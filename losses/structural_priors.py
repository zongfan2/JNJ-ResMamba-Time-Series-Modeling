# -*- coding: utf-8 -*-
"""
Structural priors and noisy-label regularizers for Deep TSO.

This module collects loss terms designed for the *structured* noise found in
TSO labels (actigraphy / self-report / heuristic algorithm outputs). Unlike
generic noisy-label methods (GCE, DLRTC), these terms encode the *shape* of
TSO -- roughly one continuous segment per 24h, plausible duration (3-11h),
and a strong circadian time-of-day prior.

Loss terms (functional API):
    - transition_count_loss : penalize fragmented TSO segments above a budget.
    - duration_prior_loss   : hinge penalty on predicted TSO duration outside a band.
    - boundary_reweighted_ce_loss : CE with caller-supplied per-position weights
                                    that down-weight uncertain boundary regions.
    - elr_loss              : Early-Learning Regularization term (Liu et al., 2020).

Stateful helpers (classes):
    - ELRMemory             : per-(segment, timestep) EMA of model predictions.
    - CircadianPriorBias    : learnable scalar that scales a fixed log-odds-of-TSO
                              vs hour-of-day prior added to logits.

Convenience:
    - measure_loss_tso_structural : combined loss with logging dict.
    - compute_boundary_weights    : precompute boundary-distance weights for a
                                    single label sequence (call at dataset build).
    - hour_from_time_channels     : recover hour-of-day from sin/cos channels.

All functions support both binary (output_channels=1) and 3-class
(output_channels=3) TSO outputs, and handle padding via x_lengths or the
-100 ignore index.

See losses/standard.py:measure_loss_tso for the conventional CE baseline
these terms compose with, and losses/dlrtc.py for the DLRTC alternative.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _tso_probs(outputs: torch.Tensor) -> torch.Tensor:
    """Per-timestep TSO probability from [B, T, C] logits (C=1 or 3)."""
    num_classes = outputs.shape[-1]
    if num_classes == 1:
        return torch.sigmoid(outputs[..., 0])
    # 3-class: TSO is class index 2 (other=0, non-wear=1, predictTSO=2).
    return torch.softmax(outputs, dim=-1)[..., 2]


def _valid_mask(x_lengths: torch.Tensor, seq_len: int) -> torch.Tensor:
    """Build a [B, T] bool mask for valid (non-padding) positions."""
    positions = torch.arange(seq_len, device=x_lengths.device).unsqueeze(0)
    return positions < x_lengths.unsqueeze(1)


# ---------------------------------------------------------------------------
# Transition-count loss
# ---------------------------------------------------------------------------

def transition_count_loss(
    outputs: torch.Tensor,
    x_lengths: torch.Tensor,
    budget: float = 2.0,
) -> torch.Tensor:
    """
    Penalize TSO segment fragmentation beyond a budget of TV units.

    A clean binary signal with k connected TSO segments has total variation
    (TV) approximately 2k -- one onset + one offset per segment. With
    ``budget=2.0`` one TSO segment per sequence is allowed free of penalty
    and only additional fragmentation is penalized.

    This is strictly stronger than plain TV smoothness
    (see ``tso_continuity_loss`` in losses/standard.py), which penalizes
    even the legitimate onset and offset.

    Args:
        outputs: [B, T, C] logits, C=1 or 3.
        x_lengths: [B] valid sequence lengths.
        budget: TV units allowed before penalty kicks in (default 2.0).

    Returns:
        loss: scalar, mean over batch of ``ReLU(TV_p - budget)``.
    """
    T = outputs.shape[1]
    probs = _tso_probs(outputs)                              # [B, T]
    valid = _valid_mask(x_lengths, T)
    valid_trans = valid[:, :-1] & valid[:, 1:]               # [B, T-1]
    diffs = torch.abs(probs[:, 1:] - probs[:, :-1])          # [B, T-1]
    tv_per_seq = (diffs * valid_trans).sum(dim=1)            # [B]
    return F.relu(tv_per_seq - budget).mean()


# ---------------------------------------------------------------------------
# Duration prior loss
# ---------------------------------------------------------------------------

def duration_prior_loss(
    outputs: torch.Tensor,
    x_lengths: torch.Tensor,
    patch_duration_hours: float,
    d_min: float = 3.0,
    d_max: float = 11.0,
) -> torch.Tensor:
    """
    Hinge penalty on predicted soft TSO duration.

    Soft duration: ``D_pred = sum_t p_t * Delta_t``. Penalize when D_pred
    falls outside [d_min, d_max] hours. Hinge form is robust to
    right-skewed sleep distributions and avoids a soft pressure toward the
    prior mean.

    Pair with ``transition_count_loss``; alone, this prior can be gamed by
    trading onset error for equal-magnitude offset error.

    Args:
        outputs: [B, T, C] logits, C=1 or 3.
        x_lengths: [B] valid sequence lengths.
        patch_duration_hours: time-resolution of each timestep, in hours
            (e.g., 1-minute patches: 1/60 ~= 0.01667).
        d_min, d_max: hinge band (hours).

    Returns:
        loss: scalar, mean over batch of hinge penalty.
    """
    T = outputs.shape[1]
    probs = _tso_probs(outputs)                              # [B, T]
    valid = _valid_mask(x_lengths, T).float()
    d_pred = (probs * valid).sum(dim=1) * patch_duration_hours   # [B]
    below = F.relu(d_min - d_pred)
    above = F.relu(d_pred - d_max)
    return (below + above).mean()


# ---------------------------------------------------------------------------
# Boundary-aware reweighting (Approach 1: label-based)
# ---------------------------------------------------------------------------

def compute_boundary_weights(label: np.ndarray, tau_steps: float = 10.0) -> np.ndarray:
    """
    Precompute boundary-distance weights for a single label sequence.

    Use at dataset construction time and cache the result; pass to
    ``boundary_reweighted_ce_loss`` as the ``weight`` argument so that the
    distance transform is not recomputed per batch.

    Weight per position: ``1 - exp(-dist / tau_steps)``, so positions far
    from any boundary get full weight (~1) and positions at the boundary
    get weight ~0.

    Args:
        label: [T] integer label sequence (0/1/2, -100 for padding).
        tau_steps: decay length in timesteps. Set to 0 to disable
            (returns all-ones).

    Returns:
        weight: [T] float32 array in [0, 1]. Padding positions are weighted 0.
    """
    T = label.shape[0]
    if tau_steps <= 0:
        w = np.ones(T, dtype=np.float32)
        w[label == -100] = 0.0
        return w
    # Boundary: position t where label[t] != label[t-1] and neither is padding.
    valid = label != -100
    transitions = (label[1:] != label[:-1]) & valid[1:] & valid[:-1]
    boundary_positions = np.where(transitions)[0] + 1  # transition assigned to right side
    if boundary_positions.size == 0:
        w = np.ones(T, dtype=np.float32)
        w[~valid] = 0.0
        return w
    positions = np.arange(T)
    dist = np.min(np.abs(positions[:, None] - boundary_positions[None, :]), axis=1)
    w = (1.0 - np.exp(-dist / float(tau_steps))).astype(np.float32)
    w[~valid] = 0.0
    return w


def boundary_reweighted_ce_loss(
    outputs: torch.Tensor,
    labels: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """
    Cross-entropy with caller-supplied per-position weights.

    Args:
        outputs: [B, T, C] logits, C=1 (binary) or 3 (three-class).
        labels: [B, T] integer class indices, -100 for padding.
        weight: [B, T] per-position weights in [0, 1].
            Use ``compute_boundary_weights`` at dataset construction.

    Returns:
        loss: scalar.
    """
    B, T, C = outputs.shape
    ignore = labels == -100
    w = weight.clone().to(outputs.device).float()
    w = w.masked_fill(ignore, 0.0)
    if C == 1:
        targets = (labels == 2).float()
        ce = F.binary_cross_entropy_with_logits(outputs[..., 0], targets, reduction='none')
    else:
        ce = F.cross_entropy(
            outputs.reshape(-1, C), labels.reshape(-1),
            ignore_index=-100, reduction='none'
        ).reshape(B, T)
    return (ce * w).sum() / w.sum().clamp(min=1.0)


# ---------------------------------------------------------------------------
# Early-Learning Regularization (ELR)
# ---------------------------------------------------------------------------

class ELRMemory:
    """
    Per-(segment, timestep) EMA buffer for Early-Learning Regularization.

    Maintains running EMA of model predictions per training segment. Used as
    a regularization target that resists drift away from the early-training
    consensus, which is empirically less corrupted by noisy labels.

    Reference: Liu et al., "Early-Learning Regularization Prevents
    Memorization of Noisy Labels", NeurIPS 2020.

    Memory cost: ~ num_segments * max_seq_len * num_classes * 4 bytes.
    For 2000 segments x 1440 timesteps x 1 class -> ~11 MB.

    Usage:
        memory = ELRMemory(num_train_segments, max_seq_len,
                           num_classes=1, beta=0.7)
        for segment_ids, x, y, x_lengths in loader:
            logits = model(x)                              # [B, T, C]
            probs = torch.sigmoid(logits[..., 0])          # binary case
            target = memory.get(segment_ids).to(device)    # [B, T]
            loss_elr = elr_loss(probs, target, x_lengths)
            (loss_ce + w_elr * loss_elr).backward()
            opt.step()
            memory.update(segment_ids,
                          probs.detach().cpu().numpy())
    """

    def __init__(self, num_segments: int, max_seq_len: int,
                 num_classes: int = 1, beta: float = 0.7, init: float = 0.5):
        if not 0.0 < beta < 1.0:
            raise ValueError(f"beta must be in (0, 1); got {beta}")
        self.num_classes = num_classes
        self.beta = beta
        if num_classes == 1:
            self.target = np.full((num_segments, max_seq_len), init, dtype=np.float32)
        else:
            self.target = np.full((num_segments, max_seq_len, num_classes),
                                  1.0 / num_classes, dtype=np.float32)

    def update(self, segment_ids, predictions: np.ndarray) -> None:
        """
        Refresh EMA targets for the given segments.

        Args:
            segment_ids: array-like of int [B].
            predictions: numpy array [B, T] (binary) or [B, T, C] (multi-class).
        """
        b = self.beta
        ids = np.asarray(segment_ids)
        # Vectorized update over the batch.
        self.target[ids] = b * self.target[ids] + (1.0 - b) * predictions

    def get(self, segment_ids) -> torch.Tensor:
        """Return EMA targets for the given segments as a CPU torch tensor."""
        ids = np.asarray(segment_ids)
        return torch.from_numpy(self.target[ids].copy())

    def save(self, path: str) -> None:
        np.savez(path, target=self.target, beta=np.array(self.beta),
                 num_classes=np.array(self.num_classes))

    @classmethod
    def load(cls, path: str) -> "ELRMemory":
        d = np.load(path)
        obj = cls.__new__(cls)
        obj.target = d['target']
        obj.beta = float(d['beta'])
        obj.num_classes = int(d['num_classes'])
        return obj


def elr_loss(probs: torch.Tensor, target: torch.Tensor,
             x_lengths: torch.Tensor) -> torch.Tensor:
    """
    Early-Learning Regularization loss term: ``-log(1 - <p, t>)``.

    Binary form: ``L = -log(1 - p*t - (1-p)*(1-t))``.
    Multi-class form: ``L = -log(1 - sum_c p_c * t_c)``.

    Args:
        probs: [B, T] in [0,1] (binary) or [B, T, C] (multi-class probs).
        target: same shape as probs, EMA of past predictions.
        x_lengths: [B] valid sequence lengths.

    Returns:
        loss: scalar, mean over valid positions.
    """
    T = probs.shape[1]
    valid = _valid_mask(x_lengths, T).to(probs.device).float()
    if probs.dim() == 2:
        agreement = probs * target + (1.0 - probs) * (1.0 - target)
    else:
        agreement = (probs * target).sum(dim=-1)             # [B, T]
    elr_per = -torch.log((1.0 - agreement).clamp(min=1e-7))  # [B, T]
    return (elr_per * valid).sum() / valid.sum().clamp(min=1.0)


# ---------------------------------------------------------------------------
# Circadian prior (output-side logit bias with learnable gamma)
# ---------------------------------------------------------------------------

def _gaussian_smooth_circular(x: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing with circular (wrap-around) boundary conditions."""
    n = len(x)
    radius = int(np.ceil(3 * sigma))
    if radius < 1 or sigma <= 0:
        return x.copy()
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    kernel = kernel / kernel.sum()
    padded = np.concatenate([x[-radius:], x, x[:radius]])
    return np.convolve(padded, kernel, mode='valid')[:n].astype(np.float32)


class CircadianPriorBias(nn.Module):
    """
    Output-side circadian bias: ``logit_t += gamma * log_p_circ(hour_t)``.

    Pre-computes a fixed log-odds-of-TSO (binary) or log-probability
    (multi-class) per hour-of-day from training labels, then biases logits
    at inference. ``gamma`` is a learnable scalar (or per-class scalar)
    controlling prior strength.

    Set ``gamma_init=0.0`` to start with the prior off and let the model
    learn how much to trust it; ``gamma_init=1.0`` to start with full
    prior weight.
    """

    def __init__(self, log_p_circ_table: np.ndarray,
                 gamma_init: float = 0.0, num_classes: int = 1):
        super().__init__()
        if num_classes == 1:
            expected_shape = (24,)
        else:
            expected_shape = (24, num_classes)
        if log_p_circ_table.shape != expected_shape:
            raise ValueError(
                f"log_p_circ_table shape {log_p_circ_table.shape} "
                f"does not match expected {expected_shape} for num_classes={num_classes}"
            )
        self.register_buffer('log_p_circ',
                             torch.from_numpy(log_p_circ_table.astype(np.float32)))
        if num_classes == 1:
            self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        else:
            self.gamma = nn.Parameter(torch.full((num_classes,), float(gamma_init)))
        self.num_classes = num_classes

    @classmethod
    def from_training_labels(
        cls,
        hours: np.ndarray,
        labels: np.ndarray,
        num_classes: int = 1,
        gamma_init: float = 0.0,
        smooth_sigma: float = 1.0,
        eps: float = 0.01,
    ) -> "CircadianPriorBias":
        """
        Build a CircadianPriorBias from flat arrays of hours and labels.

        Args:
            hours: [N] floats in [0, 24).
            labels: [N] integer labels (0/1/2, -100 ignored).
            num_classes: 1 (binary, TSO=class 2) or 3 (three-class).
            gamma_init: initial value for the learnable gamma scalar(s).
            smooth_sigma: Gaussian smoothing sigma in hour-bins (0=no smoothing).
            eps: clip probabilities to [eps, 1-eps] for numerical stability.
        """
        mask = labels != -100
        hours = hours[mask]
        labels = labels[mask]
        bin_idx = (np.floor(hours).astype(int)) % 24

        if num_classes == 1:
            tso = (labels == 2).astype(np.float32)
            counts = np.bincount(bin_idx, minlength=24).astype(np.float32)
            sums = np.bincount(bin_idx, weights=tso, minlength=24).astype(np.float32)
            p = sums / np.maximum(counts, 1.0)
            if smooth_sigma > 0:
                p = _gaussian_smooth_circular(p, smooth_sigma)
            p = np.clip(p, eps, 1 - eps)
            log_p = (np.log(p / (1 - p))).astype(np.float32)        # [24]
        else:
            p = np.zeros((24, num_classes), dtype=np.float32)
            counts = np.bincount(bin_idx, minlength=24).astype(np.float32)
            for c in range(num_classes):
                sums = np.bincount(
                    bin_idx, weights=(labels == c).astype(np.float32), minlength=24
                ).astype(np.float32)
                p[:, c] = sums / np.maximum(counts, 1.0)
            if smooth_sigma > 0:
                for c in range(num_classes):
                    p[:, c] = _gaussian_smooth_circular(p[:, c], smooth_sigma)
            p = np.clip(p, eps, 1 - eps)
            p = p / p.sum(axis=1, keepdims=True)
            log_p = np.log(p).astype(np.float32)                    # [24, C]
        return cls(log_p, gamma_init=gamma_init, num_classes=num_classes)

    def forward(self, logits: torch.Tensor, hours: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, T, C] raw model logits.
            hours: [B, T] floats in [0, 24).

        Returns:
            logits_biased: [B, T, C].
        """
        bin_idx = (torch.floor(hours).long() % 24)                  # [B, T]
        log_p = self.log_p_circ[bin_idx]                            # [B,T] or [B,T,C]
        if self.num_classes == 1:
            return logits + (self.gamma * log_p).unsqueeze(-1)
        return logits + self.gamma * log_p


def hour_from_time_channels(time_sin: torch.Tensor,
                            time_cos: torch.Tensor) -> torch.Tensor:
    """
    Recover hour-of-day from sin/cos time channels.

    Assumes the channels were encoded as
        time_sin = sin(2*pi*h/24),
        time_cos = cos(2*pi*h/24).

    Args:
        time_sin, time_cos: tensors of identical shape.

    Returns:
        hour: same shape, in [0, 24).
    """
    angle = torch.atan2(time_sin, time_cos)
    hour = (angle / (2.0 * math.pi)) * 24.0
    return hour % 24.0


# ---------------------------------------------------------------------------
# Combined loss
# ---------------------------------------------------------------------------

def measure_loss_tso_structural(
    outputs: torch.Tensor,
    labels: torch.Tensor,
    x_lengths: torch.Tensor,
    *,
    patch_duration_hours: float,
    boundary_weight: Optional[torch.Tensor] = None,
    w_trans: float = 0.0,
    w_dur: float = 0.0,
    w_elr: float = 0.0,
    trans_budget: float = 2.0,
    dur_min: float = 3.0,
    dur_max: float = 11.0,
    elr_target: Optional[torch.Tensor] = None,
) -> dict:
    """
    Combined TSO loss = (boundary-weighted) CE + structural priors + ELR.

    Args:
        outputs: [B, T, C] logits.
        labels: [B, T] integer class indices, -100 for padding.
        x_lengths: [B] valid sequence lengths.
        patch_duration_hours: time-resolution of each timestep.
        boundary_weight: optional [B, T] tensor; if provided, used instead
            of standard CE. Build with ``compute_boundary_weights`` at
            dataset construction.
        w_trans, w_dur, w_elr: scalar weights for each structural / ELR term
            (set to 0 to disable that term cheaply).
        trans_budget: TV-units budget for transition_count_loss.
        dur_min, dur_max: hinge band for duration_prior_loss (hours).
        elr_target: optional [B, T] (binary) or [B, T, C] (multi) EMA target
            from ``ELRMemory.get``. Required when ``w_elr > 0``.

    Returns:
        dict with keys 'total', 'ce', 'trans', 'dur', 'elr'. Disabled terms
        appear as zero tensors so callers can log uniformly.
    """
    out = {}

    if boundary_weight is not None:
        ce = boundary_reweighted_ce_loss(outputs, labels, boundary_weight)
    else:
        # Local import avoids a circular import at module load time.
        from .standard import measure_loss_tso
        ce = measure_loss_tso(outputs, labels, x_lengths)
    out['ce'] = ce
    total = ce

    if w_trans > 0:
        trans = transition_count_loss(outputs, x_lengths, budget=trans_budget)
        out['trans'] = trans
        total = total + w_trans * trans
    else:
        out['trans'] = torch.zeros((), device=outputs.device)

    if w_dur > 0:
        dur = duration_prior_loss(outputs, x_lengths, patch_duration_hours,
                                  d_min=dur_min, d_max=dur_max)
        out['dur'] = dur
        total = total + w_dur * dur
    else:
        out['dur'] = torch.zeros((), device=outputs.device)

    if w_elr > 0:
        if elr_target is None:
            raise ValueError("elr_target is required when w_elr > 0")
        probs = _tso_probs(outputs)
        elr = elr_loss(probs, elr_target.to(outputs.device), x_lengths)
        out['elr'] = elr
        total = total + w_elr * elr
    else:
        out['elr'] = torch.zeros((), device=outputs.device)

    out['total'] = total
    return out


__all__ = [
    'transition_count_loss',
    'duration_prior_loss',
    'compute_boundary_weights',
    'boundary_reweighted_ce_loss',
    'ELRMemory',
    'elr_loss',
    'CircadianPriorBias',
    'hour_from_time_channels',
    'measure_loss_tso_structural',
]
