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
# Structured output (C4): single-interval onset/offset boundary supervision
# ---------------------------------------------------------------------------

def tso_interval_bounds(labels, x_lengths=None, positive_class=2, ignore_index=-100):
    """Per-night (onset, offset) minute indices of the LONGEST contiguous run of
    ``positive_class`` (the single-TSO prior), within each night's valid length.

    Args:
        labels: [B, T] integer class indices (``-100`` = padding).
        x_lengths: [B] valid lengths in minutes (defaults to T).
        positive_class: the TSO class index (2 in 3-class; binary labels also use 2).
    Returns:
        onset [B] long, offset [B] long (both -1 for nights with no TSO),
        has   [B] bool (True where a TSO run exists).
    """
    B, T = labels.shape
    device = labels.device
    onset = torch.full((B,), -1, dtype=torch.long, device=device)
    offset = torch.full((B,), -1, dtype=torch.long, device=device)
    has = torch.zeros((B,), dtype=torch.bool, device=device)
    for b in range(B):
        n = int(x_lengths[b]) if x_lengths is not None else T
        n = max(1, min(n, T))
        idx = (labels[b, :n] == positive_class).nonzero(as_tuple=False).flatten().tolist()
        if not idx:
            continue
        # longest contiguous run among the positive indices
        best = (idx[0], idx[0]); s = p = idx[0]
        for j in idx[1:]:
            if j == p + 1:
                p = j
            else:
                if p - s > best[1] - best[0]:
                    best = (s, p)
                s = p = j
        if p - s > best[1] - best[0]:
            best = (s, p)
        onset[b], offset[b], has[b] = best[0], best[1], True
    return onset, offset, has


def _soft_argmax_positions(onset_logits, offset_logits):
    """Continuous expected onset/offset minute (soft-argmax) per night.

    softmax over positions -> expected index E[t] = sum_t t * p(t). Differentiable,
    and uses the per-minute spatial structure (unlike a pooled scalar regressor).
    Padding positions are assumed pre-masked to a very negative logit by the model,
    so their softmax mass is ~0 and does not pull the expectation. Returns
    (exp_onset [B], exp_offset [B]) float tensors in minute units.
    """
    T = onset_logits.shape[1]
    pos = torch.arange(T, device=onset_logits.device, dtype=onset_logits.dtype)
    exp_on = (torch.softmax(onset_logits, dim=1) * pos).sum(dim=1)
    exp_off = (torch.softmax(offset_logits, dim=1) * pos).sum(dim=1)
    return exp_on, exp_off


def interval_regression_loss(onset_logits, offset_logits, labels, x_lengths=None,
                             positive_class=2, order_weight=0.1):
    """Onset/offset REGRESSION loss for the structured single-interval head (C4).

    The per-minute onset/offset logits are turned into continuous expected
    positions by soft-argmax, normalized to a fraction of each night's valid
    length, and regressed (smooth-L1) toward the start/end of the longest GT TSO
    run. A small hinge ``relu(onset - offset)`` discourages inverted intervals.
    Nights with no TSO are skipped. Returns a scalar (0 if no night has TSO).
    """
    onset_gt, offset_gt, has = tso_interval_bounds(labels, x_lengths, positive_class)
    if not bool(has.any()):
        return onset_logits.new_zeros(())
    T = onset_logits.shape[1]
    if x_lengths is None:
        lens = onset_logits.new_full((onset_logits.shape[0],), float(T))
    else:
        lens = x_lengths.to(onset_logits.dtype).clamp(min=1.0)
    exp_on, exp_off = _soft_argmax_positions(onset_logits, offset_logits)
    pred_on, pred_off = exp_on[has] / lens[has], exp_off[has] / lens[has]
    gt_on = onset_gt[has].to(onset_logits.dtype) / lens[has]
    gt_off = offset_gt[has].to(onset_logits.dtype) / lens[has]
    reg = F.smooth_l1_loss(pred_on, gt_on) + F.smooth_l1_loss(pred_off, gt_off)
    order = torch.relu(pred_on - pred_off).mean()
    return 0.5 * reg + order_weight * order


def decode_interval(onset_logits, offset_logits):
    """Decode a single contiguous interval per night from the regression head.

    Returns the soft-argmax expected onset/offset rounded to integer minutes, with
    offset clamped to be >= onset so the decoded [onset, offset] is contiguous and
    valid by construction. Use at inference to replace post-hoc single-TSO
    enforcement. Returns (onset [B], offset [B]) long tensors.
    """
    exp_on, exp_off = _soft_argmax_positions(onset_logits, offset_logits)
    onset = exp_on.round().long()
    offset = torch.maximum(exp_off.round().long(), onset)
    return onset, offset


def cross_night_consistency_loss(outputs, x_lengths, subject_indices, ignore_index=-100):
    """Within-subject cross-night CONSISTENCY penalty (positive-only).

    Motivation: the SupCon term uses subject identity to form positives AND
    negatives, but the negative ("different subjects are dissimilar") assumption is
    weak for sleep timing — many people sleep in similar windows, so different-
    subject pairs are often false negatives. This loss keeps only the part we
    actually believe: a subject's nights should be consistent. For each subject
    with >=2 nights in the batch, it penalizes the variance of the predicted
    window's soft CENTER and soft DURATION across that subject's nights.

    It is computed from the per-minute TSO probabilities (no interval head needed),
    fully differentiable, and collapse-safe: the per-minute supervised loss keeps
    the predictions meaningful, so this only aligns same-subject windows rather than
    flattening everything. There is NO negative/repulsion term, so it is unaffected
    by batch size and by the false-negative problem. Train-only; returns a scalar
    (0 if no subject has >=2 nights in the batch).

    Args:
        outputs: [B, T, C] logits (C=1 sigmoid or C=3 softmax-TSO).
        x_lengths: [B] valid lengths in minutes.
        subject_indices: [B] long, the subject id per night.
    """
    probs = _tso_probs(outputs)                      # [B, T] soft TSO prob
    B, T = probs.shape
    pos = torch.arange(T, device=probs.device, dtype=probs.dtype)
    valid = (torch.arange(T, device=probs.device)[None, :]
             < x_lengths.to(probs.device)[:, None].clamp(max=T)).to(probs.dtype)
    p = probs * valid
    mass = p.sum(dim=1).clamp(min=1e-6)              # soft duration (minutes) per night
    center = (p * pos).sum(dim=1) / mass             # soft window center (minute index)
    lens = x_lengths.to(probs.dtype).clamp(min=1.0)
    dur_frac = mass / lens                           # fraction of night (scale-free)
    ctr_frac = center / lens
    terms = []
    for s in torch.unique(subject_indices):
        idx = (subject_indices == s).nonzero(as_tuple=False).flatten()
        if idx.numel() >= 2:
            terms.append(ctr_frac[idx].var(unbiased=False) + dur_frac[idx].var(unbiased=False))
    if not terms:
        return outputs.new_zeros(())
    return torch.stack(terms).mean()


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
    supervision_weight: Optional[torch.Tensor] = None,
    base_loss: str = "ce",
    gce_q: float = 0.7,
    gce_balance: bool = True,
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

    if boundary_weight is not None and supervision_weight is not None:
        weight = boundary_weight.to(outputs.device).float() * supervision_weight.to(outputs.device).float()
    elif boundary_weight is not None:
        weight = boundary_weight.to(outputs.device).float()
    elif supervision_weight is not None:
        weight = supervision_weight.to(outputs.device).float()
    else:
        weight = None

    if base_loss == "gce":
        from .noisy_labels import generalized_cross_entropy_loss
        ce = generalized_cross_entropy_loss(outputs, labels, q=gce_q, weight=weight,
                                            balance_classes=gce_balance)
    elif base_loss == "ce":
        if weight is not None:
            ce = boundary_reweighted_ce_loss(outputs, labels, weight)
        else:
            from .standard import measure_loss_tso
            ce = measure_loss_tso(outputs, labels, x_lengths)
    else:
        raise ValueError(f"base_loss must be 'ce' or 'gce'; got {base_loss}")
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
