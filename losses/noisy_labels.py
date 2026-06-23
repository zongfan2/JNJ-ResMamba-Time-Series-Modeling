from __future__ import annotations

import torch
import torch.nn.functional as F


def generalized_cross_entropy_loss(
    outputs: torch.Tensor,
    labels: torch.Tensor,
    *,
    q: float = 0.7,
    weight: torch.Tensor | None = None,
    balance_classes: bool = False,
    ignore_index: int = -100,
) -> torch.Tensor:
    """GCE for TSO sequence logits with padding and optional confidence weights.

    ``balance_classes`` (binary head only) adds a per-minute class-balancing
    weight identical to ``measure_loss_tso``'s pos_weight = (n_neg/n_pos).clamp(
    max=50): TSO-positive minutes are up-weighted, negatives weighted 1. This
    makes a GCE arm class-balanced EXACTLY like the CE baseline it is compared
    against — without it, GCE silently runs unbalanced while the CE baseline does
    not, confounding the CE-vs-GCE ablation. The balanced result is normalized by
    the valid-minute COUNT (not the weight sum), mirroring the BCE-with-pos_weight
    scaling in measure_loss_tso. Ignored for the 3-class head (whose CE baseline is
    likewise unweighted), so parity is preserved in both modes.
    """
    if not 0.0 < q <= 1.0:
        raise ValueError(f"q must be in (0, 1]; got {q}")

    valid = labels != ignore_index
    num_classes = outputs.shape[-1]
    targets = (labels == 2).float()  # TSO-positive indicator (binary prob + balancing)

    if num_classes == 1:
        prob_pos = torch.sigmoid(outputs[..., 0]).clamp(min=1e-7, max=1.0)
        prob_true = torch.where(targets > 0.5, prob_pos, 1.0 - prob_pos)
    else:
        safe_labels = labels.masked_fill(~valid, 0)
        probs = torch.softmax(outputs, dim=-1).clamp(min=1e-7, max=1.0)
        prob_true = probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)

    # Bound prob_true away from 0 BEFORE the power: the GCE gradient is
    # proportional to prob_true**(q-1), which is +inf at prob_true==0. In the
    # binary branch `1 - prob_pos` can hit exactly 0 (prob_pos clamped to 1.0)
    # once the model is confident on a negative minute, producing NaN gradients
    # that freeze training. Clamping here keeps the gradient finite.
    prob_true = prob_true.clamp(min=1e-7, max=1.0)

    loss = (1.0 - prob_true.pow(q)) / q
    loss = loss.masked_fill(~valid, 0.0)

    # Per-minute class-balancing weight (binary head), mirroring measure_loss_tso.
    class_w = None
    if balance_classes and num_classes == 1:
        valid_f = valid.float()
        n_pos = (targets * valid_f).sum().clamp(min=1.0)
        n_neg = ((1.0 - targets) * valid_f).sum().clamp(min=1.0)
        pos_weight = (n_neg / n_pos).clamp(max=50.0)
        class_w = torch.where(targets > 0.5, pos_weight, torch.ones_like(targets))

    if weight is not None:
        w = weight.to(outputs.device).float()
        if class_w is not None:
            w = w * class_w
        w = w.masked_fill(~valid, 0.0)
        return (loss * w).sum() / w.sum().clamp(min=1.0)

    if class_w is not None:
        # Normalize by valid-minute COUNT (not weight sum) to match the
        # BCE-with-pos_weight scaling in measure_loss_tso.
        cw = class_w.masked_fill(~valid, 0.0)
        return (loss * cw).sum() / valid.float().sum().clamp(min=1.0)

    return loss.sum() / valid.float().sum().clamp(min=1.0)


def consensus_from_annotators(
    annotator_tso: torch.Tensor,
    *,
    positive_class: int = 2,
    threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert binary annotator TSO votes [B,T,A] to class labels and confidence."""
    if annotator_tso.dim() != 3:
        raise ValueError(f"annotator_tso must have shape [B,T,A]; got {tuple(annotator_tso.shape)}")
    vote_fraction = annotator_tso.float().mean(dim=-1)
    is_tso = vote_fraction >= threshold
    labels = torch.where(
        is_tso,
        torch.full_like(vote_fraction, positive_class, dtype=torch.long),
        torch.zeros_like(vote_fraction, dtype=torch.long),
    )
    confidence = torch.maximum(vote_fraction, 1.0 - vote_fraction)
    return labels, confidence
