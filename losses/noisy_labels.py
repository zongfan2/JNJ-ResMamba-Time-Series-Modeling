from __future__ import annotations

import torch
import torch.nn.functional as F


def generalized_cross_entropy_loss(
    outputs: torch.Tensor,
    labels: torch.Tensor,
    *,
    q: float = 0.7,
    weight: torch.Tensor | None = None,
    ignore_index: int = -100,
) -> torch.Tensor:
    """GCE for TSO sequence logits with padding and optional confidence weights."""
    if not 0.0 < q <= 1.0:
        raise ValueError(f"q must be in (0, 1]; got {q}")

    valid = labels != ignore_index
    num_classes = outputs.shape[-1]

    if num_classes == 1:
        targets = (labels == 2).float()
        prob_pos = torch.sigmoid(outputs[..., 0]).clamp(min=1e-7, max=1.0)
        prob_true = torch.where(targets > 0.5, prob_pos, 1.0 - prob_pos)
    else:
        safe_labels = labels.masked_fill(~valid, 0)
        probs = torch.softmax(outputs, dim=-1).clamp(min=1e-7, max=1.0)
        prob_true = probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)

    loss = (1.0 - prob_true.pow(q)) / q
    loss = loss.masked_fill(~valid, 0.0)

    if weight is not None:
        w = weight.to(outputs.device).float().masked_fill(~valid, 0.0)
        return (loss * w).sum() / w.sum().clamp(min=1.0)

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
