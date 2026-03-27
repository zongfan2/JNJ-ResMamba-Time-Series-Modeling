# -*- coding: utf-8 -*-
"""
Dynamic Label Refinement with Temporal Consistency (DLR-TC)
Loss Functions and Label Management

This module implements the DLR-TC framework for learning from noisy labels
in wearable sensor time series data (TSO prediction).

Key Components:
1. Generalized Cross Entropy (GCE) - Robust to label noise
2. Compatibility/Anchor Loss (KL-divergence) - Prevents drift from ground truth
3. Temporal Smoothness Loss (Total Variation) - Enforces physical continuity
4. Soft Label Manager - Handles trainable label parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional


# ==================== Loss Functions ====================

class GeneralizedCrossEntropy(nn.Module):
    """
    Generalized Cross Entropy Loss (GCE)

    More robust to label noise than standard Cross Entropy.
    Interpolates between CE (q→0) and MAE (q→1).

    Reference: "Generalized Cross Entropy Loss for Training Deep Neural Networks
    with Noisy Labels" (NeurIPS 2018)

    Args:
        q: Truncation parameter (0 < q ≤ 1). Lower q = more robust to noise.
           Recommended: 0.7 for moderate noise, 0.5 for heavy noise
        num_classes: Number of classes (3 for TSO: other, non-wear, predictTSO)
        reduction: 'mean' or 'none'

    Mathematical form:
        L_GCE = (1 - sum_k(y_k * p_k^q)) / q

    where:
        - y_k is the (soft) label probability for class k
        - p_k is the model prediction probability for class k
        - q controls the robustness (lower = more robust)
    """
    def __init__(self, q: float = 0.7, num_classes: int = 3, reduction: str = 'mean'):
        super(GeneralizedCrossEntropy, self).__init__()
        self.q = q
        self.num_classes = num_classes
        self.reduction = reduction

        # Numerical stability epsilon
        self.eps = 1e-7

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            pred: Model predictions [batch_size, seq_len, num_classes] (logits or probs)
            target: Soft labels [batch_size, seq_len, num_classes] (probability distribution)
            mask: Optional binary mask [batch_size, seq_len] for valid timesteps

        Returns:
            Scalar loss value
        """
        # Ensure predictions are probabilities (apply softmax if needed)
        if pred.size(-1) == self.num_classes:
            # Check if already probabilities (sum ≈ 1) or logits
            if not torch.allclose(pred.sum(dim=-1), torch.ones_like(pred.sum(dim=-1)), atol=0.1):
                pred = F.softmax(pred, dim=-1)

        # Clamp for numerical stability
        pred = torch.clamp(pred, self.eps, 1.0)

        # Ensure target is a probability distribution
        if target.size(-1) != self.num_classes:
            # Convert hard labels to one-hot if needed
            target = F.one_hot(target.long(), num_classes=self.num_classes).float()

        # GCE formula: (1 - sum_k(y_k * p_k^q)) / q
        pred_powered = torch.pow(pred, self.q)  # p_k^q
        weighted_sum = (target * pred_powered).sum(dim=-1)  # sum_k(y_k * p_k^q)

        loss = (1.0 - weighted_sum) / self.q

        # Apply mask if provided (ignore padded positions)
        if mask is not None:
            loss = loss * mask  # Zero out invalid positions
            if self.reduction == 'mean':
                return loss.sum() / (mask.sum() + self.eps)
            elif self.reduction == 'none':
                return loss
        else:
            if self.reduction == 'mean':
                return loss.mean()
            elif self.reduction == 'none':
                return loss

        return loss


class CompatibilityLoss(nn.Module):
    """
    Compatibility/Anchor Loss using KL-Divergence

    Prevents refined soft labels from drifting too far from the original
    noisy annotations. Acts as a "reality anchor".

    Args:
        reduction: 'mean' or 'none'

    Mathematical form:
        L_Compat = KL(y_refined || y_noisy)
                 = sum_k(y_refined_k * log(y_refined_k / y_noisy_k))

    This asymmetric KL encourages y_refined to stay close to y_noisy.
    """
    def __init__(self, reduction: str = 'mean'):
        super(CompatibilityLoss, self).__init__()
        self.reduction = reduction
        self.eps = 1e-7

    def forward(self, refined_labels: torch.Tensor, noisy_labels: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            refined_labels: Soft labels being optimized [batch_size, seq_len, num_classes]
            noisy_labels: Original ground truth labels [batch_size, seq_len, num_classes]
            mask: Optional binary mask [batch_size, seq_len] for valid timesteps

        Returns:
            Scalar loss value
        """
        # Clamp for numerical stability
        refined_labels = torch.clamp(refined_labels, self.eps, 1.0)
        noisy_labels = torch.clamp(noisy_labels, self.eps, 1.0)

        # KL divergence: sum_k(p_k * log(p_k / q_k))
        # = sum_k(p_k * (log(p_k) - log(q_k)))
        kl_div = refined_labels * (torch.log(refined_labels) - torch.log(noisy_labels))
        kl_div = kl_div.sum(dim=-1)  # Sum over classes

        # Apply mask if provided
        if mask is not None:
            kl_div = kl_div * mask
            if self.reduction == 'mean':
                return kl_div.sum() / (mask.sum() + self.eps)
            elif self.reduction == 'none':
                return kl_div
        else:
            if self.reduction == 'mean':
                return kl_div.mean()
            elif self.reduction == 'none':
                return kl_div

        return kl_div


class TemporalSmoothnessLoss(nn.Module):
    """
    Temporal Smoothness Loss using Total Variation (TV)

    Enforces physical continuity by penalizing high-frequency oscillations
    in the soft label distributions. Sleep/wake states change smoothly over
    time, not erratically every minute.

    Args:
        reduction: 'mean' or 'none'
        norm: 'l1' (Total Variation) or 'l2' (squared differences)

    Mathematical form:
        L_Temp = (1/(T-1)) * sum_{t=1}^{T-1} ||y_{t+1} - y_t||_1

    where T is the sequence length.
    """
    def __init__(self, reduction: str = 'mean', norm: str = 'l1'):
        super(TemporalSmoothnessLoss, self).__init__()
        self.reduction = reduction
        self.norm = norm
        self.eps = 1e-7

    def forward(self, soft_labels: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            soft_labels: Soft label distributions [batch_size, seq_len, num_classes]
            mask: Optional binary mask [batch_size, seq_len] for valid timesteps

        Returns:
            Scalar loss value
        """
        # Calculate differences between consecutive timesteps
        # [batch_size, seq_len-1, num_classes]
        differences = soft_labels[:, 1:, :] - soft_labels[:, :-1, :]

        # Calculate norm
        if self.norm == 'l1':
            # L1 norm (Total Variation): sum of absolute values
            tv = torch.abs(differences).sum(dim=-1)  # [batch_size, seq_len-1]
        elif self.norm == 'l2':
            # L2 norm: sum of squared differences
            tv = (differences ** 2).sum(dim=-1)  # [batch_size, seq_len-1]
        else:
            raise ValueError(f"Unknown norm: {self.norm}. Use 'l1' or 'l2'.")

        # Apply mask if provided (mask for transitions)
        if mask is not None:
            # Create transition mask: valid if both t and t+1 are valid
            transition_mask = mask[:, :-1] & mask[:, 1:]  # [batch_size, seq_len-1]
            tv = tv * transition_mask

            if self.reduction == 'mean':
                # Average over valid transitions
                return tv.sum() / (transition_mask.sum() + self.eps)
            elif self.reduction == 'none':
                return tv
        else:
            if self.reduction == 'mean':
                return tv.mean()
            elif self.reduction == 'none':
                return tv

        return tv


class DLRTCLoss(nn.Module):
    """
    Combined Dynamic Label Refinement with Temporal Consistency Loss

    Combines three components:
    1. GCE: Robust prediction loss
    2. Compatibility: Anchor to original labels
    3. Temporal Smoothness: Physical continuity constraint

    Total Loss = L_GCE + alpha * L_Compat + beta * L_Temp

    Args:
        q: GCE truncation parameter (default: 0.7)
        alpha: Weight for compatibility loss (default: 0.1)
        beta: Weight for temporal smoothness loss (default: 0.5)
        num_classes: Number of classes (default: 3)
    """
    def __init__(self, q: float = 0.7, alpha: float = 0.1, beta: float = 0.5,
                 num_classes: int = 3):
        super(DLRTCLoss, self).__init__()

        self.gce = GeneralizedCrossEntropy(q=q, num_classes=num_classes, reduction='mean')
        self.compat = CompatibilityLoss(reduction='mean')
        self.temporal = TemporalSmoothnessLoss(reduction='mean', norm='l1')

        self.alpha = alpha
        self.beta = beta

    def forward(self, pred: torch.Tensor, refined_labels: torch.Tensor,
                noisy_labels: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred: Model predictions [batch_size, seq_len, num_classes]
            refined_labels: Trainable soft labels [batch_size, seq_len, num_classes]
            noisy_labels: Original ground truth [batch_size, seq_len, num_classes]
            mask: Valid timestep mask [batch_size, seq_len]

        Returns:
            Dictionary with total loss and individual components
        """
        # 1. Robust prediction loss
        loss_gce = self.gce(pred, refined_labels, mask)

        # 2. Compatibility loss (anchor to original labels)
        loss_compat = self.compat(refined_labels, noisy_labels, mask)

        # 3. Temporal smoothness loss
        loss_temp = self.temporal(refined_labels, mask)

        # Combined loss
        total_loss = loss_gce + self.alpha * loss_compat + self.beta * loss_temp

        return {
            'total': total_loss,
            'gce': loss_gce,
            'compatibility': loss_compat,
            'temporal': loss_temp
        }


# ==================== Soft Label Manager ====================

class SoftLabelManager:
    """
    Manages trainable soft label parameters for DLR-TC

    Handles:
    - Initialization from noisy labels
    - Storage as torch.nn.Parameter
    - Gradient updates during label refinement phase
    - Persistence to disk
    - Reconstruction from saved state

    The soft labels are stored per segment with unique IDs to maintain
    consistency across training iterations.
    """

    def __init__(self, num_classes: int = 3, device: torch.device = None):
        """
        Args:
            num_classes: Number of classes (3 for TSO prediction)
            device: torch device (cuda or cpu)
        """
        self.num_classes = num_classes
        self.device = device if device is not None else torch.device('cpu')

        # Storage: segment_id -> soft_labels Parameter
        # Each segment has variable length, so we store them separately
        self.soft_labels_dict = {}

        # Optimizers for label refinement (one per segment for efficiency)
        self.label_optimizers = {}

    def initialize_from_batch(self, segment_ids: list, hard_labels: torch.Tensor,
                              seq_lens: torch.Tensor, temperature: float = 1.0):
        """
        Initialize soft labels from hard labels with optional label smoothing

        Args:
            segment_ids: List of segment identifiers [batch_size]
            hard_labels: Hard class labels [batch_size, seq_len]
            seq_lens: Actual sequence lengths [batch_size]
            temperature: Softening parameter (1.0 = one-hot, higher = smoother)
        """
        batch_size = len(segment_ids)

        for i in range(batch_size):
            seg_id = segment_ids[i]
            seq_len = int(seq_lens[i])

            # Skip if already initialized
            if seg_id in self.soft_labels_dict:
                continue

            # Extract valid labels
            labels_i = hard_labels[i, :seq_len]  # [seq_len]

            # Convert to one-hot
            one_hot = F.one_hot(labels_i.long(), num_classes=self.num_classes).float()  # [seq_len, 3]

            # Optional label smoothing: soft_label = (1-ε)*one_hot + ε/K
            if temperature > 1.0:
                epsilon = 1.0 / temperature
                soft = (1.0 - epsilon) * one_hot + epsilon / self.num_classes
            else:
                soft = one_hot

            # Create trainable parameter
            soft_param = nn.Parameter(soft.clone().detach().to(self.device), requires_grad=True)

            self.soft_labels_dict[seg_id] = soft_param

    def get_batch_soft_labels(self, segment_ids: list, seq_lens: torch.Tensor,
                              max_seq_len: int) -> torch.Tensor:
        """
        Retrieve soft labels for a batch, padded to max_seq_len

        Args:
            segment_ids: List of segment identifiers [batch_size]
            seq_lens: Actual sequence lengths [batch_size]
            max_seq_len: Maximum sequence length for padding

        Returns:
            Padded soft labels [batch_size, max_seq_len, num_classes]
        """
        batch_size = len(segment_ids)

        # Initialize output tensor with uniform distribution for padding
        batch_soft = torch.zeros(batch_size, max_seq_len, self.num_classes,
                                device=self.device, dtype=torch.float32)
        batch_soft += 1.0 / self.num_classes  # Uniform prior for padded positions

        for i in range(batch_size):
            seg_id = segment_ids[i]
            seq_len = int(seq_lens[i])

            if seg_id in self.soft_labels_dict:
                soft = self.soft_labels_dict[seg_id]  # [seq_len, num_classes]

                # Apply softmax to ensure valid probability distribution
                soft_probs = F.softmax(soft, dim=-1)

                # Copy to batch tensor
                batch_soft[i, :seq_len, :] = soft_probs
            else:
                raise ValueError(f"Segment {seg_id} not found in soft label storage. "
                               f"Call initialize_from_batch() first.")

        return batch_soft

    def update_labels(self, segment_ids: list, gradients: torch.Tensor,
                     seq_lens: torch.Tensor, lr: float = 0.01):
        """
        Update soft labels using gradient descent

        Args:
            segment_ids: List of segment identifiers [batch_size]
            gradients: Gradients w.r.t. soft labels [batch_size, max_seq_len, num_classes]
            seq_lens: Actual sequence lengths [batch_size]
            lr: Learning rate for label updates
        """
        for i in range(len(segment_ids)):
            seg_id = segment_ids[i]
            seq_len = int(seq_lens[i])

            if seg_id in self.soft_labels_dict:
                # Extract gradient for this segment
                grad_i = gradients[i, :seq_len, :]  # [seq_len, num_classes]

                # Manual gradient descent (since we're optimizing labels, not model)
                with torch.no_grad():
                    self.soft_labels_dict[seg_id] -= lr * grad_i

                    # Optional: Project to simplex (ensure sum to 1 constraint)
                    # This is handled by softmax in get_batch_soft_labels()

    def save_state(self, filepath: str):
        """
        Save soft labels to disk

        Args:
            filepath: Path to save file (e.g., 'soft_labels_epoch10.pt')
        """
        # Convert parameters to regular tensors for saving
        state_dict = {}
        for seg_id, param in self.soft_labels_dict.items():
            state_dict[seg_id] = param.detach().cpu()

        torch.save({
            'soft_labels': state_dict,
            'num_classes': self.num_classes
        }, filepath)

        print(f"Soft labels saved to {filepath} ({len(state_dict)} segments)")

    def load_state(self, filepath: str):
        """
        Load soft labels from disk

        Args:
            filepath: Path to saved file
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        state_dict = checkpoint['soft_labels']
        self.num_classes = checkpoint['num_classes']

        # Reconstruct parameters
        self.soft_labels_dict = {}
        for seg_id, tensor in state_dict.items():
            param = nn.Parameter(tensor.to(self.device), requires_grad=True)
            self.soft_labels_dict[seg_id] = param

        print(f"Soft labels loaded from {filepath} ({len(state_dict)} segments)")

    def get_num_segments(self) -> int:
        """Return number of segments with soft labels"""
        return len(self.soft_labels_dict)
