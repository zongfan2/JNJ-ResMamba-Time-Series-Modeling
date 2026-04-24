# -*- coding: utf-8 -*-
"""ConvNormPool 1D CNN — Xing et al. 2024 (MDPI Sensors 24(11):3364).

Baseline replication of the deep-learning arm of the paper:
  - 3 × ConvNormPool modules (Conv1d → BatchNorm → Swish → MaxPool)
  - Global Average Pool
  - Two fully-connected layers → classification logit
  - Hidden = 128, kernel = 5

We preserve the paper's encoder faithfully, then add the standard MBA_v1
multi-task heads (classification / per-timestep mask / duration regression)
so the model fits the 6-value forward interface that
``training/train_scratch.py`` expects.  `out2` and `out3` are auxiliary —
if you want a strict paper replica, set ``wl2 = wl3 = 0`` in the YAML so
only the classification head receives gradient.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNormPool(nn.Module):
    """Conv1d → BatchNorm1d → Swish → MaxPool1d."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 5, pool_size: int = 2):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.SiLU()  # Swish
        self.pool = nn.MaxPool1d(pool_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.act(self.bn(self.conv(x))))


class MDPI2024CNN(nn.Module):
    """Xing et al. 2024 ConvNormPool CNN."""

    def __init__(
        self,
        time_length: int = 1221,
        in_channels: int = 3,
        hidden_size: int = 128,
        kernel_size: int = 5,
        pool_size: int = 2,
        num_blocks: int = 3,
        fc_dim: int = 128,
        dropout: float = 0.0,
        second_level_mask: bool = False,
        supcon_loss: bool = False,
    ):
        super().__init__()
        self.time_length = time_length
        self.return_contrastive_embedding = supcon_loss

        blocks = [ConvNormPool(in_channels, hidden_size, kernel_size, pool_size)]
        for _ in range(num_blocks - 1):
            blocks.append(ConvNormPool(hidden_size, hidden_size, kernel_size, pool_size))
        self.blocks = nn.Sequential(*blocks)

        # Paper: "average pooling layer and two fully connected layers".
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(hidden_size, fc_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Output heads (interface compatibility with train_scratch.py).
        self.classification_head = nn.Linear(fc_dim, 1)
        self.regression_head = nn.Linear(fc_dim, 1)
        mask_len = time_length // 20 if second_level_mask else time_length
        self.mask_prediction_head = nn.Linear(fc_dim, mask_len)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d,)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: Optional[list] = None,
        labels1: Optional[torch.Tensor] = None,
        labels3: Optional[torch.Tensor] = None,
        apply_mixup: bool = False,
        mixup_alpha: float = 0.2,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               Optional[torch.Tensor], Optional[torch.Tensor], Optional[tuple]]:
        """Forward.

        Args:
            x: [B, L, C] padded input tensor.
            x_lengths: per-sample true lengths (for masking out2).
        """
        batch_size, seq_len, _ = x.size()

        # Convert to [B, C, L] for Conv1d.
        h = x.permute(0, 2, 1)
        h = self.blocks(h)
        pooled = self.avg_pool(h).squeeze(-1)   # [B, hidden]
        feat = F.silu(self.fc1(pooled))
        feat = self.dropout(feat)

        out1 = self.classification_head(feat)   # [B, 1]
        out3 = self.regression_head(feat)       # [B, 1]
        out2 = self.mask_prediction_head(feat)  # [B, mask_len]

        # Flatten out2 to valid positions (matching MBA_v1 convention).
        if x_lengths is not None:
            mask_len = out2.shape[1]
            valid = (
                torch.arange(mask_len, device=out2.device).unsqueeze(0)
                < torch.tensor(x_lengths, device=out2.device, dtype=torch.long).unsqueeze(1)
            )
            out2 = out2.reshape(-1)[valid.reshape(-1)]

        con_embed = pooled if self.return_contrastive_embedding else None
        return out1, out2, out3, None, con_embed, None
