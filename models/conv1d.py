# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from typing import Optional, Tuple
import numpy as np


# 1D Convolution-based models

class Conv1DBlock(nn.Module):
    """Convolutional block with residual connection and normalization"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 15,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Ensure padding preserves sequence length with dilation
        padding = (dilation * (kernel_size - 1)) // 2
        
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection if dimensions match
        self.residual = (in_channels == out_channels)
        
        # Projection for residual if dimensions don't match
        if not self.residual:
            self.proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, L]
        Returns:
            [B, C_out, L]
        """
        residual = x
        
        # Apply convolution
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Add residual connection if possible
        if self.residual:
            x = x + residual
        else:
            x = x + self.proj(residual)
            
        return x

class Conv1DTS(nn.Module):
    """
    Convolutional 1D network for time series data.
    Replaces ViT1D with a simpler convolutional architecture.
    """
    def __init__(
        self,
        time_length: int = 1221,        # Total length of input sequence
        in_channels: int = 3,           # Number of input channels
        base_filters: int = 64,         # Base number of convolutional filters
        num_layers: int = 5,            # Number of convolutional blocks
        kernel_size: int = 15,          # Kernel size for convolutions
        dropout: float = 0.1,           # Dropout rate
        second_level_mask: bool = False,          # Predict mask at second level
        supcon_loss: bool = False,      # Whether to return contrastive embedding for for supcon loss
        reg_only: bool = False,         # Whether to use regression head only
    ):
        super().__init__()
        self.time_length = time_length
        self.in_channels = in_channels
        self.base_filters = base_filters
        self.return_contrastive_embedding = supcon_loss
        self.reg_only = reg_only

        # Initial projection to embed input
        self.input_projection = nn.Conv1d(in_channels, base_filters, kernel_size=1)
        
        # Convolutional blocks with increasing dilation
        self.conv_blocks = nn.ModuleList()
        for i in range(num_layers):
            in_channels = base_filters * (2**i) if i > 0 else base_filters
            out_channels = base_filters * (2**(i+1)) if i < num_layers-1 else base_filters * (2**i)
            dilation = 2**i
            
            self.conv_blocks.append(
                Conv1DBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
        
        # Global average pooling for classification and regression
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Final projection for different tasks
        final_dim = base_filters * (2**(num_layers-1))

        self.regression_head = nn.Linear(final_dim, 1)
        if not self.reg_only:
            self.classification_head = nn.Linear(final_dim, 1)
            
            # Mask prediction head with a separate pathway
            if second_level_mask:
                mask_length = time_length // 20
            else:
                mask_length = time_length
            
            self.mask_prediction = nn.Sequential(
                nn.Conv1d(final_dim, final_dim // 2, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv1d(final_dim // 2, 1, kernel_size=1),
                nn.AdaptiveAvgPool1d(mask_length)
            )
    
    def forward(self, x: torch.Tensor, x_lengths=None, labels1: Optional[torch.Tensor]=None, labels3: Optional[torch.Tensor]=None, apply_mixup: bool=False, mixup_alpha: float=0.2, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the Conv1D network.

        Args:
            x: Input tensor of shape [B, L, C]
            x_lengths: Original sequence lengths (for masking out2)
        """
        batch_size, seq_len, channels = x.size()
        # Initial projection
        x = x.permute(0, 2, 1)

        x = self.input_projection(x)

        # Apply convolutional blocks
        for conv_block in self.conv_blocks:
            x = conv_block(x)

        # Global pooling for classification and regression
        pooled = self.global_avg_pool(x).squeeze(-1)

        # Task-specific outputs
        if apply_mixup and self.training:
            mixed_features, mixed_labels1, mixed_labels3, mixup_lambda = latent_mixup(
                pooled, labels1, labels3, alpha=mixup_alpha
            )
            mixup_info = (mixed_labels1, mixed_labels3, mixup_lambda)
        else:
            mixed_features = pooled
            mixup_info = None

        regression_output = self.regression_head(mixed_features)

        if not self.reg_only:
            classification_output = self.classification_head(mixed_features)

            if self.return_contrastive_embedding:
                con_embed = pooled
            else:
                con_embed = None

            # Mask prediction [B, mask_length]
            mask_output = self.mask_prediction(x).squeeze(1)

            # Flatten and keep only valid positions
            if x_lengths is not None:
                mask = torch.arange(seq_len, device=x.device).unsqueeze(0) < torch.tensor(
                    x_lengths, device=x.device, dtype=torch.long
                ).unsqueeze(1)
                mask_output = mask_output.view(-1)[mask.view(-1)]
        else:
            classification_output, mask_output, con_embed = None, None, None

        attn = None
        return classification_output, mask_output, regression_output, attn, con_embed, mixup_info

    
    def _init_weights(self):
        """Initialize weights for better training"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


