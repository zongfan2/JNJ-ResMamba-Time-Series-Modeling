# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from typing import Optional, Tuple
import numpy as np
from transformers import PatchTSTConfig, PatchTSTModel, ViTModel, ViTConfig, Swinv2Model, Swinv2Config


try:
    from .vit1d import ViT1D
except ImportError:
    ViT1D = None


def create_mask(original_lengths, max_length, batch_size, device):
    mask = torch.arange(max_length, device=device).unsqueeze(0).expand(batch_size, -1)
    mask = (mask < original_lengths.unsqueeze(1)).float()
    return mask


# Specialized architectures: PatchTST, ViT, Swin, etc.

class PatchEmbedding(nn.Module):
    """
    Simple and lightweight patch embedding for raw sensor data.
    Maps [batch_size, seq_len, patch_size, channels] -> [batch_size, seq_len, num_filters]

    Uses 1D convolution + pooling for efficiency.
    Memory efficient: processes patches independently.
    """
    def __init__(self, patch_size=1200, in_channels=5, num_filters=64, reduction_factor=4):
        super(PatchEmbedding, self).__init__()

        # Simple 1D conv to extract features from each patch
        # Stride reduces patch dimension by reduction_factor
        self.conv1 = nn.Conv1d(in_channels, num_filters // 2,
                               kernel_size=7, stride=reduction_factor, padding=3)
        self.bn1 = nn.BatchNorm1d(num_filters // 2)
        self.act1 = nn.ReLU(inplace=True)

        # Second conv to further process
        self.conv2 = nn.Conv1d(num_filters // 2, num_filters,
                               kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(num_filters)
        self.act2 = nn.ReLU(inplace=True)

        # Global average pooling to get single vector per patch
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, patch_size, channels]
        Returns:
            out: [batch_size, seq_len, num_filters]
        """
        batch_size, seq_len, patch_size, channels = x.size()

        # Reshape: [B*seq_len, channels, patch_size]
        x = x.view(batch_size * seq_len, patch_size, channels)
        x = x.permute(0, 2, 1)  # [B*seq_len, channels, patch_size]

        # Apply convolutions
        x = self.act1(self.bn1(self.conv1(x)))  # [B*seq_len, num_filters//2, reduced]
        x = self.act2(self.bn2(self.conv2(x)))  # [B*seq_len, num_filters, reduced]

        # Global pooling
        x = self.pool(x).squeeze(-1)  # [B*seq_len, num_filters]

        # Reshape back
        x = x.view(batch_size, seq_len, -1)  # [B, seq_len, num_filters]

        return x


class PatchTSTHead(nn.Module):
    def __init__(self, config: PatchTSTConfig):
        super(PatchTSTHead, self).__init__()
        self.use_cls_token = config.use_cls_token
        self.pooling_type = config.pooling_type
        self.cls_only = config.cls_only
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(config.head_dropout) if config.head_dropout > 0 else nn.Identity()
        # self.linear = nn.Linear(config.num_input_channels * config.d_model, config.num_targets)
        # self.out1 = nn.Linear(config.num_input_channels * config.d_model, 1)
        self.conv = nn.Conv1d(in_channels=config.num_input_channels, out_channels=1, kernel_size=1)
        self.out1 = nn.Linear(config.d_model, 1)
        if not self.cls_only:
            # self.out2 = nn.Linear(config.num_input_channels * config.d_model, config.prediction_length)
            # self.out3 = nn.Linear(config.num_input_channels * config.d_model, 1)
            self.out2 = nn.Linear(config.d_model, config.prediction_length)
            self.out3 = nn.Linear(config.d_model, 1)

    def forward(self, embedding: torch.Tensor, x_lengths: list, max_seq_len=None):
        """
        Parameters:
            embedding (`torch.Tensor` of shape `(bs, num_channels, num_patches, d_model)` or
                     `(bs, num_channels, num_patches+1, d_model)` if `cls_token` is set to True, *required*):
                Embedding from the model
        Returns:
            `torch.Tensor` of shape `(bs, num_targets)`

        """
        if self.use_cls_token:
            # use the first output token, pooled_embedding: bs x num_channels x d_model
            pooled_embedding = embedding[:, :, 0, :]
        elif self.pooling_type == "mean":
            # pooled_embedding: [bs x num_channels x d_model]
            pooled_embedding = embedding.mean(dim=2)
        elif self.pooling_type == "max":
            # pooled_embedding: [bs x num_channels x d_model]
            pooled_embedding = embedding.max(dim=2).values
        else:
            raise ValueError(f"pooling operator {self.pooling_type} is not implemented yet")
        # pooled_embedding: bs x num_channels * d_model
        # pooled_embedding = self.flatten(pooled_embedding)
        pooled_embedding = self.conv(pooled_embedding)
        pooled_embedding = pooled_embedding.squeeze(1)
        # output: bs x n_classes
        # output = self.linear(self.dropout(pooled_embedding))
        output1 = self.out1(self.dropout(pooled_embedding))
        if not self.cls_only:
            output2 = self.out2(self.dropout(pooled_embedding))
            output3 = self.out3(self.dropout(pooled_embedding))

            mask = create_mask(torch.tensor(x_lengths, device=pooled_embedding.device), max_seq_len, len(x_lengths), pooled_embedding.device)
            mask = mask.reshape(-1).bool()
            output2 = output2.view(-1)
            output2 = output2[mask]
            attention_weights = torch.zeros(pooled_embedding.shape[0], pooled_embedding.shape[1], 1,device=pooled_embedding.device)
            return output1, output2, output3, attention_weights
        else:
            return output1

class PatchTSTNS(nn.Module):
    def __init__(self, config: PatchTSTConfig, pretrained: bool=False):
        super(PatchTSTNS, self).__init__()

        self.model = PatchTSTModel(config)
        # TODO: CONNECTION ERROR DUE TO FIREWALL
        if pretrained:
            self.model.from_pretrained("namctin/patchtst_etth1_pretrain")
        self.head = PatchTSTHead(config)
        self._max_seq_len = config.context_length

    def forward(self, x, x_lengths=None, labels1=None, labels3=None,
                apply_mixup=False, mixup_alpha=0.2, max_seq_len=None):
        # HuggingFace PatchTST requires input length == config.context_length.
        # Right-pad with zeros if the (batch-local) padded length is shorter.
        if x.dim() == 3 and x.shape[1] < self._max_seq_len:
            pad_len = self._max_seq_len - x.shape[1]
            x = F.pad(x, (0, 0, 0, pad_len))
        elif x.dim() == 3 and x.shape[1] > self._max_seq_len:
            x = x[:, : self._max_seq_len, :]
        # out2 has fixed prediction_length == context_length == self._max_seq_len,
        # so always use that as the mask size.
        max_seq_len = self._max_seq_len
        x = self.model(x)
        head_out = self.head(x["last_hidden_state"], x_lengths, max_seq_len)
        # PatchTSTHead returns (out1, out2, out3, attn) — extend to 6 values.
        # Return attn=None (not a 3D placeholder) so train_scratch.py skips it.
        if len(head_out) == 4:
            out1, out2, out3, _ = head_out
            return out1, out2, out3, None, None, None
        return head_out


class PredictHead(nn.Module):
    def __init__(self, prediction_length, 
                       d_model, 
                       reg_head=True, 
                       mask_padding=True,
                       average_mask=False,
                       average_window_size=20):
        super(PredictHead, self).__init__()
        self.reg_head = reg_head
        self.mask_padding = mask_padding
        self.out1 = nn.Linear(d_model, 1)
        self.average_mask = average_mask
        if self.average_mask:
            self.out2 = nn.Linear(d_model, prediction_length//average_window_size)
        else:
            self.out2 = nn.Linear(d_model, prediction_length)
        if self.reg_head:
            self.out3 = nn.Linear(d_model, 1)
    
    def forward(self, x, x_lengths=None, max_seq_len=None):
        # multi-task output
        out1 = self.out1(x)
        out2 = self.out2(x)
        out3 = None
        if self.reg_head:
            out3 = self.out3(x)
        attn = torch.zeros(x.shape[0], 1).to(x.device)

        if self.mask_padding and not self.average_mask:
            # mask padding area and concat all batchs into a ginlge vector 
            # TODO: concat might be wrong, since default BCELoss use mean reduction
            assert x_lengths is not None, "If mask_padding is True, x_lengths must be given."
            mask = create_mask(torch.tensor(x_lengths, device=x.device), max_seq_len, len(x_lengths), x.device)
            mask = mask.reshape(-1).bool()
            out2 = out2.view(-1)
            out2 = out2[mask]
        return out1, out2, out3


class GASFFeatureExtractor(nn.Module):
    def __init__(self, image_size):
        super(GASFFeatureExtractor, self).__init__()
        self.sum_feature_extractor = GramianAngularField(image_size=image_size, method="summation")


class ViT(nn.Module):
    def __init__(self, prediction_length, 
                 input_dim=3, 
                 d_model=1024, 
                 hidden_dropout_prob=0.1,
                 load_pretrained=False,
                 target_shape=128,
                 tcn_params={},
                 reg_head=True,
                 mask_padding=True,
                 average_mask=False,
                 average_window_size=20):
        super(ViT, self).__init__()

        self.reg_head = reg_head

        # use Conv2d FeatureExtractor
        self.feature_extractor = FeatureExtractorConv2d(input_dim, prediction_length, target_shape, **tcn_params)

        # ViT configuration
        self.config = ViTConfig(
            image_size=target_shape,  # Assuming target shape is 128
            patch_size=target_shape // 16,   # Example patch size
            num_channels=input_dim,
            hidden_size=d_model, # default 768
            num_hidden_layers=6,
            num_attention_heads=8,
            intermediate_size=2048,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=hidden_dropout_prob,
        )

        # Load the ViT model
        self.vit = ViTModel(self.config)

        if load_pretrained:
            self.vit = self.vit.from_pretrained("google/vit-base-patch16-224-in21k")

        # Output layers
        self.pool = nn.AdaptiveAvgPool1d(1)  # Global average pooling
        self.head = PredictHead(prediction_length, 
                                d_model, 
                                reg_head=reg_head, 
                                mask_padding=mask_padding,
                                average_mask=average_mask,
                                average_window_size=average_window_size)

    def forward(self, x, x_lengths=None, max_seq_len=None):
        # Forward pass through the ViT model
        x = self.feature_extractor(x.permute(0, 2, 1), x_lengths)

        outputs = self.vit(x).last_hidden_state  # Get the last hidden state
        x = self.pool(outputs.transpose(1, 2)).squeeze(-1)  # Global Average Pooling

        # Multi-task outputs
        out1, out2, out3 = self.head(x, x_lengths, max_seq_len)
        attn = torch.zeros(x.shape[0], 1).to(x.device)  # Placeholder for attention

        return out1, out2, out3, attn



class SwinT(nn.Module):
    def __init__(self, prediction_length, 
                 input_dim=3, 
                 hidden_dropout_prob=0.1,
                 load_pretrained=False,
                 target_shape=128,
                 tcn_params={},
                 reg_head=True,
                 mask_padding=True,
                 average_mask=False,
                 average_window_size=20
                 ):
        super(SwinT, self).__init__()

        self.reg_head = reg_head

        # use Conv2d FeatureExtractor
        self.feature_extractor = FeatureExtractorConv2d(input_dim, prediction_length, target_shape, **tcn_params)

        # SwinT configuration
        self.config = Swinv2Config(
            image_size=target_shape,  # Assuming target shape is 128
            patch_size=4,    # Patch size (for example)
            num_channels=input_dim,
            embed_dim=96,
            depths=[2, 2, 6, 2],  # Example depth configuration for each stage
            num_heads=[3, 6, 12, 24],  # Number of attention heads
            window_size=4,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            dropout_rate=hidden_dropout_prob,
            attention_probs_dropout_rate=hidden_dropout_prob,
            encoder_stride=16,
        )

        # Load the SwinT model
        self.swin = Swinv2Model(self.config)

        # if load_pretrained:
        #     self.swin = self.swin.from_pretrained("google/vit-base-patch16-224-in21k")

        # Output layers
        self.pool = nn.AdaptiveAvgPool1d(1)  # Global average pooling
        self.head = PredictHead(prediction_length, 
                                d_model, 
                                reg_head=reg_head, 
                                mask_padding=mask_padding,
                                average_mask=average_mask,
                                average_window_size=average_window_size)

    def forward(self, x, x_lengths=None, max_seq_len=None):
        # Forward pass through the SwinT model
        x = self.feature_extractor(x.permute(0, 2, 1), x_lengths)

        outputs = self.swin(x).last_hidden_state  # Get the last hidden state
        x = self.pool(outputs.transpose(1, 2)).squeeze(-1)  # Global Average Pooling

        # Multi-task outputs
        
        attn = torch.zeros(x.shape[0], 1).to(x.device)  # Placeholder for attention
        out1, out2, out3 = self.head(x, x_lengths, max_seq_len)
        return out1, out2, out3, attn


