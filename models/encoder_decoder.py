# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from typing import Optional, Tuple
import numpy as np

from .mamba_blocks import MBA, AffineDropPath
from .attention import (
    AttModule, AttModule_mamba, MultiHeadSelfAttentionPooling,
    AttModule_mamba_causal, AttModule_mamba_cross, AttModule_cross,
)
from .components import FeatureExtractor
from .resmamba import latent_mixup

# Encoder-decoder models and variants

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        print(d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *(-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).permute(0,2,1) # of shape (1, d_model, l)
        self.pe = nn.Parameter(pe, requires_grad=True)
#         self.register_buffer('pe', pe)
        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
#         self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        return x + self.pe[:, :, 0:x.shape[2]] 





class Encoder(nn.Module):
    def __init__(self, num_layers, r1, r2, num_f_maps, input_dim, output_dim, channel_masking_rate, att_type, alpha, drop_path_rate=0.3,encoder_only=False):
        super(Encoder, self).__init__()
        self.encoder_only = encoder_only
        self.conv_1x1 = nn.Conv1d(input_dim, num_f_maps, 1) # fc layer
        self.layers = nn.ModuleList([AttModule_mamba(2 ** i, num_f_maps, num_f_maps, 'sliding_att', 'encoder', 1, drop_path_rate=drop_path_rate) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, output_dim, 1)
        self.dropout = nn.Dropout2d(p=channel_masking_rate)
        self.channel_masking_rate = channel_masking_rate

    def forward(self, x, mask):
        if self.channel_masking_rate > 0:
            x = x.unsqueeze(2)
            x = self.dropout(x)
            x = x.squeeze(2)
        feature = self.conv_1x1(x)
        for layer in self.layers:
            feature = layer(feature, mask)
        if self.encoder_only:
            out = feature
        else:
            out = self.conv_out(feature) * mask.unsqueeze(1)#* mask[:, 0:1, :]
        return out
    
class Decoder(nn.Module):
    def __init__(self, num_layers, r1, r2, num_f_maps, input_dim, out_dim, att_type, alpha, drop_path_rate=0.3):
        super(Decoder, self).__init__()#         self.position_en = PositionalEncoding(d_model=num_f_maps)
        self.conv_1x1 = nn.Conv1d(input_dim, num_f_maps, 1)

        self.layers = nn.ModuleList([AttModule_mamba(2 ** i, num_f_maps, num_f_maps, att_type, 'decoder', alpha, drop_path_rate=drop_path_rate) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, out_dim, 1)

    def forward(self, x,  mask):
        feature = self.conv_1x1(x)
        for layer in self.layers:
            feature = layer(feature, mask)
        out = self.conv_out(feature) * mask.unsqueeze(1)
        return out


class MBA_encoder_decoder(nn.Module):
    """
    Unified encoder-decoder architecture for multi-task scratch detection.

    Supports four bottleneck modes via the `bottleneck_type` parameter:
      - "channel": Compresses channels (num_filters -> 1 -> num_filters) with
        flat feature dimensions. Uses cross-attention decoder.
      - "sequence": Compresses sequence length (seq_len -> seq_len/reduction)
        with flat feature dimensions. Uses cross-attention decoder.
      - "progressive_skip": Progressive U-shaped dims (e.g., 64->128->256->512)
        with U-Net skip connections from encoder to decoder. Uses cross-attention.
      - "progressive": Same progressive dims but NO skip connections.
        Uses self-attention decoder (no cross-attention).

    All modes share: padding mask handling, positional encoding, attention pooling,
    multi-task output heads, mixup augmentation, and contrastive learning support.
    """
    def __init__(self, input_dim,
                 num_filters=64,
                 num_encoder_layers=4,
                 num_decoder_layers=3,
                 drop_path_rate=0.3,
                 kernel_size_mba=7,
                 max_seq_len=256,
                 cls_token=False,
                 add_positional_encoding=True,
                 num_heads=4,
                 dropout_rate=0.2,
                 mba=True,
                 average_mask=False,
                 average_window_size=20,
                 supcon_loss=False,
                 reg_only=False,
                 cls_src="encoder",
                 bottleneck_type="progressive_skip",
                 seq_reduction=8,
                 **kwargs):
        """
        Args:
            input_dim: Number of input channels (e.g., 3 for accelerometer x,y,z).
            num_filters: Base feature dimension.
            num_encoder_layers: Number of encoder layers.
            num_decoder_layers: Number of decoder layers.
            drop_path_rate: Stochastic depth rate.
            kernel_size_mba: Kernel size for Mamba/attention blocks.
            max_seq_len: Maximum sequence length.
            cls_token: If True, use average pooling for CLS; else attention pooling.
            add_positional_encoding: Whether to add positional encoding.
            num_heads: Number of attention heads for pooling.
            dropout_rate: Dropout rate.
            mba: If True, use Mamba blocks; else use standard attention.
            average_mask: If True, average mask predictions over windows.
            average_window_size: Window size for mask averaging.
            supcon_loss: If True, produce contrastive embeddings.
            reg_only: If True, only produce regression output (skip cls + mask).
            cls_src: "encoder" or "decoder" — source of features for cls/reg heads.
            bottleneck_type: One of "channel", "sequence", "progressive_skip", "progressive".
            seq_reduction: Sequence compression factor (only for "sequence" mode).
        """
        super(MBA_encoder_decoder, self).__init__()

        self.bottleneck_type = bottleneck_type
        self.add_positional_encoding = add_positional_encoding
        self.max_seq_len = max_seq_len
        self.pooling = cls_token
        self.average_mask = average_mask
        self.average_window_size = average_window_size
        self.supcon_loss = supcon_loss
        self.reg_only = reg_only
        self.input_dim = input_dim
        self.num_filters = num_filters
        self.cls_src = cls_src
        self.seq_reduction = seq_reduction

        is_progressive = bottleneck_type in ("progressive_skip", "progressive")

        # --- Input projection ---
        if is_progressive:
            self.input_projection = nn.Conv1d(input_dim, num_filters, 1)
        else:
            self.input_projection = nn.Conv1d(input_dim, num_filters, 1)

        self.norm = nn.LayerNorm(num_filters)

        # --- Feature dimensions ---
        if is_progressive:
            self.encoder_dims = [num_filters * (2 ** i) for i in range(num_encoder_layers)]
            self.decoder_dims = [self.encoder_dims[-(i+1)] for i in range(num_decoder_layers)]
            pooling_dim = self.encoder_dims[-1]
        else:
            self.encoder_dims = [num_filters] * num_encoder_layers
            self.decoder_dims = [num_filters] * num_decoder_layers
            pooling_dim = num_filters

        # --- Positional encoding ---
        self.positional_encoding = PositionalEncoding(d_model=num_filters, max_len=max_seq_len)

        # --- Attention pooling ---
        self.MultiHeadSelfAttentionPooling = MultiHeadSelfAttentionPooling(
            input_dim=pooling_dim, hidden_dim=pooling_dim, num_heads=num_heads
        )

        # --- Progressive projection layers (only for progressive modes) ---
        if is_progressive:
            self.encoder_projections = nn.ModuleList([
                nn.Conv1d(self.encoder_dims[i-1] if i > 0 else num_filters, self.encoder_dims[i], 1)
                for i in range(num_encoder_layers)
            ])
            self.decoder_projections = nn.ModuleList([
                nn.Conv1d(self.decoder_dims[i-1] if i > 0 else self.encoder_dims[-1], self.decoder_dims[i], 1)
                for i in range(num_decoder_layers)
            ])
        else:
            self.encoder_projections = None
            self.decoder_projections = None

        # --- Skip connections (only for progressive_skip) ---
        if bottleneck_type == "progressive_skip":
            self.skip_projections = nn.ModuleList([
                nn.Conv1d(self.encoder_dims[-(i+1)], self.decoder_dims[i], 1)
                for i in range(min(num_decoder_layers, num_encoder_layers))
            ])
        else:
            self.skip_projections = None

        # --- Encoder layers ---
        if is_progressive:
            if not mba:
                self.encoder = nn.ModuleList([
                    AttModule(2 ** i, self.encoder_dims[i], self.encoder_dims[i], 2, 2,
                              'normal_att', 'encoder', alpha=1,
                              kernel_size=kernel_size_mba, dropout_rate=dropout_rate)
                    for i in range(num_encoder_layers)
                ])
            else:
                self.encoder = nn.ModuleList([
                    AttModule_mamba_causal(2 ** i, self.encoder_dims[i], self.encoder_dims[i],
                                          'sliding_att', 'encoder', 1,
                                          drop_path_rate=drop_path_rate, dropout_rate=dropout_rate,
                                          kernel_size=kernel_size_mba)
                    for i in range(num_encoder_layers)
                ])
        else:
            # Flat encoder (channel / sequence bottleneck)
            if not mba:
                self.encoder = nn.ModuleList([
                    AttModule(2 ** i, num_filters, num_filters, 2, 2,
                              'normal_att', 'encoder', alpha=1,
                              kernel_size=kernel_size_mba, dropout_rate=dropout_rate)
                    for i in range(num_encoder_layers)
                ])
            else:
                self.encoder = nn.ModuleList([
                    AttModule_mamba_causal(2 ** i, num_filters, num_filters,
                                          'sliding_att', 'encoder', 1,
                                          drop_path_rate=drop_path_rate, dropout_rate=dropout_rate,
                                          kernel_size=kernel_size_mba)
                    for i in range(num_encoder_layers)
                ])

        # --- Decoder layers ---
        if bottleneck_type == "progressive":
            # Self-attention decoder (no cross-attention, no skip connections)
            if not mba:
                self.decoder = nn.ModuleList([
                    AttModule(2 ** i, self.decoder_dims[i], self.decoder_dims[i], 2, 2,
                              'normal_att', 'decoder', alpha=1,
                              kernel_size=kernel_size_mba, dropout_rate=dropout_rate)
                    for i in range(num_decoder_layers)
                ])
            else:
                self.decoder = nn.ModuleList([
                    AttModule_mamba(2 ** i, self.decoder_dims[i], self.decoder_dims[i],
                                   'sliding_att', 'decoder', 1,
                                   drop_path_rate=drop_path_rate, dropout_rate=dropout_rate,
                                   kernel_size=kernel_size_mba)
                    for i in range(num_decoder_layers)
                ])
        elif is_progressive:
            # Cross-attention decoder (progressive_skip)
            if not mba:
                self.decoder = nn.ModuleList([
                    AttModule_cross(2 ** i, self.decoder_dims[i], self.decoder_dims[i], 2, 2,
                                   'cross_att', 'decoder', alpha=1,
                                   kernel_size=kernel_size_mba, dropout_rate=dropout_rate)
                    for i in range(num_decoder_layers)
                ])
            else:
                self.decoder = nn.ModuleList([
                    AttModule_mamba_cross(2 ** i, self.decoder_dims[i], self.decoder_dims[i], 2, 2,
                                         'cross_att', 'decoder', 1,
                                         drop_path_rate=drop_path_rate, dropout_rate=dropout_rate,
                                         kernel_size=kernel_size_mba)
                    for i in range(num_decoder_layers)
                ])
        else:
            # Cross-attention decoder (channel / sequence bottleneck)
            if not mba:
                self.decoder = nn.ModuleList([
                    AttModule_cross(2 ** i, num_filters, num_filters, 2, 2,
                                   'cross_att', 'decoder', alpha=1,
                                   kernel_size=kernel_size_mba, dropout_rate=dropout_rate)
                    for i in range(num_decoder_layers)
                ])
            else:
                self.decoder = nn.ModuleList([
                    AttModule_mamba_cross(2 ** i, num_filters, num_filters, 2, 2,
                                         'cross_att', 'decoder', 1,
                                         drop_path_rate=drop_path_rate, dropout_rate=dropout_rate,
                                         kernel_size=kernel_size_mba)
                    for i in range(num_decoder_layers)
                ])

        # --- Bottleneck-specific layers ---
        if bottleneck_type == "channel":
            self.encoder_compress = nn.Conv1d(num_filters, 1, 1)
            self.decoder_expand = nn.Conv1d(1, num_filters, 1)
        elif bottleneck_type == "sequence":
            self.seq_compress = nn.Conv1d(num_filters, num_filters,
                                          kernel_size=seq_reduction, stride=seq_reduction, padding=0)
            self.seq_expand = nn.ConvTranspose1d(num_filters, num_filters,
                                                  kernel_size=seq_reduction, stride=seq_reduction, padding=0)

        # --- Output heads ---
        out_dim = self.decoder_dims[-1] if is_progressive else num_filters
        cls_dim = self.encoder_dims[-1] if (is_progressive and cls_src == "encoder") else out_dim

        if not reg_only:
            self.out1 = nn.Linear(cls_dim, 1)   # Classification
            self.out2 = nn.Conv1d(out_dim, 1, 1) if is_progressive else nn.Conv1d(num_filters, 1, 1)  # Sequence labeling
        self.out3 = nn.Linear(cls_dim, 1)        # Regression

        if average_mask:
            self.scale_out2 = nn.Linear(max_seq_len, max_seq_len // average_window_size)

    def forward(self, x, labels1=None, labels3=None, apply_mixup=False, mixup_alpha=0.2,
                padding_value=-999.0, **kwargs):
        batch_size, seq_len, channels = x.size()
        x_original = x

        # --- Padding mask ---
        padding_mask_original = ~torch.any(x_original == padding_value, dim=-1)  # [B, seq_len]

        x_processed = x.clone()
        if abs(padding_value) > 100:
            padding_mask_for_replacement = torch.any(x == padding_value, dim=-1, keepdim=True)
            x_processed = torch.where(padding_mask_for_replacement, 0.0, x_processed)

        # --- Input projection ---
        x = self.input_projection(x_processed.permute(0, 2, 1))  # [B, num_filters, seq_len]
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)

        if self.add_positional_encoding:
            x = self.positional_encoding(x)

        padding_mask = padding_mask_original.float()
        current_seq_len = x.shape[2]

        # =====================================================================
        # ENCODER
        # =====================================================================
        is_progressive = self.bottleneck_type in ("progressive_skip", "progressive")

        if is_progressive:
            encoder_states = []
            encoder_out = x
            for i, (proj, layer) in enumerate(zip(self.encoder_projections, self.encoder)):
                encoder_out = proj(encoder_out)
                encoder_out = layer(encoder_out, None, padding_mask)
                encoder_states.append(encoder_out)
        else:
            encoder_out = x
            for layer in self.encoder:
                encoder_out = layer(encoder_out, None, padding_mask)

        # =====================================================================
        # ATTENTION POOLING (CLS token)
        # =====================================================================
        pooling_dim_size = self.encoder_dims[-1] if is_progressive else self.num_filters
        encoder_features = encoder_out.permute(0, 2, 1)  # [B, seq_len, dim]

        if self.pooling:
            cls_token = encoder_features[:, 1:, :].mean(dim=1)
            attention_weights = torch.zeros(batch_size, current_seq_len, 1, device=x.device)
        else:
            cls_token, _, attention_weights = self.MultiHeadSelfAttentionPooling(
                encoder_features, padding_mask
            )

        # =====================================================================
        # BOTTLENECK + DECODER
        # =====================================================================
        if self.bottleneck_type == "channel":
            # Channel bottleneck: compress channels -> sigmoid -> expand -> cross-attention decoder
            compressed = torch.sigmoid(self.encoder_compress(encoder_out))
            mask_expanded = padding_mask.unsqueeze(1)
            compressed = compressed * mask_expanded
            decoder_out = self.decoder_expand(compressed)
            masked_encoder = encoder_out * mask_expanded
            for layer in self.decoder:
                decoder_out = layer(decoder_out, masked_encoder, padding_mask)

        elif self.bottleneck_type == "sequence":
            # Sequence bottleneck: compress seq_len -> sigmoid -> expand -> cross-attention decoder
            compressed = self.seq_compress(encoder_out)
            compressed_mask = F.max_pool1d(
                padding_mask.unsqueeze(1).float(),
                kernel_size=self.seq_reduction, stride=self.seq_reduction
            ).squeeze(1)
            compressed = torch.sigmoid(compressed) * compressed_mask.unsqueeze(1)
            decoder_out = self.seq_expand(compressed)
            if decoder_out.shape[2] != current_seq_len:
                decoder_out = F.interpolate(decoder_out, size=current_seq_len, mode='linear', align_corners=False)
            mask_expanded = padding_mask.unsqueeze(1)
            masked_encoder = encoder_out * mask_expanded
            for layer in self.decoder:
                decoder_out = layer(decoder_out, masked_encoder, padding_mask)

        elif self.bottleneck_type == "progressive_skip":
            # Progressive with U-Net skip connections
            decoder_out = encoder_out
            for i, (proj, layer) in enumerate(zip(self.decoder_projections, self.decoder)):
                decoder_out = proj(decoder_out)
                if i < len(encoder_states) and i < len(self.skip_projections):
                    encoder_skip = encoder_states[-(i+1)]
                    encoder_skip_proj = self.skip_projections[i](encoder_skip)
                else:
                    encoder_skip_proj = encoder_states[-1]
                decoder_out = layer(decoder_out, encoder_skip_proj, padding_mask)

        elif self.bottleneck_type == "progressive":
            # Progressive without skip connections — self-attention decoder
            decoder_out = encoder_out
            for i, (proj, layer) in enumerate(zip(self.decoder_projections, self.decoder)):
                decoder_out = proj(decoder_out)
                decoder_out = layer(decoder_out, None, padding_mask)

        else:
            raise ValueError(f"Unknown bottleneck_type: {self.bottleneck_type}")

        # =====================================================================
        # OUTPUT HEADS
        # =====================================================================
        if self.cls_src == "encoder":
            cls_feature = cls_token
        else:
            decoder_features = decoder_out.permute(0, 2, 1)
            cls_feature, _, attention_weights = self.MultiHeadSelfAttentionPooling(
                decoder_features, padding_mask_original
            )

        # Mixup augmentation on CLS feature
        if apply_mixup and self.training:
            mixed_cls, mixed_l1, mixed_l3, lam = latent_mixup(
                cls_feature, labels1, labels3, alpha=mixup_alpha
            )
            mixup_info = (mixed_l1, mixed_l3, lam)
        else:
            mixed_cls = cls_feature
            mixup_info = None

        # Classification + regression
        if not self.reg_only:
            x1 = self.out1(mixed_cls)
            x3 = self.out3(mixed_cls)
        else:
            x1 = None
            x3 = self.out3(mixed_cls)

        # Sequence labeling
        if not self.reg_only:
            x2 = self.out2(decoder_out).squeeze(1)  # [B, seq_len]
            if self.average_mask:
                x2 = x2.view(batch_size, seq_len // self.average_window_size, self.average_window_size).mean(dim=2)
        else:
            x2 = None

        # Contrastive embedding
        contrastive_embedding = cls_feature if self.supcon_loss else None

        return x1, x2, x3, attention_weights, contrastive_embedding, mixup_info

    def train(self, mode=True):
        """Standard train method."""
        return super().train(mode)


# Backward-compatible aliases (accept positional args for setup.py compatibility)
MBA_tsm_encoder_decoder_ch_bottleneck = lambda *args, **kw: MBA_encoder_decoder(*args, bottleneck_type="channel", **kw)
MBA_tsm_encoder_decoder_seq_bottleneck = lambda *args, **kw: MBA_encoder_decoder(*args, bottleneck_type="sequence", **kw)
MBA_tsm_encoder_decoder_progressive_with_skip_connection = lambda *args, **kw: MBA_encoder_decoder(*args, bottleneck_type="progressive_skip", **kw)
MBA_tsm_encoder_decoder_progressive = lambda *args, **kw: MBA_encoder_decoder(*args, bottleneck_type="progressive", **kw)

