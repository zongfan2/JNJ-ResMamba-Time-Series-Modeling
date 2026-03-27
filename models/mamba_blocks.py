# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from typing import Optional, Tuple
import numpy as np
from mamba_ssm import Mamba


# Mamba-specific blocks and modules

class ConvFeedForward(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, kernel_size, norm="BN"):
        super(ConvFeedForward, self).__init__()
        padding = (kernel_size - 1) * dilation // 2 
        self.conv=nn.Conv1d(in_channels, out_channels, padding=padding, dilation=dilation, kernel_size=kernel_size)
        self.activation = nn.ReLU(inplace=True)
        if norm=='GN':
            self.norm = nn.GroupNorm(num_groups=4, num_channels=in_channels)
        elif norm=='BN':
            self.norm = nn.BatchNorm1d(in_channels)
        elif norm=='IN':
            self.norm = nn.InstanceNorm1d(in_channels,track_running_stats=False)
        else:
            self.norm = None

    def forward(self, x):
        out = self.conv(x)
        if self.norm is not None:
            out = self.norm(out)
        out = self.activation(out)
        return out

class ConvFeedForward_cls(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(ConvFeedForward_cls, self).__init__() 
        self.layer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layer(x)

# The follow code is modified from
# https://github.com/facebookresearch/SlowFast/blob/master/slowfast/models/common.py
def drop_path(x, drop_prob=0.0, training=False):
    """
    Stochastic Depth per sample.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()  # binarize
    output = x.div(keep_prob) * mask
    return output
    
class AffineDropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks) with a per channel scaling factor (and zero init)
    See: https://arxiv.org/pdf/2103.17239.pdf
    """

    def __init__(self, num_dim, drop_prob=0.0, init_scale_value=1e-4):
        super().__init__()
        self.scale = nn.Parameter(
            init_scale_value * torch.ones((1, num_dim, 1)),
            requires_grad=True
        )
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(self.scale * x, self.drop_prob, self.training)
    
class MaskMambaBlock(nn.Module):
    def __init__(
        self,        
        n_embd,                # dimension of the input features
        kernel_size=4,         # conv kernel size
        n_ds_stride=1,         # downsampling stride for the current layer
        drop_path_rate=0.3,         # drop path rate
    ) -> None:
        super().__init__()
        #self.mamba = ViM(n_embd, d_conv=kernel_size, use_fast_path=True, bimamba_type='v2')
        self.mamba = Mamba(n_embd,d_conv=kernel_size,use_fast_path=True)
        if n_ds_stride > 1:
            self.downsample = MaxPooler(kernel_size=3, stride=2, padding=1)
        else:
            self.downsample = None    
        self.norm = nn.LayerNorm(n_embd)
                
        # drop path
        if drop_path_rate > 0.0:
            self.drop_path = AffineDropPath(n_embd, drop_prob=drop_path_rate)
        else:
            self.drop_path = nn.Identity()

    def forward(self, x, mask):
        res = x
        x_ = x.transpose(1,2)
        x_ = self.norm(x_)
        x_ = self.mamba(x_).transpose(1, 2)
        x = x_ * mask.unsqueeze(1).to(x.dtype)
        x  = res + self.drop_path(x)
        if self.downsample is not None:
            x, mask = self.downsample(x, mask)

        return  x

class MBA(nn.Module):
    def __init__(self,input_dim,max_seq_len, n_state=16,pos_embed_dim=8,lin_embed_dim=64,BI=True,num_layers=2):
        super(MBA, self).__init__()
        
#         self.pos_embedding = PositionalEmbedding(max_seq_len, pos_embed_dim)
        self.embedding = nn.Linear(input_dim, lin_embed_dim)
        num_filters=lin_embed_dim#+pos_embed_dim
        self.cls_token = nn.Parameter(torch.randn(1, 1, num_filters))
        self.GatedAttentionMILPooling =GatedAttentionPoolingMIL(input_dim=num_filters, hidden_dim=16)
        # A Mamba model is composed of a series of MambaBlocks interleaved
        # with normalization layers (e.g. RMSNorm)
#         self.layers = nn.ModuleList([
#             nn.ModuleList(
#                 [
#                     MambaBlock(**mamba_par),
#                     RMSNorm(d_input)
#                 ]
#             )
#             for _ in range(num_layers)
#         ])
#         layers=[]
#         for i in range(num_layers):
#             layers.append(BiMambaEncoder(num_filters, n_state,BI))
#         self.mambablocks_seq = nn.Sequential(*layers)
        self.mambablocks1 =BiMambaEncoder(num_filters, n_state,BI)
        self.mambablocks2 =BiMambaEncoder(num_filters, n_state,BI)
        self.mambablocks3 =BiMambaEncoder(num_filters, n_state,BI)
        self.mambablocks4 =BiMambaEncoder(num_filters, n_state,BI)
        self.mambablocks5 =BiMambaEncoder(num_filters, n_state,BI)
        self.mambablocks6 =BiMambaEncoder(num_filters, n_state,BI)
        self.mambablocks7 =BiMambaEncoder(num_filters, n_state,BI)
        self.mambablocks8 =BiMambaEncoder(num_filters, n_state,BI)
        self.out1 = nn.Linear(num_filters, 1)
        self.out2 = nn.Linear(num_filters, 1) 
        self.out3 = nn.Linear(num_filters, 1)

    def forward(self, x, x_lengths):
        batch_size, seq_len, _ = x.size()
        x=self.embedding(x)
        # Get positional embeddings for the current sequence length
#         pos_embed = self.pos_embedding(seq_len, x.device)  # (seq_len, embedding_dim)
#         # Expand positional embeddings to match the batch size
#         pos_embed = pos_embed.unsqueeze(0).expand(batch_size, seq_len, -1)  # (batch_size, seq_len, embedding_dim)
#         # Concatenate the positional embeddings with the original input
#         x = torch.cat([x, pos_embed], dim=-1) 
        cls_token = self.cls_token.expand(batch_size, -1, -1)  # Shape: (batch_size, 1, embed_dim)
        x = torch.cat((cls_token, x), dim=1)
        mask = create_mask(torch.tensor(x_lengths,device=x.device), seq_len+1,batch_size,x.device)
#         x = self.mambablocks1({0:x,1:mask})
#         x = self.mambablocks2({0:x,1:mask})
#         x = self.mambablocks3({0:x,1:mask})
#         x = self.mambablocks4({0:x,1:mask})
#         x = self.mambablocks5({0:x,1:mask})
#         x = self.mambablocks6({0:x,1:mask})
        x = self.mambablocks7({0:x,1:mask})
        mamba_out = self.mambablocks8({0:x,1:mask})
#         mamba_out = self.mambablocks_seq({0:x,1:mask})
#         masked=masked_avg_pool(mamba_out, mask)
        masked= self.GatedAttentionMILPooling(mamba_out, mask)
        x1 = self.out1(masked)
        x2 = self.out2(mamba_out[:, 1:, :]) #use .permute(0, 2, 1) if we want to use conv1d in output
        x3= self.out3(masked)
        #if x1 is 0 then duration should be 0
#         mask_x1 =torch.round(torch.sigmoid(x1))
#         x3 = x3 * (mask_x1 != 0).float()
        
        return x1,x2,x3

    
