# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from typing import Optional, Tuple
import numpy as np


# 1D ResNet architectures

class BasicBlock1D(nn.Module):
    """Basic 1D ResNet block with two convolutional layers and residual connection"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, kernel_size=15):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, 
                               stride=stride, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, 
                               stride=1, padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck1D(nn.Module):
    """Bottleneck 1D ResNet block with three convolutional layers and residual connection"""
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, kernel_size=15):
        super(Bottleneck1D, self).__init__()
        width = out_channels
        self.conv1 = nn.Conv1d(in_channels, width, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm1d(width)
        self.conv2 = nn.Conv1d(width, width, kernel_size=kernel_size, stride=stride,
                               padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm1d(width)
        self.conv3 = nn.Conv1d(width, out_channels * self.expansion, kernel_size=1,
                               stride=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet1D(nn.Module):
    """
    ResNet architecture for 1D time series data.
    Can be configured for different depths (18, 34, 50, etc.) by specifying block type and layers.
    """
    def __init__(
        self,
        time_length: int = 1221,        # Total length of input sequence
        in_channels: int = 3,           # Number of input channels
        base_filters: int = 64,         # Base number of filters
        block_type: str = 'basic',      # 'basic' or 'bottleneck'
        layers: list = [2, 2, 2, 2],    # Number of blocks in each layer (ResNet18 style)
        kernel_size: int = 15,          # Kernel size for convolutions
        dropout: float = 0.1,           # Dropout rate
        num_classes: int = 1,           # Number of output classes
        use_cls_token: bool = True,      # Whether to use a CLS token for classification
        second_level_mask: bool = False,          # Predict mask at second level
        supcon_loss: bool = False,      # Whether to return contrastive embedding for for supcon loss
    ):
        super(ResNet1D, self).__init__()
        
        # Initialize parameters
        self.time_length = time_length
        self.in_channels = in_channels
        self.base_filters = base_filters
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.num_classes = num_classes
        self.use_cls_token = use_cls_token
        self.return_contrastive_embedding = supcon_loss
        
        # Select block type
        if block_type == 'basic':
            block = BasicBlock1D
        elif block_type == 'bottleneck':
            block = Bottleneck1D
        else:
            raise ValueError(f"Unknown block type: {block_type}")
        
        self.expansion = block.expansion
        self.inplanes = base_filters
        
        # Initial convolution and pooling
        self.input_projection = nn.Sequential(
            nn.Conv1d(in_channels, base_filters, kernel_size=kernel_size, 
                      stride=2, padding=kernel_size//2, bias=False),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        # ResNet layers
        self.layer1 = self._make_layer(block, base_filters, layers[0], stride=1, kernel_size=3)
        self.layer2 = self._make_layer(block, base_filters*2, layers[1], stride=2, kernel_size=3)
        self.layer3 = self._make_layer(block, base_filters*4, layers[2], stride=2, kernel_size=3)
        self.layer4 = self._make_layer(block, base_filters*8, layers[3], stride=2, kernel_size=3)
        
        # In relcon, 3 layers with: 
        # 1: n=3, c_in=64, base_filters=64, k=3, s=1
        # 2: n=4, c_in=64, base_filters=128, k=4, s=1
        # 3: n=6, c_in=128, base_filters=256, k=4, s=1
        # self.layer1 = self._make_layer(block, base_filters, layers[0], stride=1, kernel_size=3)
        # self.layer2 = self._make_layer(block, base_filters*2, layers[1], stride=1, kernel_size=4)
        # self.layer3 = self._make_layer(block, base_filters*4, layers[2], stride=1, kernel_size=4)
        


        # Store all ResNet blocks for consistent interface
        self.conv_blocks = [
            self.layer1, self.layer2, self.layer3, self.layer4
        ]
        # self.conv_blocks = [
        #     self.layer1, self.layer2, self.layer3
        # ]
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Feature dimension after encoding
        self.embed_dim = base_filters * 8 * block.expansion
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output heads
        self.classification_head = nn.Linear(self.embed_dim, num_classes)
        self.regression_head = nn.Linear(self.embed_dim, 1)
        
        # Mask prediction head with a separate pathway
        if second_level_mask:
            mask_length = time_length // 20
        else:
            mask_length = time_length
        
        self.mask_prediction_head = nn.Linear(self.embed_dim, mask_length)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, kernel_size=15):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, kernel_size=kernel_size))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, kernel_size=kernel_size))

        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, labels1: Optional[torch.Tensor]=None, labels3: Optional[torch.Tensor]=None, apply_mixup: bool=False, mixup_alpha: float=0.2, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through ResNet1D
        
        Args:
            x: Input tensor of shape [B, C, L]
                B = batch size, C = channels, L = sequence length
                
        Returns:
            classification_output: Classification output
            mask_prediction_output: Mask prediction output
            regression_output: Regression output
        """
        x = x.permute(0, 2, 1)
        # Input projection
        x = self.input_projection(x)
        
        # Process through ResNet blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Get pooled features
        pooled = self.global_avg_pool(x).squeeze(-1)
        # features = self.dropout(features)
        # Task-specific outputs

        if apply_mixup and self.training:
            mixed_features, mixed_labels1, mixed_labels3, mixup_lambda = latent_mixup(
                pooled, labels1, labels3, alpha=mixup_alpha
            )
            mixup_info = (mixed_labels1, mixed_labels3, mixup_lambda)
        else:
            mixed_features = pooled
            mixup_info = None
        
        # Classification output
        classification_output = self.classification_head(mixed_features)
        if self.return_contrastive_embedding:
            con_embed = self.pooled
        else:
            con_embed = None
        
        # Mask prediction output
        mask_output = self.mask_prediction_head(mixed_features)
        
        # Regression output
        regression_output = self.regression_head(mixed_features)
        
        attn = None
        return classification_output,  mask_output, regression_output, attn, con_embed, mixup_info


class ResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels,kernel_size, stride, padding)
        self.bn1 = nn.InstanceNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels,kernel_size, 1, padding)
        self.bn2 = nn.InstanceNorm1d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, 1, stride)
        else:
            self.shortcut = nn.Identity()
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

