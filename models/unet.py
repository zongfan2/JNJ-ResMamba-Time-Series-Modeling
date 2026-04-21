# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from typing import Optional, Tuple
import numpy as np
from torchvision import models

from .components import FeatureExtractorConv2d
from .specialized import PredictHead, create_mask


# U-Net and decoder-based models

class EfficientUNet(nn.Module):
    def __init__(self, prediction_length, 
                       model_name="efficientnet_v2_m", 
                       d_model=1024, 
                       input_dim=3,
                    #    tsm_horizon=64, 
                    #    pos_embed_dim=16, 
                    #    num_filters=64,
                    #    kernel_size_feature=13,
                    #    num_feature_layers=7,
                    #    tsm=False,
                    #    featurelayer="ResTCN",
                       target_shape=128,
                       tcn_params={},
                       reg_head=True,
                       mask_padding=True,
                       average_mask=False,
                       average_window_size=20):
        super(EfficientUNet, self).__init__()

        # self.featurelayer = featurelayer
        # if use out3 layer to predict scratch duration
        self.reg_head = reg_head
        self._max_seq_len = prediction_length
        self.feature_extractor = FeatureExtractorConv2d(input_dim, prediction_length, target_shape, **tcn_params)
        # if self.featurelayer == "Conv2d":
        #     self.feature_extractor = FeatureExtractorConv2d(input_dim, prediction_length, target_shape, **tcn_params)
        # else:
        #     self.feature_extractor = FeatureExtractor(tsm_horizon,
        #                                             input_dim,
        #                                             pos_embed_dim,
        #                                             num_filters=num_filters,
        #                                             kernel_size=kernel_size_feature,
        #                                             num_feature_layers=num_feature_layers,
        #                                             tsm=tsm,
        #                                             featurelayer=featurelayer)
        
        # Encoder: EfficientNet-B3 from torchvision
        if model_name == "efficientnet_v2_m":
            self.encoder = models.efficientnet_v2_m(pretrained=True)  
            self.cs = [80, 176, 1280]
        elif model_name == "efficientnet_b3":
            self.encoder = models.efficientnet_b3(pretrained=True)  
            self.cs = [48, 136, 1536]
        # Load EfficientNet-B3
        self.encoder_blocks = [
            self.encoder.features[0:4],  # First block (stem + initial conv)
            self.encoder.features[4:6],  # Second block
            self.encoder.features[6:],    # Remaining blocks
        ]
        # for v2_m: 80, 176, 1280
        # for b3: 48, 136, 1536
        # Decoding layers
        if model_name == "efficientnet_v2_m":
            self.decoder1 = nn.ConvTranspose2d(1280, 176, kernel_size=3, stride=2, padding=1)  
            self.decoder2 = nn.ConvTranspose2d(176, 80, kernel_size=3, stride=2, padding=1)
            # self.decoder3 = nn.ConvTranspose2d(80, 1, kernel_size=3, stride=2, padding=1)   
            # Batch Normalization layers
            self.bn1 = nn.BatchNorm2d(176)
            self.bn2 = nn.BatchNorm2d(80)
            self.conv = nn.Conv2d(80, d_model, kernel_size=3, stride=2, padding=1)
        else:
            self.decoder1 = nn.ConvTranspose2d(1536, 136, kernel_size=3, stride=2, padding=1)  
            self.decoder2 = nn.ConvTranspose2d(136, 48, kernel_size=3, stride=2, padding=1)
            # self.decoder3 = nn.ConvTranspose2d(48, 1, kernel_size=3, stride=2, padding=1)  
            self.bn1 = nn.BatchNorm2d(136)
            self.bn2 = nn.BatchNorm2d(48)
            self.conv = nn.Conv2d(48, d_model, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(d_model)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = PredictHead(prediction_length, 
                                d_model, 
                                reg_head=reg_head, 
                                mask_padding=mask_padding, 
                                average_mask=average_mask,
                                average_window_size=average_window_size)

    def forward(self, x, x_lengths=None, labels1=None, labels3=None,
                apply_mixup=False, mixup_alpha=0.2, max_seq_len=None, **kwargs):
        # FeatureExtractorConv2d hardcodes reshape to 33x37=1221, so input
        # must match self._max_seq_len. Right-pad batch-local shorter inputs.
        if x.dim() == 3 and x.shape[1] < self._max_seq_len:
            pad_len = self._max_seq_len - x.shape[1]
            x = F.pad(x, (0, 0, 0, pad_len))
        elif x.dim() == 3 and x.shape[1] > self._max_seq_len:
            x = x[:, : self._max_seq_len, :]
        x = self.feature_extractor(x.permute(0, 2, 1), x_lengths) # [batch size, num_filters, pred_len]

        # out2 has fixed prediction_length == self._max_seq_len, so always
        # use that for the mask (ignore batch-local pad-length passed in).
        max_seq_len = self._max_seq_len

        # Encoder Stage
        features = []  # Container for saving encoder outputs
        for block in self.encoder_blocks:
            x = block(x)  # Forward pass through each encoder block
            features.append(x)
        # Decoder Stage with Skip Connections
        x = self.decoder1(x)  # First Decoder
        x = features[1] + nn.functional.interpolate(x, size=features[1].shape[2:], mode='nearest')  # Skip connection
        x = self.bn1(x)
        x = torch.relu(x)

        x = self.decoder2(x)  # Second Decoder
        x = features[0] + nn.functional.interpolate(x, size=features[0].shape[2:], mode='nearest')  # Skip connection
        x = self.bn2(x)
        x = torch.relu(x)

        # conv
        x = self.conv(x)
        x = self.bn3(x)
        x = torch.relu(x)
        # pool
        x = self.pool(x)
        x = x.squeeze()

        out1, out2, out3 = self.head(x, x_lengths, max_seq_len)
        return out1, out2, out3, None, None, None

class EncoderStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, downsample):
        super().__init__()
        blocks = []
        blocks.append(ResBlock1D(in_channels, out_channels, stride=2 if downsample else 1))
        for _ in range(1, num_blocks):
            blocks.append(ResBlock1D(out_channels, out_channels, stride=1))
        self.stage = nn.Sequential(*blocks)
    def forward(self, x):
        return self.stage(x)

class DecoderBlock(nn.Module):
    def __init__(self, up_in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose1d(up_in_channels, skip_channels, kernel_size=2, stride=2)
        self.resblock = ResBlock1D(skip_channels * 2, out_channels)
    def forward(self, x, skip):
        x = self.up(x)
        # Align lengths for concat
        if x.size(-1) < skip.size(-1):
            x = F.pad(x, (0, skip.size(-1) - x.size(-1)))
        elif x.size(-1) > skip.size(-1):
            x = x[..., :skip.size(-1)]
        x = torch.cat([x, skip], dim=1)
        x = self.resblock(x)
        return x

class OutModule2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, norm, method):
        super(OutModule2, self).__init__()
        if norm=='GN':
            self.norm = nn.GroupNorm(num_groups=4, num_channels=hidden_dim)
        elif norm=='BN':
            self.norm = nn.BatchNorm1d(hidden_dim)
        else:
            self.norm = nn.InstanceNorm1d(hidden_dim,track_running_stats=False)
            
        if method=='FC':
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            # Second layer: hidden to output
            self.fc2 = nn.Linear(hidden_dim,hidden_dim)
            # self.fc3 = nn.Linear(hidden_dim,hidden_dim)
            # self.fc4 = nn.Linear(hidden_dim,hidden_dim)
            self.fc5 = nn.Linear(hidden_dim, output_dim)
        else:
            self.fc1 = nn.Conv1d(input_dim, hidden_dim,1)
            # Second layer: hidden to output
            self.fc2 = nn.Conv1d(hidden_dim, hidden_dim, 1)
            # self.fc3 = nn.Conv1d(hidden_dim, hidden_dim, 1)
            # self.fc4 = nn.Conv1d(hidden_dim, hidden_dim, 1)
            self.fc5 = nn.Conv1d(hidden_dim,output_dim, 1)
        
    def forward(self, x):
        if self.norm is not None:
            x=self.norm(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        # x = self.fc3(x)
        # x = F.relu(x)
        # x = self.fc4(x)
        # x = F.relu(x)
        x = self.fc5(x)
        return x

class ResNetUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, features=[64, 128, 256, 512], blocks_per_stage=[2, 2, 2, 2]):
        super().__init__()
        self.init_conv = nn.Conv1d(in_channels, features[0], kernel_size=7, stride=2, padding=3)
        self.init_bn = nn.InstanceNorm1d(features[0])
        self.init_relu = nn.ReLU(inplace=True)
        self.init_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        # Encoder
        self.encoder_stages = nn.ModuleList()
        prev_channels = features[0]
        for idx, (feat, n_blocks) in enumerate(zip(features, blocks_per_stage)):
            self.encoder_stages.append(
                EncoderStage(prev_channels, feat, num_blocks=n_blocks, downsample=(idx != 0))
            )
            prev_channels = feat
        # Bottleneck
        self.bottleneck = ResBlock1D(features[-1], features[-1]*2, stride=1)
        # Decoder
        rev_features = features[::-1]
        up_in_channels = features[-1]*2
        self.decoder_blocks = nn.ModuleList()
        for skip_channels in rev_features:
            self.decoder_blocks.append(
                DecoderBlock(up_in_channels, skip_channels, skip_channels)
            )
            up_in_channels = skip_channels

        self.out1 = OutModule2(features[0],features[0], num_classes,'IN','FC')  
        self.out2 = OutModule2(features[0],features[0], num_classes,'IN','Conv')
        self.out3 = OutModule2(features[0],features[0], num_classes,'IN','FC')
        
    def forward(self, x, x_lengths,labels1=None, labels3=None, apply_mixup=False, mixup_alpha=0.2):
        batch_size, seq_len, channels = x.size()
        mask = create_mask(torch.tensor(x_lengths,device=x.device), seq_len,batch_size,x.device)
#         x = self.input_projection(x.permute(0, 2, 1))
        contrastive_embedding=None
        x=x.permute(0, 2, 1)
        input_length = x.size(-1)
        x = self.init_conv(x)
        x = self.init_bn(x)
        x = self.init_relu(x)
        x = self.init_pool(x)
        skips = []
        for enc in self.encoder_stages:
            x = enc(x)
            skips.append(x)
        x = self.bottleneck(x)
        for dec, skip in zip(self.decoder_blocks, reversed(skips)):
            x = dec(x, skip)
        #x = self.final_conv(x)
        # Ensure output length matches input length
        if x.size(-1) < input_length:
            x = F.pad(x, (0, input_length - x.size(-1)))
        elif x.size(-1) > input_length:
            x = x[..., :input_length]
        
        attention_weights = torch.zeros(batch_size, seq_len,1,device=x.device)
        pooled_features = x[:,:,0]
        mixed_features = pooled_features
        mixup_info = None
        x1 = self.out1(mixed_features)
        x2 = self.out2(x) 
        x3= self.out3(mixed_features)


        mask=mask.reshape(-1).bool()
        x2=x2.view(-1)
        x2=x2[mask]
        attention_weights=attention_weights.reshape(-1)
        attention_weights=attention_weights[mask]

        return x1,x2,x3,attention_weights,contrastive_embedding, mixup_info

class MambaDecoderBlock(nn.Module):
    """
    Mamba-based decoder block for symmetric design with ResNet encoder.
    Replaces ResBlock1D with Mamba processing while maintaining upsampling.
    """
    def __init__(self, up_in_channels, skip_channels, out_channels, kernel_size_mba=7, drop_path_rate=0.3):
        super().__init__()
        # Upsampling layer
        # self.up = nn.ConvTranspose1d(up_in_channels, skip_channels, kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        self.channel_align = nn.Conv1d(up_in_channels+skip_channels, out_channels, 1)

        # Channel alignment before Mamba processing
        # self.channel_align = nn.Conv1d(skip_channels * 2, out_channels, 1)
        # out_channels = skip_channels * 2
        self.norm = nn.InstanceNorm1d(out_channels)

        # Mamba block for processing concatenated features
        self.mamba_block = AttModule_mamba_causal(
            dilation=1,  # No dilation needed for decoder
            in_channels=out_channels,
            out_channels=out_channels,
            att_type='sliding_att',
            stage='decoder',
            alpha=1,
            drop_path_rate=drop_path_rate,
            kernel_size=kernel_size_mba
        )

    def forward(self, x, skip, mask=None):
        # Upsample decoder features
        x = self.up(x)

        # Align sequence lengths for concatenation

        if x.size(-1) < skip.size(-1):
            x = F.pad(x, (0, skip.size(-1) - x.size(-1)))
        elif x.size(-1) > skip.size(-1):
            x = x[:, :, :skip.size(-1)]

        # Concatenate upsampled features with skip connection
        x = torch.cat([x, skip], dim=1)  # [B, skip_channels*2, seq_len]

        # Channel alignment and normalization
        x = self.channel_align(x)  # [B, out_channels, seq_len]
        x = self.norm(x)

        if mask is None:
            batch_size, seq_len = x.size(0), x.size(2)
            mask = torch.ones(batch_size, seq_len, device=x.device, dtype=torch.float32)
        # Mamba processing with optional mask
        x = self.mamba_block(x, None, mask)

        return x

class MambaDecoderStage(nn.Module):
    """
    Multi-block Mamba decoder stage to match encoder complexity.
    """
    def __init__(self, up_in_channels, skip_channels, out_channels, num_blocks=2,
                 kernel_size_mba=7, drop_path_rate=0.3):
        super().__init__()

        # First block handles upsampling and skip connection
        self.first_block = MambaDecoderBlock(
            up_in_channels, skip_channels, out_channels,
            kernel_size_mba, drop_path_rate
        )

        # Additional Mamba blocks for processing
        # self.additional_blocks = nn.ModuleList([
        #     AttModule_mamba(
        #         dilation=1,
        #         in_channels=out_channels,
        #         out_channels=out_channels,
        #         r1=2, r2=2,
        #         att_type='sliding_att',
        #         stage='decoder',
        #         alpha=1,
        #         drop_path_rate=drop_path_rate,
        #         kernel_size=kernel_size_mba
        #     ) for _ in range(num_blocks - 1)
        # ])

    def forward(self, x, skip, mask=None):
        # First block with skip connection
        x = self.first_block(x, skip, mask)

        # Additional processing blocks

        # for block in self.additional_blocks:
        #     if mask is None:
        #         batch_size, seq_len = x.size(0), x.size(2)
        #         block_mask = torch.ones(batch_size, seq_len, device=x.device, dtype=torch.float32)
        #     else:
        #         block_mask = mask
        #     x = block(x, None, block_mask)

        return x

class ResNetMambaUNet(nn.Module):
    """
    Hybrid encoder-decoder with ResNet encoder and Mamba decoder.
    Maintains symmetric design with equivalent complexity at each stage.
    """
    def __init__(self, in_channels=3, num_classes=1,
                 features=[64, 128, 256, 512],
                 blocks_per_stage=[1, 1, 1, 1],  # Reduced from [2,2,2,2] to prevent complexity
                 kernel_size_mba=7,
                 drop_path_rate=0.2,  # Reduced from 0.3
                 dropout_rate=0.2,
                 use_mask_in_decoder=True):
        super().__init__()

        self.use_mask_in_decoder = use_mask_in_decoder

        # Initial convolution (same as ResNetUNet)
        self.init_conv = nn.Conv1d(in_channels, features[0], kernel_size=7, stride=2, padding=3)
        self.init_bn = nn.InstanceNorm1d(features[0])
        self.init_relu = nn.ReLU(inplace=True)
        self.init_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # ResNet Encoder (same as ResNetUNet)
        self.encoder_stages = nn.ModuleList()
        prev_channels = features[0]
        for idx, (feat, n_blocks) in enumerate(zip(features, blocks_per_stage)):
            self.encoder_stages.append(
                EncoderStage(prev_channels, feat, num_blocks=n_blocks, downsample=(idx != 0))
            )
            prev_channels = feat

        # Bottleneck with Mamba processing - reduced alpha for stability
        self.bottleneck_conv = ResBlock1D(features[-1], features[-1]*2, stride=1)
        self.bottleneck_mamba = MaskMambaBlock(
            features[-1]*2, 
            drop_path_rate=min(drop_path_rate, 0.1),  # Very conservative drop path
            kernel_size=kernel_size_mba
        )

        # Mamba Decoder - use 3 stages to match encoder skip connections
        # Skip the deepest level since bottleneck already processed it
        rev_features = features[-2::-1]  # Exclude features[-1], reverse the rest
        rev_blocks = blocks_per_stage[-2::-1]  # Exclude blocks_per_stage[-1], reverse
        up_in_channels = features[-1]*2
        self.decoder_stages = nn.ModuleList()

        for skip_channels, n_blocks in zip(rev_features, rev_blocks):
            self.decoder_stages.append(
                MambaDecoderStage(
                    up_in_channels, skip_channels, skip_channels,
                    num_blocks=n_blocks,
                    kernel_size_mba=kernel_size_mba,
                    drop_path_rate=min(drop_path_rate, 0.2)  # Cap drop path rate for stability
                )
            )
            up_in_channels = skip_channels

        # Multi-task output heads (same as ResNetUNet)
        self.out1 = OutModule2(features[0], features[0], num_classes, 'IN', 'FC')
        self.out2 = OutModule2(features[0], features[0], num_classes, 'IN', 'Conv')
        self.out3 = OutModule2(features[0], features[0], num_classes, 'IN', 'FC')
        

    def forward(self, x, x_lengths, labels1=None, labels3=None, apply_mixup=False, mixup_alpha=0.2):
        batch_size, seq_len, channels = x.size()
        mask = create_mask(torch.tensor(x_lengths, device=x.device), seq_len, batch_size, x.device)

        contrastive_embedding = None
        x = x.permute(0, 2, 1)  # [B, C, L]
        input_length = x.size(-1)

        # Initial processing
        x = self.init_conv(x)
        x = self.init_bn(x)
        x = self.init_relu(x)
        x = self.init_pool(x)

        # Encoder with skip connections
        skips = []
        for stage in self.encoder_stages:
            x = stage(x)
            skips.append(x)

        # Bottleneck
        x = self.bottleneck_conv(x)

        # Create mask for current sequence length
        current_length = x.size(-1)
        current_mask = F.interpolate(
            mask.unsqueeze(1).float(),
            size=current_length,
            mode='nearest'
        ).squeeze(1)

        # Use mask in bottleneck only if decoder masking is enabled
        bottleneck_mask = current_mask if self.use_mask_in_decoder else None
        x = self.bottleneck_mamba(x, bottleneck_mask)

        # Mamba decoder with skip connections
        # Skip the deepest encoder output since bottleneck processed it
        for stage, skip in zip(self.decoder_stages, skips[-2::-1]):
            if self.use_mask_in_decoder:
                # Interpolate mask to match skip connection sequence length
                # The decoder stage will upsample x to match skip.size(-1)
                current_length = skip.size(-1)
                decoder_mask = F.interpolate(
                    mask.unsqueeze(1).float(),
                    size=current_length,
                    mode='nearest'
                ).squeeze(1)
            else:
                decoder_mask = None

            x = stage(x, skip, decoder_mask)

        # Final upsampling to match input length
        if x.size(-1) < input_length:
            x = F.pad(x, (0, input_length - x.size(-1)))
        elif x.size(-1) > input_length:
            x = x[..., :input_length]

        # Match ResNetUNet output format
        attention_weights = torch.zeros(batch_size, seq_len, 1, device=x.device)
        pooled_features = x[:, :, 0]  # Use first position features for classification/regression
        mixed_features = pooled_features
        mixup_info = None

        # Multi-task outputs (matching ResNetUNet)
        x1 = self.out1(mixed_features)  # Classification from pooled features
        x2 = self.out2(x)               # Sequence labeling from full sequence
        x3 = self.out3(mixed_features)  # Regression from pooled features

        # Apply mask to sequence output (matching ResNetUNet)
        mask = mask.reshape(-1).bool()
        x2 = x2.view(-1)
        x2 = x2[mask]
        attention_weights = attention_weights.reshape(-1)
        attention_weights = attention_weights[mask]

        return x1, x2, x3, attention_weights, contrastive_embedding, mixup_info
    

