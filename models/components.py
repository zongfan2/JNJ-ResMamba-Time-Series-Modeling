# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from typing import Optional, Tuple
import numpy as np
from .baselines import TCNLayer, ResTCNLayer


# Shared TCN and feature extraction components

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    
    def forward(self, x):
        if self.chomp_size == 0:
            return x
        else:
            return x[:, :, :-self.chomp_size]

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.comp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.comp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.comp1, self.relu1, self.dropout1, 
                                 self.conv2, self.comp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        # self.relu = nn.ReLU()

    def forward(self, x):
        res = x if self.downsample is None else self.downsample(x)
        out = self.net(x)
        return out + res


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()
        self.module = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            # nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            # nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(),
        )
        
    def forward(self, x):
        return self.module(x)



class FeatureExtractorConv2d(nn.Module):
    def __init__(self, in_channels, seq_len, target_shape, tcn_num_filters=64, tcn_kernel_size=7, tcn_num_blocks=4, **kwargs):
        # transform an input from (N, in_channels, seq_len) to (N, 3, target_shape, target_shape)
        super(FeatureExtractorConv2d, self).__init__()
        self.target_shape = target_shape
        tcn_layers = []
        n_inputs = in_channels
        for i in range(tcn_num_blocks):
            dilation = 2 ** i
            tcn_layers.append(TemporalBlock(n_inputs, tcn_num_filters, tcn_kernel_size, stride=1, 
                                            dilation=dilation, padding=(tcn_kernel_size-1)*dilation))
            n_inputs = tcn_num_filters
        self.tcn_encoder = nn.Sequential(*tcn_layers)
        
        
        # NOTE: flatten + linear discard the temporal information, replace it with pooling
        # in_channels = tcn_num_filters
        # OPTION1: linear + upsample layers
        # self.linear = nn.Sequential(
        #     nn.Linear(in_channels*seq_len, 128*8*8),
        #     nn.ReLU(),
        #     # nn.Dropout(0.2)
        #     )
        
        # OPTION2: 1D-to-2D reshape 
        # self.project = nn.Conv1d(tcn_num_filters, target_shape, kernel_size=1)
        self.project_only = (tcn_num_filters == in_channels)
        if self.project_only: 
            # direct upsample to target input size
            self.project = nn.Sequential(
                nn.Upsample(size=(target_shape, target_shape), mode="bilinear", align_corners=False),
                nn.Conv2d(tcn_num_filters, 3, kernel_size=3, padding=1),
                nn.BatchNorm2d(3))
        else:
            self.project = nn.Upsample(size=(32, 32), mode="bilinear", align_corners=False)
            if target_shape == 64:
                block_channels = [64]
            elif target_shape == 128:
                block_channels = [64, 32]
            elif target_shape == 256:
                block_channels = [64, 32, 16]
            else:
                raise ValueError("Target shape must be 64, 128 or 256")

            upsample_blocks = []
            for i in range(len(block_channels)-1):
                # upsample_blocks.append(nn.ConvTranspose2d(block_channels[i], block_channels[i+1], kernel_size=4, stride=2, padding=1, bias=False, ))
                # upsample_blocks.append(nn.BatchNorm2d(block_channels[i+1]))
                # upsample_blocks.append(nn.ReLU())
                upsample_blocks.append(UpsampleBlock(block_channels[i], block_channels[i+1]))
            upsample_blocks.append(nn.ConvTranspose2d(block_channels[-1], 3, kernel_size=4, stride=2, padding=1, bias=False))
            # use tanh
            # upsample_blocks.append(nn.Tanh())
            self.upsample = nn.Sequential(*upsample_blocks)

    def forward(self, x, x_lengths=None):
        x = self.tcn_encoder(x) # (N, tcn_num_filters, 1221)
        x = x.view(x.size(0), x.size(1), 33, 37) # 37*37=1221
        x = self.project(x)     # (N, tcn_num_filters, 32, 32)
        if not self.project_only:
            x = self.upsample(x)
        return x 

class FeatureExtractor(nn.Module):
    def __init__(self,
                 tsm_horizon,
                 in_channels,
                 pos_embed_dim,
                 norm="BN",
                 num_pos_channels=4,
                 kernel_size=3,
                 num_feature_layers=5,
                 num_filters=64,
                 tsm=False,
                 featurelayer="TCN"):
        super(FeatureExtractor, self).__init__()

        if pos_embed_dim != num_pos_channels:
            self.extract_pos_embed = True 
        else:
            self.extract_pos_embed = False
        layers=[]
        dilation = 1
        in_size=in_channels
        for i in range(num_feature_layers):
            # if i < 4:
            #     cur_num_filters = num_filters // 2
            # else:
            cur_num_filters = num_filters
            padding = (kernel_size - 1) * dilation // 2  # To maintain same length
            if featurelayer=="TCN":
                layers.append(TCNLayer(in_size, cur_num_filters, kernel_size, dilation, padding, norm))
            elif featurelayer=="ResNet":
                padding = (kernel_size - 1) * 1 // 2
                layers.append(ResTCNLayer(in_size, cur_num_filters, kernel_size, 1, padding, norm))
            elif featurelayer=="ResTCN":
                layers.append(ResTCNLayer(in_size, cur_num_filters, kernel_size, dilation, padding, norm))
            in_size = cur_num_filters
            dilation *= 2  # Increase dilation exponentially
        self.layers = nn.ModuleList(layers)
        self.tsm = tsm
        self.tsm_horizon = tsm_horizon

    def forward(self, x, mask, return_intermediates=False):
        """
        Forward pass with optional intermediate feature maps for UNet-style skip connections.
        
        Args:
            x: Input tensor
            mask: Input mask
            return_intermediates: If True, returns (x, intermediate_features), 
                                 else returns x only
        """
        intermediate_features = []
        
        for i, layer in enumerate(self.layers):
            x = layer(x, mask)
            intermediate_features.append(x.clone())
        
        if self.tsm:
            x = self.compute_crosscorrelation_batch(x)
        if return_intermediates:
            return x, intermediate_features
        return x    

    
    def compute_crosscorrelation_batch(self, features):
        """
        Compute the autocorrelation of the feature map.
        """ 
        batch_size, channels, length = features.size()
        unit = self.tsm_horizon * 16
        h = self.tsm_horizon
        device = features.device
        TSM = torch.zeros(batch_size, h, length, device=device)
        for i in range(batch_size):
            tsm_list=[]
            for b in range(0, length, unit):
                autocorr = torch.zeros(1, unit + h, unit + h, device=device)
                b_size = min(length - (b), unit + h)
                autocorr[0, :b_size, :b_size] = torch.corrcoef(features[i, :, b:b + b_size].T)[:, :]        
                mat = autocorr[0, :b_size, :b_size].squeeze(-0)
                mask_value = -1111
                row_indices = torch.arange(b_size + h, device=device).unsqueeze(1).repeat(1, b_size + h)  
                col_indices = torch.arange(b_size + h, device=device).unsqueeze(0).repeat(b_size + h, 1) 
                condition = ((row_indices) < (col_indices) ) & ( col_indices -row_indices<=(h))
                similar_matrix = torch.cat([torch.cat([mat,torch.zeros((b_size, h), dtype=torch.int)],dim=1),torch.zeros((h, b_size+h), dtype=torch.int)],dim=0)
                similar_matrix[~condition] = mask_value
                tsm=similar_matrix[:b_size,:][similar_matrix[:b_size,:] != mask_value].view(b_size,h)
                tsm_list.append(tsm[:min(length-(b),unit),:])
            TSM[i, :, :] =torch.cat(tsm_list,dim=0).unsqueeze(0).permute(0,2,1)
        return TSM.to(features.device) 
    
    def compute_crosscorrelation_batch_GPU(self, features):
        """
        Compute the autocorrelation of the feature map.
        """ 
        batch_size, channels, length = features.size()
        unit=self.tsm_horizon*8
        h=self.tsm_horizon
        device=features.device
        TSM = torch.zeros(batch_size, h, length,device=device)
        for i in range(batch_size):
            tsm_list=[]
            for b in range (0,length,unit):
                autocorr = torch.zeros(1,unit+h, unit+h,device=device)
                b_size=min(length-(b),unit+h)
                autocorr[0,:b_size,:b_size]=torch.corrcoef(features[i, :, b:b+b_size].T)[:,:]        
                mat=autocorr[0,:b_size,:b_size].squeeze(-0)
                mask_value=-1111
                row_indices = torch.arange(b_size+h,device=device).unsqueeze(1).repeat(1, b_size+h)  
                col_indices = torch.arange(b_size+h,device=device).unsqueeze(0).repeat(b_size+h, 1) 
                condition = ((row_indices) < (col_indices) ) & ( col_indices -row_indices<=(h))
                similar_matrix = torch.cat([torch.cat([mat,torch.zeros((b_size, h), dtype=torch.int,device=device)],dim=1),torch.zeros((h, b_size+h), dtype=torch.int,device=device)],dim=0)
                similar_matrix[~condition] = mask_value
                tsm=similar_matrix[:b_size,:][similar_matrix[:b_size,:] != mask_value].view(b_size,h)
                tsm_list.append(tsm[:min(length-(b),unit),:])
            TSM[i, :, :] =torch.cat(tsm_list,dim=0).unsqueeze(0).permute(0,2,1)
        return TSM#.to(features.device)  
    
    def compute_crosscorrelation(self, features):
        """
        Compute the autocorrelation of the feature map.
        """ 
        batch_size, channels, length = features.size()
        tsm = torch.zeros(batch_size, self.tsm_horizon, length)
        for i in range(batch_size):
            tsm_list=[]
            autocorr = torch.corrcoef(features[i, :, :].T)
            #autocorr=fast_corrcoef(features[i, :, :].T)
            for j in range(length):
                b_size=min(length-(j),self.tsm_horizon)
                tsm[i, 0:b_size, j] =autocorr[j:j+b_size,j]
        return tsm.to(features.device) 

class ConvProjection(nn.Module):
    """
    A single conv block with batch norm and dilated convolutions.
    """
    def __init__(self, in_channels, out_channels,norm,kernel_size=1,dropout_rate=0.2):
        super(ConvProjection, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels,kernel_size=kernel_size)
        if norm=='GN':
            self.norm = nn.GroupNorm(num_groups=4, num_channels=out_channels)
        elif norm=='BN':
            self.norm = nn.BatchNorm1d(out_channels)
        else:
            self.norm = nn.InstanceNorm1d(out_channels,track_running_stats=False)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class FeatureExtractorForPretraining(nn.Module):
    """
    Wrapper class that combines FeatureExtractor with specific configurations for 
    contrastive self-supervised pretraining (SimCLR, DINO, etc.). This class ensures 
    the feature extractor can be frozen during downstream fine-tuning while maintaining compatibility.
    """
    def __init__(self,
                 tsm_horizon=64,
                 in_channels=3,
                 pos_embed_dim=16,
                 norm="BN",
                 num_pos_channels=4,
                 kernel_size=3,
                 num_feature_layers=5,
                 num_filters=64,
                 tsm=False,
                 featurelayer="ResNet",
                 use_cls_token=True,
                #  projection_dim=128
                 ):
        super().__init__()
        
        self.input_projection = ConvProjection(in_channels, num_filters, norm)
        # Create the feature extractor with updated signature
        self.feature_extractor = FeatureExtractor(
            tsm_horizon=tsm_horizon,
            in_channels=num_filters,
            pos_embed_dim=pos_embed_dim,
            norm=norm,
            num_pos_channels=num_pos_channels,
            kernel_size=kernel_size,
            num_feature_layers=num_feature_layers,
            num_filters=num_filters,
            tsm=tsm,
            featurelayer=featurelayer
        )
        
        self.use_cls_token = use_cls_token
        self.num_filters = num_filters
        # self.projection_dim = projection_dim
        
        # Always create the pooling layer
        self.pooling = MultiHeadSelfAttentionPooling(
            input_dim=num_filters, hidden_dim=num_filters, num_heads=8
        )
        
        if self.use_cls_token:
            # CLS token with num_filters dimensions (after input_projection)
            self.cls_token = nn.Parameter(torch.randn(1, 1, num_filters))
        else:
            self.cls_token = None
        
        # Projection head for contrastive learning
        # self.projection_head = nn.Sequential(
        #     nn.Linear(num_filters, num_filters),
        #     nn.ReLU(),
        #     nn.Linear(num_filters, projection_dim)
        # )
        
        # Store configuration for downstream model initialization
        self.config = {
            'tsm_horizon': tsm_horizon,
            'in_channels': in_channels,
            'pos_embed_dim': pos_embed_dim,
            'norm': norm,
            'num_pos_channels': num_pos_channels,
            'kernel_size': kernel_size,
            'num_feature_layers': num_feature_layers,
            'num_filters': num_filters,
            'tsm': tsm,
            'featurelayer': featurelayer,
            'use_cls_token': use_cls_token,
            # 'projection_dim': projection_dim
        }
    
    def forward(self, x, x_lengths, **kwargs):
        """
        Forward pass for contrastive pretraining.
        
        Args:
            x: Input tensor [batch_size, seq_len, channels]
            x_lengths: Original lengths of each sequence in the batch (before padding)
        
        Returns:
            If return_features and return_projection: (sequence_features, pooled_features, projected_features)
        """

        batch_size, seq_len, _ = x.size()
        # Create mask based on x_lengths
        mask = create_mask(torch.tensor(x_lengths, device=x.device), seq_len, batch_size, x.device)
        
        # Apply input projection first
        x_projected = self.input_projection(x.permute(0, 2, 1))  # [batch, num_filters, seq]
        
        # Add CLS token after input projection (when we have richer num_filters dimensions)
        if self.use_cls_token:
            # Add CLS token after input_projection (better than 3-channel raw input)
            batch_size, num_filters, seq_len = x_projected.size()
            
            # Add CLS token with num_filters dimensions
            cls_tokens = self.cls_token.expand(batch_size, -1, -1).permute(0, 2, 1)  # [batch, num_filters, 1]
            x_with_cls = torch.cat([cls_tokens, x_projected], dim=2)  # [batch, num_filters, seq+1]
            
            # Create extended mask for CLS token + sequence (CLS token is always valid)
            cls_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=mask.device)
            extended_mask = torch.cat([cls_mask, mask], dim=1)  # [batch, seq+1]
            
            # Extract features using the feature extractor (CLS token learns through the network)
            features = self.feature_extractor(x_with_cls, extended_mask)
            # Extract the CLS token embedding (position 0, has learned through feature extractor)
            cls_embedding = features[:, :, 0]  # [batch, num_filters]
        else:
            # Standard approach without CLS token
            features = self.feature_extractor(x_projected, mask)
            
            # Apply MultiHeadSelfAttentionPooling
            features_transposed = features.permute(0, 2, 1)  # [batch, seq, num_filters]
            pooled_features, weighted_features, attention_weights = self.pooling(
                features_transposed, mask
            )
            cls_embedding = pooled_features  # Use pooled features as representation
        
        # Apply projection head for contrastive learning on CLS embedding
        # proj_features = self.projection_head(cls_embedding)  # (batch_size, projection_dim)
        # return proj_features
        return cls_embedding

    def freeze_features(self):
        """Freeze the feature extractor parameters for downstream fine-tuning."""
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def unfreeze_features(self):
        """Unfreeze the feature extractor parameters."""
        for param in self.feature_extractor.parameters():
            param.requires_grad = True
    
    def load_pretrained_features(self, checkpoint_path):
        """Load pretrained feature extractor weights."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'feature_extractor' in checkpoint:
            self.feature_extractor.load_state_dict(checkpoint['feature_extractor'])
        elif 'backbone' in checkpoint:
            if isinstance(checkpoint['backbone'], dict):
                if 'feature_extractor' in checkpoint['backbone']:
                    self.feature_extractor.load_state_dict(checkpoint['backbone']['feature_extractor'])
                else:
                    self.feature_extractor.load_state_dict(checkpoint['backbone'])
            else:
                self.feature_extractor.load_state_dict(checkpoint['backbone'].state_dict())
        elif 'model' in checkpoint:
            # Handle cases where the entire model is saved
            if 'feature_extractor' in checkpoint['model']:
                self.feature_extractor.load_state_dict(checkpoint['model']['feature_extractor'])
            else:
                # Try to load compatible weights
                self.load_state_dict(checkpoint['model'], strict=False)
        else:
            # Assume the entire checkpoint is the feature extractor state dict
            self.feature_extractor.load_state_dict(checkpoint)
    

