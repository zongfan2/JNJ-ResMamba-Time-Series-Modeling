# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from typing import Optional, Tuple
import numpy as np
from mamba_ssm import Mamba

from .mamba_blocks import MBA, AffineDropPath
from .attention import MultiHeadSelfAttentionPooling
from .components import FeatureExtractor

# Pretraining models

class MBATSMForPretraining(nn.Module):
    """
    MBA-TSM variant designed specifically for pretraining the Mamba encoders.
    This class removes the feature extractor and focuses on pretraining the Mamba blocks directly.
    When use_cls_token is True, uses the first token instead of adding a new CLS token.
    """
    def __init__(self,
                 input_dim=3,
                 num_filters=64,
                 num_encoder_layers=3,
                 drop_path_rate=0.3,
                 kernel_size_mba=7,
                 dropout_rate=0.2,
                 mba=True,
                 add_positional_encoding=True,
                 max_seq_len=2400,
                 num_heads=4,
                 use_cls_token=True,
                 pretraining=True,
                #  projection_dim=128,
                 norm="BN"):
        super(MBATSMForPretraining, self).__init__()
        
        self.add_positional_encoding = add_positional_encoding
        self.use_cls_token = use_cls_token
        self.num_filters = num_filters
        self.pretraining = pretraining
        # self.projection_dim = projection_dim
        
        # Input projection to convert raw input to num_filters dimensions
        self.input_projection = ConvProjection(input_dim, num_filters, norm)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model=num_filters, max_len=max_seq_len)
        
        # Mamba encoder layers - this is what we want to pretrain
        if not mba:
            self.encoder = nn.ModuleList([
                AttModule(2 ** i, num_filters, num_filters, 2, 2, 'normal_att', 'encoder', alpha=1,
                         kernel_size=kernel_size_mba, dropout_rate=dropout_rate) 
                for i in range(num_encoder_layers)
            ])
        else:
            self.encoder = nn.ModuleList([
                AttModule_mamba(2 ** i, num_filters, num_filters, 'sliding_att', 'encoder', 1, 
                               drop_path_rate=drop_path_rate, dropout_rate=dropout_rate, 
                               kernel_size=kernel_size_mba, norm=norm) 
                for i in range(num_encoder_layers)
            ])
        
        # Pooling for sequence-level representation
        self.pooling = MultiHeadSelfAttentionPooling(
            input_dim=num_filters, hidden_dim=num_filters, num_heads=num_heads
        )
        
        # Prediction heads for fine-tuning (following MBA_tsm design)
        if not pretraining:
            self.out1 = OutModule2(num_filters, num_filters, 1, norm, 'FC')  # Binary classification
            self.out2 = OutModule2(num_filters, num_filters, 1, norm, 'Conv')  # Sequence prediction
            self.out3 = OutModule2(num_filters, num_filters, 1, norm, 'FC')  # Regression/severity
        
        # Projection head for contrastive learning
        # self.projection_head = nn.Sequential(
        #     nn.Linear(num_filters, num_filters),
        #     nn.ReLU(),
        #     nn.Linear(num_filters, projection_dim)
        # )
        
        # Store configuration for downstream model initialization
        self.config = {
            'input_dim': input_dim,
            'num_filters': num_filters,
            'num_encoder_layers': num_encoder_layers,
            'drop_path_rate': drop_path_rate,
            'kernel_size_mba': kernel_size_mba,
            'dropout_rate': dropout_rate,
            'mba': mba,
            'add_positional_encoding': add_positional_encoding,
            'max_seq_len': max_seq_len,
            'num_heads': num_heads,
            'use_cls_token': use_cls_token,
            # 'projection_dim': projection_dim,
            'norm': norm
        }
    
    def forward(self, x, x_lengths, **kwargs):
        """
        Forward pass for both pretraining and fine-tuning.
        
        Args:
            x: Input tensor [batch_size, seq_len, channels]
            x_lengths: Original lengths of each sequence in the batch (before padding)
        
        Returns:
            If pretraining=True: pooled_features (for contrastive learning)
            If pretraining=False: (x1, x2, x3, attention_weights, None, None) following MBA_tsm format
        """
        batch_size, seq_len, _ = x.size()
        
        # Create mask based on x_lengths
        mask = create_mask(torch.tensor(x_lengths, device=x.device), seq_len, batch_size, x.device)
        
        # Apply input projection: [batch, seq, input_dim] -> [batch, input_dim, seq] -> [batch, num_filters, seq]
        x_projected = self.input_projection(x.permute(0, 2, 1))  # [batch, num_filters, seq]
        
        # Add positional encoding if enabled
        if self.add_positional_encoding:
            x_projected = self.positional_encoding(x_projected)
        
        # Apply Mamba encoder layers
        features = x_projected
        for layer in self.encoder:
            features = layer(features, None, mask)  # [batch, num_filters, seq]
        
        # Convert to [batch, seq, num_filters] for pooling
        features_transposed = features.permute(0, 2, 1)
        
        if self.pretraining:
            # Pretraining mode: return pooled features for contrastive learning
            if self.use_cls_token:
                # Use the first token as the CLS token (already learned through the network)
                cls_embedding = features_transposed[:, 0, :]  # [batch, num_filters]
                pooled_features = cls_embedding
            else:
                # Use attention pooling for sequence-level representation
                pooled_features, weighted_features, attention_weights = self.pooling(
                    features_transposed, mask
                )
            return pooled_features
        else:
            # Fine-tuning mode: return predictions following MBA_tsm format
            if self.use_cls_token:
                # Extract CLS token for classification tasks
                cls_embedding = features_transposed[:, 0, :]  # [batch, num_filters]
                  # [batch, seq-1, num_filters]
                pooled_features = cls_embedding
                attention_weights = torch.zeros(batch_size, seq_len, 1, device=x.device)  # Placeholder
            else:
                pooled_features, weighted_features, attention_weights = self.pooling(
                    features_transposed, mask
                )
            
            # Generate outputs
            x1 = self.out1(pooled_features)  # Binary classification
            x3 = self.out3(pooled_features)  # Regression/severity
            
            # For sequence-level predictions
            sequence_features_conv = features_transposed.permute(0, 2, 1)  # [batch, num_filters, seq]
            x2 = self.out2(sequence_features_conv)  # [batch, 1, seq]
            
            # Apply masking to x2 outputs
            mask_flat = mask.reshape(-1).bool()
            x2_flat = x2.view(-1)
            x2_masked = x2_flat[mask_flat]
            attention_weights_flat = attention_weights.reshape(-1)
            attention_weights_masked = attention_weights_flat[mask_flat]
            
            return x1, x2_masked, x3, attention_weights_masked, None, None
    
    def get_encoder_weights(self):
        """
        Get the encoder weights for transfer to downstream models.
        """
        return {
            'input_projection': self.input_projection.state_dict(),
            'encoder': self.encoder.state_dict(),
            'positional_encoding': self.positional_encoding.state_dict()
        }
    
    def load_pretrained_encoder(self, checkpoint_path):
        """
        Load pretrained encoder weights from checkpoint.
        Supports both direct encoder checkpoints and DINO/contrastive learning checkpoints.
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'encoder_weights' in checkpoint:
            # Direct encoder weights saved during pretraining
            weights = checkpoint['encoder_weights']
            self.input_projection.load_state_dict(weights['input_projection'])
            self.encoder.load_state_dict(weights['encoder'])
            self.positional_encoding.load_state_dict(weights['positional_encoding'])
        elif 'student.input_projection.conv.weight' in checkpoint:
            # DINO or other contrastive learning checkpoint - extract student weights
            student_weights = {}
            for key, value in checkpoint.items():
                if key.startswith('student.'):
                    # Remove 'student.' prefix
                    new_key = key[8:]  # len('student.') = 8
                    student_weights[new_key] = value
            
            # Extract encoder components
            input_proj_weights = {k.replace('input_projection.', ''): v 
                                for k, v in student_weights.items() 
                                if k.startswith('input_projection.')}
            encoder_weights = {k.replace('encoder.', ''): v 
                             for k, v in student_weights.items() 
                             if k.startswith('encoder.')}
            pos_enc_weights = {k.replace('positional_encoding.', ''): v 
                             for k, v in student_weights.items() 
                             if k.startswith('positional_encoding.')}
            
            # Load the weights
            self.input_projection.load_state_dict(input_proj_weights)
            self.encoder.load_state_dict(encoder_weights)
            self.positional_encoding.load_state_dict(pos_enc_weights)
        else:
            # Try to load compatible weights (fallback)
            self.load_state_dict(checkpoint, strict=False)
    
    def set_mode(self, pretraining=True):
        """
        Switch between pretraining and fine-tuning modes.
        
        Args:
            pretraining: If True, model is in pretraining mode. If False, fine-tuning mode.
        """
        self.pretraining = pretraining
        
        # Add prediction heads if switching to fine-tuning mode
        if not pretraining and not hasattr(self, 'out1'):
            norm = self.config.get('norm', 'BN')
            self.out1 = OutModule2(self.num_filters, self.num_filters, 1, norm, 'FC')
            self.out2 = OutModule2(self.num_filters, self.num_filters, 1, norm, 'Conv')
            self.out3 = OutModule2(self.num_filters, self.num_filters, 1, norm, 'FC')
    
    def freeze_encoder(self):
        """
        Freeze the encoder parameters including input_projection and positional_encoding.
        Useful for fine-tuning scenarios where you want to keep the pretrained encoder frozen.
        """
        # Freeze input projection
        for param in self.input_projection.parameters():
            param.requires_grad = False
        
        # Freeze positional encoding
        for param in self.positional_encoding.parameters():
            param.requires_grad = False
        
        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        """
        Unfreeze the encoder parameters including input_projection and positional_encoding.
        Allows all encoder components to be trainable.
        """
        # Unfreeze input projection
        for param in self.input_projection.parameters():
            param.requires_grad = True
        
        # Unfreeze positional encoding
        for param in self.positional_encoding.parameters():
            param.requires_grad = True
        
        # Unfreeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = True
    


