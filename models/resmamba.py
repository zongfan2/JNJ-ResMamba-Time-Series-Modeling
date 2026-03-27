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
from .attention import MultiHeadSelfAttentionPooling, MaskedMaxAvgPooling
from .components import FeatureExtractor

# Primary ResMamba models

def masked_avg_pool(x, mask):
    # Sum the activations only for the valid (non-padded) positions
    sum_x = (x * mask.unsqueeze(-1)).sum(dim=1)  # Shape: (batch_size, num_filters)
    valid_count = mask.sum(dim=1)  # Shape: (batch_size)
    avg_x = sum_x / valid_count.unsqueeze(1)  # Normalize by number of valid elements
    return avg_x

def create_mask(original_lengths, max_length,batch_size,device):
    """
    Creates a binary mask based on the original sequence lengths.
    
    Args:
    - original_lengths (Tensor): The original (un-padded) lengths of each sequence in the batch.
    - max_length (int): The length of the padded sequences (typically the maximum sequence length in the batch).
    
    Returns:
    - mask (Tensor): A binary mask where 1 indicates valid data and 0 indicates padding.
    """
    # Create a tensor of shape (batch_size, max_length) where each row contains 
    # indices from 0 to max_length-1, representing each time step in the sequence.
    mask = torch.arange(max_length,device=device).unsqueeze(0).expand(batch_size, -1)
    # The mask will have 1s where the sequence is valid and 0s where padding occurs
    mask = (mask < original_lengths.unsqueeze(1)).long()
    
    return mask

    
class MBA_tsm(nn.Module):
    def __init__(self,
                 input_dim, 
                 n_state=16,
                 pos_embed_dim=16,
                 num_filters=64,
                 BI=True,
                 num_feature_layers=7,
                 num_encoder1_layers=3,
                 num_encoder2_layers=3,
                 dilation1=True,
                 dilation2=True,
                 drop_path_rate=0.3,
                 tsm_horizon=64,
                 kernel_size_feature=3,
                 kernel_size_mba1=7,
                 kernel_size_mba2=7,
                 channel_masking_rate=0,
                 dropout_rate=0.2,
                 tsm=False,
                 mba=True,
                 add_positional_encoding=True,
                 max_seq_len=2400,
                 pretraining=False,
                 cls_token=True,
                 num_heads=4,
                 featurelayer="TCN",
                 supcon_loss=False,
                 skip_connect=False,
                 skip_cross_attention=False,
                 norm1='IN',
                 norm2='IN',
                 norm3='BN',
                 pooling_type="attention"):
        super(MBA_tsm, self).__init__()
        
        self.input_projection = ConvProjection(input_dim, num_filters,norm1)# Input projection to transform raw signals to model dimension
        self.add_positional_encoding=add_positional_encoding
        self.pretraining = pretraining
        self.use_cls_token = cls_token
        self.cls_token = nn.Parameter(torch.randn(1, num_filters,1))
        self.num_feature_layers = num_feature_layers
        self.skip_connect = skip_connect
        self.skip_cross_attention = skip_cross_attention
        self.feature_extractor = FeatureExtractor(tsm_horizon,num_filters,pos_embed_dim,num_filters=num_filters,kernel_size=kernel_size_feature,num_feature_layers=num_feature_layers,tsm=tsm,featurelayer=featurelayer,norm=norm1)
        # self.feature_extractor2 = FeatureExtractor(tsm_horizon,num_filters,pos_embed_dim,num_filters=num_filters,kernel_size=kernel_size_feature,num_feature_layers=num_feature_layers,tsm=tsm,featurelayer='ResTCN',norm=norm1)
        # self.positional_encoding = PositionalEncoding(d_model=num_filters,max_len=max_seq_len+1 if self.use_cls_token else max_seq_len)
        self.positional_encoding = PositionalEncoding(d_model=num_filters,max_len=max_seq_len+1)

        self.GatedAttentionMILPooling =GatedAttentionPoolingMIL(input_dim=num_filters, hidden_dim=16)   
        # Choose pooling type
        if pooling_type == 'concat':
            self.pooling = MaskedMaxAvgPooling(input_dim=num_filters, combination_type='concat')
            self.pooling_output_dim = 2 * num_filters  # Concatenated max and avg
        elif pooling_type == 'weighted':
            self.pooling = MaskedMaxAvgPooling(input_dim=num_filters, combination_type='weighted')
            self.pooling_output_dim = num_filters
        elif pooling_type == 'avg_only':
            self.pooling = MaskedMaxAvgPooling(input_dim=num_filters, combination_type='avg_only')
            self.pooling_output_dim = num_filters
        elif pooling_type == 'max_only':
            self.pooling = MaskedMaxAvgPooling(input_dim=num_filters, combination_type='max_only')
            self.pooling_output_dim = num_filters
        elif pooling_type == 'attention':
            self.pooling = MultiHeadSelfAttentionPooling(input_dim=num_filters, hidden_dim=num_filters, num_heads=num_heads)
            self.pooling_output_dim = num_filters
        else:
            raise ValueError(f"Unknown pooling_type: {pooling_type}. Choose from ['weighted', 'avg_only', 'max_only', 'attention']")

        self.pooling_type = pooling_type
        self.supcon_loss = supcon_loss
        self.channel_masking_rate = channel_masking_rate
        self.dropout = nn.Dropout2d(p=channel_masking_rate)


        # Choose encoder type based on skip connection configuration
        if self.skip_connect and self.skip_cross_attention:
            # Use cross-attention modules for skip connections
            self.encoder1 = nn.ModuleList([AttModule_mamba_cross(2 ** i, num_filters, num_filters, 1, 1, 'sliding_att', 'encoder', 1, drop_path_rate=drop_path_rate, kernel_size=kernel_size_mba1, dropout_rate=dropout_rate, norm="IN") for i in range(num_encoder1_layers)])
        else:
            # Use standard attention modules
            self.encoder1 = nn.ModuleList([AttModule_mamba(2 ** i, num_filters, num_filters, 'sliding_att', 'encoder', 1, drop_path_rate=drop_path_rate,dropout_rate=dropout_rate,kernel_size=kernel_size_mba1, norm=norm2) for i in range(num_encoder1_layers)])

        self.out1 = OutModule2(self.pooling_output_dim,num_filters, 1, norm3, 'FC')  
        self.out2 = OutModule2(num_filters,num_filters, 1, norm3, 'Conv')
        self.out3 = OutModule2(self.pooling_output_dim,num_filters, 1, norm3, 'FC')
        
        # if pretraining:
        #     self.out1_pretrain = nn.Conv1d(num_filters,1, 1)
        #     self.out2_pretrain = nn.Conv1d(num_filters,1, 1)
        #     self.out3_pretrain = nn.Conv1d(num_filters,1, 1)
        
        if self.supcon_loss:
            self.proj = nn.Sequential(
                nn.Linear(num_filters, num_filters),
                nn.ReLU(),
                nn.Linear(num_filters, 128)
            )

    def forward(self, x, x_lengths,labels1=None, labels3=None, apply_mixup=False, mixup_alpha=0.2):

        batch_size, seq_len, channels = x.size()       
        # Create padding mask from original input
        if isinstance(x_lengths, torch.Tensor):
            x_lengths = x_lengths.to(x.device)
        else:
            x_lengths = torch.tensor(x_lengths,device=x.device)
        mask = create_mask(x_lengths, seq_len,batch_size,x.device)
        
        x = self.input_projection(x.permute(0, 2, 1))

        token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((token, x), dim=2)
        mask=torch.cat((torch.ones(batch_size,1,device=x.device), mask), dim=1)
        
        if self.num_feature_layers>0:
            # Store intermediate feature extractor outputs for UNet-style skip connections
            if self.skip_connect:
                # Get intermediate features from each layer of feature extractor
                x, feature_maps = self.feature_extractor(x, mask, return_intermediates=True)
            else:
                x = self.feature_extractor(x, mask)
            
            # # If in pretraining mode, return pooled features from feature extractor
            # if self.pretraining:
            #     # Choose pooling method based on use_cls_token
            #     if self.use_cls_token:
            #         # Use CLS token (first position) for pooling
            #         pooled_features = x[:, :, 0]
            #     else:
            #         # Use multi-head self-attention pooling
            #         pooled_features, _, _ = self.pooling(x[:, :, 1:].permute(0, 2, 1), mask[:, 1:])
            #     return pooled_features
            
        if self.add_positional_encoding:
            x = self.positional_encoding(x)
        
            if self.channel_masking_rate > 0:
                x = x.unsqueeze(2)
                x = self.dropout(x)
                x = x.squeeze(2)
                
        # Apply encoder layers with UNet-style skip connections
        if self.skip_connect and self.num_feature_layers > 0:
            # UNet-style: connect each encoder layer with reversed feature layer (U-shape)
            # First encoder layer gets last feature map, last encoder layer gets first feature map
            for i, layer in enumerate(self.encoder1):
                if self.skip_cross_attention:
                    # Use cross-attention to integrate skip connections
                    if i < len(feature_maps):
                        skip_idx = len(feature_maps) - 1 - i  # Reverse the index for U-Net structure
                        encoder_states = feature_maps[skip_idx]  # Skip connection as encoder states
                        x = layer(x, encoder_states, mask)  # Cross-attention integration
                    else:
                        x = layer(x, None, mask)  # No skip connection available
                else:
                    # Use simple addition for skip connections
                    x = layer(x, None, mask)
                    if i < len(feature_maps):
                        skip_idx = len(feature_maps) - 1 - i  # Reverse the index for U-Net structure
                        x = x + feature_maps[skip_idx]  # UNet-style skip connection
        else:
            # Standard forward without skip connections
            for layer in self.encoder1:
                x = layer(x, None, mask)
        
        # Choose pooling method based on use_cls_token
        if self.use_cls_token:
            # Use CLS token (first position) for pooling
            pooled_features = x[:, :, 0]
        else:
            # Use multi-head self-attention pooling
            pooled_features, _, _ = self.pooling(x[:, :, 1:].permute(0, 2, 1), mask[:, 1:])

        if self.pretraining:
            return pooled_features

        attention_weights = torch.zeros(batch_size, seq_len, 1, device=x.device)
        
        x1 = self.out1(pooled_features)
        x3 = self.out3(pooled_features)
        x2 = self.out2(x[:, :, 1:]) 

        mask = mask[:, 1:].reshape(-1).bool()
        x2 = x2.view(-1)
        x2 = x2[mask]
        attention_weights = attention_weights.reshape(-1)
        attention_weights = attention_weights[mask]

        return x1, x2, x3, attention_weights, None, None
    
    def set_mode(self, pretraining=True):
        """
        Switch between pretraining and fine-tuning modes.
        
        Args:
            pretraining: If True, model is in pretraining mode. If False, fine-tuning mode.
        """
        self.pretraining = pretraining
    
    
    def freeze_encoder(self):
        """
        Freeze the encoder parameters including input_projection, feature_extractor, 
        positional_encoding, and encoder1 layers.
        """
        # Freeze input projection
        for param in self.input_projection.parameters():
            param.requires_grad = False
        
        # Freeze feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        # Freeze positional encoding
        for param in self.positional_encoding.parameters():
            param.requires_grad = False
        
        # Freeze encoder1 layers
        for layer in self.encoder1:
            for param in layer.parameters():
                param.requires_grad = False
    
    def unfreeze_encoder(self):
        """
        Unfreeze the encoder parameters.
        """
        # Unfreeze input projection
        for param in self.input_projection.parameters():
            param.requires_grad = True
        
        # Unfreeze feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = True
        
        # Unfreeze positional encoding
        for param in self.positional_encoding.parameters():
            param.requires_grad = True
        
        # Unfreeze encoder1 layers
        for layer in self.encoder1:
            for param in layer.parameters():
                param.requires_grad = True
    
    def freeze_feature_extractor(self):
        """
        Freeze only the feature extractor for hybrid pretraining approach.
        This allows fine-tuning of Mamba encoder while keeping pretrained features frozen.
        """
        # Freeze input projection
        for param in self.input_projection.parameters():
            param.requires_grad = False
        
        # Freeze feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        # Keep positional encoding and encoder1 layers trainable
        print("Frozen feature extractor (input_projection + feature_extractor)")
        print("Mamba encoder layers remain trainable for fine-tuning")
    
    def unfreeze_feature_extractor(self):
        """
        Unfreeze the feature extractor for end-to-end training.
        """
        # Unfreeze input projection
        for param in self.input_projection.parameters():
            param.requires_grad = True
        
        # Unfreeze feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = True
    
    def load_pretrained_weights(self, checkpoint_path):
        """
        Load pretrained encoder weights from checkpoint for fine-tuning.
        Supports both direct encoder checkpoints and contrastive learning checkpoints.
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'encoder_weights' in checkpoint:
            # Direct encoder weights saved during pretraining
            weights = checkpoint['encoder_weights']
            self.input_projection.load_state_dict(weights['input_projection'])
            self.feature_extractor.load_state_dict(weights['feature_extractor'])
            self.positional_encoding.load_state_dict(weights['positional_encoding'])
            for i, layer_weights in enumerate(weights['encoder1']):
                self.encoder1[i].load_state_dict(layer_weights)
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
            feature_ext_weights = {k.replace('feature_extractor.', ''): v 
                                 for k, v in student_weights.items() 
                                 if k.startswith('feature_extractor.')}
            pos_enc_weights = {k.replace('positional_encoding.', ''): v 
                             for k, v in student_weights.items() 
                             if k.startswith('positional_encoding.')}
            encoder1_weights = {k.replace('encoder1.', ''): v 
                              for k, v in student_weights.items() 
                              if k.startswith('encoder1.')}
            
            # Load the weights
            self.input_projection.load_state_dict(input_proj_weights)
            self.feature_extractor.load_state_dict(feature_ext_weights)
            self.positional_encoding.load_state_dict(pos_enc_weights)
            
            # Load encoder1 layer weights
            encoder1_layers = {}
            for key, value in encoder1_weights.items():
                layer_idx = int(key.split('.')[0])
                param_key = '.'.join(key.split('.')[1:])
                if layer_idx not in encoder1_layers:
                    encoder1_layers[layer_idx] = {}
                encoder1_layers[layer_idx][param_key] = value
            
            for layer_idx, layer_weights in encoder1_layers.items():
                self.encoder1[layer_idx].load_state_dict(layer_weights)
        else:
            # Try to load compatible weights (fallback)
            self.load_state_dict(checkpoint, strict=False)
        
        print(f"Loaded pretrained weights from {checkpoint_path}")

class MBA_tsm_with_padding(nn.Module):
    def __init__(self,input_dim, 
                      n_state=16,
                      pos_embed_dim=16,
                      num_filters=64,
                      BI=True,
                      num_feature_layers=5,
                      num_encoder_layers=3,
                      drop_path_rate=0.3,
                      tsm_horizon=64,
                      kernel_size_feature=3,
                      kernel_size_mba=7,
                      dropout_rate=0.2,
                      tsm=False,
                      mba=True,
                      add_positional_encoding=True,
                      max_seq_len=2400,
                      cls_token=False,
                      num_heads=4,
                      featurelayer="TCN",
                      average_mask=False,
                      average_window_size=20,
                      supcon_loss=False,
                      reg_only=False,
                      use_feature_extractor=True,
                      norm="BN"):
        super(MBA_tsm_with_padding, self).__init__()
        self.add_positional_encoding=add_positional_encoding
        self.cls_token = nn.Parameter(torch.randn(1, num_filters,1))
        self.pooling=cls_token
        self.use_feature_extractor = use_feature_extractor
        if self.use_feature_extractor:
            self.feature_extractor = FeatureExtractor(tsm_horizon,
                                                      input_dim,
                                                      pos_embed_dim,
                                                      num_filters=num_filters,
                                                      kernel_size=kernel_size_feature,
                                                      num_feature_layers=num_feature_layers,
                                                      tsm=tsm,
                                                      featurelayer=featurelayer)
        else:
            self.input_projection = nn.Conv1d(input_dim, num_filters, 1)
            self.norm = nn.LayerNorm(num_filters)
            # BN
            # self.norm = nn.BatchNorm1d(num_filters)
            # IN
            # self.norm = nn.InstanceNorm1d(num_filters)
        self.positional_encoding = PositionalEncoding(d_model=num_filters,max_len=max_seq_len)
        self.GatedAttentionMILPooling =GatedAttentionPoolingMIL(input_dim=num_filters, hidden_dim=16)   
        self.MultiHeadSelfAttentionPooling = MultiHeadSelfAttentionPooling(input_dim=num_filters, hidden_dim=num_filters, num_heads=num_heads)
        self.average_mask = average_mask
        self.average_window_size = average_window_size
        self.supcon_loss = supcon_loss
        self.reg_only = reg_only

        if not mba:
            self.encoder = nn.ModuleList([AttModule(2 ** i, num_filters, num_filters, 2, 2,'normal_att', 'encoder', alpha=1,kernel_size=kernel_size_mba,dropout_rate=dropout_rate) for i in range(num_encoder_layers)])
        else:
            self.encoder = nn.ModuleList([AttModule_mamba(2 ** i, num_filters, num_filters, 'sliding_att', 'encoder', 1, drop_path_rate=drop_path_rate,dropout_rate=dropout_rate,kernel_size=kernel_size_mba, norm=norm) for i in range(num_encoder_layers)])
        
        if not reg_only:
            self.out1 = nn.Linear(num_filters, 1)  
            self.out2 = nn.Conv1d(num_filters, 1, 1)
        self.out3 = nn.Linear(num_filters, 1)

        if self.average_mask:
            self.scale_out2 = nn.Linear(max_seq_len, max_seq_len//average_window_size)
        

    def forward(self, x, labels1=None, labels3=None, apply_mixup=False, mixup_alpha=0.2, padding_value=-999.0, replaced_padding_value=-0.5, **kwargs):
        batch_size, seq_len, channels = x.size()    
        
        x_orig = x
        x_for_features = x.clone()
        padding_mask_for_replacement = torch.any(x == padding_value, dim=-1, keepdim=True)
        x_for_features = torch.where(padding_mask_for_replacement, replaced_padding_value, x_for_features)
        if self.use_feature_extractor:
            x = x_for_features.permute(0, 2, 1)
            x = self.feature_extractor(x)
        else:
            # Project input to model dimension: [B, seq_len, input_dim] -> [B, seq_len, num_filters]
            x = self.input_projection(x_for_features.permute(0, 2, 1))
            # Transpose for conv layers: [B, seq_len, num_filters] -> [B, num_filters, seq_len]
            x = self.norm(x.permute(0, 2, 1))
            x = x.permute(0, 2, 1)
        
        if self.add_positional_encoding:
            x = self.positional_encoding(x)

        # padding_mask = create_mask(torch.tensor(x_lens, device=x.device), seq_len, batch_size, x.device).float()
        # check padding value for each position
        padding_mask = ~torch.any(x_orig == padding_value, dim=-1)
        # padding_mask = padding_mask.float()
        cur_seq_len = x.shape[2]
        if cur_seq_len != seq_len:
            padding_mask = F.interpolate(
                padding_mask.float().unsqueeze(1), 
                size=cur_seq_len,
                mode="nearest" 
            ).squeeze(1).bool().float()
        else:
            padding_mask = padding_mask.float()

        # No masking for fixed length padded inputs
        if len(self.encoder)>0:
            for layer in self.encoder:
                x = layer(x, None, padding_mask)  # use padding mask
        
        if self.pooling :
            attention_weights = torch.zeros(batch_size, cur_seq_len, 1, device=x.device)
            pooled_features = x[:, :, 0]
        else:
            # Use padding mask for attention pooling
            pooled_features,weighted_features,attention_weights=self.MultiHeadSelfAttentionPooling(x.permute(0, 2, 1), padding_mask)
        
        if apply_mixup and self.training:
            mixed_features, mixed_labels1, mixed_labels3, mixup_lambda = latent_mixup(
                pooled_features, labels1, labels3, alpha=mixup_alpha
            )
            mixup_info = (mixed_labels1, mixed_labels3, mixup_lambda)
        else:
            mixed_features = pooled_features
            mixup_info = None
            
        if not self.reg_only:
            x1 = self.out1(mixed_features)
            x2 = self.out2(x) 

            if self.supcon_loss:
                contrastive_embedding = pooled_features
            else:
                contrastive_embedding = None

            # Keep padding - do not remove any values from x2
            x2 = x2.squeeze(1)
            if self.average_mask:
                x2 = self.scale_out2(x2)
            # Return all attention weights without masking
            attention_weights = attention_weights.reshape(-1)
        else:
            x1, x2, attention_weights, contrastive_embedding = None, None, None, None
            
        x3= self.out3(mixed_features)
        return x1, x2, x3, attention_weights, contrastive_embedding, mixup_info
    
    def load_pretrained_feature_extractor(self, checkpoint_path, freeze=True):
        """
        Load a pretrained FeatureExtractor into this MBA_tsm model.
        
        Args:
            checkpoint_path: Path to the pretrained feature extractor checkpoint
            freeze: Whether to freeze the feature extractor parameters
        """
        import torch
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        feature_extractor_state_dict = {}
        
        for key, value in checkpoint.items():
            if key.startswith('backbone.feature_extractor.'):
                # Remove 'backbone.feature_extractor.' prefix
                new_key = key[len('backbone.feature_extractor.'):]
                feature_extractor_state_dict[new_key] = value
        
        # If no backbone keys found, assume entire checkpoint is feature extractor
        if not feature_extractor_state_dict:
            feature_extractor_state_dict = checkpoint
        
        # Load the state dict
        try:
            self.feature_extractor.load_state_dict(feature_extractor_state_dict, strict=True)
            print(f"Successfully loaded feature extractor weights with {len(feature_extractor_state_dict)} parameters")
        except RuntimeError as e:
            print("Attempting to load with strict=False...")
            missing_keys, unexpected_keys = self.feature_extractor.load_state_dict(feature_extractor_state_dict, strict=False)
            if missing_keys:
                print(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys}")
        
        if freeze:
            # Freeze feature extractor parameters
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
            # set feature extractor to eval mode to freeze BatchNorm
            self.feature_extractor.eval()
            print("FeatureExtractor parameters frozen for fine-tuning")
        else:
            print("FeatureExtractor parameters remain trainable")
    
    def unfreeze_feature_extractor(self):
        """Unfreeze the feature extractor parameters."""
        for param in self.feature_extractor.parameters():
            param.requires_grad = True
        print("FeatureExtractor parameters unfrozen")
    
    def train(self, mode=True):
        """Override train method to keep frozen feature extractor in eval mode."""
        super().train(mode)
        # If feature extractor is frozen (no gradients), keep it in eval mode
        # if hasattr(self, 'feature_extractor'):
        #     frozen_params = [p for p in self.feature_extractor.parameters() if not p.requires_grad]
        #     if len(frozen_params) > 0:
        #         self.feature_extractor.eval()
        return self

  
class MBA_patch(nn.Module):
    """
    Mamba model using patch embeddings from PatchEmbedWithPos1D.
    Similar to MBA_tsm_with_padding but uses patch-based input processing like ViT1D.
    """
    
    def __init__(
        self,
        max_seq_len: int = 1221,            # Maximum sequence length
        window_size: int = 60,              # Patch window size
        stride: int = 20,                   # Patch stride
        in_channels: int = 3,               # Number of input channels
        embed_dim = None,                   # Embedding dimension (defaults to patch_dim)
        use_embedding: bool = False,        # Whether to apply linear embedding to patches
        num_encoder_layers: int = 6,        # Number of Mamba encoder layers
        drop_path_rate: float = 0.3,        # Drop path rate for Mamba blocks
        kernel_size_mba: int = 7,           # Kernel size for Mamba blocks
        dropout_rate: float = 0.2,          # Dropout rate
        padding_value: float = -999.0,      # Padding value
        padding_threshold: float = 0.5,     # Padding threshold
        num_heads: int = 4,                 # Number of attention heads for pooling
        supcon_loss: bool = False,          # Whether to return contrastive embeddings
        reg_only: bool = False,             # Whether to do regression only
        cls_token: bool = False,            # Whether to use CLS token pooling
    ):
        super(MBA_patch, self).__init__()
        
        self.max_seq_len = max_seq_len
        self.window_size = window_size
        self.stride = stride
        self.in_channels = in_channels
        self.use_embedding = use_embedding
        self.num_encoder_layers = num_encoder_layers
        self.supcon_loss = supcon_loss
        self.reg_only = reg_only
        self.pooling = cls_token
        
        # Import PatchEmbedWithPos1D locally to avoid circular imports
        try:
            from models.embed import PatchEmbedWithPos1D
        except:
            try:
                from embed import PatchEmbedWithPos1D
            except:
                # Fallback for relative imports
                import sys
                import os
                sys.path.insert(0, os.path.dirname(__file__))
                from embed import PatchEmbedWithPos1D
        
        # Patch embedding with positional encoding and CLS token
        self.patch_embed = PatchEmbedWithPos1D(
            max_seq_len=max_seq_len,
            window_size=window_size,
            stride=stride,
            in_channels=in_channels,
            embed_dim=embed_dim,
            padding_value=padding_value,
            padding_threshold=padding_threshold,
            use_embedding=use_embedding,
            pos_learnable=False,
            include_cls=True,
            dropout=dropout_rate
        )
        
        # Get actual embedding dimension from patch embedding
        self.embed_dim = self.patch_embed.patch_embed.embed_dim
        
        # Multi-head self-attention pooling
        self.MultiHeadSelfAttentionPooling = MultiHeadSelfAttentionPooling(
            input_dim=self.embed_dim, 
            hidden_dim=self.embed_dim, 
            num_heads=num_heads
        )
        
        # Mamba encoder layers using existing AttModule_mamba
        if num_encoder_layers > 0:
            self.encoder = nn.ModuleList([
                AttModule_mamba_cls(
                    dilation=min(16, 2**i), #2**i
                    in_channels=self.embed_dim, 
                    out_channels=self.embed_dim, 
                    r1=2, 
                    r2=2,
                    att_type='sliding_att', 
                    stage='encoder', 
                    alpha=1, 
                    drop_path_rate=drop_path_rate,
                    dropout_rate=dropout_rate,
                    kernel_size=kernel_size_mba
                ) for i in range(num_encoder_layers)
            ])
        else:
            self.encoder = nn.ModuleList([])
        
        # Output heads
        if not reg_only:
            self.out1 = nn.Linear(self.embed_dim, 1)  # Classification
            self.out2 = nn.Conv1d(self.embed_dim, 1, 1)  # Mask prediction
            expected_patches = (max_seq_len - window_size) // stride + 1
            target_length = max_seq_len // stride 
            self.scale_out2 = nn.Linear(expected_patches, target_length)
        
        self.out3 = nn.Linear(self.embed_dim, 1)  # Regression
    
    def forward(
        self, 
        x,
        labels1=None,
        labels3=None,
        apply_mixup=False,
        mixup_alpha=0.2,
        **kwargs
    ):
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, L, C]
            labels1: Classification labels
            labels3: Regression labels
            apply_mixup: Whether to apply mixup
            mixup_alpha: Mixup alpha parameter
            **kwargs: Additional keyword arguments
            
        Returns:
            x1: Classification output [B, 1]
            x2: Mask prediction output [B, seq_len] 
            x3: Regression output [B, 1]
            attention_weights: Attention weights from pooling
            contrastive_embedding: Pooled features if supcon_loss=True
            mixup_info: Mixup information if apply_mixup=True
        """
        assert x.dim() == 3, "Input must be of shape [B, L, C]"
        batch_size, seq_len, channels = x.shape
        
        # Convert to [B, C, L] for patch embedding
        x = x.permute(0, 2, 1)  # [B, C, L]

        # Get patch embeddings with CLS token and positional encoding
        x, padding_mask = self.patch_embed(x)  # [B, 1+num_patches, embed_dim]
        
        # Convert to [B, embed_dim, seq_len] format for Mamba blocks
        current_seq_len = x.shape[1]  # 1 + num_patches
        x = x.permute(0, 2, 1)  # [B, embed_dim, 1+num_patches]
        
        # Create padding mask in float format for AttModule_mamba
        if padding_mask is not None:
            padding_mask_float = (~padding_mask).float()  # [B, 1+num_patches], True for valid tokens
            # padding_mask_float = (padding_mask).float()
        else:
            padding_mask_float = torch.ones(batch_size, current_seq_len, device=x.device)
        
        # Apply Mamba encoder layers
        if len(self.encoder) > 0:
            for layer in self.encoder:
                x = layer(x, None, padding_mask_float)  # AttModule_mamba(x, f, mask)
        
        # Convert back to [B, seq_len, embed_dim] for pooling
        x = x.permute(0, 2, 1)  # [B, 1+num_patches, embed_dim]
        
        # Pool features
        if self.pooling:  # Use CLS token
            pooled_features = x[:, 0]  # [B, embed_dim] - CLS token
            attention_weights = torch.zeros(batch_size, current_seq_len, 1, device=x.device)
        else:
            # Use multi-head attention pooling
            pooled_features, weighted_features, attention_weights = self.MultiHeadSelfAttentionPooling(
                x, padding_mask_float
            )
        
        # Apply mixup if requested
        if apply_mixup and self.training and labels1 is not None and labels3 is not None:
            mixed_features, mixed_labels1, mixed_labels3, mixup_lambda = latent_mixup(
                pooled_features, labels1, labels3, alpha=mixup_alpha
            )
            mixup_info = (mixed_labels1, mixed_labels3, mixup_lambda)
        else:
            mixed_features = pooled_features
            mixup_info = None
        
        # Generate outputs
        if not self.reg_only:
            x1 = self.out1(mixed_features)  # [B, 1] Classification
            
            # For mask prediction, convert back to conv format
            # x_conv = x.permute(0, 2, 1)  # [B, embed_dim, seq_len]
            # x2 = self.out2(x_conv)  # [B, 1, seq_len]
            # x2 = x2.squeeze(1)  # [B, seq_len]
            patch_tokens = x[:, 1:, :]
            x2 = self.out2(patch_tokens.permute(0, 2, 1)).squeeze(1)
            
            x2 = self.scale_out2(x2)  # [B, seq_len // average_window_size]
            
            if self.supcon_loss:
                contrastive_embedding = pooled_features  # Use pooled features directly
            else:
                contrastive_embedding = None
        else:
            x1, x2, attention_weights, contrastive_embedding = None, None, None, None
        
        x3 = self.out3(mixed_features)  # [B, 1] Regression
        
        # Reshape attention weights to match expected format
        if attention_weights is not None:
            attention_weights = attention_weights.reshape(-1)
        
        return x1, x2, x3, attention_weights, contrastive_embedding, mixup_info

class MBA4TSO(nn.Module):
    """
    MBA model for status prediction with 4-channel input and 3-class sequence-level output.
    Input: [batch_size, seq_len, 4] (x, y, z, temperature)
    Output: [batch_size, seq_len, 3] (other, non-wear, predictTSO masks)
    Supports UNet-style skip connections for better segmentation.
    """
    def __init__(self,
                 input_dim=4,
                 n_state=16,
                 pos_embed_dim=16,
                 num_filters=64,
                 num_feature_layers=7,
                 num_encoder_layers=3,
                 drop_path_rate=0.3,
                 tsm_horizon=64,
                 kernel_size_feature=3,
                 kernel_size_mba=7,
                 dropout_rate=0.2,
                 tsm=False,
                 add_positional_encoding=True,
                 max_seq_len=2400,
                 num_heads=4,
                 featurelayer="TCN",
                 skip_connect=True,
                 skip_cross_attention=False,
                 norm1='IN',
                 norm2='IN',
                 output_channels=3):
        super(MBA4TSO, self).__init__()

        self.input_projection = ConvProjection(input_dim, num_filters, norm1)
        self.add_positional_encoding = add_positional_encoding
        self.num_feature_layers = num_feature_layers
        self.output_channels = output_channels
        self.skip_connect = skip_connect
        self.skip_cross_attention = skip_cross_attention

        # Feature extraction
        self.feature_extractor = FeatureExtractor(
            tsm_horizon, num_filters, pos_embed_dim,
            num_filters=num_filters,
            kernel_size=kernel_size_feature,
            num_feature_layers=num_feature_layers,
            tsm=tsm,
            featurelayer=featurelayer,
            norm=norm1
        )

        self.positional_encoding = PositionalEncoding(d_model=num_filters, max_len=max_seq_len)

        # Choose encoder type based on skip connection configuration
        if self.skip_connect and self.skip_cross_attention:
            # Use cross-attention modules for skip connections
            self.encoder = nn.ModuleList([
                AttModule_mamba_cross(
                    2 ** i, num_filters, num_filters, 1, 1,
                    'sliding_att', 'encoder', 1,
                    drop_path_rate=drop_path_rate,
                    kernel_size=kernel_size_mba,
                    dropout_rate=dropout_rate
                ) for i in range(num_encoder_layers)
            ])
        else:
            # Use standard attention modules
            self.encoder = nn.ModuleList([
                AttModule_mamba(
                    2 ** i, num_filters, num_filters,
                    'sliding_att', 'encoder', 1,
                    drop_path_rate=drop_path_rate,
                    dropout_rate=dropout_rate,
                    kernel_size=kernel_size_mba,
                    norm=norm2
                ) for i in range(num_encoder_layers)
            ])

        # Output projection: num_filters -> output_channels (3)
        self.output_projection = nn.Conv1d(num_filters, output_channels, kernel_size=1)

    def forward(self, x, x_lengths):
        """
        Args:
            x: [batch_size, seq_len, 4] input tensor
            x_lengths: [batch_size] original sequence lengths

        Returns:
            output: [batch_size, seq_len, 3] class logits for each timestep
        """
        batch_size, seq_len, channels = x.size()

        # Create padding mask
        if isinstance(x_lengths, torch.Tensor):
            x_lengths_tensor = x_lengths.clone().detach()
        else:
            x_lengths_tensor = torch.tensor(x_lengths, device=x.device)
        mask = create_mask(x_lengths_tensor, seq_len, batch_size, x.device)

        # Input projection: [B, 4, L] -> [B, num_filters, L]
        x = self.input_projection(x.permute(0, 2, 1))

        # Feature extraction with optional skip connections
        if self.num_feature_layers > 0:
            if self.skip_connect:
                # Get intermediate features from each layer of feature extractor
                x, feature_maps = self.feature_extractor(x, mask, return_intermediates=True)
            else:
                x = self.feature_extractor(x, mask)

            # Update mask if sequence length changed
            if x.size(2) != mask.size(1):
                new_seq_len = x.size(2)
                mask = F.interpolate(
                    mask.unsqueeze(1).float(),
                    size=new_seq_len,
                    mode='nearest'
                ).squeeze(1).bool()

        # Positional encoding
        if self.add_positional_encoding:
            x = self.positional_encoding(x)

        # Apply encoder layers with UNet-style skip connections
        if self.skip_connect and self.num_feature_layers > 0:
            # UNet-style: connect each encoder layer with reversed feature layer (U-shape)
            # First encoder layer gets last feature map, last encoder layer gets first feature map
            for i, layer in enumerate(self.encoder):
                if self.skip_cross_attention:
                    # Use cross-attention to integrate skip connections
                    if i < len(feature_maps):
                        skip_idx = len(feature_maps) - 1 - i  # Reverse the index for U-Net structure
                        encoder_states = feature_maps[skip_idx]  # Skip connection as encoder states
                        x = layer(x, encoder_states, mask)  # Cross-attention integration
                    else:
                        x = layer(x, None, mask)  # No skip connection available
                else:
                    # Use simple addition for skip connections
                    x = layer(x, None, mask)
                    if i < len(feature_maps):
                        skip_idx = len(feature_maps) - 1 - i  # Reverse the index for U-Net structure
                        # Interpolate skip connection if sizes don't match
                        if x.size(2) != feature_maps[skip_idx].size(2):
                            skip_resized = F.interpolate(
                                feature_maps[skip_idx],
                                size=x.size(2),
                                mode='linear',
                                align_corners=False
                            )
                            x = x + skip_resized
                        else:
                            x = x + feature_maps[skip_idx]  # UNet-style skip connection
        else:
            # Standard forward without skip connections
            for layer in self.encoder:
                x = layer(x, None, mask)

        # Output projection: [B, num_filters, L] -> [B, output_channels, L]
        output = self.output_projection(x)

        # Interpolate back to original sequence length if needed
        if output.size(2) != seq_len:
            output = F.interpolate(output, size=seq_len, mode='linear', align_corners=False)

        # Permute to [B, L, 3]
        output = output.permute(0, 2, 1)

        return output


def latent_mixup(features, labels1=None, labels3=None, alpha=0.2):
    """
    Applies mixup to features and corresponding labels for classification and reg head only.
    
    Args:
        features: Feature tensor [batch_size, feature_dim]
        labels: Classification labels [batch_size]
        alpha: Beta distribution parameter for mixing strength
        
    Returns:
        mixed_features: Mixed feature tensor
        mixed_labels1: Mixed classification labels with lambda weights
        mixed_labels3: Mixed regression labels with lambda weights
        mixup_lambda: Mixing coefficient
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = features.shape[0]
    
    # Create permutation indices for mixing
    indices = torch.randperm(batch_size, device=features.device)
    
    # Mix features in latent space
    mixed_features = lam * features + (1 - lam) * features[indices]
    
    if labels1 is not None:
        mixed_labels1 = labels1[indices]
    else:
        mixed_labels1 = None

    if labels3 is not None:
        mixed_labels3 = lam * labels3 + (1 - lam) * labels3[indices]
    else:
        mixed_labels3 = None
    # Return original labels, permuted labels, and lambda for loss calculation
    return mixed_features, mixed_labels1, mixed_labels3, lam

