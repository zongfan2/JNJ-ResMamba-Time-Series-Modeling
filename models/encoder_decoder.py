# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from typing import Optional, Tuple
import numpy as np

from .mamba_blocks import MBA, AffineDropPath
from .attention import MultiHeadSelfAttentionPooling
from .components import FeatureExtractor

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
    
# class MBA_tsm(nn.Module):
#     def __init__(self,input_dim, 
#                       n_state=16,
#                       pos_embed_dim=16,
#                       num_filters=64,
#                       BI=True,
#                       num_feature_layers=5,
#                       num_encoder_layers=3,
#                       drop_path_rate=0.3,
#                       tsm_horizon=64,
#                       kernel_size_feature=3,
#                       kernel_size_mba=7,
#                       dropout_rate=0.2,
#                       tsm=False,
#                       mba=True,
#                       add_positional_encoding=True,
#                       max_seq_len=2400,
#                       pretraining=False,
#                       cls_token=False,
#                       num_heads=4,
#                       featurelayer="TCN",
#                       random_padding=False,
#                       average_mask=False,
#                       average_window_size=20,
#                       supcon_loss=False,
#                       reg_only=False,
#                       norm="BN"):
#         super(MBA_tsm, self).__init__()
#         self.add_positional_encoding=add_positional_encoding
#         self.pretraining = pretraining
#         self.cls_token = nn.Parameter(torch.randn(1, num_filters,1))
#         self.pooling=cls_token
#         self.feature_extractor = FeatureExtractor(tsm_horizon,input_dim,pos_embed_dim,num_filters=num_filters,kernel_size=kernel_size_feature,num_feature_layers=num_feature_layers,tsm=tsm,featurelayer=featurelayer)
#         self.positional_encoding = PositionalEncoding(d_model=num_filters,max_len=max_seq_len)
#         self.GatedAttentionMILPooling =GatedAttentionPoolingMIL(input_dim=num_filters, hidden_dim=16)   
#         self.MultiHeadSelfAttentionPooling = MultiHeadSelfAttentionPooling(input_dim=num_filters, hidden_dim=num_filters, num_heads=num_heads)
#         self.random_padding = random_padding
#         self.average_mask = average_mask
#         self.average_window_size = average_window_size
#         self.supcon_loss = supcon_loss
#         self.reg_only = reg_only

#         if not mba:
#             self.encoder = nn.ModuleList([AttModule(2 ** i, num_filters, num_filters, 2, 2,'normal_att', 'encoder', alpha=1,kernel_size=kernel_size_mba,dropout_rate=dropout_rate) for i in range(num_encoder_layers)])
#         else:
#             self.encoder = nn.ModuleList([AttModule_mamba(2 ** i, num_filters, num_filters, 'sliding_att', 'encoder', 1, drop_path_rate=drop_path_rate,dropout_rate=dropout_rate,kernel_size=kernel_size_mba, norm=norm) for i in range(num_encoder_layers)])
        
#         if not reg_only:
#             self.out1 = nn.Linear(num_filters, 1)  
#             self.out2 = nn.Conv1d(num_filters, 1, 1)
#         self.out3 = nn.Linear(num_filters, 1)

#         if self.average_mask:
#             self.scale_out2 = nn.Linear(max_seq_len, max_seq_len//average_window_size)
        
#         self.out1_pretrain = nn.Conv1d(num_filters, 1, 1)
#         self.out2_pretrain = nn.Conv1d(num_filters, 1, 1)
#         self.out3_pretrain = nn.Conv1d(num_filters, 1, 1)

#         if self.supcon_loss:
#             self.proj = nn.Sequential(
#                 nn.Linear(num_filters, num_filters),
#                 nn.ReLU(),
#                 nn.Linear(num_filters, 128)
#             )

#     def forward(self, x, x_lengths, labels1=None, labels3=None, apply_mixup=False, mixup_alpha=0.2):
#         #global xx,mask,mask2,lengths
#         batch_size, seq_len, channels = x.size()       
#         x=self.feature_extractor(x.permute(0, 2, 1))
         
#         if self.pretraining: #If pretraining, then predict x,y,z
#             if self.add_positional_encoding:
#                 x = self.positional_encoding(x)
#             mask = create_mask(torch.tensor(x_lengths,device=x.device), seq_len,batch_size,x.device)
#             if len(self.encoder)>0:
#                 for layer in self.encoder:
#                     x = layer(x,None, mask)
#             # pooled_features,attention_weights= self.GatedAttentionMILPooling(x.permute(0, 2, 1), mask)
#             # pooled_features,weighted_features,attention_weights=self.MultiHeadSelfAttentionPooling(x[:,:,1:].permute(0, 2, 1), mask[:,1:])
#             pooled_features,weighted_features,attention_weights=self.MultiHeadSelfAttentionPooling(x.permute(0, 2, 1), mask)
#             mask=mask.reshape(-1).bool()
#             x1 = self.out1_pretrain(x).view(-1)[mask]
#             x2 = self.out2_pretrain(x).view(-1)[mask]
#             x3= self.out3_pretrain(x).view(-1)[mask]
#         else: # else, predict scratch, scratch sequence, and duration
#             # token = self.cls_token.expand(batch_size, -1, -1)
#             # x = torch.cat((token, x), dim=2)
#             if self.add_positional_encoding:
#                 x = self.positional_encoding(x)

#             mask = create_mask(torch.tensor(x_lengths,device=x.device), seq_len,batch_size,x.device)
#             # mask=torch.cat((torch.ones(batch_size,1,device=x.device),mask), dim=1)
#             if len(self.encoder)>0:
#                 for layer in self.encoder:
#                     x = layer(x,None, mask)
#             # pooled_features,attention_weights= self.GatedAttentionMILPooling(x.permute(0, 2, 1), mask)
#             # pooled_features,weighted_features,attention_weights=self.MultiHeadSelfAttentionPooling(x[:,:,1:].permute(0, 2, 1), mask[:,1:])
#             # mask=mask[:,1:].reshape(-1).bool()
            
#             if self.pooling :
#                 attention_weights = torch.zeros(batch_size, seq_len,1,device=x.device)
#                 pooled_features = x[:, :, 0]
#             else:
#                 pooled_features,weighted_features,attention_weights=self.MultiHeadSelfAttentionPooling(x.permute(0, 2, 1), mask)
            
#             if apply_mixup and self.training:
#                 mixed_features, mixed_labels1, mixed_labels3, mixup_lambda = latent_mixup(
#                     pooled_features, labels1, labels3, alpha=mixup_alpha
#                 )
#                 mixup_info = (mixed_labels1, mixed_labels3, mixup_lambda)
#             else:
#                 mixed_features = pooled_features
#                 mixup_info = None
#             if not self.reg_only:
#                 x1 = self.out1(mixed_features)
#                 x2 = self.out2(x) 

#                 if self.supcon_loss:
#                     contrastive_embedding = self.proj(pooled_features)
#                 else:
#                     contrastive_embedding = None

#                 # if random padding
#                 mask=mask.reshape(-1).bool()
#                 if not self.random_padding:
#                     x2=x2.view(-1)
#                     x2=x2[mask]
#                 else:
#                     x2 = x2.squeeze(1)
#                     if self.average_mask:
#                         x2 = self.scale_out2(x2)
#                 attention_weights=attention_weights.reshape(-1)
#                 attention_weights=attention_weights[mask]
#             # x3= self.out3(torch.cat((x[:,:,0], pooled_features), dim=1))
#             else:
#                 x1, x2, attention_weights, contrastive_embedding = None, None, None, None
#             x3= self.out3(mixed_features)
#         return x1, x2, x3, attention_weights, contrastive_embedding, mixup_info


# class MBA_tsm(nn.Module):
#     def __init__(self,
#                  input_dim, 
#                  n_state=16,
#                  pos_embed_dim=16,
#                  num_filters=64,
#                  BI=True,
#                  num_feature_layers=7,
#                  num_encoder1_layers=3,
#                  num_encoder2_layers=3,
#                  dilation1=True,
#                  dilation2=True,
#                  drop_path_rate=0.3,
#                  tsm_horizon=64,
#                  kernel_size_feature=3,
#                  kernel_size_mba1=7,
#                  kernel_size_mba2=7,
#                  channel_masking_rate=0,
#                  dropout_rate=0.2,
#                  tsm=False,
#                  mba=True,
#                  add_positional_encoding=True,
#                  max_seq_len=2400,
#                  pretraining=False,
#                  use_cls_token=True,
#                  num_heads=4,
#                  featurelayer="TCN",
#                  supcon_loss=False,
#                  skip_connect=False,
#                  skip_cross_attention=False,
#                  norm1='IN',
#                  norm2='IN',
#                  norm3='BN',
#                  pooling_type='attention'):
#         super(MBA_tsm, self).__init__()
        
#         self.input_projection = ConvProjection(input_dim, num_filters,norm1)# Input projection to transform raw signals to model dimension
#         self.add_positional_encoding=add_positional_encoding
#         self.pretraining = pretraining
#         self.use_cls_token = use_cls_token
#         self.cls_token = nn.Parameter(torch.randn(1, num_filters,1))
#         self.num_feature_layers = num_feature_layers
#         self.skip_connect = skip_connect
#         self.skip_cross_attention = skip_cross_attention
#         self.feature_extractor = FeatureExtractor(tsm_horizon,num_filters,pos_embed_dim,num_filters=num_filters,kernel_size=kernel_size_feature,num_feature_layers=num_feature_layers,tsm=tsm,featurelayer=featurelayer,norm=norm1)
#         # self.feature_extractor2 = FeatureExtractor(tsm_horizon,num_filters,pos_embed_dim,num_filters=num_filters,kernel_size=kernel_size_feature,num_feature_layers=num_feature_layers,tsm=tsm,featurelayer='ResTCN',norm=norm1)
#         self.positional_encoding = PositionalEncoding(d_model=num_filters,max_len=max_seq_len)
#         self.GatedAttentionMILPooling =GatedAttentionPoolingMIL(input_dim=num_filters, hidden_dim=16)

#         # Choose pooling type
#         if pooling_type == 'concat':
#             self.pooling = MaskedMaxAvgPooling(input_dim=num_filters, combination_type='concat')
#             self.pooling_output_dim = 2 * num_filters  # Concatenated max and avg
#         elif pooling_type == 'weighted':
#             self.pooling = MaskedMaxAvgPooling(input_dim=num_filters, combination_type='weighted')
#             self.pooling_output_dim = num_filters
#         elif pooling_type == 'masked_avg':
#             self.pooling = MaskedMaxAvgPooling(input_dim=num_filters, combination_type='avg_only')
#             self.pooling_output_dim = num_filters
#         elif pooling_type == 'masked_max':
#             self.pooling = MaskedMaxAvgPooling(input_dim=num_filters, combination_type='max_only')
#             self.pooling_output_dim = num_filters
#         elif pooling_type == 'attention':
#             self.pooling = MultiHeadSelfAttentionPooling(input_dim=num_filters, hidden_dim=num_filters, num_heads=num_heads)
#             self.pooling_output_dim = num_filters
#         else:
#             raise ValueError(f"Unknown pooling_type: {pooling_type}. Choose from ['masked_max_avg', 'masked_avg', 'masked_max', 'attention']")

#         self.pooling_type = pooling_type
#         self.supcon_loss = supcon_loss
#         self.channel_masking_rate = channel_masking_rate
#         self.dropout = nn.Dropout2d(p=channel_masking_rate)


#         # Choose encoder type based on skip connection configuration
#         if self.skip_connect and self.skip_cross_attention:
#             # Use cross-attention modules for skip connections
#             self.encoder1 = nn.ModuleList([AttModule_mamba_cross(2 ** i, num_filters, num_filters, 1, 1, 'sliding_att', 'encoder', 1, drop_path_rate=drop_path_rate, kernel_size=kernel_size_mba1, dropout_rate=dropout_rate) for i in range(num_encoder1_layers)])
#         else:
#             # Use standard attention modules
#             self.encoder1 = nn.ModuleList([AttModule_mamba(2 ** i, num_filters, num_filters, 'sliding_att', 'encoder', 1, drop_path_rate=drop_path_rate,dropout_rate=dropout_rate,kernel_size=kernel_size_mba1,norm=norm2) for i in range(num_encoder1_layers)])

#         # Output layers - adjust input dimension based on pooling type
#         self.out1 = OutModule2(self.pooling_output_dim, num_filters, 1, norm3, 'FC')
#         self.out2 = OutModule2(num_filters, num_filters, 1, norm3, 'Conv')  # Uses conv features, not pooled
#         self.out3 = OutModule2(self.pooling_output_dim, num_filters, 1, norm3, 'FC')
        
#         if pretraining:
#             self.out1_pretrain = nn.Conv1d(num_filters,1, 1)
#             self.out2_pretrain = nn.Conv1d(num_filters,1, 1)
#             self.out3_pretrain = nn.Conv1d(num_filters,1, 1)
        
#         if self.supcon_loss:
#             self.proj = nn.Sequential(
#                 nn.Linear(num_filters, num_filters),
#                 nn.ReLU(),
#                 nn.Linear(num_filters, 128)
#             )

#     def forward(self, x, x_lengths,labels1=None, labels3=None, apply_mixup=False, mixup_alpha=0.2):

#         batch_size, seq_len, channels = x.size()       
#         # Create padding mask from original input
#         mask = create_mask(torch.tensor(x_lengths,device=x.device), seq_len,batch_size,x.device)
        
#         x = self.input_projection(x.permute(0, 2, 1))

#         token = self.cls_token.expand(batch_size, -1, -1)
#         x = torch.cat((token, x), dim=2)
#         mask=torch.cat((torch.ones(batch_size,1,device=x.device), mask), dim=1)
        
#         if self.num_feature_layers>0:
#             # Store intermediate feature extractor outputs for UNet-style skip connections
#             if self.skip_connect:
#                 # Get intermediate features from each layer of feature extractor
#                 x, feature_maps = self.feature_extractor(x, mask, return_intermediates=True)
#             else:
#                 x = self.feature_extractor(x, mask)
            
#             # If in pretraining mode, return pooled features from feature extractor
#             if self.pretraining:
#                 # Choose pooling method based on use_cls_token
#                 if self.use_cls_token:
#                     # Use CLS token (first position) for pooling
#                     pooled_features = x[:, :, 0]
#                 else:
#                     # Use chosen pooling method, exclude CLS token (position 0)
#                     pooled_features, _, _ = self.pooling(x[:, :, 1:].permute(0, 2, 1), mask[:, 1:])
#                 return pooled_features
            
#         if self.add_positional_encoding:
#             x = self.positional_encoding(x)
        
#             if self.channel_masking_rate > 0:
#                 x = x.unsqueeze(2)
#                 x = self.dropout(x)
#                 x = x.squeeze(2)
                
#         # Apply encoder layers with UNet-style skip connections
#         if self.skip_connect and self.num_feature_layers > 0:
#             # UNet-style: connect each encoder layer with reversed feature layer (U-shape)
#             # First encoder layer gets last feature map, last encoder layer gets first feature map
#             for i, layer in enumerate(self.encoder1):
#                 if self.skip_cross_attention:
#                     # Use cross-attention to integrate skip connections
#                     if i < len(feature_maps):
#                         skip_idx = len(feature_maps) - 1 - i  # Reverse the index for U-Net structure
#                         encoder_states = feature_maps[skip_idx]  # Skip connection as encoder states
#                         x = layer(x, encoder_states, mask)  # Cross-attention integration
#                     else:
#                         x = layer(x, None, mask)  # No skip connection available
#                 else:
#                     # Use simple addition for skip connections
#                     x = layer(x, None, mask)
#                     if i < len(feature_maps):
#                         skip_idx = len(feature_maps) - 1 - i  # Reverse the index for U-Net structure
#                         x = x + feature_maps[skip_idx]  # UNet-style skip connection
#         else:
#             # Standard forward without skip connections
#             for layer in self.encoder1:
#                 x = layer(x, None, mask)
        
#         # Choose pooling method based on use_cls_token
#         if self.use_cls_token:
#             # Use CLS token (first position) for pooling
#             pooled_features = x[:, :, 0]
#         else:
#             # Use chosen pooling method, exclude CLS token (position 0)
#             pooled_features, _, _ = self.pooling(x[:, :, 1:].permute(0, 2, 1), mask[:, 1:])
        
#         # If in pretraining mode, return pooled features for contrastive learning
#         if self.pretraining:
#             return pooled_features
        
#         attention_weights = torch.zeros(batch_size, seq_len, 1, device=x.device)
        
#         x1 = self.out1(pooled_features)
#         x3 = self.out3(pooled_features)
#         x2 = self.out2(x[:, :, 1:]) 

#         mask = mask[:, 1:].reshape(-1).bool()
#         x2 = x2.view(-1)
#         x2 = x2[mask]
#         attention_weights = attention_weights.reshape(-1)
#         attention_weights = attention_weights[mask]

#         return x1, x2, x3, attention_weights, None, None
    
#     def set_mode(self, pretraining=True):
#         """
#         Switch between pretraining and fine-tuning modes.
        
#         Args:
#             pretraining: If True, model is in pretraining mode. If False, fine-tuning mode.
#         """
#         self.pretraining = pretraining
    
    
#     def freeze_encoder(self):
#         """
#         Freeze the encoder parameters including input_projection, feature_extractor, 
#         positional_encoding, and encoder1 layers.
#         """
#         # Freeze input projection
#         for param in self.input_projection.parameters():
#             param.requires_grad = False
        
#         # Freeze feature extractor
#         for param in self.feature_extractor.parameters():
#             param.requires_grad = False
        
#         # Freeze positional encoding
#         for param in self.positional_encoding.parameters():
#             param.requires_grad = False
        
#         # Freeze encoder1 layers
#         for layer in self.encoder1:
#             for param in layer.parameters():
#                 param.requires_grad = False
    
#     def unfreeze_encoder(self):
#         """
#         Unfreeze the encoder parameters.
#         """
#         # Unfreeze input projection
#         for param in self.input_projection.parameters():
#             param.requires_grad = True
        
#         # Unfreeze feature extractor
#         for param in self.feature_extractor.parameters():
#             param.requires_grad = True
        
#         # Unfreeze positional encoding
#         for param in self.positional_encoding.parameters():
#             param.requires_grad = True
        
#         # Unfreeze encoder1 layers
#         for layer in self.encoder1:
#             for param in layer.parameters():
#                 param.requires_grad = True
    
#     def freeze_feature_extractor(self):
#         """
#         Freeze only the feature extractor for hybrid pretraining approach.
#         This allows fine-tuning of Mamba encoder while keeping pretrained features frozen.
#         """
#         # Freeze input projection
#         for param in self.input_projection.parameters():
#             param.requires_grad = False
        
#         # Freeze feature extractor
#         for param in self.feature_extractor.parameters():
#             param.requires_grad = False
        
#         # Keep positional encoding and encoder1 layers trainable
#         print("Frozen feature extractor (input_projection + feature_extractor)")
#         print("Mamba encoder layers remain trainable for fine-tuning")
    
#     def unfreeze_feature_extractor(self):
#         """
#         Unfreeze the feature extractor for end-to-end training.
#         """
#         # Unfreeze input projection
#         for param in self.input_projection.parameters():
#             param.requires_grad = True
        
#         # Unfreeze feature extractor
#         for param in self.feature_extractor.parameters():
#             param.requires_grad = True
    
#     def load_pretrained_weights(self, checkpoint_path):
#         """
#         Load pretrained encoder weights from checkpoint for fine-tuning.
#         Supports both direct encoder checkpoints and contrastive learning checkpoints.
#         """
#         checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
#         if 'encoder_weights' in checkpoint:
#             # Direct encoder weights saved during pretraining
#             weights = checkpoint['encoder_weights']
#             self.input_projection.load_state_dict(weights['input_projection'])
#             self.feature_extractor.load_state_dict(weights['feature_extractor'])
#             self.positional_encoding.load_state_dict(weights['positional_encoding'])
#             for i, layer_weights in enumerate(weights['encoder1']):
#                 self.encoder1[i].load_state_dict(layer_weights)
#         elif 'student.input_projection.conv.weight' in checkpoint:
#             # DINO or other contrastive learning checkpoint - extract student weights
#             student_weights = {}
#             for key, value in checkpoint.items():
#                 if key.startswith('student.'):
#                     # Remove 'student.' prefix
#                     new_key = key[8:]  # len('student.') = 8
#                     student_weights[new_key] = value
            
#             # Extract encoder components
#             input_proj_weights = {k.replace('input_projection.', ''): v 
#                                 for k, v in student_weights.items() 
#                                 if k.startswith('input_projection.')}
#             feature_ext_weights = {k.replace('feature_extractor.', ''): v 
#                                  for k, v in student_weights.items() 
#                                  if k.startswith('feature_extractor.')}
#             pos_enc_weights = {k.replace('positional_encoding.', ''): v 
#                              for k, v in student_weights.items() 
#                              if k.startswith('positional_encoding.')}
#             encoder1_weights = {k.replace('encoder1.', ''): v 
#                               for k, v in student_weights.items() 
#                               if k.startswith('encoder1.')}
            
#             # Load the weights
#             self.input_projection.load_state_dict(input_proj_weights)
#             self.feature_extractor.load_state_dict(feature_ext_weights)
#             self.positional_encoding.load_state_dict(pos_enc_weights)
            
#             # Load encoder1 layer weights
#             encoder1_layers = {}
#             for key, value in encoder1_weights.items():
#                 layer_idx = int(key.split('.')[0])
#                 param_key = '.'.join(key.split('.')[1:])
#                 if layer_idx not in encoder1_layers:
#                     encoder1_layers[layer_idx] = {}
#                 encoder1_layers[layer_idx][param_key] = value
            
#             for layer_idx, layer_weights in encoder1_layers.items():
#                 self.encoder1[layer_idx].load_state_dict(layer_weights)
#         else:
#             # Try to load compatible weights (fallback)
#             self.load_state_dict(checkpoint, strict=False)
        
#         print(f"Loaded pretrained weights from {checkpoint_path}")


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
    
    
class MBA_tsm_encoder_decoder_ch_bottleneck(nn.Module):
    """
    Encoder-decoder architecture for multi-task scratch detection.
    Encoder processes input, decoder generates outputs with cross-attention.
    """
    def __init__(self, input_dim, 
                 num_filters=64,
                 num_encoder_layers=2,
                 num_decoder_layers=1,  # Single decoder for one class
                 drop_path_rate=0.3,
                 kernel_size_mba=7,
                 max_seq_len=256,
                 cls_token=False,
                 add_positional_encoding=True,
                 num_heads=4,
                 dropout_rate=0.2,
                 mba=True,
                 supcon_loss=False,
                 cls_src="encoder"):  # "encoder" or "decoder" for classification source
        super(MBA_tsm_encoder_decoder_ch_bottleneck, self).__init__()
        
        self.add_positional_encoding = add_positional_encoding
        self.max_seq_len = max_seq_len
        self.pooling = cls_token
        self.supcon_loss = supcon_loss
        self.cls_src = cls_src
        self.num_filters = num_filters
    
        self.positional_encoding = PositionalEncoding(d_model=num_filters, max_len=max_seq_len)  # +1 for CLS token
        self.MultiHeadSelfAttentionPooling = MultiHeadSelfAttentionPooling(input_dim=num_filters, hidden_dim=num_filters, num_heads=num_heads)

        # Encoder layers (bidirectional)
        if not mba:
            self.encoder = nn.ModuleList([AttModule(2 ** i, num_filters, num_filters, 2, 2, 'normal_att', 'encoder', alpha=1, 
                                                  kernel_size=kernel_size_mba, dropout_rate=dropout_rate) for i in range(num_encoder_layers)])
        else:
            self.encoder = nn.ModuleList([AttModule_mamba_causal(2 ** i, num_filters, num_filters, 'sliding_att', 'encoder', 1, 
                                                        drop_path_rate=drop_path_rate, dropout_rate=dropout_rate, 
                                                        kernel_size=kernel_size_mba,
                                                        ) for i in range(num_encoder_layers)])

        # Decoder layers (cross-attention following MyTransformer approach)
        if not mba:
            self.decoder = nn.ModuleList([AttModule_cross(2 ** i, num_filters, num_filters, 2, 2, 'cross_att', 'decoder', alpha=1, 
                                                        kernel_size=kernel_size_mba, dropout_rate=dropout_rate) for i in range(num_decoder_layers)])
        else:
            self.decoder = nn.ModuleList([AttModule_mamba_cross(2 ** i, num_filters, num_filters, 2, 2, 'cross_att', 'decoder', 1, 
                                                              drop_path_rate=drop_path_rate, dropout_rate=dropout_rate, 
                                                              kernel_size=kernel_size_mba) for i in range(num_decoder_layers)])
        
        self.encoder_expand = nn.Conv1d(input_dim, num_filters, 1)

        # Compression layer after encoder: [B, num_filters, seq_len] -> [B, 1, seq_len]
        self.encoder_compress = nn.Conv1d(num_filters, 1, 1)
        
        # Expansion layer before decoder: [B, 1, seq_len] -> [B, num_filters, seq_len]
        self.decoder_expand = nn.Conv1d(1, num_filters, 1)
        self.encoder_norm = nn.LayerNorm(num_filters)
        # BN
        # self.encoder_norm = nn.BatchNorm1d(num_filters)
        # IN
        # self.encoder_norm = nn.InstanceNorm1d(num_filters)
        self.decoder_compress = nn.Conv1d(num_filters, 1, 1) # Sequence labeling head
        
        # All three output heads use pooled sequence (max_seq_len dimension)
        # self.out1 = nn.Linear(max_seq_len, 1)   # Classification head   
        # self.out3 = nn.Linear(max_seq_len, 1)   # Regression head
        self.out1 = nn.Linear(num_filters, 1)
        self.out3 = nn.Linear(num_filters, 1)


    def forward(self, x, padding_value=-999.0, **kwargs):
        batch_size, seq_len, channels = x.size()       
        x_original = x
        padding_mask_original = ~torch.any(x_original == padding_value, dim=-1)  # [B, seq_len]

        # Handle extreme padding values
        x_for_features = x.clone()

        if abs(padding_value) > 100:
            padding_mask_for_replacement = torch.any(x == padding_value, dim=-1, keepdim=True)
            x_for_features = torch.where(padding_mask_for_replacement, 0.0, x_for_features)

        # Transpose for conv layers: [B, seq_len, num_filters] -> [B, num_filters, seq_len]
        x = x_for_features.permute(0, 2, 1) 
        # Project input to model dimension: [B, input_dim, seq_len] -> [B, num_filters, seq_len]
        x = self.encoder_expand(x)
        x = self.encoder_norm(x.permute(0, 2, 1)).permute(0, 2, 1)
         
        if self.add_positional_encoding:
            x = self.positional_encoding(x)
        
        current_seq_len = x.shape[2]  # seq_len (includes CLS token)
        padding_mask = padding_mask_original.float()
        
        # Encoder pass - no skip connections needed
        encoder_out = x  # [B, num_filters, seq_len]
        for layer in self.encoder:
            encoder_out = layer(encoder_out, None, padding_mask)
        
        # Compress encoder output: [B, num_filters, seq_len] -> [B, 1, seq_len]
        compressed_encoder = self.encoder_compress(encoder_out)  # [B, 1, seq_len]
        
        # Apply softmax and mask for information bottleneck
        # compressed_bottleneck = F.softmax(compressed_encoder, dim=1)  # [B, 1, seq_len] - probability bottleneck
        compressed_bottleneck = torch.sigmoid(compressed_encoder)
        decoder_mask_expanded = padding_mask.unsqueeze(1)  # [B, 1, seq_len]
        compressed_bottleneck = compressed_bottleneck * decoder_mask_expanded
        
        # Expand for decoder input: [B, 1, seq_len] -> [B, num_filters, seq_len]
        decoder_out = self.decoder_expand(compressed_bottleneck)  # [B, num_filters, seq_len]
    
        # Following video-mamba-suite MyTransformer: use compressed bottleneck and encoder features both masked
        masked_encoder_out = encoder_out * decoder_mask_expanded  # [B, num_filters, seq_len]
        # Apply decoder layers following MyTransformer approach
        for layer in self.decoder:
            # Pass both masked inputs to decoder layer (following MyTransformer pattern)
            decoder_out = layer(decoder_out, masked_encoder_out, padding_mask)
        
        # Always use CLS token (position 0) for task predictions
        attention_weights = torch.zeros(batch_size, current_seq_len, 1, device=x.device)
        # pooled_features = self.decoder_compress(decoder_out).squeeze(1)  # [B, seq_len]
        
        # # Multi-task outputs using from specified source
        # x1 = self.out1(pooled_features)  # Classification from CLS token
        # x3 = self.out3(pooled_features)  # Regression from CLS token
        # x2 = pooled_features
        decoder_out_transposed = decoder_out.permute(0, 2, 1)  # [B, seq_len, num_filters]
        
        # Use multi-head attention pooling for global representation
        cls_token, weighted_features, attention_weights = self.MultiHeadSelfAttentionPooling(
            decoder_out_transposed, padding_mask_original
        )  # [B, num_filters], [B, seq_len, 1]
        # Multi-task outputs
        x1 = self.out1(cls_token)  # Classification from attention pooled features
        x3 = self.out3(cls_token)  # Regression from attention pooled features
        x2 = self.decoder_compress(decoder_out).squeeze(1)  # [B, seq_len] - Sequence labeling

        if self.supcon_loss:
            contrastive_embedding = pooled_features 
        else:
            contrastive_embedding = None
        mixup_info = None
        return x1, x2, x3, attention_weights, contrastive_embedding, mixup_info


class MBA_tsm_encoder_decoder_seq_bottleneck(nn.Module):
    """
    Encoder-decoder architecture with sequence-level information bottleneck.
    Compresses sequence length from seq_len to seq_len/reduction while keeping num_filters channels.
    """
    def __init__(self, input_dim, 
                 num_filters=64,
                 num_encoder_layers=2,
                 num_decoder_layers=1,
                 drop_path_rate=0.3,
                 kernel_size_mba=7,
                 max_seq_len=256,
                 cls_token=False,
                 add_positional_encoding=True,
                 num_heads=4,
                 dropout_rate=0.2,
                 mba=True,
                 supcon_loss=False,
                 cls_src="encoder",
                 seq_reduction=8):  # Sequence compression factor
        super(MBA_tsm_encoder_decoder_seq_bottleneck, self).__init__()
        
        self.add_positional_encoding = add_positional_encoding
        self.max_seq_len = max_seq_len
        self.pooling = cls_token
        self.supcon_loss = supcon_loss
        self.cls_src = cls_src
        self.num_filters = num_filters
        self.seq_reduction = seq_reduction
        self.compressed_seq_len = max_seq_len // seq_reduction
        
        self.positional_encoding = PositionalEncoding(d_model=num_filters, max_len=max_seq_len)
        self.MultiHeadSelfAttentionPooling = MultiHeadSelfAttentionPooling(input_dim=num_filters, hidden_dim=num_filters, num_heads=num_heads)

        # Encoder layers
        if not mba:
            self.encoder = nn.ModuleList([AttModule(2 ** i, num_filters, num_filters, 2, 2, 'normal_att', 'encoder', alpha=1, 
                                                  kernel_size=kernel_size_mba, dropout_rate=dropout_rate) for i in range(num_encoder_layers)])
        else:
            self.encoder = nn.ModuleList([AttModule_mamba_causal(2 ** i, num_filters, num_filters, 'sliding_att', 'encoder', 1, 
                                                        drop_path_rate=drop_path_rate, dropout_rate=dropout_rate, 
                                                        kernel_size=kernel_size_mba) for i in range(num_encoder_layers)])
        
        # Decoder layers (cross-attention)
        if not mba:
            self.decoder = nn.ModuleList([AttModule_cross(2 ** i, num_filters, num_filters, 2, 2, 'cross_att', 'decoder', alpha=1, 
                                                        kernel_size=kernel_size_mba, dropout_rate=dropout_rate) for i in range(num_decoder_layers)])
        else:
            self.decoder = nn.ModuleList([AttModule_mamba_cross(2 ** i, num_filters, num_filters, 2, 2, 'cross_att', 'decoder', 1, 
                                                              drop_path_rate=drop_path_rate, dropout_rate=dropout_rate, 
                                                              kernel_size=kernel_size_mba) for i in range(num_decoder_layers)])
        
        # Sequence compression/expansion layers
        self.encoder_expand = nn.Conv1d(input_dim, num_filters, 1)
        self.encoder_norm = nn.LayerNorm(num_filters)
        # BN
        # self.encoder_norm = nn.BatchNorm1d(num_filters)
        # IN
        # self.encoder_norm = nn.InstanceNorm1d(num_filters)

        # Sequence compression: [B, num_filters, seq_len] -> [B, num_filters, seq_len/reduction] 
        self.seq_compress = nn.Conv1d(num_filters, num_filters, kernel_size=seq_reduction, stride=seq_reduction, padding=0)
        
        # Sequence expansion: [B, num_filters, seq_len/reduction] -> [B, num_filters, seq_len]
        self.seq_expand = nn.ConvTranspose1d(num_filters, num_filters, kernel_size=seq_reduction, stride=seq_reduction, padding=0)
        
        # Output heads for sequence predictions  
        self.decoder_compress = nn.Conv1d(num_filters, 1, 1)  # Sequence labeling head
        
        # Task-specific heads using attention pooling
        self.out1 = nn.Linear(num_filters, 1)   # Classification from attention pooled features   
        self.out3 = nn.Linear(num_filters, 1)   # Regression from attention pooled features

    def forward(self, x, padding_value=-999.0, **kwargs):
        batch_size, seq_len, channels = x.size()       
        x_original = x
        padding_mask_original = ~torch.any(x_original == padding_value, dim=-1)  # [B, seq_len]

        # Handle extreme padding values
        x_for_features = x.clone()
        if abs(padding_value) > 100:
            padding_mask_for_replacement = torch.any(x == padding_value, dim=-1, keepdim=True)
            x_for_features = torch.where(padding_mask_for_replacement, 0.0, x_for_features)

        # Transpose for conv layers: [B, seq_len, channels] -> [B, channels, seq_len]
        x_for_features = x_for_features.permute(0, 2, 1) 
        # Project input to model dimension: [B, channels, seq_len] -> [B, num_filters, seq_len]
        x = self.encoder_expand(x_for_features)
        x = self.norm(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        
        if self.add_positional_encoding:
            x = self.positional_encoding(x)
        
        current_seq_len = x.shape[2]  # seq_len
        padding_mask = padding_mask_original.float()
        
        # Encoder pass
        encoder_out = x  # [B, num_filters, seq_len]
        for layer in self.encoder:
            encoder_out = layer(encoder_out, None, padding_mask)
        
        # Sequence compression: [B, num_filters, seq_len] -> [B, num_filters, seq_len/reduction]
        compressed_encoder = self.seq_compress(encoder_out)  # [B, num_filters, compressed_seq_len]
        compressed_seq_len = compressed_encoder.shape[2]
        
        # Create compressed padding mask
        compressed_padding_mask = F.max_pool1d(
            padding_mask.unsqueeze(1).float(), 
            kernel_size=self.seq_reduction, 
            stride=self.seq_reduction
        ).squeeze(1)  # [B, compressed_seq_len]
        
        # Apply sigmoid activation for information bottleneck
        compressed_bottleneck = torch.sigmoid(compressed_encoder)  # [B, num_filters, compressed_seq_len]
        compressed_bottleneck = compressed_bottleneck * compressed_padding_mask.unsqueeze(1)
        
        # Expand back to original sequence length: [B, num_filters, compressed_seq_len] -> [B, num_filters, seq_len]
        decoder_out = self.seq_expand(compressed_bottleneck)  # [B, num_filters, seq_len]
        
        # Ensure decoder output matches original sequence length
        if decoder_out.shape[2] != current_seq_len:
            decoder_out = F.interpolate(decoder_out, size=current_seq_len, mode='linear', align_corners=False)
        
        # Masked encoder output for cross-attention
        padding_mask_expanded = padding_mask.unsqueeze(1)  # [B, 1, seq_len]
        masked_encoder = encoder_out * padding_mask_expanded  # [B, num_filters, seq_len]
        
        # Apply decoder layers with cross-attention
        for layer in self.decoder:
            decoder_out = layer(decoder_out, masked_encoder, padding_mask)
        
        # Generate outputs using attention pooling
        # Convert to [B, seq_len, num_filters] for attention pooling
        decoder_out_transposed = decoder_out.permute(0, 2, 1)  # [B, seq_len, num_filters]
        
        # Use multi-head attention pooling for global representation
        cls_token, weighted_features, attention_weights = self.MultiHeadSelfAttentionPooling(
            decoder_out_transposed, padding_mask_original
        )  # [B, num_filters], [B, seq_len, 1]
        # Multi-task outputs
        x1 = self.out1(cls_token)  # Classification from attention pooled features
        x3 = self.out3(cls_token)  # Regression from attention pooled features
        x2 = self.decoder_compress(decoder_out).squeeze(1)  # [B, seq_len] - Sequence labeling
        
        if self.supcon_loss:
            contrastive_embedding = cls_token  # Use attention pooled features for contrastive learning
        else:
            contrastive_embedding = None
        
        mixup_info = None
        return x1, x2, x3, attention_weights, contrastive_embedding, mixup_info

class MBA_tsm_encoder_decoder_progressive_with_skip_connection(nn.Module):
    """
    Pure encoder-decoder architecture for multi-task scratch detection.
    Encoder processes raw accelerometer signals directly without FeatureExtractor.
    Decoder generates outputs with cross-attention to encoder states.
    """
    def __init__(self, input_dim, 
                 n_state=16,
                 pos_embed_dim=16,
                 num_filters=64,
                 BI=True,
                 num_encoder_layers=4,
                 num_decoder_layers=3,
                 drop_path_rate=0.3,
                 kernel_size_mba=7,
                 max_seq_len=1220,
                 cls_token=True,
                 add_positional_encoding=True,
                 num_heads=4,
                 dropout_rate=0.2,
                 mba=True,
                 average_mask=False,
                 average_window_size=20,
                 supcon_loss=False,
                 cls_src="encoder"):
        super(MBA_tsm_encoder_decoder_progressive_with_skip_connection, self).__init__()
        
        self.add_positional_encoding = add_positional_encoding
        self.max_seq_len = max_seq_len
        self.pooling = cls_token
        self.average_mask = average_mask
        self.average_window_size = average_window_size
        self.supcon_loss = supcon_loss
        self.input_dim = input_dim
        self.num_filters = num_filters
        self.cls_src = cls_src

        # Input projection to transform raw signals to model dimension
        self.input_projection = nn.Conv1d(input_dim, num_filters, 1)
        self.norm = nn.LayerNorm(num_filters)
        # BN
        # self.norm = nn.BatchNorm1d(num_filters)
        # IN
        # self.norm = nn.InstanceNorm1d(num_filters)
        
        # Progressive depth reduction (U-Net style) architecture
        # Define progressive feature dimensions FIRST
        self.encoder_dims = [num_filters * (2 ** i) for i in range(num_encoder_layers)]  # [64, 128, 256, 512]
        self.decoder_dims = [self.encoder_dims[-(i+1)] for i in range(num_decoder_layers)]  # [512, 256, 128] (reverse)
        
        # Optional positional encoding
        if add_positional_encoding:
            self.positional_encoding = PositionalEncoding(d_model=num_filters, max_len=max_seq_len)
        
        # Pooling layer for final representations (use final encoder dimension)
        self.MultiHeadSelfAttentionPooling = MultiHeadSelfAttentionPooling(
            input_dim=self.encoder_dims[-1], hidden_dim=self.encoder_dims[-1], num_heads=num_heads
        )
        
        # Add dimension projection layers for progressive feature expansion/reduction
        self.encoder_projections = nn.ModuleList([
            nn.Conv1d(self.encoder_dims[i-1] if i > 0 else num_filters, self.encoder_dims[i], 1)
            for i in range(num_encoder_layers)
        ])
        
        self.decoder_projections = nn.ModuleList([
            nn.Conv1d(self.decoder_dims[i-1] if i > 0 else self.encoder_dims[-1], self.decoder_dims[i], 1)
            for i in range(num_decoder_layers)
        ])
        
        # Skip connection projection layers to match dimensions
        self.skip_projections = nn.ModuleList([
            nn.Conv1d(self.encoder_dims[-(i+1)], self.decoder_dims[i], 1)
            for i in range(min(num_decoder_layers, num_encoder_layers))
        ])

        # Encoder layers with progressive feature dimensions
        if not mba:
            self.encoder = nn.ModuleList([
                AttModule(2 ** i, self.encoder_dims[i], self.encoder_dims[i], 2, 2, 'normal_att', 'encoder', alpha=1, 
                         kernel_size=kernel_size_mba, dropout_rate=dropout_rate) 
                for i in range(num_encoder_layers)
            ])
        else:
            self.encoder = nn.ModuleList([
                AttModule_mamba_causal(2 ** i, self.encoder_dims[i], self.encoder_dims[i], 'sliding_att', 'encoder', 1, 
                               drop_path_rate=drop_path_rate, dropout_rate=dropout_rate, 
                               kernel_size=kernel_size_mba) 
                for i in range(num_encoder_layers)
            ])

        # Decoder layers with progressive feature dimensions
        if not mba:
            self.decoder = nn.ModuleList([
                AttModule_cross(2 ** i, self.decoder_dims[i], self.decoder_dims[i], 2, 2, 'cross_att', 'decoder', alpha=1, 
                               kernel_size=kernel_size_mba, dropout_rate=dropout_rate) 
                for i in range(num_decoder_layers)
            ])
        else:
            self.decoder = nn.ModuleList([
                AttModule_mamba_cross(2 ** i, self.decoder_dims[i], self.decoder_dims[i], 2, 2, 'cross_att', 'decoder', 1, 
                                     drop_path_rate=drop_path_rate, dropout_rate=dropout_rate, 
                                     kernel_size=kernel_size_mba) 
                for i in range(num_decoder_layers)
            ])
        
        # Task-specific output heads (use final encoder/decoder dimensions)
        if self.cls_src == "encoder":
            self.out1 = nn.Linear(self.encoder_dims[-1], 1)  # Classification head (from encoder CLS)
            self.out3 = nn.Linear(self.encoder_dims[-1], 1)  # Regression head (from encoder CLS)
        else:
            self.out1 = nn.Linear(self.decoder_dims[-1], 1)
            self.out3 = nn.Linear(self.decoder_dims[-1], 1)
        self.out2 = nn.Conv1d(self.decoder_dims[-1], 1, 1)  # Sequence labeling head (from decoder)


        if self.average_mask:
            self.scale_out2 = nn.Linear(max_seq_len, max_seq_len//average_window_size)

        # Remove projection layers - use raw cls_token directly for contrastive learning

    def forward(self, x, labels1=None, labels3=None, apply_mixup=False, mixup_alpha=0.2, padding_value=-999.0,  **kwargs):
        batch_size, seq_len, channels = x.size()       
        x_original = x
        
        # Create padding mask from original input
        padding_mask_original = ~torch.any(x_original == padding_value, dim=-1)  # [B, seq_len]
        
        # Handle extreme padding values for processing
        x_processed = x.clone()
        if abs(padding_value) > 100:
            padding_mask_for_replacement = torch.any(x == padding_value, dim=-1, keepdim=True)
            x_processed = torch.where(padding_mask_for_replacement, 0.0, x_processed)
        
        # Project input to model dimension: [B, seq_len, input_dim] -> [B, seq_len, num_filters]
        x = self.input_projection(x_processed.permute(0, 2, 1))
        
        # Transpose for conv layers: [B, seq_len, num_filters] -> [B, num_filters, seq_len]
        x = self.norm(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        
        # Add positional encoding if enabled
        if self.add_positional_encoding:
            x = self.positional_encoding(x)

        # Use original padding mask (sequence length unchanged)
        padding_mask = padding_mask_original.float()
        current_seq_len = seq_len  # No change in sequence length
        
        # Progressive encoder pass with dimension expansion
        encoder_states = []
        encoder_out = x  # Start with [B, num_filters, seq_len]
        
        for i, (proj, layer) in enumerate(zip(self.encoder_projections, self.encoder)):
            # Progressive dimension expansion: num_filters -> encoder_dims[i]
            encoder_out = proj(encoder_out)  # [B, encoder_dims[i], seq_len]
            encoder_out = layer(encoder_out, None, padding_mask)
            encoder_states.append(encoder_out)
        
        # Extract CLS token from encoder output for information bottleneck
        # encoder_out shape: [B, encoder_dims[-1], seq_len] -> permute to [B, seq_len, encoder_dims[-1]]
        encoder_features = encoder_out.permute(0, 2, 1)  # [B, seq_len, encoder_dims[-1]]
        
        # Multi-head attention pooling to get global representation
        cls_token, weighted_features, attention_weights = self.MultiHeadSelfAttentionPooling(
            encoder_features, padding_mask
        )
        
        # SEQUENCE PREDICTION: Choose between encoder-only or encoder-decoder approach
        # Option 1: Use decoder (comment out for encoder-only)
        # DECODER PASS: Use full encoder features instead of CLS token bottleneck
        # Use encoder output directly as decoder input (preserves spatial information)
        decoder_input = encoder_out  # [B, num_filters, seq_len] - already in correct format
        
        # Use the original padding mask for decoder (respects actual sequence lengths)
        decoder_mask = padding_mask  # Use same mask as encoder
        
        # Progressive decoder pass with dimension reduction and skip connections
        decoder_out = decoder_input  # Initialize decoder with encoder features [B, encoder_dims[-1], seq_len]
        
        for i, (proj, layer) in enumerate(zip(self.decoder_projections, self.decoder)):
            # Progressive dimension reduction: decoder_dims[i-1] -> decoder_dims[i]
            decoder_out = proj(decoder_out)  # [B, decoder_dims[i], seq_len]
            
            # Skip connection from corresponding encoder layer
            if i < len(encoder_states) and i < len(self.skip_projections):
                # U-Net style: reverse order (last encoder -> first decoder)
                encoder_skip = encoder_states[-(i+1)]  # [B, encoder_dims[-(i+1)], seq_len]
                # Project skip connection to match decoder dimension
                encoder_skip_proj = self.skip_projections[i](encoder_skip)  # [B, decoder_dims[i], seq_len]
            else:
                encoder_skip_proj = encoder_states[-1]  # Fallback to final encoder state
            
            # Decoder processes with cross-attention to encoder skip connection
            decoder_out = layer(decoder_out, encoder_skip_proj, decoder_mask)
        
        # Apply mixup augmentation to CLS token if enabled
        if self.cls_src == "encoder":
            cls_feature = cls_token
        else:
            cls_feature = decoder_out[:, :, 0]
            # cls_feature = decoder_out[:, :, 1:].mean(dim=-1)

        if apply_mixup and self.training:
            mixed_cls_features, mixed_labels1, mixed_labels3, mixup_lambda = latent_mixup(
                cls_feature, labels1, labels3, alpha=mixup_alpha
            )
            mixup_info = (mixed_labels1, mixed_labels3, mixup_lambda)
        else:
            mixed_cls_features = cls_feature
            mixup_info = None
  
        # ENCODER OUTPUTS: Classification and Regression from CLS token
        x1 = self.out1(mixed_cls_features)  # Classification from encoder CLS: [B, 1]
        x3 = self.out3(mixed_cls_features)  # Regression from encoder CLS: [B, 1]

        # DECODER OUTPUT: Sequence mask prediction
        x2 = self.out2(decoder_out)     # Sequence labeling from decoder: [B, 1, seq_len]
        x2 = x2.squeeze(1)              # [B, seq_len]
        
        # Option 2: Use encoder output directly (uncomment for encoder-only)
        # x2 = self.out2(encoder_out)    # Sequence labeling from encoder: [B, 1, seq_len] 
        # x2 = x2.squeeze(1)             # [B, seq_len]

        # Apply mask averaging if enabled
        if self.average_mask:
            x2 = x2.view(batch_size, seq_len//self.average_window_size, self.average_window_size).mean(dim=2)

        # Contrastive embedding from encoder CLS token
        if self.supcon_loss:
            contrastive_embedding = cls_feature  # Use raw CLS token directly
        else:
            contrastive_embedding = None

        return x1, x2, x3, attention_weights, contrastive_embedding, mixup_info

    def train(self, mode=True):
        """Standard train method - no feature extractor to handle specially."""
        return super().train(mode)


class MBA_tsm_encoder_decoder_progressive(nn.Module):
    """
    Simplified progressive encoder-decoder architecture without skip connections.
    Uses progressive dimension expansion/reduction for efficient information flow.
    """
    def __init__(self, input_dim, 
                 n_state=16,
                 pos_embed_dim=16,
                 num_filters=64,
                 BI=True,
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
                 reg_only=False):
        super(MBA_tsm_encoder_decoder_progressive, self).__init__()
        
        self.add_positional_encoding = add_positional_encoding
        self.max_seq_len = max_seq_len
        self.pooling = cls_token
        self.average_mask = average_mask
        self.average_window_size = average_window_size
        self.supcon_loss = supcon_loss
        self.reg_only = reg_only
        self.input_dim = input_dim
        self.num_filters = num_filters

        # Input projection to transform raw signals to model dimension
        self.input_projection = nn.Linear(input_dim, num_filters)
        
        # Progressive dimension architecture (simplified - no skip connections)
        # Define progressive feature dimensions
        self.encoder_dims = [num_filters * (2 ** i) for i in range(num_encoder_layers)]  # [64, 128, 256, 512]
        self.decoder_dims = [self.encoder_dims[-(i+1)] for i in range(num_decoder_layers)]  # [512, 256, 128] (reverse)
        
        # Optional positional encoding
        if add_positional_encoding:
            self.positional_encoding = PositionalEncoding(d_model=num_filters, max_len=max_seq_len)
        
        # Pooling layer for final representations (use final encoder dimension)
        self.MultiHeadSelfAttentionPooling = MultiHeadSelfAttentionPooling(
            input_dim=self.encoder_dims[-1], hidden_dim=self.encoder_dims[-1], num_heads=num_heads
        )
        
        # Dimension projection layers for progressive feature expansion/reduction
        self.encoder_projections = nn.ModuleList([
            nn.Conv1d(self.encoder_dims[i-1] if i > 0 else num_filters, self.encoder_dims[i], 1)
            for i in range(num_encoder_layers)
        ])
        
        self.decoder_projections = nn.ModuleList([
            nn.Conv1d(self.decoder_dims[i-1] if i > 0 else self.encoder_dims[-1], self.decoder_dims[i], 1)
            for i in range(num_decoder_layers)
        ])
        
        # NO skip connection projections - simplified architecture

        # Encoder layers with progressive feature dimensions
        if not mba:
            self.encoder = nn.ModuleList([
                AttModule(2 ** i, self.encoder_dims[i], self.encoder_dims[i], 2, 2, 'normal_att', 'encoder', alpha=1, 
                         kernel_size=kernel_size_mba, dropout_rate=dropout_rate) 
                for i in range(num_encoder_layers)
            ])
        else:
            self.encoder = nn.ModuleList([
                AttModule_mamba_causal(2 ** i, self.encoder_dims[i], self.encoder_dims[i], 'sliding_att', 'encoder', 1, 
                               drop_path_rate=drop_path_rate, dropout_rate=dropout_rate, 
                               kernel_size=kernel_size_mba) 
                for i in range(num_encoder_layers)
            ])

        # Decoder layers with self-attention only (no cross-attention)
        if not mba:
            self.decoder = nn.ModuleList([
                AttModule(2 ** i, self.decoder_dims[i], self.decoder_dims[i], 2, 2, 'normal_att', 'decoder', alpha=1, 
                         kernel_size=kernel_size_mba, dropout_rate=dropout_rate) 
                for i in range(num_decoder_layers)
            ])
        else:
            self.decoder = nn.ModuleList([
                AttModule_mamba(2 ** i, self.decoder_dims[i], self.decoder_dims[i], 'sliding_att', 'decoder', 1, 
                               drop_path_rate=drop_path_rate, dropout_rate=dropout_rate, 
                               kernel_size=kernel_size_mba) 
                for i in range(num_decoder_layers)
            ])
        
        # Task-specific output heads (use final encoder/decoder dimensions)
        if not reg_only:
            self.out1 = nn.Linear(self.encoder_dims[-1], 1)  # Classification head (from encoder CLS)
            self.out2 = nn.Conv1d(self.decoder_dims[-1], 1, 1)  # Sequence labeling head (from decoder)
        self.out3 = nn.Linear(self.encoder_dims[-1], 1)  # Regression head (from encoder CLS)

        if self.average_mask:
            self.scale_out2 = nn.Linear(max_seq_len, max_seq_len//average_window_size)

    def forward(self, x, labels1=None, labels3=None, apply_mixup=False, mixup_alpha=0.2, padding_value=-999.0, **kwargs):
        batch_size, seq_len, channels = x.size()       
        x_original = x
        
        # Create padding mask from original input
        padding_mask_original = ~torch.any(x_original == padding_value, dim=-1)  # [B, seq_len]
        
        # Handle extreme padding values for processing
        x_processed = x.clone()
        if abs(padding_value) > 100:
            padding_mask_for_replacement = torch.any(x == padding_value, dim=-1, keepdim=True)
            x_processed = torch.where(padding_mask_for_replacement, 0.0, x_processed)
        
        # Project input to model dimension: [B, seq_len, input_dim] -> [B, seq_len, num_filters]
        x = self.input_projection(x_processed)
        
        # Transpose for conv layers: [B, seq_len, num_filters] -> [B, num_filters, seq_len]
        x = x.permute(0, 2, 1)
        
        # Add positional encoding if enabled
        if self.add_positional_encoding:
            x = self.positional_encoding(x)

        # Use original padding mask (sequence length unchanged)
        padding_mask = padding_mask_original.float()
        current_seq_len = seq_len  # No change in sequence length
        
        # Progressive encoder pass with dimension expansion (no skip connection storage needed)
        encoder_out = x  # Start with [B, num_filters, seq_len]
        
        for i, (proj, layer) in enumerate(zip(self.encoder_projections, self.encoder)):
            # Progressive dimension expansion: num_filters -> encoder_dims[i]
            encoder_out = proj(encoder_out)  # [B, encoder_dims[i], seq_len]
            encoder_out = layer(encoder_out, None, padding_mask)  # Self-attention only
        
        # Extract CLS token from encoder output
        # encoder_out shape: [B, encoder_dims[-1], seq_len] -> permute to [B, seq_len, encoder_dims[-1]]
        encoder_features = encoder_out.permute(0, 2, 1)  # [B, seq_len, encoder_dims[-1]]
        
        # Get CLS token (first position) from encoder
        if self.pooling:
            cls_token = encoder_features[:, 1:, :].mean(dim=1)  # [B, encoder_dims[-1]] - average pool positions 1:
            attention_weights = torch.zeros(batch_size, current_seq_len, 1, device=x.device)
        else:
            # Multi-head attention pooling to get global representation
            cls_token, weighted_features, attention_weights = self.MultiHeadSelfAttentionPooling(
                encoder_features, padding_mask
            )
        
        # Apply mixup augmentation to CLS token if enabled
        if apply_mixup and self.training:
            mixed_cls_features, mixed_labels1, mixed_labels3, mixup_lambda = latent_mixup(
                cls_token, labels1, labels3, alpha=mixup_alpha
            )
            mixup_info = (mixed_labels1, mixed_labels3, mixup_lambda)
        else:
            mixed_cls_features = cls_token
            mixup_info = None
        
        # ENCODER OUTPUTS: Classification and Regression from CLS token
        if not self.reg_only:
            x1 = self.out1(mixed_cls_features)  # Classification from encoder CLS: [B, 1]
            x3 = self.out3(mixed_cls_features)  # Regression from encoder CLS: [B, 1]
        else:
            x1, x3 = None, self.out3(mixed_cls_features)
        
        # SEQUENCE PREDICTION: Simplified decoder (no skip connections)
        if not self.reg_only:
            # Use original padding mask for decoder
            decoder_mask = padding_mask
            
            # Progressive decoder pass with dimension reduction (NO skip connections)
            decoder_out = encoder_out  # Initialize decoder with encoder features [B, encoder_dims[-1], seq_len]
            
            for i, (proj, layer) in enumerate(zip(self.decoder_projections, self.decoder)):
                # Progressive dimension reduction: decoder_dims[i-1] -> decoder_dims[i]
                decoder_out = proj(decoder_out)  # [B, decoder_dims[i], seq_len]
                
                # Decoder processes with SELF-ATTENTION ONLY (no cross-attention/skip connections)
                decoder_out = layer(decoder_out, None, decoder_mask)
            
            # DECODER OUTPUT: Sequence mask prediction
            x2 = self.out2(decoder_out)     # Sequence labeling from decoder: [B, 1, seq_len]
            x2 = x2.squeeze(1)              # [B, seq_len]

            # Apply mask averaging if enabled
            if self.average_mask:
                x2 = x2.view(batch_size, seq_len//self.average_window_size, self.average_window_size).mean(dim=2)

            # Contrastive embedding from encoder CLS token
            if self.supcon_loss:
                contrastive_embedding = cls_token  # Use raw CLS token directly
            else:
                contrastive_embedding = None

            return x1, x2, x3, attention_weights, contrastive_embedding, mixup_info
        else:
            return None, None, x3, attention_weights, None, mixup_info

    def train(self, mode=True):
        """Standard train method - no feature extractor to handle specially."""
        return super().train(mode)

