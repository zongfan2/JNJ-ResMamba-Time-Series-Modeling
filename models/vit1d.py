import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple
# TODO: fix version error for flash-attn
# from flash_attn import flash_attn_func, flash_attn_qkvpacked_func
# from flash_attn.modules.mha import FlashMultiHeadAttention
try:
    from embed import PatchEmbed1D, PositionalEmbedding2D, MotionTransitionEmbedding, EventFrequencyEmbedding, PatchEmbedWithPos1D
except:
    from .embed import PatchEmbed1D, PositionalEmbedding2D, MotionTransitionEmbedding, EventFrequencyEmbedding, PatchEmbedWithPos1D

class TransformerEncoderBlock(nn.Module):
    """
    Standard Transformer encoder block with multi-head attention.
    """
    def __init__(
        self,
        embed_dim: int = 384,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attention_type: str = 'standard',  # 'standard' or 'flash'
    ):
        super().__init__()
        self.attention_type = attention_type
        # Multi-head attention
        if self.attention_type == 'flash':
            self.attn = FlashMultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            attention_dropout=dropout,
            device=None,  # Will use the device of input tensors
            dtype=None,   # Will use the dtype of input tensors
        )
        else:
            # Standard PyTorch multi-head attention
            self.attn = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # MLP block
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention with residual connection
        x = self.norm1(x)
        if self.attention_type == 'flash':
            flash_mask = None
            if mask is not None:
                 # Flash attention expects a mask with True for tokens to attend to
                # (opposite of PyTorch's MultiheadAttention)
                flash_mask = ~mask if mask.dtype == torch.bool else mask
            attn_output = self.attn(
                x,
                attn_mask=flash_mask,  # Flash attention has different mask format
                need_weights=False
            )[0]  # FlashAttention returns tuple (output, weights)
        
        else:
            attn_output, _ = self.attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=mask
            )
        x = x + attn_output
        
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerDecoderBlock(nn.Module):
    """
    Transformer decoder block with both self-attention and cross-attention.
    """
    def __init__(
        self,
        embed_dim: int = 384,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attention_type: str = 'standard',  # 'standard' or 'flash'
    ):
        super().__init__()
        
        self.attention_type = attention_type
        if self.attention_type == 'flash':
            # Self-attention using Flash Attention
            self.self_attn = FlashMultiHeadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                attention_dropout=dropout
            )
            
            # Cross-attention using Flash Attention
            self.cross_attn = FlashMultiHeadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                attention_dropout=dropout
            )
        else:
            # Self-attention
            self.self_attn = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            
            # Cross-attention to connect with encoder
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        
        # MLP block
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        self_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention with residual
        x = self.norm1(x)
        if self.attention_type == 'flash':
            flash_mask = None
            if self_mask is not None:
                # Flash attention expects a mask with True for tokens to attend to
                # (opposite of PyTorch's MultiheadAttention)
                flash_mask = ~self_mask if self_mask.dtype == torch.bool else self_mask
            self_attn_output = self.self_attn(
                x,
                attn_mask=flash_mask,  # Flash attention has different mask format
                need_weights=False
            )[0]  # FlashAttention returns tuple (output, weights)
        else:

            self_attn_output, _ = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_mask
            )
        x = x + self_attn_output
        
        x = self.norm2(x)
        if self.attention_type == 'flash':
            # Cross-attention with encoder output
            flash_mask = None
            if memory_mask is not None:
                # Flash attention expects a mask with True for tokens to attend to
                # (opposite of PyTorch's MultiheadAttention)
                flash_mask = ~memory_mask if memory_mask.dtype == torch.bool else memory_mask
            cross_attn_output = self.cross_attn(
                query=x,
                key=memory,
                value=memory,
                attn_mask=flash_mask,  # Flash attention has different mask format
                need_weights=False
            )[0]
        else:
            # Cross-attention with encoder output
            cross_attn_output, _ = self.cross_attn(
                query=x,
                key=memory,
                value=memory,
                key_padding_mask=memory_mask
            )
        x = x + cross_attn_output
        
        # MLP
        x = x + self.mlp(self.norm3(x))
        return x


def classification_mixup(features, labels, alpha=0.2):
    """
    Applies mixup to features and corresponding labels for classification head only.
    
    Args:
        features: Feature tensor [batch_size, feature_dim]
        labels: Classification labels [batch_size]
        alpha: Beta distribution parameter for mixing strength
        
    Returns:
        mixed_features: Mixed feature tensor
        mixed_labels: Original and mixed labels with lambda weights
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
    
    # Return original labels, permuted labels, and lambda for loss calculation
    return mixed_features, labels, labels[indices], lam
    

class PredictorHead(nn.Module):
    """ Predict head for classification, regression, and mask prediction tasks.
    """
    def __init__(self, embed_dim: int, mask_length: int=1, num_classes: int = 1, return_contrastive_embedding: bool=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.return_contrastive_embedding = return_contrastive_embedding
        self.mask_length = mask_length
        self.classification_head = nn.Linear(embed_dim, num_classes)
        self.regression_head = nn.Linear(embed_dim, 1)  # For regression tasks
        self.mask_prediction_head = nn.Conv1d(embed_dim, 1, 1)  # For spatial mask prediction using patch tokens


    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor]=None, apply_mixup: bool=False, mixup_alpha=0.2) -> Tuple[torch.Tensor, list]:
        """
        Predict head for classification, regression, and mask prediction tasks.
        Args:
            x: Input tensor of shape [B, 1+num_patches, embed_dim]
        Returns:
            Output tensor of shape [B, 1] for binary prediction, [B, 1] for regression, [B, 1] for mask prediction
        """
        # use cls token

        cls_token = x[:, 0]  # [B, embed_dim]

        # apply mixup 
        if apply_mixup and self.training:
            mixed_features, orig_labels, mixed_labels, mixup_lambda = classification_mixup(
                cls_token, labels, alpha=mixup_alpha
            )
            classification_output = self.classification_head(mixed_features)
            mixup_info = (mixed_labels, mixup_lambda)
        else:
            classification_output = self.classification_head(cls_token)  # [B, num_classes]
            mixup_info = None
        # Mask prediction using patch tokens to preserve spatial information
        patch_tokens = x[:, 1:, :]  # [B, num_patches, embed_dim] - exclude CLS token
        x_conv_patches = patch_tokens.permute(0, 2, 1)  # [B, embed_dim, num_patches]
        mask_prediction_output = self.mask_prediction_head(x_conv_patches)  # [B, 1, num_patches]
        mask_prediction_output = mask_prediction_output.squeeze(1)  # [B, num_patches]

        # Interpolate to target mask length if needed
        if self.mask_length > 1 and mask_prediction_output.size(1) != self.mask_length:
            mask_prediction_output = F.interpolate(
                mask_prediction_output.unsqueeze(1),  # [B, 1, num_patches]
                size=self.mask_length,
                mode='linear',
                align_corners=False
            ).squeeze(1)  # [B, mask_length]
        # use cls token for regression
        regression_output = self.regression_head(cls_token)  # [B, 1]
        
        # Use the CLS token embedding directly as contrastive embedding
        if self.return_contrastive_embedding:
            con_embed = cls_token  # [B, embed_dim]
        else:
            con_embed = None

        return classification_output, mask_prediction_output, regression_output, con_embed, mixup_info

class ViT1D(nn.Module):
    """
    Vision Transformer for 1D time series data with encoder-decoder architecture.
    Can be used for event classification, mask prediction, and duration estimation.
    """
    def __init__(
        self,
        time_length: int = 1200,        # Total length in seconds
        patch_size: int = 200,          # Patch size in seconds
        stride: int = 20,               # Stride for patch extraction
        in_channels: int = 3,         # Number of input channels
        embed_dim: Optional[int] = None, # Embedding dimension (defaults to patch_dim)
        use_embedding: bool = False,    # Whether to apply linear embedding to patches
        encoder_layers: int = 12,     # Number of encoder layers
        decoder_layers: int = 4,      # Number of decoder layers
        num_heads: int = 6,           # Number of attention heads
        mlp_ratio: float = 4.0,       # Ratio for MLP hidden dimension
        dropout: float = 0.1,         # Dropout rate
        load_weights: Optional[str] = None,  # Path to load pretrained weights
        use_motion_embed: bool = False,  # Whether to use motion transition embedding
        use_freq_embed: bool = False,    # Whether to use event frequency embedding
        max_event_count: int = 10,       # Maximum event count for frequency embedding
        padding_value: float = -999.0,  # Value to identify missing data
        padding_threshold: float = 0.5, # Threshold to determine if a patch is missing
        attention_type: str = 'standard',  # 'standard' or 'flash'
        average_mask: bool = False,    # whether to average overlapping mask predictions
        supcon_loss: bool = False, # use supervised contrastive loss or not
    ):
        super().__init__()
        
        # Calculate derived parameters
        self.time_length = time_length
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.stride = stride
        self.use_embedding = use_embedding
        self.num_time_patches = (time_length - patch_size) // stride + 1
        self.num_patches = self.num_time_patches * in_channels
        self.load_weights = load_weights
        self.use_motion_embed = use_motion_embed
        self.use_freq_embed = use_freq_embed
        self.max_event_count = max_event_count
        self.supcon_loss = supcon_loss

        # Combined patch embedding with positional encoding and CLS token
        self.patch_embed = PatchEmbedWithPos1D(
            max_seq_len=time_length,
            window_size=patch_size,
            stride=stride,
            in_channels=in_channels,
            embed_dim=embed_dim,
            padding_value=padding_value,
            padding_threshold=padding_threshold,
            use_embedding=use_embedding,
            pos_learnable=True,
            include_cls=True,  # Include CLS token
            dropout=dropout
        )
        
        # Get actual embedding dimension from patch embedding
        self.embed_dim = self.patch_embed.patch_embed.embed_dim
        
        if self.use_motion_embed:
            # Motion transition embedding for capturing state transitions
            self.motion_embed = MotionTransitionEmbedding(embed_dim=self.embed_dim, dropout=dropout)

        # Frequency embedding (for event frequency information)
        if self.use_freq_embed:
            self.freq_embed = EventFrequencyEmbedding(
                max_count=self.max_event_count,
                embed_dim=self.embed_dim,
                dropout=dropout
            )

        self.attention_type = attention_type

        # Encoder
        encoder_blocks = []
        for _ in range(encoder_layers):
            encoder_blocks.append(
                TransformerEncoderBlock(
                    embed_dim=self.embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attention_type=attention_type
                )
            )
        self.encoder = nn.ModuleList(encoder_blocks)
        
        # Decoder (for mask prediction or reconstruction)
        self.decoder = None
        if decoder_layers > 0:
            decoder_blocks = []
            for _ in range(decoder_layers):
                decoder_blocks.append(
                    TransformerDecoderBlock(
                        embed_dim=self.embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        attention_type=attention_type
                    )
                )
            self.decoder = nn.ModuleList(decoder_blocks)
        
        # if second_level_mask:
        #     mask_len = time_length // 20
        # else:
        #     mask_len = time_length

        if average_mask:
            mask_len = time_length // stride
        else:
            mask_len = time_length
        self.predictor = PredictorHead(self.embed_dim, mask_length=mask_len, num_classes=1, return_contrastive_embedding=supcon_loss)  # Default to regression
        # Initialize weights
        if self.load_weights:
            self._load_pretrained_weights()
        else:   
            self._init_weights()
    
    def _init_weights(self):
        # Initialize weights for all modules
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def _load_pretrained_weights(self):
        pretrained_weights = "/mnt/data/GENEActive-featurized/results/DL/geneactive_20hz_2s_b1s_imerit_sleep_nonwearOR_5smerge_nograv_change/ns_detect-lsm2-aim-pretrain/training/model_weights/vit1d_test_subject_FOLD4_weights.pth"
        


    def embedding(self, x: torch.Tensor,
                        motion_mask: Optional[torch.Tensor] = None,
                        frequency_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert input time series data into patches and embed them.
        
        Args:
            x: Input tensor of shape [B, C, L] where
               B = batch size, C = channels, L = time length
               
        Returns:
            Embedded patches with CLS token and positional encoding [B, 1+num_patches, embed_dim]
            Padding mask [B, 1+num_patches]
        """

        # Create patches, add positional embeddings and CLS token
        x, padding_mask = self.patch_embed(x)  # [B, 1+num_patches, embed_dim], [B, 1+num_patches]

        if self.use_motion_embed and motion_mask is not None:
            # Apply motion transition embedding if enabled
            x = self.motion_embed(x, motion_mask.to(x.device))

        if self.use_freq_embed and frequency_mask is not None:
            x = self.freq_embed(x, frequency_mask.to(x.device))
        
        return x, padding_mask

    def forward_embedding(
        self, 
        patch_embedding: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, list]:
        # forward using patch embedding
        x = patch_embedding
        assert x.dim() == 3, "Input must be of shape [B, num_patches, embed_dim]"

        # Apply encoder layers
        encoder_outputs = []
        
        for layer in self.encoder:
            x = layer(x, mask=attention_mask)
            encoder_outputs.append(x)
        
        # Return final output and intermediate features
        return x, encoder_outputs
    
    def forward_encoder(self, x, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, list]:
        # forward using input x
        assert x.dim() == 3, "Input must be of shape [B, L, C]"
        x = x.permute(0, 2, 1)  # [B, C, L]
        
        if attention_mask is None:
            x, attention_mask = self.embedding(x)  # [B, 1+num_patches, embed_dim]
        else:
            x, _ = self.embedding(x)

        # Apply encoder layers
        encoder_outputs = []
        
        for layer in self.encoder:
            x = layer(x, mask=attention_mask)
            encoder_outputs.append(x)
        
        # Return final output and intermediate features
        return x, encoder_outputs
    
    def forward(self, x, x_lengths=None, labels1=None, labels3=None,
                apply_mixup=False, mixup_alpha=0.2):
        # forward using input x
        assert x.dim() == 3, "Input must be of shape [B, L, C]"

        # PatchEmbedWithPos1D expects fixed time_length; pad batch-local inputs.
        if x.shape[1] < self.time_length:
            pad_len = self.time_length - x.shape[1]
            x = F.pad(x, (0, 0, 0, pad_len))
        elif x.shape[1] > self.time_length:
            x = x[:, : self.time_length, :]

        x = x.permute(0, 2, 1)  # [B, C, L]

        x, attention_mask = self.embedding(x)  # [B, 1+num_patches, embed_dim]

        # MultiheadAttention's key_padding_mask expects bool; float masks
        # trigger a perf-warning conversion on every call.
        if attention_mask is not None and attention_mask.dtype != torch.bool:
            attention_mask = attention_mask.bool()

        # Apply encoder layers
        x, _ = self.forward_embedding(x, attention_mask)

        # Predict using CLS token (first token)
        output1, output2, output3, con_embed, mixup_info = self.predictor(
            x, labels=labels1, apply_mixup=apply_mixup, mixup_alpha=mixup_alpha
        )

        # Mask-aware flatten of out2: loss expects concat of true-length values,
        # but out2 is fixed [B, mask_length]. Keep only positions j < x_lengths[i].
        if x_lengths is not None and output2 is not None and output2.dim() == 2:
            mask_len = output2.shape[1]
            valid = torch.arange(mask_len, device=output2.device).unsqueeze(0) < torch.tensor(
                x_lengths, device=output2.device, dtype=torch.long
            ).unsqueeze(1)
            output2 = output2.reshape(-1)[valid.reshape(-1)]

        attn = None
        return output1, output2, output3, attn, con_embed, mixup_info
        
    
    def forward_decoder(
        self,
        encoder_output: torch.Tensor,
        encoder_features: list,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Use encoder output as initial decoder input
        x = encoder_output
        
        # Apply decoder layers with cross-attention to encoder features
        if self.decoder is not None:
            for i, layer in enumerate(self.decoder):
                # Use different encoder features for each decoder layer
                encoder_idx = len(encoder_features) - 1 - min(i, len(encoder_features) - 1)
                memory = encoder_features[encoder_idx]
                
                x = layer(x, memory, self_mask=mask, memory_mask=mask)
        
        return x

if __name__ == "__main__":
    print("Test passed")