import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple, Dict


class PatchEmbed1D(nn.Module):
    """
    Convert 1D time series data into patches and embed them.
    Handles multiple channels with a shared kernel.
    """
    def __init__(
        self, 
        time_length: int = 1200,    # Total length in seconds
        patch_size: int = 200,      # Patch size in seconds
        stride: int = 20,           # Stride for patch extraction
        in_channels: int = 3,       # Number of input channels
        embed_dim: int = 384,       # Embedding dimension
        missing_value: float = -999.0,  # Value to identify missing data
        return_missing_mask: bool = False,  # Whether to return missing mask
        missing_patch_thres: float = 0.5, # Threshold to determine if a patch is missing
    ):
        super().__init__()
        self.time_length = time_length
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.stride = stride
        self.missing_value = missing_value
        self.return_missing_mask = return_missing_mask
        self.missing_patch_thres = missing_patch_thres
        
        # Calculate num_patches
        self.num_time_patches = (time_length - patch_size) // stride + 1
        self.num_patches = self.num_time_patches * in_channels
        
        # Linear projection for embedding patches
        # Each patch contains signal data for one patch_size across one channel
        self.proj = nn.Linear(patch_size, embed_dim)
        # self.proj = nn.Linear(in_channels * patch_size, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
    
    def get_padding_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Identify real missing values in input data"""
        B, C, L = x.shape
        device = x.device
        
        # Check for missing values (NaN or special value)
        is_missing = torch.isnan(x) if self.missing_value == float('nan') else (x == self.missing_value)
         
        # Create patches using unfold to match patch embeddings
        # is_missing_padded = F.pad(is_missing, (self.patch_size // 2 - self.stride, self.patch_size // 2))
        
        # Create patches [B, C, num_patches, patch_size]
        patches_missing = is_missing.unfold(2, self.patch_size, self.stride)
        patches_missing = patches_missing.contiguous().view(B, C, -1, self.patch_size)
        patches_missing = patches_missing.permute(0, 1, 2, 3).reshape(B, self.num_patches, self.patch_size)
        # A patch is considered missing if the ratio of missing values exceeds the threshold
        padding_mask = patches_missing.float().mean(dim=-1) > self.missing_patch_thres  # [B, num_patches*C]
        padding_mask = padding_mask.bool().to(device)
        
        return padding_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, C, L] where
               B = batch size, C = channels, L = time length
        Returns:
            Embedded patches of shape [B, C, num_patches, embed_dim]
        """
        B, C, L = x.shape
        assert C == self.in_channels, f"Expected {self.in_channels} channels, got {C}"
        assert L == self.time_length, f"Expected length {self.time_length}, got {L}"
        
        padding_mask = None
        if self.return_missing_mask:
            padding_mask = self.get_padding_mask(x)
        # replace missing values with zeros
        # if self.missing_value is not None:
        #     x = torch.where(x == self.missing_value, torch.zeros_like(x), x)
        
        # Generate patches along time dimension by sliding window with stride
        # [B, C, L] -> [B, C, num_time_patches, patch_size]. 
        # Add zeros padding to the sequence length [B, C, patch_size+L-1]
        # x = F.pad(x, (self.patch_size // 2 - self.stride, self.patch_size // 2))
        x = x.unfold(2, self.patch_size, self.stride)
        x = x.contiguous().view(B, C, -1, self.patch_size)

        # Process each channel independently with the same kernel
        patches = []
        for c in range(C):
            # Extract this channel and embed each patch
            channel_patches = x[:, c]  # [B, num_time_patches, patch_size]
            channel_patches = self.proj(channel_patches)  # [B, num_time_patches, embed_dim]
            channel_patches = self.norm(channel_patches)
            patches.append(channel_patches)

        # [B, C, num_time_patches, embed_dim] -> [B, num_patches, embed_dim]
        x = torch.cat(patches, dim=1)  # Stack along channel dimension

        # Permute and reshape to [B, num_time_patches, C*patch_size]
        # x = x.permute(0, 2, 1, 3).reshape(B, self.num_time_patches, -1)
        # x = self.proj(x)  # [B, num_time_patches, embed_dim]
        # x = self.norm(x)  # Normalize the embeddings
        return x, padding_mask
     

class PositionalEmbedding2D(nn.Module):
    """
    2D positional embedding for encoding temporal position and signal channel.
    """
    def __init__(
        self,
        num_time_patches: int,  # Number of patches along time dimension
        in_channels: int,       # Number of signal channels
        embed_dim: int,         # Embedding dimension
        dropout: float = 0.1,   # Dropout rate
    ):
        super().__init__()
        self.num_time_patches = num_time_patches
        self.in_channels = in_channels
        
        # Total number of position embeddings (num_time_patches * in_channels)
        num_positions = num_time_patches * in_channels
        
        # Create learnable position embeddings
        # We use separate embedding vectors for temporal positions and channels
        self.time_embed = nn.Parameter(torch.zeros(num_time_patches, embed_dim // 2))
        self.channel_embed = nn.Parameter(torch.zeros(in_channels, embed_dim // 2))
        
        # Learnable embedding for CLS token
        self.cls_token_embed = nn.Parameter(torch.zeros(1, embed_dim))
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize embeddings with sinusoidal patterns
        self._init_weights()
    
    def _init_weights(self):
        # Initialize temporal embeddings with sinusoidal pattern
        pos = torch.arange(self.num_time_patches).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.time_embed.shape[1], 2) * 
                           -(math.log(10000.0) / self.time_embed.shape[1]))
        
        # Create a temporary tensor with the right values
        time_embed_init = torch.zeros_like(self.time_embed.data)
        time_embed_init[:, 0::2] = torch.sin(pos * div_term)
        time_embed_init[:, 1::2] = torch.cos(pos * div_term)
        
        self.time_embed.data.copy_(time_embed_init)
        # Initialize channel embeddings with small random values
        nn.init.normal_(self.channel_embed, std=0.02)
        nn.init.normal_(self.cls_token_embed, std=0.02)
    
    def forward(self, x: torch.Tensor, include_cls: bool = True) -> torch.Tensor:
        """
        Args:
            x: Embedded patches [B, num_patches, embed_dim]
            include_cls: Whether to include position embedding for CLS token
        Returns:
            x with positional embeddings added
        """
        batch_size, seq_len, _ = x.shape
        
        # Get device
        device = x.device
        
        # # Build 2D positional embeddings
        # pos_embeddings = torch.zeros_like(x)
        # for t in range(self.num_time_patches):
        #     for c in range(self.in_channels):
        #         # Compute position in the sequence
        #         pos = t + c * self.num_time_patches
        #         if pos < seq_len:  # Check if position is valid
        #             # Combine temporal and channel embeddings
        #             time_emb = self.time_embed[t]
        #             channel_emb = self.channel_embed[c]
        #             pos_embedding = torch.cat([time_emb, channel_emb], dim=0)
        #             pos_embeddings[:, pos] = pos_embedding
        content_start = 1 if include_cls else 0
        content_len = seq_len - content_start
        positions = torch.arange(min(content_len, self.num_time_patches * self.in_channels), device=device)
        t_indices = positions % self.num_time_patches
        c_indices = positions // self.num_time_patches
        # gather embeddings
        t_embeds = self.time_embed[t_indices]
        c_embeds = self.channel_embed[c_indices]

        # combine embeddings and reshape
        pos_embeddings = torch.zeros_like(x)
        pos_embeddings[:, content_start:content_start+content_len] = torch.cat([t_embeds, c_embeds], dim=-1).unsqueeze(0)

        # Apply position embeddings
        # if include_cls and seq_len > self.num_time_patches * self.in_channels:
        if include_cls:
             # First token is CLS token
            pos_embeddings[:, 0] = self.cls_token_embed.squeeze(0)
        
        # Add positional embeddings and apply dropout
        x = x + pos_embeddings.to(device)
        return self.dropout(x)

class MotionTransitionEmbedding(nn.Module):
    """
    Captures motion transition patterns between different states in time series data
    """
    def __init__(self, embed_dim, dropout=0.1, ignore_padding=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.ignore_padding = ignore_padding

        # Embedding for motion states (0 = padding, 1 = no transition, 2 = transition)
        if ignore_padding:
            self.motion_embed = nn.Embedding(2, embed_dim)  # Only 1 and 2
        else:
            self.motion_embed = nn.Embedding(3, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, motion_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Patch embeddings [B, num_patches, embed_dim]
            motion_mask: Binary mask indicating transitions [B, num_patches]
                         (0 = padding, 1 = no transition, 2 = transition)
        Returns:
            Fused embeddings [B, num_patches, embed_dim]
        """
        # Ensure motion_mask has the right shape
        if motion_mask.dim() == 1:
            motion_mask = motion_mask.unsqueeze(0).expand(x.size(0), -1)
        motion_mask = motion_mask.long()
        
        # Get motion embeddings
        motion_embeddings = self.motion_embed(motion_mask)

        x = x + motion_embeddings
        return self.dropout(x)
    

class EventFrequencyEmbedding(nn.Module):
    """
    Embeds event frequencies within time windows and fuses them with patch embeddings
    """
    def __init__(
        self, 
        embed_dim: int,
        max_count: int = 10,  # Maximum event count to have unique embeddings for
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_count = max_count

        # Embedding for event frequencies (0 to max_count)
        # +1 for frequencies exceeding max_count
        self.event_embed = nn.Embedding(max_count + 1, embed_dim)

        # Normalization factor for frequencies exceeding max_count
        self.count_scale = nn.Parameter(torch.tensor(0.1))
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize
        self._init_weights()
        
    def _init_weights(self):
        # Initialize with small random values
        nn.init.normal_(self.event_embed.weight, std=0.02)
        
    def forward(self, x: torch.Tensor, x_freq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Patch embeddings [B, num_patches, embed_dim]
            x_freq: Global event frequency for each sequence [B]
                     Values are non-negative integers
        Returns:
            Fused embeddings [B, num_patches, embed_dim]
        """
        # Ensure x_freq has the right shape
        batch_size, num_patches, _ = x.shape
        if x_freq.dim() > 1:
            x_freq = x_freq.squeeze()

        # Cap the frequencies at max_count
        capped_freq = torch.clamp(x_freq, max=self.max_count)

        # Get event frequency embeddings
        freq_embeddings = self.event_embed(capped_freq.long())

        # Scale embeddings for frequencies exceeding max_count
        scale_factor = 1.0 + self.count_scale * torch.relu(x_freq.float() - self.max_count).unsqueeze(-1)
        freq_embeddings = freq_embeddings * scale_factor

        # Expand to all patches [B, num_patches, embed_dim]
        freq_embeddings = freq_embeddings.unsqueeze(1).expand(-1, num_patches, -1)

        # Add to input embeddings
        x = x + freq_embeddings
        
        return self.dropout(x)



class PatchEmbedViT1D(nn.Module):
    """
    ViT-style patch embedding for 1D time series data.
    Converts input (B, C, seq_len) to patches (B, num_patches, window_size * C).
    Handles padding values and generates padding masks for patches.
    """
    
    def __init__(
        self,
        window_size: int = 200,           # Size of each patch/window
        stride: int = 20,                 # Stride for sliding window
        in_channels: int = 3,             # Number of input channels
        embed_dim: Optional[int] = None,  # Embedding dimension (defaults to patch_dim)
        padding_value: float = -999.0,    # Value used for padding
        padding_threshold: float = 0.5,   # Threshold for patch-level padding detection
        use_embedding: bool = True,      # Whether to apply linear embedding
    ):
        super().__init__()
        self.window_size = window_size
        self.stride = stride
        self.in_channels = in_channels
        self.padding_value = padding_value
        self.padding_threshold = padding_threshold
        self.use_embedding = use_embedding
        
        # Input dimension for each patch (flattened multi-channel window)
        self.patch_dim = window_size * in_channels
        
        # Set embedding dimension
        if embed_dim is None:
            self.embed_dim = self.patch_dim
        else:
            self.embed_dim = embed_dim
        
        # Linear projection to embed patches
        if use_embedding:
            self.proj = nn.Linear(self.patch_dim, self.embed_dim)
            self.norm = nn.LayerNorm(self.embed_dim)
    
    def calculate_num_patches(self, seq_len: int) -> int:
        """Calculate number of patches for given sequence length."""
        return max(0, (seq_len - self.window_size) // self.stride + 1)
    
    def create_patches(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract patches from input using sliding window.
        
        Args:
            x: Input tensor [B, C, L]
            
        Returns:
            patches: Tensor [B, num_patches, window_size * C]
        """
        B, C, L = x.shape
        
        # Calculate number of patches
        num_patches = self.calculate_num_patches(L)
        
        if num_patches == 0:
            # Return empty tensor with correct shape
            return torch.zeros(B, 0, self.patch_dim, device=x.device, dtype=x.dtype)
        
        # Use unfold to create sliding windows
        # x.unfold(dim, size, step) creates windows along dimension 'dim'
        patches = x.unfold(2, self.window_size, self.stride)  # [B, C, num_patches, window_size]
        
        # Reshape to [B, num_patches, C, window_size] then flatten channels and window
        patches = patches.permute(0, 2, 1, 3)  # [B, num_patches, C, window_size]
        patches = patches.contiguous().view(B, num_patches, -1)  # [B, num_patches, C * window_size]
        
        return patches
    
    def create_padding_mask(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Create padding mask for patches based on padding threshold.
        
        Args:
            patches: Patch tensor [B, num_patches, window_size * C]
            
        Returns:
            padding_mask: Boolean tensor [B, num_patches] where True indicates padding
        """
        # Check for padding values
        if torch.isnan(torch.tensor(self.padding_value)):
            is_padding = torch.isnan(patches)
        else:
            is_padding = (patches == self.padding_value)
        
        # Calculate ratio of padding values per patch
        padding_ratio = is_padding.float().mean(dim=-1)  # [B, num_patches]
        
        # A patch is considered padding if ratio exceeds threshold
        padding_mask = padding_ratio > self.padding_threshold
        
        return padding_mask
    
    def replace_extreme_padding(self, patches: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Replace extreme padding values with zeros to prevent issues with normalization.
        Only replaces values in patches marked as padding.
        
        Args:
            patches: Patch tensor [B, num_patches, window_size * C]
            padding_mask: Boolean tensor [B, num_patches]
            
        Returns:
            patches_cleaned: Patches with extreme values replaced
        """
        patches_cleaned = patches.clone()
        
        # Only replace extreme values (absolute value > 100) in padded patches
        if abs(self.padding_value) > 100:
            # Create mask for extreme values in padded patches
            extreme_mask = (torch.abs(patches) > 100) & padding_mask.unsqueeze(-1)
            patches_cleaned = torch.where(extreme_mask, 0.0, patches_cleaned)
            
        return patches_cleaned
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of patch embedding.
        
        Args:
            x: Input tensor [B, C, L]
            
        Returns:
            embeddings: Patch embeddings [B, num_patches, embed_dim]
            padding_mask: Padding mask [B, num_patches]
        """
        # Create patches
        patches = self.create_patches(x)  # [B, num_patches, window_size * C]
        
        # Create padding mask
        padding_mask = self.create_padding_mask(patches)  # [B, num_patches]
        
        # Replace extreme padding values for stable training
        patches_clean = self.replace_extreme_padding(patches, padding_mask)
        
        # Apply linear projection if enabled
        if self.use_embedding:
            embeddings = self.proj(patches_clean)  # [B, num_patches, embed_dim]
            embeddings = self.norm(embeddings)
        else:
            embeddings = patches_clean
            
        return embeddings, padding_mask
    
    def get_patch_info(self, seq_len: int) -> Dict[str, int]:
        """Get information about patch dimensions for given sequence length."""
        num_patches = self.calculate_num_patches(seq_len)
        return {
            'num_patches': num_patches,
            'patch_dim': self.patch_dim,
            'embed_dim': self.embed_dim if self.use_embedding else self.patch_dim,
            'window_size': self.window_size,
            'stride': self.stride
        }


class PatchPositionalEmbedding1D(nn.Module):
    """
    Positional embedding for 1D patch sequences with variable lengths.
    Supports batches with different sequence lengths up to max_seq_len.
    """
    
    def __init__(
        self,
        max_seq_len: int,                 # Maximum sequence length
        window_size: int = 200,           # Patch window size
        stride: int = 20,                 # Patch stride
        embed_dim: int = 600,             # Embedding dimension
        dropout: float = 0.1,             # Dropout rate
        learnable: bool = True,           # Whether to use learnable or sinusoidal embeddings
        include_cls: bool = False,        # Whether to include CLS token position
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.window_size = window_size
        self.stride = stride
        self.embed_dim = embed_dim
        self.learnable = learnable
        self.include_cls = include_cls
        
        # Calculate maximum number of patches
        self.max_num_patches = max(0, (max_seq_len - window_size) // stride + 1)
        
        # Add 1 for CLS token if needed
        total_positions = self.max_num_patches + (1 if include_cls else 0)
        
        if learnable:
            # Learnable positional embeddings
            self.pos_embed = nn.Parameter(torch.zeros(1, total_positions, embed_dim))
            self._init_learnable_weights()
        else:
            # Fixed sinusoidal positional embeddings
            self.register_buffer('pos_embed', self._create_sinusoidal_embeddings(total_positions, embed_dim))
            
        # CLS token embedding if needed
        if include_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.normal_(self.cls_token, std=0.02)
            
        self.dropout = nn.Dropout(dropout)
    
    def _init_learnable_weights(self):
        """Initialize learnable positional embeddings."""
        nn.init.normal_(self.pos_embed, std=0.02)
    
    def _create_sinusoidal_embeddings(self, num_positions: int, embed_dim: int) -> torch.Tensor:
        """Create fixed sinusoidal positional embeddings."""
        pos_embed = torch.zeros(1, num_positions, embed_dim)
        position = torch.arange(0, num_positions).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                           -(math.log(10000.0) / embed_dim))
        
        pos_embed[0, :, 0::2] = torch.sin(position * div_term)
        pos_embed[0, :, 1::2] = torch.cos(position * div_term)
        
        return pos_embed
    
    def calculate_num_patches(self, seq_len: int) -> int:
        """Calculate number of patches for given sequence length."""
        return max(0, (seq_len - self.window_size) // self.stride + 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional embeddings to patch embeddings.
        
        Args:
            x: Patch embeddings [B, num_patches, embed_dim]
            
        Returns:
            x_pos: Embeddings with positional encoding [B, num_patches_with_cls, embed_dim]
        """
        B, num_patches, embed_dim = x.shape
        assert embed_dim == self.embed_dim, f"Expected embed_dim {self.embed_dim}, got {embed_dim}"
        
        # Add CLS token if needed
        if self.include_cls:
            cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
            x = torch.cat([cls_tokens, x], dim=1)  # [B, num_patches + 1, embed_dim]
            num_patches += 1
        
        # Get positional embeddings for the actual sequence length
        pos_embeddings = self.pos_embed[:, :num_patches, :]  # [1, num_patches, embed_dim]
        
        # Apply positional embeddings
        x = x + pos_embeddings.expand(B, -1, -1)
        
        return self.dropout(x)
    
    def get_embedding_info(self, seq_len: int) -> Dict[str, int]:
        """Get information about embeddings for given sequence length."""
        num_patches = self.calculate_num_patches(seq_len)
        total_tokens = num_patches + (1 if self.include_cls else 0)
        
        return {
            'num_patches': num_patches,
            'total_tokens': total_tokens,
            'cls_included': self.include_cls,
            'max_num_patches': self.max_num_patches,
            'embed_dim': self.embed_dim
        }


class PatchEmbedWithPos1D(nn.Module):
    """
    Combined patch embedding and positional encoding for 1D time series.
    Handles variable sequence lengths and provides complete patch preprocessing.
    """
    
    def __init__(
        self,
        max_seq_len: int,                 # Maximum sequence length
        window_size: int = 200,           # Patch window size
        stride: int = 20,                 # Patch stride
        in_channels: int = 3,             # Number of input channels
        embed_dim: Optional[int] = None,  # Embedding dimension (defaults to patch_dim)
        padding_value: float = -999.0,    # Value used for padding
        padding_threshold: float = 0.5,   # Threshold for patch-level padding detection
        use_embedding: bool = False,      # Whether to apply linear embedding
        pos_learnable: bool = True,       # Whether positional embeddings are learnable
        include_cls: bool = False,        # Whether to include CLS token
        dropout: float = 0.1,             # Dropout rate
    ):
        super().__init__()
        
        # Patch embedding layer
        self.patch_embed = PatchEmbedViT1D(
            window_size=window_size,
            stride=stride,
            in_channels=in_channels,
            embed_dim=embed_dim,
            padding_value=padding_value,
            padding_threshold=padding_threshold,
            use_embedding=use_embedding
        )
        
        # Get actual embedding dimension
        actual_embed_dim = self.patch_embed.embed_dim
        
        # Positional embedding layer
        self.pos_embed = PatchPositionalEmbedding1D(
            max_seq_len=max_seq_len,
            window_size=window_size,
            stride=stride,
            embed_dim=actual_embed_dim,
            dropout=dropout,
            learnable=pos_learnable,
            include_cls=include_cls
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass combining patch embedding and positional encoding.
        
        Args:
            x: Input tensor [B, C, L] where L can vary across batch
            
        Returns:
            embeddings: Patch embeddings with positional encoding [B, num_patches_with_cls, embed_dim]
            padding_mask: Padding mask [B, num_patches_with_cls]
        """
        # Get patch embeddings and padding mask
        patch_embeddings, padding_mask = self.patch_embed(x)  # [B, num_patches, embed_dim], [B, num_patches]
        
        # Add positional embeddings
        embeddings = self.pos_embed(patch_embeddings)  # [B, num_patches_with_cls, embed_dim]
        
        # Extend padding mask for CLS token if included
        if self.pos_embed.include_cls:
            B = padding_mask.shape[0]
            cls_mask = torch.zeros(B, 1, dtype=padding_mask.dtype, device=padding_mask.device)
            padding_mask = torch.cat([cls_mask, padding_mask], dim=1)  # [B, num_patches + 1]
        
        return embeddings, padding_mask
    
    def get_info(self, seq_len: int) -> Dict[str, int]:
        """Get comprehensive information about patch and positional embeddings."""
        patch_info = self.patch_embed.get_patch_info(seq_len)
        pos_info = self.pos_embed.get_embedding_info(seq_len)
        
        return {**patch_info, **pos_info}
      