import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import Optional, Tuple, Dict, List
try:
    from .vit1d import ViT1D
except:
    from vit1d import ViT1D


class MAEPretrainer(nn.Module):
    """
    Masked Autoencoder for ViT-1D pretraining.
    """
    def __init__(
        self,
        vit_model: ViT1D,
        mask_ratio: float = 0.75,
        norm_pixel_loss: bool = True,
    ):
        super().__init__()
        self.vit = vit_model
        self.mask_ratio = mask_ratio
        self.norm_pixel_loss = norm_pixel_loss
        self.mask_token = nn.Parameter(torch.zeros(1, 1, vit_model.embed_dim))  # Mask token for decoder
        # Create decoder-specific components if needed
        if vit_model.decoder is None:
            # If the ViT doesn't have a built-in decoder, create one
            self.decoder = nn.ModuleList([
                TransformerDecoderBlock(
                    embed_dim=vit_model.embed_dim,
                    num_heads=vit_model.num_heads if hasattr(vit_model, 'num_heads') else 6,
                    mlp_ratio=4.0,
                    dropout=0.1
                ) for _ in range(4)  # 4 decoder layers
            ])
        else:
            self.decoder = vit_model.decoder
        self.reconstruction_head = nn.Linear(self.vit.embed_dim, self.vit.patch_size)
    
    def random_masking(self, x, mask_ratio):
        """
        Perform random masking by replacing mask_ratio tokens with a mask token.
        Args:
            x: Input sequence [B, N, D]
            mask_ratio: Ratio of tokens to mask
        """
        B, N, D = x.shape  # batch, sequence length, dimension
        
        # Skip CLS token for masking
        x_no_cls = x[:, 1:, :]
        B, N_no_cls, D = x_no_cls.shape
        
        # Calculate number of tokens to keep
        len_keep = int(N_no_cls * (1 - mask_ratio))
        
        # Generate random permutation indices
        noise = torch.rand(B, N_no_cls, device=x.device)  # noise in [0, 1]
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)  # restore original order
        
        # Keep the first len_keep indices
        ids_keep = ids_shuffle[:, :len_keep]
        
        # Gather the kept tokens
        x_kept = torch.gather(x_no_cls, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # Generate mask: 0 is keep, 1 is remove
        mask = torch.ones([B, N_no_cls], device=x.device)
        mask[:, :len_keep] = 0
        # Restore order of the mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        # Prepend CLS token
        cls_tokens = x[:, :1, :]
        x_masked = torch.cat([cls_tokens, x_kept], dim=1)
        
        # For decoder: Full sequence with masked token indicator
        mask_full = torch.cat([torch.zeros([B, 1], device=x.device), mask], dim=1)
        
        return x_masked, mask_full, ids_restore
    
    def forward_encoder(self, x, mask_ratio):
        """
        Encode input with masking
        """
        # replace missing values with zeros
        if self.vit.patch_embed.missing_value is not None:
            x = torch.where(x == self.vit.patch_embed.missing_value, torch.zeros_like(x), x)
        # Create patch embeddings
        x, _ = self.vit.patch_embed(x)
        
        # Add CLS token
        cls_tokens = self.vit.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embeddings
        x = self.vit.pos_embed(x)
        
        # Random masking
        x_masked, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        # Apply encoder
        latent, encoder_features = self.vit.forward_embedding(x_masked)
        
        return latent, mask, ids_restore, encoder_features
    
    def forward_decoder(self, latent, ids_restore, encoder_features):
        """
        Decode from latent representation
        """
        B, _, C = latent.shape
    
        # Add mask tokens for masked positions
        # First, expand mask tokens to batch size
        mask_tokens = self.mask_token.expand(B, ids_restore.shape[1] - latent.shape[1] + 1, -1)
        
        # Concatenate latent vectors (without CLS) with mask tokens
        x_ = torch.cat([latent[:, 1:], mask_tokens], dim=1)
        
        # Reorder tokens to original sequence using ids_restore
        x_ = torch.gather(x_, dim=1, 
                        index=ids_restore.unsqueeze(-1).repeat(1, 1, latent.shape[-1]))
        
        # Add back the CLS token
        x = torch.cat([latent[:, :1], x_], dim=1)
        # Decoder forward pass
        x = self.vit.forward_decoder(x, encoder_features, None)
        
        # Predict original values
        pred = self.reconstruction_head(x[:, 1:])  # Skip CLS token
        
        return pred
    
    def forward(self, x, mask_ratio=None):
        """
        Forward function for MAE pretraining
        """
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        
        # Get device
        device = x.device
        
        # Encoder: randomly mask patches and encode visible ones
        latent, mask, ids_restore, encoder_features = self.forward_encoder(x, mask_ratio)
        
        # Decoder: predict pixel values for reconstruction
        pred = self.forward_decoder(latent, ids_restore, encoder_features)
        # Calculate loss on masked tokens only
        loss = self.calculate_loss(x, pred, mask)
        
        return loss, pred, mask
    
    def calculate_loss(self, x, pred, mask):
        """
        Compute reconstruction loss
        """
        # Get original values for masked patches
        B, C, L = x.shape

        # Reshape x to match predicted patch values
        if self.vit.patch_embed.missing_value is not None:
            x = torch.where(x == self.vit.patch_embed.missing_value, torch.zeros_like(x), x)
        
        # Generate patches along time dimension by sliding window with stride
        # [B, C, L] -> [B, C, num_time_patches, patch_size]. 
        # Add zeros padding to the sequence length [B, C, patch_size+L-1]
        # x = F.pad(x, (self.vit.patch_size // 2, self.vit.patch_size // 2))
        x = x.unfold(2, self.vit.patch_size, self.vit.stride)
        x = x.contiguous().view(B, C, -1, self.vit.patch_size)
        num_patches = x.size(2)
        x = x.permute(0, 1, 2, 3).reshape(B, C * num_patches, self.vit.patch_size)
        
        print(x.shape, pred.shape, mask.shape)
        # Apply mask: only compute loss on masked patches
        mask = mask[:, 1:]  # Remove CLS token
        
        # Normalize target if needed
        if self.norm_pixel_loss:
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True)
            x = (x - mean) / (var + 1.e-6)**.5
        
        # Calculate MSE loss
        loss = (pred - x) ** 2
        loss = loss.mean(dim=-1)  # [B, L], mean loss per patch
        
        # Apply mask and compute final loss
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        
        return loss
    
    
  
class AIMPretrainer(nn.Module):
    """
    Adaptive Attentive Masking(AIM) pretraining for ViT1D.
    Combines inherited (real) and artificial masks with dropout removal for efficiency.
    """
    def __init__(
        self,
        vit_model: nn.Module,
        mask_ratio: float = 0.5,         # Ratio for artificial masking
        dropout_rate: float = 0.2,      # Ratio of tokens to fully drop for efficiency
        norm_target: bool = True,
        missing_value: float = -999.0,
        missing_patch_thres: float = 0.5,      # Threshold to determine if a patch is missing
    ):
        super().__init__()
        self.vit = vit_model
        self.mask_ratio = mask_ratio
        # TODO: check if dropout_rate = artificial_mask_ratio
        self.dropout_rate = dropout_rate 
        self.norm_target = norm_target
        self.missing_value = missing_value
        self.missing_patch_thres = missing_patch_thres
        
        # Learnable mask token (shared between inherited and artificial masks)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.vit.embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        
        # Reconstruction head
        self.decoder_head = nn.Linear(self.vit.embed_dim, self.vit.patch_size)
    
    def generate_inherited_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Identify real missing values in input data"""
        B, C, L = x.shape
        device = x.device
        
        # Check for missing values (NaN or special value)
        is_missing = torch.isnan(x) if self.missing_value == float('nan') else (x == self.missing_value)
         
        # Create patches using unfold to match patch embeddings
        # pad_size = self.vit.num_time_patches * self.vit.stride + self.vit.patch_size - L
        # pad_size = self.vit.patch_size 
        # is_missing_padded = F.pad(is_missing, (pad_size//2, pad_size//2))
        
        # Create patches [B, C, num_patches, patch_size]
        patches_missing = is_missing.unfold(2, self.vit.patch_size, self.vit.stride)
        patches_missing = patches_missing.contiguous().view(B, C, -1, self.vit.patch_size)
        num_patches = patches_missing.size(2)
        patches_missing = patches_missing.permute(0, 1, 2, 3).reshape(B, C * num_patches, self.vit.patch_size)
        # A patch is considered missing if the ratio of missing values exceeds the threshold
        inherited_mask = patches_missing.float().mean(dim=-1) >= self.missing_patch_thres  # [B, num_patches*C]
        inherited_mask = inherited_mask.bool().to(device)
        
        return inherited_mask
    

    def generate_artificial_mask(self, inherited_mask: torch.Tensor) -> torch.Tensor:
        """
        Create random artificial masks based on target mask ratio, accounting for inherited masks.
        Only apply to non-inherited-masked positions.
        """
        B, num_patches = inherited_mask.shape
        device = inherited_mask.device
        
        # Calculate how many tokens to artificially mask
        artificial_mask = torch.zeros_like(inherited_mask)
        
        artificial_token_count = []

        for b in range(B):
            # Count inherited masks for this batch item
            inherited_count = inherited_mask[b].sum().item()
            inherited_ratio = inherited_count / num_patches
            
            # Determine artificial mask ratio (target - inherited, but not less than 0)
            artificial_ratio = max(0, self.mask_ratio - inherited_ratio)
            
            # Apply artificial masking only on valid (non-inherited masked) tokens
            valid_tokens = ~inherited_mask[b]
            valid_count = valid_tokens.sum().item()
            
            # Number of tokens to artificially mask
            num_to_mask = int(artificial_ratio * num_patches)
            num_to_mask = min(num_to_mask, valid_count)  # Can't mask more than available
            
            artificial_token_count.append(num_to_mask)
            if num_to_mask > 0:
                # Get indices of valid tokens
                valid_indices = torch.nonzero(valid_tokens).squeeze(-1)
                
                # Randomly select tokens to mask
                perm = torch.randperm(len(valid_indices), device=device)
                mask_indices = valid_indices[perm[:num_to_mask]]
                
                # Set selected indices to be masked
                artificial_mask[b, mask_indices] = True
        
        return artificial_mask, artificial_token_count
    
    def select_dropout_tokens(self, inherited_mask: torch.Tensor, artificial_mask: torch.Tensor, dropout_count: list=[], dropout_rate: float=0.25) -> torch.Tensor:
        """Determine which tokens to drop entirely during encoding for efficiency"""
        B, N = inherited_mask.shape
        device = inherited_mask.device
    
        # Combined mask (both inherited and artificial)
        combined_mask = (inherited_mask | artificial_mask)
        if not dropout_count:
            # If no specific dropout counts provided, calculate based on dropout_rate
            dropout_count = (combined_mask.sum(dim=1) * dropout_rate).int().tolist()
        
        dropout_mask = torch.zeros_like(combined_mask)
        
        for b in range(B):
            num_to_drop = dropout_count[b]
            # dropping already masked tokens
            masked_indices = torch.nonzero(combined_mask[b]).squeeze(-1)
            
            if num_to_drop > len(masked_indices):
                num_to_drop = len(masked_indices)
            
            perm = torch.randperm(len(masked_indices), device=device)
            selected_indices = masked_indices[perm[:num_to_drop]]
        
            dropout_mask[b, selected_indices] = 1
            
        return combined_mask, dropout_mask.bool()

    def process_encoder_input(self, patch_embeddings: torch.Tensor, 
                             combined_mask: torch.Tensor, 
                             dropout_mask: torch.Tensor,
                             ):
        """Process input for encoder: apply masking and create attention masks"""
        B, C, L = patch_embeddings.shape
        device = patch_embeddings.device
        
        # 3. Determine which tokens to keep 
        keep_mask = ~dropout_mask  # [B, num_patches]
        
        # 5. Create input tensor with only kept tokens
        num_kept = keep_mask.sum(dim=1)  # [B]
        max_kept = num_kept.max().item()
        
        x_kept = torch.zeros(B, max_kept, self.vit.embed_dim, device=device)
        attention_mask = torch.ones(B, max_kept, device=device).bool()  # 1 = position to ignore for transformer encoder
        
        # Store mapping info for each batch item to track positions
        position_mapping = []
        
        for b in range(B):
            # Get indices of tokens to keep
            keep_indices = torch.nonzero(keep_mask[b]).squeeze(-1)
            kept_count = len(keep_indices)
            
            # Copy kept tokens
            x_kept[b, :kept_count] = patch_embeddings[b, keep_indices]
            
            # attention masking for ignored tokens
            attention_mask[b, :kept_count] = False
            
            # Replace masked positions (both inherited and artificial) with mask token
            for i, idx in enumerate(keep_indices):
                if combined_mask[b, idx]:
                    x_kept[b, i] = self.mask_token
            
            # Store mapping from kept positions to original positions
            mapping = {}
            for i, orig_idx in enumerate(keep_indices):
                mapping[i] = orig_idx.item()
            position_mapping.append(mapping)
        
        # check the kept tokens and update attention mask if it's masked in the original input
        for b in range(B):
            for i in range(num_kept[b]):
                orig_idx = position_mapping[b][i]
                if combined_mask[b, orig_idx]:
                    attention_mask[b, i] = True

        return x_kept, attention_mask, position_mapping

    def reinsert_dropout_tokens(self, encoded_features: torch.Tensor, 
                                      patch_embeddings: torch.Tensor,
                                      position_mapping: list, 
                                      dropout_mask: torch.Tensor, 
                                      combined_mask: torch.Tensor):
        """
        Reinsert dropped tokens in original order before decoding.
        """
        B, _, D = encoded_features.shape
        device = encoded_features.device
        
        # Create full-sized output with all positions
        num_patches = dropout_mask.shape[1]
        full_features = torch.zeros(B, num_patches, D, device=device)

        
        for b in range(B):
            # Fill in the encoded features for kept positions
            for kept_idx, orig_idx in position_mapping[b].items():
                if kept_idx < encoded_features.size(1):  # Safety check
                    full_features[b, orig_idx] = encoded_features[b, kept_idx]
            
            # Fill in dropout positions with original patch embeddings
            for i in range(num_patches):
                if dropout_mask[b, i]:
                    full_features[b, i] = patch_embeddings[b, i]
        
        return full_features
    
    def compute_reconstruction_loss(self, original: torch.Tensor, 
                                          reconstructed: torch.Tensor, 
                                          artificial_mask: torch.Tensor):
        """Calculate reconstruction loss only on artificially masked tokens"""
        # Process original patches
        B, C, L = original.shape
        device = original.device
        
        # Create original patches
        # pad_size = self.vit.patch_size
        # if pad_size > 0:
        #     original_padded = F.pad(original, (pad_size//2, pad_size//2))
        # else:
        #     original_padded = original
            
        # Extract patches [B, C, num_patches, patch_size]
        original_patches = original.unfold(2, self.vit.patch_size, self.vit.stride)
        
        # Reshape to match reconstructed output [B, C*num_patches, patch_size]
        num_patches = original_patches.size(2)
        original_flat = original_patches.permute(0, 1, 2, 3).reshape(B, C * num_patches, self.vit.patch_size)
        
        if self.norm_target:
            mean = original_flat.mean(dim=-1, keepdim=True)
            var = original_flat.var(dim=-1, keepdim=True)
            original_flat = (original_flat - mean) / (var + 1e-6).sqrt()
        
        # Only compute loss on artificially masked tokens 
        # Exclude inherited masked tokens (no ground truth available)
        total_valid = artificial_mask.sum()
        if total_valid > 0:
            # Flatten tensors for element-wise comparison
            flat_original = original_flat.reshape(-1, original_flat.size(-1))
            flat_reconstructed = reconstructed.reshape(-1, reconstructed.size(-1))
            flat_valid_mask = artificial_mask.reshape(-1)
            
            # Select valid positions
            valid_original = flat_original[flat_valid_mask]
            valid_reconstructed = flat_reconstructed[flat_valid_mask]
            
            # Compute MSE
            loss = F.mse_loss(valid_reconstructed, valid_original)
        else:
            # No valid positions to compute loss on
            loss = torch.tensor(0.0, device=device)
        
        return loss
    
    def forward(self, x: torch.Tensor,
                      motion_mask: torch.Tensor = None,
                      frequency_mask: torch.Tensor = None,
                      ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for AIM pretraining.
        
        Args:
            x: Input tensor [B, C, L]
            motion_mask: Tensor indicating motion information [B, num_patches]
            frequency_mask: Tensor indicating frequency information [B]
            
        Returns:
            Dictionary with loss and reconstructed output
        """
        B, C, L = x.shape
        
        # inherited masks from missing data
        inherited_mask = self.generate_inherited_mask(x)
        
        # artificial masks on observed data
        artificial_mask, artificial_token_count = self.generate_artificial_mask(inherited_mask)
        
        # update inherited mask and artifical mask to include CLS token
        inherited_mask = torch.cat([torch.zeros(B, 1, device=x.device, dtype=torch.bool), inherited_mask], dim=1)
        artificial_mask = torch.cat([torch.zeros(B, 1, device=x.device, dtype=torch.bool), artificial_mask], dim=1)

        # select tokens to drop entirely for efficiency
        combined_mask, dropout_mask = self.select_dropout_tokens(inherited_mask, artificial_mask, artificial_token_count) 
        if frequency_mask is not None:
            frequency_mask = torch.cat([torch.zeros(B, 1, device=x.device, dtype=torch.bool), frequency_mask], dim=1)

        patch_embeddings, _ = self.vit.embedding(x, motion_mask=motion_mask,
                                                 frequency_mask=frequency_mask)  # [B, num_patches, embed_dim]
        # Step 4: Process input for encoder (apply masking, prepare attention mask)
        x_kept, attention_mask, position_mapping = self.process_encoder_input(
            patch_embeddings, combined_mask, dropout_mask
        )
        
        # Step 5: Run encoder with attention masking
        encoded_output, encoder_features = self.vit.forward_embedding(x_kept, attention_mask=attention_mask)
        
        # Step 6: CRITICAL - Reinsert dropped tokens before decoding
        full_features = self.reinsert_dropout_tokens(
            encoded_output, patch_embeddings, position_mapping, dropout_mask, combined_mask
        )

        # Step 7: Decode and reconstruct
        decoded_features = self.vit.forward_decoder(full_features, encoder_features) # [B, num_patches, embed_dim]
        reconstructed = self.decoder_head(decoded_features) # [B, num_patches, patch_size]
        # remove CLS token from reconstruction
        reconstructed = reconstructed[:, 1:]
        artificial_mask = artificial_mask[:, 1:]
        # Step 8: Compute loss only on artificially masked tokens
        loss = self.compute_reconstruction_loss(
            x, reconstructed, artificial_mask
        )
        
        return {
            "loss": loss,
            "reconstructed": reconstructed,
            "artificial_mask": artificial_mask,
            "inherited_mask": inherited_mask,
            "dropout_mask": dropout_mask
        }


class ProjectionHead(nn.Module):
    """Projection head for contrastive learning"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, norm_last_layer: bool = True):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.norm_last_layer = norm_last_layer
        self.last_layer = nn.utils.weight_norm(nn.Linear(output_dim, output_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)
        if self.norm_last_layer:
            x = F.normalize(self.last_layer(x), dim=-1)
        return x


class TimeSeriesAugmentations:
    """Augmentations for time series data"""
    @staticmethod
    def jitter(x: torch.Tensor, sigma: float = 0.05) -> torch.Tensor:
        return x + torch.randn_like(x) * sigma
    
    @staticmethod
    def scaling(x: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
        factor = torch.randn(x.size(0), 1, 1, device=x.device) * sigma + 1.0
        return x * factor
    
    @staticmethod
    def time_shift(x: torch.Tensor, max_shift: int = 10) -> torch.Tensor:
        B, C, L = x.shape
        shifts = torch.randint(-max_shift, max_shift+1, (B,), device=x.device)
        shifted_x = torch.zeros_like(x)
        
        for i in range(B):
            shift = shifts[i].item()
            if shift > 0:
                shifted_x[i, :, shift:] = x[i, :, :-shift]
                shifted_x[i, :, :shift] = x[i, :, :1]  # Repeat first value
            elif shift < 0:
                shift = -shift
                shifted_x[i, :, :-shift] = x[i, :, shift:]
                shifted_x[i, :, -shift:] = x[i, :, -1:]  # Repeat last value
            else:
                shifted_x[i] = x[i]
                
        return shifted_x
    
    @staticmethod
    def channel_shuffle(x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        shuffled_x = torch.zeros_like(x)
        
        for i in range(B):
            channel_indices = torch.randperm(C)
            shuffled_x[i] = x[i, channel_indices]
            
        return shuffled_x
    
    @staticmethod
    def apply_augmentations(x: torch.Tensor, x_lengths: list, aug_strength: float = 0.5) -> torch.Tensor:
        """Apply a random set of augmentations with controlled strength only to non-padded areas"""
        # Select which augmentations to apply
        aug_types = ['jitter', 'scaling', 'time_shift', 'channel_shuffle']
        num_augs = random.randint(1, len(aug_types))
        selected_augs = random.sample(aug_types, num_augs)
        
        x_aug = x.clone()
        
        if x_lengths is not None:
            # Apply augmentations only to non-padded regions
            batch_size = x.shape[0]
            
            for i in range(batch_size):
                # Get valid (non-padded) length for this sample
                valid_length = x_lengths[i]

                # Extract only valid sequence
                valid_seq = x_aug[i, :valid_length, :].unsqueeze(0)  # [1, valid_len, channels]
                
                # Apply selected augmentations only to valid sequence
                for aug in selected_augs:
                    if aug == 'jitter':
                        valid_seq = TimeSeriesAugmentations.jitter(valid_seq, sigma=0.05 * aug_strength)
                    elif aug == 'scaling':
                        valid_seq = TimeSeriesAugmentations.scaling(valid_seq, sigma=0.1 * aug_strength)
                    elif aug == 'time_shift':
                        valid_seq = TimeSeriesAugmentations.time_shift(valid_seq, max_shift=int(10 * aug_strength))
                    elif aug == 'channel_shuffle':
                        if random.random() < aug_strength:
                            valid_seq = TimeSeriesAugmentations.channel_shuffle(valid_seq)
                
                # Put augmented sequence back, keep padding unchanged
                x_aug[i, :valid_length, :] = valid_seq.squeeze(0)
                # x_aug[i, valid_length:, :] remains as original padding
        else:
            # Fallback: apply augmentations to entire sequence if no x_lengths provided
            for aug in selected_augs:
                if aug == 'jitter':
                    x_aug = TimeSeriesAugmentations.jitter(x_aug, sigma=0.05 * aug_strength)
                elif aug == 'scaling':
                    x_aug = TimeSeriesAugmentations.scaling(x_aug, sigma=0.1 * aug_strength)
                elif aug == 'time_shift':
                    x_aug = TimeSeriesAugmentations.time_shift(x_aug, max_shift=int(10 * aug_strength))
                elif aug == 'channel_shuffle':
                    if random.random() < aug_strength:
                        x_aug = TimeSeriesAugmentations.channel_shuffle(x_aug)
                    
        return x_aug


class DINOPretrainer(nn.Module):
    """
    Updated DINO pretraining for time series models.
    Supports flexible model architectures with improved feature extraction.
    """
    def __init__(
        self,
        base_model: nn.Module,
        feature_dim: int = 64,
        projection_dim: int = 128,
        projection_hidden: int = 64,
        momentum: float = 0.996,
        temperature_student: float = 0.1,
        temperature_teacher: float = 0.04,
        center_momentum: float = 0.9,
        global_crops: int = 2,
        local_crops: int = 6,
    ):
        super().__init__()
        self.student = base_model
        self.teacher = self._create_teacher_model(base_model)
        
        # Disable gradient updates for teacher
        for p in self.teacher.parameters():
            p.requires_grad = False
            
        # Get feature dimension from model
        self.feat_dim = feature_dim
        
        # Create projection heads
        self.student_head = ProjectionHead(self.feat_dim, projection_hidden, projection_dim)
        self.teacher_head = ProjectionHead(self.feat_dim, projection_hidden, projection_dim)
        
        # Copy student weights to teacher head
        self._update_teacher(m=0)
        
        # Initialize center for DINO loss
        self.register_buffer("center", torch.zeros(1, projection_dim))
        
        # Hyperparameters
        self.momentum = momentum
        self.temperature_student = temperature_student
        self.temperature_teacher = temperature_teacher
        self.center_momentum = center_momentum
        self.global_crops = global_crops
        self.local_crops = local_crops
        
    def _create_teacher_model(self, model):
        import copy
        teacher = copy.deepcopy(model)
        return teacher

    
    def _update_teacher(self, m=None):
        """Update teacher model with EMA of student"""
        momentum = self.momentum if m is None else m
        
        # Update teacher model weights
        with torch.no_grad():
            for param_student, param_teacher in zip(self.student.parameters(), 
                                                   self.teacher.parameters()):
                param_teacher.data.mul_(momentum).add_(
                    (1 - momentum) * param_student.detach().data
                )
                
            # Update teacher head weights
            for param_student, param_teacher in zip(self.student_head.parameters(),
                                                  self.teacher_head.parameters()):
                param_teacher.data.mul_(momentum).add_(
                    (1 - momentum) * param_student.detach().data
                )
    
    def _create_crops(self, x, x_lengths):
        """Create global and local crops from input"""
        crops = []
        crops_lens = []
        
        # Global crops (full size with different augmentations)
        for i in range(self.global_crops):
            crop = TimeSeriesAugmentations.apply_augmentations(x, x_lengths, aug_strength=0.3)
            crops.append(crop)
            crops_lens.append(x_lengths)
        
        # Local crops (smaller temporal windows with stronger augmentations)
        B, C, L = x.shape
        for _ in range(self.local_crops):
            # Create temporal crops only within non-padding areas
            crop = torch.zeros_like(x)  # Initialize with same shape
            crop_lengths = []
            
            for i in range(B):
                valid_length = x_lengths[i]
                # Determine crop size within valid sequence
                crop_ratio = random.uniform(0.5, 0.8)
                crop_size = max(1, int(valid_length * crop_ratio))
                
                # Random start position within valid sequence
                max_start = max(0, valid_length - crop_size)
                start_idx = random.randint(0, max_start) if max_start > 0 else 0
                
                # Extract crop from valid sequence only
                valid_crop = x[i, :, start_idx:start_idx + crop_size]  # [C, crop_size]
                # Interpolate crop back to original sequence length
                if crop_size != L:
                    valid_crop = F.interpolate(
                        valid_crop.unsqueeze(0), size=L, mode='linear', align_corners=False
                    ).squeeze(0)
                
                crop[i] = valid_crop
                # Scale crop length back to full size
                scaled_crop_length = min(L, int(crop_size * (L / valid_length)) if valid_length > 0 else 0)
                crop_lengths.append(scaled_crop_length)
            
            # Apply strong augmentations with proper crop lengths
            crop = TimeSeriesAugmentations.apply_augmentations(crop, crop_lengths, aug_strength=0.8)
            crops.append(crop)
            crops_lens.append(crop_lengths)
            
        return crops, crops_lens
                
    def forward(self, x, x_lengths) -> Dict:
        """
        Forward pass for DINO pretraining
        
        Args:
            x: Input tensor [B, C, L]
            
        Returns:
            Dict containing loss and other metrics
        """
        # Create crops
        x = x.permute(0, 2, 1)
        crops, crops_lens = self._create_crops(x, x_lengths)
        
        # Get student features for all crops
        student_features = []
        for crop, crop_lengths in zip(crops, crops_lens):
            feat = self.student(crop.permute(0, 2, 1), crop_lengths)
            student_features.append(feat)
        
        # Get teacher features (only for global crops)
        teacher_features = []
        with torch.no_grad():
            for i in range(self.global_crops):
                feat = self.teacher(crops[i].permute(0, 2, 1), crops_lens[i])
                teacher_features.append(feat)
        
        # Apply projection heads
        student_projections = [self.student_head(feat) for feat in student_features]
        
        with torch.no_grad():
            teacher_projections = [self.teacher_head(feat) for feat in teacher_features]
            
            # Update center using teacher projections (before softmax)
            all_teacher_projections = torch.cat(teacher_projections, dim=0)
            self.center = self.center * self.center_momentum + \
                          all_teacher_projections.mean(dim=0, keepdim=True) * (1 - self.center_momentum)
            
            # Compute teacher probabilities with centering
            teacher_probs = []
            for proj in teacher_projections:
                # Center first, then apply temperature and softmax
                centered_logits = (proj - self.center) / self.temperature_teacher
                prob = F.softmax(centered_logits, dim=-1)
                teacher_probs.append(prob)
        
        # Compute DINO loss
        loss = 0.0
        n_loss_terms = 0
        
        for i, teacher_prob in enumerate(teacher_probs):
            for j, student_proj in enumerate(student_projections):
                if i != j:  # Don't compare crop with itself
                    student_logits = student_proj / self.temperature_student
                    loss += -torch.sum(teacher_prob * F.log_softmax(student_logits, dim=-1), dim=-1).mean()
                    n_loss_terms += 1
        
        loss = loss / n_loss_terms if n_loss_terms > 0 else loss
        
        # Update teacher model
        self._update_teacher()
        
        return {
            "loss": loss
        }


class SimCLRPretrainer(nn.Module):
    """
    SimCLR (Simple Contrastive Learning of Visual Representations) pretrainer for time series models.
    Follows the design pattern of DINOPretrainer but uses SimCLR's contrastive loss.
    """
    def __init__(
        self,
        base_model: nn.Module,
        feature_dim: int = 64,
        projection_dim: int = 128,
        projection_hidden: int = 64,
        temperature: float = 0.1,
        num_views: int = 2,
        extra_views: int = 0,
    ):
        super().__init__()
        self.encoder = base_model
        
        # Get feature dimension from model
        self.feat_dim = feature_dim
        
        # Create projection head
        self.projection_head = ProjectionHead(self.feat_dim, projection_hidden, projection_dim)
        
        # Hyperparameters
        self.temperature = temperature
        self.num_views = num_views
        self.extra_views = extra_views
        
    def _create_crops(self, x, x_lengths):
        """Create augmented views for SimCLR contrastive learning"""
        crops = []
        crops_lens = []
        
        # Create augmented views of the same input (standard SimCLR approach)
        for i in range(self.num_views):
            crop = TimeSeriesAugmentations.apply_augmentations(x, x_lengths, aug_strength=0.4)
            crops.append(crop)
            crops_lens.append(x_lengths)
        
        # Optionally add more augmented views for stronger contrastive learning
        if self.extra_views > 0:
            for _ in range(self.extra_views):
                # Create additional augmented views with varying strength
                aug_strength = random.uniform(0.3, 0.7)
                crop = TimeSeriesAugmentations.apply_augmentations(x, x_lengths, aug_strength=aug_strength)
                crops.append(crop)
                crops_lens.append(x_lengths)
            
        return crops, crops_lens
    
    def _contrastive_loss(self, features_list):
        """
        Compute SimCLR contrastive loss using cosine similarity.
        
        Args:
            features_list: List of projected features from different views
            
        Returns:
            Contrastive loss value
        """
        # Concatenate all features
        all_features = torch.cat(features_list, dim=0)  # [N_total, projection_dim]
        
        # Normalize features
        all_features = F.normalize(all_features, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(all_features, all_features.T) / self.temperature
        
        batch_size = features_list[0].size(0)
        n_views = len(features_list)
        
        # Create mask to exclude self-similarity
        mask = torch.eye(similarity_matrix.size(0), dtype=torch.bool).to(all_features.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -float('inf'))
        
        # Compute contrastive loss
        loss = 0.0
        n_loss_terms = 0
        
        for i in range(n_views):
            for j in range(n_views):
                if i != j:  # Don't compare view with itself
                    # Get features for current view pair
                    start_i, end_i = i * batch_size, (i + 1) * batch_size
                    start_j, end_j = j * batch_size, (j + 1) * batch_size
                    
                    # Similarities for positive pairs (same sample, different views)
                    pos_sim = similarity_matrix[start_i:end_i, start_j:end_j].diag()
                    
                    # All similarities for denominator (excluding self)
                    neg_mask = torch.ones_like(similarity_matrix[start_i:end_i]).bool()
                    neg_mask[:, start_i:end_i] = False  # Exclude self-similarities
                    
                    # Compute log softmax for contrastive loss
                    logits = similarity_matrix[start_i:end_i]
                    logits_max = torch.max(logits, dim=1, keepdim=True)[0]
                    logits = logits - logits_max.detach()
                    
                    # Positive logits
                    pos_logits = pos_sim - logits_max.squeeze()
                    
                    # Denominator: sum of exp over all negatives + positive
                    exp_logits = torch.exp(logits)
                    exp_logits = exp_logits.masked_fill(~neg_mask, 0.0)
                    denominator = exp_logits.sum(dim=1) + torch.exp(pos_logits)
                    
                    # Compute loss: -log(exp(pos) / denominator)
                    view_loss = -pos_logits + torch.log(denominator)
                    loss += view_loss.mean()
                    n_loss_terms += 1
        
        return loss / n_loss_terms if n_loss_terms > 0 else loss
                
    def forward(self, x, x_lengths) -> Dict:
        """
        Forward pass for SimCLR pretraining
        
        Args:
            x: Input tensor [B, L, C]
            x_lengths: Original lengths of each sequence in the batch
            
        Returns:
            Dict containing loss and other metrics
        """
        # x = x.permute(0, 2, 1)
        # Create crops

        crops, crops_lens = self._create_crops(x, x_lengths)
        
        # Get features for all crops
        features = []
        for crop, crop_lengths in zip(crops, crops_lens):
            feat = self.encoder(crop, crop_lengths)
            features.append(feat)
        
        # Apply projection head
        projected_features = [self.projection_head(feat) for feat in features]
        
        # Compute contrastive loss
        loss = self._contrastive_loss(projected_features)
        
        return {
            "loss": loss,
            "n_crops": len(crops),
            "features": features,
            "projected_features": projected_features
        }


class RelConPretrainer(nn.Module):
    """
    RelCon (Relative Contrastive Learning) implementation for time series.
    Based on https://github.com/maxxu05/relcon
    
    RelCon learns representations by preserving relative similarities between
    temporal segments, capturing hierarchical structures in time series data.
    """
    def __init__(
        self,
        base_model: nn.Module,
        feature_dim: int = 64,
        projection_dim: int = 128,
        projection_hidden: int = 256,
        temperature: float = 0.1,
        lambda_temporal: float = 5.0,
        num_candidates: int = 8,
        within_subject_ratio: float = 0.5,
        num_views: int = 2,
    ):
        super().__init__()
        self.backbone = base_model
        self.feat_dim = feature_dim
        self.temperature = temperature
        self.lambda_temporal = lambda_temporal
        self.num_candidates = num_candidates
        self.within_subject_ratio = within_subject_ratio
        self.num_views = num_views
        
        # Projection head for contrastive learning
        self.projection_head = ProjectionHead(
            input_dim=self.feat_dim,
            hidden_dim=projection_hidden,
            output_dim=projection_dim,
            norm_last_layer=True
        )
        
        # Distance measure for candidate ranking
        self.distance_net = nn.Sequential(
            nn.Linear(self.feat_dim * 2, projection_hidden),
            nn.ReLU(),
            nn.Linear(projection_hidden, projection_hidden // 2),
            nn.ReLU(),
            nn.Linear(projection_hidden // 2, 1),
            nn.Sigmoid()
        )
    
    def _create_views(self, x, x_lengths):
        """Create augmented views for RelCon"""
        views = []
        
        # Create multiple augmented views
        for _ in range(self.num_views):
            aug_strength = random.uniform(0.3, 0.6)
            view = TimeSeriesAugmentations.apply_augmentations(x, x_lengths, aug_strength=aug_strength)
            views.append(view)
        
        return views
    

    def _extract_features(self, x, x_lengths):
        """Extract features from the backbone model"""
        try:
            # Try calling with x_lengths first (for models that support it)
            features = self.backbone(x, x_lengths)
            return features
        except TypeError:
            # Fallback: call without x_lengths
            features = self.backbone(x)
            return features
    
    def _generate_candidates(self, anchor_features, all_features, subject_ids=None):
        """
        Generate candidate signals for relative contrastive learning.
        
        Args:
            anchor_features: Features of anchor signals [B, D]
            all_features: Pool of all available features [N, D]
            subject_ids: Subject IDs for within/between subject sampling
            
        Returns:
            candidate_features: Selected candidate features [B, num_candidates, D]
            distances: Distances between anchor and candidates [B, num_candidates]
        """
        B, D = anchor_features.shape
        N = all_features.shape[0]
        
        # If we have subject IDs, implement within/between subject sampling
        if subject_ids is not None:
            # This would be implemented for multi-subject scenarios
            # For now, use random sampling
            pass
        
        # Simple random candidate selection for this implementation
        candidate_indices = torch.randint(0, N, (B, self.num_candidates), device=anchor_features.device)
        candidate_features = all_features[candidate_indices]  # [B, num_candidates, D]
        
        # Compute distances using the distance network
        anchor_expanded = anchor_features.unsqueeze(1).expand(-1, self.num_candidates, -1)
        
        # Concatenate anchor and candidate features
        pairs = torch.cat([anchor_expanded, candidate_features], dim=-1)  # [B, num_candidates, 2*D]
        
        # Compute distances - handle potential dimension mismatch
        expected_input_dim = 2 * D
        actual_pairs_dim = pairs.shape[-1]
        
        if expected_input_dim != actual_pairs_dim:
            # If dimensions don't match, fall back to simple euclidean distance
            distances = torch.norm(anchor_expanded - candidate_features, dim=-1)  # [B, num_candidates]
        else:
            distances = self.distance_net(pairs.view(-1, 2*D)).view(B, self.num_candidates)
        
        return candidate_features, distances
    
    def _compute_relcon_loss(self, anchor_proj, candidate_proj, distances):
        """
        Compute RelCon loss based on relative similarities.
        
        Args:
            anchor_proj: Projected anchor features [B, proj_dim]
            candidate_proj: Projected candidate features [B, num_candidates, proj_dim]
            distances: Distance rankings [B, num_candidates]
            
        Returns:
            RelCon loss
        """
        B, num_candidates, proj_dim = candidate_proj.shape
        
        # Normalize features
        anchor_norm = F.normalize(anchor_proj, dim=1)  # [B, proj_dim]
        candidate_norm = F.normalize(candidate_proj, dim=2)  # [B, num_candidates, proj_dim]
        
        # Compute similarities
        similarities = torch.bmm(
            candidate_norm, 
            anchor_norm.unsqueeze(-1)
        ).squeeze(-1)  # [B, num_candidates]
        
        # Scale by temperature
        similarities = similarities / self.temperature
        
        # RelCon loss: rank candidates by distance and encourage similarity to match distance
        # Closer candidates (smaller distance) should have higher similarity
        
        # Sort candidates by distance (ascending order)
        sorted_distances, sort_indices = torch.sort(distances, dim=1)
        sorted_similarities = torch.gather(similarities, 1, sort_indices)
        
        # Create target probabilities based on distance ranking
        # Exponential decay: closer candidates get higher probability
        target_probs = torch.exp(-self.lambda_temporal * sorted_distances)
        target_probs = target_probs / target_probs.sum(dim=1, keepdim=True)
        
        # Compute cross-entropy loss
        log_probs = F.log_softmax(sorted_similarities, dim=1)
        loss = -torch.sum(target_probs * log_probs, dim=1).mean()
        
        return loss
    
    def forward(self, x: torch.Tensor, x_lengths, subject_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for RelCon pretraining.
        
        Args:
            x: Input tensor [B, seq_len, channels]
            x_lengths: Sequence lengths for each sample
            subject_ids: Optional subject identifiers for sampling strategy
            
        Returns:
            Dict containing loss and features
        """
        # Convert x_lengths to list if it's a tensor
        if isinstance(x_lengths, torch.Tensor):
            x_lengths = x_lengths.tolist()
        
        # Create augmented views
        views = self._create_views(x, x_lengths)
        
        # Extract features from views
        features_list = []
        for view in views:
            features = self._extract_features(view, x_lengths)
            features_list.append(features)
        
        # Use first view as anchor, second as candidate pool
        anchor_features = features_list[0]
        candidate_pool_features = features_list[1] if len(features_list) > 1 else features_list[0]
        
        # Generate candidates and compute distances
        candidate_features, distances = self._generate_candidates(
            anchor_features, candidate_pool_features, subject_ids
        )
        
        # Project features
        anchor_proj = self.projection_head(anchor_features)
        candidate_proj = self.projection_head(candidate_features.view(-1, self.feat_dim))
        candidate_proj = candidate_proj.view(x.size(0), self.num_candidates, -1)
        
        # Compute RelCon loss
        loss = self._compute_relcon_loss(anchor_proj, candidate_proj, distances)
        
        return {
            "loss": loss,
            "features": anchor_features.detach(),
            "projected_features": anchor_proj.detach(),
            "distances": distances.detach()
        }

if __name__ == "__main__":
    x = torch.randn(2, 3, 20)  # Example input tensor
    x[0, 1, 5:10] = -999.0  # Simulating missing values
    x[1, 2, 15:20] = -999.0  #
    vit = ViT1D(time_length=20, patch_size=10, stride=10, in_channels=3, embed_dim=20, encoder_layers=4, decoder_layers=2, num_heads=2, mlp_ratio=2.0, dropout=0.1, missing_value=-999.0, missing_patch_thres=0.5, return_missing_mask=True)
    pretrainer = AIMPretrainer(vit_model=vit, mask_ratio=0.5)
    # pretrainer = MAEPretrainer(vit_model=vit, mask_ratio=0.5, norm_pixel_loss=True)

    # x, mask = vit.embedding(x)
    # print(x.shape)  # Should print the shape of the output tensor
    # print(mask) 

    # # inherited mask
    # expected_output = torch.tensor([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]])  # Example output tensor
    # mask = pretrainer.generate_inherited_mask(x)
    # print(mask.shape)  # Should print the shape of the mask tensor
    # print(mask.int())

    # # artificial mask
    # artificial_mask, counts = pretrainer.generate_artificial_mask(mask)
    # print(artificial_mask.shape)  # Should print the shape of the artificial mask tensor
    # print(artificial_mask.int())
    # print(counts)  
    # expected_counts = [3, 3]

    # # select dropout tokens
    # dropout_counts = [3, 2]  # Example dropout ratios for each batch item
    # # dropout_counts = []
    # combined_mask, dropout_mask = pretrainer.select_dropout_tokens(mask, artificial_mask, dropout_counts)
    # print(combined_mask.int())
    # print(dropout_mask.int())

    # # Process input for encoder
    # patch_embeddings = vit.patch_embed(x)
    # x_kept, attention_mask, position_mapping = pretrainer.process_encoder_input(patch_embeddings, combined_mask, dropout_mask)
    # print(x_kept.shape)  # Should print the shape of the processed input tensor
    # print(attention_mask.int())  # Should print the shape of the attention mask tensor
    # print(attention_mask.shape)  # Should print the shape of the attention mask tensor
    # print(position_mapping)  # Should print the position mapping for each batch item

    # expected attention mask
    # exp_attn_mask = []
    # for i, maps in enumerate(position_mapping):
    #     attn_mask_b = [combined_mask[i][orig_idx].int().item() for orig_idx in maps.values()]
    #     exp_attn_mask.append(attn_mask_b)
    # print("expected attention mask:", exp_attn_mask)

    # # reinsert dropout tokens
    # full_features = pretrainer.reinsert_dropout_tokens(
    #     x_kept, patch_embeddings, position_mapping, dropout_mask, combined_mask
    # )
    # print(full_features.shape)  # Should print the shape of the full features tensor

    # # check inserted dropout tokens match original input
    # for b in range(len(dropout_mask)):
    #     for i in range(dropout_mask.shape[1]):
    #         if dropout_mask[b, i]:
    #             assert torch.all(full_features[b, i] == patch_embeddings[b, i]), f"Mismatch at batch {b}, token {i}"

    # forward
    res = pretrainer.forward(x)
    # print(res[0].item(), res[1].shape, res[2].shape) 
    print(res["loss"].item())  # Should print the loss value
    print(res["reconstructed"].shape)  # Should print the shape of the reconstructed tensor
    print(res["artificial_mask"].int())  # Should print the artificial mask tensor
    print(res["inherited_mask"].int())  # Should print the inherited mask tensor
    print(res["dropout_mask"].int())  # Should print the dropout mask tensor