"""
Distributed contrastive learning pretrainers with cross-GPU negative sampling.
Designed for DINO and SimCLR with proper handling of x_lengths and batch distribution.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import random
import copy
from typing import List, Dict, Any, Optional, Tuple

try:
    from .pretrainer import ProjectionHead, TimeSeriesAugmentations
except:
    from pretrainer import ProjectionHead, TimeSeriesAugmentations


def setup_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', rank % torch.cuda.device_count()))
        
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        
        return True, rank, world_size, local_rank
    return False, 0, 1, 0


def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def gather_tensor_with_lengths(tensor: torch.Tensor, lengths: List[int]) -> Tuple[torch.Tensor, List[int]]:
    """
    Gather tensors and their corresponding lengths from all GPUs.
    Handles variable-length sequences properly by converting lengths to tensor.
    
    Args:
        tensor: Local tensor [batch_size, seq_len, features]
        lengths: List of sequence lengths for local batch
        
    Returns:
        gathered_tensor: All tensors concatenated [world_batch_size, seq_len, features]
        gathered_lengths: All lengths concatenated as list
    """
    if not dist.is_initialized():
        return tensor, lengths
    
    world_size = dist.get_world_size()
    
    # Convert lengths to tensor for gathering
    lengths_tensor = torch.tensor(lengths, dtype=torch.long, device=tensor.device)
    
    # Gather tensors
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    
    # Gather lengths
    gathered_lengths_tensors = [torch.zeros_like(lengths_tensor) for _ in range(world_size)]
    dist.all_gather(gathered_lengths_tensors, lengths_tensor)
    
    # Concatenate results
    gathered_tensor = torch.cat(gathered_tensors, dim=0)
    gathered_lengths = []
    for lt in gathered_lengths_tensors:
        gathered_lengths.extend(lt.tolist())
    
    return gathered_tensor, gathered_lengths


class DistributedDINOPretrainer(nn.Module):
    """
    Distributed DINO pretrainer with cross-GPU negative sampling.
    Maintains the same interface as DINOPretrainer but handles distribution internally.
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
        gather_negatives: bool = True,  # Whether to gather negatives from all GPUs
    ):
        super().__init__()
        
        # Setup distributed
        self.is_distributed, self.rank, self.world_size, self.local_rank = setup_distributed()
        
        # Initialize models
        self.student = base_model
        self.teacher = self._create_teacher_model(base_model)
        
        # Wrap in DDP if distributed
        if self.is_distributed:
            self.student = DDP(self.student, device_ids=[self.local_rank])
            # Teacher doesn't need DDP as it has no gradients
        
        # Disable gradient updates for teacher
        for p in self.teacher.parameters():
            p.requires_grad = False
            
        self.feat_dim = feature_dim
        
        # Create projection heads
        self.student_head = ProjectionHead(self.feat_dim, projection_hidden, projection_dim)
        self.teacher_head = ProjectionHead(self.feat_dim, projection_hidden, projection_dim)
        
        # Wrap projection heads in DDP if distributed
        if self.is_distributed:
            self.student_head = DDP(self.student_head, device_ids=[self.local_rank])
            # Teacher head doesn't need DDP
        
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
        self.gather_negatives = gather_negatives
        
    def _create_teacher_model(self, model):
        teacher = copy.deepcopy(model)
        return teacher
    
    def _update_teacher(self, m=None):
        """Update teacher model with EMA of student"""
        momentum = self.momentum if m is None else m
        
        # Get unwrapped models for parameter access
        student_model = self.student.module if isinstance(self.student, DDP) else self.student
        student_head = self.student_head.module if isinstance(self.student_head, DDP) else self.student_head
        
        with torch.no_grad():
            # Update teacher model weights
            for param_student, param_teacher in zip(student_model.parameters(), 
                                                   self.teacher.parameters()):
                param_teacher.data.mul_(momentum).add_(
                    (1 - momentum) * param_student.detach().data
                )
                
            # Update teacher head weights
            for param_student, param_teacher in zip(student_head.parameters(),
                                                   self.teacher_head.parameters()):
                param_teacher.data.mul_(momentum).add_(
                    (1 - momentum) * param_student.detach().data
                )
    
    def _create_crops(self, x, x_lengths):
        """Create global and local crops for DINO"""
        crops = []
        crops_lens = []
        
        # Global crops (standard augmentations)
        for i in range(self.global_crops):
            crop = TimeSeriesAugmentations.apply_augmentations(x, x_lengths, aug_strength=0.5)
            crops.append(crop)
            crops_lens.append(x_lengths)
        
        # Local crops (stronger augmentations + temporal cropping)
        for i in range(self.local_crops):
            # Apply temporal cropping first
            crop, crop_lens = TimeSeriesAugmentations.temporal_crop(x, x_lengths, crop_ratio=0.7)
            # Then apply augmentations
            crop = TimeSeriesAugmentations.apply_augmentations(crop, crop_lens, aug_strength=0.8)
            crops.append(crop)
            crops_lens.append(crop_lens)
        
        return crops, crops_lens
    
    def _compute_dino_loss(self, student_features, teacher_features):
        """
        Compute DINO loss with proper cross-GPU negative sampling.
        
        Args:
            student_features: List of student features for each crop
            teacher_features: List of teacher features for each crop (only global crops)
        """
        total_loss = 0
        num_loss_terms = 0
        
        # Gather features from all GPUs if distributed and gather_negatives is True
        if self.is_distributed and self.gather_negatives:
            # Gather all student and teacher features across GPUs
            gathered_student_features = []
            gathered_teacher_features = []
            
            for sf in student_features:
                gathered_sf = [torch.zeros_like(sf) for _ in range(self.world_size)]
                dist.all_gather(gathered_sf, sf)
                gathered_student_features.append(torch.cat(gathered_sf, dim=0))
            
            for tf in teacher_features:
                gathered_tf = [torch.zeros_like(tf) for _ in range(self.world_size)]
                dist.all_gather(gathered_tf, tf)
                gathered_teacher_features.append(torch.cat(gathered_tf, dim=0))
                
            student_features = gathered_student_features
            teacher_features = gathered_teacher_features
        
        # DINO loss: student learns from teacher on global crops
        for i, s_feat in enumerate(student_features):
            for j, t_feat in enumerate(teacher_features):
                if i != j:  # No self-prediction
                    # Student prediction
                    s_prob = F.log_softmax((s_feat - self.center) / self.temperature_student, dim=-1)
                    
                    # Teacher prediction (no gradients)
                    with torch.no_grad():
                        t_prob = F.softmax((t_feat - self.center) / self.temperature_teacher, dim=-1)
                    
                    # Cross-entropy loss
                    loss = -torch.sum(t_prob * s_prob, dim=-1).mean()
                    total_loss += loss
                    num_loss_terms += 1
        
        return total_loss / num_loss_terms if num_loss_terms > 0 else torch.tensor(0.0, device=student_features[0].device)
    
    def _update_center(self, teacher_features):
        """Update center for DINO loss with proper distributed handling"""
        if len(teacher_features) == 0:
            return
            
        # Compute batch center
        batch_center = torch.cat(teacher_features, dim=0).mean(dim=0, keepdim=True)
        
        # If distributed, average across all GPUs
        if self.is_distributed:
            dist.all_reduce(batch_center, op=dist.ReduceOp.SUM)
            batch_center /= self.world_size
        
        # Update center with momentum
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
    
    def forward(self, x, x_lengths):
        """Forward pass with distributed handling"""
        # Convert x_lengths to list if it's a tensor (for distributed compatibility)
        if isinstance(x_lengths, torch.Tensor):
            x_lengths = x_lengths.tolist()
        
        # Create crops
        crops, crops_lens = self._create_crops(x, x_lengths)
        
        # Process through student
        student_features = []
        for crop, crop_len in zip(crops, crops_lens):
            # Get features from model
            if hasattr(self.student, 'module'):  # DDP wrapped
                feat = self.student.module(crop, crop_len)
            else:
                feat = self.student(crop, crop_len)
            
            # Project features
            if hasattr(self.student_head, 'module'):  # DDP wrapped
                proj_feat = self.student_head.module(feat)
            else:
                proj_feat = self.student_head(feat)
            
            student_features.append(proj_feat)
        
        # Process global crops through teacher (no gradients)
        teacher_features = []
        with torch.no_grad():
            for i in range(self.global_crops):
                crop, crop_len = crops[i], crops_lens[i]
                feat = self.teacher(crop, crop_len)
                proj_feat = self.teacher_head(feat)
                teacher_features.append(proj_feat)
        
        # Compute DINO loss
        loss = self._compute_dino_loss(student_features, teacher_features)
        
        # Update center
        self._update_center(teacher_features)
        
        # Update teacher with EMA
        self._update_teacher()
        
        return {
            "loss": loss,
            "student_features": student_features,
            "teacher_features": teacher_features
        }


class DistributedSimCLRPretrainer(nn.Module):
    """
    Distributed SimCLR pretrainer with cross-GPU negative sampling.
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
        gather_negatives: bool = True,
    ):
        super().__init__()
        
        # Setup distributed
        self.is_distributed, self.rank, self.world_size, self.local_rank = setup_distributed()
        
        # Initialize encoder
        self.encoder = base_model
        
        # Wrap in DDP if distributed
        if self.is_distributed:
            self.encoder = DDP(self.encoder, device_ids=[self.local_rank])
        
        self.feat_dim = feature_dim
        
        # Create projection head
        self.projection_head = ProjectionHead(self.feat_dim, projection_hidden, projection_dim)
        
        # Wrap projection head in DDP if distributed
        if self.is_distributed:
            self.projection_head = DDP(self.projection_head, device_ids=[self.local_rank])
        
        # Hyperparameters
        self.temperature = temperature
        self.num_views = num_views
        self.extra_views = extra_views
        self.gather_negatives = gather_negatives
        
    def _create_views(self, x, x_lengths):
        """Create augmented views for SimCLR"""
        views = []
        views_lens = []
        
        # Create standard augmented views
        for i in range(self.num_views):
            view = TimeSeriesAugmentations.apply_augmentations(x, x_lengths, aug_strength=0.4)
            views.append(view)
            views_lens.append(x_lengths)
        
        # Create additional views if specified
        if self.extra_views > 0:
            for _ in range(self.extra_views):
                aug_strength = random.uniform(0.3, 0.7)
                view = TimeSeriesAugmentations.apply_augmentations(x, x_lengths, aug_strength=aug_strength)
                views.append(view)
                views_lens.append(x_lengths)
        
        return views, views_lens
    
    def _compute_simclr_loss(self, features_list):
        """
        Compute SimCLR contrastive loss with cross-GPU negative sampling.
        
        Args:
            features_list: List of projected features for each view
        """
        # Concatenate all views
        all_features = torch.cat(features_list, dim=0)  # [num_views * batch_size, proj_dim]
        
        # Gather features from all GPUs if distributed and gather_negatives is True
        if self.is_distributed and self.gather_negatives:
            gathered_features = [torch.zeros_like(all_features) for _ in range(self.world_size)]
            dist.all_gather(gathered_features, all_features)
            all_features = torch.cat(gathered_features, dim=0)
        
        # Normalize features
        all_features = F.normalize(all_features, dim=-1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(all_features, all_features.T) / self.temperature
        
        # Create labels for positive pairs
        batch_size = len(features_list[0])
        if self.is_distributed and self.gather_negatives:
            total_batch_size = batch_size * self.world_size
        else:
            total_batch_size = batch_size
        
        labels = torch.arange(len(features_list)).repeat_interleave(total_batch_size)
        labels = labels.to(all_features.device)
        
        # Create mask to remove self-similarities
        mask = torch.eye(len(all_features), device=all_features.device).bool()
        similarity_matrix.masked_fill_(mask, float('-inf'))
        
        # Compute InfoNCE loss
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss
    
    def forward(self, x, x_lengths):
        """Forward pass with distributed handling"""
        # Convert x_lengths to list if it's a tensor
        if isinstance(x_lengths, torch.Tensor):
            x_lengths = x_lengths.tolist()
        
        # Create views
        views, views_lens = self._create_views(x, x_lengths)
        
        # Process through encoder and projection head
        features_list = []
        for view, view_len in zip(views, views_lens):
            # Get features from encoder
            if hasattr(self.encoder, 'module'):  # DDP wrapped
                feat = self.encoder.module(view, view_len)
            else:
                feat = self.encoder(view, view_len)
            
            # Project features
            if hasattr(self.projection_head, 'module'):  # DDP wrapped
                proj_feat = self.projection_head.module(feat)
            else:
                proj_feat = self.projection_head(feat)
            
            features_list.append(proj_feat)
        
        # Compute SimCLR loss
        loss = self._compute_simclr_loss(features_list)
        
        return {
            "loss": loss,
            "features": features_list
        }


def create_distributed_pretrainer(pretrainer_type: str, base_model: nn.Module, **kwargs):
    """
    Factory function to create distributed pretrainers.
    
    Args:
        pretrainer_type: 'dino' or 'simclr'
        base_model: The base model to be pretrained
        **kwargs: Additional arguments for the pretrainer
    """
    if pretrainer_type.lower() == 'dino':
        return DistributedDINOPretrainer(base_model, **kwargs)
    elif pretrainer_type.lower() == 'simclr':
        return DistributedSimCLRPretrainer(base_model, **kwargs)
    else:
        raise ValueError(f"Unknown pretrainer type: {pretrainer_type}")