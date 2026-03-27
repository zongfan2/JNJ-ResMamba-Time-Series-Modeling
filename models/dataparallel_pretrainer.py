"""
DataParallel contrastive learning pretrainers for single-server multi-GPU setups.
Much simpler than DistributedDataParallel and perfect for your use case.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import copy
from typing import List, Dict

try:
    from .pretrainer import ProjectionHead, TimeSeriesAugmentations
except:
    from pretrainer import ProjectionHead, TimeSeriesAugmentations


class DataParallelDINOPretrainer(nn.Module):
    """
    DataParallel DINO pretrainer with cross-GPU negative sampling.
    Simpler than DistributedDataParallel but still allows larger effective batch sizes.
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
        device_ids: List[int] = None,  # [0, 1, 2, 3] for 4 GPUs
    ):
        super().__init__()
        
        # Setup devices
        self.device_ids = device_ids if device_ids else list(range(torch.cuda.device_count()))
        self.primary_device = torch.device(f'cuda:{self.device_ids[0]}')
        
        # Initialize models
        self.student = base_model.to(self.primary_device)
        self.teacher = self._create_teacher_model(base_model).to(self.primary_device)
        
        # Wrap student in DataParallel (teacher stays on primary device)
        if len(self.device_ids) > 1:
            self.student = nn.DataParallel(self.student, device_ids=self.device_ids)
        
        # Disable gradient updates for teacher
        for p in self.teacher.parameters():
            p.requires_grad = False
            
        self.feat_dim = feature_dim
        
        # Create projection heads
        self.student_head = ProjectionHead(self.feat_dim, projection_hidden, projection_dim).to(self.primary_device)
        self.teacher_head = ProjectionHead(self.feat_dim, projection_hidden, projection_dim).to(self.primary_device)
        
        # Wrap student head in DataParallel
        if len(self.device_ids) > 1:
            self.student_head = nn.DataParallel(self.student_head, device_ids=self.device_ids)
        
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
        teacher = copy.deepcopy(model)
        return teacher
    
    def _update_teacher(self, m=None):
        """Update teacher model with EMA of student"""
        momentum = self.momentum if m is None else m
        
        # Get unwrapped models for parameter access
        student_model = self.student.module if isinstance(self.student, nn.DataParallel) else self.student
        student_head = self.student_head.module if isinstance(self.student_head, nn.DataParallel) else self.student_head
        
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
        
        # Convert x_lengths to tensor if it's a list (for DataParallel compatibility)
        if isinstance(x_lengths, list):
            x_lengths_tensor = torch.tensor(x_lengths, dtype=torch.long, device=x.device)
        else:
            x_lengths_tensor = x_lengths
        
        # Global crops (standard augmentations)
        for i in range(self.global_crops):
            crop = TimeSeriesAugmentations.apply_augmentations(x, x_lengths, aug_strength=0.5)
            crops.append(crop)
            crops_lens.append(x_lengths_tensor)  # Use tensor instead of list
        
        # Local crops (stronger augmentations + temporal cropping)
        for i in range(self.local_crops):
            # Apply temporal cropping first if available
            if hasattr(TimeSeriesAugmentations, 'temporal_crop'):
                crop, crop_lens = TimeSeriesAugmentations.temporal_crop(x, x_lengths, crop_ratio=0.7)
                # Then apply augmentations
                crop = TimeSeriesAugmentations.apply_augmentations(crop, crop_lens, aug_strength=0.8)
                crops.append(crop)
                # Convert crop_lens to tensor for DataParallel
                if isinstance(crop_lens, list):
                    crop_lens = torch.tensor(crop_lens, dtype=torch.long, device=x.device)
                crops_lens.append(crop_lens)
            else:
                # Fallback: just use stronger augmentations
                crop = TimeSeriesAugmentations.apply_augmentations(x, x_lengths, aug_strength=0.8)
                crops.append(crop)
                crops_lens.append(x_lengths_tensor)  # Use tensor instead of list
        
        return crops, crops_lens
    
    def _compute_dino_loss(self, student_features, teacher_features):
        """
        Compute DINO loss with cross-GPU negative sampling.
        DataParallel automatically handles gathering features from all GPUs.
        """
        total_loss = 0
        num_loss_terms = 0
        
        # DINO loss: student learns from teacher on global crops
        for i, s_feat in enumerate(student_features):
            for j, t_feat in enumerate(teacher_features):
                if i != j:  # No self-prediction
                    # Move center to same device as features
                    center = self.center.to(s_feat.device)
                    
                    # Student prediction
                    s_prob = F.log_softmax((s_feat - center) / self.temperature_student, dim=-1)
                    
                    # Teacher prediction (no gradients)
                    with torch.no_grad():
                        t_prob = F.softmax((t_feat - center) / self.temperature_teacher, dim=-1)
                    
                    # Cross-entropy loss
                    loss = -torch.sum(t_prob * s_prob, dim=-1).mean()
                    total_loss += loss
                    num_loss_terms += 1
        
        return total_loss / num_loss_terms if num_loss_terms > 0 else torch.tensor(0.0, device=self.primary_device)
    
    def _update_center(self, teacher_features):
        """Update center for DINO loss"""
        if len(teacher_features) == 0:
            return
            
        # Compute batch center from all teacher features
        batch_center = torch.cat(teacher_features, dim=0).mean(dim=0, keepdim=True)
        
        # Ensure center is on same device as batch_center
        center = self.center.to(batch_center.device)
        
        # Update center with momentum
        self.center = center * self.center_momentum + batch_center * (1 - self.center_momentum)
        
    def forward(self, x, x_lengths):
        """Forward pass with DataParallel handling"""
        # Convert x_lengths to list if it's a tensor
        if isinstance(x_lengths, torch.Tensor):
            x_lengths = x_lengths.tolist()
        
        # Ensure input is on primary device
        x = x.to(self.primary_device)
        
        # Create crops
        crops, crops_lens = self._create_crops(x, x_lengths)
        
        # Process through student (DataParallel handles multi-GPU automatically)
        student_features = []
        for crop, crop_len in zip(crops, crops_lens):
            # crop = crop.to(self.primary_device)
            # Get features from model
            feat = self.student(crop, crop_len)
            # Project features
            proj_feat = self.student_head(feat)
            student_features.append(proj_feat)
        
        # Process global crops through teacher (single GPU)
        teacher_features = []
        with torch.no_grad():
            for i in range(self.global_crops):
                crop, crop_len = crops[i], crops_lens[i]
                crop = crop.to(self.primary_device)
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


class DataParallelSimCLRPretrainer(nn.Module):
    """
    DataParallel SimCLR pretrainer with cross-GPU negative sampling.
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
        device_ids: List[int] = None,
    ):
        super().__init__()
        
        # Setup devices
        self.device_ids = device_ids if device_ids else list(range(torch.cuda.device_count()))
        self.primary_device = torch.device(f'cuda:{self.device_ids[0]}')
        
        # Initialize encoder
        self.encoder = base_model.to(self.primary_device)
        
        # Wrap in DataParallel
        if len(self.device_ids) > 1:
            self.encoder = nn.DataParallel(self.encoder, device_ids=self.device_ids)
        
        self.feat_dim = feature_dim
        
        # Create projection head
        self.projection_head = ProjectionHead(self.feat_dim, projection_hidden, projection_dim).to(self.primary_device)
        
        # Wrap projection head in DataParallel
        if len(self.device_ids) > 1:
            self.projection_head = nn.DataParallel(self.projection_head, device_ids=self.device_ids)
        
        # Hyperparameters
        self.temperature = temperature
        self.num_views = num_views
        self.extra_views = extra_views
        
    def _create_views(self, x, x_lengths):
        """Create augmented views for SimCLR"""
        views = []
        views_lens = []
        
        # Convert x_lengths to tensor if it's a list (for DataParallel compatibility)
        if isinstance(x_lengths, list):
            x_lengths_tensor = torch.tensor(x_lengths, dtype=torch.long, device=x.device)
        else:
            x_lengths_tensor = x_lengths
        
        # Create standard augmented views
        for i in range(self.num_views):
            view = TimeSeriesAugmentations.apply_augmentations(x, x_lengths, aug_strength=0.4)
            views.append(view)
            views_lens.append(x_lengths_tensor)  # Use tensor instead of list
        
        # Create additional views if specified
        if self.extra_views > 0:
            for _ in range(self.extra_views):
                aug_strength = random.uniform(0.3, 0.7)
                view = TimeSeriesAugmentations.apply_augmentations(x, x_lengths, aug_strength=aug_strength)
                views.append(view)
                views_lens.append(x_lengths_tensor)  # Use tensor instead of list
        
        return views, views_lens
    
    def _compute_simclr_loss(self, features_list):
        """
        Compute SimCLR contrastive loss.
        DataParallel automatically handles cross-GPU features.
        """
        # Concatenate all views
        all_features = torch.cat(features_list, dim=0)  # [num_views * batch_size, proj_dim]
        
        # Normalize features
        all_features = F.normalize(all_features, dim=-1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(all_features, all_features.T) / self.temperature
        
        # Create labels for positive pairs
        batch_size = len(features_list[0])
        labels = torch.arange(len(features_list)).repeat_interleave(batch_size)
        labels = labels.to(all_features.device)
        
        # Create mask to remove self-similarities
        mask = torch.eye(len(all_features), device=all_features.device).bool()
        similarity_matrix.masked_fill_(mask, float('-inf'))
        
        # Compute InfoNCE loss
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss
    
    def forward(self, x, x_lengths):
        """Forward pass with DataParallel handling"""
        # Convert x_lengths to list if it's a tensor
        if isinstance(x_lengths, torch.Tensor):
            x_lengths = x_lengths.tolist()
        
        # Ensure input is on primary device
        x = x.to(self.primary_device)
        
        # Create views
        views, views_lens = self._create_views(x, x_lengths)
        
        # Process through encoder and projection head (DataParallel handles multi-GPU)
        features_list = []
        for view, view_len in zip(views, views_lens):
            # view = view.to(self.primary_device)
            # Get features from encoder
            feat = self.encoder(view, view_len)
            # Project features
            proj_feat = self.projection_head(feat)
            features_list.append(proj_feat)
        
        # Compute SimCLR loss
        loss = self._compute_simclr_loss(features_list)
        
        return {
            "loss": loss,
            "features": features_list
        }


def create_dataparallel_pretrainer(pretrainer_type: str, base_model: nn.Module, device_ids: List[int] = None, **kwargs):
    """
    Factory function to create DataParallel pretrainers.
    
    Args:
        pretrainer_type: 'dino' or 'simclr'
        base_model: The base model to be pretrained
        device_ids: List of GPU IDs to use [0, 1, 2, 3] for 4 GPUs
        **kwargs: Additional arguments for the pretrainer
    """
    if device_ids is None:
        device_ids = list(range(torch.cuda.device_count()))
    
    if pretrainer_type.lower() == 'dino':
        return DataParallelDINOPretrainer(base_model, device_ids=device_ids, **kwargs)
    elif pretrainer_type.lower() == 'simclr':
        return DataParallelSimCLRPretrainer(base_model, device_ids=device_ids, **kwargs)
    else:
        raise ValueError(f"Unknown pretrainer type: {pretrainer_type}")