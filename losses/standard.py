# -*- coding: utf-8 -*-
"""
Split from DL_helpers.py - Modular code structure
@author: MBoukhec (original)
"""

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, verbose=False, restore_best_weights=True):
        """
        Early stopping to halt training when validation loss does not improve.
        
        Parameters:
        patience (int): How many epochs to wait after the last improvement before stopping.
        min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        verbose (bool): Whether to print messages when the validation loss improves or not.
        restore_best_weights (bool): Whether to restore model weights from the epoch with the best validation loss.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_weights = None
    
    def __call__(self, val_loss, model):
        # If this is the first validation loss, initialize best_score
        if self.best_score is None:
            self.best_score = val_loss
            self.best_weights = model.state_dict()
            return False
        
        # If the validation loss improved, reset counter and update best score
        if val_loss < self.best_score - self.min_delta:
            self.best_score = val_loss
            self.best_weights = model.state_dict()
            self.counter = 0
            if self.verbose:
                print(f"Validation loss improved: {val_loss:.6f}")
            return False
        
        # If no improvement, increment the counter
        self.counter += 1
        if self.verbose:
            print(f"Validation loss did not improve for {self.counter} epochs.")
        
        # If patience is exceeded, return True to stop training
        if self.counter >= self.patience:
            self.early_stop = True
            return True
        
        return False
    
    def restore(self, model):
        """Restore model weights from the best epoch."""
        if self.restore_best_weights:
            model.load_state_dict(self.best_weights)
            
            

class Retraining:
    def __init__(self, verbose=False):
        """
        Early stopping to halt training when validation loss does not improve.
        
        Parameters:
        patience (int): How many epochs to wait after the last improvement before stopping.
        min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        verbose (bool): Whether to print messages when the validation loss improves or not.
        restore_best_weights (bool): Whether to restore model weights from the epoch with the best validation loss.
        """
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_weights = None
        self.best_index = None
    
    def check_performance(self, val_loss, model):
        # If this is the first validation loss, initialize best_score
        self.counter += 1
        if self.best_score is None:
            self.best_score = val_loss
            self.best_weights = model.state_dict()
            self.best_index = self.counter
        else:
            # If the validation loss improved, reset counter and update best score
            if val_loss < self.best_score :
                self.best_score = val_loss
                self.best_weights = model.state_dict()
                self.best_index = self.counter 
                if self.verbose:
                    print(f"Validation loss improved. Best model is from training {self.best_index}")
            else:
                # If no improvement, increment the counter
                if self.verbose:
                    print(f"Validation loss did not improve. Best model is from training {self.best_index}")   
    
    def restore(self, model):
        """Restore model weights from the best epoch."""
        if self.verbose:
            print(f"Best model restored from training {self.best_index}")
        model.load_state_dict(self.best_weights)
        return model
    

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = (self.alpha[targets] * (1 - pt) ** self.gamma * ce_loss).mean()
        return loss


    

class GCELoss(nn.Module):
    def __init__(self, num_classes=2, q=0.9):
        super(GCELoss, self).__init__()
        self.q = q
        self.num_classes = num_classes
        self.eps = 1e-9  # Define eps here

    def forward(self, pred, labels):
        # Convert logits to probabilities
        pred = torch.sigmoid(pred)  # Use sigmoid for binary classification
        pred = torch.stack([1 - pred, pred], dim=1)  # Shape [batch_size, 2]
        pred = torch.clamp(pred, min=self.eps, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        loss = (1. - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.q)) / self.q
        return loss.mean()



class GCELoss2(nn.Module):
    def __init__(self, num_classes=2, q=0.9):
        super(GCELoss2, self).__init__()
        self.q = q
        self.num_classes = num_classes
        self.eps = 1e-9  # Define eps here

    def forward(self, pred, labels):
        # Convert logits to probabilities
        pred = torch.sigmoid(pred)  # Use sigmoid for binary classification
        pred = torch.stack([1 - pred, pred], dim=1)  # Shape [batch_size, 2]
        pred = torch.clamp(pred, min=self.eps, max=1.0)
        #label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        label_one_hot = pred
        loss = (1. - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.q)) / self.q
        return loss


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor
    

def measure_loss(outputs,batch_labels,pos_weight):
    """
    Measure loss without reduction and mask the padded values to prevent loss calculation from padded values.
    """
    #loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight,reduction='none')(outputs, batch_labels.view(-1).float())
    criterion = GCELoss2(q=0.9)
    loss = criterion(outputs, batch_labels.view(-1).float())
    loss_mask = batch_labels != 2
    loss_masked = loss.where(loss_mask, torch.tensor(0.0))
    loss= loss_masked.sum() / loss_mask.sum()  # tensor(0.3802)
    return loss

class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor
    

def measure_loss_pretrain(outputs1,outputs2,outputs3,label1,label2,label3,wl1=1.0, wl2=1, wl3=0.1, pos_weight_l1 = False,pos_weight_l2 = False):
    mask = (label1 != -999) #detect the padded values    
    
    label1,label2,label3 = label1[mask],label2[mask],label3[mask]
    
    loss1 = F.mse_loss(outputs1,label1.float()) 
    loss2 = F.mse_loss(outputs2,label2.float())
    loss3 = F.mse_loss(outputs3,label3.float())
    
    
    return loss1+loss2+loss3,loss1,loss2,loss3,label1,label2,label3



def measure_loss_multitask(outputs1,
                           outputs2,
                           outputs3,
                           label1,
                           label2,
                           label3,
                           wl1=1.0, 
                           wl2=1, 
                           wl3=0.1, 
                           contrastive_embedding=None,
                           mixup_label1=None,
                           mixup_label3=None,
                           mixup_lambda=1.0,
                           pos_weight_l1=False,
                           pos_weight_l2=False, 
                           ):
    """
    Measure loss without reduction and mask the padded values to prevent loss calculation from padded values.
    """
    weight1, weight2 = None, None
    
    if pos_weight_l1:
        weight1 = torch.ones([len(label1)],device=outputs1.device) * 3

    if mixup_label1 is not None and mixup_lambda < 1.0:
        loss1_orig = nn.BCEWithLogitsLoss(pos_weight=weight1, reduction='none')(outputs1, label1.float())
        loss1_mixup = nn.BCEWithLogitsLoss(pos_weight=weight1, reduction='none')(outputs1, mixup_label1.float())
        loss1 = (mixup_lambda * loss1_orig + (1 - mixup_lambda) * loss1_mixup).mean()
    else:
        loss1 = nn.BCEWithLogitsLoss(pos_weight=weight1)(outputs1, label1.float())

    if contrastive_embedding is not None:
        supcon_loss = SupConLoss(temperature=0.07)(contrastive_embedding, label1.float())
        loss1 = 0.5*loss1 + 0.5 * supcon_loss
    
    if not pos_weight_l2:
        weight2 = torch.ones([len(label2)],device=outputs1.device)*10
    loss2 = nn.BCEWithLogitsLoss(pos_weight=weight2)(outputs2,label2.view(-1).float())

    if mixup_label1 is not None and mixup_lambda < 1.0:
        loss3_orig = F.mse_loss(outputs3, label3.float())
        loss3_mixup = F.mse_loss(outputs3, mixup_label3.float())
        loss3 = (mixup_lambda * loss3_orig + (1 - mixup_lambda) * loss3_mixup).mean()
    else:
        loss3 = F.mse_loss(outputs3, label3.float())
    return wl1*loss1+wl2*loss2+wl3*loss3, loss1, loss2, loss3



def measure_loss_reg(outputs3, label1, label3, wl1=1.0, wl3=1.0, pos_thres=1.0):
    # measure classification and regression losses. The classification labels are generated from regression outputs. 
    loss3 = F.mse_loss(outputs3, label3.float())
    # if outputs3 > pos_thres, treate it as positive
    outputs1 = (outputs3 > pos_thres).float()
    loss1 = nn.BCEWithLogitsLoss()(outputs1, label1.float())
    # loss1_orig = nn.BCEWithLogitsLoss(reduction='none')(outputs1, label1.float())
    # loss1_mixup = nn.BCEWithLogitsLoss(reduction='none')(outputs1, mixup_label1.float())
    # loss1 = (mixup_lambda * loss1_orig + (1 - mixup_lambda) * loss1_mixup).mean()
    loss = wl1 * loss1 + loss3 * wl3
    return loss, loss1, loss3



def measure_loss_cls(outputs1,label1,pos_weight_l1=False):
    """
    Measure loss without reduction and mask the padded values to prevent loss calculation from padded values.
    """
    weight1 = torch.ones([len(label1)],device=outputs1.device)*3

    if pos_weight_l1:
        loss1 = nn.BCEWithLogitsLoss(pos_weight=weight1)(outputs1,label1.float()) #pos_weight=weight1
    else:
        loss1 = nn.BCEWithLogitsLoss()(outputs1,label1.float()) #pos_weight=weight1

    return loss1



def measure_loss_tso(outputs, labels, x_lengths):
    """
    Measure loss for TSO prediction. Supports binary (output_channels=1) and
    3-class (output_channels=3) modes based on output shape.

    Binary mode: maps labels to TSO (class 2) -> 1, other/non-wear -> 0, uses BCE.
    3-class mode: uses cross-entropy with ignore_index=-100 for padding.

    Args:
        outputs: [batch_size, seq_len, C] - model predictions (logits), C=1 or 3
        labels: [batch_size, seq_len] - ground truth class indices (0/1/2, -100=padding)
        x_lengths: [batch_size] - original sequence lengths before padding

    Returns:
        total_loss: scalar tensor
    """
    batch_size, seq_len, num_classes = outputs.size()

    if num_classes == 1:
        # Binary mode: TSO (2) -> 1, everything else -> 0
        ignore_mask = (labels == -100)
        binary_labels = (labels == 2).float()
        outputs_flat = outputs.reshape(-1)
        labels_flat = binary_labels.reshape(-1)
        ignore_flat = ignore_mask.reshape(-1)
        valid_labels = labels_flat[~ignore_flat]
        # Compute pos_weight from this batch to counteract class imbalance.
        # pos_weight = #negatives / #positives (clamped to [1, 50] for stability).
        n_pos = valid_labels.sum().clamp(min=1)
        n_neg = (1 - valid_labels).sum().clamp(min=1)
        pos_weight = (n_neg / n_pos).clamp(max=50.0)
        loss_per_token = F.binary_cross_entropy_with_logits(
            outputs_flat, labels_flat, reduction='none',
            pos_weight=pos_weight.expand_as(outputs_flat)
        )
        loss_per_token = loss_per_token * (~ignore_flat).float()
        return loss_per_token.sum() / (~ignore_flat).float().sum().clamp(min=1)
    else:
        # 3-class mode: cross-entropy, padding positions (label=-100) are ignored
        outputs_flat = outputs.reshape(-1, num_classes)
        labels_flat = labels.reshape(-1)
        return F.cross_entropy(outputs_flat, labels_flat, ignore_index=-100)



def tso_continuity_loss(outputs, x_lengths, alpha=0.1):
    """
    Continuity regularization for TSO predictions to encourage single, continuous TSO period.

    Penalizes fragmented TSO predictions by measuring the total variation (switches)
    in TSO probability over time. Encourages smooth, continuous TSO segments.

    OPTIMIZED: Fully vectorized implementation for GPU efficiency.

    Optimizations:
    1. Fully vectorized - no Python loops
    2. Uses masking instead of slicing for valid sequences
    3. Single GPU kernel for all operations
    4. Eliminates intermediate list allocations

    Args:
        outputs: [batch_size, seq_len, 3] - model predictions (logits)
        x_lengths: [batch_size] - valid sequence lengths
        alpha: Weight for continuity loss (default: 0.1)

    Returns:
        continuity_loss: Scalar tensor - penalty for fragmented TSO predictions

    Intuition:
        - If TSO probability smoothly goes 0 -> 0.8 -> 1.0 -> 0.8 -> 0: Low loss (continuous)
        - If TSO probability jumps 0 -> 1 -> 0 -> 1 -> 0: High loss (fragmented)
    """
    batch_size, seq_len, _ = outputs.shape
    device = outputs.device

    # Get TSO probabilities
    num_classes = outputs.shape[-1]
    if num_classes == 1:
        tso_probs = torch.sigmoid(outputs[:, :, 0])   # [batch_size, seq_len]
    else:
        tso_probs = torch.softmax(outputs, dim=-1)[:, :, 2]  # [batch_size, seq_len]

    # Create mask for valid positions: [batch_size, seq_len]
    positions = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, seq_len]
    valid_mask = positions < x_lengths.unsqueeze(1)  # [batch_size, seq_len]

    # Create mask for valid transitions (exclude last timestep and invalid regions)
    # Transition from t to t+1 is valid if both t and t+1 are valid
    valid_transition_mask = valid_mask[:, :-1] & valid_mask[:, 1:]  # [batch_size, seq_len-1]

    # Calculate differences between consecutive timesteps (vectorized)
    diffs = torch.abs(tso_probs[:, 1:] - tso_probs[:, :-1])  # [batch_size, seq_len-1]

    # Apply mask and sum per sequence
    masked_diffs = diffs * valid_transition_mask  # Zero out invalid transitions
    switches_per_seq = masked_diffs.sum(dim=1)  # [batch_size]

    # Normalize by valid transition count (valid_len - 1)
    # Clamp to avoid division by zero for sequences with length <= 1
    valid_transition_counts = valid_transition_mask.sum(dim=1).clamp(min=1)  # [batch_size]
    normalized_switches = switches_per_seq / valid_transition_counts  # [batch_size]

    # Return mean across batch
    return alpha * normalized_switches.mean()



def measure_loss_tso_with_continuity(outputs, labels, x_lengths, continuity_weight=0.1):
    """
    TSO loss with continuity regularization for single-period prediction.

    Combines standard cross-entropy loss with continuity regularization to encourage
    the model to predict single, continuous TSO periods rather than fragmented periods.

    Args:
        outputs: [batch_size, seq_len, 3] - model predictions (logits)
        labels: [batch_size, seq_len] - ground truth labels
        x_lengths: [batch_size] - valid sequence lengths
        continuity_weight: Weight for continuity loss (default: 0.1)
                          Higher = stronger enforcement of continuity
                          0.0 = no continuity regularization (standard CE only)

    Returns:
        total_loss: Combined classification + continuity loss
        class_loss: Standard cross-entropy loss
        cont_loss: Continuity regularization loss
    """
    # Standard classification loss
    class_loss = measure_loss_tso(outputs, labels, x_lengths)

    # Continuity regularization (encourage smooth, continuous TSO predictions)
    if continuity_weight > 0:
        cont_loss = tso_continuity_loss(outputs, x_lengths, alpha=1.0)
        total_loss = class_loss + continuity_weight * cont_loss
    else:
        cont_loss = torch.tensor(0.0, device=outputs.device)
        total_loss = class_loss

    return total_loss, class_loss, cont_loss



class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss


class SupConLossV2(nn.Module):
    """Supervised Contrastive Learning loss"""
    # No multi-view requirement. Contrast between different samples in the batch
    # Fixed implementation based on https://github.com/HobbitLong/SupContrast/blob/master/losses.py
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, features, labels):
        """
        Args:
            features: hidden vector of shape [batch_size, embed_dim]
            labels: ground truth of shape [batch_size]
        """
        batch_size = features.shape[0]
        if batch_size < 2:
            return torch.tensor(0.0, device=features.device, requires_grad=True)
            
        labels = labels.contiguous().view(-1, 1)
        
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Mask of positive pairs (same class, excluding self)
        mask = torch.eq(labels, labels.T).float()
        
        # Compute similarity matrix
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)
        
        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # Create mask to exclude self-contrasts (diagonal)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size, device=features.device).view(-1, 1),
            0
        )
        
        # Mask out self-contrast cases  
        mask = mask * logits_mask
        
        # Compute log_prob: log(exp(sim_i_j) / sum_k(exp(sim_i_k))) where k != i
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        
        # Compute mean of log-likelihood over positive pairs
        # Only consider samples that have positive pairs
        mask_sum = mask.sum(1)
        valid_samples = mask_sum > 0
        
        if valid_samples.sum() == 0:
            return torch.tensor(0.0, device=features.device, requires_grad=True)
        
        mean_log_prob_pos = (mask * log_prob).sum(1)[valid_samples] / mask_sum[valid_samples]
        
        # Loss: negative mean of log-likelihood
        loss = -mean_log_prob_pos.mean()
        return loss


def measure_loss_multitask_with_padding(outputs1,
                           outputs2,
                           outputs3,
                           label1,
                           label2,
                           label3,
                           wl1=1.0, 
                           wl2=1, 
                           wl3=0.1, 
                           contrastive_embedding=None,
                           mixup_label1=None,
                           mixup_label3=None,
                           mixup_lambda=1.0,
                           padding_position="tail",
                           pos_weight_l1=False,
                           pos_weight_l2=False,
                           x_lengths=[],
                           seq_start_idx=[],
                           average_mask=False,
                           average_window_size=20,
                           ignore_padding_in_mask_loss=True):
    """
    Measure loss without reduction and mask the padded values to prevent loss calculation from padded values.
    
    Args:
        outputs1, outputs2, outputs3: Model outputs for the three tasks
        label1, label2, label3: Ground truth labels
        wl1, wl2, wl3: Loss weights for the three tasks
        padding_position: "tail" or "random" - specifies where padding was applied
        pos_weight_l1, pos_weight_l2: Whether to use positive class weighting
        x_lengths: Original sequence lengths (before padding)
        seq_start_idx: Start indices for sequences (when using random padding)
        average_mask: Whether to apply window averaging to downsample masks
        average_window_size: Size of window for averaging
        ignore_padding_in_loss: Whether to ignore padding areas when calculating loss2
    """
    weight1, weight2 = None, None
    if pos_weight_l1:
        weight1 = torch.ones([len(label1)], device=outputs1.device) * 2

    # Task 1: Binary classification loss
    if mixup_label1 is not None and mixup_lambda < 1.0:
        loss1_orig = nn.BCEWithLogitsLoss(pos_weight=weight1, reduction='none')(outputs1, label1.float())
        loss1_mixup = nn.BCEWithLogitsLoss(pos_weight=weight1, reduction='none')(outputs1, mixup_label1.float())
        loss1 = (mixup_lambda * loss1_orig + (1 - mixup_lambda) * loss1_mixup).mean()
    else:
        loss1 = nn.BCEWithLogitsLoss(pos_weight=weight1)(outputs1, label1.float())

    if contrastive_embedding is not None:
        supcon_loss = SupConLossV2(temperature=0.07)(contrastive_embedding, label1.float())
        # loss1 += supcon_loss
        # loss1 = supcon_loss
        loss1 = 0.5 * loss1 + 0.5 * supcon_loss
        

    # Handle mask averaging if enabled
    if average_mask:
        B, L = label2.shape
        target_length = L // average_window_size

        # Use F.interpolate with nearest mode for binary labels (better than averaging + threshold)
        label2_float = label2.float().unsqueeze(1)  # Add channel dimension: [B, 1, L]
        label2_interpolated = F.interpolate(label2_float, size=target_length, mode='nearest')
        label2 = label2_interpolated.squeeze(1).long()  # Remove channel dim and convert back to long

        # CRITICAL: Recalculate x_lengths and seq_start_idx for the new target_length
        # This is essential for proper mask calculation after averaging
        if len(x_lengths) > 0:
            # Simply divide by average_window_size
            x_lengths = [max(1, length // average_window_size) for length in x_lengths]

            # Scale seq_start_idx by dividing by average_window_size
            if len(seq_start_idx) > 0:
                seq_start_idx = [idx // average_window_size for idx in seq_start_idx]
    # Task 2: Binary sequence classification loss
    if pos_weight_l2:
        weight2 = torch.ones([label2.shape[1]], device=outputs2.device) * 10
    
    # Create padding mask if needed
    if ignore_padding_in_mask_loss:
        B, L = label2.shape
        non_padding = torch.zeros((B, L), dtype=torch.bool, device=outputs2.device)
        
        # Fill mask based on padding position
        if padding_position == "tail":
            # For tail padding, only the first x_lengths elements are valid
            for i in range(len(x_lengths)):
                non_padding[i, :x_lengths[i]] = True
        else:  # "random" padding
            # For random padding, valid elements depend on start indices
            for i in range(len(x_lengths)):
                non_padding[i, seq_start_idx[i]:seq_start_idx[i]+x_lengths[i]] = True
        
        # Calculate loss with reduction='none' to apply mask
        element_losses = nn.BCEWithLogitsLoss(pos_weight=weight2, reduction='none')(outputs2, label2.float())
        
        # Apply mask and calculate mean over non-padded elements only
        loss2 = (element_losses * non_padding.float()).sum() / non_padding.sum()
    else:
        # Standard loss calculation without ignoring padding
        loss2 = nn.BCEWithLogitsLoss(pos_weight=weight2)(outputs2, label2.float())
    
    # Task 3: Regression loss for scratch duration
    if mixup_label1 is not None and mixup_lambda < 1.0:
        loss3_orig = F.mse_loss(outputs3, label3.float())
        loss3_mixup = F.mse_loss(outputs3, mixup_label3.float())
        loss3 = (mixup_lambda * loss3_orig + (1 - mixup_lambda) * loss3_mixup).mean()
    else:
        loss3 = F.mse_loss(outputs3, label3.float())

    # Extract unpadded areas for evaluation metrics
    if ignore_padding_in_mask_loss:
        if padding_position == "tail":
            # For tail padding, just take the first x_lengths elements
            unpad_outputs2 = [
                outputs2[i, :x_lengths[i]]
                for i in range(len(x_lengths))
            ]
            unpad_label2 = [
                label2[i, :x_lengths[i]]
                for i in range(len(x_lengths))
            ]
        else:  # "random" padding
            # For random padding, use the sequence start indices
            unpad_outputs2 = [
                outputs2[i, seq_start_idx[i]:seq_start_idx[i]+x_lengths[i]]
                for i in range(len(x_lengths))
            ]
            unpad_label2 = [
                label2[i, seq_start_idx[i]:seq_start_idx[i]+x_lengths[i]]
                for i in range(len(x_lengths))
            ]

        # Concatenate the unpadded sequences
        unpad_outputs2 = torch.cat(unpad_outputs2, dim=0)
        unpad_label2 = torch.cat(unpad_label2, dim=0)
    else:
        # Fallback: if no x_lengths provided, just flatten everything
        unpad_outputs2 = outputs2.view(-1)
        unpad_label2 = label2.view(-1)
    
    # Calculate weighted total loss
    total_loss = wl1*loss1 + wl2*loss2 + wl3*loss3
    # total_loss = loss1
    # if torch.isnan(total_loss):
    #     print(f"loss1: {loss1.item()}, loss2: {loss2.item()}, loss3: {loss3.item()}")
    
    return total_loss, loss1, loss2, loss3, unpad_outputs2, unpad_label2



