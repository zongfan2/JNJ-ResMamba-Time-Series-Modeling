# -*- coding: utf-8 -*-
"""
Losses module - Loss functions and training utilities
"""

from .standard import (
    FocalLoss,
    GCELoss,
    GCELoss2,
    SupConLoss,
    SupConLossV2,
    EarlyStopper,
    EarlyStopping,
    Retraining,
    CosineWarmupScheduler,
    measure_loss,
    measure_loss_pretrain,
    measure_loss_multitask,
    measure_loss_multitask_with_padding,
    measure_loss_cls,
    measure_loss_reg,
    measure_loss_tso,
    measure_loss_tso_with_continuity,
    tso_continuity_loss,
)

from . import dlrtc
from . import structural_priors
from .structural_priors import (
    transition_count_loss,
    duration_prior_loss,
    compute_boundary_weights,
    boundary_reweighted_ce_loss,
    ELRMemory,
    elr_loss,
    CircadianPriorBias,
    hour_from_time_channels,
    measure_loss_tso_structural,
)

__all__ = [
    # Classes
    'FocalLoss',
    'GCELoss',
    'GCELoss2',
    'SupConLoss',
    'SupConLossV2',
    'EarlyStopper',
    'EarlyStopping',
    'Retraining',
    'CosineWarmupScheduler',
    # Functions
    'measure_loss',
    'measure_loss_pretrain',
    'measure_loss_multitask',
    'measure_loss_multitask_with_padding',
    'measure_loss_cls',
    'measure_loss_reg',
    'measure_loss_tso',
    'measure_loss_tso_with_continuity',
    'tso_continuity_loss',
    # Structural priors (Deep TSO)
    'transition_count_loss',
    'duration_prior_loss',
    'compute_boundary_weights',
    'boundary_reweighted_ce_loss',
    'ELRMemory',
    'elr_loss',
    'CircadianPriorBias',
    'hour_from_time_channels',
    'measure_loss_tso_structural',
    # Modules
    'dlrtc',
    'structural_priors',
]
