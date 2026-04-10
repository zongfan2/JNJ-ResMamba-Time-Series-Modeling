#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Self-supervised pretraining on UKB accelerometer data for Deep Scratch.

Loads UKB H5 (produced by data/preprocess_ukb_h5.py), creates an
MBA_v1_ForPretraining encoder, wraps it in DINOPretrainer, and trains.

Saves encoder weights that can be directly loaded into MBA_v1 for finetuning.

Improvements over base DINO implementation:
  - Teacher EMA momentum cosine schedule (0.996 → 1.0) per original DINO paper
  - AMP (mixed precision) for ~2x speedup and lower memory
  - Cosine LR decay with linear warmup (not OneCycleLR)
  - Weight decay excluded from BatchNorm / bias params
  - Larger projection head (hidden=512, out=256)
  - Rolling checkpoint: saves every 10 epochs, keeps latest 3

Usage (Domino):
    python3.11 /mnt/code/munge/predictive_modeling/code/JNJ-ResMamba-Time-Series-Modeling/training/pretrain_ukb.py \
        --config experiments/configs/pretrain_ukb_dino.yaml
"""

import subprocess
import os
import sys

# ---------------------------------------------------------------------------
# Project root on sys.path so that `from models import ...` works from Domino
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

import joblib

shell_script = '''
cd munge/predictive_modeling
sudo python3.11 -m pip install -r requirement-ml.txt
sudo python3.11 -m pip install -e .
sudo python3.11 -m pip install optuna==4.3.0 seaborn ray TensorboardX torcheval ruptures mamba-ssm[causal-conv1d]==2.2.2
'''
result = subprocess.run(shell_script, shell=True, capture_output=True, text=True)

import argparse
import gc
import math
import yaml
import shutil
import glob as globmod
from datetime import datetime
from pprint import pprint

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from models.resmamba import MBA_v1_ForPretraining
from models.setup import setup_model
from models.pretrainer import DINOPretrainer
from models.dataparallel_pretrainer import DataParallelDINOPretrainer
from data.loading_ukb_h5 import (
    UKBPretrainDataset,
    make_ukb_collate_fn,
    read_ukb_h5_metadata,
    list_ukb_segment_indices,
    split_indices_by_segment,
)
from torch.utils.data import DataLoader
from losses.standard import EarlyStopping

torch.cuda.empty_cache()
gc.collect()


# ============================================================================
# Argument parsing with YAML config support
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='UKB Self-Supervised Pretraining')
    parser.add_argument('--config', type=str, default='',
                        help='Path to YAML config file')
    # Data
    parser.add_argument('--ukb_h5_path', type=str, default='',
                        help='Path to UKB H5 file (from preprocess_ukb_h5.py)')
    parser.add_argument('--max_segments', type=int, default=0,
                        help='Max segments to load (0 = all)')
    parser.add_argument('--max_seq_len', type=int, default=1221,
                        help='Cap sequence length for padding (default 1221 = 61s at 20Hz, '
                             'matching production). Segments longer than this are truncated.')
    parser.add_argument('--scaler_path', type=str, default='',
                        help='Path to a pre-fitted StandardScaler (.joblib/.bin). '
                             'If provided, UKB data is transformed with this scaler '
                             'instead of fitting a fresh one — use the production JNJ '
                             'scaler here so the pretrained encoder learns in the same '
                             'input distribution JNJ finetuning will use.')
    # Model
    parser.add_argument('--model', type=str, default='mbav1_pretrain',
                        help='Model architecture for pretraining')
    parser.add_argument('--pretraining_method', type=str, default='DINO',
                        help='Pretraining method: DINO, SimCLR')
    # Training
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--optim', type=str, default='AdamW')
    parser.add_argument('--weight_decay', type=float, default=0.04,
                        help='Weight decay (excluded from BN/bias)')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')
    parser.add_argument('--gradient_clip', type=float, default=1.0)
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Linear warmup epochs for LR scheduler')
    parser.add_argument('--use_amp', type=str, default='True',
                        help='Use automatic mixed precision')
    # DINO-specific
    parser.add_argument('--momentum_start', type=float, default=0.996,
                        help='Teacher EMA momentum start')
    parser.add_argument('--momentum_end', type=float, default=1.0,
                        help='Teacher EMA momentum end (cosine schedule)')
    parser.add_argument('--projection_dim', type=int, default=256,
                        help='DINO projection head output dim')
    parser.add_argument('--projection_hidden', type=int, default=512,
                        help='DINO projection head hidden dim')
    # Checkpoint
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--keep_last_n', type=int, default=3,
                        help='Keep only the last N intermediate checkpoints')
    # GPU
    parser.add_argument('--num_gpu', type=str, default='NA')
    # DataLoader
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of DataLoader worker processes for H5 reads. '
                             'Each worker opens its own h5py handle.')
    parser.add_argument('--pin_memory', type=str, default='True',
                        help='Pin host memory for faster GPU transfer.')
    parser.add_argument('--persistent_workers', type=str, default='True',
                        help='Keep DataLoader workers alive across epochs '
                             '(avoids re-opening H5 every epoch).')
    # Output
    parser.add_argument('--output_dir', type=str, default='',
                        help='Output directory for weights and logs')
    parser.add_argument('--clear_tracker', type=str, default='False')
    return parser.parse_args()


def merge_yaml_config(args, raw_cli_args=None):
    """Merge YAML config into args.

    Priority: CLI explicitly provided > YAML > argparse defaults.
    We compare against argparse defaults to detect which args the user
    actually typed on the command line vs. left at their default.
    """
    if not args.config or not os.path.exists(args.config):
        return args
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f) or {}
    # Flatten nested YAML sections
    flat = {}
    for key, val in cfg.items():
        if isinstance(val, dict):
            flat.update(val)
        else:
            flat[key] = val

    # Build set of arg names that the user explicitly passed on the CLI
    cli_provided = set()
    if raw_cli_args is not None:
        for tok in raw_cli_args:
            if tok.startswith('--'):
                cli_provided.add(tok.lstrip('-').replace('-', '_'))

    for key, val in flat.items():
        key_under = key.replace('-', '_')
        if hasattr(args, key_under):
            # Only skip override if the user explicitly typed this arg on CLI
            if key_under not in cli_provided:
                setattr(args, key_under, val)
    return args


# ============================================================================
# GPU config
# ============================================================================

def parse_gpu_config(num_gpu_arg):
    if num_gpu_arg == "NA":
        return False, None, "cuda:0"
    if "," in num_gpu_arg:
        device_ids = [int(x.strip()) for x in num_gpu_arg.split(",")]
        if len(device_ids) > 1:
            return True, device_ids, f"cuda:{device_ids[0]}"
        return False, None, f"cuda:{device_ids[0]}"
    try:
        num = int(num_gpu_arg)
        if num <= 1:
            return False, None, f"cuda:{max(0, num)}"
        device_ids = list(range(num))
        return True, device_ids, "cuda:0"
    except ValueError:
        return False, None, "cuda:0"


# ============================================================================
# Teacher EMA momentum schedule (DINO paper: cosine from m_start → m_end)
# ============================================================================

def get_momentum_schedule(epoch, total_epochs, m_start=0.996, m_end=1.0):
    """Cosine schedule for teacher EMA momentum, per DINO paper."""
    return m_end - (m_end - m_start) * (math.cos(math.pi * epoch / total_epochs) + 1) / 2


# ============================================================================
# Weight-decay param groups (exclude BN and bias)
# ============================================================================

def get_param_groups(model, weight_decay):
    """Separate params into decay / no-decay groups.
    BatchNorm params and bias terms should not have weight decay."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'norm' in name or 'bn' in name or 'bias' in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0},
    ]


# ============================================================================
# Cosine LR scheduler with linear warmup
# ============================================================================

class CosineWithWarmup(torch.optim.lr_scheduler._LRScheduler):
    """Cosine annealing with linear warmup, per-step."""

    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6,
                 last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            scale = step / max(1, self.warmup_steps)
        else:
            progress = (step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps)
            scale = max(0.0,
                        0.5 * (1.0 + math.cos(math.pi * progress)))
        return [
            max(self.min_lr, base_lr * scale) for base_lr in self.base_lrs
        ]


# ============================================================================
# Contrastive pretraining loop (reusable for DINO / SimCLR)
# ============================================================================

def contrastive_pretrain_epoch(pretrainer, loader, train_mode, device,
                               optimizer, scheduler, max_seq_len=None,
                               gradient_clip=1.0, scaler=None):
    """Run one epoch of contrastive pretraining with optional AMP.

    Consumes an already-built ``torch.utils.data.DataLoader`` that yields
    ``(batch_data, x_lens)`` pairs from ``UKBPretrainDataset`` +
    ``collate_ukb_batch``. All segment grouping and padding happens inside
    the DataLoader workers — this loop only moves tensors to GPU and runs
    the forward/backward pass.
    """
    epoch_loss = 0.0
    batches = 0
    use_amp = scaler is not None

    for batch_data, x_lens in loader:
        # Collate runs on CPU (in workers); move to GPU here.
        batch_data = batch_data.to(device, non_blocking=True)
        x_lens = x_lens.to(device, non_blocking=True)

        # Defensive cap — collate already truncated, but guard against a
        # mis-configured collate_fn being passed in.
        if max_seq_len is not None and batch_data.size(1) > max_seq_len:
            batch_data = batch_data[:, :max_seq_len, :]
            x_lens = torch.clamp(x_lens, max=max_seq_len)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)  # slightly faster than zero_grad()

        # Forward with optional AMP
        if use_amp and train_mode:
            with torch.cuda.amp.autocast():
                res = pretrainer(batch_data, x_lens)
                loss = res["loss"]
        else:
            res = pretrainer(batch_data, x_lens)
            loss = res["loss"]

        if train_mode:
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  Warning: NaN/Inf loss, skipping batch.")
                continue

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(pretrainer.parameters(),
                                               max_norm=gradient_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(pretrainer.parameters(),
                                               max_norm=gradient_clip)
                optimizer.step()

            scheduler.step()

        if not (torch.isnan(loss) or torch.isinf(loss)):
            epoch_loss += loss.item()
        batches += 1

    avg_loss = epoch_loss / max(batches, 1)
    return avg_loss, batches


# ============================================================================
# Rolling checkpoint: save every N epochs, keep latest K
# ============================================================================

def save_rolling_checkpoint(state_dict, weights_dir, method, epoch,
                            keep_last_n=3):
    """Save checkpoint and remove old ones beyond keep_last_n."""
    ckpt_path = os.path.join(weights_dir,
                             f'{method}_epoch{epoch}.pth')
    torch.save(state_dict, ckpt_path)
    print(f"  Saved checkpoint: {ckpt_path}")

    # Find all intermediate checkpoints (excluding _best and _final)
    pattern = os.path.join(weights_dir, f'{method}_epoch*.pth')
    existing = sorted(globmod.glob(pattern),
                      key=os.path.getmtime)

    # Remove oldest beyond keep_last_n
    while len(existing) > keep_last_n:
        old = existing.pop(0)
        os.remove(old)
        print(f"  Removed old checkpoint: {os.path.basename(old)}")


# ============================================================================
# Weight transfer: pretrained encoder → MBA_v1
# ============================================================================

def load_pretrained_encoder_into_mbav1(mbav1_model, pretrain_ckpt_path,
                                       device=None):
    """
    Load pretrained encoder weights from a DINO pretraining checkpoint into
    an MBA_v1 model.  The checkpoint stores the full DINOPretrainer state_dict
    so we extract the student (= MBA_v1_ForPretraining) sub-keys.

    Matching keys:  input_projection.*, cls_token, encoder.*, positional_encoding.*
    """
    if device is None:
        device = torch.device('cpu')
    ckpt = torch.load(pretrain_ckpt_path, map_location=device)

    # Handle full DINOPretrainer state_dict
    # Keys may be 'student.module.*' (DataParallel) or 'student.*' (single-GPU)
    encoder_state = {}
    for key, val in ckpt.items():
        if key.startswith('student.module.'):
            encoder_state[key[len('student.module.'):]] = val
        elif key.startswith('student.'):
            encoder_state[key[len('student.'):]] = val

    if not encoder_state:
        # Fallback: already a clean encoder dict (e.g. _encoder_only.pth)
        encoder_state = ckpt

    mbav1_keys = set(mbav1_model.state_dict().keys())
    matched = {k: v for k, v in encoder_state.items() if k in mbav1_keys}

    missing, unexpected = mbav1_model.load_state_dict(matched, strict=False)
    print(f"Loaded {len(matched)} pretrained encoder parameters into MBA_v1")
    if missing:
        print(f"  Missing (randomly initialised): {len(missing)} params "
              f"({', '.join(list(missing)[:5])}{'...' if len(missing)>5 else ''})")
    return mbav1_model


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()
    args = merge_yaml_config(args, raw_cli_args=sys.argv[1:])

    use_amp = str(args.use_amp).lower() == 'true'

    print("=" * 60)
    print(" UKB Self-Supervised Pretraining for Deep Scratch")
    print("=" * 60)
    print(f"  UKB H5:            {args.ukb_h5_path}")
    print(f"  Model:             {args.model}")
    print(f"  Method:            {args.pretraining_method}")
    print(f"  Batch size:        {args.batch_size}")
    print(f"  Epochs:            {args.epochs}")
    print(f"  LR:                {args.lr}")
    print(f"  Weight decay:      {args.weight_decay}")
    print(f"  Optimizer:         {args.optim}")
    print(f"  AMP:               {use_amp}")
    print(f"  Warmup epochs:     {args.warmup_epochs}")
    print(f"  Teacher momentum:  {args.momentum_start} → {args.momentum_end}")
    print(f"  Projection:        hidden={args.projection_hidden}, out={args.projection_dim}")
    print(f"  Checkpoint:        every {args.save_every} epochs, keep {args.keep_last_n}")
    print(f"  Output dir:        {args.output_dir}")
    print("=" * 60)

    # ----- Load UKB data -----
    # New pipeline: build a torch Dataset that reads segments directly from
    # H5 (one handle per DataLoader worker). No pandas DataFrame, no
    # Categorical, no groupby. Pretraining has no labels and the H5 already
    # stores each segment as its own group, so flattening to rows and
    # grouping back was pure overhead.
    metadata = read_ukb_h5_metadata(args.ukb_h5_path)
    print(f"UKB metadata keys: {list(metadata.keys())}")

    # Resolve the pre-fitted scaler ONCE. It is stored inside the Dataset
    # and applied per segment in __getitem__, so the downstream encoder
    # sees data in the same distribution the JNJ finetuning pipeline uses.
    cols = ['x', 'y', 'z']
    scaler_data = None
    if args.scaler_path and os.path.exists(args.scaler_path):
        print(f"Loading pre-fitted scaler: {args.scaler_path}")
        scaler_data = joblib.load(args.scaler_path)
        if not hasattr(scaler_data, 'transform'):
            raise ValueError(
                f"Loaded object from {args.scaler_path} is not a fitted "
                f"scaler (type: {type(scaler_data).__name__})")
        n_expected = getattr(scaler_data, 'n_features_in_', None)
        if n_expected is not None and n_expected != len(cols):
            raise ValueError(
                f"Scaler expects {n_expected} features but UKB pretraining "
                f"uses {len(cols)} ({cols}). Scaler fitted on wrong columns?"
            )
        print(f"  mean:  {scaler_data.mean_}")
        print(f"  scale: {scaler_data.scale_}")
        print(
            f"Applied pre-fitted scaler to {cols} "
            f"(UKB data now in JNJ-compatible distribution)"
        )
    else:
        if args.scaler_path:
            print(
                f"WARNING: scaler_path={args.scaler_path} does not exist. "
                f"Proceeding with raw (un-scaled) UKB data. For a pretrained "
                f"encoder that transfers to JNJ finetuning, pass a valid "
                f"--scaler_path pointing to the production JNJ scaler."
            )
        else:
            print(
                "WARNING: no --scaler_path provided; feeding raw UKB "
                "(x, y, z) into the encoder. The downstream JNJ scaler will "
                "not match."
            )

    max_seq_len = args.max_seq_len
    print(f"Capped max_seq_len: {max_seq_len}")

    # Build train/val segment-index splits at the segment level.
    all_indices = list_ukb_segment_indices(args.ukb_h5_path,
                                           max_segments=args.max_segments)
    train_idx, val_idx = split_indices_by_segment(all_indices,
                                                  val_fraction=0.1, seed=42)
    print(f"Train segments: {len(train_idx):,}, "
          f"Val segments: {len(val_idx):,}")

    train_dataset = UKBPretrainDataset(
        h5_path=args.ukb_h5_path,
        indices=train_idx,
        scaler=scaler_data,
        max_seq_len=max_seq_len,
    )
    val_dataset = UKBPretrainDataset(
        h5_path=args.ukb_h5_path,
        indices=val_idx,
        scaler=scaler_data,
        max_seq_len=max_seq_len,
        verbose=False,
    )

    # ----- GPU -----
    use_dataparallel, device_ids, device = parse_gpu_config(args.num_gpu)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if "cpu" in str(device):
        print("WARNING: Running on CPU — this will be very slow.")

    # ----- Model params (matching param_mba_v1 from train_scratch.py) -----
    best_params = {
        'batch_size': args.batch_size,
        'lr': args.lr,
        'dropout': 0.5,
        'droppath': 0.3,
        'kernel_f': 13,
        'num_feature_layers': 9,
        'featurelayer': 'ResNet',
        'num_filters': 128,
        'norm1': 'BN',
        'optim': args.optim,
    }

    # ----- Create base model -----
    # +1 to account for CLS token prepended in MBA_v1_ForPretraining.forward()
    base_model = setup_model(
        model_name=args.model,
        input_tensor_size=3,
        max_seq_len=max_seq_len + 1,
        best_params=best_params,
        pretraining=True,
        num_classes=1,
    )
    print(f"Base model: {args.model}")
    total_params = sum(p.numel() for p in base_model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # ----- Wrap in pretrainer -----
    num_filters = best_params['num_filters']

    # With DataParallel, the batch is split across GPUs automatically.
    # batch_size here is the TOTAL batch fed in; each GPU gets batch_size/N.
    # User should set a larger --batch_size when using multiple GPUs.
    effective_batch_size = args.batch_size
    if use_dataparallel and device_ids:
        print(f"Multi-GPU ({len(device_ids)} GPUs): total batch={effective_batch_size}, "
              f"per-GPU={effective_batch_size // len(device_ids)}")

    if args.pretraining_method.upper() == "DINO":
        if use_dataparallel and device_ids:
            # DataParallel DINO: splits batches across GPUs, handles teacher EMA
            pretrainer = DataParallelDINOPretrainer(
                base_model=base_model,
                feature_dim=num_filters,
                projection_dim=args.projection_dim,
                projection_hidden=args.projection_hidden,
                momentum=args.momentum_start,
                temperature_student=0.1,
                temperature_teacher=0.04,
                center_momentum=0.9,
                global_crops=2,
                local_crops=2,
                device_ids=device_ids,
            )
            print(f"Using DataParallelDINOPretrainer on GPUs: {device_ids}")
        else:
            pretrainer = DINOPretrainer(
                base_model=base_model,
                feature_dim=num_filters,
                projection_dim=args.projection_dim,
                projection_hidden=args.projection_hidden,
                momentum=args.momentum_start,
                temperature_student=0.1,
                temperature_teacher=0.04,
                center_momentum=0.9,
                global_crops=2,
                local_crops=2,
            )
            pretrainer = pretrainer.to(device)
            print(f"Using single-GPU DINOPretrainer on {device}")
    else:
        raise ValueError(f"Unsupported pretraining method: {args.pretraining_method}")

    print(f"Pretrainer: {args.pretraining_method}")
    total_params = sum(p.numel() for p in pretrainer.parameters())
    print(f"  Total parameters: {total_params:,}")
    print(f"  Effective batch size: {effective_batch_size}")

    # ----- Optimizer with separated weight decay -----
    param_groups = get_param_groups(pretrainer, args.weight_decay)
    lr = args.lr
    if args.optim == 'AdamW':
        optimizer = optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))
    elif args.optim == 'RMSprop':
        optimizer = optim.RMSprop(param_groups, lr=lr)
    elif args.optim == 'RAdam':
        optimizer = optim.RAdam(param_groups, lr=lr)
    else:
        optimizer = optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))

    # ----- DataLoaders -----
    pin_memory = str(args.pin_memory).lower() == 'true'
    persistent_workers = (
        str(args.persistent_workers).lower() == 'true' and args.num_workers > 0
    )
    collate_fn = make_ukb_collate_fn(max_seq_len=max_seq_len,
                                     padding_value=0.0)
    train_loader = DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=effective_batch_size,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2) if args.num_workers > 0 else 0,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=persistent_workers,
    )
    nb_steps = len(train_loader)
    print(f"Train batches/epoch: {nb_steps}, "
          f"val batches/epoch: {len(val_loader)}")
    print(f"DataLoader: num_workers={args.num_workers}, "
          f"pin_memory={pin_memory}, "
          f"persistent_workers={persistent_workers}")

    # ----- LR scheduler: cosine decay with linear warmup -----
    total_steps = nb_steps * args.epochs
    warmup_steps = nb_steps * args.warmup_epochs
    scheduler = CosineWithWarmup(optimizer, warmup_steps=warmup_steps,
                                  total_steps=total_steps)
    print(f"LR schedule: {warmup_steps} warmup steps, "
          f"{total_steps} total steps")

    # ----- AMP scaler -----
    amp_scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # ----- Output dirs -----
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    weights_dir = os.path.join(output_dir, 'weights')
    logs_dir    = os.path.join(output_dir, 'logs')
    plots_dir   = os.path.join(output_dir, 'plots')
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    if args.clear_tracker.lower() == 'true' and os.path.exists(output_dir):
        for sub in ['weights', 'logs', 'plots']:
            p = os.path.join(output_dir, sub)
            if os.path.exists(p):
                shutil.rmtree(p, ignore_errors=True)
                os.makedirs(p, exist_ok=True)
        print("Cleared previous output.")

    # Save config
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)

    # ----- Training loop -----
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    print(f"\nStarting training at {datetime.now()}")
    print("-" * 60)

    for epoch in range(args.epochs):
        # Update teacher EMA momentum (cosine schedule)
        current_momentum = get_momentum_schedule(
            epoch, args.epochs, args.momentum_start, args.momentum_end
        )
        pretrainer.momentum = current_momentum

        # Train
        pretrainer.train()
        train_loss, train_steps = contrastive_pretrain_epoch(
            pretrainer, train_loader, True, device,
            optimizer, scheduler, max_seq_len=max_seq_len,
            gradient_clip=args.gradient_clip, scaler=amp_scaler,
        )
        train_losses.append(train_loss)

        # Validate
        with torch.no_grad():
            pretrainer.eval()
            val_loss, val_steps = contrastive_pretrain_epoch(
                pretrainer, val_loader, False, device,
                optimizer, scheduler, max_seq_len=max_seq_len,
            )
        val_losses.append(val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{args.epochs}]  "
              f"train={train_loss:.4f} ({train_steps} steps)  "
              f"val={val_loss:.4f} ({val_steps} steps)  "
              f"lr={current_lr:.2e}  "
              f"ema_m={current_momentum:.4f}  "
              f"{datetime.now()}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(weights_dir,
                                     f'{args.pretraining_method}_best.pth')
            torch.save(pretrainer.state_dict(), best_path)
            print(f"  -> New best val_loss={val_loss:.4f}")

        # Early stopping
        if early_stopping(val_loss, pretrainer):
            print(f"Early stopping at epoch {epoch+1}")
            early_stopping.restore(pretrainer)
            break

        # Rolling checkpoint every N epochs (keep latest K)
        if (epoch + 1) % args.save_every == 0:
            save_rolling_checkpoint(
                pretrainer.state_dict(), weights_dir,
                args.pretraining_method, epoch + 1,
                keep_last_n=args.keep_last_n,
            )

        gc.collect()
        torch.cuda.empty_cache()

    # ----- Save final outputs -----
    final_path = os.path.join(weights_dir, f'{args.pretraining_method}_final.pth')
    torch.save(pretrainer.state_dict(), final_path)
    print(f"\nFinal weights saved to {final_path}")

    # Encoder-only weights for easy loading into MBA_v1
    # Handle both single-GPU (student.*) and DataParallel (student.module.*) keys
    encoder_state = {}
    for key, val in pretrainer.state_dict().items():
        if key.startswith('student.module.'):
            # DataParallel: strip 'student.module.'
            encoder_state[key[len('student.module.'):]] = val
        elif key.startswith('student.'):
            # Single-GPU: strip 'student.'
            encoder_state[key[len('student.'):]] = val
    encoder_path = os.path.join(weights_dir,
                                f'{args.pretraining_method}_encoder_only.pth')
    torch.save(encoder_state, encoder_path)
    print(f"Encoder-only weights saved to {encoder_path}")

    # Save loss history
    loss_df = pd.DataFrame({
        'epoch': range(1, len(train_losses) + 1),
        'train_loss': train_losses,
        'val_loss': val_losses,
    })
    loss_df.to_csv(os.path.join(logs_dir, 'loss_history.csv'), index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(loss_df['epoch'], loss_df['train_loss'], 'b-', label='Train loss')
    ax.plot(loss_df['epoch'], loss_df['val_loss'], 'r-', label='Val loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'UKB {args.pretraining_method} Pretraining')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'loss_curve.png'), dpi=300)
    plt.close()

    print(f"\nDone! {datetime.now()}")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Weights: {weights_dir}")
    print(f"To finetune MBA_v1:")
    print(f"  python3.11 training/train_scratch.py \\")
    print(f"    --config ... --pretrained_weights_path {encoder_path}")


if __name__ == '__main__':
    main()
