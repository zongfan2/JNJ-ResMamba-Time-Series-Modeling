#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UKB MAE (Masked Autoencoder) pretraining for the MBA_v1 encoder.

This script is a sibling of ``pretrain_ukb.py`` (DINO) and shares the same
data pipeline. The only differences are:

  * It wraps ``MBA_v1_ForPretraining`` in ``MBAMAEPretrainer`` instead of
    ``DINOPretrainer`` — a SimMIM-style masked autoencoder that keeps masked
    tokens in-place so MBA_v1's locality-sensitive encoder (TCN + Mamba)
    sees the full temporal grid.
  * No teacher EMA, no contrastive augmentation stack, no DataParallel MAE
    wrapper — MAE is a single-network objective so it runs on one GPU or
    via plain ``nn.DataParallel`` if desired (not wired by default).

Saved checkpoints use the same ``student.*`` key convention as the DINO
pretrainer so that ``load_pretrained_encoder_into_mbav1`` in the main
training scripts continues to work unchanged.

Usage (Domino):
    python3.11 training/pretrain_ukb_mae.py \
        --config experiments/configs/pretrain_ukb_mae.yaml
"""

import subprocess
import os
import sys

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

from models.setup import setup_model
from models.pretrainer import MBAMAEPretrainer
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
    parser = argparse.ArgumentParser(description='UKB MAE Pretraining')
    parser.add_argument('--config', type=str, default='',
                        help='Path to YAML config file')
    # Data
    parser.add_argument('--ukb_h5_path', type=str, default='')
    parser.add_argument('--max_segments', type=int, default=0)
    parser.add_argument('--max_seq_len', type=int, default=1221,
                        help='Cap sequence length. Default 1221 = 61s@20Hz, '
                             'matching the production JNJ pipeline.')
    parser.add_argument('--scaler_path', type=str, default='',
                        help='Path to the pre-fitted JNJ StandardScaler so '
                             'UKB pretraining sees the same input '
                             'distribution as downstream fine-tuning.')

    # Model
    parser.add_argument('--model', type=str, default='mbav1_pretrain',
                        help='Base encoder architecture (must be mbav1_pretrain '
                             'or a compatible MBA_v1_ForPretraining).')
    parser.add_argument('--pretraining_method', type=str, default='MAE',
                        help='Only used for checkpoint naming.')

    # MAE-specific
    parser.add_argument('--patch_size', type=int, default=20,
                        help='Width of each masking block in samples '
                             '(20 ≈ 1s at 20Hz).')
    parser.add_argument('--mask_ratio', type=float, default=0.6,
                        help='Fraction of non-overlapping blocks masked.')
    parser.add_argument('--decoder_depth', type=int, default=2)
    parser.add_argument('--decoder_dim', type=int, default=0,
                        help='Decoder width. 0 = num_filters // 2.')
    parser.add_argument('--decoder_kernel_size', type=int, default=7)
    parser.add_argument('--decoder_dropout', type=float, default=0.1)
    parser.add_argument('--decoder_drop_path', type=float, default=0.1)
    parser.add_argument('--decoder_norm', type=str, default='BN')
    parser.add_argument('--norm_pixel_loss', type=str, default='True')

    # Training
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1.5e-4)
    parser.add_argument('--optim', type=str, default='AdamW')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='Weight decay (excluded from BN/bias).')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience.')
    parser.add_argument('--gradient_clip', type=float, default=1.0)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--use_amp', type=str, default='True')

    # Checkpoint
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--keep_last_n', type=int, default=3)

    # GPU / DataLoader
    parser.add_argument('--num_gpu', type=str, default='NA')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pin_memory', type=str, default='True')
    parser.add_argument('--persistent_workers', type=str, default='True')

    # Output
    parser.add_argument('--output_dir', type=str, default='')
    parser.add_argument('--clear_tracker', type=str, default='False')
    return parser.parse_args()


def merge_yaml_config(args, raw_cli_args=None):
    """Flat merge of YAML into argparse. CLI wins over YAML."""
    if not args.config or not os.path.exists(args.config):
        return args
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f) or {}
    flat = {}
    for key, val in cfg.items():
        if isinstance(val, dict):
            flat.update(val)
        else:
            flat[key] = val
    cli_provided = set()
    if raw_cli_args is not None:
        for tok in raw_cli_args:
            if tok.startswith('--'):
                cli_provided.add(tok.lstrip('-').replace('-', '_'))
    for key, val in flat.items():
        key_under = key.replace('-', '_')
        if hasattr(args, key_under) and key_under not in cli_provided:
            setattr(args, key_under, val)
    return args


# ============================================================================
# GPU config (matches pretrain_ukb.py)
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
# Weight-decay param groups
# ============================================================================

def get_param_groups(model, weight_decay):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'norm' in name or 'bn' in name or 'bias' in name or 'mask_token' in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0},
    ]


# ============================================================================
# Cosine LR with warmup (copy of pretrain_ukb.py's)
# ============================================================================

class CosineWithWarmup(torch.optim.lr_scheduler._LRScheduler):
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
            scale = max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return [max(self.min_lr, base_lr * scale) for base_lr in self.base_lrs]


# ============================================================================
# MAE epoch — simpler than DINO: just forward and backprop
# ============================================================================

def mae_pretrain_epoch(pretrainer, loader, train_mode, device,
                       optimizer, scheduler, max_seq_len=None,
                       gradient_clip=1.0, scaler=None):
    """Single MAE pretraining epoch.

    Mirrors ``contrastive_pretrain_epoch`` from ``pretrain_ukb.py`` so both
    scripts have an identical surface to the data pipeline.
    """
    epoch_loss = 0.0
    batches = 0
    use_amp = scaler is not None

    for batch_data, x_lens in loader:
        batch_data = batch_data.to(device, non_blocking=True)
        x_lens = x_lens.to(device, non_blocking=True)

        if max_seq_len is not None and batch_data.size(1) > max_seq_len:
            batch_data = batch_data[:, :max_seq_len, :]
            x_lens = torch.clamp(x_lens, max=max_seq_len)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        if use_amp and train_mode:
            with torch.cuda.amp.autocast():
                res = pretrainer(batch_data, x_lens)
                loss = res["loss"]
        else:
            res = pretrainer(batch_data, x_lens)
            loss = res["loss"]

        # nn.DataParallel gathers per-GPU scalar losses along dim 0,
        # producing a vector. Reduce to a scalar for backward and checks.
        if loss.dim() > 0:
            loss = loss.mean()

        if train_mode:
            if torch.isnan(loss) or torch.isinf(loss):
                print("  Warning: NaN/Inf loss, skipping batch.")
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
# Rolling checkpoint (matches DINO script)
# ============================================================================

def save_rolling_checkpoint(state_dict, weights_dir, method, epoch,
                            keep_last_n=3):
    ckpt_path = os.path.join(weights_dir, f'{method}_epoch{epoch}.pth')
    torch.save(state_dict, ckpt_path)
    print(f"  Saved checkpoint: {ckpt_path}")

    pattern = os.path.join(weights_dir, f'{method}_epoch*.pth')
    existing = sorted(globmod.glob(pattern), key=os.path.getmtime)
    while len(existing) > keep_last_n:
        old = existing.pop(0)
        os.remove(old)
        print(f"  Removed old checkpoint: {os.path.basename(old)}")


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()
    args = merge_yaml_config(args, raw_cli_args=sys.argv[1:])

    use_amp = str(args.use_amp).lower() == 'true'
    norm_pixel_loss = str(args.norm_pixel_loss).lower() == 'true'

    print("=" * 60)
    print(" UKB MAE Pretraining for Deep Scratch (MBA_v1 backbone)")
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
    print(f"  Patch size:        {args.patch_size}")
    print(f"  Mask ratio:        {args.mask_ratio}")
    print(f"  Decoder depth:     {args.decoder_depth}")
    print(f"  Decoder dim:       {args.decoder_dim or 'num_filters/2'}")
    print(f"  Norm pixel loss:   {norm_pixel_loss}")
    print(f"  Output dir:        {args.output_dir}")
    print("=" * 60)

    # ----- Load UKB data (same pipeline as DINO) -----
    metadata = read_ukb_h5_metadata(args.ukb_h5_path)
    print(f"UKB metadata keys: {list(metadata.keys())}")

    cols = ['x', 'y', 'z']
    scaler_data = None
    if args.scaler_path and os.path.exists(args.scaler_path):
        print(f"Loading pre-fitted scaler: {args.scaler_path}")
        scaler_data = joblib.load(args.scaler_path)
        if not hasattr(scaler_data, 'transform'):
            raise ValueError(
                f"Loaded object from {args.scaler_path} is not a fitted "
                f"scaler (type: {type(scaler_data).__name__})"
            )
        n_expected = getattr(scaler_data, 'n_features_in_', None)
        if n_expected is not None and n_expected != len(cols):
            raise ValueError(
                f"Scaler expects {n_expected} features but UKB pretraining "
                f"uses {len(cols)} ({cols})."
            )
        print(f"  mean:  {scaler_data.mean_}")
        print(f"  scale: {scaler_data.scale_}")
    else:
        if args.scaler_path:
            print(
                f"WARNING: scaler_path={args.scaler_path} does not exist. "
                "Proceeding with raw (un-scaled) UKB data."
            )
        else:
            print(
                "WARNING: no --scaler_path provided; feeding raw UKB "
                "(x, y, z) into the encoder."
            )

    max_seq_len = args.max_seq_len
    print(f"Capped max_seq_len: {max_seq_len}")

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

    # ----- Model params (matching param_mba_v1 in train_scratch.py) -----
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
    num_filters = best_params['num_filters']
    input_dim = 3

    # ----- Base encoder (MBA_v1_ForPretraining) -----
    # +1 to account for the CLS token prepended inside the encoder.
    base_model = setup_model(
        model_name=args.model,
        input_tensor_size=input_dim,
        max_seq_len=max_seq_len + 1,
        best_params=best_params,
        pretraining=True,
        num_classes=1,
    )
    print(f"Base model: {args.model}")
    print(f"  Total parameters (encoder only): "
          f"{sum(p.numel() for p in base_model.parameters()):,}")

    # ----- Wrap in MAE pretrainer -----
    decoder_dim = args.decoder_dim if args.decoder_dim > 0 else None
    pretrainer = MBAMAEPretrainer(
        base_model=base_model,
        num_filters=num_filters,
        input_dim=input_dim,
        patch_size=args.patch_size,
        mask_ratio=args.mask_ratio,
        decoder_depth=args.decoder_depth,
        decoder_dim=decoder_dim,
        decoder_kernel_size=args.decoder_kernel_size,
        decoder_dropout=args.decoder_dropout,
        decoder_drop_path=args.decoder_drop_path,
        decoder_norm=args.decoder_norm,
        norm_pixel_loss=norm_pixel_loss,
    )
    pretrainer = pretrainer.to(device)
    if use_dataparallel and device_ids:
        # Plain DataParallel wrapping the MAE pretrainer.
        # The student's keys become 'student.module.*' which
        # load_pretrained_encoder_into_mbav1() already handles.
        pretrainer = nn.DataParallel(pretrainer, device_ids=device_ids)
        print(f"Wrapped in nn.DataParallel on GPUs: {device_ids}")

    total_params = sum(p.numel() for p in pretrainer.parameters())
    print(f"Pretrainer total parameters: {total_params:,}")

    # ----- Optimizer -----
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
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
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

    # ----- LR schedule -----
    total_steps = nb_steps * args.epochs
    warmup_steps = nb_steps * args.warmup_epochs
    scheduler = CosineWithWarmup(optimizer, warmup_steps=warmup_steps,
                                  total_steps=total_steps)
    print(f"LR schedule: {warmup_steps} warmup steps, "
          f"{total_steps} total steps")

    amp_scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # ----- Output dirs -----
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    weights_dir = os.path.join(output_dir, 'weights')
    logs_dir = os.path.join(output_dir, 'logs')
    plots_dir = os.path.join(output_dir, 'plots')
    for d in (weights_dir, logs_dir, plots_dir):
        os.makedirs(d, exist_ok=True)

    if args.clear_tracker.lower() == 'true' and os.path.exists(output_dir):
        for sub in ['weights', 'logs', 'plots']:
            p = os.path.join(output_dir, sub)
            if os.path.exists(p):
                shutil.rmtree(p, ignore_errors=True)
                os.makedirs(p, exist_ok=True)
        print("Cleared previous output.")

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
        pretrainer.train()
        train_loss, train_steps = mae_pretrain_epoch(
            pretrainer, train_loader, True, device,
            optimizer, scheduler, max_seq_len=max_seq_len,
            gradient_clip=args.gradient_clip, scaler=amp_scaler,
        )
        train_losses.append(train_loss)

        with torch.no_grad():
            pretrainer.eval()
            val_loss, val_steps = mae_pretrain_epoch(
                pretrainer, val_loader, False, device,
                optimizer, scheduler, max_seq_len=max_seq_len,
            )
        val_losses.append(val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{args.epochs}]  "
              f"train={train_loss:.4f} ({train_steps} steps)  "
              f"val={val_loss:.4f} ({val_steps} steps)  "
              f"lr={current_lr:.2e}  "
              f"{datetime.now()}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(weights_dir,
                                     f'{args.pretraining_method}_best.pth')
            torch.save(pretrainer.state_dict(), best_path)
            print(f"  -> New best val_loss={val_loss:.4f}")

        if early_stopping(val_loss, pretrainer):
            print(f"Early stopping at epoch {epoch+1}")
            early_stopping.restore(pretrainer)
            break

        if (epoch + 1) % args.save_every == 0:
            save_rolling_checkpoint(
                pretrainer.state_dict(), weights_dir,
                args.pretraining_method, epoch + 1,
                keep_last_n=args.keep_last_n,
            )

        gc.collect()
        torch.cuda.empty_cache()

    # ----- Save final outputs -----
    final_path = os.path.join(weights_dir,
                              f'{args.pretraining_method}_final.pth')
    torch.save(pretrainer.state_dict(), final_path)
    print(f"\nFinal weights saved to {final_path}")

    # Encoder-only extraction for downstream loading into MBA_v1.
    # Mirrors the DINO script's naming convention: 'student.*' or
    # 'student.module.*' -> stripped prefix.
    encoder_state = {}
    for key, val in pretrainer.state_dict().items():
        if key.startswith('module.student.module.'):  # DataParallel MAE + inner DP
            encoder_state[key[len('module.student.module.'):]] = val
        elif key.startswith('module.student.'):       # nn.DataParallel on MAE
            encoder_state[key[len('module.student.'):]] = val
        elif key.startswith('student.module.'):       # inner DataParallel only
            encoder_state[key[len('student.module.'):]] = val
        elif key.startswith('student.'):              # single-GPU
            encoder_state[key[len('student.'):]] = val
    encoder_path = os.path.join(weights_dir,
                                f'{args.pretraining_method}_encoder_only.pth')
    torch.save(encoder_state, encoder_path)
    print(f"Encoder-only weights saved to {encoder_path}")

    loss_df = pd.DataFrame({
        'epoch': range(1, len(train_losses) + 1),
        'train_loss': train_losses,
        'val_loss': val_losses,
    })
    loss_df.to_csv(os.path.join(logs_dir, 'loss_history.csv'), index=False)

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
