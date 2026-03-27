# -*- coding: utf-8 -*-
"""
TSO Status Prediction Script - H5 Version
==========================================
Predicts 3-class status: 'other', 'non-wear', 'predictTSO'
Uses MBA4TSO_Patch model with raw sensor patches (20Hz data)
Input: Preprocessed H5 file created by convert_parquet_to_h5.py

This version uses H5 files for much faster data loading compared to parquet files.
All preprocessing (scaling, time encoding) is done once during H5 conversion.

Key Differences from Parquet Version:
- No per-batch file I/O or decompression
- No repeated preprocessing (scaler, time encoding)
- Direct memory-mapped access to preprocessed data
- Significantly faster training (10-100x speedup expected)
"""
import subprocess

import joblib

shell_script = '''
cd munge/predictive_modeling
sudo python3.11 -m pip install -r requirements-tso.txt
sudo python3.11 -m pip install -e .
sudo python3.11 -m pip install optuna==4.3.0 seaborn ray TensorboardX torcheval ruptures mamba-ssm[causal-conv1d]==2.2.2
'''
result = subprocess.run(shell_script, shell=True, capture_output=True, text=True)

import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint
import time
import h5py

from models import setup_model
from data import (
)
from losses import (
    measure_loss_tso,
    measure_loss_tso_with_continuity,
    batch_enforce_single_tso
)
from utils import (
    EarlyStopping,
    calculate_metrics_nn,
    smooth_predictions,
    smooth_predictions_combined,
    plot_tso_learning_curves
)


# ==================== H5 Dataset Class ====================
class H5Dataset:
    """
    Fast dataset class for loading preprocessed H5 data.

    Provides efficient random access to segments using memory mapping.
    """
    def __init__(self, h5_file, indices=None):
        """
        Args:
            h5_file: Path to H5 file
            indices: Optional array of indices to use (for train/val split)
        """
        self.h5_file = h5_file
        self.h5f = h5py.File(h5_file, 'r')

        # Load metadata
        self.num_segments = self.h5f.attrs['num_segments']
        self.max_seq_length = self.h5f.attrs['max_seq_length']
        self.samples_per_second = self.h5f.attrs['samples_per_second']
        self.max_len = self.h5f.attrs['max_len']
        self.num_channels = self.h5f.attrs['num_channels']

        # Use subset of indices if provided (for train/val split)
        if indices is not None:
            self.indices = indices
        else:
            self.indices = np.arange(self.num_segments)

        # Datasets (memory-mapped, not loaded into RAM)
        self.X = self.h5f['X']  # [num_segments, max_len, num_channels] - 5 or 6 channels
        self.Y = self.h5f['Y']  # [num_segments, max_len, 2]
        self.seq_lengths = self.h5f['seq_lengths'][:]  # Load lengths into RAM (small)
        self.segment_names = self.h5f['segment_names'][:]  # Load names into RAM

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """Get a single segment."""
        actual_idx = self.indices[idx]
        return {
            'X': self.X[actual_idx],  # [max_len, num_channels] - 5 or 6 channels
            'Y': self.Y[actual_idx],  # [max_len, 2]
            'seq_length': self.seq_lengths[actual_idx],
            'segment': self.segment_names[actual_idx]
        }

    def close(self):
        """Close H5 file."""
        self.h5f.close()


# ==================== Batch Generator ====================
def batch_generator_h5(dataset, batch_size, shuffle=False):
    """
    Generate batches of indices for H5 dataset.

    Args:
        dataset: H5Dataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle indices

    Yields:
        List of indices for each batch
    """
    indices = np.arange(len(dataset))

    if shuffle:
        np.random.shuffle(indices)

    for start_idx in range(0, len(indices), batch_size):
        batch_indices = indices[start_idx:start_idx + batch_size]
        yield batch_indices


# ==================== Visualization Function ====================
def visualize_batch_predictions_h5(pad_X, pad_Y, predictions, x_lens, batch_samples, output_folder,
                                   smooth_preds=True, smooth_method='majority_vote', smooth_window=5):
    """
    Visualize predictions for H5 dataset samples.
    Reuses visualization logic from parquet version.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Convert to numpy if tensors
    if torch.is_tensor(pad_X):
        pad_X = pad_X.cpu().numpy()
    if torch.is_tensor(pad_Y):
        pad_Y = pad_Y.cpu().numpy()
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    if torch.is_tensor(x_lens):
        x_lens = x_lens.cpu().numpy()

    batch_size, seq_len, patch_size, num_channels = pad_X.shape
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    segment_ids = [sample.get('segment', f"sample_{i}") for i, sample in enumerate(batch_samples)]

    # Determine channel names based on num_channels
    if num_channels == 6:
        channel_names = "x, y, z, temperature, time_sin, time_cos"
    else:  # 5 channels - backward compatible, keep old name "time_cyclic"
        channel_names = "x, y, z, temperature, time_cyclic"

    print(f"\n{'='*80}")
    print(f"BATCH VISUALIZATION WITH PREDICTIONS (H5 Dataset)")
    print(f"{'='*80}")
    print(f"Batch size: {batch_size}")
    print(f"Sequence length (minutes): {seq_len}")
    print(f"Patch size (samples/minute): {patch_size}")
    print(f"Channels: {num_channels} [{channel_names}]")

    for sample_idx in range(batch_size):
        seg_id = segment_ids[sample_idx] if sample_idx < len(segment_ids) else f"sample_{sample_idx}"
        seq_len_i = int(x_lens[sample_idx])

        # Get valid data
        X_valid = pad_X[sample_idx, :seq_len_i, :, :]
        y_labels = pad_Y[sample_idx, :seq_len_i]
        y_preds_logits = predictions[sample_idx:sample_idx+1, :seq_len_i, :]
        y_preds_raw = np.argmax(y_preds_logits[0], axis=-1)

        # Apply smoothing if enabled
        if smooth_preds:
            if smooth_method == "combined":
                y_preds_logits_smoothed = smooth_predictions_combined(
                    y_preds_logits,
                    methods=['majority_vote', 'min_segment'],
                    window_size=smooth_window,
                    min_segment_length=3
                )
            else:
                y_preds_logits_smoothed = smooth_predictions(
                    y_preds_logits,
                    method=smooth_method,
                    window_size=smooth_window
                )
            y_preds = np.argmax(y_preds_logits_smoothed[0], axis=-1)
            accuracy_raw = (y_preds_raw == y_labels).sum() / len(y_labels) * 100
            accuracy_smoothed = (y_preds == y_labels).sum() / len(y_labels) * 100
            accuracy = accuracy_smoothed
        else:
            y_preds = y_preds_raw
            accuracy = (y_preds == y_labels).sum() / len(y_labels) * 100
            accuracy_raw = accuracy
            accuracy_smoothed = accuracy

        accuracy_threshold = 90.0

        # Determine output subfolder
        sample_output_folder = output_folder
        if accuracy < accuracy_threshold:
            sample_output_folder = os.path.join(output_folder, 'low_acc')
        os.makedirs(sample_output_folder, exist_ok=True)

        # Create figure (simplified version - you can expand this)
        num_rows = 3  # Ground truth, predictions, accuracy
        fig, axes = plt.subplots(num_rows, 1, figsize=(16, 9))

        title = f'Sample {sample_idx}: {seg_id} ({seq_len_i} minutes)\n'
        if smooth_preds:
            title += f'Raw Acc: {accuracy_raw:.1f}% | Smoothed Acc: {accuracy_smoothed:.1f}%'
        else:
            title += f'Accuracy: {accuracy:.1f}%'
        fig.suptitle(title, fontsize=14, fontweight='bold')

        time_minutes = np.arange(seq_len_i)

        # Accelerometer magnitude (simplified)
        x_mean = X_valid[:, :, 0].mean(axis=1)
        y_mean = X_valid[:, :, 1].mean(axis=1)
        z_mean = X_valid[:, :, 2].mean(axis=1)
        magnitude = np.sqrt(x_mean**2 + y_mean**2 + z_mean**2)

        axes[0].plot(time_minutes, magnitude, linewidth=1.5, alpha=0.8)
        axes[0].set_ylabel('Acc Magnitude')
        axes[0].set_title('Accelerometer Magnitude (minute-level)')
        axes[0].grid(True, alpha=0.3)

        # Ground truth
        axes[1].plot(time_minutes, y_labels, linewidth=2, color='black')
        axes[1].set_ylabel('Label')
        axes[1].set_title('Ground Truth (0=other, 1=non-wear, 2=predictTSO)')
        axes[1].set_yticks([0, 1, 2])
        axes[1].grid(True, alpha=0.3)

        # Predictions
        axes[2].plot(time_minutes, y_preds, linewidth=2, color='red', label='Predicted', alpha=0.7)
        axes[2].plot(time_minutes, y_labels, linewidth=1, color='black', linestyle='--', label='Ground Truth', alpha=0.5)
        axes[2].set_ylabel('Label')
        axes[2].set_xlabel('Time (minutes)')
        axes[2].set_title(f'Predictions - Acc: {accuracy:.1f}%')
        axes[2].set_yticks([0, 1, 2])
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_file = os.path.join(sample_output_folder,
                                f"batch_predictions_{sample_idx}_{seg_id.replace('/', '_')}_{timestamp}_Acc={accuracy:.2f}.png")
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {plot_file}")

    print(f"{'='*80}\n")


# ==================== Training Function ====================
def run_model_tso_h5(model, dataset, batch_size, train_mode, device, optimizer, scheduler,
                    w_other=1.0, w_nonwear=1.0, w_tso=1.0,
                    max_seq_len=1440, patch_size=1200, padding_value=0.0, verbose=False,
                    visualize_batch=False, output_folder=None,
                    smooth_preds=False, smooth_method='majority_vote', smooth_window=5,
                    continuity_weight=0.0, enforce_single_tso=False,
                    min_gap_minutes=30, min_duration_minutes=10):
    """
    Run model on H5 dataset for one epoch.

    Args:
        model: Model instance
        dataset: H5Dataset instance
        batch_size: Batch size
        train_mode: Training or evaluation mode
        device: torch device
        optimizer: Optimizer
        scheduler: LR scheduler
        w_other, w_nonwear, w_tso: Loss weights (unused in current implementation)
        max_seq_len: Max sequence length in minutes
        patch_size: Samples per minute
        padding_value: Padding value
        verbose: Print detailed metrics
        visualize_batch: Visualize batches
        output_folder: Visualization output folder
        smooth_preds: Apply smoothing to predictions
        smooth_method: Smoothing method
        smooth_window: Smoothing window size
        continuity_weight: Weight for continuity loss (0.0 = disabled, 0.1 = recommended)
        enforce_single_tso: If True, post-process to enforce single TSO period
        min_gap_minutes: Minimum gap to merge TSO segments (for post-processing)
        min_duration_minutes: Minimum TSO duration to keep (for post-processing)

    Returns:
        model: Updated model
        metrics: Dict with loss and metrics
        predictions: Dict with predictions and labels
    """
    epoch_loss = 0.0
    epoch_class_loss = 0.0
    epoch_cont_loss = 0.0
    batches = 0

    all_preds_tso = []
    all_labels_tso = []
    # 3-class only
    all_preds_other = []
    all_preds_nonwear = []
    all_labels_other = []
    all_labels_nonwear = []

    for batch_indices in batch_generator_h5(dataset, batch_size=batch_size, shuffle=train_mode):
        t1 = time.time()

        # Prepare batch
        pad_X, pad_Y, x_lens, batch_samples = add_padding_tso_patch_h5(
            dataset, batch_indices, device,
            max_seq_len=max_seq_len,
            patch_size=patch_size,
            padding_value=padding_value,
            num_channels=dataset.num_channels
        )
        load_time = time.time() - t1

        # if batches == 0:
        #     print(f"First batch loading time: {load_time:.4f}s")

        # Forward pass
        outputs = model(pad_X, x_lens)

        # Visualize if requested
        if visualize_batch and output_folder is not None and batches == 0:
            visualize_batch_predictions_h5(pad_X, pad_Y, outputs, x_lens, batch_samples,
                                          output_folder + "/debug_predictions",
                                          smooth_preds=smooth_preds,
                                          smooth_method=smooth_method,
                                          smooth_window=smooth_window)

        # Calculate loss (with optional continuity regularization)
        if continuity_weight > 0:
            total_loss, class_loss, cont_loss = measure_loss_tso_with_continuity(
                outputs, pad_Y, x_lens, continuity_weight=continuity_weight
            )
            epoch_class_loss += class_loss.item()
            epoch_cont_loss += cont_loss.item()
        else:
            total_loss = measure_loss_tso(outputs, pad_Y, x_lens)

        # Backward pass if training
        if train_mode:
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

        # Record loss
        epoch_loss += total_loss.item()
        batches += 1

        # Get predictions
        preds = torch.sigmoid(outputs).cpu().detach().numpy()

        # Apply post-processing to enforce single TSO period if requested
        num_out_channels = outputs.shape[-1]
        if enforce_single_tso:
            if num_out_channels == 1:
                # Binary: threshold sigmoid to get 0/1 class indices
                pred_classes = (torch.sigmoid(outputs[:, :, 0]) > 0.5).long()
            else:
                pred_classes = torch.argmax(outputs, dim=-1)  # [batch, seq_len]
            pred_classes = batch_enforce_single_tso(
                pred_classes, x_lens,
                min_gap_minutes=min_gap_minutes,
                min_duration_minutes=min_duration_minutes
            )
            pred_classes_np = pred_classes.cpu().numpy() if torch.is_tensor(pred_classes) else pred_classes
            for i in range(len(x_lens)):
                valid_len = int(x_lens[i])
                for j in range(valid_len):
                    if num_out_channels == 1:
                        preds[i, j, 0] = float(pred_classes_np[i, j])
                    else:
                        if pred_classes_np[i, j] == 2:  # TSO
                            preds[i, j, :] = [0.0, 0.0, 1.0]
                        elif pred_classes_np[i, j] == 1:  # Non-wear
                            preds[i, j, :] = [0.0, 1.0, 0.0]
                        else:  # Other
                            preds[i, j, :] = [1.0, 0.0, 0.0]

        labels = pad_Y.cpu().numpy()

        # Collect valid predictions
        for i in range(len(x_lens)):
            valid_len = int(x_lens[i])
            valid_preds = preds[i, :valid_len, :]
            valid_labels = labels[i, :valid_len]
            valid_mask = valid_labels >= 0  # exclude padding (-100)

            if num_out_channels == 1:
                # Binary: TSO label = (class 2), everything else = 0
                all_preds_tso.extend(valid_preds[valid_mask, 0].tolist())
                all_labels_tso.extend((valid_labels[valid_mask] == 2).astype(int).tolist())
            else:
                # 3-class: convert labels to one-hot
                labels_onehot = np.zeros((valid_len, 3))
                for j, lbl in enumerate(valid_labels):
                    if lbl >= 0 and lbl < 3:
                        labels_onehot[j, lbl] = 1

                all_preds_other.extend(valid_preds[:, 0].tolist())
                all_preds_nonwear.extend(valid_preds[:, 1].tolist())
                all_preds_tso.extend(valid_preds[:, 2].tolist())

                all_labels_other.extend(labels_onehot[:, 0].tolist())
                all_labels_nonwear.extend(labels_onehot[:, 1].tolist())
                all_labels_tso.extend(labels_onehot[:, 2].tolist())

    # Convert to numpy arrays
    all_preds_tso = np.array(all_preds_tso)
    all_labels_tso = np.array(all_labels_tso)

    # Calculate metrics
    avg_loss = epoch_loss / batches if batches > 0 else 0
    avg_class_loss = epoch_class_loss / batches if batches > 0 else 0
    avg_cont_loss = epoch_cont_loss / batches if batches > 0 else 0

    f1_tso = f1_score(all_labels_tso, (all_preds_tso > 0.5).astype(int), zero_division=0)

    if num_out_channels == 1:
        # Binary: only TSO metric
        f1_avg = f1_tso
        accuracy = np.mean((all_preds_tso > 0.5).astype(int) == all_labels_tso.astype(int))
        metrics = {
            'loss': avg_loss,
            'class_loss': avg_class_loss,
            'cont_loss': avg_cont_loss,
            'accuracy': accuracy,
            'f1_avg': f1_avg,
            'f1_tso': f1_tso
        }
        predictions = {
            'preds_tso': all_preds_tso,
            'labels_tso': all_labels_tso
        }
    else:
        all_preds_other = np.array(all_preds_other)
        all_preds_nonwear = np.array(all_preds_nonwear)
        all_labels_other = np.array(all_labels_other)
        all_labels_nonwear = np.array(all_labels_nonwear)

        f1_other = f1_score(all_labels_other, (all_preds_other > 0.5).astype(int), zero_division=0)
        f1_nonwear = f1_score(all_labels_nonwear, (all_preds_nonwear > 0.5).astype(int), zero_division=0)
        f1_avg = (f1_other + f1_nonwear + f1_tso) / 3

        all_preds_stacked = np.stack([all_preds_other, all_preds_nonwear, all_preds_tso], axis=1)
        all_labels_stacked = np.stack([all_labels_other, all_labels_nonwear, all_labels_tso], axis=1)
        pred_classes = np.argmax(all_preds_stacked, axis=1)
        true_classes = np.argmax(all_labels_stacked, axis=1)
        accuracy = np.mean(pred_classes == true_classes)

        metrics = {
            'loss': avg_loss,
            'class_loss': avg_class_loss,
            'cont_loss': avg_cont_loss,
            'accuracy': accuracy,
            'f1_avg': f1_avg,
            'f1_other': f1_other,
            'f1_nonwear': f1_nonwear,
            'f1_tso': f1_tso
        }
        predictions = {
            'preds_other': all_preds_other,
            'preds_nonwear': all_preds_nonwear,
            'preds_tso': all_preds_tso,
            'labels_other': all_labels_other,
            'labels_nonwear': all_labels_nonwear,
            'labels_tso': all_labels_tso
        }

    if verbose:
        extra = "" if num_out_channels == 1 else (
            f" | F1 other: {metrics.get('f1_other', 0):.4f}"
            f" | F1 nonwear: {metrics.get('f1_nonwear', 0):.4f}"
        )
        print(f"  Loss: {avg_loss:.4f} | F1 avg: {f1_avg:.4f} | F1 TSO: {f1_tso:.4f}{extra}")

    return model, metrics, predictions


# ==================== Main Script ====================
parser = argparse.ArgumentParser(description='TSO Status Prediction with H5 Data')
parser.add_argument('--input_h5', type=str, required=True, help='Path to input H5 file')
parser.add_argument('--split_file', type=str, default=None, help='Path to train/val split file (.npz)')
parser.add_argument('--output', type=str, required=True, help='Output folder name')
parser.add_argument('--model', type=str, default="mba4tso_patch", help='Model name')
parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
parser.add_argument('--num_gpu', type=str, default="0", help='GPU device ID')
parser.add_argument('--multi_gpu', action='store_true', help='Enable multi-GPU training')
parser.add_argument('--training_iterations', type=int, default=1, help='Number of training iterations')
parser.add_argument('--test_only', action='store_true', help='Skip training, only test')
parser.add_argument('--visualize', action='store_true', help='Visualize validation and test batches')
parser.add_argument('--smooth_predictions', action='store_true', help='Apply prediction smoothing')
parser.add_argument('--smooth_method', type=str, default='combined',
                   choices=['majority_vote', 'median_filter', 'moving_average', 'gaussian', 'min_segment', 'combined'],
                   help='Smoothing method')
parser.add_argument('--smooth_window', type=int, default=5, help='Smoothing window size')
parser.add_argument('--val_size', type=float, default=0.1, help='Validation set size (if no split file provided)')

# Single TSO period enforcement arguments
parser.add_argument('--continuity_weight', type=float, default=0.0,
                   help='Weight for continuity loss (0.0=disabled, 0.1-0.5=reasonable range). Default: 0.0')
parser.add_argument('--enforce_single_tso', action='store_true',
                   help='Post-process predictions to enforce single TSO period')
parser.add_argument('--min_gap_minutes', type=int, default=30,
                   help='Merge TSO segments within this gap (minutes). Default: 30')
parser.add_argument('--min_duration_minutes', type=int, default=10,
                   help='Filter out TSO segments shorter than this (minutes). Default: 10')

# Fine-tuning arguments
parser.add_argument('--pretrained_model', type=str, default=None,
                   help='Path to pretrained model checkpoint (.pt) for fine-tuning')
parser.add_argument('--freeze_layers', type=str, default=None,
                   choices=['none', 'encoder', 'feature_extractor', 'all_except_projection'],
                   help='Which layers to freeze during fine-tuning. Default: none (train all layers)')
parser.add_argument('--finetune_lr', type=float, default=None,
                   help='Learning rate for fine-tuning (lower than training from scratch). If not set, uses best_params["lr"]')

args = parser.parse_args()

# ==================== Setup ====================
# Multi-GPU setup
if args.multi_gpu and torch.cuda.is_available():
    gpu_ids = [int(x) for x in args.num_gpu.split(',')]
    device = torch.device(f"cuda:{gpu_ids[0]}")
    num_gpus = len(gpu_ids)
    print(f"Multi-GPU mode: Using {num_gpus} GPUs: {gpu_ids}")
else:
    device = torch.device(f"cuda:{args.num_gpu}" if torch.cuda.is_available() else "cpu")
    gpu_ids = None
    num_gpus = 1
    print(f"Single GPU/CPU mode: {device}")

print(f"Starting training at: {datetime.now()}")

# Create output folders
results_folder = f"/mnt/data/GENEActive-featurized/results/DL/{args.output}"
training_output_folder = os.path.join(results_folder, "training/")
predictions_output_folder = os.path.join(training_output_folder, "predictions/")
learning_plots_output_folder = os.path.join(training_output_folder, "learning_plots/")
train_models_folder = os.path.join(training_output_folder, "model_weights/")
checkpoint_folder = os.path.join(training_output_folder, "checkpoints/")
training_logs_folder = os.path.join(training_output_folder, "training_logs/")

def create_folder(folders):
    for folder in folders:
        os.makedirs(folder, exist_ok=True)

create_folder([results_folder, training_output_folder, predictions_output_folder,
               learning_plots_output_folder, train_models_folder, training_logs_folder, checkpoint_folder])

# Save arguments
with open(os.path.join(results_folder, "user_arguments.txt"), "w") as f:
    f.write(str(args))

# ==================== Model Hyperparameters ====================
best_params = {
    'batch_size': 24,
    'num_filters': 128,
    'dropout': 0.3,
    'droppath': 0.3,
    'kernel_f': 3,
    'kernel_MBA': 7,
    'num_feature_layers': 6,
    'blocks_MBA': 5,
    'featurelayer': 'ResNet',
    'lr': args.finetune_lr if args.finetune_lr is not None else 0.001,
    'w_other': 1.0,
    'w_nonwear': 1.0,
    'w_tso': 1.0,
    'padding_value': 0.0,
    'patch_size': 1200,
    'patch_channels': 6,
    'norm1': 'BN',
    'norm2': 'GN',
    'output_channels': 1,  # 1=binary (TSO vs not), 3=three-class (other/non-wear/TSO)
    'skip_connect': True,
    'skip_cross_attention': True,
    'use_scaler': False,
    'pretrained_model_path': args.pretrained_model,
    'freeze_layers': args.freeze_layers
}

print("Model hyperparameters:")
pprint(best_params)

# Print fine-tuning configuration if enabled
if args.pretrained_model is not None:
    print("\n" + "="*60)
    print("FINE-TUNING MODE")
    print("="*60)
    print(f"  Pretrained model: {args.pretrained_model}")
    print(f"  Freeze layers: {args.freeze_layers if args.freeze_layers else 'none (train all)'}")
    print(f"  Fine-tune LR: {best_params['lr']:.6f}")
    print("="*60 + "\n")

# ==================== Load H5 Data ====================
print(f"\nLoading H5 data from: {args.input_h5}")

# Load train/val split
if args.split_file is not None and os.path.exists(args.split_file):
    print(f"Loading split from: {args.split_file}")
    split_data = np.load(args.split_file)
    train_indices = split_data['train']
    val_indices = split_data['val']
else:
    print(f"No split file provided, creating random split (val_size={args.val_size})")
    # Open H5 to get total count
    with h5py.File(args.input_h5, 'r') as h5f:
        num_segments = h5f.attrs['num_segments']

    indices = np.arange(num_segments)
    np.random.shuffle(indices)
    val_count = int(num_segments * args.val_size)
    train_indices = indices[val_count:]
    val_indices = indices[:val_count]

print(f"Train segments: {len(train_indices)}")
print(f"Val segments: {len(val_indices)}")

# Create datasets
dataset_train = H5Dataset(args.input_h5, indices=train_indices)
dataset_val = H5Dataset(args.input_h5, indices=val_indices)
dataset_test = dataset_val  # Use validation set as test for now

max_seq_len = 1440  # 24 hours in minutes

print(f"\nDataset info:")
print(f"  Max sequence length: {dataset_train.max_seq_length}s = {max_seq_len} minutes")
print(f"  Samples per second: {dataset_train.samples_per_second}Hz")
print(f"  Max samples per segment: {dataset_train.max_len:,}")
print(f"  Number of channels: {dataset_train.num_channels}")
if dataset_train.num_channels == 6:
    print(f"  Time encoding: Sin+Cos (unique 2D representation)")
elif dataset_train.num_channels == 5:
    print(f"  Time encoding: Sin only (backward compatible)")
else:
    print(f"  Time encoding: Unknown format")

# ==================== Training Loop ====================
for iteration in range(args.training_iterations):
    print(f"\n{'='*80}")
    print(f"Training iteration {iteration+1}/{args.training_iterations}")
    print(f"{'='*80}\n")

    # Setup model
    model = setup_model(args.model, None, max_seq_len, best_params,
                      pretraining=False, num_classes=best_params['output_channels'])
    model = model.to(device)

    # Load pretrained weights for fine-tuning
    if args.pretrained_model is not None:
        print(f"\nLoading pretrained weights from: {args.pretrained_model}")
        checkpoint = torch.load(args.pretrained_model, map_location=device)

        # Load state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print("✓ Pretrained weights loaded successfully")
        else:
            model.load_state_dict(checkpoint, strict=False)
            print("✓ Pretrained weights loaded (raw state dict)")

        # Freeze layers based on strategy
        if args.freeze_layers == 'encoder':
            print("  Freezing encoder layers...")
            for name, param in model.named_parameters():
                if 'encoder' in name:
                    param.requires_grad = False
            print("  ✓ Encoder frozen")

        elif args.freeze_layers == 'feature_extractor':
            print("  Freezing feature extractor...")
            for name, param in model.named_parameters():
                if 'feature_extractor' in name or 'patch_embedding' in name:
                    param.requires_grad = False
            print("  ✓ Feature extractor + patch embedding frozen")

        elif args.freeze_layers == 'all_except_projection':
            print("  Freezing all layers except output projection...")
            for name, param in model.named_parameters():
                if 'output_projection' not in name:
                    param.requires_grad = False
            print("  ✓ All layers frozen except output_projection")

        print(f"  Fine-tuning strategy: {args.freeze_layers if args.freeze_layers else 'train all layers'}")

    # Wrap with DataParallel for multi-GPU
    if args.multi_gpu and gpu_ids is not None and len(gpu_ids) > 1:
        print(f"Wrapping model with DataParallel for GPUs: {gpu_ids}")
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    if frozen_params > 0:
        print(f"Frozen parameters: {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")

    # Setup optimizer and scheduler
    optimizer = optim.RMSprop(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=best_params['lr']
    )

    nb_steps = len(train_indices) // best_params['batch_size']
    print(f"Steps per epoch: {nb_steps}")
    # One smooth cosine decay over all training steps (no warm restarts).
    # CosineAnnealingWarmRestarts with T_0=nb_steps restarts every epoch,
    # periodically spiking the LR and causing val F1 to collapse to 0.
    total_steps = nb_steps * args.epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_accuracy': [], 'val_accuracy': [],
        'train_f1_avg': [], 'val_f1_avg': [],
        'train_f1_tso': [], 'val_f1_tso': [],
    }
    if best_params['output_channels'] == 3:
        history.update({'train_f1_other': [], 'train_f1_nonwear': [],
                        'val_f1_other': [], 'val_f1_nonwear': []})

    # Early stopping
    early_stopping = EarlyStopping(patience=10, verbose=True)
    torch.cuda.empty_cache()

    # Print single TSO period enforcement configuration
    print("\n" + "="*60)
    print("Single TSO Period Enforcement Configuration:")
    print("="*60)
    if args.continuity_weight > 0:
        print(f"  ✓ Continuity loss ENABLED (weight: {args.continuity_weight})")
        print(f"    - Applies during: Training & Validation")
        print(f"    - Penalizes fragmented TSO predictions")
    else:
        print(f"  ✗ Continuity loss DISABLED (weight: 0.0)")

    if args.enforce_single_tso:
        print(f"  ✓ Post-processing ENABLED")
        print(f"    - Applies during: Testing ONLY")
        print(f"    - Min gap for merging: {args.min_gap_minutes} minutes")
        print(f"    - Min TSO duration: {args.min_duration_minutes} minutes")
    else:
        print(f"  ✗ Post-processing DISABLED")

    if args.continuity_weight == 0 and not args.enforce_single_tso:
        print(f"\n  Note: Using baseline approach (no single TSO enforcement)")
    elif args.continuity_weight > 0 and args.enforce_single_tso:
        print(f"\n  Note: Using HYBRID approach (recommended for best results)")
        print(f"        - Training: Soft constraint via continuity loss")
        print(f"        - Testing: Hard constraint via post-processing")
    print("="*60 + "\n")

    # Training loop
    if not args.test_only:
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch+1}/{args.epochs}")

            # Train
            model.train()
            t1 = time.time()
            model, train_metrics, _ = run_model_tso_h5(
                model, dataset_train, best_params['batch_size'], True, device, optimizer, scheduler,
                max_seq_len=max_seq_len,
                patch_size=best_params['patch_size'],
                padding_value=best_params['padding_value'],
                verbose=(epoch % 5 == 0),
                continuity_weight=args.continuity_weight,
                enforce_single_tso=False,  # No post-processing during training
                min_gap_minutes=args.min_gap_minutes,
                min_duration_minutes=args.min_duration_minutes
            )
            print("Time cost for training: ", time.time()-t1)

            # Record train metrics
            history['train_loss'].append(train_metrics['loss'])
            history['train_accuracy'].append(train_metrics['accuracy'])
            history['train_f1_avg'].append(train_metrics['f1_avg'])
            history['train_f1_tso'].append(train_metrics['f1_tso'])
            if best_params['output_channels'] == 3:
                history['train_f1_other'].append(train_metrics['f1_other'])
                history['train_f1_nonwear'].append(train_metrics['f1_nonwear'])

            # Print with optional continuity loss breakdown
            if args.continuity_weight > 0:
                print(f"  Train - Loss: {train_metrics['loss']:.4f} (Class: {train_metrics['class_loss']:.4f}, Cont: {train_metrics['cont_loss']:.4f}), "
                      f"Acc: {train_metrics['accuracy']:.4f}, F1 avg: {train_metrics['f1_avg']:.4f}")
            else:
                print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, F1 avg: {train_metrics['f1_avg']:.4f}")

            # Validation
            if epoch % 1 == 0:
                model.eval()
                with torch.no_grad():
                    _, val_metrics, _ = run_model_tso_h5(
                        model, dataset_val, best_params['batch_size'], False, device, optimizer, scheduler,
                        max_seq_len=max_seq_len,
                        patch_size=best_params['patch_size'],
                        padding_value=best_params['padding_value'],
                        verbose=False,
                        visualize_batch=args.visualize,
                        output_folder=training_output_folder,
                        smooth_preds=args.smooth_predictions,
                        smooth_method=args.smooth_method,
                        smooth_window=args.smooth_window,
                        continuity_weight=args.continuity_weight,
                        enforce_single_tso=False,  # No post-processing during validation
                        min_gap_minutes=args.min_gap_minutes,
                        min_duration_minutes=args.min_duration_minutes
                    )

                history['val_loss'].append(val_metrics['loss'])
                history['val_accuracy'].append(val_metrics['accuracy'])
                history['val_f1_avg'].append(val_metrics['f1_avg'])
                history['val_f1_tso'].append(val_metrics['f1_tso'])
                if best_params['output_channels'] == 3:
                    history['val_f1_other'].append(val_metrics['f1_other'])
                    history['val_f1_nonwear'].append(val_metrics['f1_nonwear'])

                # Print with optional continuity loss breakdown
                if args.continuity_weight > 0:
                    print(f"  Val   - Loss: {val_metrics['loss']:.4f} (Class: {val_metrics['class_loss']:.4f}, Cont: {val_metrics['cont_loss']:.4f}), "
                          f"Acc: {val_metrics['accuracy']:.4f}, F1 avg: {val_metrics['f1_avg']:.4f}")
                else:
                    print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1 avg: {val_metrics['f1_avg']:.4f}")

                # Early stopping check
                early_stopping(val_metrics['loss'], model)
                if early_stopping.early_stop:
                    print("Early stopping triggered")
                    break

            # Save best model
            if val_metrics['loss'] == early_stopping.best_score:
                model_path = os.path.join(train_models_folder,
                                        f"best_model_iter_{iteration}.pt")
                model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_to_save.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': train_metrics['loss'],
                    'train_accuracy': train_metrics['accuracy'],
                    'train_f1_avg': train_metrics['f1_avg'],
                    'val_loss': val_metrics['loss'],
                    'val_accuracy': val_metrics['accuracy'],
                    'val_f1_avg': val_metrics['f1_avg'],
                    'history': history,
                    'best_params': best_params,
                    'num_gpus': num_gpus
                }, model_path)
                print(f"  -> Saved best model at epoch {epoch}")

        # Load best model
        checkpoint_path = os.path.join(train_models_folder, f"best_model_iter_{iteration}.pt")
        checkpoint = torch.load(checkpoint_path)
        model_to_load = model.module if isinstance(model, torch.nn.DataParallel) else model
        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

    # Test evaluation
    print(f"\nEvaluating on test set...")
    with torch.no_grad():
        _, test_metrics, test_predictions = run_model_tso_h5(
            model, dataset_test, best_params['batch_size'], False, device, optimizer, scheduler,
            max_seq_len=max_seq_len,
            patch_size=best_params['patch_size'],
            padding_value=best_params['padding_value'],
            verbose=True,
            visualize_batch=args.visualize,
            output_folder=training_output_folder,
            smooth_preds=args.smooth_predictions,
            smooth_method=args.smooth_method,
            smooth_window=args.smooth_window,
            continuity_weight=args.continuity_weight,
            enforce_single_tso=args.enforce_single_tso,
            min_gap_minutes=args.min_gap_minutes,
            min_duration_minutes=args.min_duration_minutes
        )

    # Calculate comprehensive metrics
    if best_params['output_channels'] == 1:
        pred_classes = (test_predictions['preds_tso'] > 0.5).astype(int)
        true_classes = test_predictions['labels_tso'].astype(int)
    else:
        pred_classes = np.argmax(
            np.stack([test_predictions['preds_other'],
                     test_predictions['preds_nonwear'],
                     test_predictions['preds_tso']], axis=1),
            axis=1
        )
        true_classes = np.argmax(
            np.stack([test_predictions['labels_other'],
                     test_predictions['labels_nonwear'],
                     test_predictions['labels_tso']], axis=1),
            axis=1
        )

    comprehensive_metrics = calculate_metrics_nn(true_classes, pred_classes, classification=True)

    # Print results
    print(f"\n{'='*80}")
    print(f"TEST RESULTS - Iteration {iteration}")
    print(f"{'='*80}")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"\nOverall Metrics:")
    print(f"  Accuracy: {comprehensive_metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {comprehensive_metrics['balanced_accuracy']:.4f}")
    print(f"  F1 (macro): {comprehensive_metrics['f1_score_macro']:.4f}")
    print(f"  F1 (weighted): {comprehensive_metrics['f1_score_weighted']:.4f}")
    print(f"{'='*80}\n")

    # Save results
    results = {
        'iteration': iteration,
        'test_metrics': test_metrics,
        'comprehensive_metrics': comprehensive_metrics,
        'history': history
    }

    results_file = os.path.join(predictions_output_folder, f"results_iter_{iteration}.joblib")
    joblib.dump(results, results_file)
    print(f"Results saved to {results_file}")

    # Plot learning curves
    if not args.test_only and len(history.get('train_loss', [])) > 0:
        plot_tso_learning_curves(
            history=history,
            output_filepath=os.path.join(learning_plots_output_folder,
                                        f"training_history_iter_{iteration}.png")
        )
        print(f"Learning curves saved\n")

# Close datasets
dataset_train.close()
dataset_val.close()

print(f"\n{'='*80}")
print("Training complete!")
print(f"{'='*80}")
