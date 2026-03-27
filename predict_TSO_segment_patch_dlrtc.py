# -*- coding: utf-8 -*-
"""
TSO Status Prediction Script with Dynamic Label Refinement (DLR-TC)
Predicts 3-class status: 'other', 'non-wear', 'predictTSO'

This script implements Dynamic Label Refinement with Temporal Consistency (DLR-TC),
a novel approach for learning from noisy labels in wearable sensor data.

Key Features:
1. Two-phase training: Warmup → Joint label-model refinement
2. Generalized Cross Entropy (GCE) for noise robustness
3. Temporal smoothness constraints for physical plausibility
4. Trainable soft labels that evolve during training

Reference Implementation: Based on theoretical framework combining:
- Zhang & Sabuncu (2018): "Generalized Cross Entropy Loss"
- Li et al. (2020): "DivideMix" label correction
- Custom temporal consistency constraints for time series

Uses MBA4TSO_Patch model with raw sensor patches (20Hz data)
Input: Raw accelerometer data at 20Hz instead of aggregated features
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
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint
import time
import h5py  # For H5 data loading 

from Helpers.DL_models import setup_model
from Helpers.DL_helpers import (
    add_padding_tso_patch_h5,  # Padding for H5 data
    measure_loss_tso,
    EarlyStopping,
    smooth_predictions,  # Post-processing for predictions
    smooth_predictions_combined,  # Combined smoothing methods
    plot_tso_learning_curves  # Visualization
)

# DLR-TC specific imports
from Helpers.dlrtc_losses import (
    GeneralizedCrossEntropy,
    CompatibilityLoss,
    TemporalSmoothnessLoss,
    DLRTCLoss,
    SoftLabelManager
)


# ==================== Visualization Function ====================
def visualize_batch_predictions(pad_X, pad_Y, predictions, x_lens, batch_df, output_folder, seg_column='segment',
                                smooth_preds=True, smooth_method='majority_vote', smooth_window=5):
    """
    Visualize raw sensor patches, ground truth, and predictions for quality checking.

    Args:
        pad_X: [batch_size, seq_len, patch_size, 5] - raw sensor patches
        pad_Y: [batch_size, seq_len] - ground truth labels (minute-level)
        predictions: [batch_size, seq_len, 3] - model predictions (logits)
        x_lens: [batch_size] - original sequence lengths
        batch_df: DataFrame - batch metadata
        output_folder: str - output directory
        seg_column: str - segment column name
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

    segment_ids = batch_df[seg_column].unique()[:batch_size]

    # Determine channel names based on num_channels
    if num_channels == 6:
        channel_names = "x, y, z, temperature, time_sin, time_cos"
    elif num_channels == 5:
        channel_names = "x, y, z, temperature, time_sin"
    else:
        channel_names = "unknown"

    # Summary statistics
    print(f"\n{'='*80}")
    print(f"BATCH VISUALIZATION WITH PREDICTIONS")
    print(f"{'='*80}")
    print(f"Batch size: {batch_size}")
    print(f"Sequence length (minutes): {seq_len}")
    print(f"Patch size (samples/minute): {patch_size}")
    print(f"Channels: {num_channels} [{channel_names}]")

    # Get split_type for each segment if available (for organizing output folders)
    split_types = {}
    if 'split_type' in batch_df.columns:
        for idx, seg_id in enumerate(segment_ids):
            split_type = batch_df[batch_df[seg_column] == seg_id]['split_type'].iloc[0]
            split_types[seg_id] = split_type

    # Plot first 3 samples
    # num_samples = min(3, batch_size)
    num_samples = batch_size
    for sample_idx in range(num_samples):
        seg_id = segment_ids[sample_idx] if sample_idx < len(segment_ids) else f"sample_{sample_idx}"
        seq_len_i = int(x_lens[sample_idx])

        # Get valid data first to calculate accuracy early
        X_valid = pad_X[sample_idx, :seq_len_i, :, :]  # [seq_len_i, patch_size, 5]
        y_labels = pad_Y[sample_idx, :seq_len_i]  # [seq_len_i]
        y_preds_logits = predictions[sample_idx:sample_idx+1, :seq_len_i, :]  # [1, seq_len_i, 3]
        y_preds_raw = np.argmax(y_preds_logits[0], axis=-1)  # [seq_len_i] - raw predictions

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
            y_preds = np.argmax(y_preds_logits_smoothed[0], axis=-1)  # [seq_len_i] - smoothed predictions
            accuracy_raw = (y_preds_raw == y_labels).sum() / len(y_labels) * 100
            accuracy_smoothed = (y_preds == y_labels).sum() / len(y_labels) * 100
            accuracy = accuracy_smoothed  # Use smoothed accuracy for folder placement
        else:
            y_preds = y_preds_raw
            accuracy = (y_preds == y_labels).sum() / len(y_labels) * 100
            accuracy_raw = accuracy
            accuracy_smoothed = accuracy

        accuracy_threshold = 90.0

        # Determine output subfolder based on split_type first
        if seg_id in split_types:
            sample_output_folder = os.path.join(output_folder, split_types[seg_id])
        else:
            sample_output_folder = output_folder

        # Then add low_acc subfolder if accuracy is below threshold
        if accuracy < accuracy_threshold:
            sample_output_folder = os.path.join(sample_output_folder, 'low_acc')

        os.makedirs(sample_output_folder, exist_ok=True)

        # Create time labels for x-axis (19:00 to 19:00 next day)
        # Start at 19:00 (1140 minutes from midnight)
        start_hour = 19
        time_labels = []
        for minute in range(seq_len_i):
            total_minutes = (start_hour * 60 + minute) % 1440  # Wrap around at 24 hours
            hour = (total_minutes // 60) % 24
            min_val = total_minutes % 60
            time_labels.append(f"{hour:02d}:{min_val:02d}")

        # For plotting, use minute indices
        time_minutes = np.arange(seq_len_i)

        # Count labels
        label_counts_gt = {
            'other': (y_labels == 0).sum(),
            'non-wear': (y_labels == 1).sum(),
            'predictTSO': (y_labels == 2).sum()
        }
        label_counts_pred = {
            'other': (y_preds == 0).sum(),
            'non-wear': (y_preds == 1).sum(),
            'predictTSO': (y_preds == 2).sum()
        }

        # Create figure - add extra row if smoothing is enabled
        num_rows = 5 if smooth_preds else 4
        fig, axes = plt.subplots(num_rows, 1, figsize=(16, 3*num_rows))

        title = f'Sample {sample_idx}: {seg_id} ({seq_len_i} minutes)\n'
        if smooth_preds:
            title += f'GT: {label_counts_gt} | Raw Acc: {accuracy_raw:.1f}% | Smoothed Acc: {accuracy_smoothed:.1f}%'
        else:
            title += f'GT: {label_counts_gt} | Pred: {label_counts_pred} | Acc: {accuracy:.1f}%'
        fig.suptitle(title, fontsize=14, fontweight='bold')

        # === ROW 1: Raw sensor data (minute-level aggregates) - All channels in one plot ===
        # Aggregate each minute's patch to mean values for visualization
        x_minute_mean = X_valid[:, :, 0].mean(axis=1)  # x-axis mean per minute
        y_minute_mean = X_valid[:, :, 1].mean(axis=1)  # y-axis mean per minute
        z_minute_mean = X_valid[:, :, 2].mean(axis=1)  # z-axis mean per minute
        temp_minute_mean = X_valid[:, :, 3].mean(axis=1)  # temperature mean per minute
        time_minute_mean = X_valid[:, :, 4].mean(axis=1)  # time_cyclic mean per minute

        # Helper function to set hourly x-axis ticks
        def set_hourly_xticks(ax, seq_len, start_hour=19):
            """Set x-axis to show hourly ticks from start_hour to start_hour (24h cycle)"""
            # Calculate tick positions (every 60 minutes)
            tick_positions = np.arange(0, seq_len, 60)
            tick_labels = []
            for pos in tick_positions:
                hour = (start_hour + pos // 60) % 24
                tick_labels.append(f"{hour:02d}:00")
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=45, ha='right')
            ax.set_xlim(0, seq_len - 1)

        # Accelerometer data (X, Y, Z)
        axes[0].plot(time_minutes, x_minute_mean, label='X-axis', alpha=0.8, linewidth=1.5)
        axes[0].plot(time_minutes, y_minute_mean, label='Y-axis', alpha=0.8, linewidth=1.5, color='orange')
        axes[0].plot(time_minutes, z_minute_mean, label='Z-axis', alpha=0.8, linewidth=1.5, color='green')
        axes[0].set_ylabel('Acceleration (g)')
        axes[0].set_title('Raw Accelerometer Data (minute-level mean)')
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)
        set_hourly_xticks(axes[0], seq_len_i)

        # Temperature and Time cyclic
        axes[1].plot(time_minutes, temp_minute_mean, label='Temperature', alpha=0.8, linewidth=1.5, color='red')
        ax1_twin = axes[1].twinx()
        ax1_twin.plot(time_minutes, time_minute_mean, label='Time Cyclic', alpha=0.8, linewidth=1.5, color='purple')
        axes[1].set_ylabel('Temperature', color='red')
        ax1_twin.set_ylabel('Time Cyclic', color='purple')
        axes[1].set_title('Temperature and Time Encoding (minute-level mean)')
        axes[1].tick_params(axis='y', labelcolor='red')
        ax1_twin.tick_params(axis='y', labelcolor='purple')
        axes[1].grid(True, alpha=0.3)
        # Combine legends
        lines1, labels1 = axes[1].get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        axes[1].legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        set_hourly_xticks(axes[1], seq_len_i)

        # === ROW 2: Ground Truth Labels ===
        axes[2].plot(time_minutes, y_labels, linewidth=2, color='black', label='Ground Truth')
        axes[2].set_ylabel('Label Class')
        axes[2].set_title('Ground Truth Labels (0=other, 1=non-wear, 2=predictTSO)')
        axes[2].set_yticks([0, 1, 2])
        axes[2].set_yticklabels(['other', 'non-wear', 'predictTSO'])
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        # Add colored background regions
        for label_val, color in [(0, 'lightblue'), (1, 'lightcoral'), (2, 'lightgreen')]:
            label_regions = (y_labels == label_val)
            if label_regions.any():
                axes[2].fill_between(time_minutes, 0, 2.5, where=label_regions, alpha=0.2, color=color)
        set_hourly_xticks(axes[2], seq_len_i)

        # === ROW 3: Raw Predictions (if smoothing enabled) ===
        if smooth_preds:
            axes[3].plot(time_minutes, y_preds_raw, linewidth=2, color='orange', label='Raw Predicted', alpha=0.7)
            axes[3].plot(time_minutes, y_labels, linewidth=1, color='black', linestyle='--', label='Ground Truth', alpha=0.5)
            axes[3].set_ylabel('Label Class')
            axes[3].set_title(f'Raw Predictions (before smoothing) - Acc: {accuracy_raw:.1f}%')
            axes[3].set_yticks([0, 1, 2])
            axes[3].set_yticklabels(['other', 'non-wear', 'predictTSO'])
            axes[3].legend()
            axes[3].grid(True, alpha=0.3)
            # Add colored background for raw predictions
            for label_val, color in [(0, 'lightblue'), (1, 'lightcoral'), (2, 'lightgreen')]:
                pred_regions = (y_preds_raw == label_val)
                if pred_regions.any():
                    axes[3].fill_between(time_minutes, 0, 2.5, where=pred_regions, alpha=0.15, color=color)
            set_hourly_xticks(axes[3], seq_len_i)

            # === ROW 4: Smoothed Predictions ===
            axes[4].plot(time_minutes, y_preds, linewidth=2, color='red', label=f'Smoothed ({smooth_method}, w={smooth_window})', alpha=0.7)
            axes[4].plot(time_minutes, y_labels, linewidth=1, color='black', linestyle='--', label='Ground Truth', alpha=0.5)
            axes[4].set_ylabel('Label Class')
            axes[4].set_xlabel('Time (HH:MM)')
            axes[4].set_title(f'Smoothed Predictions - Acc: {accuracy_smoothed:.1f}% (Δ: {accuracy_smoothed-accuracy_raw:+.1f}%)')
            axes[4].set_yticks([0, 1, 2])
            axes[4].set_yticklabels(['other', 'non-wear', 'predictTSO'])
            axes[4].legend()
            axes[4].grid(True, alpha=0.3)
            # Add colored background for smoothed predictions
            for label_val, color in [(0, 'lightblue'), (1, 'lightcoral'), (2, 'lightgreen')]:
                pred_regions = (y_preds == label_val)
                if pred_regions.any():
                    axes[4].fill_between(time_minutes, 0, 2.5, where=pred_regions, alpha=0.15, color=color)
            set_hourly_xticks(axes[4], seq_len_i)
        else:
            # === ROW 3: Predictions (no smoothing) ===
            axes[3].plot(time_minutes, y_preds, linewidth=2, color='red', label='Predicted', alpha=0.7)
            axes[3].plot(time_minutes, y_labels, linewidth=1, color='black', linestyle='--', label='Ground Truth', alpha=0.5)
            axes[3].set_ylabel('Label Class')
            axes[3].set_xlabel('Time (HH:MM)')
            axes[3].set_title('Predictions vs Ground Truth')
            axes[3].set_yticks([0, 1, 2])
            axes[3].set_yticklabels(['other', 'non-wear', 'predictTSO'])
            axes[3].legend()
            axes[3].grid(True, alpha=0.3)
            # Add colored background for predictions
            for label_val, color in [(0, 'lightblue'), (1, 'lightcoral'), (2, 'lightgreen')]:
                pred_regions = (y_preds == label_val)
                if pred_regions.any():
                    axes[3].fill_between(time_minutes, 0, 2.5, where=pred_regions, alpha=0.15, color=color)
            axes[3].text(0.02, 0.95, f'Accuracy: {accuracy:.1f}%',
                        transform=axes[3].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            set_hourly_xticks(axes[3], seq_len_i)

        plt.tight_layout()
        # save accuracy to 0.01 decimal places
        plot_file = os.path.join(sample_output_folder, f"batch_predictions_{sample_idx}_{seg_id.replace('/', '_')}_{timestamp}_Acc={accuracy:.2f}.png")
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {plot_file}")

    print(f"{'='*80}\n")

# NOTE: plot_tso_learning_curves is now imported from DL_helpers
# This matches the implementation in predict_TSO_segment_patch_h5.py

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
            'X': self.X[actual_idx],  # [max_len, num_channels]
            'Y': self.Y[actual_idx],  # [max_len, 2]
            'seq_length': self.seq_lengths[actual_idx],
            'segment': self.segment_names[actual_idx]
        }

    def close(self):
        """Close H5 file."""
        self.h5f.close()


def batch_generator_h5(dataset, batch_size, shuffle=False):
    """
    Generate batches of indices for H5 dataset.

    Args:
        dataset: H5Dataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle indices

    Yields:
        List of sample dictionaries for each batch
    """
    indices = np.arange(len(dataset))

    if shuffle:
        np.random.shuffle(indices)

    for start_idx in range(0, len(indices), batch_size):
        batch_indices = indices[start_idx:start_idx + batch_size]
        yield batch_indices


# ==================== Argument Parser ====================
parser = argparse.ArgumentParser(description='TSO Status Prediction with DLR-TC (Dynamic Label Refinement) - H5 Format')
parser.add_argument('--input_h5', type=str, required=True, help='Path to input H5 file')
parser.add_argument('--output', type=str, required=True, help='Output folder name')
parser.add_argument('--model', type=str, default="mba4tso_patch", help='Model name (default: mba4tso_patch)')
parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs (overridden by dlrtc_config)')
parser.add_argument('--num_gpu', type=str, default="0", help='GPU device ID (single GPU) or comma-separated list (multi-GPU, e.g., "0,1,2,3")')
parser.add_argument('--multi_gpu', action='store_true', help='Enable multi-GPU training with DataParallel')
parser.add_argument('--training_iterations', type=int, default=1, help='Number of training iterations')
parser.add_argument('--test_only', action='store_true', help='Skip training and only run testing with existing models')
parser.add_argument('--visualize', action='store_true', help='Visualize validation and test batches')
parser.add_argument('--smooth_predictions', action='store_true', help='Apply prediction smoothing for visualization')
parser.add_argument('--smooth_method', type=str, default='combined',
                   choices=['majority_vote', 'median_filter', 'moving_average', 'gaussian', 'min_segment', 'combined'],
                   help='Smoothing method for predictions')
parser.add_argument('--smooth_window', type=int, default=5, help='Smoothing window size (should be odd, e.g., 5, 7, 9)')

# Data split arguments
parser.add_argument('--split_file', type=str, help='Path to .npz file with train/val split indices')
parser.add_argument('--val_size', type=float, default=0.2, help='Validation split ratio (default: 0.2)')

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

# Helper function to create folders
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
    'batch_size': 24,  # Reduced due to larger memory footprint of raw patches
    'num_filters': 128,
    'dropout': 0.3,
    'droppath': 0.3,
    'kernel_f': 3,
    'kernel_MBA': 7,
    'num_feature_layers': 6,  # Reduced from 7 for efficiency
    'blocks_MBA': 5,
    'featurelayer': 'ResNet',
    'lr': 0.001,
    'w_other': 1.0,
    'w_nonwear': 1.0,
    'w_tso': 1.0,
    'padding_value': 0.0,
    'patch_size': 1200,  # 60 seconds * 20Hz
    'patch_channels': 5,  # x, y, z, temperature, time_cyclic (updated dynamically based on use_sincos)
    'norm1': 'BN',
    'norm2': 'GN',  # Changed from 'GN' to 'IN' to match MBA4TSO default
    'output_channels': 3,
    'skip_connect': True,  # Enable UNet-style skip connections
    'skip_cross_attention': True,  # Use simple addition instead of cross-attention

    # Data preprocessing configuration (MUST match training!)
    'use_sincos': True,  # If True: 6 channels (sin+cos), If False: 5 channels (sin only)
    'scaler_path': "/mnt/data/GENEActive-featurized/results/DL/UKB_v2/mbav1_scaler.joblib",  # Path to pretrained scaler (.joblib). Set to None for no scaling.
                          # Example: '/mnt/data/GENEActive-featurized/results/DL/UKB_v2/mbav1_scaler.joblib'

    # Model weights for testing (used when --test_only flag is set)
    'model_path': None,  # Path to pretrained model checkpoint (.pt file). Required for test_only mode.
                         # Example: '/mnt/data/GENEActive-featurized/results/DL/UKB_v2/TSO_predict-mba4tso_patch-bs=24/training/model_weights/best_model_iter_0.pt'

    # Reproducibility
    'random_seed': 42,  # Random seed for numpy, torch, and python random. Set to None for non-deterministic behavior.
}


print("Model hyperparameters:")
pprint(best_params)

# ==================== Data Preprocessing Configuration ====================
# Update patch_channels based on time encoding method
use_sincos = best_params['use_sincos']
best_params['patch_channels'] = 6 if use_sincos else 5
num_channels = best_params['patch_channels']

print(f"\nData Preprocessing Configuration:")
print(f"  Time encoding: {'sin+cos (6 channels)' if use_sincos else 'sin only (5 channels)'}")
print(f"  Total input channels: {num_channels}")

# Load scaler if path is provided in best_params
scaler = None
scaler_path = best_params.get('scaler_path')
if scaler_path:
    print(f"  Scaler: Loading from {scaler_path}")
    scaler = joblib.load(scaler_path)
    print(f"  Scaler loaded: {type(scaler).__name__}")
    print(f"  Scaling strategy: Scale during loading (cached for efficiency)")
else:
    print(f"  Scaler: None (raw data, no scaling)")

print(f"  IMPORTANT: These settings must match your training configuration!\n")

# ==================== Validate Configuration ====================
# Check if test_only mode requires model_path
if args.test_only:
    model_path = best_params.get('model_path')
    if model_path is None:
        raise ValueError(
            "ERROR: --test_only mode requires 'model_path' in best_params!\n"
            "Please set: best_params['model_path'] = '/path/to/your/model.pt'\n"
            "Example: best_params['model_path'] = '/mnt/data/.../best_model_iter_0.pt'"
        )
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"ERROR: Model checkpoint not found: {model_path}\n"
            f"Please verify the path in best_params['model_path']"
        )
    print(f"Test-only mode: Will load model from {model_path}\n")

# ==================== Load H5 Data ====================
print(f"\n{'='*80}")
print(f"Loading H5 data from: {args.input_h5}")
print(f"{'='*80}\n")

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

    # Set random seed for reproducibility
    random_seed = best_params.get('random_seed', 42)
    np.random.seed(random_seed)

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
    print(f"  Time encoding: Sin+Cos (6 channels)")
elif dataset_train.num_channels == 5:
    print(f"  Time encoding: Sin only (5 channels)")

# Update use_sincos based on H5 metadata
use_sincos = (dataset_train.num_channels == 6)
best_params['use_sincos'] = use_sincos
best_params['patch_channels'] = dataset_train.num_channels

# ==================== DLR-TC Training Functions ====================

def run_model_tso_patch_dlrtc_warmup(model, dataset, batch_size, train_mode, device, optimizer, scheduler,
                                     max_seq_len=None, patch_size=1200, padding_value=0.0,
                                     verbose=False):
    """
    Phase 1: Warmup training with standard Cross Entropy

    Purpose: Let the model learn easy, clean patterns before label refinement begins.
    This provides a good initialization for the joint refinement phase.

    Args:
        model: The model to train
        dataset: H5Dataset instance
        batch_size: Batch size
        train_mode: Whether to train (True) or evaluate (False)
        device: Device to use
        optimizer: Optimizer instance
        scheduler: Learning rate scheduler
        max_seq_len: Maximum sequence length in minutes
        patch_size: Patch size (samples per patch)
        padding_value: Padding value
        verbose: Whether to print verbose output

    Returns:
        model, metrics, predictions
    """
    epoch_loss = 0.0
    batches = 0

    all_preds_other = []
    all_preds_nonwear = []
    all_preds_tso = []
    all_labels_other = []
    all_labels_nonwear = []
    all_labels_tso = []

    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    for batch_indices in batch_generator_h5(dataset, batch_size=batch_size, shuffle=train_mode):

        # Prepare batch data
        pad_X, pad_Y, x_lens, batch_samples = add_padding_tso_patch_h5(
            dataset, batch_indices, device,
            max_seq_len=max_seq_len,
            patch_size=patch_size,
            padding_value=padding_value,
            num_channels=dataset.num_channels
        )

        # Forward pass
        outputs = model(pad_X, x_lens)  # [batch_size, seq_len, 3]

        # Calculate standard CE loss
        # Reshape for CrossEntropyLoss: [batch*seq, 3] and [batch*seq]
        outputs_flat = outputs.view(-1, 3)
        labels_flat = pad_Y.view(-1).long()
        
        loss = criterion(outputs_flat, labels_flat)

        # Backward pass if training
        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

        # Record loss
        epoch_loss += loss.item()
        batches += 1

        # Get predictions
        preds = torch.softmax(outputs, dim=-1).cpu().detach().numpy()  # [batch_size, seq_len, 3]
        labels = pad_Y.cpu().numpy()  # [batch_size, seq_len]

        # Collect valid predictions
        for i in range(len(x_lens)):
            valid_len = int(x_lens[i])
            valid_preds = preds[i, :valid_len, :]
            valid_labels = labels[i, :valid_len]

            # Convert to one-hot
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

    # Calculate metrics
    all_preds_other = np.array(all_preds_other)
    all_preds_nonwear = np.array(all_preds_nonwear)
    all_preds_tso = np.array(all_preds_tso)
    all_labels_other = np.array(all_labels_other)
    all_labels_nonwear = np.array(all_labels_nonwear)
    all_labels_tso = np.array(all_labels_tso)

    avg_loss = epoch_loss / batches if batches > 0 else 0

    f1_other = f1_score(all_labels_other, (all_preds_other > 0.5).astype(int), zero_division=0)
    f1_nonwear = f1_score(all_labels_nonwear, (all_preds_nonwear > 0.5).astype(int), zero_division=0)
    f1_tso = f1_score(all_labels_tso, (all_preds_tso > 0.5).astype(int), zero_division=0)
    f1_avg = (f1_other + f1_nonwear + f1_tso) / 3

    all_preds_stacked = np.stack([all_preds_other, all_preds_nonwear, all_preds_tso], axis=1)
    all_labels_stacked = np.stack([all_labels_other, all_labels_nonwear, all_labels_tso], axis=1)
    pred_classes = np.argmax(all_preds_stacked, axis=1)
    true_classes = np.argmax(all_labels_stacked, axis=1)
    accuracy = np.mean(pred_classes == true_classes)

    metrics = {
        'loss': avg_loss,
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
        print(f"  [Warmup] Loss: {avg_loss:.4f} | F1 avg: {f1_avg:.4f} | "
              f"F1 other: {f1_other:.4f} | F1 nonwear: {f1_nonwear:.4f} | F1 TSO: {f1_tso:.4f}")

    return model, metrics, predictions


def run_model_tso_patch_dlrtc_joint(model, dataset, batch_size, device,
                                    model_optimizer, scheduler,
                                    soft_label_manager, dlrtc_loss,
                                    label_lr=0.01, max_seq_len=None,
                                    patch_size=1200, padding_value=0.0,
                                    verbose=False):
    """
    Phase 2: Joint refinement - alternating between label updates and model updates

    For each batch:
        1. Update soft labels (freeze model)
        2. Update model parameters (freeze labels)

    Args:
        model: The model to train
        dataset: H5Dataset instance
        batch_size: Batch size
        device: Device to use
        model_optimizer: Optimizer for model parameters
        scheduler: Learning rate scheduler
        soft_label_manager: SoftLabelManager instance
        dlrtc_loss: DLRTCLoss instance
        label_lr: Learning rate for label updates
        max_seq_len: Maximum sequence length in minutes
        patch_size: Patch size (samples per patch)
        padding_value: Padding value
        verbose: Whether to print verbose output

    Returns:
        model, soft_label_manager, metrics, predictions
    """
    epoch_loss = 0.0
    epoch_loss_gce = 0.0
    epoch_loss_compat = 0.0
    epoch_loss_temp = 0.0
    batches = 0

    all_preds_other = []
    all_preds_nonwear = []
    all_preds_tso = []
    all_labels_other = []
    all_labels_nonwear = []
    all_labels_tso = []

    for batch_indices in batch_generator_h5(dataset, batch_size=batch_size, shuffle=True):

        # Prepare batch data
        pad_X, pad_Y, x_lens, batch_samples = add_padding_tso_patch_h5(
            dataset, batch_indices, device,
            max_seq_len=max_seq_len,
            patch_size=patch_size,
            padding_value=padding_value,
            num_channels=dataset.num_channels
        )

        # Get segment IDs from batch_samples
        segment_ids = [sample['segment'].decode('utf-8') if isinstance(sample['segment'], bytes) else sample['segment']
                       for sample in batch_samples]

        # Initialize soft labels if needed (first time seeing these segments)
        soft_label_manager.initialize_from_batch(segment_ids, pad_Y, x_lens, temperature=1.0)

        # Get current soft labels (trainable)
        max_len = pad_X.size(1)
        batch_soft_labels = soft_label_manager.get_batch_soft_labels(segment_ids, x_lens, max_len)

        # Convert hard labels to one-hot for compatibility loss
        pad_Y_onehot = torch.zeros(pad_Y.size(0), pad_Y.size(1), 3, device=device)
        for i in range(pad_Y.size(0)):
            for j in range(int(x_lens[i])):
                if pad_Y[i, j] >= 0 and pad_Y[i, j] < 3:
                    pad_Y_onehot[i, j, int(pad_Y[i, j])] = 1.0

        # Create mask for valid positions (boolean)
        mask = torch.zeros(pad_Y.size(0), pad_Y.size(1), device=device, dtype=torch.bool)
        for i in range(len(x_lens)):
            mask[i, :int(x_lens[i])] = True

        # === Step 1: Update soft labels (freeze model) ===
        model.eval()  # Freeze model

        # Collect raw parameters that require gradients
        raw_params = []
        for seg_id in segment_ids:
            if seg_id in soft_label_manager.soft_labels_dict:
                raw_params.append(soft_label_manager.soft_labels_dict[seg_id])

        # Forward pass to compute gradients w.r.t. raw parameters
        outputs = model(pad_X, x_lens)  # [batch_size, seq_len, 3]

        # Reconstruct batch_soft_labels with gradient tracking
        batch_soft_labels_grad = torch.zeros(batch_soft_labels.size(), device=device, requires_grad=False)
        for i, seg_id in enumerate(segment_ids):
            seq_len = int(x_lens[i])
            if seg_id in soft_label_manager.soft_labels_dict:
                soft = soft_label_manager.soft_labels_dict[seg_id]
                soft_probs = F.softmax(soft, dim=-1)
                batch_soft_labels_grad[i, :seq_len, :] = soft_probs

        # Calculate DLR-TC loss
        loss_dict = dlrtc_loss(outputs.detach(), batch_soft_labels_grad, pad_Y_onehot, mask)
        label_loss = loss_dict['total']

        # Compute gradients w.r.t. raw parameters
        if raw_params:
            grads = torch.autograd.grad(label_loss, raw_params, create_graph=False, allow_unused=True)

            # Update soft labels using computed gradients
            for seg_id, grad in zip(segment_ids, grads):
                if grad is not None and seg_id in soft_label_manager.soft_labels_dict:
                    with torch.no_grad():
                        soft_label_manager.soft_labels_dict[seg_id] -= label_lr * grad

        # === Step 2: Update model (freeze labels) ===
        model.train()
        
        # Get updated soft labels (no grad)
        with torch.no_grad():
            batch_soft_labels_updated = soft_label_manager.get_batch_soft_labels(segment_ids, x_lens, max_len)
        
        # Forward pass
        outputs = model(pad_X, x_lens)
        
        # Calculate DLR-TC loss
        loss_dict = dlrtc_loss(outputs, batch_soft_labels_updated, pad_Y_onehot, mask)
        total_loss = loss_dict['total']
        
        # Backward pass for model
        model_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        model_optimizer.step()
        scheduler.step()

        # Record losses
        epoch_loss += total_loss.item()
        epoch_loss_gce += loss_dict['gce'].item()
        epoch_loss_compat += loss_dict['compatibility'].item()
        epoch_loss_temp += loss_dict['temporal'].item()
        batches += 1

        # Get predictions
        preds = torch.softmax(outputs, dim=-1).cpu().detach().numpy()
        labels = pad_Y.cpu().numpy()

        # Collect valid predictions
        for i in range(len(x_lens)):
            valid_len = int(x_lens[i])
            valid_preds = preds[i, :valid_len, :]
            valid_labels = labels[i, :valid_len]

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

    # Calculate metrics
    all_preds_other = np.array(all_preds_other)
    all_preds_nonwear = np.array(all_preds_nonwear)
    all_preds_tso = np.array(all_preds_tso)
    all_labels_other = np.array(all_labels_other)
    all_labels_nonwear = np.array(all_labels_nonwear)
    all_labels_tso = np.array(all_labels_tso)

    avg_loss = epoch_loss / batches if batches > 0 else 0
    avg_loss_gce = epoch_loss_gce / batches if batches > 0 else 0
    avg_loss_compat = epoch_loss_compat / batches if batches > 0 else 0
    avg_loss_temp = epoch_loss_temp / batches if batches > 0 else 0

    f1_other = f1_score(all_labels_other, (all_preds_other > 0.5).astype(int), zero_division=0)
    f1_nonwear = f1_score(all_labels_nonwear, (all_preds_nonwear > 0.5).astype(int), zero_division=0)
    f1_tso = f1_score(all_labels_tso, (all_preds_tso > 0.5).astype(int), zero_division=0)
    f1_avg = (f1_other + f1_nonwear + f1_tso) / 3

    all_preds_stacked = np.stack([all_preds_other, all_preds_nonwear, all_preds_tso], axis=1)
    all_labels_stacked = np.stack([all_labels_other, all_labels_nonwear, all_labels_tso], axis=1)
    pred_classes = np.argmax(all_preds_stacked, axis=1)
    true_classes = np.argmax(all_labels_stacked, axis=1)
    accuracy = np.mean(pred_classes == true_classes)

    metrics = {
        'loss': avg_loss,
        'loss_gce': avg_loss_gce,
        'loss_compat': avg_loss_compat,
        'loss_temp': avg_loss_temp,
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
        print(f"  [Joint] Loss: {avg_loss:.4f} (GCE: {avg_loss_gce:.4f}, "
              f"Compat: {avg_loss_compat:.4f}, Temp: {avg_loss_temp:.4f}) | "
              f"F1 avg: {f1_avg:.4f} | F1 other: {f1_other:.4f} | "
              f"F1 nonwear: {f1_nonwear:.4f} | F1 TSO: {f1_tso:.4f}")

    return model, soft_label_manager, metrics, predictions


def run_model_tso_patch_dlrtc_eval(model, dataset, batch_size, device,
                                   max_seq_len=None, patch_size=1200,
                                   padding_value=0.0, verbose=False):
    """
    Evaluation mode (no training, no label refinement)

    Args:
        model: The model to evaluate
        dataset: H5Dataset instance
        batch_size: Batch size
        device: Device to use
        max_seq_len: Maximum sequence length in minutes
        patch_size: Patch size (samples per patch)
        padding_value: Padding value
        verbose: Whether to print verbose output

    Returns:
        metrics, predictions
    """
    all_preds_other = []
    all_preds_nonwear = []
    all_preds_tso = []
    all_labels_other = []
    all_labels_nonwear = []
    all_labels_tso = []

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    epoch_loss = 0.0
    batches = 0

    model.eval()

    with torch.no_grad():
        for batch_indices in batch_generator_h5(dataset, batch_size=batch_size, shuffle=False):

            # Prepare batch data
            pad_X, pad_Y, x_lens, batch_samples = add_padding_tso_patch_h5(
                dataset, batch_indices, device,
                max_seq_len=max_seq_len,
                patch_size=patch_size,
                padding_value=padding_value,
                num_channels=dataset.num_channels
            )

            # Forward pass
            outputs = model(pad_X, x_lens)

            # Calculate loss for monitoring
            outputs_flat = outputs.view(-1, 3)
            labels_flat = pad_Y.view(-1).long()
            loss = criterion(outputs_flat, labels_flat)
            epoch_loss += loss.item()
            batches += 1

            # Get predictions
            preds = torch.softmax(outputs, dim=-1).cpu().numpy()
            labels = pad_Y.cpu().numpy()

            # Collect valid predictions
            for i in range(len(x_lens)):
                valid_len = int(x_lens[i])
                valid_preds = preds[i, :valid_len, :]
                valid_labels = labels[i, :valid_len]

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

    # Calculate metrics
    all_preds_other = np.array(all_preds_other)
    all_preds_nonwear = np.array(all_preds_nonwear)
    all_preds_tso = np.array(all_preds_tso)
    all_labels_other = np.array(all_labels_other)
    all_labels_nonwear = np.array(all_labels_nonwear)
    all_labels_tso = np.array(all_labels_tso)

    avg_loss = epoch_loss / batches if batches > 0 else 0

    f1_other = f1_score(all_labels_other, (all_preds_other > 0.5).astype(int), zero_division=0)
    f1_nonwear = f1_score(all_labels_nonwear, (all_preds_nonwear > 0.5).astype(int), zero_division=0)
    f1_tso = f1_score(all_labels_tso, (all_preds_tso > 0.5).astype(int), zero_division=0)
    f1_avg = (f1_other + f1_nonwear + f1_tso) / 3

    all_preds_stacked = np.stack([all_preds_other, all_preds_nonwear, all_preds_tso], axis=1)
    all_labels_stacked = np.stack([all_labels_other, all_labels_nonwear, all_labels_tso], axis=1)
    pred_classes = np.argmax(all_preds_stacked, axis=1)
    true_classes = np.argmax(all_labels_stacked, axis=1)
    accuracy = np.mean(pred_classes == true_classes)

    metrics = {
        'loss': avg_loss,
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
        print(f"  [Eval] Loss: {avg_loss:.4f} | F1 avg: {f1_avg:.4f} | "
              f"F1 other: {f1_other:.4f} | F1 nonwear: {f1_nonwear:.4f} | F1 TSO: {f1_tso:.4f}")

    return metrics, predictions



# ==================== Main Training Loop with DLR-TC ====================
print(f"\n{'='*80}")
print(f"STARTING DLR-TC TRAINING")
print(f"{'='*80}\n")

# DLR-TC hyperparameters
# Optimized for large-scale dataset (34K+ segments, 900GB data)
dlrtc_config = {
    # Phase durations (TOTAL: 35 epochs ≈ 45-50 hours training)
    'warmup_epochs': 10,  # Warmup with standard CE (increased for large dataset)
                          # Large datasets need more epochs to learn general patterns
                          # Expect: F1 plateau around epoch 7-9

    'joint_epochs': 25,   # Joint label-model refinement (increased for thorough refinement)
                          # More segments = more labels to refine = more epochs needed
                          # Expect: F1 peak around epoch 18-22, then stabilize

    # Loss hyperparameters
    'gce_q': 0.7,        # GCE truncation parameter (0.5=very robust, 0.9=less robust)
                         # 0.7 is good default for moderate label noise

    'alpha': 0.1,        # Compatibility loss weight (anchor to original labels)
                         # Higher = stay closer to original (0.05-0.5 range)
                         # 0.1 allows refinement while preventing drift

    'beta': 0.5,         # Temporal smoothness weight (continuity constraint)
                         # Higher = smoother predictions (0.2-1.0 range)
                         # 0.5 balances smoothness and event detection

    'label_lr': 0.01,    # Learning rate for soft label updates
                         # Higher = faster refinement but risk instability (0.001-0.05 range)
                         # 0.01 is stable for most cases
}

# Alternative configurations for different scenarios:
#
# CONSERVATIVE (limited compute, ~30-35 hours):
#   warmup_epochs=7, joint_epochs=18
#
# AGGRESSIVE (very noisy labels, strong compute, ~60-70 hours):
#   warmup_epochs=15, joint_epochs=35
#
# QUICK TEST (hyperparameter search, ~10-12 hours):
#   warmup_epochs=3, joint_epochs=7

print("DLR-TC Configuration:")
for key, value in dlrtc_config.items():
    print(f"  {key}: {value}")
print()

# Training loop (single iteration for H5)
for iteration in range(args.training_iterations):
    print(f"\n{'='*60}")
    print(f"Iteration {iteration + 1}/{args.training_iterations}")
    print(f"{'='*60}")

    # Setup model
    model = setup_model(args.model, None, max_seq_len, best_params,
                      pretraining=False, num_classes=3)
    model = model.to(device)

    # Wrap with DataParallel for multi-GPU
    if args.multi_gpu and gpu_ids is not None and len(gpu_ids) > 1:
        print(f"Wrapping model with DataParallel for GPUs: {gpu_ids}")
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}, Trainable: {trainable_params:,}")

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=best_params['lr'], weight_decay=1e-5)

    # Calculate total steps for H5 data
    nb_steps = len(dataset_train) // best_params['batch_size']
    total_steps = nb_steps * (dlrtc_config['warmup_epochs'] + dlrtc_config['joint_epochs'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    # Initialize DLR-TC components
    dlrtc_loss = DLRTCLoss(
        q=dlrtc_config['gce_q'],
        alpha=dlrtc_config['alpha'],
        beta=dlrtc_config['beta'],
        num_classes=3
    )
    soft_label_manager = SoftLabelManager(num_classes=3, device=device)

    # Training history
    history = {
        'train_loss': [], 'train_accuracy': [], 'train_f1_avg': [],
        'train_f1_other': [], 'train_f1_nonwear': [], 'train_f1_tso': [],
        'val_loss': [], 'val_accuracy': [], 'val_f1_avg': [],
        'val_f1_other': [], 'val_f1_nonwear': [], 'val_f1_tso': [],
        'train_loss_gce': [], 'train_loss_compat': [], 'train_loss_temp': []
    }

    # Early stopping
    early_stopping = EarlyStopping(patience=10, verbose=True)

    # ========== PHASE 1: WARMUP TRAINING ==========
    print(f"\n{'='*60}")
    print(f"PHASE 1: WARMUP TRAINING ({dlrtc_config['warmup_epochs']} epochs)")
    print(f"{'='*60}")
    print("Purpose: Learn easy patterns with standard Cross Entropy")
    print()

    for epoch in range(dlrtc_config['warmup_epochs']):
        print(f"\n[Warmup] Epoch {epoch+1}/{dlrtc_config['warmup_epochs']}")

        # Train
        model.train()
        model, train_metrics, _ = run_model_tso_patch_dlrtc_warmup(
            model, dataset_train, best_params['batch_size'], True, device, optimizer, scheduler,
            max_seq_len=max_seq_len, patch_size=best_params['patch_size'],
            padding_value=best_params['padding_value'], verbose=(epoch % 5 == 0)
        )

        history['train_loss'].append(train_metrics['loss'])
        history['train_accuracy'].append(train_metrics['accuracy'])
        history['train_f1_avg'].append(train_metrics['f1_avg'])
        history['train_f1_other'].append(train_metrics['f1_other'])
        history['train_f1_nonwear'].append(train_metrics['f1_nonwear'])
        history['train_f1_tso'].append(train_metrics['f1_tso'])

        print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, "
              f"F1 avg: {train_metrics['f1_avg']:.4f}")

        # Validation
        val_metrics, _ = run_model_tso_patch_dlrtc_eval(
            model, dataset_val, best_params['batch_size'], device,
            max_seq_len=max_seq_len, patch_size=best_params['patch_size'],
            padding_value=best_params['padding_value'], verbose=False
        )

        history['val_loss'].append(val_metrics['loss'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_f1_avg'].append(val_metrics['f1_avg'])
        history['val_f1_other'].append(val_metrics['f1_other'])
        history['val_f1_nonwear'].append(val_metrics['f1_nonwear'])
        history['val_f1_tso'].append(val_metrics['f1_tso'])

        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
              f"F1 avg: {val_metrics['f1_avg']:.4f}")

        # Early stopping check
        early_stopping(val_metrics['loss'], model)
        if early_stopping.early_stop:
            print("Early stopping triggered during warmup")
            break

    print(f"\n{'='*60}")
    print(f"Warmup phase completed. Best val loss: {early_stopping.best_score:.4f}")
    print(f"{'='*60}")

    # ========== PHASE 2: JOINT REFINEMENT ==========
    print(f"\n{'='*60}")
    print(f"PHASE 2: JOINT LABEL-MODEL REFINEMENT ({dlrtc_config['joint_epochs']} epochs)")
    print(f"{'='*60}")
    print("Purpose: Refine noisy labels while training model")
    print()

    # Reset early stopping for joint phase
    early_stopping_joint = EarlyStopping(patience=10, verbose=True)

    for epoch in range(dlrtc_config['joint_epochs']):
        print(f"\n[Joint] Epoch {epoch+1}/{dlrtc_config['joint_epochs']}")

        # Joint training
        model.train()
        model, soft_label_manager, train_metrics, _ = run_model_tso_patch_dlrtc_joint(
            model, dataset_train, best_params['batch_size'], device, optimizer, scheduler,
            soft_label_manager, dlrtc_loss, label_lr=dlrtc_config['label_lr'],
            max_seq_len=max_seq_len, patch_size=best_params['patch_size'],
            padding_value=best_params['padding_value'], verbose=(epoch % 5 == 0)
        )

        history['train_loss'].append(train_metrics['loss'])
        history['train_accuracy'].append(train_metrics['accuracy'])
        history['train_f1_avg'].append(train_metrics['f1_avg'])
        history['train_f1_other'].append(train_metrics['f1_other'])
        history['train_f1_nonwear'].append(train_metrics['f1_nonwear'])
        history['train_f1_tso'].append(train_metrics['f1_tso'])
        history['train_loss_gce'].append(train_metrics['loss_gce'])
        history['train_loss_compat'].append(train_metrics['loss_compat'])
        history['train_loss_temp'].append(train_metrics['loss_temp'])

        print(f"  Train - Loss: {train_metrics['loss']:.4f} (GCE: {train_metrics['loss_gce']:.4f}, "
              f"Compat: {train_metrics['loss_compat']:.4f}, Temp: {train_metrics['loss_temp']:.4f})")
        print(f"          Acc: {train_metrics['accuracy']:.4f}, F1 avg: {train_metrics['f1_avg']:.4f}")

        # Validation
        val_metrics, _ = run_model_tso_patch_dlrtc_eval(
            model, dataset_val, best_params['batch_size'], device,
            max_seq_len=max_seq_len, patch_size=best_params['patch_size'],
            padding_value=best_params['padding_value'], verbose=False
        )

        history['val_loss'].append(val_metrics['loss'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_f1_avg'].append(val_metrics['f1_avg'])
        history['val_f1_other'].append(val_metrics['f1_other'])
        history['val_f1_nonwear'].append(val_metrics['f1_nonwear'])
        history['val_f1_tso'].append(val_metrics['f1_tso'])

        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
              f"F1 avg: {val_metrics['f1_avg']:.4f}")

        # Early stopping check
        early_stopping_joint(val_metrics['loss'], model)
        if early_stopping_joint.early_stop:
            print("Early stopping triggered during joint refinement")
            break

        # Save best model
        if val_metrics['loss'] == early_stopping_joint.best_score:
            model_path = os.path.join(train_models_folder,
                                    f"best_model_iter_{iteration}.pt")

            # Handle DataParallel: save underlying model, not wrapper
            model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model

            torch.save({
                'epoch': dlrtc_config['warmup_epochs'] + epoch,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'val_f1_avg': val_metrics['f1_avg'],
                'history': history,
                'dlrtc_config': dlrtc_config
            }, model_path)
            print(f"  -> Saved best model")

            # Save soft labels
            soft_label_path = os.path.join(train_models_folder,
                                          f"soft_labels_iter_{iteration}.pt")
            soft_label_manager.save_state(soft_label_path)

    print(f"\n{'='*60}")
    print(f"Joint refinement completed. Best val loss: {early_stopping_joint.best_score:.4f}")
    print(f"Refined {soft_label_manager.get_num_segments()} segment labels")
    print(f"{'='*60}")

    # ========== TESTING ==========
    print(f"\nEvaluating on test set...")
    model.eval()
    test_metrics, test_predictions = run_model_tso_patch_dlrtc_eval(
        model, dataset_test, best_params['batch_size'], device,
        max_seq_len=max_seq_len, patch_size=best_params['patch_size'],
        padding_value=best_params['padding_value'], verbose=True
    )

    print(f"\nTest Results:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  F1 avg: {test_metrics['f1_avg']:.4f}")
    print(f"  F1 other: {test_metrics['f1_other']:.4f}")
    print(f"  F1 nonwear: {test_metrics['f1_nonwear']:.4f}")
    print(f"  F1 TSO: {test_metrics['f1_tso']:.4f}")

    # Plot training history
    plot_tso_learning_curves(
        history=history,
        output_filepath=os.path.join(learning_plots_output_folder,
                                    f"dlrtc_training_history_iter_{iteration}.png")
    )
    print(f"Training history plot saved to learning_plots folder")

    # Save results in JSON format (human-readable and portable)
    results = {
        'iteration': iteration,
        'test_metrics': test_metrics,
        'history': history,
        'dlrtc_config': dlrtc_config,
        'num_refined_labels': soft_label_manager.get_num_segments(),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model': args.model,
        'total_epochs': dlrtc_config['warmup_epochs'] + dlrtc_config['joint_epochs'],
    }

    # Save as JSON
    results_file_json = os.path.join(predictions_output_folder,
                                    f"dlrtc_results_iter_{iteration}.json")
    with open(results_file_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_file_json}")

    # Also save as joblib for backward compatibility (contains full history arrays)
    results_file_joblib = os.path.join(predictions_output_folder,
                                      f"dlrtc_results_iter_{iteration}.joblib")
    joblib.dump(results, results_file_joblib)
    print(f"Results also saved to {results_file_joblib} (for programmatic access)")

    # Clean up
    torch.cuda.empty_cache()

print(f"\n{'='*80}")
print(f"DLR-TC TRAINING COMPLETE")
print(f"{'='*80}")
print(f"Completed at: {datetime.now()}")

