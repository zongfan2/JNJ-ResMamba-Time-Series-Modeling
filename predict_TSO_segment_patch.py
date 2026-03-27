# -*- coding: utf-8 -*-
"""
TSO Status Prediction Script - Patched Raw Sensor Input Version
Predicts 3-class status: 'other', 'non-wear', 'predictTSO'
Uses MBA4TSO_Patch model with raw sensor patches (20Hz data)
Input: Raw accelerometer data at 20Hz instead of aggregated features
"""
import subprocess

import joblib

shell_script = '''
sudo python3.11 -m pip install -r munge/predictive_modeling/requirements-tso.txt
sudo python3.11 -m pip install -e .
sudo python3.11 -m pip install optuna==4.3.0 seaborn ray TensorboardX torcheval ruptures  mamba-ssm[causal-conv1d]==2.2.2
'''
# shell_script = '''
# sudo python3.11 -m pip install -r munge/predictive_modeling/requirements-tso.txt
# '''
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

from Helpers.DL_models import setup_model
from Helpers.DL_helpers import (
    load_data_tso_patch,  # Load raw sensor data for patched TSO prediction
    load_data_tso_patch_biobank,
    batch_generator,
    add_padding_tso_patch,  # Use patched version instead of add_padding_TSO
    measure_loss_tso,
    EarlyStopping,
    get_nb_steps,
    smooth_predictions,  # Post-processing for predictions
    smooth_predictions_combined,  # Combined smoothing methods
    plot_tso_learning_curves  # Visualization
)

# NOTE: smooth_predictions and smooth_predictions_combined are now imported from DL_helpers
# This matches the implementation in predict_TSO_segment_patch_h5.py


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

# ==================== Argument Parser ====================
parser = argparse.ArgumentParser(description='TSO Status Prediction with Raw Sensor Patches')
parser.add_argument('--input_data_folder', type=str, required=True, help='Path to input data folder')
parser.add_argument('--output', type=str, required=True, help='Output folder name')
parser.add_argument('--model', type=str, default="mba4tso_patch", help='Model name (default: mba4tso_patch)')
parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
parser.add_argument('--num_gpu', type=str, default="0", help='GPU device ID')
parser.add_argument('--testing', type=str, default="LOFO", help='Testing strategy: LOFO or LOSO')
parser.add_argument('--training_iterations', type=int, default=1, help='Number of training iterations')
parser.add_argument('--clear_tracker', type=bool, default=False, required=False, help='Clear Job Tracking Folder')
parser.add_argument('--test_only', action='store_true', help='Skip training and only run testing with existing models')
parser.add_argument('--visualize', action='store_true', help='Visualize validation and test batches')
parser.add_argument('--smooth_predictions', action='store_true', help='Apply prediction smoothing for visualization')
parser.add_argument('--smooth_method', type=str, default='combined',
                   choices=['majority_vote', 'median_filter', 'moving_average', 'gaussian', 'min_segment', 'combined'],
                   help='Smoothing method for predictions')
parser.add_argument('--smooth_window', type=int, default=5, help='Smoothing window size (should be odd, e.g., 5, 7, 9)')

args = parser.parse_args()

# ==================== Setup ====================
device = torch.device(f"cuda:{args.num_gpu}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Starting training at: {datetime.now()}")

# Handle clear_tracker flag
clear_tracker_flag = str(args.clear_tracker)

# Create output folders
input_data_folder = args.input_data_folder
results_folder_name = args.output
dataset_name = os.path.basename(input_data_folder.rstrip("/raw"))
results_folder = f"/mnt/data/GENEActive-featurized/results/DL/{dataset_name}/{results_folder_name}/"
# results_folder = f"/mnt/data/GENEActive-featurized/results/DL/UKB_v2/{results_folder_name}/"

training_output_folder = os.path.join(results_folder, "training/")
job_tracking_folder = os.path.join(training_output_folder, "job_tracking_folder")
predictions_output_folder = os.path.join(training_output_folder, "predictions/")
confusion_matrix_plots_folder = os.path.join(training_output_folder, "confusion_matrix_plots/")
learning_plots_output_folder = os.path.join(training_output_folder, "learning_plots/")
train_models_folder = os.path.join(training_output_folder, "model_weights/")
train_models_folder_joblib = os.path.join(training_output_folder, "model_weights_joblib/")
checkpoint_folder = os.path.join(training_output_folder, "checkpoints/")
training_logs_folder = os.path.join(training_output_folder, "training_logs/")
processed_data_folder = os.path.join(input_data_folder.rstrip("/raw"), "processed_patch/")

# Clear job tracking folder if requested
if clear_tracker_flag.lower() == "true" and os.path.exists(job_tracking_folder):
    import shutil
    shutil.rmtree(job_tracking_folder, ignore_errors=True)
    print(f"Cleared job tracking folder: {job_tracking_folder}")

# Helper function to create folders
def create_folder(folders):
    for folder in folders:
        os.makedirs(folder, exist_ok=True)

create_folder([results_folder, training_output_folder, job_tracking_folder, predictions_output_folder,
               confusion_matrix_plots_folder, learning_plots_output_folder,
               train_models_folder, train_models_folder_joblib, training_logs_folder, processed_data_folder, checkpoint_folder])

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
    'patch_channels': 6,  # x, y, z, temperature, time_cyclic (updated dynamically based on use_sincos)
    'norm1': 'BN',
    'norm2': 'GN',  # Changed from 'GN' to 'IN' to match MBA4TSO default
    'output_channels': 1,
    'skip_connect': True,  # Enable UNet-style skip connections
    'skip_cross_attention': True,  # Use simple addition instead of cross-attention

    # Data preprocessing configuration (MUST match training!)
    'use_sincos': True,  # If True: 6 channels (sin+cos), If False: 5 channels (sin only)
    'scaler_path': "/mnt/data/GENEActive-featurized/results/DL/UKB_v2/mbav1_scaler.joblib",  # Path to pretrained scaler (.joblib). Set to None for no scaling.
                          # Example: '/mnt/data/GENEActive-featurized/results/DL/UKB_v2/mbav1_scaler.joblib'

    # Model weights for testing (used when --test_only flag is set)
    'model_path': "/mnt/data/GENEActive-featurized/results/DL/geneactive_20hz_3s_b1s_production_writeall/TSO_predict-mba4tso_patch-bs=24-dim=128-ch=6-no-ukb-binary/training/model_weights/best_model_iter_0.pt",  # Path to pretrained model checkpoint (.pt file). Required for test_only mode.
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

# ==================== Load Data ====================
print(f"Loading data from: {args.input_data_folder}")

# Load RAW sensor data (20Hz) for patched input
# Pass scaler object directly - no hardcoded paths!
df = load_data_tso_patch(args.input_data_folder, max_seq_length=86400, scaler=scaler)
# df = load_data_tso_patch_biobank(args.input_data_folder, max_seq_length=86400)

print(f"Data loaded. Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Calculate max sequence length in minutes (should be 1440 = 24 hours)
# Raw data at 20Hz: 60 seconds * 20 samples = 1200 samples per minute
samples_per_minute = 60 * 20
# max_seq_len = (df.groupby('segment').size().max() // samples_per_minute) + 1
max_seq_len = 1440
print(f"Max sequence length: {max_seq_len} minutes")

# Data statistics
print(f"\nData statistics:")
print(f"  Total 20Hz samples: {len(df)}")
print(f"  Total unique segments: {df['segment'].nunique()}")
print(f"  Total PIDs: {df['PID'].nunique()}")
if 'predictTSO' in df.columns:
    print(f"  predictTSO samples: {df[df['predictTSO']==True].shape[0]} ({100*df[df['predictTSO']==True].shape[0]/len(df):.1f}%)")
if 'non-wear' in df.columns:
    print(f"  non-wear samples: {df[df['non-wear']==1].shape[0]} ({100*df[df['non-wear']==1].shape[0]/len(df):.1f}%)")

# ==================== Define Splits ====================
if args.testing == "production":
    df["FOLD"] = "All"
    PID_name = "FOLD"
    splits = df[PID_name].unique()
    print(f"\nTesting strategy: {args.testing} (train on all data)")
elif args.testing == "LOSO":
    PID_name = "PID"
    splits = df[PID_name].unique()
    print(f"\nTesting strategy: {args.testing}, Number of splits: {len(splits)}")
else:  # LOFO
    PID_name = "FOLD"
    splits = df[PID_name].unique()
    print(f"\nTesting strategy: {args.testing}, Number of splits: {len(splits)}")

# ==================== Training Function ====================
def run_model_tso_patch(model, df, batch_size, train_mode, device, optimizer, scheduler,
                        stratify=False, w_other=1.0, w_nonwear=1.0, w_tso=1.0,
                        max_seq_len=None, patch_size=1200, padding_value=0.0, verbose=False,
                        visualize_batch=False, output_folder=None,
                        smooth_preds=False, smooth_method='majority_vote', smooth_window=5):
    """
    Run model on raw patched data for one epoch

    Args:
        visualize_batch: bool - whether to visualize batches
        output_folder: str - folder to save visualization
        smooth_preds: bool - apply smoothing to predictions in visualization
        smooth_method: str - smoothing method (majority_vote, median_filter, etc.)
        smooth_window: int - smoothing window size

    Returns:
        model: updated model
        metrics: dict with loss and accuracy metrics
        predictions: dict with predictions and labels
    """
    epoch_loss = 0.0
    batches = 0

    all_preds_tso = []
    all_labels_tso = []
    # 3-class only
    all_preds_other = []
    all_preds_nonwear = []
    all_labels_other = []
    all_labels_nonwear = []

    seg_column = 'segment'

    for batch in batch_generator(df=df, batch_size=batch_size, stratify=stratify,
                                 shuffle=train_mode, seg_column=seg_column):

        # Prepare batch data with raw sensor patches
        # Output: [batch_size, seq_len_minutes, patch_size, num_channels]
        # num_channels from best_params (5 or 6 depending on use_sincos)
        t1 = time.time()
        pad_X, pad_Y, x_lens = add_padding_tso_patch(
            batch,
            device=device,
            seg_column=seg_column,
            max_seq_len=max_seq_len,
            patch_size=patch_size,
            padding_value=padding_value,
            use_sincos=use_sincos,  # From best_params
            scaler=None  # Data already scaled in load_data_tso_patch() - don't scale again!
        )
        print("Time cost for loading:", time.time()-t1)

        # Forward pass
        outputs = model(pad_X, x_lens)  # [batch_size, seq_len, 3]

        # Visualize first batch if requested
        if visualize_batch and output_folder is not None:
            visualize_batch_predictions(pad_X, pad_Y, outputs, x_lens, batch,
                                       output_folder + "/debug_predictions", seg_column,
                                       smooth_preds=smooth_preds,
                                       smooth_method=smooth_method,
                                       smooth_window=smooth_window)
            # visualize_batch = False  # Only visualize once

        # Calculate loss (pad_Y is minute-level labels [batch_size, seq_len])
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

        # Get predictions per minute
        num_out_channels = outputs.shape[-1]
        preds = torch.sigmoid(outputs).cpu().detach().numpy()  # [batch_size, seq_len, C]
        labels = pad_Y.cpu().numpy()  # [batch_size, seq_len]

        # Collect valid predictions (ignore padded minutes where label == -100)
        for i in range(len(x_lens)):
            valid_len = int(x_lens[i])
            valid_preds = preds[i, :valid_len, :]
            valid_labels = labels[i, :valid_len]
            valid_mask = valid_labels >= 0  # exclude padding (-100)

            if num_out_channels == 1:
                all_preds_tso.extend(valid_preds[valid_mask, 0].tolist())
                all_labels_tso.extend((valid_labels[valid_mask] == 2).astype(int).tolist())
            else:
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
    f1_tso = f1_score(all_labels_tso, (all_preds_tso > 0.5).astype(int), zero_division=0)

    if num_out_channels == 1:
        f1_avg = f1_tso
        accuracy = np.mean((all_preds_tso > 0.5).astype(int) == all_labels_tso.astype(int))
        metrics = {
            'loss': avg_loss,
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

# ==================== Training Loop ====================
for split_id in splits:
    print(f"\n{'='*80}")
    print(f"Processing split: {split_id}")
    print(f"{'='*80}")

    # Split data - use boolean masks to avoid creating DataFrame copies
    # This saves massive amounts of memory (50GB df -> just boolean arrays ~few MB)
    if args.testing == "production":
        # Production mode: use all data
        test_mask = np.ones(len(df), dtype=bool)
        train_mask_full = np.ones(len(df), dtype=bool)
    else:
        test_mask = (df[PID_name] == split_id).values
        train_mask_full = (df[PID_name] != split_id).values

    # Further split train into train/val (80/20) - use segment-level splitting
    # Set random seed for reproducible train/val split
    random_seed = best_params.get('random_seed', 42)
    np.random.seed(random_seed)

    # Get segments from train mask
    train_segments = df.loc[train_mask_full, 'segment'].unique()
    np.random.shuffle(train_segments)
    val_size = int(len(train_segments) * 0.2)
    val_segments = set(train_segments[:val_size])

    # Create boolean masks for val and train (memory efficient - just bool arrays)
    val_mask = train_mask_full & df['segment'].isin(val_segments).values
    train_mask = train_mask_full & ~df['segment'].isin(val_segments).values

    # Create DataFrame views (NOT copies) - these reference the original df
    # No memory duplication here
    df_test = df[test_mask].copy()  # Need copy to add split_type
    df_train = df[train_mask].copy()
    df_val = df[val_mask].copy()

    # Add split_type column to track train/val split for production mode visualization
    df_train['split_type'] = 'train'
    df_val['split_type'] = 'val'
    if args.testing == "production":
        # For production mode, mark test data with split_type based on segment membership
        df_test['split_type'] = df_test['segment'].apply(
            lambda x: 'val' if x in val_segments else 'train'
        )

    print(f"Train segments: {df_train['segment'].nunique()}, "
          f"Val segments: {df_val['segment'].nunique()}, "
          f"Test segments: {df_test['segment'].nunique()}")

    # Delete masks to free memory (though they're small)
    del test_mask, train_mask_full, val_mask, train_mask, train_segments

    # Training iterations
    for iteration in range(args.training_iterations):
        print(f"\nTraining iteration {iteration+1}/{args.training_iterations}")

        # Setup model
        model = setup_model(args.model, None, max_seq_len, best_params,
                          pretraining=False, num_classes=best_params.get('output_channels', 3))
        model = model.to(device)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}, Trainable: {trainable_params:,}")

        # Check if test-only mode
        if args.test_only:
            print("Test-only mode: Loading existing model for evaluation...")

            # Get model path from best_params
            checkpoint_path = best_params.get('model_path')

            if checkpoint_path is None:
                print("ERROR: test_only mode requires 'model_path' in best_params!")
                print("Please set best_params['model_path'] = '/path/to/model.pt'")
                print("Skipping this split.")
                continue

            if not os.path.exists(checkpoint_path):
                print(f"ERROR: Model checkpoint not found: {checkpoint_path}")
                print("Please verify the path in best_params['model_path']")
                print("Skipping this split.")
                continue

            print(f"Loading model from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            print(f"Model loaded successfully!")

            # Load history from checkpoint if available, otherwise use empty dict
            _default_history = {
                'train_loss': [], 'val_loss': [],
                'train_accuracy': [], 'val_accuracy': [],
                'train_f1_avg': [], 'val_f1_avg': [],
                'train_f1_tso': [], 'val_f1_tso': [],
            }
            if best_params.get('output_channels', 3) == 3:
                _default_history.update({'train_f1_other': [], 'train_f1_nonwear': [],
                                         'val_f1_other': [], 'val_f1_nonwear': []})
            history = checkpoint.get('history', _default_history)

            # Skip training, go directly to testing
            optimizer = None
            scheduler = None
        else:
            # Setup optimizer
            optimizer = optim.RMSprop(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=best_params['lr']
            )

            # Setup scheduler
            nb_steps = get_nb_steps(df=df_train, batch_size=best_params['batch_size'],
                                   stratify='undersample_TSO', shuffle=True)
            print(f"Number of steps per epoch: {nb_steps}")
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=nb_steps, T_mult=1)

            # Training history
            history = {
                'train_loss': [], 'val_loss': [],
                'train_accuracy': [], 'val_accuracy': [],
                'train_f1_avg': [], 'val_f1_avg': [],
                'train_f1_tso': [], 'val_f1_tso': [],
            }
            if best_params.get('output_channels', 3) == 3:
                history.update({'train_f1_other': [], 'train_f1_nonwear': [],
                                 'val_f1_other': [], 'val_f1_nonwear': []})

            # Early stopping
            early_stopping = EarlyStopping(patience=10, verbose=True)
            torch.cuda.empty_cache()

        # Training loop (skip if test_only mode)
        if not args.test_only:
            for epoch in range(args.epochs):
                print(f"\nEpoch {epoch+1}/{args.epochs}")

                # Train
                model.train()
                model, train_metrics, _ = run_model_tso_patch(
                    model, df_train, best_params['batch_size'], True, device, optimizer, scheduler,
                    stratify=False,
                    w_other=best_params['w_other'],
                    w_nonwear=best_params['w_nonwear'],
                    w_tso=best_params['w_tso'],
                    max_seq_len=max_seq_len,
                    patch_size=best_params['patch_size'],
                    padding_value=best_params['padding_value'],
                    verbose=(epoch % 5 == 0),
                )

                # Record train metrics
                history['train_loss'].append(train_metrics['loss'])
                history['train_accuracy'].append(train_metrics['accuracy'])
                history['train_f1_avg'].append(train_metrics['f1_avg'])
                history['train_f1_tso'].append(train_metrics['f1_tso'])
                if best_params.get('output_channels', 3) == 3:
                    history['train_f1_other'].append(train_metrics['f1_other'])
                    history['train_f1_nonwear'].append(train_metrics['f1_nonwear'])

                print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, F1 avg: {train_metrics['f1_avg']:.4f}")

                # Validation
                if epoch % 1 == 0:  # Validate every epoch
                    model.eval()
                    with torch.no_grad():
                        _, val_metrics, _ = run_model_tso_patch(
                            model, df_val, best_params['batch_size'], False, device, optimizer, scheduler,
                            stratify=False,
                            w_other=best_params['w_other'],
                            w_nonwear=best_params['w_nonwear'],
                            w_tso=best_params['w_tso'],
                            max_seq_len=max_seq_len,
                            patch_size=best_params['patch_size'],
                            padding_value=best_params['padding_value'],
                            verbose=False,
                            visualize_batch=args.visualize,
                            output_folder=training_output_folder,
                            smooth_preds=args.smooth_predictions,
                            smooth_method=args.smooth_method,
                            smooth_window=args.smooth_window
                        )

                    history['val_loss'].append(val_metrics['loss'])
                    history['val_accuracy'].append(val_metrics['accuracy'])
                    history['val_f1_avg'].append(val_metrics['f1_avg'])
                    history['val_f1_tso'].append(val_metrics['f1_tso'])
                    if best_params.get('output_channels', 3) == 3:
                        history['val_f1_other'].append(val_metrics['f1_other'])
                        history['val_f1_nonwear'].append(val_metrics['f1_nonwear'])

                    print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1 avg: {val_metrics['f1_avg']:.4f}")

                    # Early stopping check
                    early_stopping(val_metrics['loss'], model)
                    if early_stopping.early_stop:
                        print("Early stopping triggered")
                        break

                # Save best model
                if val_metrics['loss'] == early_stopping.best_score:
                    model_path = os.path.join(train_models_folder,
                                            f"best_model_split_{split_id}_iter_{iteration}.pt")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_metrics['loss'],
                        'val_f1_avg': val_metrics['f1_avg'],
                        'history': history
                    }, model_path)
                    print(f"  -> Saved best model")

            # Load best model for evaluation (only if we trained)
            print(f"\nLoading best model for evaluation...")
            # pretrained_weight_folder = "/mnt/data/GENEActive-featurized/results/DL/UKB_v2/TSO_predict-mba4tso_patch-bs=16-dim=128-ukb/training/model_weights"
            # pretrained_weight_folder = "/mnt/data/GENEActive-featurized/results/DL/UKB_v2/TSO_predict-mba4tso_patch-bs=16-dim=128-ukb-gpu=4/training/model_weights"

            checkpoint_path = os.path.join(train_models_folder,
            # checkpoint_path = os.path.join(pretrained_weight_folder,
                                          f"best_model_split_{split_id}_iter_{iteration}.pt")
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

        # Evaluation on test set
        print(f"Evaluating on test set...")
        with torch.no_grad():
            _, test_metrics, test_predictions = run_model_tso_patch(
                model, df_test, best_params['batch_size'], False, device, optimizer, scheduler,
                stratify=False,
                w_other=best_params['w_other'],
                w_nonwear=best_params['w_nonwear'],
                w_tso=best_params['w_tso'],
                max_seq_len=max_seq_len,
                patch_size=best_params['patch_size'],
                padding_value=best_params['padding_value'],
                verbose=True,
                visualize_batch=args.visualize,
                output_folder=training_output_folder,
                smooth_preds=args.smooth_predictions,
                smooth_method=args.smooth_method,
                smooth_window=args.smooth_window
            )

        print(f"\nTest Results:")
        print(f"  Loss: {test_metrics['loss']:.4f}")
        print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  F1 avg: {test_metrics['f1_avg']:.4f}")
        print(f"  F1 TSO: {test_metrics['f1_tso']:.4f}")
        if best_params.get('output_channels', 3) == 3:
            print(f"  F1 other: {test_metrics['f1_other']:.4f}")
            print(f"  F1 nonwear: {test_metrics['f1_nonwear']:.4f}")

        # Plot training history (only if we actually trained)
        if not args.test_only and len(history.get('train_loss', [])) > 0:
            plot_tso_learning_curves(
                history=history,
                output_filepath=os.path.join(learning_plots_output_folder,
                                            f"training_history_split_{split_id}_iter_{iteration}.png")
            )
            print(f"Training history plot saved to learning_plots folder")

        # Save results
        results = {
            'split_id': split_id,
            'iteration': iteration,
            'test_metrics': test_metrics,
            'history': history
        }

        results_file = os.path.join(predictions_output_folder,
                                   f"results_split_{split_id}_iter_{iteration}.joblib")
        joblib.dump(results, results_file)
        print(f"Results saved to {results_file}")

    # Clean up split data to free memory before next split
    del df_train, df_val, df_test
    torch.cuda.empty_cache()
    print(f"Completed split {split_id} - memory cleaned")

print(f"\n{'='*80}")
print("Training complete!")
print(f"{'='*80}")

