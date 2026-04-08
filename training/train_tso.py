# -*- coding: utf-8 -*-
"""
TSO Status Prediction Script
Predicts 3-class status: 'other', 'non-wear', 'predictTSO'
Uses MBA4TSO model with undersample_TSO batch generation
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
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint

from models import setup_model
from data import (
    load_data_tso,
    batch_generator,
    add_padding_TSO,
    get_nb_steps,
)
from losses import (
    measure_loss_tso
)
from utils import (
    EarlyStopping
)

# ==================== Visualization Function ====================
def visualize_batch_data(pad_X, pad_Y, x_lens, batch_df, output_folder, seg_column='segment', predictions=None):
    """
    Visualize and save first batch data for quality checking.

    Args:
        pad_X: [batch_size, seq_len, num_features] - feature data
        pad_Y: [batch_size, seq_len] - ground truth labels (minute-level)
        x_lens: [batch_size] - original sequence lengths
        batch_df: DataFrame - batch metadata
        output_folder: str - output folder path
        seg_column: str - segment column name
        predictions: [batch_size, seq_len, 3] - model predictions (logits), optional
    """
    os.makedirs(output_folder, exist_ok=True)

    # Convert to numpy if tensors
    if torch.is_tensor(pad_X):
        pad_X = pad_X.cpu().numpy()
    if torch.is_tensor(pad_Y):
        pad_Y = pad_Y.cpu().numpy()
    if torch.is_tensor(x_lens):
        x_lens = x_lens.cpu().numpy()

    batch_size, seq_len, num_features = pad_X.shape
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Feature names (29 features for TSO task)
    # 2 base + 27 statistical (9 features × 3 axes: x, y, z)
    feature_names = (
        ['temperature', 'time_cyclic'] +
        ['x_mean', 'x_std', 'x_min', 'x_max', 'x_q25', 'x_q75', 'x_skew', 'x_kurt', 'x_cv'] +
        ['y_mean', 'y_std', 'y_min', 'y_max', 'y_q25', 'y_q75', 'y_skew', 'y_kurt', 'y_cv'] +
        ['z_mean', 'z_std', 'z_min', 'z_max', 'z_q25', 'z_q75', 'z_skew', 'z_kurt', 'z_cv']
    )

    segment_ids = batch_df[seg_column].unique()[:batch_size]

    # Summary statistics
    print(f"\n{'='*80}")
    print(f"BATCH VISUALIZATION")
    print(f"{'='*80}")
    print(f"Batch size: {batch_size}")
    print(f"Sequence length (padded): {seq_len}")
    print(f"Number of features: {num_features}")
    print(f"\nSequence lengths:")
    for i, (seg_id, seq_len_i) in enumerate(zip(segment_ids, x_lens)):
        # pad_Y[i] is now minute-level labels [seq_len]
        y_valid = pad_Y[i, :seq_len_i]
        label_counts = {
            'other': (y_valid == 0).sum().item(),
            'non-wear': (y_valid == 1).sum().item(),
            'predictTSO': (y_valid == 2).sum().item()
        }
        print(f"  {seg_id}: {seq_len_i} minutes, labels: {label_counts}")

    print(f"\nLabel distribution (all valid minutes):")
    # Only count valid (non-padded) labels
    valid_mask = pad_Y != -100
    valid_labels = pad_Y[valid_mask]
    print(f"  Other (0): {(valid_labels == 0).sum()}")
    print(f"  Non-wear (1): {(valid_labels == 1).sum()}")
    print(f"  PredictTSO (2): {(valid_labels == 2).sum()}")

    # Process predictions if provided
    pred_classes = None
    if predictions is not None:
        if torch.is_tensor(predictions):
            predictions = predictions.cpu().numpy()
        # Convert logits to class predictions
        pred_classes = np.argmax(predictions, axis=-1)  # [batch_size, seq_len]

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
        X_valid = pad_X[sample_idx, :seq_len_i, :]
        time_minutes = np.arange(seq_len_i)

        # Determine output subfolder based on split_type
        if seg_id in split_types:
            sample_output_folder = os.path.join(output_folder, split_types[seg_id])
        else:
            sample_output_folder = output_folder
        os.makedirs(sample_output_folder, exist_ok=True)

        # Get minute-level labels for this segment from pad_Y
        if torch.is_tensor(pad_Y):
            y_labels = pad_Y[sample_idx, :seq_len_i].cpu().numpy()  # [seq_len_i]
        else:
            y_labels = pad_Y[sample_idx, :seq_len_i]  # [seq_len_i]

        # Determine number of subplots (add 1 if predictions provided)
        num_plots = 7 if predictions is not None else 6
        fig, axes = plt.subplots(num_plots, 1, figsize=(14, 16 if predictions is not None else 14))
        # Count label distribution for this segment
        label_counts = {
            'other': (y_labels == 0).sum(),
            'non-wear': (y_labels == 1).sum(),
            'predictTSO': (y_labels == 2).sum()
        }
        fig.suptitle(f'Sample {sample_idx}: {seg_id} ({seq_len_i} minutes)\nLabels: {label_counts}',
                     fontsize=14, fontweight='bold')

        # Time embeddings
        axes[0].plot(time_minutes, X_valid[:, 0], label='temperature', alpha=0.7)
        axes[0].plot(time_minutes, X_valid[:, 1], label='time_cyclic', alpha=0.7)
        axes[0].set_ylabel('Value')
        axes[0].set_title('Temporal Features')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # X, Y, Z mean values
        # Features layout: [temperature, time_cyclic, x_9_features, y_9_features, z_9_features]
        for ch_idx, channel in enumerate(['x', 'y', 'z']):
            mean_idx = 2 + ch_idx * 9  # 2 base features + channel_idx * 9
            std_idx = mean_idx + 1
            axes[ch_idx + 1].plot(time_minutes, X_valid[:, mean_idx], label=f'{channel}_mean')
            axes[ch_idx + 1].fill_between(time_minutes,
                                         X_valid[:, mean_idx] - X_valid[:, std_idx],
                                         X_valid[:, mean_idx] + X_valid[:, std_idx],
                                         alpha=0.3)
            axes[ch_idx + 1].set_ylabel(f'{channel.upper()}-axis')
            axes[ch_idx + 1].set_title(f'{channel.upper()}-axis (mean ± std)')
            axes[ch_idx + 1].legend()
            axes[ch_idx + 1].grid(True, alpha=0.3)

        # CV (coefficient of variation is the 9th feature for each channel, index 8 within each group)
        axes[4].plot(time_minutes, X_valid[:, 10], label='x_cv', alpha=0.7)  # 2 + 8 = 10
        axes[4].plot(time_minutes, X_valid[:, 19], label='y_cv', alpha=0.7)  # 2 + 9 + 8 = 19
        axes[4].plot(time_minutes, X_valid[:, 28], label='z_cv', alpha=0.7)  # 2 + 18 + 8 = 28
        axes[4].set_ylabel('CV')
        axes[4].set_title('Coefficient of Variation')
        axes[4].legend()
        axes[4].grid(True, alpha=0.3)

        # Minute-level labels (ground truth)
        axes[5].plot(time_minutes, y_labels, linewidth=2, color='black')
        axes[5].set_ylabel('Label Class')
        if predictions is None:
            axes[5].set_xlabel('Time (minutes)')
        axes[5].set_title('Ground Truth Labels (0=other, 1=non-wear, 2=predictTSO)')
        axes[5].set_yticks([0, 1, 2])
        axes[5].set_yticklabels(['other', 'non-wear', 'predictTSO'])
        axes[5].grid(True, alpha=0.3)
        # Add colored background regions for different labels
        for label_val, color in [(0, 'lightblue'), (1, 'lightcoral'), (2, 'lightgreen')]:
            label_regions = (y_labels == label_val)
            if label_regions.any():
                axes[5].fill_between(time_minutes, 0, 2.5, where=label_regions, alpha=0.2, color=color)

        # Predictions subplot (if provided)
        if predictions is not None:
            y_pred = pred_classes[sample_idx, :seq_len_i]
            axes[6].plot(time_minutes, y_pred, linewidth=2, color='red', alpha=0.7, label='Predicted')
            axes[6].plot(time_minutes, y_labels, linewidth=1, color='black', linestyle='--', alpha=0.5, label='Ground Truth')
            axes[6].set_ylabel('Label Class')
            axes[6].set_xlabel('Time (minutes)')
            axes[6].set_title('Predictions vs Ground Truth')
            axes[6].set_yticks([0, 1, 2])
            axes[6].set_yticklabels(['other', 'non-wear', 'predictTSO'])
            axes[6].legend()
            axes[6].grid(True, alpha=0.3)
            # Add colored background for predictions
            for label_val, color in [(0, 'lightblue'), (1, 'lightcoral'), (2, 'lightgreen')]:
                pred_regions = (y_pred == label_val)
                if pred_regions.any():
                    axes[6].fill_between(time_minutes, 0, 2.5, where=pred_regions, alpha=0.15, color=color)
            # Calculate accuracy for this segment
            accuracy = (y_pred == y_labels).sum() / len(y_labels) * 100
            axes[6].text(0.02, 0.95, f'Accuracy: {accuracy:.1f}%',
                        transform=axes[6].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plot_file = os.path.join(sample_output_folder, f"batch_sample_{sample_idx}_{seg_id.replace('/', '_')}_{timestamp}.png")
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {plot_file}")

    # Feature statistics heatmap
    feature_stats = []
    for feat_idx in range(num_features):
        valid_values = []
        for sample_idx in range(batch_size):
            seq_len_i = int(x_lens[sample_idx])
            valid_values.extend(pad_X[sample_idx, :seq_len_i, feat_idx].tolist())
        valid_values = np.array(valid_values)
        feature_stats.append({
            'feature': feature_names[feat_idx],
            'mean': np.mean(valid_values),
            'std': np.std(valid_values),
            'min': np.min(valid_values),
            'max': np.max(valid_values)
        })

    stats_df = pd.DataFrame(feature_stats)
    # fig, ax = plt.subplots(figsize=(10, 12))
    # stats_matrix = stats_df[['mean', 'std', 'min', 'max']].T
    # sns.heatmap(stats_matrix, annot=True, fmt='.3f', cmap='RdYlBu_r',
    #             xticklabels=feature_names, yticklabels=['mean', 'std', 'min', 'max'],
    #             ax=ax, cbar_kws={'label': 'Value'})
    # ax.set_title('Feature Statistics Heatmap')
    # plt.tight_layout()
    # heatmap_file = os.path.join(output_folder, f"batch_feature_heatmap_{timestamp}.png")
    # plt.savefig(heatmap_file, dpi=150, bbox_inches='tight')
    # plt.close()
    # print(f"Saved: {heatmap_file}")

    stats_csv = os.path.join(output_folder, f"batch_feature_stats_{timestamp}.csv")
    stats_df.to_csv(stats_csv, index=False)
    print(f"Saved: {stats_csv}")
    print(f"{'='*80}\n")

# ==================== Argument Parser ====================
parser = argparse.ArgumentParser(description='TSO Status Prediction Training Pipeline')
parser.add_argument('--input_data_folder', type=str, required=True, help='Path to input data folder')
parser.add_argument('--output', type=str, required=True, help='Output folder name')
parser.add_argument('--model', type=str, default="mba4tso", help='Model name (default: mba4tso)')
parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
parser.add_argument('--num_gpu', type=str, default="0", help='GPU device ID')
parser.add_argument('--testing', type=str, default="LOFO", help='Testing strategy: LOFO or LOSO')
parser.add_argument('--training_iterations', type=int, default=1, help='Number of training iterations')
parser.add_argument('--clear_tracker', type=bool, default=False, required=False, help='Clear Job Tracking Folder')
parser.add_argument('--test_only', action='store_true', help='Skip training and only run testing with existing models')
parser.add_argument('--visualize', action='store_true', help='Visualize validation and test batches')

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
training_output_folder = os.path.join(results_folder, "training/")
job_tracking_folder = os.path.join(training_output_folder, "job_tracking_folder")
predictions_output_folder = os.path.join(training_output_folder, "predictions/")
confusion_matrix_plots_folder = os.path.join(training_output_folder, "confusion_matrix_plots/")
learning_plots_output_folder = os.path.join(training_output_folder, "learning_plots/")
train_models_folder = os.path.join(training_output_folder, "model_weights/")
train_models_folder_joblib = os.path.join(training_output_folder, "model_weights_joblib/")
checkpoint_folder = os.path.join(training_output_folder, "checkpoints/")
training_logs_folder = os.path.join(training_output_folder, "training_logs/")
processed_data_folder = os.path.join(input_data_folder.rstrip("/raw"), "processed/")

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

# ==================== Model Hyperparameters (based on param_16) ====================
best_params = {
    'batch_size': 16,
    'num_filters': 128,
    'dropout': 0.5,
    'droppath': 0.3,
    'kernel_f': 13,
    'kernel_MBA': 9,
    'num_feature_layers': 9,
    'blocks_MBA': 8,
    'optim': 'RMSprop',
    'featurelayer': 'ResNet',
    'norm1': 'BN',
    'norm2': 'GN',
    'lr': 0.001,
    'skip_connect': True,
    'skip_cross_attention': False,  # Changed to False for simpler skip connections
    'output_channels': 3,
    # TSO-specific weights
    'w_other': 1.0,
    'w_nonwear': 1.0,
    'w_tso': 1.0,
    'padding_value': 0.0,
}

print("Model hyperparameters:")
pprint(best_params)

# ==================== Load Data ====================
print(f"\nLoading data from: {args.input_data_folder}")
# Load minute-level aggregated data (24h window -> 1440 minutes with 36 features)
df = load_data_tso(args.input_data_folder, max_seq_length=86400, group=True)

print(f"Data loaded. Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Check required columns (29 features + labels + metadata)
# 2 base (temperature, time_cyclic) + 27 statistical (9 features × 3 axes: x, y, z)
feature_cols = (
    ['temperature', 'time_cyclic'] +  # 2 base features
    ['x_mean', 'x_std', 'x_min', 'x_max', 'x_q25', 'x_q75', 'x_skew', 'x_kurt', 'x_cv'] +  # 9 x-axis features
    ['y_mean', 'y_std', 'y_min', 'y_max', 'y_q25', 'y_q75', 'y_skew', 'y_kurt', 'y_cv'] +  # 9 y-axis features
    ['z_mean', 'z_std', 'z_min', 'z_max', 'z_q25', 'z_q75', 'z_skew', 'z_kurt', 'z_cv']    # 9 z-axis features
)
# required_cols = feature_cols + ['non-wear', 'predictTSO', 'segment', 'PID',]
# missing_cols = [col for col in required_cols if col not in df.columns]
# if missing_cols:
#     raise ValueError(f"Missing required columns: {missing_cols}")

# Calculate max sequence length (should be 1440 minutes = 24 hours)
# Note: minute_id ranges from 0-1439, so max is 1440 unique values
max_seq_len = df.groupby('segment')['minute_id'].max().max() + 1
print(f"Max sequence length: {max_seq_len} minutes")

# Data statistics
print(f"\nData statistics:")
print(f"  Total minute-level records: {len(df)}")
print(f"  Total unique segments: {df['segment'].nunique()}")
print(f"  Total PIDs: {df['PID'].nunique()}")
print(f"  predictTSO minutes: {df[df['predictTSO']==True].shape[0]} ({100*df[df['predictTSO']==True].shape[0]/len(df):.1f}%)")
print(f"  non-wear minutes: {df[df['non-wear']==1].shape[0]} ({100*df[df['non-wear']==1].shape[0]/len(df):.1f}%)")
other_segs = df[(df['predictTSO']==False) & (df['non-wear']!=1)]['segment'].nunique()
print(f"  other segments: {other_segs}")

# ==================== Define Splits ====================
if args.testing == "production":
    # Production mode: train on all data, no test split
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
def run_model_tso(model, df, batch_size, train_mode, device, optimizer, scheduler,
                  stratify=False, w_other=1.0, w_nonwear=1.0, w_tso=1.0,
                  max_seq_len=None, padding_value=0.0, verbose=False,
                  visualize_batch=False, output_folder=None):
    """
    Run model on data for one epoch

    Returns:
        model: updated model
        metrics: dict with loss and accuracy metrics
        predictions: dict with predictions and labels
    """
    epoch_loss = 0.0
    batches = 0

    all_preds_other = []  # Will store predicted class indices
    all_labels_other = []  # Will store true class indices
    all_segments = []  # Will store segment IDs
    all_positions = []  # Will store positions from data (1-indexed)
    all_minute_ids = []  # Will store minute_ids

    seg_column = 'segment'

    for batch in batch_generator(df=df, batch_size=batch_size, stratify=stratify,
                                 shuffle=train_mode, seg_column=seg_column):

        # Prepare batch data (30 feature channels)
        pad_X, pad_Y, x_lens = add_padding_TSO(
            batch,
            device=device,
            seg_column=seg_column,
            max_seq_len=max_seq_len,
            padding_value=padding_value
        )

        # Forward pass
        outputs = model(pad_X, x_lens)

        # Visualize batch if requested (with predictions)
        if visualize_batch and output_folder is not None:
            visualize_batch_data(pad_X, pad_Y, x_lens, batch, output_folder+"/debug_batch", seg_column, predictions=outputs)

        # Calculate loss
        # Use cross-entropy loss
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

        # Extract minute-level predictions
        # outputs: [batch, seq_len, 3], pad_Y: [batch, seq_len]
        pred_classes_seq = torch.argmax(outputs, dim=-1)  # [batch, seq_len]

        # Get segment IDs for this batch
        batch_segments = batch[seg_column].unique()

        # Only keep valid (non-padded) predictions and labels
        for i, length in enumerate(x_lens):
            # Convert length to int
            length_int = int(length)

            # Get valid portion
            pred_valid = pred_classes_seq[i, :length_int].cpu().numpy()  # [length]
            true_valid = pad_Y[i, :length_int].cpu().numpy()  # [length]

            # Get segment ID
            segment_id = batch_segments[i] if i < len(batch_segments) else f"unknown_{i}"

            # Get minute_ids and position for this segment from the batch dataframe
            seg_data = batch[batch[seg_column] == segment_id].sort_values('minute_id').iloc[:length_int]
            minute_ids = seg_data['minute_id'].values if 'minute_id' in seg_data.columns else np.arange(length_int)
            positions = seg_data['position'].values if 'position' in seg_data.columns else np.arange(1, length_int + 1)

            # Store minute-level predictions and labels with metadata
            all_preds_other.append(pred_valid)
            all_labels_other.append(true_valid)
            all_segments.extend([segment_id] * length)
            all_positions.extend(positions)
            all_minute_ids.extend(minute_ids)

        if verbose and (batches % 10 == 0):
            print(f"  Batch {batches}: Loss={total_loss.item():.4f}")

    # Calculate metrics
    avg_loss = epoch_loss / batches if batches > 0 else 0.0

    # Concatenate all predictions and labels
    all_pred_classes = np.concatenate(all_preds_other)  # [N] predicted class indices
    all_true_classes = np.concatenate(all_labels_other)  # [N] true class indices

    # Calculate per-class metrics
    # Convert to binary: 1 if class matches, 0 otherwise
    pred_classes_other = (all_pred_classes == 0).astype(int)
    pred_classes_nonwear = (all_pred_classes == 1).astype(int)
    pred_classes_tso = (all_pred_classes == 2).astype(int)

    true_classes_other = (all_true_classes == 0).astype(int)
    true_classes_nonwear = (all_true_classes == 1).astype(int)
    true_classes_tso = (all_true_classes == 2).astype(int)

    f1_other = f1_score(true_classes_other, pred_classes_other, average='binary', zero_division=0)
    f1_nonwear = f1_score(true_classes_nonwear, pred_classes_nonwear, average='binary', zero_division=0)
    f1_tso = f1_score(true_classes_tso, pred_classes_tso, average='binary', zero_division=0)

    acc_other = accuracy_score(true_classes_other, pred_classes_other)
    acc_nonwear = accuracy_score(true_classes_nonwear, pred_classes_nonwear)
    acc_tso = accuracy_score(true_classes_tso, pred_classes_tso)

    metrics = {
        'loss': avg_loss,
        'f1_other': f1_other,
        'f1_nonwear': f1_nonwear,
        'f1_tso': f1_tso,
        'acc_other': acc_other,
        'acc_nonwear': acc_nonwear,
        'acc_tso': acc_tso,
        'f1_avg': (f1_other + f1_nonwear + f1_tso) / 3,
    }

    predictions = {
        'segment': all_segments,  # Segment IDs [N]
        'position': all_positions,  # Position within segment (1-indexed) [N]
        'minute_id': all_minute_ids,  # Minute IDs [N]
        'pred_other': pred_classes_other,  # Binary predictions for "other"
        'pred_nonwear': pred_classes_nonwear,  # Binary predictions for "non-wear"
        'pred_tso': pred_classes_tso,  # Binary predictions for "predictTSO"
        'true_other': true_classes_other,  # Binary labels for "other"
        'true_nonwear': true_classes_nonwear,  # Binary labels for "non-wear"
        'true_tso': true_classes_tso,  # Binary labels for "predictTSO"
    }

    return model, metrics, predictions

# ==================== Training Loop for Each Split ====================
for split_id in splits:
    print(f"\n{'='*80}")
    print(f"Processing split: {split_id}")
    print(f"{'='*80}")

    # Check if already processed
    split_marker = os.path.join(job_tracking_folder, str(split_id))
    # if os.path.exists(split_marker):
    #     print(f"Split {split_id} already processed. Skipping.")
    #     continue

    # Mark as in progress
    with open(split_marker, 'w') as f:
        f.write("processing")

    # Split data
    if args.testing == "production":
        # Production mode: use all data for training
        df_test = df.copy()
        df_train = df.copy()
    else:
        # Cross-validation mode: split by PID/FOLD
        df_test = df[df[PID_name] == split_id].copy()
        df_train = df[df[PID_name] != split_id].copy()

    # Further split train into train/val (80/20) - randomly split by segments
    unique_segments = df_train['segment'].unique()
    np.random.shuffle(unique_segments)
    val_size = int(len(unique_segments) * 0.2)
    val_segments = unique_segments[:val_size]

    df_val = df_train[df_train['segment'].isin(val_segments)].copy()
    df_train = df_train[~df_train['segment'].isin(val_segments)].copy()

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

    # Scale features (all 28 statistical features for x, y, z and temperature)
    scaler = StandardScaler()
    c_to_scale = (
        ['temperature'] +
        ['x_mean', 'x_std', 'x_min', 'x_max', 'x_q25', 'x_q75', 'x_skew', 'x_kurt', 'x_cv'] +
        ['y_mean', 'y_std', 'y_min', 'y_max', 'y_q25', 'y_q75', 'y_skew', 'y_kurt', 'y_cv'] +
        ['z_mean', 'z_std', 'z_min', 'z_max', 'z_q25', 'z_q75', 'z_skew', 'z_kurt', 'z_cv']
    )
    df_train.loc[:, c_to_scale] = scaler.fit_transform(df_train[c_to_scale])
    df_val.loc[:, c_to_scale] = scaler.transform(df_val[c_to_scale])
    df_test.loc[:, c_to_scale] = scaler.transform(df_test[c_to_scale])

    # Training iterations
    for iteration in range(args.training_iterations):
        print(f"\nTraining iteration {iteration+1}/{args.training_iterations}")

        # Setup model
        input_dim = len(feature_cols)  # 2 base features + 27 statistical features (9 per axis × 3 axes: x, y, z)
        model = setup_model(args.model, input_dim, max_seq_len, best_params,
                          pretraining=False, num_classes=3)
        model = model.to(device)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}, Trainable: {trainable_params:,}")

        # Check if test-only mode
        if args.test_only:
            print("Test-only mode: Loading existing model for evaluation...")
            checkpoint_path = os.path.join(train_models_folder,
                                          f"best_model_split_{split_id}_iter_{iteration}.pt")
            if not os.path.exists(checkpoint_path):
                print(f"ERROR: Model checkpoint not found: {checkpoint_path}")
                print("Skipping this split.")
                continue

            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

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
                'train_f1_avg': [], 'val_f1_avg': [],
                'train_f1_other': [], 'train_f1_nonwear': [], 'train_f1_tso': [],
                'val_f1_other': [], 'val_f1_nonwear': [], 'val_f1_tso': [],
            }

            # Early stopping
            early_stopping = EarlyStopping(patience=15, verbose=True)
            torch.cuda.empty_cache()

        # Training loop (skip if test_only mode)
        if not args.test_only:
            for epoch in range(args.epochs):
                print(f"\nEpoch {epoch+1}/{args.epochs}")

                # Train
                model.train()
                model, train_metrics, _ = run_model_tso(
                    model, df_train, best_params['batch_size'], True, device, optimizer, scheduler,
                    stratify=False,
                    w_other=best_params['w_other'],
                    w_nonwear=best_params['w_nonwear'],
                    w_tso=best_params['w_tso'],
                    max_seq_len=max_seq_len,
                    padding_value=best_params['padding_value'],
                    verbose=(epoch % 5 == 0),
                )

                # Record train metrics
                history['train_loss'].append(train_metrics['loss'])
                history['train_f1_avg'].append(train_metrics['f1_avg'])
                history['train_f1_other'].append(train_metrics['f1_other'])
                history['train_f1_nonwear'].append(train_metrics['f1_nonwear'])
                history['train_f1_tso'].append(train_metrics['f1_tso'])

                # Validate
                model.eval()
                with torch.no_grad():
                    _, val_metrics, _ = run_model_tso(
                        model, df_val, best_params['batch_size'], False, device, optimizer, scheduler,
                        stratify=False,
                        w_other=best_params['w_other'],
                        w_nonwear=best_params['w_nonwear'],
                        w_tso=best_params['w_tso'],
                        max_seq_len=max_seq_len,
                        padding_value=best_params['padding_value'],
                        verbose=False,
                        visualize_batch=args.visualize,
                        output_folder=training_output_folder
                    )

                # Record val metrics
                history['val_loss'].append(val_metrics['loss'])
                history['val_f1_avg'].append(val_metrics['f1_avg'])
                history['val_f1_other'].append(val_metrics['f1_other'])
                history['val_f1_nonwear'].append(val_metrics['f1_nonwear'])
                history['val_f1_tso'].append(val_metrics['f1_tso'])

                current_time = datetime.now().strftime("%H:%M:%S")
                print(f"  [{current_time}] Train - Loss: {train_metrics['loss']:.4f}, F1_avg: {train_metrics['f1_avg']:.4f} "
                      f"(other: {train_metrics['f1_other']:.3f}, nonwear: {train_metrics['f1_nonwear']:.3f}, tso: {train_metrics['f1_tso']:.3f})")
                print(f"  [{current_time}] Val   - Loss: {val_metrics['loss']:.4f}, F1_avg: {val_metrics['f1_avg']:.4f} "
                      f"(other: {val_metrics['f1_other']:.3f}, nonwear: {val_metrics['f1_nonwear']:.3f}, tso: {val_metrics['f1_tso']:.3f})")

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
            checkpoint_path = os.path.join(train_models_folder,
                                          f"best_model_split_{split_id}_iter_{iteration}.pt")
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Evaluation on test set
        print(f"Evaluating on test set...")
        with torch.no_grad():
            _, test_metrics, test_predictions = run_model_tso(
                model, df_test, best_params['batch_size'], False, device, optimizer, scheduler,
                stratify=False,
                w_other=best_params['w_other'],
                w_nonwear=best_params['w_nonwear'],
                w_tso=best_params['w_tso'],
                max_seq_len=max_seq_len,
                padding_value=best_params['padding_value'],
                verbose=False,
                visualize_batch=args.visualize,
                output_folder=training_output_folder
            )

        # Print test results
        print(f"\nTest Results for Split {split_id}, Iteration {iteration+1}:")
        print("="*60)
        print(f"  Loss: {test_metrics['loss']:.4f}")
        print(f"  F1_avg: {test_metrics['f1_avg']:.4f}")
        print(f"  Other    - F1: {test_metrics['f1_other']:.4f}, Acc: {test_metrics['acc_other']:.4f}")
        print(f"  Non-wear - F1: {test_metrics['f1_nonwear']:.4f}, Acc: {test_metrics['acc_nonwear']:.4f}")
        print(f"  PredTSO  - F1: {test_metrics['f1_tso']:.4f}, Acc: {test_metrics['acc_tso']:.4f}")

        # Save predictions
        pred_df = pd.DataFrame({
            'segment': test_predictions['segment'],
            'position': test_predictions['position'],
            'minute_id': test_predictions['minute_id'],
            'pred_other': test_predictions['pred_other'],
            'pred_nonwear': test_predictions['pred_nonwear'],
            'pred_tso': test_predictions['pred_tso'],
            'true_other': test_predictions['true_other'],
            'true_nonwear': test_predictions['true_nonwear'],
            'true_tso': test_predictions['true_tso'],
        })
        pred_df.to_csv(os.path.join(predictions_output_folder,
                                    f"predictions_split_{split_id}_iter_{iteration}.csv"),
                      index=False)

        # Save test metrics
        with open(os.path.join(predictions_output_folder,
                              f"test_metrics_split_{split_id}_iter_{iteration}.txt"), 'w') as f:
            f.write(f"Test Metrics:\n")
            for key, value in test_metrics.items():
                f.write(f"{key}: {value:.4f}\n")

        # Plot training history (only if we actually trained)
        if not args.test_only:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            # Total loss
            axes[0, 0].plot(history['train_loss'], label='Train', linewidth=2)
            axes[0, 0].plot(history['val_loss'], label='Val', linewidth=2)
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Total Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            # Average F1
            axes[0, 1].plot(history['train_f1_avg'], label='Train', linewidth=2)
            axes[0, 1].plot(history['val_f1_avg'], label='Val', linewidth=2)
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('F1 Score')
            axes[0, 1].set_title('Average F1 Score')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            # F1 per class - Train
            axes[1, 0].plot(history['train_f1_other'], label='Other', linewidth=2)
            axes[1, 0].plot(history['train_f1_nonwear'], label='Non-wear', linewidth=2)
            axes[1, 0].plot(history['train_f1_tso'], label='PredictTSO', linewidth=2)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('F1 Score')
            axes[1, 0].set_title('Train F1 per Class')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            # F1 per class - Val
            axes[1, 1].plot(history['val_f1_other'], label='Other', linewidth=2)
            axes[1, 1].plot(history['val_f1_nonwear'], label='Non-wear', linewidth=2)
            axes[1, 1].plot(history['val_f1_tso'], label='PredictTSO', linewidth=2)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('F1 Score')
            axes[1, 1].set_title('Val F1 per Class')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(learning_plots_output_folder,
                                    f"training_history_split_{split_id}_iter_{iteration}.png"),
                       dpi=150, bbox_inches='tight')
            plt.close()

    print(f"\nCompleted split {split_id}")
    print("="*80)

print(f"\nTraining completed at: {datetime.now()}")
print(f"Results saved to: {results_folder}")


