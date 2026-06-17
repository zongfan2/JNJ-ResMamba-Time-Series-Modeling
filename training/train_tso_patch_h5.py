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
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint
import time
import h5py
import yaml

# Make the repo root importable so `from models/data/losses ...` works regardless
# of the cwd or whether `pip install -e .` succeeded (the shell_script above can
# `cd` to the wrong place on nested Domino checkouts).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import setup_model
from data import (
    add_padding_tso_patch_h5,
)
from losses import (
    measure_loss_tso,
    measure_loss_tso_with_continuity,
    # Structural priors and noisy-label regularizers (Deep TSO).
    measure_loss_tso_structural,
    SupConLossV2,
    ELRMemory,
    CircadianPriorBias,
    hour_from_time_channels,
    compute_boundary_weights,
    consensus_from_annotators,
)
from evaluation.postprocessing import batch_enforce_single_tso
# Pre-existing utils imports were stale: EarlyStopping lives in losses/standard.py
# and the rest live under evaluation/. Pull each from its real module so the
# script can actually import.
from losses.standard import EarlyStopping
from evaluation.metrics import calculate_metrics_nn, plot_tso_learning_curves
from evaluation.postprocessing import smooth_predictions, smooth_predictions_combined
from evaluation.tso_validation import extract_tso_interval, cross_night_consistency, interval_agreement


# ==================== H5 Dataset Class ====================
def _decode_h5_string(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _subject_from_segment_name(segment_name):
    text = _decode_h5_string(segment_name)
    return text.split("_")[0] if "_" in text else text


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
        self.subject_ids = self.h5f["subject_ids"][:] if "subject_ids" in self.h5f else None
        self.Y_annotators = self.h5f["Y_annotators"] if "Y_annotators" in self.h5f else None
        self.Y_gt = self.h5f["Y_gt"] if "Y_gt" in self.h5f else None

        subjects = []
        for idx in self.indices:
            if self.subject_ids is not None:
                subjects.append(_decode_h5_string(self.subject_ids[idx]))
            else:
                subjects.append(_subject_from_segment_name(self.segment_names[idx]))
        self.subject_to_index = {subject: i for i, subject in enumerate(sorted(set(subjects)))}
        # Per-position subject string (aligned with __getitem__ ordering) so the
        # subject-grouped batch generator can group nights without loading X.
        self.subjects = subjects

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """Get a single segment.

        ``segment_id`` is the H5-file index (constant across train/val splits)
        and is used as the key for ELRMemory and any other per-segment auxiliary
        state. ``segment`` is the (potentially non-integer) human-readable name.
        """
        actual_idx = self.indices[idx]
        segment_name = self.segment_names[actual_idx]
        subject = (
            _decode_h5_string(self.subject_ids[actual_idx])
            if self.subject_ids is not None
            else _subject_from_segment_name(segment_name)
        )
        sample = {
            "X": self.X[actual_idx],
            "Y": self.Y[actual_idx],
            "seq_length": self.seq_lengths[actual_idx],
            "segment": segment_name,
            "segment_id": int(actual_idx),
            "subject": subject,
            "subject_index": self.subject_to_index[subject],
        }
        if self.Y_annotators is not None:
            sample["Y_annotators"] = self.Y_annotators[actual_idx]
        if self.Y_gt is not None:
            sample["Y_gt"] = self.Y_gt[actual_idx]
        return sample

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


def subject_grouped_batch_generator(dataset, batch_size, nights_per_group=4):
    """Yield batches that keep same-subject nights together so the cross-night
    SupCon objective has positive pairs. Each night is emitted once per epoch.

    Plain random batching almost never co-locates multiple nights of one
    subject, so SupCon would otherwise see zero positives and contribute nothing.
    """
    by_subject = defaultdict(list)
    for pos in range(len(dataset)):
        by_subject[dataset.subjects[pos]].append(pos)

    groups = []
    for positions in by_subject.values():
        positions = list(positions)
        np.random.shuffle(positions)
        for start in range(0, len(positions), nights_per_group):
            groups.append(positions[start:start + nights_per_group])
    np.random.shuffle(groups)

    batch = []
    for group in groups:
        if batch and len(batch) + len(group) > batch_size:
            yield np.array(batch, dtype=np.int64)
            batch = []
        batch.extend(group)
    if batch:
        yield np.array(batch, dtype=np.int64)


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
def threshold_diagnostics(probs, labels):
    """Separate ranking ability from threshold calibration for binary TSO.

    A model can score F1=0 at the hard 0.5 threshold yet still rank TSO
    minutes correctly: its sigmoid probabilities are simply compressed below
    0.5 (common under class imbalance / smoothness priors). These read-only
    diagnostics distinguish the two failure modes from the SAME continuous
    probabilities the F1 is computed from:

      - auc           : ranking quality, threshold-independent. ~0.5 => the
                        model learned nothing; >>0.5 => it ranks but the
                        threshold/scale is wrong (cheap fix, no retrain).
      - best_f1       : the F1 achievable at the best single threshold.
      - best_threshold: where that maximum occurs (==0.5 would mean 0.5 is
                        already fine; <<0.5 confirms probability compression).
      - pos_rate      : positive (TSO) fraction of valid minutes.
      - pred_mean_prob: mean predicted TSO probability (collapse => near 0).

    Args:
        probs: 1-D array of continuous TSO probabilities (sigmoid outputs).
        labels: 1-D array of 0/1 TSO labels, same length as probs.

    Returns:
        dict with the five keys above; AUC/best-threshold are NaN when only
        one class is present (both undefined with a single label value).
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels).astype(int)
    out = {
        "pos_rate": float(labels.mean()) if labels.size else float("nan"),
        "pred_mean_prob": float(probs.mean()) if probs.size else float("nan"),
        "auc": float("nan"),
        "best_f1": float("nan"),
        "best_threshold": float("nan"),
    }
    if labels.size == 0 or labels.min() == labels.max():
        return out  # AUC and a "best" threshold are undefined with one class
    try:
        out["auc"] = float(roc_auc_score(labels, probs))
    except ValueError:
        pass
    # Sweep candidate thresholds at the prob quantiles (bounded, data-adaptive).
    candidates = np.unique(np.quantile(probs, np.linspace(0.02, 0.98, 49)))
    best_f1, best_t = 0.0, 0.5
    for t in candidates:
        f1 = f1_score(labels, (probs > t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    out["best_f1"] = float(best_f1)
    out["best_threshold"] = best_t
    return out


def build_cv_runs(input_h5, testing, num_folds, val_size, single_fold, seed=42, split_file=None):
    """Build a list of (tag, train_idx, val_idx, test_idx) cross-validation runs.

    Replaces the leaky random-by-night split (which let a subject's nights land
    in both train and test) with subject-independent CV mirroring Deep Scratch's
    --testing LOFO/LOSO:

      - "LOFO": leave one FOLD out as the test set. Uses the H5 ``fold_ids`` if
        present; otherwise derives ``num_folds`` folds deterministically from
        subject ids (sorted unique subjects round-robin into FOLD1..FOLDk) so no
        subject spans train and test.
      - "LOSO": leave one SUBJECT out (one run per subject).
      - "production": train and test on ALL segments (no held-out set).
      - "random": legacy — use split_file if given else a random val carve; a
        single run, test == val, NOT subject-independent.

    For LOFO/LOSO the held-out fold/subject IS the test set; the validation set
    (model selection / early stopping only) is a random ``val_size`` carve from
    the remaining segments. The reported test metric is therefore on a
    subject-independent held-out group.
    """
    with h5py.File(input_h5, 'r') as h5f:
        num_segments = int(h5f.attrs['num_segments'])
        subjects = (h5f['subject_ids'][:].astype(str)
                    if 'subject_ids' in h5f else np.array([''] * num_segments))
        folds = (h5f['fold_ids'][:].astype(str)
                 if 'fold_ids' in h5f else np.array([''] * num_segments))
    all_idx = np.arange(num_segments)
    rng = np.random.RandomState(seed)

    def _carve_val(pool):
        shuf = np.array(sorted(pool)); rng.shuffle(shuf)
        n_val = max(1, int(len(shuf) * val_size))
        return np.sort(shuf[n_val:]), np.sort(shuf[:n_val])  # train, val

    if testing == 'production':
        return [('All', all_idx, all_idx, all_idx)]

    if testing == 'random':
        if split_file and os.path.exists(split_file):
            sd = np.load(split_file)
            tr, va = np.asarray(sd['train']), np.asarray(sd['val'])
        else:
            shuf = all_idx.copy(); rng.shuffle(shuf)
            n_val = int(num_segments * val_size)
            tr, va = np.sort(shuf[n_val:]), np.sort(shuf[:n_val])
        return [('random', tr, va, va)]  # test == val (legacy, leaky)

    # ---- subject-independent CV ----
    if testing == 'LOSO':
        if (subjects == '').all():
            raise ValueError("LOSO requires subject_ids in the H5.")
        key = subjects
    else:  # LOFO
        if (folds != '').any():
            key = folds  # use the parquet's FOLD labels verbatim
        else:
            uniq = sorted(s for s in set(subjects.tolist()) if s != '')
            if not uniq:
                raise ValueError("LOFO needs fold_ids or subject_ids in the H5.")
            subj_fold = {s: f"FOLD{(i % num_folds) + 1}" for i, s in enumerate(uniq)}
            key = np.array([subj_fold.get(s, 'FOLD1') for s in subjects])
            print(f"[LOFO] no FOLD labels in H5; derived {num_folds} folds from "
                  f"{len(uniq)} subjects (round-robin).")

    groups = sorted(set(key.tolist()))
    if single_fold:
        if single_fold not in groups:
            raise ValueError(f"--single_fold {single_fold!r} not in {groups}")
        groups = [single_fold]

    runs = []
    for g in groups:
        test_idx = np.sort(all_idx[key == g])
        remaining = all_idx[key != g]
        if len(remaining) == 0:
            raise ValueError(f"Held-out group {g!r} contains all segments; cannot train.")
        train_idx, val_idx = _carve_val(remaining)
        runs.append((str(g), train_idx, val_idx, test_idx))
    return runs


def run_model_tso_h5(model, dataset, batch_size, train_mode, device, optimizer, scheduler,
                    w_other=1.0, w_nonwear=1.0, w_tso=1.0,
                    max_seq_len=1440, patch_size=1200, padding_value=0.0, verbose=False,
                    visualize_batch=False, output_folder=None,
                    smooth_preds=False, smooth_method='majority_vote', smooth_window=5,
                    continuity_weight=0.0, enforce_single_tso=False,
                    min_gap_minutes=30, min_duration_minutes=10,
                    # ----- Structural priors and noisy-label regularizers -----
                    elr_memory=None, w_elr=0.0,
                    w_trans=0.0, trans_budget=2.0,
                    w_dur=0.0, dur_min=3.0, dur_max=11.0,
                    boundary_tau_steps=0.0,
                    circadian_bias=None,
                    base_loss="ce", gce_q=0.7,
                    w_supcon=0.0, supcon_temperature=0.07,
                    use_consensus_weight=False,
                    patch_duration_hours=None):
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
    # Per-sample label-free TSO interval records (onset/offset/duration/segments).
    interval_records = []
    gt_records = []        # per-night {"model": interval_agreement, "vanhees": interval_agreement}
    gt_pm, model_pm, vanhees_pm = [], [], []   # aligned per-minute GT / model / van Hees (0/1)
    nonfinite_batches = 0   # batches whose loss was NaN/Inf (model not learning if high)
    nan_grad_batches = 0    # batches with finite loss but NaN/Inf gradient (step skipped)

    if w_supcon > 0 and train_mode:
        batch_iter = subject_grouped_batch_generator(dataset, batch_size)
    else:
        batch_iter = batch_generator_h5(dataset, batch_size=batch_size, shuffle=train_mode)
    for batch_indices in batch_iter:
        t1 = time.time()

        # Prepare batch
        pad_X, pad_Y, x_lens, batch_samples, pad_Y_annotators, pad_Y_gt = add_padding_tso_patch_h5(
            dataset, batch_indices, device,
            max_seq_len=max_seq_len,
            patch_size=patch_size,
            padding_value=padding_value,
            num_channels=dataset.num_channels
        )
        pad_Y_raw = pad_Y.clone()  # van Hees predictTSO, before any Phase-2 consensus relabel
        load_time = time.time() - t1

        # Raw accelerometry often has NaN/Inf (sensor gaps, missing temperature).
        # A single non-finite value -> NaN logits -> NaN loss -> NaN weights after
        # one optimizer step, which freezes the whole run. Sanitize and warn once.
        if not torch.isfinite(pad_X).all():
            if not getattr(run_model_tso_h5, "_warned_nan_input", False):
                n_bad = int((~torch.isfinite(pad_X)).sum().item())
                print(f"  [warn] non-finite values in input batch ({n_bad}); replacing "
                      f"with {padding_value}. Clean the H5 (rebuild) or apply a scaler.")
                run_model_tso_h5._warned_nan_input = True
            pad_X = torch.nan_to_num(pad_X, nan=float(padding_value), posinf=0.0, neginf=0.0)

        # if batches == 0:
        #     print(f"First batch loading time: {load_time:.4f}s")

        # Forward pass
        if w_supcon > 0 and train_mode:
            outputs, embedding = model(pad_X, x_lens, return_embedding=True)
        else:
            outputs = model(pad_X, x_lens)
            embedding = None

        # Visualize if requested
        if visualize_batch and output_folder is not None and batches == 0:
            visualize_batch_predictions_h5(pad_X, pad_Y, outputs, x_lens, batch_samples,
                                          output_folder + "/debug_predictions",
                                          smooth_preds=smooth_preds,
                                          smooth_method=smooth_method,
                                          smooth_window=smooth_window)

        # --- Circadian logit bias (output-side, learnable gamma) -----------
        # Pulls in the population time-of-day prior. Channels are assumed to be
        # [x, y, z, temperature, time_sin, time_cos] for 6-channel data. For
        # 5-channel data the caller is responsible for passing circadian_bias=None.
        if circadian_bias is not None:
            # pad_X: [B, T_min, patch_size, num_channels]. Average sin/cos across
            # the patch to get per-minute time channels, then recover hour-of-day.
            ts_per_min = pad_X[..., 4].mean(dim=2)   # [B, T_min]
            tc_per_min = pad_X[..., 5].mean(dim=2)   # [B, T_min]
            hours_per_min = hour_from_time_channels(ts_per_min, tc_per_min)
            outputs = circadian_bias(outputs, hours_per_min)

        # --- Boundary reweighting precompute (label-based, Approach 1) -----
        boundary_weight = None
        if boundary_tau_steps > 0:
            labels_np = pad_Y.cpu().numpy()
            bw = np.stack([
                compute_boundary_weights(labels_np[i], tau_steps=boundary_tau_steps)
                for i in range(labels_np.shape[0])
            ])
            boundary_weight = torch.from_numpy(bw).to(device)

        supervision_weight = None
        if use_consensus_weight:
            if pad_Y_annotators is None:
                raise ValueError("--use_consensus_weight requires Y_annotators in the H5 file")
            # Capture masks BEFORE relabeling — consensus_labels has no -100.
            padding_mask = pad_Y == -100
            nonwear_mask = pad_Y == 1
            consensus_labels, supervision_weight = consensus_from_annotators(
                pad_Y_annotators, positive_class=2
            )
            keep_original = padding_mask | nonwear_mask
            pad_Y = torch.where(keep_original, pad_Y, consensus_labels)
            # Zero supervision on padded minutes using the ORIGINAL mask so GCE
            # never trains on padding (pad_Y no longer contains -100 after the
            # torch.where above).
            supervision_weight = supervision_weight.masked_fill(padding_mask, 0.0)

        # --- ELR target lookup (per-segment EMA of past predictions) -------
        elr_target = None
        if w_elr > 0 and elr_memory is not None:
            segment_ids = np.array(
                [s.get('segment_id', i) for i, s in enumerate(batch_samples)],
                dtype=np.int64,
            )
            elr_target = elr_memory.get(segment_ids).to(device)

        # --- Calculate loss -----------------------------------------------
        use_structural = (
            base_loss != "ce"
            or supervision_weight is not None
            or w_trans > 0 or w_dur > 0 or w_elr > 0
            or boundary_weight is not None
        )
        if use_structural:
            loss_dict = measure_loss_tso_structural(
                outputs, pad_Y, x_lens,
                patch_duration_hours=patch_duration_hours,
                boundary_weight=boundary_weight,
                supervision_weight=supervision_weight,
                base_loss=base_loss, gce_q=gce_q,
                w_trans=w_trans, w_dur=w_dur, w_elr=w_elr,
                trans_budget=trans_budget, dur_min=dur_min, dur_max=dur_max,
                elr_target=elr_target,
            )
            total_loss = loss_dict['total']
            # Map structural-loss components onto the existing class/cont
            # tracking fields so downstream printing keeps working unchanged.
            class_loss = loss_dict['ce']
            cont_loss = loss_dict['trans'] + loss_dict['dur'] + loss_dict['elr']
            epoch_class_loss += class_loss.item()
            epoch_cont_loss += cont_loss.item()
        elif continuity_weight > 0:
            total_loss, class_loss, cont_loss = measure_loss_tso_with_continuity(
                outputs, pad_Y, x_lens, continuity_weight=continuity_weight
            )
            epoch_class_loss += class_loss.item()
            epoch_cont_loss += cont_loss.item()
        else:
            total_loss = measure_loss_tso(outputs, pad_Y, x_lens)

        if w_supcon > 0 and train_mode and embedding is not None:
            subject_indices = torch.tensor(
                [s["subject_index"] for s in batch_samples],
                dtype=torch.long,
                device=device,
            )
            # SupCon needs at least one subject with >= 2 nights in the batch;
            # otherwise it has no positive pairs and is undefined/unstable.
            _, subject_counts = torch.unique(subject_indices, return_counts=True)
            if (subject_counts >= 2).any():
                supcon_loss = SupConLossV2(temperature=supcon_temperature)(embedding, subject_indices)
                total_loss = total_loss + w_supcon * supcon_loss

        # Backward pass if training
        if train_mode:
            optimizer.zero_grad()
            if not torch.isfinite(total_loss):
                # A non-finite loss would write NaN/Inf into the weights on
                # optimizer.step() and freeze the entire run. Skip this batch
                # (weights unchanged) instead of poisoning the model.
                if not getattr(run_model_tso_h5, "_warned_nan_loss", False):
                    input_finite = bool(torch.isfinite(pad_X).all())
                    logits_finite = bool(torch.isfinite(outputs).all())
                    print(f"  [warn] non-finite loss; skipping optimizer step. "
                          f"input_finite={input_finite} logits_finite={logits_finite} "
                          f"logit_min={float(outputs.min())} logit_max={float(outputs.max())}. "
                          f"logits_finite=False with input_finite=True => the MODEL is "
                          f"emitting NaN (almost always unscaled input — rebuild the H5 "
                          f"with a scaler). A per-epoch skipped-batch count is printed below.")
                    run_model_tso_h5._warned_nan_loss = True
            else:
                total_loss.backward()
                # clip_grad_norm_ returns the PRE-clip total norm. A finite loss
                # can still yield a NaN/Inf gradient (e.g. backward through a
                # fully-masked / sub-minute segment, or an unstable attention).
                # Stepping then scales EVERY weight by a NaN clip coefficient and
                # poisons the whole model (that is what froze the run: batch 1
                # stepped on a NaN grad, then all later batches were NaN). So skip
                # the step whenever the gradient is non-finite.
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0, error_if_nonfinite=False)
                if torch.isfinite(grad_norm):
                    optimizer.step()
                    scheduler.step()

                    # --- ELR memory update ----------------------------------
                    # Refresh per-segment EMA of predictions even during ELR
                    # warmup (w_elr == 0 but memory provided) so the target is
                    # meaningful as soon as the loss term kicks in.
                    if elr_memory is not None:
                        with torch.no_grad():
                            if outputs.shape[-1] == 1:
                                probs_for_mem = torch.sigmoid(outputs[..., 0]).cpu().numpy()
                            else:
                                probs_for_mem = torch.softmax(outputs, dim=-1).cpu().numpy()
                        segment_ids = np.array(
                            [s.get('segment_id', i) for i, s in enumerate(batch_samples)],
                            dtype=np.int64,
                        )
                        elr_memory.update(segment_ids, probs_for_mem)
                else:
                    # Finite loss but NaN/Inf gradient: stepping would scale every
                    # weight by a NaN clip coefficient and poison the model. Skip.
                    nan_grad_batches += 1
                    if not getattr(run_model_tso_h5, "_warned_nan_grad", False):
                        print("  [warn] finite loss but non-finite GRADIENT; skipping step "
                              "to avoid poisoning all weights. Usually backward through a "
                              "fully-masked/sub-minute segment — run the seq_length<1200 check.")
                        run_model_tso_h5._warned_nan_grad = True

        # Record loss (skip non-finite so the reported average stays meaningful)
        loss_val = total_loss.item()
        if np.isfinite(loss_val):
            epoch_loss += loss_val
        else:
            nonfinite_batches += 1
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

            # Label-free TSO interval extraction (per night, predicted classes only).
            if num_out_channels == 1:
                pred_seq = (valid_preds[:, 0] > 0.5).astype(int)
            else:
                pred_seq = np.argmax(valid_preds, axis=1)
            interval = extract_tso_interval(pred_seq, timestep_minutes=1.0)
            interval["segment"] = _decode_h5_string(batch_samples[i]["segment"])
            interval["subject"] = batch_samples[i]["subject"]
            interval_records.append(interval)

            if pad_Y_gt is not None:
                gt_seq = pad_Y_gt[i, :valid_len].cpu().numpy().astype(int)
                vanhees_seq = (pad_Y_raw[i, :valid_len] == 2).cpu().numpy().astype(int)
                tso_class = 2 if num_out_channels > 1 else 1
                model_bin = (pred_seq == tso_class).astype(int)
                gt_records.append({
                    "model": interval_agreement(pred_seq, gt_seq, timestep_minutes=1.0),
                    "vanhees": interval_agreement(vanhees_seq, gt_seq, timestep_minutes=1.0),
                })
                gt_pm.extend(gt_seq.tolist())
                model_pm.extend(model_bin.tolist())
                vanhees_pm.extend(vanhees_seq.tolist())

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
    # Average over batches that produced a FINITE loss (so a run where most
    # batches NaN-out doesn't masquerade as a tiny loss). Surface the skip count.
    finite_batches = max(batches - nonfinite_batches, 1)
    avg_loss = epoch_loss / finite_batches
    avg_class_loss = epoch_class_loss / finite_batches
    avg_cont_loss = epoch_cont_loss / finite_batches
    if nonfinite_batches > 0:
        print(f"  [warn] {nonfinite_batches}/{batches} batches had non-finite loss "
              f"and were skipped — the model is NOT training on those. If this is most "
              f"of them, fix the NaN source (scale the input) before reading metrics.")
    if nan_grad_batches > 0:
        print(f"  [warn] {nan_grad_batches}/{batches} batches had a non-finite GRADIENT "
              f"(finite loss) and their step was skipped. If most batches, the model "
              f"isn't learning — suspect sub-minute segments or the cross-attention path "
              f"(try skip_cross_attention=False in the config).")

    f1_tso = f1_score(all_labels_tso, (all_preds_tso > 0.5).astype(int), zero_division=0)

    if num_out_channels == 1:
        # Binary: only TSO metric
        f1_avg = f1_tso
        accuracy = np.mean((all_preds_tso > 0.5).astype(int) == all_labels_tso.astype(int))
        diag = threshold_diagnostics(all_preds_tso, all_labels_tso)
        metrics = {
            'loss': avg_loss,
            'class_loss': avg_class_loss,
            'cont_loss': avg_cont_loss,
            'accuracy': accuracy,
            'f1_avg': f1_avg,
            'f1_tso': f1_tso,
            **diag,
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

        diag = threshold_diagnostics(all_preds_tso, all_labels_tso)
        metrics = {
            'loss': avg_loss,
            'class_loss': avg_class_loss,
            'cont_loss': avg_cont_loss,
            'accuracy': accuracy,
            'f1_avg': f1_avg,
            'f1_other': f1_other,
            'f1_nonwear': f1_nonwear,
            'f1_tso': f1_tso,
            **diag,
        }
        predictions = {
            'preds_other': all_preds_other,
            'preds_nonwear': all_preds_nonwear,
            'preds_tso': all_preds_tso,
            'labels_other': all_labels_other,
            'labels_nonwear': all_labels_nonwear,
            'labels_tso': all_labels_tso
        }

    if interval_records:
        consistency = cross_night_consistency(interval_records)
        metrics.update({
            "mean_pred_tso_duration_hours": float(np.nanmean([x["duration_hours"] for x in interval_records])),
            "mean_pred_tso_segment_count": float(np.nanmean([x["segment_count"] for x in interval_records])),
            **consistency,
        })
    predictions["interval_records"] = interval_records

    if gt_records:
        def _nanmean(key, who):
            vals = [r[who][key] for r in gt_records if not np.isnan(r[who][key])]
            return float(np.mean(vals)) if vals else float("nan")
        for who, tag in (("model", "gt_model"), ("vanhees", "gt_vanhees")):
            metrics[f"{tag}_onset_mae_min"] = _nanmean("onset_mae_min", who)
            metrics[f"{tag}_offset_mae_min"] = _nanmean("offset_mae_min", who)
            metrics[f"{tag}_duration_err_h"] = _nanmean("duration_err_h", who)
            iou_vals = [r[who]["iou"] for r in gt_records if not np.isnan(r[who]["iou"])]
            metrics[f"{tag}_iou"] = float(np.mean(iou_vals)) if iou_vals else float("nan")
        gt_arr = np.array(gt_pm); model_arr = np.array(model_pm); vh_arr = np.array(vanhees_pm)
        metrics["gt_model_f1"] = f1_score(gt_arr, model_arr, zero_division=0)
        metrics["gt_vanhees_f1"] = f1_score(gt_arr, vh_arr, zero_division=0)
        metrics["gt_model_balacc"] = float(balanced_accuracy_score(gt_arr, model_arr))
        metrics["gt_vanhees_balacc"] = float(balanced_accuracy_score(gt_arr, vh_arr))
        metrics["gt_n_nights_both_tso"] = int(sum(
            1 for r in gt_records if r["model"]["pred_has_tso"] and r["model"]["gt_has_tso"]))
        metrics["gt_pred_has_tso_rate"] = float(np.mean([r["model"]["pred_has_tso"] for r in gt_records]))
        metrics["gt_gt_has_tso_rate"] = float(np.mean([r["model"]["gt_has_tso"] for r in gt_records]))

    if verbose:
        extra = "" if num_out_channels == 1 else (
            f" | F1 other: {metrics.get('f1_other', 0):.4f}"
            f" | F1 nonwear: {metrics.get('f1_nonwear', 0):.4f}"
        )
        print(f"  Loss: {avg_loss:.4f} | F1 avg: {f1_avg:.4f} | F1 TSO: {f1_tso:.4f}{extra}")

    return model, metrics, predictions


# ==================== Main Script ====================
parser = argparse.ArgumentParser(description='TSO Status Prediction with H5 Data')
parser.add_argument('--input_h5', type=str, default=None, help='Path to input H5 file (or set data.input_h5 in --config)')
parser.add_argument('--split_file', type=str, default=None, help='Path to train/val split file (.npz); only used when --testing random')
parser.add_argument('--testing', type=str, default='random',
                    choices=['random', 'LOFO', 'LOSO', 'production'],
                    help='CV strategy. random=split_file/random val (legacy, NOT subject-independent); '
                         'LOFO=leave-one-fold-out (subject-independent, matches Deep Scratch); '
                         'LOSO=leave-one-subject-out; production=train and test on all segments.')
parser.add_argument('--single_fold', type=str, default='',
                    help='Restrict LOFO/LOSO to one held-out fold/subject (e.g. FOLD4) — smoke tests or per-fold Domino jobs.')
parser.add_argument('--num_folds', type=int, default=4,
                    help='Number of LOFO folds when the H5 has no FOLD labels (derived from subject ids).')
parser.add_argument('--output', type=str, default=None, help='Output folder name (or set training.output in --config)')
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
# ---- Structural priors and noisy-label regularizers (Deep TSO) ----
parser.add_argument('--w_trans', type=float, default=0.0,
                    help='Weight for transition-count loss (single-segment prior).')
parser.add_argument('--trans_budget', type=float, default=2.0,
                    help='TV-units budget for transition-count loss (default 2 = one segment per sequence).')
parser.add_argument('--w_dur', type=float, default=0.0,
                    help='Weight for duration prior loss (hinge on soft TSO duration).')
parser.add_argument('--dur_min', type=float, default=3.0,
                    help='Lower bound (hours) of plausible TSO duration band.')
parser.add_argument('--dur_max', type=float, default=11.0,
                    help='Upper bound (hours) of plausible TSO duration band.')
parser.add_argument('--w_elr', type=float, default=0.0,
                    help='Weight for Early-Learning Regularization (Liu et al., NeurIPS 2020).')
parser.add_argument('--elr_beta', type=float, default=0.7,
                    help='EMA decay for ELR memory; higher = slower drift toward recent predictions.')
parser.add_argument('--elr_warmup_epochs', type=int, default=5,
                    help='Disable ELR loss until after this epoch (hard cutoff; memory still updates).')
parser.add_argument('--boundary_tau_steps', type=float, default=0.0,
                    help='Boundary-distance decay (in timesteps) for per-position loss reweighting; 0 disables.')
parser.add_argument('--enable_circadian', action='store_true',
                    help='Enable circadian logit-bias module (requires 6-channel data with time_sin/time_cos).')
parser.add_argument('--circadian_gamma_init', type=float, default=0.0,
                    help='Initial value for learnable circadian gamma scalar.')
parser.add_argument('--circadian_smooth_sigma', type=float, default=1.0,
                    help='Gaussian smoothing sigma (in hour-bins) for log_p_circ table.')

parser.add_argument('--finetune_lr', type=float, default=None,
                   help='Learning rate for fine-tuning (lower than training from scratch). If not set, uses best_params["lr"]')

parser.add_argument("--base_loss", type=str, default="ce", choices=["ce", "gce"],
                    help="Base supervised loss for TSO.")
parser.add_argument("--gce_q", type=float, default=0.7,
                    help="GCE q parameter when --base_loss=gce.")
parser.add_argument("--w_supcon", type=float, default=0.0,
                    help="Weight for cross-night supervised contrastive loss.")
parser.add_argument("--supcon_temperature", type=float, default=0.07,
                    help="Temperature for SupConLossV2.")
parser.add_argument("--use_consensus_weight", action="store_true",
                    help="Use Y_annotators agreement as per-minute supervision confidence.")
parser.add_argument("--projection_dim", type=int, default=128,
                    help="Projection dimension for the TSO night embedding head.")
parser.add_argument("--skip_connect", type=lambda v: str(v).lower() in ("1", "true", "yes"),
                    default=True, help="Enable U-Net-style skip connections.")
parser.add_argument("--skip_cross_attention",
                    type=lambda v: str(v).lower() in ("1", "true", "yes"), default=False,
                    help="Use the cross-attention skip path. Default False = the "
                         "self-attention path covered by the factory test.")
parser.add_argument("--config", type=str, default="", help="Path to YAML config file.")
parser.add_argument("--output_root", type=str, default="/mnt/data/GENEActive-featurized/results/DL",
                    help="Root folder for Domino training outputs.")
parser.add_argument("--batch_size", type=int, default=24, help="Batch size.")


def _flatten_config(config):
    flat = {}
    for section in ("data", "model", "training", "loss", "evaluation"):
        values = config.get(section, {})
        if isinstance(values, dict):
            flat.update(values)
    loss_components = config.get("loss", {}).get("components", {})
    if isinstance(loss_components, dict):
        flat.update(loss_components)
    return flat


def _apply_config_defaults(parser, argv):
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default="")
    known, _ = pre_parser.parse_known_args(argv)
    if not known.config:
        return
    with open(known.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    flat = _flatten_config(config)
    defaults = {}
    mapping = {
        "input_h5": "input_h5",
        "split_file": "split_file",
        "output": "output",
        "output_root": "output_root",
        "architecture": "model",
        "epochs": "epochs",
        "batch_size": "batch_size",
        "lr": "finetune_lr",
        "base_loss": "base_loss",
        "gce_q": "gce_q",
        "w_supcon": "w_supcon",
        "supcon_temperature": "supcon_temperature",
        "use_consensus_weight": "use_consensus_weight",
        "projection_dim": "projection_dim",
        "skip_connect": "skip_connect",
        "skip_cross_attention": "skip_cross_attention",
        "w_trans": "w_trans",
        "w_dur": "w_dur",
        "w_elr": "w_elr",
        "boundary_tau_steps": "boundary_tau_steps",
        "enforce_single_tso": "enforce_single_tso",
        "testing": "testing",
        "single_fold": "single_fold",
        "num_folds": "num_folds",
    }
    for key, arg_name in mapping.items():
        if key in flat:
            defaults[arg_name] = flat[key]
    parser.set_defaults(**defaults)


_apply_config_defaults(parser, sys.argv[1:])
args = parser.parse_args()

# input_h5 / output may come from --config (data.input_h5 / training.output) or
# the CLI. They are not argparse-`required` because config values are injected via
# set_defaults, which does not satisfy argparse's required check.
if not args.input_h5:
    parser.error("input_h5 is required: pass --input_h5 or set data.input_h5 in --config")
if not args.output:
    parser.error("output is required: pass --output or set training.output in --config")

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
results_folder = os.path.join(args.output_root, args.output)
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
    'batch_size': args.batch_size,
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
    'skip_connect': args.skip_connect,
    'skip_cross_attention': args.skip_cross_attention,  # default False = tested path
    'use_scaler': False,
    'pretrained_model_path': args.pretrained_model,
    'freeze_layers': args.freeze_layers,
    "projection_dim": args.projection_dim,
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

# ---- Build cross-validation runs (subject-independent for LOFO/LOSO) ----
cv_runs = build_cv_runs(
    args.input_h5, args.testing, args.num_folds, args.val_size,
    args.single_fold, seed=42, split_file=args.split_file,
)
# random/production preserve the --training_iterations repeat; LOFO/LOSO use the
# folds themselves as the iterations (one held-out group per run).
if args.testing in ('LOFO', 'LOSO'):
    runs = cv_runs
else:
    runs = cv_runs * args.training_iterations
print(f"\nTesting strategy: {args.testing} -> {len(runs)} run(s): {[r[0] for r in runs]}")

max_seq_len = 1440  # 24 hours in minutes

# ==================== Training / CV Loop ====================
for iteration, (split_tag, train_indices, val_indices, test_indices) in enumerate(runs):
    print(f"\n{'='*80}")
    print(f"CV run {iteration+1}/{len(runs)}  (testing={args.testing}, held-out={split_tag})")
    print(f"{'='*80}\n")

    # Per-fold datasets. For LOFO/LOSO the test set is the held-out, subject-
    # independent group; val is a carve-out from the remaining training data
    # (used only for model selection / early stopping).
    dataset_train = H5Dataset(args.input_h5, indices=train_indices)
    dataset_val = H5Dataset(args.input_h5, indices=val_indices)
    dataset_test = H5Dataset(args.input_h5, indices=test_indices)
    print(f"  Segments — train: {len(train_indices)}, val: {len(val_indices)}, test: {len(test_indices)}")
    print(f"  Channels: {dataset_train.num_channels} "
          f"({'x,y,z,temp,time_sin,time_cos' if dataset_train.num_channels == 6 else 'x,y,z,temp,time_sin'})")

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

    # ==================== Structural Priors / ELR / Circadian Bias ====================
    # Patch duration in hours: patch_size samples / samples_per_second / 3600 s/h.
    # For the default 1200 samples @ 20Hz that is exactly 1 minute = 1/60 h.
    patch_duration_hours = (
        best_params['patch_size'] / float(dataset_train.samples_per_second) / 3600.0
    )

    # Per-segment EMA buffer for Early-Learning Regularization. Indexed by the
    # H5-file segment_id (constant across train/val splits), so a single
    # ELRMemory works whether the same segment appears in train or val.
    elr_memory = None
    if args.w_elr > 0:
        elr_memory = ELRMemory(
            num_segments=dataset_train.num_segments,
            max_seq_len=max_seq_len,
            num_classes=best_params['output_channels'],
            beta=args.elr_beta,
        )
        print(f"[ELR] Memory initialized: {dataset_train.num_segments} segments "
              f"x {max_seq_len} steps, beta={args.elr_beta}, "
              f"warmup_epochs={args.elr_warmup_epochs}")

    # Circadian logit-bias module: learnable gamma * fixed log_p_circ(hour).
    # log_p_circ is fit from training labels at startup; gamma is trained jointly.
    circadian_bias = None
    if args.enable_circadian:
        if dataset_train.num_channels != 6:
            print(f"[Circadian] WARNING: need 6 channels (have {dataset_train.num_channels}); "
                  f"disabling circadian bias.")
        else:
            print("[Circadian] Fitting log_p_circ from training labels...")
            sample_n = min(200, len(train_indices))
            rng = np.random.default_rng(42)
            sampled = rng.choice(train_indices, size=sample_n, replace=False)
            hours_list, labels_list = [], []
            for sid in sampled:
                x = dataset_train.X[sid]  # [max_len, num_channels]
                y = dataset_train.Y[sid]  # [max_len, 2]
                L = int(dataset_train.seq_lengths[sid])
                if L <= 0:
                    continue
                ts = x[:L, 4].astype(np.float32)
                tc = x[:L, 5].astype(np.float32)
                hours = ((np.arctan2(ts, tc) / (2.0 * np.pi)) * 24.0) % 24.0
                predtso = y[:L, 0]
                nonwear = y[:L, 1]
                lab = np.where(
                    predtso > 0, 2,
                    np.where(nonwear > 0, 1, 0)
                ).astype(np.int64)
                hours_list.append(hours)
                labels_list.append(lab)
            if hours_list:
                all_hours = np.concatenate(hours_list)
                all_labels = np.concatenate(labels_list)
                circadian_bias = CircadianPriorBias.from_training_labels(
                    all_hours, all_labels,
                    num_classes=best_params['output_channels'],
                    gamma_init=args.circadian_gamma_init,
                    smooth_sigma=args.circadian_smooth_sigma,
                ).to(device)
                print(f"[Circadian] Bias module ready "
                      f"(gamma_init={args.circadian_gamma_init}, "
                      f"smooth_sigma={args.circadian_smooth_sigma}, "
                      f"fit_segments={len(hours_list)})")

    # Setup optimizer and scheduler
    optimizer = optim.RMSprop(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=best_params['lr']
    )
    # Add the learnable circadian gamma to the optimizer as a separate param
    # group so it inherits the LR schedule but stays distinguishable.
    if circadian_bias is not None:
        optimizer.add_param_group({'params': list(circadian_bias.parameters())})
        print("[Circadian] Added gamma parameter group to optimizer")

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
        'val_selection_score': [],
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
            # Hard cutoff: ELR loss is 0 until the warmup epoch is reached.
            # The memory itself still accumulates predictions during warmup so
            # the EMA target is meaningful the moment the loss term kicks in.
            elr_active = args.w_elr if epoch >= args.elr_warmup_epochs else 0.0
            model, train_metrics, _ = run_model_tso_h5(
                model, dataset_train, best_params['batch_size'], True, device, optimizer, scheduler,
                max_seq_len=max_seq_len,
                patch_size=best_params['patch_size'],
                padding_value=best_params['padding_value'],
                verbose=(epoch % 5 == 0),
                continuity_weight=args.continuity_weight,
                enforce_single_tso=False,  # No post-processing during training
                min_gap_minutes=args.min_gap_minutes,
                min_duration_minutes=args.min_duration_minutes,
                # ----- Structural priors / ELR / circadian bias -----
                elr_memory=elr_memory, w_elr=elr_active,
                w_trans=args.w_trans, trans_budget=args.trans_budget,
                w_dur=args.w_dur, dur_min=args.dur_min, dur_max=args.dur_max,
                boundary_tau_steps=args.boundary_tau_steps,
                circadian_bias=circadian_bias,
                base_loss=args.base_loss,
                gce_q=args.gce_q,
                w_supcon=args.w_supcon,
                supcon_temperature=args.supcon_temperature,
                use_consensus_weight=args.use_consensus_weight,
                patch_duration_hours=patch_duration_hours,
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
            diag_str = (f" | AUC {train_metrics.get('auc', float('nan')):.3f}"
                        f" | best-F1 {train_metrics.get('best_f1', float('nan')):.3f}"
                        f"@thr {train_metrics.get('best_threshold', float('nan')):.2f}"
                        f" | mean-prob {train_metrics.get('pred_mean_prob', float('nan')):.3f}")
            if train_metrics.get('cont_loss', 0) or train_metrics.get('class_loss', 0):
                print(f"  Train - Loss: {train_metrics['loss']:.4f} (CE: {train_metrics['class_loss']:.4f}, priors: {train_metrics['cont_loss']:.4f}), "
                      f"Acc: {train_metrics['accuracy']:.4f}, F1@0.5: {train_metrics['f1_avg']:.4f}{diag_str}")
            else:
                print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, F1@0.5: {train_metrics['f1_avg']:.4f}{diag_str}")

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
                        min_duration_minutes=args.min_duration_minutes,
                        # ----- Structural priors carry through so val loss is comparable -----
                        # ELR is training-only; pass w_elr=0 and no memory updates happen.
                        elr_memory=None, w_elr=0.0,
                        w_trans=args.w_trans, trans_budget=args.trans_budget,
                        w_dur=args.w_dur, dur_min=args.dur_min, dur_max=args.dur_max,
                        boundary_tau_steps=args.boundary_tau_steps,
                        circadian_bias=circadian_bias,
                        base_loss=args.base_loss,
                        gce_q=args.gce_q,
                        w_supcon=0.0,
                        supcon_temperature=args.supcon_temperature,
                        use_consensus_weight=args.use_consensus_weight,
                        patch_duration_hours=patch_duration_hours,
                    )

                # NOTE: selection_score is a label-free-leaning robustness gate,
                # not a TSO-accuracy metric — val loss is still vs. noisy labels.
                selection_score = (
                    val_metrics["loss"]
                    + 0.05 * val_metrics.get("mean_pred_tso_segment_count", 0.0)
                    + 0.05 * abs(val_metrics.get("mean_pred_tso_duration_hours", 7.0) - 7.0)
                )
                val_metrics["selection_score"] = selection_score

                history['val_loss'].append(val_metrics['loss'])
                history['val_accuracy'].append(val_metrics['accuracy'])
                history['val_f1_avg'].append(val_metrics['f1_avg'])
                history['val_f1_tso'].append(val_metrics['f1_tso'])
                history["val_selection_score"].append(selection_score)
                if best_params['output_channels'] == 3:
                    history['val_f1_other'].append(val_metrics['f1_other'])
                    history['val_f1_nonwear'].append(val_metrics['f1_nonwear'])

                # Print with optional continuity loss breakdown
                vdiag_str = (f" | AUC {val_metrics.get('auc', float('nan')):.3f}"
                             f" | best-F1 {val_metrics.get('best_f1', float('nan')):.3f}"
                             f"@thr {val_metrics.get('best_threshold', float('nan')):.2f}"
                             f" | mean-prob {val_metrics.get('pred_mean_prob', float('nan')):.3f}")
                if val_metrics.get('cont_loss', 0) or val_metrics.get('class_loss', 0):
                    print(f"  Val   - Loss: {val_metrics['loss']:.4f} (CE: {val_metrics['class_loss']:.4f}, priors: {val_metrics['cont_loss']:.4f}), "
                          f"Acc: {val_metrics['accuracy']:.4f}, F1@0.5: {val_metrics['f1_avg']:.4f}{vdiag_str}")
                else:
                    print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1@0.5: {val_metrics['f1_avg']:.4f}{vdiag_str}")

                # Early stopping check (uses label-free-leaning selection score).
                early_stopping(selection_score, model)
                if early_stopping.early_stop:
                    print("Early stopping triggered")
                    break

            # Save best model (best epoch is now defined by selection_score).
            if selection_score == early_stopping.best_score:
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
                    'val_selection_score': selection_score,
                    'history': history,
                    'best_params': best_params,
                    'num_gpus': num_gpus
                }, model_path)
                print(f"  -> Saved best model at epoch {epoch}")

        # Load best model if one was saved. A degenerate run (e.g. no epoch ever
        # improved the selection score) leaves no checkpoint — fall back to the
        # current in-memory weights instead of crashing on a missing file.
        checkpoint_path = os.path.join(train_models_folder, f"best_model_iter_{iteration}.pt")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model_to_load = model.module if isinstance(model, torch.nn.DataParallel) else model
            model_to_load.load_state_dict(checkpoint['model_state_dict'])
        elif args.test_only:
            # In --test_only there is no training to fall back on: an untrained
            # (random) model would silently produce garbage metrics. Fail loudly so
            # the user points --output at the trained run's folder (e.g. its
            # DOMINO_RUN_ID-suffixed name) instead of evaluating random weights.
            raise FileNotFoundError(
                f"--test_only but no checkpoint at {checkpoint_path}. Set --output to the "
                f"trained run's folder so the trained weights are loaded.")
        else:
            print(f"  [warn] no best-model checkpoint at {checkpoint_path} "
                  f"(no epoch improved the selection score); using current weights.")
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
            min_duration_minutes=args.min_duration_minutes,
            # ----- Structural priors / circadian bias (test) -----
            elr_memory=None, w_elr=0.0,
            w_trans=args.w_trans, trans_budget=args.trans_budget,
            w_dur=args.w_dur, dur_min=args.dur_min, dur_max=args.dur_max,
            boundary_tau_steps=args.boundary_tau_steps,
            circadian_bias=circadian_bias,
            base_loss=args.base_loss,
            gce_q=args.gce_q,
            w_supcon=0.0,
            supcon_temperature=args.supcon_temperature,
            use_consensus_weight=args.use_consensus_weight,
            patch_duration_hours=patch_duration_hours,
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
    if 'auc' in test_metrics:
        print(f"\nThreshold diagnostics (does the model rank TSO minutes, or is 0.5 just wrong?):")
        print(f"  AUC (rank quality):     {test_metrics.get('auc', float('nan')):.4f}  "
              f"(~0.5 = learned nothing; >>0.5 = ranks well, threshold/scale is the problem)")
        print(f"  F1 @ 0.5 threshold:     {test_metrics.get('f1_tso', float('nan')):.4f}")
        print(f"  best F1 @ best thr:     {test_metrics.get('best_f1', float('nan')):.4f} "
              f"@ thr={test_metrics.get('best_threshold', float('nan')):.3f}")
        print(f"  TSO positive rate:      {test_metrics.get('pos_rate', float('nan')):.4f}  "
              f"| mean predicted prob: {test_metrics.get('pred_mean_prob', float('nan')):.4f}")
    print(f"{'='*80}\n")

    # Save results
    results = {
        'iteration': iteration,
        'split_tag': split_tag,        # held-out fold/subject for this CV run
        'testing': args.testing,
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

    # Close this fold's datasets before the next CV run.
    dataset_train.close()
    dataset_val.close()
    dataset_test.close()

print(f"\n{'='*80}")
print("Training complete!")
print(f"{'='*80}")
