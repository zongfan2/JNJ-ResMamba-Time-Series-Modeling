# -*- coding: utf-8 -*-
"""
Generate qualitative prediction samples for Figure 8 of the Deep Scratch paper.

Loads a trained MBA_v1 checkpoint, runs inference on test data, and saves
the first N samples with raw accelerometer signals (x, y, z), ground-truth
scratch labels, model predictions, and mask probabilities.

Usage:
    python evaluation/predict_qualitative.py \
        --config experiments/configs/scratch_mbav1_deepscratch.yaml \
        --checkpoint /path/to/model_weights/mbav1_test_subject_FOLD3_weights.pth \
        --output_dir papers/deep_scratch/fig/qualitative \
        --n_samples 50

The output CSVs can then be used with a plotting script to generate
Figure 8 panels: (a) correct detection, (b) missed event, (c) false positive.
"""

import os
import sys
import argparse
import yaml
import joblib
import numpy as np
import pandas as pd
import torch

# Add project root to sys.path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from models.resmamba import MBA_v1
from models.setup import setup_model
from data.loading import load_data
from data.padding import add_padding


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate qualitative prediction samples for Figure 8."
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML experiment config (e.g. experiments/configs/scratch_mbav1_deepscratch.yaml)"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to trained model checkpoint (.pth or model.pt)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="papers/deep_scratch/fig/qualitative",
        help="Directory to save output CSVs"
    )
    parser.add_argument(
        "--n_samples", type=int, default=50,
        help="Number of segments to save (default: 50)"
    )
    parser.add_argument(
        "--data_folder", type=str, default=None,
        help="Override data folder from config"
    )
    parser.add_argument(
        "--scaler_path", type=str, default=None,
        help="Override scaler path from config"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device (cuda or cpu). Auto-detected if not set."
    )
    return parser.parse_args()


def load_config(config_path):
    """Load YAML config and return flattened settings."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def build_model(cfg, device, checkpoint_path):
    """Instantiate MBA_v1 from the hardcoded param set used in training."""
    # Match the param set from train_scratch.py (param_mba_v1)
    best_params = {
        "batch_size": 32,
        "num_filters": 128,
        "dropout": 0.5,
        "droppath": 0.3,
        "kernel_f": 13,
        "kernel_MBA1": 9,
        "num_feature_layers": 9,
        "blocks_MBA1": 8,
        "wl1": 0.5,
        "wl2": 0.9,
        "wl3": 0.5,
        "optim": "RMSprop",
        "pos_weight_l1": False,
        "pos_weight_l2": True,
        "featurelayer": "ResNet",
        "norm1": "BN",
        "norm2": "GN",
        "norm3": "BN",
        "lr": 0.001,
        "channel_masking_rate": 0,
        "cls_token": True,
        "stratify": "undersample",
    }

    # Apply any model overrides from config
    overrides = cfg.get("model", {}).get("overrides", {})
    if overrides:
        best_params.update(overrides)

    input_dim = 3  # accelerometer x, y, z

    # Infer max_seq_len from checkpoint's positional encoding shape so the
    # model architecture matches the saved weights exactly.
    max_seq_len = _infer_max_seq_len(checkpoint_path, device, fallback=1221)
    print(f"Using max_seq_len={max_seq_len} (from checkpoint PE shape)")

    model = setup_model("mbav1", input_dim, max_seq_len, best_params,
                        pretraining=False)
    model = model.to(device)
    return model, best_params


def _infer_max_seq_len(checkpoint_path, device, fallback=1221):
    """Read positional_encoding.pe shape from checkpoint to get max_seq_len."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict):
        for key in ["state_dict", "model"]:
            if key in checkpoint:
                state_dict = checkpoint[key]
                break
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Look for positional_encoding.pe — shape is [1, num_filters, max_seq_len]
    for k, v in state_dict.items():
        if "positional_encoding.pe" in k:
            return v.shape[-1]
    return fallback


def load_checkpoint(model, checkpoint_path, device):
    """Load model weights from checkpoint file."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict):
        for key in ["state_dict", "model"]:
            if key in checkpoint:
                state_dict = checkpoint[key]
                break
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Handle DDP-wrapped models (keys prefixed with "module.")
    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k.replace("module.", "")] = v

    model.load_state_dict(cleaned, strict=False)
    print(f"Loaded checkpoint from {checkpoint_path}")
    return model


def run_inference(model, df, device, n_samples, seg_column="segment"):
    """Run model inference on the first n_samples segments and collect results.

    Iterates over segments directly (not via batch_generator) because
    batch_generator skips batches with fewer than 2 segments, which would
    prevent single-segment inference.  We group pairs of segments into
    mini-batches of 2 to satisfy BatchNorm constraints, then unpack per
    segment.
    """
    model.eval()

    all_segments = df[seg_column].unique()
    n_to_process = min(n_samples, len(all_segments))
    segments_to_run = all_segments[:n_to_process]
    results = []

    with torch.no_grad():
        # Process in pairs (BatchNorm1d needs batch_size >= 2 in eval mode
        # for safety, though eval() uses running stats).  Pairs also
        # make add_padding work correctly.
        for i in range(0, len(segments_to_run), 2):
            seg_batch = segments_to_run[i:i + 2]
            batch = df[df[seg_column].isin(seg_batch)]

            # If only one segment left, duplicate it so add_padding works
            single_seg = len(seg_batch) == 1
            if single_seg:
                batch = pd.concat([batch, batch], ignore_index=True)
                # Give the duplicate a distinct segment name so groupby works
                dup_mask = batch.index >= len(batch) // 2
                batch.loc[dup_mask, seg_column] = batch.loc[dup_mask, seg_column] + "__dup"

            # Prepare batch for model (pads to max length in batch)
            batch_data, label1, label2, label3, x_lens = add_padding(
                batch, device, seg_column
            )

            # Forward pass
            outputs1, outputs2, outputs3, _, _, _ = model(
                batch_data, x_lens
            )

            # Post-process each segment in the batch
            pr1_prob_all = torch.sigmoid(outputs1).cpu().numpy().flatten()
            pr3_all = outputs3.cpu().numpy().flatten()
            pr2_prob_raw = torch.sigmoid(outputs2).cpu().numpy().flatten()

            # pr2 is masked by MBA_v1 — it concatenates valid positions
            # from all segments in the batch.  Split by x_lens.
            offset = 0
            n_in_batch = 1 if single_seg else len(seg_batch)
            for j in range(n_in_batch):
                seg_name = seg_batch[j]
                seg_rows = df[df[seg_column] == seg_name]
                seq_len = int(x_lens[j])

                # Raw signals for visualization
                raw_x = seg_rows["x"].values[:seq_len].copy()
                raw_y = seg_rows["y"].values[:seq_len].copy()
                raw_z = seg_rows["z"].values[:seq_len].copy()

                # Ground truth
                gt_scratch = seg_rows["scratch"].values[:seq_len].copy() if "scratch" in seg_rows.columns else np.zeros(seq_len)
                gt_segment_scratch = seg_rows["segment_scratch"].values[0] if "segment_scratch" in seg_rows.columns else 0
                gt_scratch_duration = seg_rows["scratch_duration"].values[0] if "scratch_duration" in seg_rows.columns else 0

                # Predictions for this segment
                pr1_prob = float(pr1_prob_all[j])
                pr1 = float(pr1_prob >= 0.5)
                pr3 = float(pr3_all[j])

                # Per-timestep mask predictions (pr2 is flat-concatenated
                # across valid positions of all segments in the batch)
                pr2_prob = pr2_prob_raw[offset:offset + seq_len]
                pr2 = (pr2_prob >= 0.5).astype(float)
                offset += seq_len

                sample_df = pd.DataFrame({
                    "segment": seg_name,
                    "timestep": np.arange(seq_len),
                    "x": raw_x,
                    "y": raw_y,
                    "z": raw_z,
                    "gt_scratch": gt_scratch,
                    "gt_segment_scratch": gt_segment_scratch,
                    "gt_scratch_duration": gt_scratch_duration,
                    "pr1": pr1,
                    "pr1_prob": pr1_prob,
                    "pr2": pr2,
                    "pr2_prob": pr2_prob,
                    "pr3": pr3,
                })
                results.append(sample_df)

            # Skip the duplicate's pr2 offset if we padded a single segment
            if single_seg:
                offset += int(x_lens[1])

            processed = min(i + 2, n_to_process)
            if processed % 10 == 0 or processed == n_to_process:
                print(f"  Processed {processed}/{n_to_process} segments")

    print(f"Inference complete: {len(results)} segments processed")
    if not results:
        raise RuntimeError(
            f"No segments were processed. DataFrame has {len(df)} rows and "
            f"{df[seg_column].nunique()} unique segments. Check data loading."
        )
    return pd.concat(results, ignore_index=True)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Device selection
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths from config or CLI overrides
    data_folder = args.data_folder or cfg.get("data", {}).get("input_folder")

    if not data_folder:
        raise ValueError("Data folder must be specified via --data_folder or in config YAML")

    # Load data using load_data() which adds required columns:
    # segment_scratch, scratch_duration, position_segment, etc.
    print(f"Loading data from {data_folder} ...")
    df = load_data(data_folder, motion_filter=True)
    print(f"Loaded {len(df)} rows, {df['segment'].nunique()} segments")

    # Standardize using the saved scaler
    scaler_path_resolved = args.scaler_path or cfg.get("data", {}).get("scaler_path")
    if scaler_path_resolved and os.path.exists(scaler_path_resolved):
        scaler = joblib.load(scaler_path_resolved)
        print(f"Loaded scaler from {scaler_path_resolved}")
        list_features = ["x", "y", "z"]
        df[list_features] = scaler.transform(df[list_features])

    # Build and load model
    print("Building model ...")
    model, best_params = build_model(cfg, device, args.checkpoint)
    model = load_checkpoint(model, args.checkpoint, device)

    # Run inference
    print(f"Running inference on first {args.n_samples} segments ...")
    results_df = run_inference(
        model, df, device, n_samples=args.n_samples
    )

    # Save outputs
    os.makedirs(args.output_dir, exist_ok=True)

    # Save combined CSV with all samples
    combined_path = os.path.join(args.output_dir, "qualitative_samples.csv")
    results_df.to_csv(combined_path, index=False)
    print(f"Saved combined results to {combined_path}")

    # Also save a summary CSV (one row per segment) for quick inspection
    summary_rows = []
    for seg, grp in results_df.groupby("segment", sort=False):
        gt_has_scratch = grp["gt_segment_scratch"].iloc[0]
        pred_scratch = grp["pr1"].iloc[0]
        gt_scratch_steps = grp["gt_scratch"].sum()
        pred_scratch_steps = grp["pr2"].sum()
        mean_pr2_prob = grp["pr2_prob"].mean()
        max_pr2_prob = grp["pr2_prob"].max()

        # Classify the example type for Figure 8 panel selection
        if gt_has_scratch and pred_scratch:
            example_type = "true_positive"
        elif gt_has_scratch and not pred_scratch:
            example_type = "false_negative"
        elif not gt_has_scratch and pred_scratch:
            example_type = "false_positive"
        else:
            example_type = "true_negative"

        summary_rows.append({
            "segment": seg,
            "seq_len": len(grp),
            "gt_segment_scratch": gt_has_scratch,
            "gt_scratch_duration": grp["gt_scratch_duration"].iloc[0],
            "gt_scratch_steps": gt_scratch_steps,
            "pr1": pred_scratch,
            "pr1_prob": grp["pr1_prob"].iloc[0],
            "pr2_scratch_steps": pred_scratch_steps,
            "pr2_prob_mean": mean_pr2_prob,
            "pr2_prob_max": max_pr2_prob,
            "pr3": grp["pr3"].iloc[0],
            "example_type": example_type,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.output_dir, "qualitative_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary to {summary_path}")

    # Print distribution of example types
    print("\nExample type distribution:")
    for etype, count in summary_df["example_type"].value_counts().items():
        print(f"  {etype}: {count}")


if __name__ == "__main__":
    main()
