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
from data.loading import load_sequence_data
from data.padding import add_padding
from data.batching import batch_generator


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


def build_model(cfg, device):
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
    max_seq_len = 1220  # default for 3s segments at 20Hz with margin
    model = setup_model("mbav1", input_dim, max_seq_len, best_params,
                        pretraining=False)
    model = model.to(device)
    return model, best_params


def load_checkpoint(model, checkpoint_path, device):
    """Load model weights from checkpoint file."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict):
        # Try common key names
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


def run_inference(model, df, scaler, device, n_samples, seg_column="segment"):
    """Run model inference on the first n_samples segments and collect results."""
    model.eval()

    segments_seen = 0
    results = []

    with torch.no_grad():
        for batch in batch_generator(df=df, batch_size=1, stratify=False,
                                     shuffle=False, seg_column=seg_column):
            if segments_seen >= n_samples:
                break

            seg_name = batch[seg_column].iloc[0]

            # Extract raw signals before standardization for visualization
            raw_x = batch["x"].values.copy()
            raw_y = batch["y"].values.copy()
            raw_z = batch["z"].values.copy()

            # Ground truth labels
            gt_scratch = batch["scratch"].values.copy() if "scratch" in batch.columns else np.zeros(len(batch))
            gt_segment_scratch = batch["segment_scratch"].values[0] if "segment_scratch" in batch.columns else 0
            gt_scratch_duration = batch.get("scratch_duration", pd.Series([0])).values[0] if "scratch_duration" in batch.columns else 0

            # Prepare batch for model (adds padding, standardizes)
            batch_data, label1, label2, label3, x_lens = add_padding(
                batch, device, seg_column
            )

            # Forward pass
            outputs1, outputs2, outputs3, _, _, _ = model(
                batch_data, x_lens
            )

            # Post-process predictions
            pr1_prob = torch.sigmoid(outputs1).cpu().numpy().flatten()
            pr1 = (pr1_prob >= 0.5).astype(float)
            pr2_prob_raw = torch.sigmoid(outputs2).cpu().numpy().flatten()
            pr3 = outputs3.cpu().numpy().flatten()

            # Get the actual sequence length (before padding)
            seq_len = int(x_lens[0])

            # pr2 is already masked to valid positions by MBA_v1.forward(),
            # so pr2_prob_raw has exactly seq_len elements
            pr2_prob = pr2_prob_raw[:seq_len]
            pr2 = (pr2_prob >= 0.5).astype(float)

            # Trim raw signals and ground truth to actual length
            raw_x = raw_x[:seq_len]
            raw_y = raw_y[:seq_len]
            raw_z = raw_z[:seq_len]
            gt_scratch = gt_scratch[:seq_len]

            # Build per-timestep DataFrame for this segment
            sample_df = pd.DataFrame({
                "segment": seg_name,
                "timestep": np.arange(seq_len),
                "x": raw_x,
                "y": raw_y,
                "z": raw_z,
                "gt_scratch": gt_scratch,
                "gt_segment_scratch": gt_segment_scratch,
                "gt_scratch_duration": gt_scratch_duration,
                "pr1": pr1[0],
                "pr1_prob": pr1_prob[0],
                "pr2": pr2,
                "pr2_prob": pr2_prob,
                "pr3": pr3[0],
            })
            results.append(sample_df)
            segments_seen += 1

            if segments_seen % 10 == 0:
                print(f"  Processed {segments_seen}/{n_samples} segments")

    print(f"Inference complete: {segments_seen} segments processed")
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
    scaler_path = args.scaler_path or cfg.get("data", {}).get("scaler_path")

    if not data_folder:
        raise ValueError("Data folder must be specified via --data_folder or in config YAML")

    # Load scaler
    scaler = None
    if scaler_path and os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print(f"Loaded scaler from {scaler_path}")

    # Load data
    print(f"Loading data from {data_folder} ...")
    list_features = ["x", "y", "z"]
    df = load_sequence_data(
        path=data_folder,
        remove_subject_filter=None,
        include_subject_filter=None,
        list_features=list_features,
        motion_filter=True,
    )
    print(f"Loaded {len(df)} rows, {df['segment'].nunique()} segments")

    # Standardize using the saved scaler
    if scaler is not None:
        df[list_features] = scaler.transform(df[list_features])

    # Build and load model
    print("Building model ...")
    model, best_params = build_model(cfg, device)
    model = load_checkpoint(model, args.checkpoint, device)

    # Run inference
    print(f"Running inference on first {args.n_samples} segments ...")
    results_df = run_inference(
        model, df, scaler, device, n_samples=args.n_samples
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
