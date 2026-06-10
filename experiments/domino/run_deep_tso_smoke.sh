#!/usr/bin/env bash
set -euo pipefail

# input_h5 / split_file / output_root come from the config YAML
# (deep_tso_phase1_gce_supcon.yaml) by default. Export any of INPUT_H5 /
# SPLIT_FILE / OUTPUT_ROOT only to OVERRIDE the config for this run.

cmd=(
  python3.11 training/train_tso_patch_h5.py
  --config experiments/configs/deep_tso_phase1_gce_supcon.yaml
  --output "deep_tso_smoke_${DOMINO_RUN_ID:-manual}"
  --epochs 2
  --batch_size 4
  --val_size 0.05
  --num_gpu 0
)
[[ -n "${INPUT_H5:-}" ]]    && cmd+=(--input_h5 "${INPUT_H5}")
[[ -n "${SPLIT_FILE:-}" ]]  && cmd+=(--split_file "${SPLIT_FILE}")
[[ -n "${OUTPUT_ROOT:-}" ]] && cmd+=(--output_root "${OUTPUT_ROOT}")

"${cmd[@]}"
