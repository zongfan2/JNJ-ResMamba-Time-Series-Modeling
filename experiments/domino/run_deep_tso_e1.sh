#!/usr/bin/env bash
set -euo pipefail

# E1 — architecture ablation (paper §E1 / Table 4, claim C1).
# Full backbone vs one-component-removed variants, all class-balanced CE on the same
# subject-independent 4-fold LOFO split. input_h5 / split_file / output_root come from
# each config YAML; export INPUT_H5 / SPLIT_FILE / OUTPUT_ROOT only to OVERRIDE them.
#
# SMOKE FIRST (one fold) before the full sweep, e.g.:
#   python3.11 training/train_tso_patch_h5.py \
#     --config experiments/configs/deep_tso_e1_no_mamba.yaml \
#     --single_fold FOLD1 --epochs 2 --num_gpu 0
#
# 5 arms x 4 folds = 20 runs (~100 min/run). Comment out arms to run a subset.
configs=(
  experiments/configs/deep_tso_e1_full.yaml       # full backbone (reference row)
  experiments/configs/deep_tso_e1_no_mamba.yaml   # - Mamba state-space encoder
  experiments/configs/deep_tso_e1_no_resnet.yaml  # - ResNet feature extractor (skips auto-off)
  experiments/configs/deep_tso_e1_no_skip.yaml    # - U-Net skip connections
  experiments/configs/deep_tso_e1_no_patch.yaml   # - convolutional patch embedding (stat embed)
)

for config in "${configs[@]}"; do
  name="$(basename "${config}" .yaml)"
  cmd=(
    python3.11 training/train_tso_patch_h5.py
    --config "${config}"
    --output "${name}_${DOMINO_RUN_ID:-manual}"
    --num_gpu 0
  )
  [[ -n "${INPUT_H5:-}" ]]    && cmd+=(--input_h5 "${INPUT_H5}")
  [[ -n "${SPLIT_FILE:-}" ]]  && cmd+=(--split_file "${SPLIT_FILE}")
  [[ -n "${OUTPUT_ROOT:-}" ]] && cmd+=(--output_root "${OUTPUT_ROOT}")
  "${cmd[@]}"
done
