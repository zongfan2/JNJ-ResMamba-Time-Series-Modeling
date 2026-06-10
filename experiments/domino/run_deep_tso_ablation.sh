#!/usr/bin/env bash
set -euo pipefail

# input_h5 / split_file / output_root come from each config YAML by default.
# Export any of INPUT_H5 / SPLIT_FILE / OUTPUT_ROOT only to OVERRIDE the configs.

configs=(
  experiments/configs/deep_tso_phase1_baseline.yaml
  experiments/configs/deep_tso_phase1_gce.yaml
  experiments/configs/deep_tso_phase1_gce_supcon.yaml
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
