#!/usr/bin/env bash
set -euo pipefail

: "${INPUT_H5:?Set INPUT_H5 to the Domino H5 path}"
: "${SPLIT_FILE:=}"
: "${OUTPUT_ROOT:=/mnt/data/GENEActive-featurized/results/DL}"

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
    --input_h5 "${INPUT_H5}"
    --output "${name}_${DOMINO_RUN_ID:-manual}"
    --output_root "${OUTPUT_ROOT}"
    --num_gpu 0
  )
  if [[ -n "${SPLIT_FILE}" ]]; then
    cmd+=(--split_file "${SPLIT_FILE}")
  fi
  "${cmd[@]}"
done
