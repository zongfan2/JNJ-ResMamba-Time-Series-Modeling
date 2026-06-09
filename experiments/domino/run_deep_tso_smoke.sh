#!/usr/bin/env bash
set -euo pipefail

: "${INPUT_H5:?Set INPUT_H5 to the Domino H5 path}"
: "${OUTPUT_ROOT:=/mnt/data/GENEActive-featurized/results/DL}"

python training/train_tso_patch_h5.py \
  --config experiments/configs/deep_tso_phase1_gce_supcon.yaml \
  --input_h5 "${INPUT_H5}" \
  --output "deep_tso_smoke_${DOMINO_RUN_ID:-manual}" \
  --output_root "${OUTPUT_ROOT}" \
  --epochs 2 \
  --batch_size 4 \
  --val_size 0.05 \
  --num_gpu 0
