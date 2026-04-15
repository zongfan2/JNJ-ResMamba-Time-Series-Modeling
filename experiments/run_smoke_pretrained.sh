#!/usr/bin/env bash
# run_smoke_pretrained.sh
# Smoke test for MBA_v1 fine-tuning from UKB DINO pretrained weights.
# Trains a single fold (FOLD4) for a few epochs to verify the transfer path
# end-to-end before launching the full ablation sweep.
#
# Usage (from Domino):
#   bash /mnt/code/munge/predictive_modeling/code/JNJ-ResMamba-Time-Series-Modeling/experiments/run_smoke_pretrained.sh
#
# Override the checkpoint to test the MAE path:
#   bash .../run_smoke_pretrained.sh \
#       --pretrained_model_path /mnt/data/.../pretrain_mae/weights/MAE_encoder_only.pth

set -euo pipefail

ROOT_DIR="/mnt/code/munge/predictive_modeling/code/JNJ-ResMamba-Time-Series-Modeling"

python3.11 "${ROOT_DIR}/training/train_scratch.py" \
    --config "${ROOT_DIR}/experiments/configs/smoke_pretrained_mbav1.yaml" \
    "$@"
