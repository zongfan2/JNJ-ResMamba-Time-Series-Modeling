#!/usr/bin/env bash
# run_pretrain_ukb_mae.sh
# Self-supervised MAE pretraining on UKB accelerometer data.
# Produces encoder weights that can be loaded into MBA_v1 for finetuning.
#
# Usage (from Domino):
#   bash /mnt/code/munge/predictive_modeling/code/JNJ-ResMamba-Time-Series-Modeling/experiments/run_pretrain_ukb_mae.sh

set -euo pipefail

ROOT_DIR="/mnt/code/munge/predictive_modeling/code/JNJ-ResMamba-Time-Series-Modeling"

python3.11 "${ROOT_DIR}/training/pretrain_ukb_mae.py" \
    --config "${ROOT_DIR}/experiments/configs/pretrain_ukb_mae.yaml" \
    "$@"
