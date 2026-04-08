#!/usr/bin/env bash
# run_pretrain_ukb_dino.sh
# Self-supervised DINO pretraining on UKB accelerometer data.
# Produces encoder weights that can be loaded into MBA_v1 for finetuning.
#
# Usage (from Domino):
#   bash /mnt/code/munge/predictive_modeling/code/JNJ-ResMamba-Time-Series-Modeling/experiments/run_pretrain_ukb_dino.sh

set -euo pipefail

ROOT_DIR="/mnt/code/munge/predictive_modeling/code/JNJ-ResMamba-Time-Series-Modeling"

python3.11 "${ROOT_DIR}/training/pretrain_ukb.py" \
    --config "${ROOT_DIR}/experiments/configs/pretrain_ukb_dino.yaml" \
    "$@"
