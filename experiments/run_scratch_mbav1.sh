#!/usr/bin/env bash
# run_scratch_mbav1.sh
# Launch Deep Scratch training with MBA_v1 using YAML config.
#
# Usage (from Domino):
#   bash /mnt/code/munge/predictive_modeling/code/JNJ-ResMamba-Time-Series-Modeling/experiments/run_scratch_mbav1.sh
#
# CLI args can override YAML values:
#   bash experiments/run_scratch_mbav1.sh experiments/configs/scratch_mbatsm_deeptso.yaml --epochs 100

set -euo pipefail

ROOT_DIR="/mnt/code/munge/predictive_modeling/code/JNJ-ResMamba-Time-Series-Modeling"
CONFIG="${1:-${ROOT_DIR}/experiments/configs/scratch_mbav1_deepscratch.yaml}"
shift 2>/dev/null || true  # consume config arg, pass remaining to train_scratch.py

echo "============================================"
echo " Deep Scratch — Training from YAML config"
echo " config: ${CONFIG}"
echo "============================================"

python3.11 "${ROOT_DIR}/training/train_scratch.py" \
    --config "${CONFIG}" \
    "$@"
