#!/usr/bin/env bash
# run_scratch_mbav1.sh
# Launch Deep Scratch training with MBA_v1 using YAML config.
#
# Usage:
#   bash experiments/run_scratch_mbav1.sh
#   bash experiments/run_scratch_mbav1.sh experiments/configs/scratch_mbatsm_deeptso.yaml
#
# CLI args can still override YAML values:
#   bash experiments/run_scratch_mbav1.sh --epochs 100 --num_gpu 1

set -euo pipefail

CONFIG="${1:-experiments/configs/scratch_mbatsm_deeptso.yaml}"
shift 2>/dev/null || true  # consume config arg, pass remaining to train_scratch.py

echo "============================================"
echo " Deep Scratch — Training from YAML config"
echo " config: ${CONFIG}"
echo "============================================"

python training/train_scratch.py \
    --config "${CONFIG}" \
    "$@"
