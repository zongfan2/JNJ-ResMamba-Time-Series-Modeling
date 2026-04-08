#!/usr/bin/env bash
# run_scratch_mbav1.sh
# Launch Deep Scratch training with MBA_v1 (production architecture)
# on Domino or local GPU environments.
#
# Usage:
#   bash experiments/run_scratch_mbav1.sh
#
# This script mirrors the YAML config: experiments/configs/scratch_mbatsm_deeptso.yaml
# Edit variables below or override via environment:
#   INPUT_FOLDER=/path/to/data bash experiments/run_scratch_mbav1.sh

set -euo pipefail

# ---------- configurable defaults ----------
INPUT_FOLDER="${INPUT_FOLDER:-/mnt/data/Nocturnal-scratch/geneactive_20hz_3s_b1s_production_train_van_new_enh_lth-rth/raw/}"
MODEL="${MODEL:-mbav1}"
OUTPUT="${OUTPUT:-ns_detect-mbav1-bs=32-param_mba_v1}"
EXECUTION_MODE="${EXECUTION_MODE:-train}"
EPOCHS="${EPOCHS:-200}"
NUM_GPU="${NUM_GPU:-0}"
SCALER_PATH="${SCALER_PATH:-/mnt/code/munge/predictive_modeling/std_scaler_3s.bin}"
PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH:-}"
FREEZE_ENCODER="${FREEZE_ENCODER:-False}"
DATA_AUGMENTATION="${DATA_AUGMENTATION:-0}"
# -------------------------------------------

echo "============================================"
echo " Deep Scratch — MBA_v1 Training"
echo "============================================"
echo " input_folder : ${INPUT_FOLDER}"
echo " model        : ${MODEL}"
echo " output       : ${OUTPUT}"
echo " epochs       : ${EPOCHS}"
echo " gpu          : ${NUM_GPU}"
echo " scaler       : ${SCALER_PATH}"
echo "============================================"

python training/train_scratch.py \
    --input_data_folder "${INPUT_FOLDER}" \
    --model "${MODEL}" \
    --output "${OUTPUT}" \
    --execution_mode "${EXECUTION_MODE}" \
    --epochs "${EPOCHS}" \
    --num_gpu "${NUM_GPU}" \
    --scaler_path "${SCALER_PATH}" \
    --data_augmentation "${DATA_AUGMENTATION}" \
    --clear_tracker True \
    --pretrained_model_path "${PRETRAINED_MODEL_PATH}" \
    --freeze_encoder "${FREEZE_ENCODER}"

echo ""
echo "Done. Results written to: /mnt/data/GENEActive-featurized/results/DL/.../${OUTPUT}/"
