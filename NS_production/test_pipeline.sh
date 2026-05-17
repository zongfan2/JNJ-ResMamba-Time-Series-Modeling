#!/usr/bin/env bash
# test_dl_tso.sh
# Test NS_pipeline with the DL TSO prediction algorithm.
# Processes a single sample: writes raw output and generates a plot.
# No scratch detection is run.
#
# Usage:
#   bash test_dl_tso.sh [data_folder] [target_folder] [model_path] [device]
#
# Defaults (edit these or pass as positional arguments):
#   DATA_FOLDER  – folder containing one NOPROD .csv sensor file
#   OUT_FOLDER   – where results are written
#   MODEL_PATH   – path to the trained mba4tso_patch .pth checkpoint
#   DEVICE       – torch device (cpu | cuda:0)

set -euo pipefail

# ---------- configurable defaults ----------
DATA_FOLDER="/mnt/imported/data/NOPRODNA0029/for_s3"
# OUT_FOLDER="/mnt/data/Nocturnal-scratch/geneactive_20hz_3s_b1s_production_test_scratch/test_tso_preprocess"
# MODEL_PATH="/mnt/data/GENEActive-featurized/results/DL/UKB_v2/TSO_predict-mba4tso_patch-bs=24-dim=128-ch=6/training/model_weights/best_model_iter_0.pt"
OUT_FOLDER="/mnt/data/Nocturnal-scratch/geneactive_20hz_3s_b1s_production_test_scratch/DeepTSO-NOPROD-ch=6-binary"
# MODEL_PATH="/mnt/data/GENEActive-featurized/results/DL/geneactive_20hz_3s_b1s_production_writeall/TSO_predict-mba4tso_patch-bs=24-dim=128-ch=6/training/model_weights/best_model_iter_0.pt"
MODEL_PATH="/mnt/data/GENEActive-featurized/results/DL/geneactive_20hz_3s_b1s_production_writeall/TSO_predict-mba4tso_patch-bs=24-dim=128-ch=6-no-ukb-binary/training/model_weights/best_model_iter_0.pt"
# MODEL_PATH="/mnt/data/GENEActive-featurized/results/DL/UKB_v2/TSO_predict-mba4tso_patch-bs=24-dim=128-ch=6-dlrtc/training/model_weights/best_model_iter_0.pt"

DEVICE="cuda:0"
# -------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PIPELINE="${SCRIPT_DIR}/NS-pipeline.py"
MODEL_NAME="mba4tso_patch"

echo "============================================"
echo " NS_pipeline DL-TSO debug run"
echo "============================================"
echo " data_folder : ${DATA_FOLDER}"
echo " target_folder: ${OUT_FOLDER}"
echo " model_path  : ${MODEL_PATH}"
echo " device      : ${DEVICE}"
echo "============================================"

mkdir -p "${OUT_FOLDER}"

python "${PIPELINE}" \
    --data_folder  "${DATA_FOLDER}" \
    --target_folder "${OUT_FOLDER}" \
    --TSO_algo     dl \
    --tso_model_path  "${MODEL_PATH}" \
    --tso_model_name  "${MODEL_NAME}" \
    --tso_device   "${DEVICE}" \
    --source       NOPROD \
    --write_raw \
    --plot

echo ""
echo "Done. Results written to: ${OUT_FOLDER}"