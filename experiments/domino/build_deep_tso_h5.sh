#!/usr/bin/env bash
# build_deep_tso_h5.sh
# Build the SUPERVISED Deep TSO training H5 (X / Y / seq_lengths / segment_names)
# from the LABELLED GENEActive production parquet, using training/convert_h5.py.
#
# Source data: the GENEActive *production_train* set, whose parquet carries the
# traditional-algorithm TSO labels (predictTSO from van Hees enh lth-rth, non-wear).
# This is NOT UKB: experiments/run_preprocess_ukb.sh turns UKB into an UNLABELLED
# pretraining H5 (/segments/.../x,y,z) for DINO/MAE; TSO training needs the
# labelled, full-day contract produced here.
#
# Prerequisite: verify the parquet actually has the label columns first, else
# convert_h5.py silently writes all-zero labels:
#   python3.11 test-tools/check_parquet_columns.py --input_folder "$RAW_DIR"
#
# Usage (from repo root on Domino):
#   bash experiments/domino/build_deep_tso_h5.sh
# Override any path via env vars:
#   RAW_DIR=... OUTPUT_H5=... SCALER_PATH=... VAL_SIZE=0.1 bash experiments/domino/build_deep_tso_h5.sh

set -euo pipefail

: "${RAW_DIR:=/mnt/data/Nocturnal-scratch/geneactive_20hz_3s_b1s_production_train_van_new_enh_lth-rth/raw/}"
: "${OUTPUT_H5:=/mnt/data/GENEActive-featurized/h5/deep_tso_20hz_sincos.h5}"
: "${VAL_SIZE:=0.1}"
# Scaler applied to x,y,z at build time. Default = the SAME scaler the Deep
# Scratch experiment uses on this exact data (scratch_mbav1_deepscratch.yaml),
# so the TSO H5 shares Deep Scratch's input representation and stays comparable
# for the downstream scratch proxy. (NOT std_scaler_3s.bin — that's only
# train_scratch.py's fallback default, which the experiment overrides.)
# Set SCALER_PATH="" to disable scaling.
: "${SCALER_PATH:=/mnt/data/GENEActive-featurized/results/DL/UKB_v2/mbav1_scaler.joblib}"

mkdir -p "$(dirname "${OUTPUT_H5}")"

cmd=(
  python3.11 training/convert_h5.py
  --input_folder "${RAW_DIR}"
  --output_h5 "${OUTPUT_H5}"
  --use_sincos True
  --create_split
  --val_size "${VAL_SIZE}"
)
if [[ -n "${SCALER_PATH}" ]]; then
  cmd+=(--scaler_path "${SCALER_PATH}")
fi

echo "Building TSO H5: ${OUTPUT_H5}"
"${cmd[@]}"
echo "Done. Split file: ${OUTPUT_H5%.h5}_split.npz"
