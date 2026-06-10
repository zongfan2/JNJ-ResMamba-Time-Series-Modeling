#!/usr/bin/env bash
# build_deep_tso_h5.sh
# Build the SUPERVISED Deep TSO training H5 (X / Y / seq_lengths / segment_names)
# from UKB raw parquet, using training/convert_h5.py.
#
# NOTE: This is NOT the same as experiments/run_preprocess_ukb.sh, which produces
# an UNLABELLED pretraining H5 (/segments/.../x,y,z) for DINO/MAE. TSO training
# needs the labelled, full-day contract produced here.
#
# Prerequisite: the parquet files must contain `predictTSO` and `non-wear`
# columns (the traditional-algorithm labels). convert_h5.py silently writes
# all-zero labels if they are absent, so verify columns on one file first:
#   python -c "import pandas as pd; print(pd.read_parquet('<one_file>').columns.tolist())"
#
# Usage (from repo root on Domino):
#   bash experiments/domino/build_deep_tso_h5.sh
# Override any path via env vars:
#   UKB_RAW=... OUTPUT_H5=... SCALER_PATH=... VAL_SIZE=0.1 bash experiments/domino/build_deep_tso_h5.sh

set -euo pipefail

: "${UKB_RAW:=/mnt/imported/data/NocturnalScratch_Analysis/UKB_v2/raw/}"
: "${OUTPUT_H5:=/mnt/data/GENEActive-featurized/h5/deep_tso_20hz_sincos.h5}"
: "${VAL_SIZE:=0.1}"
# Leave SCALER_PATH empty for no scaling; set it to reuse the scaler the prior
# TSO/scratch models were trained with (e.g. .../std_scaler_3s.bin).
: "${SCALER_PATH:=}"

mkdir -p "$(dirname "${OUTPUT_H5}")"

cmd=(
  python training/convert_h5.py
  --input_folder "${UKB_RAW}"
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
