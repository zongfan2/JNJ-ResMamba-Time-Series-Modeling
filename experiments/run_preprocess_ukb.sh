#!/usr/bin/env bash
# run_preprocess_ukb.sh
# Preprocess UKB accelerometer data into H5 for Deep Scratch pretraining.
#
# Usage (from Domino):
#   bash /mnt/code/munge/predictive_modeling/code/JNJ-ResMamba-Time-Series-Modeling/experiments/run_preprocess_ukb.sh

set -euo pipefail

ROOT_DIR="/mnt/code/munge/predictive_modeling/code/JNJ-ResMamba-Time-Series-Modeling"

python3.11 "${ROOT_DIR}/data/preprocess_ukb_h5.py" \
    --input_folder /mnt/imported/data/NocturnalScratch_Analysis/UKB_v2/raw/ \
    --output_h5 /mnt/data/GENEActive-featurized/results/DL/UKB_v2/ukb_pretrain_20hz.h5 \
    --min_segment_samples 100 \
    --filter_nonwear \
    --filter_stationary \
    "$@"
