#!/usr/bin/env bash
# run_ablation.sh
# Deep Scratch ablation study launcher (paper §6 Ablation Study).
#
# Each variant is a separate Domino job. Pass the variant name as $1; the
# script resolves it to experiments/configs/ablation/ablation_<variant>.yaml
# and invokes training/train_scratch.py.
#
# Usage (from Domino):
#   bash /mnt/code/munge/predictive_modeling/code/JNJ-ResMamba-Time-Series-Modeling/experiments/run_ablation.sh <variant>
#
# Variants:
#   full_no_pretrain      — MBA_v1 trained from scratch (ablation baseline)
#   full_with_dino        — MBA_v1 fine-tuned from UKB DINO encoder
#   full_with_mae         — MBA_v1 fine-tuned from UKB MAE  encoder
#   freeze_encoder_dino   — DINO-init, freeze encoder (decoder + heads train)
#   freeze_encdec_dino    — DINO-init, freeze encoder + decoder (heads only)
#   freeze_encoder_mae    — MAE-init,  freeze encoder
#   freeze_encdec_mae     — MAE-init,  freeze encoder + decoder
#   no_mask_head          — wl2 = 0
#   no_mamba              — blocks_MBA1 = 0
#   no_resnet             — num_feature_layers = 0
#   no_cross_attn         — use_skip_cross_attention = false
#   no_cross_attn_dino    — no_cross_attn + DINO pretrained encoder
#   no_balanced           — stratify = false
#   cls_only              — wl2 = wl3 = 0
#
# Baselines (comparison models):
#   baseline_resnet1d     — 1D ResNet-18
#   baseline_mtcna2       — Multi-Task TCN with Attention
#   baseline_patchtst     — Patch Time Series Transformer
#   baseline_efficientunet — EfficientNet U-Net
#   baseline_conv1dts     — Dilated 1D CNN
#   baseline_vit1d        — 1D Vision Transformer
#   baseline_bilstm       — Bidirectional LSTM
#   baseline_mahadevan2021 — classical: RandomForest + 36 hand-crafted features
#   baseline_ji2023       — classical: LightGBM + 36 hand-crafted features (TDA/DL omitted)
#
# Extra CLI args after the variant are forwarded to the training script, e.g.:
#   bash run_ablation.sh no_mamba --num_gpu 2
#
# Classical baselines (mahadevan2021 / ji2023) dispatch to
# training/train_classical.py; everything else uses training/train_scratch.py.

set -euo pipefail

ROOT_DIR="/mnt/code/munge/predictive_modeling/code/JNJ-ResMamba-Time-Series-Modeling"
ABLATION_DIR="${ROOT_DIR}/experiments/configs/ablation"

VALID_VARIANTS=(
    full_no_pretrain
    full_with_dino
    full_with_mae
    freeze_encoder_dino
    freeze_encdec_dino
    freeze_encoder_mae
    freeze_encdec_mae
    no_mask_head
    no_mamba
    no_resnet
    no_cross_attn
    no_cross_attn_dino
    no_balanced
    cls_only
    baseline_resnet1d
    baseline_mtcna2
    baseline_patchtst
    baseline_efficientunet
    baseline_conv1dts
    baseline_vit1d
    baseline_bilstm
    baseline_mahadevan2021
    baseline_ji2023
)

if [[ $# -lt 1 ]]; then
    echo "Error: missing variant argument." >&2
    echo "Usage: $0 <variant> [extra args for train_scratch.py]" >&2
    echo "Valid variants: ${VALID_VARIANTS[*]}" >&2
    exit 2
fi

VARIANT="$1"
shift

MATCHED=0
for v in "${VALID_VARIANTS[@]}"; do
    if [[ "$v" == "$VARIANT" ]]; then
        MATCHED=1
        break
    fi
done
if [[ $MATCHED -eq 0 ]]; then
    echo "Error: unknown variant '$VARIANT'." >&2
    echo "Valid variants: ${VALID_VARIANTS[*]}" >&2
    exit 2
fi

CONFIG="${ABLATION_DIR}/ablation_${VARIANT}.yaml"
if [[ ! -f "$CONFIG" ]]; then
    echo "Error: config not found: $CONFIG" >&2
    exit 2
fi

echo "=============================================="
echo " Deep Scratch ablation run"
echo "   variant: ${VARIANT}"
echo "   config:  ${CONFIG}"
echo "=============================================="

# Classical ML baselines use a different runner.
case "${VARIANT}" in
    baseline_mahadevan2021|baseline_ji2023)
        RUNNER="${ROOT_DIR}/training/train_classical.py"
        ;;
    *)
        RUNNER="${ROOT_DIR}/training/train_scratch.py"
        ;;
esac

python3.11 "${RUNNER}" \
    --config "${CONFIG}" \
    "$@"
