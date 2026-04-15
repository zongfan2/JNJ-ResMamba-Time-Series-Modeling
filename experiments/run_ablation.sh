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
#   full_with_pretrain    — MBA_v1 fine-tuned from UKB MAE encoder
#   no_mask_head          — wl2 = 0
#   no_mamba              — blocks_MBA1 = 0
#   no_resnet             — num_feature_layers = 0
#   no_cross_attn         — use_skip_cross_attention = false
#   no_balanced           — stratify = false
#   cls_only              — wl2 = wl3 = 0
#
# Extra CLI args after the variant are forwarded to train_scratch.py, e.g.:
#   bash run_ablation.sh no_mamba --num_gpu 2

set -euo pipefail

ROOT_DIR="/mnt/code/munge/predictive_modeling/code/JNJ-ResMamba-Time-Series-Modeling"
ABLATION_DIR="${ROOT_DIR}/experiments/configs/ablation"

VALID_VARIANTS=(
    full_no_pretrain
    full_with_pretrain
    no_mask_head
    no_mamba
    no_resnet
    no_cross_attn
    no_balanced
    cls_only
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

python3.11 "${ROOT_DIR}/training/train_scratch.py" \
    --config "${CONFIG}" \
    "$@"
