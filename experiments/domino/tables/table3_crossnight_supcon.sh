#!/usr/bin/env bash
# TABLE 3 — Cross-night SupCon isolation (E3).  Claim C3: the decisive matched pair
# CE -> CE+SupCon (single change) buys cross-subject stability, not peak accuracy.
# Same recipe both arms, the only difference is the SupCon term.
# Cross-dataset: train/val = UKB (predictTSO); test = ALL of noprod (inTSO anchor).
# Stability read = across-noprod-subject IoU std (gt_model_iou_subj_std).
set -euo pipefail
source "$(dirname "$0")/_common.sh"

echo "== Table 3: CE -> CE+SupCon =="
run_arm deep_tso_phase1_baseline       # CE   (control)
run_arm deep_tso_phase1_ce_supcon      # CE + SupCon  (single change)

echo
echo "Read the matched delta (IoU, IoU-std across subjects, F1-vs-inTSO):"
echo "  bash experiments/domino/tables/inspect.sh   # uses --baseline deep_tso_phase1_baseline"
