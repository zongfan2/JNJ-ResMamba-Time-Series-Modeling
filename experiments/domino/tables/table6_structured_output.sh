#!/usr/bin/env bash
# TABLE 6 — Structured single-interval output (E4).  Claim C4: enforce one contiguous
# TSO interval inside the model vs per-minute argmax + post-hoc smoothing.
# #seg = mean predicted TSO segments before post-processing (1 = single clean interval).
# Cross-dataset: train/val = UKB (predictTSO); test = ALL of noprod (inTSO anchor).
# Uncertainty = spread across noprod test subjects (gt_*_subj_std).
set -euo pipefail
source "$(dirname "$0")/_common.sh"

echo "== Table 6: structured output =="
run_arm deep_tso_phase1_baseline      # row: per-minute argmax + post-hoc smoothing (reused)
run_arm deep_tso_phase1_structural    # row: CE + gated structural priors (reused from Table 2)

# Onset-offset regression head (ours, C4). Smoke (1 epoch) first — the head + loss were
# unit-tested but the full trainer integration must be confirmed on Domino.
smoke deep_tso_phase1_interval
run_arm deep_tso_phase1_interval      # row: onset-offset regression head (ours)

echo
echo "Aggregate (the #seg column comes from mean_pred_tso_segment_count):"
echo "  bash experiments/domino/tables/inspect.sh"
