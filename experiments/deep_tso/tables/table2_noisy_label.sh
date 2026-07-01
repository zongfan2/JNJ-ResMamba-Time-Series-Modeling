#!/usr/bin/env bash
# TABLE 2 — Noisy-label loss-family comparison (E2).  Claim C2: generic i.i.d.-robust
# losses don't help van-Hees-quality labels and can collapse. All arms class-balanced.
# Cross-dataset: train/val = UKB (predictTSO); test = ALL of noprod (inTSO anchor).
# Uncertainty = spread across noprod test subjects (gt_*_subj_std).
set -euo pipefail
source "$(dirname "$0")/_common.sh"

echo "== Table 2: noisy-label ablation =="
run_arm deep_tso_phase1_baseline          # CE (strong, class-balanced baseline)
run_arm deep_tso_phase1_gce               # GCE (i.i.d.-robust)            -> expect <= CE
run_arm deep_tso_phase1_gce_supcon        # GCE + SupCon
run_arm deep_tso_phase1_gce_elr           # GCE + ELR   -> EXPECTED to collapse (a result)
run_arm deep_tso_phase1_structural        # CE + gated transition/duration priors
run_arm deep_tso_phase1_structural_3class # same, 3-class (avoids 0.5-threshold collapse)

echo
echo "Aggregate: bash experiments/deep_tso/tables/inspect.sh"
