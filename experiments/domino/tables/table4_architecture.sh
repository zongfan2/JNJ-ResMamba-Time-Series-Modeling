#!/usr/bin/env bash
# TABLE 4 — Architecture component ablation (E1).  Claim C1: each backbone component
# earns its place. Each removal is a one-line delta from the full model.
# Cross-dataset: train/val = UKB (predictTSO); test = ALL of noprod (inTSO anchor).
# Uncertainty = spread across noprod test subjects (gt_*_subj_std).
set -euo pipefail
source "$(dirname "$0")/_common.sh"

echo "== Table 4: component removals =="
run_arm deep_tso_e1_full          # Full MBA4TSO-Patch (reference row)
run_arm deep_tso_e1_no_skip       # - U-Net skip connections
run_arm deep_tso_e1_no_mamba      # - Mamba state-space blocks
run_arm deep_tso_e1_no_patch      # - convolutional patch embedding (stat embed)
run_arm deep_tso_e1_no_resnet     # - residual convolutional feature extractor

cat <<'NOTE'

[external baseline rows — not trained here]
  Sundararajan RF, U-Time: run their published pipelines on the same nights and add
  as rows. The full-backbone arm (deep_tso_e1_full) is the same recipe as
  deep_tso_phase1_baseline; they should agree within run-to-run noise.

Aggregate: bash experiments/domino/tables/inspect.sh
NOTE
