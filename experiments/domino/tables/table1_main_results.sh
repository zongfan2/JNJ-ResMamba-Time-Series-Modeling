#!/usr/bin/env bash
# TABLE 1 — Primary metrics vs the inTSO anchor (E5a).  Claim: honest headline.
#
# Rows: van-Hees HDCZA / Sundararajan RF / U-Time are EXTERNAL reference rows (not
# trained here — see notes), plus our two trained models below. All metrics (IoU,
# onset/offset MAE, F1-vs-inTSO, f1_tso) are emitted by every run and aggregated by
# inspect_tso_results.py.  Cross-dataset: train/val = UKB (predictTSO); test = ALL of
# noprod (inTSO anchor). Uncertainty = spread across noprod test subjects (gt_*_subj_std).
set -euo pipefail
source "$(dirname "$0")/_common.sh"

echo "== Table 1: main results vs inTSO =="
run_arm deep_tso_phase1_baseline       # row: MBA4TSO-Patch, CE
run_arm deep_tso_phase1_ce_supcon      # row: MBA4TSO-Patch, CE+SupCon

cat <<'NOTE'

[external rows — not trained by this script]
  - van-Hees HDCZA: the training labeler scored vs inTSO; produced by the data
    pipeline and printed as the "vanhees" reference by inspect_tso_results.py.
  - Sundararajan RF / U-Time: run their published pipelines on the same nights (or
    cite published agreement) and add as rows manually.

Aggregate the trained rows:
  bash experiments/domino/tables/inspect.sh
NOTE
