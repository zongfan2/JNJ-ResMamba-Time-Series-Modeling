#!/usr/bin/env bash
# TABLE 5 — Cross-night consistency, label-free (E5b).  Within-subject std of predicted
# onset/offset/duration across a subject's NOPROD test nights (lower = more stable).
# Every run emits these, so the CE and CE+SupCon rows reuse the Table 1/3 runs; only the
# positive-only C3' arm is new.
# Cross-dataset: train/val = UKB (predictTSO, ~7 nights/subj -> strong C3' positives);
# test = ALL of noprod. All arms share the batch-48 4-GPU regime (see _common.sh).
set -euo pipefail
source "$(dirname "$0")/_common.sh"

echo "== Table 5: cross-night consistency std =="
run_arm deep_tso_phase1_baseline      # CE          (reused)
run_arm deep_tso_phase1_ce_supcon     # CE+SupCon   (reused)
run_arm deep_tso_phase1_consistency   # CE+consistency (C3', positive-only)

cat <<'NOTE'

Watch epoch 1 of the consistency run for a NONZERO consistency term — UKB has ~7
nights/subject, so it should fire strongly. The label-free std here is measured on the
noprod TEST nights (per subject), so it is not circular with the UKB training objective.

Aggregate: bash experiments/deep_tso/tables/inspect.sh
NOTE
