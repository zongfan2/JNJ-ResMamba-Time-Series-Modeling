#!/usr/bin/env bash
# inspect.sh — aggregate all completed runs into per-arm test metrics (mean +/- across-subject std) and the
# paired-vs-baseline deltas that fill every table. Reads the trusted gt_* metrics vs
# inTSO (Tables 1-4,6) and the cross-night consistency std (Table 5); f1_tso is fidelity
# to the noisy labeler, never the headline.
set -euo pipefail
source "$(dirname "$0")/_common.sh"

python3.11 "${ROOT}/test-tools/inspect_tso_results.py" --plots \
  --baseline deep_tso_phase1_baseline \
  "$OUTPUT_ROOT"
