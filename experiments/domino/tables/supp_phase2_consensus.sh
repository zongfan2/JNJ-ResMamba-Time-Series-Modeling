#!/usr/bin/env bash
# supp_phase2_consensus.sh — PHASE 2 (annotator-consensus weighting).
# SUPPLEMENTARY: not a numbered paper table yet (the method is built but unwritten).
# Extends C2: instead of trusting van-Hees predictTSO alone, relabel each minute by the
# MAJORITY VOTE of >=2 algorithms (van Hees / Sadeh / Cole-Kripke ...) and weight the
# per-minute loss by inter-algorithm AGREEMENT, so disputed boundary minutes (where the
# structured noise lives) are down-weighted. 4-fold subject-independent LOFO on the NOPROD
# consensus H5.
#
# The decisive comparison is the matched pair (single change = consensus weighting):
#   deep_tso_phase2_consensus_ctrl  (GCE+SupCon, consensus OFF)
#   deep_tso_phase2_consensus       (GCE+SupCon, consensus ON)
#
# PREREQUISITE — build the consensus H5 with >=2 DIFFERENT algorithms' binary TSO
# columns stored as Y_annotators (they must be distinct algorithms, NOT a GT column):
#   1. Confirm the columns exist in the parquet:
#        python3.11 -c "import pandas as pd,glob; \
#          print(sorted(pd.read_parquet(sorted(glob.glob('<RAW_DIR>/*.parquet*'))[0]).columns))"
#   2. Build (example columns — replace with the ones your parquet actually has):
#        ANNOTATOR_COLUMNS='predictTSO,sadeh,cole_kripke' \
#        OUTPUT_H5=/mnt/data/GENEActive-featurized/h5/deep_tso_20hz_sincos_consensus.h5 \
#        bash experiments/domino/build_deep_tso_h5.sh
#      (convert_h5 hard-fails on a missing annotator column, so a bad name stops early.)
set -euo pipefail
source "$(dirname "$0")/_common.sh"

CONSENSUS_H5="/mnt/data/GENEActive-featurized/h5/deep_tso_20hz_sincos_consensus.h5"
if [[ ! -f "$CONSENSUS_H5" ]]; then
  echo "[err ] consensus H5 not found: $CONSENSUS_H5" >&2
  echo "       build it first (see the PREREQUISITE block at the top of this script)." >&2
  exit 1
fi

echo "== Phase 2: annotator-consensus weighting =="
run_arm deep_tso_phase2_consensus_ctrl   # GCE+SupCon, consensus OFF (control)
run_arm deep_tso_phase2_consensus        # GCE+SupCon, consensus ON  (single change)

cat <<'NOTE'

Read the matched delta (ctrl -> consensus) on the gt_* metrics vs inTSO:
  bash experiments/domino/tables/inspect.sh
Watch epoch 1: if Y_annotators is missing the run errors immediately
("--use_consensus_weight requires Y_annotators in the H5"). If consensus helps, this
becomes a new paper table/section (needs writing — it is not in the paper yet).
NOTE
