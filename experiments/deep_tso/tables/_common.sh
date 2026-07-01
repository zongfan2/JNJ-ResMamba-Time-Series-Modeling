#!/usr/bin/env bash
# _common.sh — shared setup for the per-table run scripts (source this, don't run).
#
# Every paper table is a CROSS-DATASET run: train/val carved from UKB (predictTSO),
# test = ALL of noprod (the cleaner inTSO anchor). One split (not k-fold); the
# stability numbers come from the spread across noprod TEST SUBJECTS (gt_*_subj_std),
# aggregated by test-tools/inspect_tso_results.py. Arms are SHARED across tables (the
# CE baseline feeds Tables 1,2,3,5,6), so run_arm SKIPS any arm whose output already
# exists — run the table scripts in any order and each arm trains exactly once.
# Set FORCE=1 to retrain.
#
# Override paths/GPU via env vars:
#   ROOT=...  OUTPUT_ROOT=...  GPU=0,1,2,3   FORCE=1
set -euo pipefail

ROOT="${ROOT:-/mnt/code/munge/predictive_modeling/code/JNJ-ResMamba-Time-Series-Modeling}"
CFG="${ROOT}/experiments/configs/deep_tso"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/data/GENEActive-featurized/results/DL/DeepTSO-JNJ}"
GPU="${GPU:-0,1,2,3}"           # configs use batch_size 48 -> 4-GPU DataParallel by default

# run_arm <config-basename-without-.yaml> [extra train args...]
# Trains one cross-dataset arm (UKB -> noprod). Output dir name = the config basename,
# so the same arm requested by two tables resolves to one directory and trains once.
run_arm() {
  local name="$1"; shift
  local cfg="${CFG}/${name}.yaml"
  local out="${OUTPUT_ROOT}/${name}"
  [[ -f "$cfg" ]] || { echo "[err ] missing config: $cfg" >&2; return 1; }
  if [[ -d "$out" && -z "${FORCE:-}" ]]; then
    echo "[skip] ${name} (exists; FORCE=1 to retrain)"
    return 0
  fi
  echo "[run ] ${name}  gpu=${GPU}  ${*:-}"
  python3.11 "${ROOT}/training/train_tso_patch_h5.py" \
    --config "$cfg" --output "$name" --output_root "$OUTPUT_ROOT" \
    --num_gpu "$GPU" --multi_gpu "$@"
}

# smoke <config-basename> — 1 epoch end-to-end; use before a full table run to confirm
# the arm trains + evaluates without crashing. NOTE: cross-dataset has no folds, so this
# still trains on the FULL UKB set for one epoch (not instant) — it is a correctness
# check, not a sub-second test.
smoke() {
  local name="$1"; shift
  echo "[smoke] ${name} (1 epoch, full UKB -> noprod)"
  python3.11 "${ROOT}/training/train_tso_patch_h5.py" \
    --config "${CFG}/${name}.yaml" --output "${name}_smoke" --output_root "$OUTPUT_ROOT" \
    --epochs 1 --num_gpu "$GPU" --multi_gpu "$@"
}
