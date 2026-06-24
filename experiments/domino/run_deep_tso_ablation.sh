#!/usr/bin/env bash
set -euo pipefail

# input_h5 / split_file / output_root come from each config YAML by default.
# Export any of INPUT_H5 / SPLIT_FILE / OUTPUT_ROOT only to OVERRIDE the configs.

# Noisy-label ablation (E2) + SupCon isolation (E3). ~7 arms x 4 LOFO folds is a
# large sweep (~100 min/run) — comment out arms to run a subset.
configs=(
  experiments/configs/deep_tso_phase1_baseline.yaml          # CE (class-balanced) — the strong baseline
  experiments/configs/deep_tso_phase1_ce_supcon.yaml         # CE + SupCon — decisive SupCon isolation (E3)
  experiments/configs/deep_tso_phase1_gce.yaml               # GCE (i.i.d.-robust) — now class-balanced
  experiments/configs/deep_tso_phase1_gce_supcon.yaml        # GCE + SupCon
  experiments/configs/deep_tso_phase1_gce_elr.yaml           # GCE + ELR (expected to collapse)
  experiments/configs/deep_tso_phase1_structural.yaml        # CE + transition/duration priors (gated) — structure-aware
  experiments/configs/deep_tso_phase1_structural_3class.yaml # same, 3-class (avoids 0.5-threshold collapse)
  experiments/configs/deep_tso_phase1_interval.yaml          # CE + structured single-interval head (C4, E4)
)

for config in "${configs[@]}"; do
  name="$(basename "${config}" .yaml)"
  cmd=(
    python3.11 training/train_tso_patch_h5.py
    --config "${config}"
    --output "${name}_${DOMINO_RUN_ID:-manual}"
    --num_gpu 0
  )
  [[ -n "${INPUT_H5:-}" ]]    && cmd+=(--input_h5 "${INPUT_H5}")
  [[ -n "${SPLIT_FILE:-}" ]]  && cmd+=(--split_file "${SPLIT_FILE}")
  [[ -n "${OUTPUT_ROOT:-}" ]] && cmd+=(--output_root "${OUTPUT_ROOT}")
  "${cmd[@]}"
done
