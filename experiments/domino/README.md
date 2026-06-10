# Deep TSO Noisy-Label — Domino Runbook

Scripts and configs for the noisy-label Deep TSO experiments. **All of this runs
on Domino** — the local machine has no `mamba_ssm`, GPU, or training data.

## Scripts

| Script | Purpose |
|---|---|
| `deep_tso_setup.sh` | Install Python deps (run once per workspace/job). |
| `build_deep_tso_h5.sh` | Build the **supervised** TSO H5 (`X`/`Y`/`seq_lengths`/`segment_names`) from the labelled **GENEActive production** parquet via `training/convert_h5.py`, plus a `_split.npz`. |
| `run_deep_tso_smoke.sh` | 2-epoch smoke run (gce+supcon arm) to confirm the path works. |
| `run_deep_tso_ablation.sh` | Phase-1 ablation: CE baseline → GCE → GCE+SupCon. |

> `build_deep_tso_h5.sh` is **not** `experiments/run_preprocess_ukb.sh`. The latter
> produces an *unlabelled pretraining* H5 (`/segments/.../x,y,z`) for DINO/MAE.
> TSO training needs the labelled, full-day contract built here.

## End-to-end

```bash
# 0. deps + verify the parquet actually carries the labels
bash experiments/domino/deep_tso_setup.sh
python3.11 test-tools/check_parquet_columns.py \
  --input_folder /mnt/data/Nocturnal-scratch/geneactive_20hz_3s_b1s_production_train_van_new_enh_lth-rth/raw/
#   exit 0 => predictTSO & non-wear present in all sampled files

# 1. build the TSO training H5 (+ split) from the labelled GENEActive production data
#    scaler optional: set SCALER_PATH to match how prior models were trained
bash experiments/domino/build_deep_tso_h5.sh
#   -> /mnt/data/GENEActive-featurized/h5/deep_tso_20hz_sincos.h5 (+ _split.npz)

# 2. smoke
export INPUT_H5=/mnt/data/GENEActive-featurized/h5/deep_tso_20hz_sincos.h5
export OUTPUT_ROOT=/mnt/data/GENEActive-featurized/results/DL
bash experiments/domino/run_deep_tso_smoke.sh

# 3. Phase-1 ablation
export SPLIT_FILE=/mnt/data/GENEActive-featurized/h5/deep_tso_20hz_sincos_split.npz
bash experiments/domino/run_deep_tso_ablation.sh
```

Run a single config (CLI overrides YAML):

```bash
python3.11 training/train_tso_patch_h5.py \
  --config experiments/configs/deep_tso_phase1_gce_supcon.yaml \
  --input_h5 "$INPUT_H5" --output my_run --output_root "$OUTPUT_ROOT" --num_gpu 0
```

## Notes / gotchas

- `convert_h5.py` writes **all-zero labels** if `predictTSO`/`non-wear` are absent —
  always run `check_parquet_columns.py` first.
- `--use_sincos` in `convert_h5.py` is `type=bool` (argparse footgun): `--use_sincos False`
  still evaluates `True`. Default is already `True` (6 channels) — leave it.
- **Cross-night SupCon** needs ≥2 nights per subject in the data; otherwise the
  zero-positive guard makes it a no-op (no error). Subject identity is read from
  `subject_ids` if present, else the `segment_names` prefix (`<subject>_...`).
- **Phase-2 consensus** needs ≥2 traditional-algorithm label columns in the parquet
  (e.g. Sadeh, Cole–Kripke, van Hees). Rebuild with
  `... --annotator_columns "sadeh,cole_kripke,van_hees"` and run the phase-2 config
  with `--use_consensus_weight`. A single `predictTSO` track is enough for Phase 1.
- Phase-1 model selection uses a **label-free-leaning** `selection_score` (val loss is
  still vs. noisy labels) — a robustness gate, not proof of TSO accuracy. The primary
  downstream scratch-proxy validation is deferred (see the plan's Post-Phase Runway).

See `docs/superpowers/plans/2026-06-09-deep-tso-noisy-labels-domino.md` for the full plan.
