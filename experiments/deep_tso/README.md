# Deep TSO Noisy-Label — Domino Runbook

Scripts and configs for the noisy-label Deep TSO experiments. **All of this runs
on Domino** — the local machine has no `mamba_ssm`, GPU, or training data.

## Scripts

| Script | Purpose |
|---|---|
| `deep_tso_setup.sh` | Install Python deps (run once per workspace/job). |
| `build_deep_tso_h5.sh` | Build the **supervised** TSO H5 (`X`/`Y`/`seq_lengths`/`segment_names`) from the labelled **GENEActive production** parquet via `training/convert_h5.py`, plus a `_split.npz`. |
| `run_deep_tso_smoke.sh` | 2-epoch smoke run (gce+supcon arm) to confirm the path works. |
| `tables/` | Per-paper-table scripts (E1-E6); see tables/README.md. |

> `build_deep_tso_h5.sh` is **not** `experiments/pretrain/run_preprocess_ukb.sh`. The latter
> produces an *unlabelled pretraining* H5 (`/segments/.../x,y,z`) for DINO/MAE.
> TSO training needs the labelled, full-day contract built here.

## End-to-end

```bash
# 0. deps + verify the parquet actually carries the labels
bash experiments/deep_tso/deep_tso_setup.sh
python3.11 test-tools/check_parquet_columns.py \
  --input_folder /mnt/data/Nocturnal-scratch/geneactive_20hz_3s_b1s_production_train_van_new_enh_lth-rth/raw/
#   exit 0 => predictTSO & non-wear present in all sampled files

# 1. build the TSO training H5 (+ split) from the labelled GENEActive production data
#    scaler defaults to the SAME one Deep Scratch uses (UKB_v2/mbav1_scaler.joblib);
#    override with SCALER_PATH=... or SCALER_PATH="" to disable.
bash experiments/deep_tso/build_deep_tso_h5.sh
#   -> /mnt/data/GENEActive-featurized/h5/deep_tso_20hz_sincos.h5 (+ _split.npz)

# 2. smoke — paths come from the config YAML; no env vars needed
bash experiments/deep_tso/run_deep_tso_smoke.sh

# 3. Run experiments (one script per paper table)
bash experiments/deep_tso/tables/run_all_tables.sh

# (optional) override the config paths for a run:
#   INPUT_H5=/some/other.h5 OUTPUT_ROOT=/some/root bash experiments/deep_tso/run_deep_tso_smoke.sh
```

Run a single config. `input_h5`, `split_file`, `output`, and `output_root` all come
from the YAML (`data.*` / `training.*`); pass a flag only to **override** a config
value (CLI wins over YAML):

```bash
# fully config-driven:
python3.11 training/train_tso_patch_h5.py \
  --config experiments/configs/deep_tso/deep_tso_phase1_gce_supcon.yaml --num_gpu 0

# e.g. override just the output name:
python3.11 training/train_tso_patch_h5.py \
  --config experiments/configs/deep_tso/deep_tso_phase1_gce_supcon.yaml --output my_run --num_gpu 0
```

The `run_deep_tso_*.sh` scripts still pass `--input_h5`/`--output`/`--output_root`
via env vars on purpose (per-run output names with `DOMINO_RUN_ID`, smoke overrides).

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
