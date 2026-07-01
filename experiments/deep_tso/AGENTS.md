# Deep TSO — agent onboarding

Context for an AI agent (Claude Code / Codex) working on the **Deep TSO** project. Read
this before touching TSO code, configs, experiments, or the paper. The human-facing guide
is `docs/deep_tso/README.md`; the full design spec / Domino runbook is
`docs/deep_tso/experiment_plan.md`. Paths below are relative to the repo root.

## What this project is

Detecting **Time-in-Sleep-Opportunity (TSO)** — the nocturnal in-bed window — from 20 Hz
wrist accelerometry, framed as **learning a structured sleep window from noisy algorithmic
labels**. The training label is the van Hees `predictTSO` heuristic (noisy, with noise
*structured at onset/offset boundaries*, not i.i.d.). There is **no PSG gold standard**, so
evaluation is done against a cleaner diary-assisted `inTSO` anchor. Backbone is
`MBA4TSO_Patch` — a linear-time Mamba state-space model over the 24 h day.

Contributions: **C1** backbone · **C2** structured-label-noise lens · **C3** cross-night
SupCon · **C3′** positive-only cross-night consistency · **C4** structured single-interval
output · **C5** cross-cohort gold-standard-free validation.

## The single most important thing: cross-cohort design

**Every experiment trains/validates on UKB (`predictTSO` only) and tests on ALL of noprod
(which has the `inTSO` anchor).** The cohorts are disjoint. This is set in each config via
`input_h5: …/ukb_20hz_sincos.h5` + `test_h5: …/deep_tso_20hz_sincos.h5` — the presence of
`test_h5` triggers the single-split cross-dataset code path in the trainer (no k-folds).

- **Uncertainty is the spread across noprod TEST SUBJECTS**, not CV folds. The trainer emits
  `gt_model_iou_subj_mean` / `gt_model_iou_subj_std` (and onset/offset subj_std) — these are
  the stability numbers for C3/C3′. There is **no LOFO / no per-fold std** anymore; if you
  see "fold" framing in older docs it is stale.
- Do **not** re-introduce `testing: LOFO` / `split_file` into the table configs.

## Where things live

| Piece | Path |
|---|---|
| Trainer (the one that matters) | `training/train_tso_patch_h5.py` |
| Model + factory | `models/resmamba.py` (`MBA4TSO_Patch`), `models/setup.py` |
| Losses | `losses/structural_priors.py` (priors, `cross_night_consistency_loss`), `losses/noisy_labels.py` (GCE, consensus) |
| Configs (one per arm) | `experiments/configs/deep_tso/` |
| Run scripts (this dir) | `build_deep_tso_h5.sh`, `deep_tso_setup.sh`, `run_deep_tso_{ablation,e1,smoke}.sh` |
| Per-paper-table scripts | `tables/` (one script per table; `_common.sh` has `run_arm`) |
| Results inspector | `test-tools/inspect_tso_results.py` |
| Data diagnostics | `test-tools/check_ukb_nights_per_subject.py`, `check_parquet_columns.py` |
| Paper (LaTeX, untracked) | `papers/JNJ_deepTSO/` (`main.tex`, `body.tex`) |
| Design spec / runbook | `docs/deep_tso/experiment_plan.md` |

Config → table mapping and the run commands are in `tables/README.md`.

## Runs on Domino only

The local machine has **no `mamba_ssm`, no GPU, and no training data**. Do not try to run
training locally. Local work = editing code/configs/paper, `py_compile`/`bash -n` checks,
and reading downloaded result CSVs. Real runs go on Domino (paths under `/mnt/…`,
`python3.11`, 4-GPU DataParallel).

## Conventions & hard-won gotchas (read before running)

1. **Building the UKB training H5 needs `GT_COLUMN=""`.** UKB has no `inTSO`, and
   `convert_h5.py` hard-fails on a missing GT column. The build script uses `=` (not `:=`)
   so an explicit empty value is respected. Also verify `predictTSO` actually exists in the
   parquet (`check_parquet_columns.py`) or the labels silently come out all-zero.
2. **Multi-GPU:** configs are `batch_size: 48` for 4 GPUs. Pass `--num_gpu 0,1,2,3
   --multi_gpu`. A comma-separated `--num_gpu` **without** `--multi_gpu` falls back to the
   first GPU (with a warning) rather than crashing. Batches smaller than the GPU count are
   routed to `model.module` on GPU 0 (no data dropped) — don't "fix" this by dropping them.
3. **`f1_tso` is fidelity to the noisy labeler, never accuracy.** Headline = the `gt_*`
   metrics vs `inTSO`. Report van-Hees as a *reference row* (it only reaches IoU ≈0.87 /
   F1 ≈0.92 vs `inTSO`), not a beaten baseline.
4. **Cross-night terms need ≥2 nights/subject in a batch** (subject-grouped batching kicks
   in automatically when `w_supcon>0` or `w_consistency>0`). UKB has ~7 nights/subject, so
   they fire strongly; confirm a nonzero consistency term at epoch 1.
5. **C3′ consistency is positive-only** — a within-subject variance penalty on the predicted
   window's soft center + duration, with **no negative/repulsion term** (unlike SupCon).
   Motivation: SupCon's "different subjects are dissimilar" negatives are false-negative-prone
   for sleep timing.
6. **Config layout** mirrors the two papers: `experiments/configs/{deep_tso,deep_scratch,
   pretrain}/`. Keep new TSO configs under `deep_tso/`. `deep_tso_ukb2noprod.yaml` is a
   SUPERSEDED template — don't use it; each arm has its own config.
7. **Verify before claiming done:** `python3 -m py_compile` (use a 3.11+ interpreter for
   files with `match`), `bash -n` on scripts, and check YAML has no duplicate keys.

## Typical tasks

- **Add an experiment arm:** copy the closest `experiments/configs/deep_tso/deep_tso_*.yaml`,
  change the loss/model knobs, keep the cross-dataset `data:` block and `batch_size: 48`, add
  a `run_arm <name>` line to the relevant `tables/tableN_*.sh`.
- **Run a table:** `bash experiments/deep_tso/tables/tableN_*.sh` then
  `bash experiments/deep_tso/tables/inspect.sh`.
- **Change a metric:** the eval/metrics live in `run_model_tso_h5` in
  `training/train_tso_patch_h5.py`; per-subject spread is computed there and surfaced by
  `inspect_tso_results.py`.
- **Edit the paper:** `papers/JNJ_deepTSO/{main,body}.tex` (untracked — the human keeps paper
  content out of git). Keep results framed as cross-cohort; the old within-cohort LOFO
  numbers are marked "prior in-domain, pending".
