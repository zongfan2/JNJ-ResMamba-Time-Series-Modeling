# Deep TSO

Deep learning for **Time-in-Sleep-Opportunity (TSO)** detection — the nocturnal in-bed
window — from wrist accelerometry, framed as **learning a structured sleep window from
noisy algorithmic labels**. This is the front door for the Deep TSO paper (IMWUT paper 2)
and its experiments.

## The idea in one paragraph

TSO is the denominator for most digital sleep biomarkers, but at scale it is labeled by
heuristic algorithms (van Hees HDCZA, Sadeh, Cole–Kripke) whose noise is **structured at
the onset/offset boundaries**, not i.i.d. A model trained on such labels risks merely
re-learning the labeler, and there is no PSG gold standard to check it. We address this
with (C1) a linear-time Mamba backbone over the 24 h day, (C2) a structured-label-noise
treatment, (C3/C3′) cross-night regularizers that use subject identity, (C4) a structured
single-interval output, and (C5) a **cross-cohort** validation protocol.

## Cross-cohort design (important)

Every experiment **trains/validates on UKB** (`predictTSO` labels only) and **tests on
ALL of noprod** (which carries the cleaner diary-assisted `inTSO` anchor). The cohorts are
disjoint, so test agreement measures genuine cross-cohort generalization — it cannot be
re-learning the same labeler on the same participants. Uncertainty comes from the spread
**across noprod test subjects** (`gt_*_subj_std`), not from CV folds.

## Where things live

| Piece | Path |
|---|---|
| Paper (LaTeX) | `papers/JNJ_deepTSO/` (`main.tex`, `body.tex`) |
| Design spec / Domino runbook | [`experiment_plan.md`](experiment_plan.md) |
| Configs (one per arm) | `experiments/configs/deep_tso/` |
| Run scripts (Domino) | `experiments/deep_tso/` (`build_*`, `run_*`, `tables/`) |
| Per-paper-table scripts | `experiments/deep_tso/tables/` (one script per table) |
| Trainer | `training/train_tso_patch_h5.py` |
| Model | `models/resmamba.py` (`MBA4TSO_Patch`), `models/setup.py` |
| Losses | `losses/structural_priors.py`, `losses/noisy_labels.py` |
| Results inspector | `test-tools/inspect_tso_results.py` |
| Data diagnostics | `test-tools/check_ukb_nights_per_subject.py`, `check_parquet_columns.py` |

## Quick start (on Domino)

```bash
cd $REPO_ROOT
bash experiments/deep_tso/deep_tso_setup.sh          # deps (mamba_ssm), once

# 1. build data: UKB training H5 (no inTSO -> GT_COLUMN="") + noprod test H5
RAW_DIR=<ukb_labelled_parquet> OUTPUT_H5=.../ukb_20hz_sincos.h5 GT_COLUMN="" \
  bash experiments/deep_tso/build_deep_tso_h5.sh
bash experiments/deep_tso/build_deep_tso_h5.sh        # noprod (default paths)

# 2. run experiments — one script per paper table (shared arms train once)
bash experiments/deep_tso/tables/run_all_tables.sh
# or a single table:
bash experiments/deep_tso/tables/table3_crossnight_supcon.sh

# 3. aggregate results (per-arm inTSO metrics + across-test-subject stability std)
bash experiments/deep_tso/tables/inspect.sh
```

Full per-experiment commands, prerequisites, and what-to-check: **[`experiment_plan.md`](experiment_plan.md)**.

## Contributions → tables

| | Contribution | Experiment | Table |
|---|---|---|---|
| C1 | MBA4TSO-Patch backbone | E1 architecture ablation | 4 |
| C2 | Structured-label-noise lens (GCE/ELR don't help) | E2 loss-family ablation | 2 |
| C3 | Cross-night SupCon (subject identity as positive key) | E3 SupCon isolation | 3 |
| C3′ | Positive-only cross-night consistency (no negatives) | E3 consistency arm | 3, 5 |
| C4 | Structured single-interval output | E4 structured output | 6 |
| C5 | Cross-cohort, gold-standard-free validation | E5 protocol | 1, 5 |

## Reading results honestly

- Headline = the `gt_*` metrics **vs the `inTSO` anchor** on the noprod test set.
- `f1_tso` (agreement vs the noisy `predictTSO` training label) is **fidelity to the
  labeler, never accuracy** — never report it as the headline.
- The van-Hees algorithm appears as a **reference row** (its own agreement vs `inTSO`), not
  a beaten baseline: even the labeler only reaches IoU ≈0.87 / F1 ≈0.92 vs `inTSO`.
- Stability (C3/C3′) is the **across-test-subject std** of IoU, not a fold std.
