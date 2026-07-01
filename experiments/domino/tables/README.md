# Per-table run scripts (Deep TSO paper)

One script per paper table. **Every run is cross-dataset: train/val carved from UKB
(`predictTSO`), test = ALL of noprod (the cleaner `inTSO` anchor).** One split, not
k-fold — uncertainty for the stability claims comes from the spread across noprod TEST
subjects (`gt_*_subj_std`). All runs write under `DeepTSO-JNJ/`. Tables share arms (the
CE baseline feeds five tables); `run_arm` **skips any arm whose output already exists**,
so order doesn't matter and each arm trains once. Aggregation is done by `inspect.sh`.

Prerequisite: build the **UKB** training H5 (`ukb_20hz_sincos.h5`, `GT_COLUMN=""`) and
the **noprod** test H5 (`deep_tso_20hz_sincos.h5`). Configs default to batch 48 on 4
GPUs (`--multi_gpu`).

| Script | Table | Arms |
|---|---|---|
| `table1_main_results.sh` | 1 — main vs inTSO (E5a) | baseline, ce_supcon (+ external refs) |
| `table2_noisy_label.sh` | 2 — loss family (E2) | baseline, gce, gce_supcon, gce_elr, structural, structural_3class |
| `table3_crossnight_supcon.sh` | 3 — SupCon isolation (E3) | baseline → ce_supcon |
| `table4_architecture.sh` | 4 — component removals (E1) | e1_full, e1_no_{skip,mamba,patch,resnet} |
| `table5_crossnight_consistency.sh` | 5 — consistency std (E5b) | baseline, ce_supcon, consistency *(multi-GPU)* |
| `table6_structured_output.sh` | 6 — structured output (E4) | baseline, structural, interval |
| `supp_phase2_consensus.sh` | — (supplementary, unwritten) | Phase 2: consensus-weighting ctrl → on |

## Usage (Domino)

```bash
cd $REPO_ROOT                      # the JNJ-ResMamba repo on /mnt/code
bash experiments/domino/deep_tso_setup.sh         # deps, once

# one table at a time:
bash experiments/domino/tables/table2_noisy_label.sh
bash experiments/domino/tables/inspect.sh

# or everything, then aggregate:
bash experiments/domino/tables/run_all_tables.sh
```

Prerequisites: build the **UKB** training H5 (`ukb_20hz_sincos.h5`, with `GT_COLUMN=""`)
and the **noprod** test H5 (`deep_tso_20hz_sincos.h5`) via
`experiments/domino/build_deep_tso_h5.sh`. Confirm UKB is populated and multi-night with
`test-tools/check_ukb_nights_per_subject.py`.

## Knobs (env vars, see `_common.sh`)

- `GPU=0,1,2,3` (default; configs are batch-48 multi-GPU). Single GPU: `GPU=0` (may OOM at 48).
- `FORCE=1` — retrain an arm even if its output exists.
- `ROOT=` / `OUTPUT_ROOT=` — repo / results roots.

Full per-experiment design rationale: `docs/deep_tso_experiment_plan.md`.
Always read the `gt_*` metrics vs inTSO as the headline; `f1_tso` is fidelity to the
noisy van-Hees label, never accuracy.
