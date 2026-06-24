# Deep TSO — running the paper experiments on Domino

Companion to the design spec (`papers/deep_tso/EXPERIMENT_PLAN.md`, local-only).
All outputs land under `/mnt/data/GENEActive-featurized/results/DL/DeepTSO-JNJ/`.

## What the experiments answer

| Sweep | Arms | Paper claim |
|---|---|---|
| **E2/E3 ablation** (`run_deep_tso_ablation.sh`) | baseline (CE) · ce_supcon · gce · gce_supcon · gce_elr · structural · structural_3class | C2 (structured-noise lens), C3 (SupCon isolation) |
| E1 architecture | `--skip_connect`/`--skip_cross_attention`/`--output_channels` toggles | C1 (backbone) — *configs TBD* |
| E4 structured output | `deep_tso_phase1_interval.yaml` (onset/offset head) | C4 — head + loss built; verify on Domino |

All arms are **4-fold LOFO** (subject-independent), class-balanced (CE *and* GCE get
`pos_weight`, so loss families are comparable), outputs nested under `DeepTSO-JNJ`.

## 1. Smoke-test one arm/one fold first (~quick)

Before the full sweep, confirm an arm runs end-to-end on a single fold:

```bash
python3.11 training/train_tso_patch_h5.py \
  --config experiments/configs/deep_tso_phase1_ce_supcon.yaml \
  --single_fold FOLD1 --output deep_tso_smoke --num_gpu 0
```

Check the per-epoch log shows: a non-zero `F1 TSO`, the `AUC ... best-F1 ...@thr` diagnostic,
and (for SupCon arms) a `[supcon] fired on N/M steps` line. For `structural*` arms confirm
F1 is **non-zero after the prior warmup** (epoch > `prior_warmup_epochs`, default 5) — if it
collapses to 0, the gating/3-class needs revisiting.

> **Smoke-test the interval arm first.** `deep_tso_phase1_interval.yaml` (the structured
> single-interval head, C4) had its head + loss unit-tested locally, but the full model
> forward + trainer integration could not be run locally (no `mamba_ssm`). Run one fold
> (`--config .../deep_tso_phase1_interval.yaml --single_fold FOLD1`) and confirm the loss is
> finite and decreasing before launching the sweep. The head currently adds a training-time
> boundary loss; using its decoded interval to *replace* post-hoc smoothing at eval is the
> remaining wire-up (see EXPERIMENT_PLAN D4).

## 2. Full noisy-label sweep (E2 + E3)

```bash
bash experiments/domino/run_deep_tso_ablation.sh
```

7 arms × 4 folds ≈ 28 runs (~100 min each). Comment out arms in the `configs=(...)` array
to run a subset. The decisive comparison is **baseline → ce_supcon** (pure SupCon effect on
the fair, class-balanced base). `gce_elr` is expected to collapse — that is a result, not a bug.

## 3. Inspect (fold aggregates + paired significance)

```bash
python3.11 test-tools/inspect_tso_results.py --plots \
  /mnt/data/GENEActive-featurized/results/DL/DeepTSO-JNJ
```

Read in this order:
1. **`!! INCOMPLETE: n/4 folds`** — if present, an arm crashed a fold; don't trust its mean.
2. **FOLD AGGREGATES** — per-arm `mean ± std` across folds. Headline = `IoU vs GT` and
   `onset MAE min` (model vs the cleaner `inTSO` anchor), plus the variance (C3 stability).
3. **PAIRWISE SIGNIFICANCE vs baseline** — per-fold paired deltas, win-count, and a paired-t
   p-value (small n=4 → read the per-fold win-count + mean delta as the honest signal).
   Use `--baseline <substr>` to change the reference arm.

`f1_tso` is **fidelity to the noisy van-Hees training label**, not accuracy — never report it
as the headline. The trusted metrics are the `gt_*` ones vs `inTSO`.

## Validation tiers (C5)
- **(a) `inTSO` anchor** — `gt_model_*` vs `gt_vanhees_*` (in every run; needs `Y_gt` in the H5).
- **(b) cross-night consistency** — `mean_onset/offset/duration_std` (label-free; in every run).
- **(c) downstream scratch proxy** — *not yet built (D6)*; needs a fixed Deep Scratch checkpoint
  + independent scratch GT.

## Key knobs (all YAML- or CLI-settable)
`--testing {LOFO,LOSO,production}` · `--num_folds` · `--single_fold FOLDk` ·
`--base_loss {ce,gce}` · `--gce_balance/--no_gce_balance` · `--gce_q` ·
`--w_supcon` `--supcon_temperature` · `--w_trans --w_dur --prior_warmup_epochs` ·
`--w_elr --elr_warmup_epochs` · `--output_channels {1,3}` · `--use_consensus_weight`.
