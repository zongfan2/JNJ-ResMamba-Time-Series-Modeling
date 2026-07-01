# Deep TSO — Domino Experiment Plan

Design spec for the experiments in the Deep TSO paper (`main.tex` / `body.tex`),
written as a Domino runbook. Each section maps a paper experiment (**E1–E5**, the
**C3′** consistency arm, the **Phase‑2** consensus arm, and the **cross‑dataset**
UKB→noprod study) to the concrete config(s), command, what to check, and which
paper table/figure it fills.

> **Everything here runs on Domino.** The local machine has no `mamba_ssm`, no GPU,
> and none of the training data. Do not attempt these locally — local is for editing
> configs, the paper, and inspecting downloaded result CSVs only.

Companion: `experiments/deep_tso/tables/README.md` (per-table run scripts),
`experiments/deep_tso/README.md` (data-build runbook).

---

## 0. Conventions, paths, and the shared protocol

**Repo root (Domino):**
`/mnt/code/munge/predictive_modeling/code/JNJ-ResMamba-Time-Series-Modeling`
(referred to below as `$ROOT`; all commands are run from `$ROOT`).

**Data artifacts**

| Artifact | Path | Built by |
|---|---|---|
| Supervised TSO H5 (noprod, has `inTSO`=`Y_gt`) | `/mnt/data/GENEActive-featurized/h5/deep_tso_20hz_sincos.h5` (+ `_split.npz`) | `experiments/deep_tso/build_deep_tso_h5.sh` |
| Consensus TSO H5 (adds `Y_annotators`) | `/mnt/data/GENEActive-featurized/h5/deep_tso_20hz_sincos_consensus.h5` | `build_deep_tso_h5.sh` with `ANNOTATOR_COLUMNS=...` |
| UKB supervised H5 (predictTSO, no `inTSO`) | `/mnt/data/GENEActive-featurized/h5/ukb_20hz_sincos.h5` | `build_deep_tso_h5.sh` with `RAW_DIR=<ukb> GT_COLUMN=""` |
| UKB unlabelled pretrain H5 (DINO/MAE) | `/mnt/data/GENEActive-featurized/results/DL/UKB_v2/ukb_pretrain_20hz.h5` | `experiments/pretrain/run_preprocess_ukb.sh` |

**Outputs:** every run nests under
`/mnt/data/GENEActive-featurized/results/DL/DeepTSO-JNJ/<output>/`.

**Shared protocol (applies to all supervised arms unless noted):**

- **Cross-cohort split (THE design):** train/val on **UKB** (`predictTSO` only; 10%
  random carve = validation for selection/early-stopping), test on **ALL of noprod**
  (carries the cleaner `inTSO` anchor). One split, not k-fold — set by `input_h5: ukb…`
  + `test_h5: …noprod…` in every config. The cohorts are disjoint, so test = genuine
  cross-cohort generalization. (The old in-domain noprod LOFO design has been removed.)
- **Class balancing:** CE uses `BCEWithLogitsLoss(pos_weight = n_neg/n_pos)` (per-batch,
  capped at 50); the GCE family is wrapped with the **same** `pos_weight` so loss
  families are comparable. This is the "fairness fix".
- **Metrics** (computed every run, vs the cleaner `inTSO` anchor on the noprod test):
  **IoU**, **onset MAE**, **offset MAE** (minutes), **F1**. Uncertainty = **mean ± std
  across noprod test subjects** (`gt_*_subj_std`; the stability evidence for C3/C3′,
  replacing per-fold std). `f1_tso` (agreement vs the noisy van-Hees `predictTSO`
  training label) is **fidelity to the labeler, never the headline**.
- **Single-TSO constraint:** `enforce_single_tso: true` (post-hoc) for all arms except
  E4's structured head, which produces one interval by construction.
- **Batch size:** every config uses `batch_size: 48` across **4 GPUs** (`--multi_gpu`,
  ~12/GPU); the raw-patch model is activation-heavy, so a single 22 GB GPU OOMs at 48 —
  drop `GPU=0` + `batch_size` for a single-GPU debug only.

**Setup (once per Domino workspace/job):**
```bash
bash experiments/deep_tso/deep_tso_setup.sh    # installs deps incl. mamba_ssm
```

**Smoke-test discipline:** before any full sweep, run **one arm for 1 epoch**
(`--epochs 1`; cross-dataset has no folds, so this trains on full UKB once) and confirm the per-epoch log shows a non-zero
`F1 TSO`, the `AUC … best-F1 …@thr` diagnostic, and (for cross-night arms) the
`[supcon] fired on N/M steps` line. Only then launch the sweep.

---

## 1. Data builds (prerequisite for everything)

```bash
# 0. verify the labelled production parquet actually carries the label columns
python3.11 test-tools/check_parquet_columns.py \
  --input_folder /mnt/data/Nocturnal-scratch/geneactive_20hz_3s_b1s_production_train_van_new_enh_lth-rth/raw/
#    exit 0  => predictTSO & non-wear present in all sampled files

# 1a. supervised TSO H5 (noprod) — the main training/eval store (has inTSO as Y_gt)
bash experiments/deep_tso/build_deep_tso_h5.sh
#    -> deep_tso_20hz_sincos.h5 (+ _split.npz)

# 1b. consensus H5 (Phase 2 only) — adds per-algorithm tracks as Y_annotators
ANNOTATOR_COLUMNS="predictTSO,sadeh,cole_kripke" \
OUTPUT_H5=/mnt/data/GENEActive-featurized/h5/deep_tso_20hz_sincos_consensus.h5 \
  bash experiments/deep_tso/build_deep_tso_h5.sh

# 1c. UKB supervised H5 (cross-dataset study only) — predictTSO labels, no inTSO
RAW_DIR=<ukb_labelled_parquet_dir> \
OUTPUT_H5=/mnt/data/GENEActive-featurized/h5/ukb_20hz_sincos.h5 \
GT_COLUMN="" \
  bash experiments/deep_tso/build_deep_tso_h5.sh
```

---

## 2. Experiment matrix → runs

All supervised commands follow the same shape; paths come from the YAML, flags only
**override**:
```bash
python3.11 training/train_tso_patch_h5.py --config <cfg.yaml> --num_gpu 0,1,2,3 --multi_gpu
```

### E1 — Architecture ablation & baselines  *(paper §E1, Table 4 / "Component removals")*

**Claim:** C1 (the MBA4TSO-Patch backbone earns its components).

**(a) Component removals** — one config per removal, each a one-line delta from the
explicit full-backbone reference. Run the whole group with `tables/table4_architecture.sh`:

```bash
bash experiments/deep_tso/tables/table4_architecture.sh   # full + 4 removals = 5 cross-dataset runs
```

| Arm | Config | Delta from full |
|---|---|---|
| Full backbone (reference) | `deep_tso_e1_full.yaml` | — (`blocks_mba 5`, `num_feature_layers 6`, `skip_connect true`, `patch_embed conv`) |
| − Mamba state-space blocks | `deep_tso_e1_no_mamba.yaml` | `blocks_mba: 0` (encoder removed) |
| − conv patch embedding | `deep_tso_e1_no_patch.yaml` | `patch_embed: stat` (mean/std summary + linear, no intra-minute conv) |
| − ResNet feature extractor | `deep_tso_e1_no_resnet.yaml` | `num_feature_layers: 0` (also auto-disables U-Net skips) |
| − U-Net skip connections | `deep_tso_e1_no_skip.yaml` | `skip_connect: false` |

These are enabled by new config/CLI knobs `blocks_mba`, `num_feature_layers`,
`featurelayer`, and `patch_embed` (previously hardcoded / absent in the trainer; now read
from the YAML `model:` block). The full-backbone arm is the same recipe as
`deep_tso_phase1_baseline.yaml` with the architecture pinned explicitly. The `− conv patch
embedding` arm uses the new `StatPatchEmbedding` module (`models/specialized.py`). Other
knobs still available as CLI/config overrides: `skip_cross_attention` (self- vs
cross-attention skip path) and `output_channels {1,3}` (binary vs 3-class head).

> Paper §E1 / Table 4 list exactly these five rows (Full, −U-Net skip, −Mamba, −conv patch
> embedding, −feature extractor) — paper and configs are now in sync.

**(b) Reference baselines** (rows in Table 4, not trained by us as TSO heads unless noted):
- **van-Hees HDCZA** — the training labeler itself, scored vs `inTSO` (the reference row;
  it is *not* a beaten baseline). Produced by the data pipeline / `inspect_tso_results.py`.
- **Sundararajan RF** and **U-Time** — external sleep/wake references; run their published
  pipelines on the same nights or cite published agreement, then add as Table-4 rows.
- Optional DL baselines already scaffolded under `experiments/configs/deep_scratch/ablation/`
  (`ablation_baseline_{bilstm,conv1dts,patchtst,resnet1d,vit1d,…}.yaml`) — these are Deep
  Scratch-oriented; only include in the TSO paper if re-pointed at the TSO H5.

### E2 — Noisy-label ablation  *(paper §E2, Table 2)*

**Claim:** C2 (generic i.i.d.-robust losses do not help van-Hees-quality labels; the
noise is structured at the boundaries). **All arms class-balanced.**

```bash
bash experiments/deep_tso/tables/table2_noisy_label.sh
```
This sweeps (each = one cross-dataset run, train UKB / test noprod):

| Arm | Config | Expectation |
|---|---|---|
| CE (strong baseline) | `deep_tso_phase1_baseline.yaml` | reference |
| GCE | `deep_tso_phase1_gce.yaml` | ≤ CE (no help) |
| GCE + SupCon | `deep_tso_phase1_gce_supcon.yaml` | stability gain |
| GCE + ELR | `deep_tso_phase1_gce_elr.yaml` | **collapses** to all-negative (a result, not a bug) |
| CE + structural priors (gated) | `deep_tso_phase1_structural.yaml` | structure-aware |
| CE + structural priors, 3-class | `deep_tso_phase1_structural_3class.yaml` | avoids 0.5-threshold collapse |
| CE + interval head (C4) | `deep_tso_phase1_interval.yaml` | see E4 |

~7 arms = 7 cross-dataset runs (~100 min/run). Comment out arms in the `configs=(…)`
array for a subset.

### E3 — Cross-night regularizer isolation  *(paper §E3, Table 3 + Table 5)*

**Claim:** C3 / C3′ (subject-identity regularization buys cross-subject *stability*, not
peak accuracy). Two matched single-change pairs off the **same class-balanced CE base**:

**(i) SupCon (contrastive) — C3.** The decisive `CE → CE+SupCon` pair:
```bash
# CE baseline (control) — already run in E2
python3.11 training/train_tso_patch_h5.py \
  --config experiments/configs/deep_tso/deep_tso_phase1_baseline.yaml --num_gpu 0,1,2,3 --multi_gpu
# CE + SupCon
python3.11 training/train_tso_patch_h5.py \
  --config experiments/configs/deep_tso/deep_tso_phase1_ce_supcon.yaml --num_gpu 0,1,2,3 --multi_gpu
```

**(ii) Consistency (positive-only) — C3′ (NEW).** The `CE → CE+consistency` pair, which
keeps same-subject attraction but **drops the contrastive negatives** (the variance
penalty on the predicted window's soft center & duration):
```bash
# CE + consistency  (config sets loss.components.w_consistency: 0.1)
python3.11 training/train_tso_patch_h5.py \
  --config experiments/configs/deep_tso/deep_tso_phase1_consistency.yaml \
  --num_gpu 0,1,2,3 --multi_gpu
# matched control = the SAME config with the term off (single change):
python3.11 training/train_tso_patch_h5.py \
  --config experiments/configs/deep_tso/deep_tso_phase1_consistency.yaml \
  --output deep_tso_phase1_consistency_ce_ctrl --w_consistency 0.0 \
  --num_gpu 0,1,2,3 --multi_gpu
```

> **Batch-size caveat.** `deep_tso_phase1_consistency.yaml` uses `batch_size: 48` (sized
> for a 4-GPU A40 box; the cross-night terms benefit from more subjects×nights per batch).
> On a single 22 GB GPU this OOMs — either run `--multi_gpu` on 4 GPUs, or drop to
> `--batch_size 8` for an apples-to-apples comparison with the single-GPU E2 baseline (at
> the cost of fewer same-subject groups per batch). Pick one regime and keep it identical
> across the matched pair. When `w_consistency>0` the trainer auto-switches to
> `subject_grouped_batch_generator` (≥2 nights/subject), so no extra flag is needed.

Fills: **Table 3** (CE / CE+SupCon / Δ), **Table 5** rows `CE`, `CE+SupCon`,
`CE+consistency`, and Figure `fig:consistency`.

### E4 — Structured single-interval output  *(paper §E4, Table 6)*

**Claim:** C4 (move the single-interval constraint into the model via an onset/offset
regression head, vs per-minute argmax + post-hoc smoothing).

```bash
# SMOKE FIRST — confirm the interval loss is finite & decreasing (1 epoch, full UKB)
python3.11 training/train_tso_patch_h5.py \
  --config experiments/configs/deep_tso/deep_tso_phase1_interval.yaml \
  --epochs 1 --num_gpu 0,1,2,3 --multi_gpu
# full run (cross-dataset)
python3.11 training/train_tso_patch_h5.py \
  --config experiments/configs/deep_tso/deep_tso_phase1_interval.yaml --num_gpu 0,1,2,3 --multi_gpu
```
Compare three decode regimes in Table 6: (1) per-minute argmax + post-hoc smoothing
(the E2 baseline), (2) soft structural priors (`deep_tso_phase1_structural*`), (3) the
onset/offset head (`deep_tso_phase1_interval.yaml`).

> **Wire-up note.** The head's training-time boundary loss (`w_interval: 0.5`) is built
> and unit-tested; using its *decoded* interval to **replace** post-hoc smoothing at eval
> is the remaining integration (EXPERIMENT_PLAN item D4). Verify on Domino before claiming
> Table 6's structured-decode row.

### E5 — Three-tier validation protocol  *(paper §E5, Tables 1 & 5, Figure `fig:valid`)*

Not separate runs — three *views* computed from the runs above (no gold standard exists):

- **(a) inTSO-anchor agreement** — `gt_model_*` vs `gt_vanhees_*` (IoU, onset/offset MAE,
  F1). In every run that has `Y_gt` in the H5. → Table 1.
- **(b) Label-free cross-night consistency** — across-night std of predicted
  onset/offset/duration (`mean_*_std`). In every run, no labels needed. → Table 5.
- **(c) Downstream scratch proxy** — feed each TSO source into a **fixed** Deep Scratch
  model, score scratch F1/index vs independent scratch GT (non-circular). **Not yet built
  (item D6):** needs a frozen Deep Scratch checkpoint + the scratch-GT nights. Build before
  claiming tier (c).

---

## 3. Additional studies

### Phase 2 — Annotator-consensus weighting  *(extends C2/C3)*

Uses the **consensus** H5; weights the per-minute supervision by inter-algorithm
agreement (`use_consensus_weight: true`) on top of GCE+SupCon:
```bash
python3.11 training/train_tso_patch_h5.py \
  --config experiments/configs/deep_tso/deep_tso_phase2_consensus.yaml --num_gpu 0,1,2,3 --multi_gpu
```
Requires data build **1b**. Compare against the GCE+SupCon arm from E2.

### Cross-dataset — train UKB / test noprod  *(generalization stress test)*

Train/val carved from **UKB** (predictTSO only); test = **all** of noprod (carries
`inTSO`). The `--test_h5` path in the config triggers cross-dataset mode:
```bash
# baseline (both cross-night terms off)
python3.11 training/train_tso_patch_h5.py \
  --config experiments/configs/deep_tso/deep_tso_ukb2noprod.yaml --num_gpu 0,1,2,3 --multi_gpu
# choose the cross-night arm at run time:
#   + SupCon       : --w_supcon 0.1
#   + consistency  : --w_consistency 0.1
#   + both         : --w_supcon 0.1 --w_consistency 0.1
```
Requires data build **1c**. This is the strongest test of the C3/C3′ stability claim
(train and test subjects are entirely disjoint *and* from different cohorts).

---

## 4. Inspection & aggregation

```bash
python3.11 test-tools/inspect_tso_results.py --plots \
  /mnt/data/GENEActive-featurized/results/DL/DeepTSO-JNJ
```
Read in order (each arm is a single cross-dataset run, so there is one result per arm,
not a fold aggregate):
1. **GT (inTSO) — model vs van Hees**: the headline `IoU` / `onset MAE` / `offset MAE` /
   `F1` of the model and the van-Hees reference, both vs the `inTSO` anchor on the noprod
   test.
2. **across-test-subject line**: `IoU mean ± std` (and onset/offset ± std) over the noprod
   test subjects — **this std is the C3/C3′ stability number** that replaces per-fold std.
3. Compare arms by their across-subject means and stds (e.g. CE vs CE+SupCon vs
   CE+consistency); `--baseline <substr>` sets the reference arm for deltas.

Never promote `f1_tso` (fidelity to the noisy labeler) to a headline — the trusted
metrics are the `gt_*` ones vs `inTSO`.

---

## 5. Execution order & rough budget

1. `deep_tso_setup.sh` (once) → **§1** data builds (1a always; 1b for Phase 2; 1c for
   cross-dataset).
2. Smoke (1 epoch) ce_supcon and interval before the full sweep.
3. **E2 + E3** (`tables/table2_noisy_label.sh` + `table3_crossnight_supcon.sh`) — the core of the paper.
4. **E3(ii)** consistency pair (multi-GPU).
5. **E4** interval (`deep_tso_phase1_interval.yaml`); **E1** component removals
   (`tables/table4_architecture.sh`).
6. Phase 2 and cross-dataset (optional / generalization).
7. `inspect_tso_results.py` → fill Tables 1–6.

**Budget:** ~100 min/run × ~40 supervised runs (E1–E4 + pairs) ≈ a few GPU-days on one
22 GB GPU; the multi-GPU cross-night and cross-dataset arms are faster per epoch but use
4 GPUs. Sequence the sweeps; don't run all configs blind before the 1-epoch smoke
passes.

---

## Open items (blocking specific table cells)

- **E4 eval wire-up (D4):** decoded interval should replace post-hoc smoothing at test.
- **E5(c) downstream proxy (D6):** fixed Deep Scratch checkpoint + scratch-GT nights.
- **Final E2 numbers** must be the class-balanced re-run; the preliminary values in the
  paper draft predate the fairness fix and are labelled "(prelim)".
