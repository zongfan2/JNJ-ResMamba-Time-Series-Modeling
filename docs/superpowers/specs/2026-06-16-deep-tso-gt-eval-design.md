# Deep TSO — Ground-Truth (`inTSO`) Evaluation Design

**Date:** 2026-06-16
**Status:** Approved design (pre-implementation)
**Sub-project 1 of 3** in the "close the gap to the innovation report" effort.
(2 = direct onset/offset regression; 3 = multiple noisy annotators. Both deferred to
their own spec → plan → build cycles. This sub-project is the *measuring stick* for #2.)

## Problem

Deep TSO currently trains on the noisy van Hees `predictTSO` label **and** scores
against that same noisy label (per-minute F1). Per-minute F1 vs `predictTSO` is
dominated by the easy sleep interior and is blind to onset/offset quality, so a
high number means "reproduces van Hees," not "better TSO." The dataset
(`geneactive_..._production_train_van_new_enh_lth-rth`) also carries `inTSO` /
`TSOSTART` / `TSOEND`, which the team confirms can be treated as **ground truth**,
present for (effectively) all nights.

## Goal

Add a purely-additive **GT evaluation** that answers the core question: does the
model — trained on noisy van Hees — **beat van Hees itself** when both are scored
against `inTSO` GT, using boundary-sensitive metrics? GT is used for **reporting
only**: it never enters the loss, training, or model selection (that may come
later as a separate change).

## Decisions (locked)

1. **Integrated** into the pipeline (store GT in the H5; compute metrics in the
   test pass; land them in the results joblib). Evaluate existing checkpoints
   cheaply with `--test_only` — no retrain.
2. **Three-way, both metric families:** score the **model** and **van Hees**
   predictions against **`inTSO` GT**, with per-minute (F1, balanced accuracy)
   AND interval metrics (onset MAE, offset MAE, IoU, duration error).
3. **Reporting only** — training/early-stopping unchanged; GT never touches the loss.

## Components

### 1. Data contract — `training/convert_h5.py`
- New CLI `--gt_column` (default `inTSO`; empty string disables GT).
- In `load_and_preprocess_segment`: if the GT column exists, read the per-row
  bool as `int8` (error if it was requested but missing — no silent zeros);
  return it under a `gt` key. If `--gt_column` is empty, skip.
- In `convert_parquet_to_h5`: create a presence-guarded dataset **`Y_gt`**
  `(num_segments, max_len)` `int8`, written per segment (zero-padded to `max_len`),
  with the same failed-file trim/resize handling as `Y`. Store
  `h5f.attrs["gt_column"]` for provenance.
- Aggregation to per-minute happens later (in padding), identical to `predictTSO`.

### 2. Loading — `training/train_tso_patch_h5.py` (`H5Dataset`) + `data/padding.py`
- `H5Dataset.__init__`: `self.Y_gt = h5f["Y_gt"] if "Y_gt" in h5f else None`.
- `H5Dataset.__getitem__`: include `"Y_gt"` in the sample when present.
- `add_padding_tso_patch_h5`: aggregate `Y_gt` to per-minute with the **same
  `np.any` rule** used for `pad_Y` (a minute is GT-TSO if any sample in it is),
  and return a **6th value `pad_Y_gt`** (`[batch, minutes]` int, or `None`).
  Update callers: `run_model_tso_h5` unpacks 6; the three `train_tso_dlrtc.py`
  sites take `…, _ = add_padding_tso_patch_h5(...)`.

### 3. Metrics — `evaluation/tso_validation.py` + `run_model_tso_h5`
New pure-numpy helper:
```
interval_agreement(pred_classes, gt_classes, *, timestep_minutes=1.0) -> dict
  # longest contiguous TSO interval on each side (reuse _segments/extract logic)
  returns {
    "pred_has_tso": bool, "gt_has_tso": bool,
    "onset_mae_min": float|nan,   # |pred_onset - gt_onset|, only when BOTH have TSO
    "offset_mae_min": float|nan,  # |pred_offset - gt_offset|, only when BOTH have TSO
    "iou": float,                 # overlap/union of the two intervals; 0 if exactly
                                  #   one has an interval; nan if NEITHER does
    "duration_err_h": float|nan,  # (pred_dur - gt_dur) in hours, only when BOTH have TSO
  }
```
Semantics (explicit, to avoid NaN-poisoned means):
- `onset/offset MAE` and `duration_err` are defined **only on nights where both
  pred and GT have a TSO interval**; aggregate with `np.nanmean` and also report
  the count of such nights.
- `iou` is defined on all nights (0 when exactly one side has an interval),
  averaged over nights where at least one side has an interval.
- Report `pred_has_tso` / `gt_has_tso` rates so "predicts no TSO" is visible.

In `run_model_tso_h5`, when `pad_Y_gt is not None`, in the per-sample loop compute
`interval_agreement` and per-minute agreement for **two predictors** vs GT:
- **model**: the predicted per-minute classes (already derived as `pred_seq`).
- **van Hees**: the **raw** `predictTSO` minutes. Capture `pad_Y_raw = pad_Y.clone()`
  immediately after `add_padding` (BEFORE the Phase-2 consensus relabel), and take
  van Hees TSO = `(pad_Y_raw == 2)`. This keeps the teacher baseline correct even
  in consensus runs.

Aggregate into `metrics` with explicit keys (only present when GT is available):
```
gt_model_f1, gt_model_balacc,
gt_model_onset_mae_min, gt_model_offset_mae_min, gt_model_iou, gt_model_duration_err_h,
gt_vanhees_f1, gt_vanhees_balacc,
gt_vanhees_onset_mae_min, gt_vanhees_offset_mae_min, gt_vanhees_iou, gt_vanhees_duration_err_h,
gt_n_nights_both_tso, gt_pred_has_tso_rate, gt_gt_has_tso_rate
```
These flow into `test_metrics` → the `results_iter_*.joblib`. The existing
noisy-label metrics are unchanged and kept.

### 4. Build + run — `experiments/deep_tso/build_deep_tso_h5.sh`
- New `GT_COLUMN` env (default `inTSO`) → `--gt_column`. Rebuild the H5 once so it
  carries `Y_gt`.
- Evaluate already-trained checkpoints with `--test_only` (loads the best model,
  runs the test pass only) to populate the `gt_*` metrics without retraining.

### 5. Inspector — `test-tools/inspect_tso_results.py`
- Surface the `gt_*` keys, and print a compact **model vs van Hees vs GT** table
  per run (onset MAE, offset MAE, IoU, F1) so "did the model beat its teacher
  against GT?" is one glance.

## Testing / verification
- **Local (pure numpy):** unit-test `interval_agreement` — both-have-TSO MAE/IoU/
  duration, exactly-one-has-TSO (IoU 0, MAE nan), neither (nan/iou nan), perfect
  overlap (IoU 1, MAE 0). Run with the py3.7 env.
- **Local isolation:** extend the `add_padding` isolation check to assert
  `pad_Y_gt` aggregates correctly and is `None` when absent.
- **Byte-compile** the H5/training touch points (mamba-only to run on Domino).
- **Domino:** rebuild H5 with `GT_COLUMN=inTSO`, `--test_only` a phase-1
  checkpoint, confirm `gt_*` keys appear and the inspector renders the table.

## Backward compatibility & scope guards
- Every GT addition is **presence-guarded**: an H5 without `Y_gt` produces no
  `gt_*` metrics and no errors; existing phase-1 H5s and runs are unaffected.
- GT never enters the loss, the training target, or model selection. Purely
  additive evaluation.
- `--gt_column` requested-but-missing raises (no silent-zero GT), mirroring the
  label-column guard.

## Out of scope (explicitly)
- Onset/offset **regression** model head (sub-project 2 — its own brainstorm next).
- GT-driven **model selection** / early stopping (possible fast follow-up).
- Generating Sadeh/Cole-Kripke labels / multi-annotator consensus (sub-project 3).
- Per-subject GT aggregation, clinical metrics (TST/sleep-efficiency) — could be
  added later; not needed for the core "model vs teacher vs GT" question.

## Risks / notes
- If `inTSO` turns out to be only on a subset of nights, the `gt_*` means simply
  cover that subset; the `gt_n_nights_both_tso` count makes the support explicit.
- `inTSO` is assumed reliable GT per the team; if it is itself an algorithm,
  these become "agreement with inTSO," not true accuracy — interpret accordingly.
