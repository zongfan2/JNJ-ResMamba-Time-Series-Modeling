# Deep TSO Ground-Truth (`inTSO`) Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a purely-additive evaluation that scores both the model and van Hees `predictTSO` against `inTSO` ground truth (per-minute F1/balanced-acc + interval onset/offset MAE, IoU, duration error), so we can see whether the model beats its noisy teacher.

**Architecture:** Store `inTSO` as a per-minute `Y_gt` track in the H5 (build-time), surface it through `H5Dataset`/`add_padding_tso_patch_h5` as a 6th batch value, and compute GT metrics in `run_model_tso_h5`'s evaluation loop using a new pure-numpy `interval_agreement` helper. GT is reporting-only — never in the loss/target/selection.

**Tech Stack:** PyTorch, H5Py, NumPy, scikit-learn metrics, existing Deep TSO modules. All non-mamba code is unit/byte-testable locally with `/Users/zongfan/opt/miniconda3/envs/pytorch/bin/python`; the model/training paths run on Domino.

Design spec: `docs/superpowers/specs/2026-06-16-deep-tso-gt-eval-design.md`.

---

## File Structure
- Modify `evaluation/tso_validation.py`: add `interval_agreement()`.
- Modify `evaluation/__init__.py`: export it.
- Create `tests/test_deep_tso_gt_eval.py`: unit tests for `interval_agreement`.
- Modify `training/convert_h5.py`: `--gt_column` → store `Y_gt`.
- Modify `training/train_tso_patch_h5.py`: `H5Dataset` reads `Y_gt`; `run_model_tso_h5` computes GT metrics.
- Modify `data/padding.py`: aggregate `Y_gt` per-minute, return 6th value `pad_Y_gt`.
- Modify `training/train_tso_dlrtc.py`: absorb the new 6th return at 3 call sites.
- Modify `experiments/domino/build_deep_tso_h5.sh`: `GT_COLUMN` env.
- Modify `test-tools/inspect_tso_results.py`: surface `gt_*` keys.

---

### Task 1: `interval_agreement` metric helper

**Files:**
- Modify: `evaluation/tso_validation.py`
- Modify: `evaluation/__init__.py`
- Test: `tests/test_deep_tso_gt_eval.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_deep_tso_gt_eval.py`:

```python
import numpy as np


def test_interval_agreement_identical():
    from evaluation.tso_validation import interval_agreement
    a = np.array([0, 0, 1, 1, 1, 0])
    r = interval_agreement(a, a, timestep_minutes=1.0)
    assert r["pred_has_tso"] and r["gt_has_tso"]
    assert r["onset_mae_min"] == 0.0 and r["offset_mae_min"] == 0.0
    assert r["iou"] == 1.0 and r["duration_err_h"] == 0.0


def test_interval_agreement_shifted():
    from evaluation.tso_validation import interval_agreement
    pred = np.array([0, 0, 1, 1, 1, 0, 0])   # [2,5)
    gt = np.array([0, 0, 0, 1, 1, 1, 0])     # [3,6)
    r = interval_agreement(pred, gt, timestep_minutes=1.0)
    assert r["onset_mae_min"] == 1.0 and r["offset_mae_min"] == 1.0
    assert abs(r["iou"] - 0.5) < 1e-9           # inter=2, union=4
    assert r["duration_err_h"] == 0.0           # same length


def test_interval_agreement_one_missing():
    from evaluation.tso_validation import interval_agreement
    r = interval_agreement(np.array([0, 0, 0, 0]), np.array([0, 1, 1, 0]))
    assert r["pred_has_tso"] is False and r["gt_has_tso"] is True
    assert r["iou"] == 0.0
    assert np.isnan(r["onset_mae_min"]) and np.isnan(r["duration_err_h"])


def test_interval_agreement_neither():
    from evaluation.tso_validation import interval_agreement
    z = np.array([0, 0, 0])
    r = interval_agreement(z, z)
    assert r["pred_has_tso"] is False and r["gt_has_tso"] is False
    assert np.isnan(r["iou"]) and np.isnan(r["onset_mae_min"])
```

- [ ] **Step 2: Run the tests (expect ImportError)**

```bash
/Users/zongfan/opt/miniconda3/envs/pytorch/bin/python -c "import sys; sys.path.insert(0,'.'); import tests.test_deep_tso_gt_eval as t; t.test_interval_agreement_identical()"
```
Expected before implementation: `ImportError: cannot import name 'interval_agreement'`.

- [ ] **Step 3: Implement `interval_agreement`**

In `evaluation/tso_validation.py`, add after `extract_tso_interval` (it reuses that function):

```python
def interval_agreement(pred_classes: np.ndarray, gt_classes: np.ndarray,
                       *, timestep_minutes: float = 1.0) -> dict:
    """Compare predicted vs GT TSO intervals (longest contiguous block each side).

    onset/offset MAE and duration error are defined only when BOTH sides have a
    TSO interval (else NaN). IoU is 0 when exactly one side has an interval, NaN
    when neither does.
    """
    pred = extract_tso_interval(np.asarray(pred_classes), timestep_minutes=timestep_minutes)
    gt = extract_tso_interval(np.asarray(gt_classes), timestep_minutes=timestep_minutes)
    pred_has = pred["segment_count"] > 0
    gt_has = gt["segment_count"] > 0
    out = {
        "pred_has_tso": bool(pred_has),
        "gt_has_tso": bool(gt_has),
        "onset_mae_min": np.nan,
        "offset_mae_min": np.nan,
        "duration_err_h": np.nan,
        "iou": np.nan,
    }
    if pred_has and gt_has:
        out["onset_mae_min"] = abs(pred["onset_minute"] - gt["onset_minute"])
        out["offset_mae_min"] = abs(pred["offset_minute"] - gt["offset_minute"])
        out["duration_err_h"] = pred["duration_hours"] - gt["duration_hours"]
        inter = max(0.0, min(pred["offset_minute"], gt["offset_minute"])
                    - max(pred["onset_minute"], gt["onset_minute"]))
        union = ((pred["offset_minute"] - pred["onset_minute"])
                 + (gt["offset_minute"] - gt["onset_minute"]) - inter)
        out["iou"] = float(inter / union) if union > 0 else 0.0
    elif pred_has != gt_has:
        out["iou"] = 0.0
    return out
```

- [ ] **Step 4: Export it**

In `evaluation/__init__.py`, change the tso_validation import to include it and add to `__all__`:

```python
from .tso_validation import extract_tso_interval, cross_night_consistency, interval_agreement
```
Add `'interval_agreement',` to `__all__`.

- [ ] **Step 5: Run the tests (expect pass) + byte-compile**

```bash
/Users/zongfan/opt/miniconda3/envs/pytorch/bin/python -c "
import sys; sys.path.insert(0,'.')
import tests.test_deep_tso_gt_eval as t
t.test_interval_agreement_identical(); t.test_interval_agreement_shifted()
t.test_interval_agreement_one_missing(); t.test_interval_agreement_neither()
print('ALL 4 PASS')
"
/Users/zongfan/opt/miniconda3/envs/pytorch/bin/python -m py_compile evaluation/tso_validation.py evaluation/__init__.py tests/test_deep_tso_gt_eval.py
```
Expected: `ALL 4 PASS`, clean compile.

- [ ] **Step 6: Commit**

```bash
git add evaluation/tso_validation.py evaluation/__init__.py tests/test_deep_tso_gt_eval.py
git commit -m "feat: add interval_agreement (onset/offset MAE, IoU) for GT eval

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Store `inTSO` as `Y_gt` in the H5

**Files:**
- Modify: `training/convert_h5.py`

Mirrors the existing `subject_ids` / `Y_annotators` handling. Anchors (current): `load_and_preprocess_segment` adds `'subject': current_subject,` (~:169); `ds_subjects` created (~:332); `ds_Y_annotators` block (~:338-345); write loop has `ds_segments[idx]=...`/`ds_subjects[idx]=...` (~:426-427) and the `if ds_Y_annotators is not None:` annotator write (~:417); trim block has `if ds_Y_annotators is not None:` resize (~:445).

- [ ] **Step 1: Read the GT column in `load_and_preprocess_segment`**

Change the signature to add `gt_column=None`:

```python
def load_and_preprocess_segment(
    file,
    scaler,
    max_samples,
    samples_per_second=20,
    use_sincos=True,
    annotator_columns=None,
    gt_column=None,
):
```

Just before `return result`, add (read the per-row GT bool; error if requested-but-missing):

```python
        if gt_column:
            if gt_column not in df.columns:
                raise ValueError(f"GT column {gt_column!r} not found in {file}")
            result["gt"] = df[gt_column].values.astype(np.int8)
```

- [ ] **Step 2: Add `gt_column` to `convert_parquet_to_h5` + create `Y_gt`**

Change `convert_parquet_to_h5(...)` signature to add `gt_column=None,`. Near the top of the body (next to `annotator_columns = annotator_columns or []`) the value is already a string or None — no normalization needed.

After the `ds_subjects = h5f.create_dataset("subject_ids", ...)` block, add:

```python
        ds_Y_gt = None
        if gt_column:
            h5f.attrs["gt_column"] = gt_column
            ds_Y_gt = h5f.create_dataset(
                "Y_gt",
                shape=(num_segments, max_len),
                dtype=np.int8,
                chunks=(1, min(1200, max_len)),
                compression="gzip",
                compression_opts=4,
            )
```

Pass `gt_column` into the loader call (where `annotator_columns=annotator_columns,` is passed):

```python
                gt_column=gt_column,
```

After the `ds_subjects[idx] = segment_data["subject"]` write, add (pad to max_len like Y):

```python
            if ds_Y_gt is not None:
                gt_seg = segment_data["gt"]
                if seg_len < max_len:
                    gt_seg = np.concatenate([gt_seg, np.zeros(max_len - seg_len, dtype=np.int8)])
                ds_Y_gt[idx] = gt_seg
```

In the trim block (next to the `ds_Y_annotators` resize), add:

```python
            if ds_Y_gt is not None:
                ds_Y_gt.resize((actual_segments, max_len))
```

- [ ] **Step 3: Add the CLI arg + pass it**

In the `if __name__ == "__main__":` parser, add:

```python
    parser.add_argument("--gt_column", type=str, default="",
                        help="Per-row binary GT TSO column (e.g. inTSO) -> stored as Y_gt for evaluation.")
```

In the `convert_parquet_to_h5(...)` call, add:

```python
        gt_column=args.gt_column or None,
```

- [ ] **Step 4: Byte-compile**

```bash
/Users/zongfan/opt/miniconda3/envs/pytorch/bin/python -m py_compile training/convert_h5.py
```
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add training/convert_h5.py
git commit -m "feat: convert_h5 --gt_column stores inTSO as Y_gt for GT eval

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Surface `Y_gt` through `H5Dataset` and padding

**Files:**
- Modify: `training/train_tso_patch_h5.py` (`H5Dataset` only)
- Modify: `data/padding.py`
- Modify: `training/train_tso_dlrtc.py`
- Test: `tests/test_deep_tso_gt_eval.py` (padding isolation check)

- [ ] **Step 1: `H5Dataset` reads `Y_gt`**

In `H5Dataset.__init__` (after the `self.Y_annotators = ...` line, ~:125), add:

```python
        self.Y_gt = self.h5f["Y_gt"] if "Y_gt" in self.h5f else None
```

In `H5Dataset.__getitem__`, where the sample dict is returned, add `Y_gt` when present (place beside the existing `if self.Y_annotators is not None: sample["Y_annotators"] = ...`):

```python
        if self.Y_gt is not None:
            sample["Y_gt"] = self.Y_gt[actual_idx]
```

- [ ] **Step 2: `add_padding_tso_patch_h5` aggregates `Y_gt` → 6th return**

In `data/padding.py`, in the collection loop (after `if "Y_annotators" in sample:` block, ~:599) add collection:

```python
        gt_batch.append(sample.get("Y_gt"))
```
And initialize before the loop (next to `annotator_batch = []`):

```python
    gt_batch = []
    has_gt = False
```
Set `has_gt = True` inside the loop when present — replace the `gt_batch.append(...)` above with:

```python
        if "Y_gt" in sample:
            has_gt = True
            gt_batch.append(sample["Y_gt"])
```

After `pad_Y` is created (next to the `pad_Y_annotators = None` block, ~:612), add:

```python
    pad_Y_gt = None
    if has_gt:
        gt_arr = np.stack(gt_batch)  # [batch, max_len]
        pad_Y_gt = np.zeros((batch_size, num_minutes_max), dtype=np.int64)
```

Inside the per-minute loop (next to the `if pad_Y_annotators is not None:` block, ~:649), add:

```python
                if pad_Y_gt is not None:
                    gt_minute = gt_arr[i, m * patch_size:(m + 1) * patch_size]
                    pad_Y_gt[i, m] = int(np.any(gt_minute))
```

Before the return, convert + return 6 values:

```python
    if pad_Y_gt is not None:
        pad_Y_gt = torch.from_numpy(pad_Y_gt).to(device)

    return pad_X, pad_Y, x_lens, segments_batch, pad_Y_annotators, pad_Y_gt
```

- [ ] **Step 3: Update the `train_tso_dlrtc.py` callers (3 sites)**

At lines ~659, ~806, ~1013, change each:

```python
        pad_X, pad_Y, x_lens, batch_samples, _ = add_padding_tso_patch_h5(
```
to:

```python
        pad_X, pad_Y, x_lens, batch_samples, _, _ = add_padding_tso_patch_h5(
```
(Preserve each site's indentation — line 1013 is more indented.)

- [ ] **Step 4: Local isolation check for the padding aggregation**

Append to `tests/test_deep_tso_gt_eval.py`:

```python
def test_add_padding_returns_pad_y_gt():
    import torch
    from data.padding import add_padding_tso_patch_h5

    class _DS:
        num_channels = 6
        def __init__(self, with_gt):
            self.with_gt = with_gt
            self._x = __import__("numpy").zeros((120, 6), "float32")
            self._y = __import__("numpy").zeros((120, 2), "int8"); self._y[:60, 0] = 1
            self._g = __import__("numpy").zeros((120,), "int8"); self._g[:60] = 1  # GT minute0
        def __getitem__(self, i):
            d = {"X": self._x, "Y": self._y, "seq_length": 120,
                 "segment": "S%d_0_d" % i, "segment_id": i}
            if self.with_gt:
                d["Y_gt"] = self._g
            return d

    out = add_padding_tso_patch_h5(_DS(True), [0], torch.device("cpu"),
                                   max_seq_len=1440, patch_size=60, num_channels=6)
    assert len(out) == 6
    pad_Y_gt = out[5]
    assert pad_Y_gt is not None and tuple(pad_Y_gt.shape) == (1, 2)
    assert pad_Y_gt[0, 0].item() == 1 and pad_Y_gt[0, 1].item() == 0

    out2 = add_padding_tso_patch_h5(_DS(False), [0], torch.device("cpu"),
                                    max_seq_len=1440, patch_size=60, num_channels=6)
    assert len(out2) == 6 and out2[5] is None
```

Run it + byte-compile:

```bash
/Users/zongfan/opt/miniconda3/envs/pytorch/bin/python -c "
import sys; sys.path.insert(0,'.')
import tests.test_deep_tso_gt_eval as t
t.test_add_padding_returns_pad_y_gt(); print('PADDING GT CHECK PASS')
"
/Users/zongfan/opt/miniconda3/envs/pytorch/bin/python -m py_compile data/padding.py training/train_tso_dlrtc.py
```
Expected: `PADDING GT CHECK PASS`, clean compile. (`train_tso_patch_h5.py` is byte-compiled in Task 4.)

- [ ] **Step 5: Commit**

```bash
git add training/train_tso_patch_h5.py data/padding.py training/train_tso_dlrtc.py tests/test_deep_tso_gt_eval.py
git commit -m "feat: surface Y_gt through H5Dataset + add_padding (6th return pad_Y_gt)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Compute GT metrics in `run_model_tso_h5`

**Files:**
- Modify: `training/train_tso_patch_h5.py`

Anchors (current): the `add_padding_tso_patch_h5(...)` call unpacks 5 values (~:429); the consensus block starts with `supervision_weight = None` (~:478); accumulators incl. `interval_records = []` (~:419); the per-sample loop computes `pred_seq` and `valid_labels` (~:662-696); the `metrics = {...}` dict is built after the loop (~:685+ for binary, with 3-class additions).

- [ ] **Step 1: Import `balanced_accuracy_score` and `interval_agreement`**

Add `balanced_accuracy_score` to the existing `from sklearn.metrics import ...` line, and add to the evaluation import:

```python
from evaluation.tso_validation import extract_tso_interval, cross_night_consistency, interval_agreement
```
(If the file currently imports only `extract_tso_interval, cross_night_consistency`, extend it.)

- [ ] **Step 2: Unpack `pad_Y_gt` and capture raw van Hees labels**

Change the padding unpack (~:429) to 6 values:

```python
        pad_X, pad_Y, x_lens, batch_samples, pad_Y_annotators, pad_Y_gt = add_padding_tso_patch_h5(
```

Immediately after that call (before the forward / consensus), capture the raw teacher labels:

```python
        pad_Y_raw = pad_Y.clone()  # van Hees predictTSO, before any Phase-2 consensus relabel
```

- [ ] **Step 3: Add GT accumulators**

Next to `interval_records = []` (~:419), add:

```python
    gt_records = []        # per-night {"model": interval_agreement, "vanhees": interval_agreement}
    gt_pm, model_pm, vanhees_pm = [], [], []   # aligned per-minute GT / model / van Hees (0/1)
```

- [ ] **Step 4: Per-sample GT collection**

Inside the per-sample loop, right after `interval_records.append(interval)` (~:675), add:

```python
            if pad_Y_gt is not None:
                gt_seq = pad_Y_gt[i, :valid_len].cpu().numpy().astype(int)
                vanhees_seq = (pad_Y_raw[i, :valid_len] == 2).cpu().numpy().astype(int)
                tso_class = 2 if num_out_channels > 1 else 1
                model_bin = (pred_seq == tso_class).astype(int)
                gt_records.append({
                    "model": interval_agreement(pred_seq, gt_seq, timestep_minutes=1.0),
                    "vanhees": interval_agreement(vanhees_seq, gt_seq, timestep_minutes=1.0),
                })
                gt_pm.extend(gt_seq.tolist())
                model_pm.extend(model_bin.tolist())
                vanhees_pm.extend(vanhees_seq.tolist())
```

- [ ] **Step 5: Aggregate into `metrics` after it is built**

After the `metrics = {...}` dict is constructed (and after any 3-class keys are added to it, just before `predictions = {...}`/the return), add:

```python
    if gt_records:
        def _nanmean(key, who):
            vals = [r[who][key] for r in gt_records if not np.isnan(r[who][key])]
            return float(np.mean(vals)) if vals else float("nan")
        for who, tag in (("model", "gt_model"), ("vanhees", "gt_vanhees")):
            metrics[f"{tag}_onset_mae_min"] = _nanmean("onset_mae_min", who)
            metrics[f"{tag}_offset_mae_min"] = _nanmean("offset_mae_min", who)
            metrics[f"{tag}_duration_err_h"] = _nanmean("duration_err_h", who)
            iou_vals = [r[who]["iou"] for r in gt_records if not np.isnan(r[who]["iou"])]
            metrics[f"{tag}_iou"] = float(np.mean(iou_vals)) if iou_vals else float("nan")
        gt_arr = np.array(gt_pm); model_arr = np.array(model_pm); vh_arr = np.array(vanhees_pm)
        metrics["gt_model_f1"] = f1_score(gt_arr, model_arr, zero_division=0)
        metrics["gt_vanhees_f1"] = f1_score(gt_arr, vh_arr, zero_division=0)
        metrics["gt_model_balacc"] = float(balanced_accuracy_score(gt_arr, model_arr))
        metrics["gt_vanhees_balacc"] = float(balanced_accuracy_score(gt_arr, vh_arr))
        metrics["gt_n_nights_both_tso"] = int(sum(
            1 for r in gt_records if r["model"]["pred_has_tso"] and r["model"]["gt_has_tso"]))
        metrics["gt_pred_has_tso_rate"] = float(np.mean([r["model"]["pred_has_tso"] for r in gt_records]))
        metrics["gt_gt_has_tso_rate"] = float(np.mean([r["model"]["gt_has_tso"] for r in gt_records]))
```

(If `metrics` is built in two branches — binary vs 3-class — place this block after BOTH, i.e. after the metric dict exists on every path and before the function returns. `metrics` must already be defined.)

- [ ] **Step 6: Byte-compile**

```bash
/opt/homebrew/bin/python3 -m py_compile training/train_tso_patch_h5.py
```
(Uses homebrew py3.14 because the file has `match/case`; the pytorch env can't compile it but Domino runs it. Expected: clean.)

- [ ] **Step 7: Commit**

```bash
git add training/train_tso_patch_h5.py
git commit -m "feat: compute model-vs-vanHees-vs-GT metrics against inTSO in eval

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Build-script GT flag + inspector surfacing

**Files:**
- Modify: `experiments/domino/build_deep_tso_h5.sh`
- Modify: `test-tools/inspect_tso_results.py`

- [ ] **Step 1: Add `GT_COLUMN` to the build script**

In `experiments/domino/build_deep_tso_h5.sh`, next to the `ANNOTATOR_COLUMNS` default, add:

```bash
# Per-row GT TSO column -> stored as Y_gt for evaluation. Default inTSO; "" disables.
: "${GT_COLUMN:=inTSO}"
```

In the `cmd=(...)` assembly, after the annotator-columns append, add:

```bash
if [[ -n "${GT_COLUMN}" ]]; then
  cmd+=(--gt_column "${GT_COLUMN}")
fi
```

- [ ] **Step 2: Verify the build script**

```bash
bash -n experiments/domino/build_deep_tso_h5.sh && echo "bash -n OK"
tmp=$(mktemp -d); printf '#!/usr/bin/env bash\necho PYCALL "$@"\n' > "$tmp/python3.11"; chmod +x "$tmp/python3.11"
PATH="$tmp:$PATH" OUTPUT_H5="$tmp/x.h5" SCALER_PATH="" RAW_DIR=/r bash experiments/domino/build_deep_tso_h5.sh 2>&1 | grep -o "gt_column inTSO"
rm -rf "$tmp"
```
Expected: `bash -n OK` and `gt_column inTSO`.

- [ ] **Step 3: Surface `gt_*` in the inspector**

In `test-tools/inspect_tso_results.py`, after the existing per-run metric printing, add a GT block. In `inspect(path)`, after the label-free TSO metrics print, add:

```python
    gt_keys = [k for k in tm if k.startswith("gt_")]
    if gt_keys:
        print("  GT (inTSO) — model vs van Hees:")
        rows = [("onset MAE (min)", "gt_model_onset_mae_min", "gt_vanhees_onset_mae_min"),
                ("offset MAE (min)", "gt_model_offset_mae_min", "gt_vanhees_offset_mae_min"),
                ("IoU", "gt_model_iou", "gt_vanhees_iou"),
                ("F1 vs GT", "gt_model_f1", "gt_vanhees_f1"),
                ("balanced acc", "gt_model_balacc", "gt_vanhees_balacc")]
        print(f"    {'metric':18s} {'model':>10s} {'vanHees':>10s}")
        for name, mk, vk in rows:
            print(f"    {name:18s} {fmt(tm.get(mk)):>10s} {fmt(tm.get(vk)):>10s}")
        print(f"    nights both-TSO: {fmt(tm.get('gt_n_nights_both_tso'))} | "
              f"pred-has-TSO: {fmt(tm.get('gt_pred_has_tso_rate'))} | "
              f"gt-has-TSO: {fmt(tm.get('gt_gt_has_tso_rate'))}")
```

Also add the headline GT columns to the comparison row dict (in `inspect`'s `return {...}`) and the `print_comparison` cols:

```python
        "gt_model_iou": tm.get("gt_model_iou"),
        "gt_model_onset_mae": tm.get("gt_model_onset_mae_min"),
```
and add `("gt_model_iou", 9), ("gt_model_onset_mae", 12)` to the `cols` list in `print_comparison`.

- [ ] **Step 4: Self-test the inspector with synthetic gt_* metrics**

```bash
/Users/zongfan/opt/miniconda3/envs/pytorch/bin/python - <<'PY'
import os, tempfile, subprocess, sys, joblib
d = os.path.join(tempfile.mkdtemp(), "run", "training", "predictions"); os.makedirs(d)
joblib.dump({"iteration": 0,
             "test_metrics": {"loss": 0.3, "f1_tso": 0.88,
                              "gt_model_onset_mae_min": 9.0, "gt_vanhees_onset_mae_min": 14.0,
                              "gt_model_iou": 0.82, "gt_vanhees_iou": 0.74,
                              "gt_model_f1": 0.86, "gt_vanhees_f1": 0.80,
                              "gt_model_balacc": 0.88, "gt_vanhees_balacc": 0.83,
                              "gt_n_nights_both_tso": 40, "gt_pred_has_tso_rate": 1.0,
                              "gt_gt_has_tso_rate": 1.0},
             "comprehensive_metrics": {"balanced_accuracy": 0.87, "f1_score_macro": 0.86, "accuracy": 0.84},
             "history": {"val_loss": [0.5, 0.4], "val_selection_score": [0.9, 0.7]}},
            os.path.join(d, "results_iter_0.joblib"))
r = subprocess.run([sys.executable, "test-tools/inspect_tso_results.py", os.path.dirname(os.path.dirname(d))],
                   capture_output=True, text=True)
print(r.stdout); assert "GT (inTSO)" in r.stdout and "vanHees" in r.stdout, r.stdout
print("INSPECTOR GT CHECK PASS")
PY
/Users/zongfan/opt/homebrew/bin/python3 -m py_compile test-tools/inspect_tso_results.py 2>/dev/null || python3 -m py_compile test-tools/inspect_tso_results.py
```
Expected: GT table rendered, `INSPECTOR GT CHECK PASS`.

- [ ] **Step 5: Commit**

```bash
git add experiments/domino/build_deep_tso_h5.sh test-tools/inspect_tso_results.py
git commit -m "feat: GT_COLUMN build flag + GT metrics in results inspector

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Domino verification

**Files:**
- No source edits unless a failure is found.

- [ ] **Step 1: Rebuild the H5 with GT, run local tests on Domino**

```bash
bash experiments/domino/deep_tso_setup.sh
python -m pytest tests/test_deep_tso_gt_eval.py tests/test_deep_tso_noisy_labels.py tests/test_deep_tso_validation.py -q
GT_COLUMN=inTSO bash experiments/domino/build_deep_tso_h5.sh
python3.11 -c "import h5py; f=h5py.File('/mnt/data/GENEActive-featurized/h5/deep_tso_20hz_sincos.h5'); print('Y_gt' in f, f.attrs.get('gt_column'))"
```
Expected: tests pass; `True inTSO`.

- [ ] **Step 2: Evaluate an existing checkpoint with `--test_only`**

```bash
python3.11 training/train_tso_patch_h5.py \
  --config experiments/configs/deep_tso_phase1_gce.yaml --num_gpu 0 --test_only
```
Expected: test runs, no errors; the printed/saved `test_metrics` now contain `gt_model_*` and `gt_vanhees_*` keys.

- [ ] **Step 3: Inspect the GT comparison**

```bash
python3.11 test-tools/inspect_tso_results.py /mnt/data/GENEActive-featurized/results/DL/deep_tso_phase1_gce_*
```
Expected: the "GT (inTSO) — model vs van Hees" table prints; read off whether the model's onset/offset MAE / IoU / F1-vs-GT beat van Hees's.

- [ ] **Step 4: Commit any summary**

```bash
git add -A && git commit -m "docs: Deep TSO GT-eval verification notes" || echo "nothing to commit"
```

---

## Self-Review
- **Spec coverage:** convert_h5 `Y_gt` (Task 2) ✓; loading/`pad_Y_gt` (Task 3) ✓; `interval_agreement` + per-minute + interval, three-way model/vanHees vs GT, raw-predictTSO captured before consensus (Tasks 1, 4) ✓; build `GT_COLUMN` + `--test_only` (Tasks 5–6) ✓; inspector surfacing (Task 5) ✓; reporting-only, presence-guarded, GT never in loss/selection ✓; metric semantics (MAE only both-present, IoU 0/NaN rules, signed duration error) encoded in `interval_agreement` + aggregation ✓.
- **Placeholder scan:** all steps contain concrete code/commands; no TBD/TODO.
- **Type consistency:** `add_padding_tso_patch_h5` returns 6 values everywhere it's consumed (run_model_tso_h5 + 3 dlrtc sites); `interval_agreement` keys (`onset_mae_min`, `offset_mae_min`, `iou`, `duration_err_h`, `pred_has_tso`, `gt_has_tso`) match the aggregation in Task 4 and the inspector keys in Task 5; metric keys (`gt_model_*`, `gt_vanhees_*`) consistent across Tasks 4–5.
