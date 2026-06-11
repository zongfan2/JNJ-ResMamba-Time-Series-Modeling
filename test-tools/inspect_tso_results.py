#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inspect Deep TSO training results saved as results_iter_*.joblib.

Each file (written by training/train_tso_patch_h5.py) is a dict:
    {iteration, test_metrics, comprehensive_metrics, history}

Prints, per file: test metrics, comprehensive metrics, a training-history
summary (best epoch + key curves), and the label-free TSO interval / cross-night
consistency metrics. With multiple files it also prints a comparison table —
handy for the CE / GCE / GCE+SupCon ablation.

Usage:
    # one file
    python3.11 test-tools/inspect_tso_results.py /path/to/results_iter_0.joblib
    # a run folder, a glob, or the whole results root (default)
    python3.11 test-tools/inspect_tso_results.py /mnt/data/GENEActive-featurized/results/DL
"""

import argparse
import glob
import os
import sys

import joblib
import numpy as np

DEFAULT_ROOT = "/mnt/data/GENEActive-featurized/results/DL"

# label-free TSO metrics recorded in test_metrics (evaluation/tso_validation.py)
TSO_KEYS = [
    "mean_pred_tso_duration_hours",
    "mean_pred_tso_segment_count",
    "subjects_with_multiple_nights",
    "mean_onset_std_minutes",
    "mean_offset_std_minutes",
    "mean_duration_std_hours",
]


def find_files(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            out += glob.glob(os.path.join(p, "**", "results_iter_*.joblib"), recursive=True)
        elif any(c in p for c in "*?["):
            out += glob.glob(p, recursive=True)
        else:
            out.append(p)
    return sorted(set(out))


def run_name(path):
    # .../<RUN>/training/predictions/results_iter_N.joblib  -> <RUN>
    d = os.path.dirname(os.path.dirname(os.path.dirname(path)))
    return os.path.basename(d) or path


def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, (float, np.floating)):
        return "nan" if np.isnan(v) else f"{float(v):.4f}"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return str(v)


def print_metrics(title, d):
    print(f"  {title}:")
    if not isinstance(d, dict) or not d:
        print(f"    (none)")
        return
    for k in sorted(d):
        v = d[k]
        if isinstance(v, (list, np.ndarray)):
            print(f"    {k:32s} array{np.asarray(v).shape}")
        elif isinstance(v, dict):
            print(f"    {k:32s} {{{len(v)} keys}}")
        else:
            print(f"    {k:32s} {fmt(v)}")


def history_summary(hist):
    print("  history:")
    if not isinstance(hist, dict) or not hist:
        print("    (none)")
        return
    sel_key = "val_selection_score" if hist.get("val_selection_score") else "val_loss"
    sel = hist.get(sel_key) or []
    n = len(sel)
    print(f"    epochs: {n}  (best chosen by min {sel_key})")
    if n:
        finite = [(i, s) for i, s in enumerate(sel) if s is not None and np.isfinite(s)]
        best = min(finite, key=lambda t: t[1])[0] if finite else int(np.argmin(sel))
        print(f"    best epoch: {best + 1}")
        for k in ("val_loss", "val_f1_tso", "val_f1_avg", "val_accuracy",
                  "val_selection_score", "train_loss", "train_f1_tso"):
            seq = hist.get(k)
            if seq and len(seq) > best:
                print(f"    {k:24s} best={fmt(seq[best])}  last={fmt(seq[-1])}")


def inspect(path):
    print("=" * 78)
    print(run_name(path))
    print(path)
    print("=" * 78)
    data = joblib.load(path)
    if not isinstance(data, dict):
        print(f"  top-level object is {type(data).__name__}, not the expected dict.")
        return None

    print(f"  iteration: {data.get('iteration')}")
    tm = data.get("test_metrics", {}) or {}
    cm = data.get("comprehensive_metrics", {}) or {}
    print_metrics("test_metrics", tm)
    print_metrics("comprehensive_metrics", cm)

    present_tso = {k: tm[k] for k in TSO_KEYS if k in tm}
    if present_tso:
        print("  label-free TSO metrics:")
        for k in TSO_KEYS:
            if k in present_tso:
                print(f"    {k:32s} {fmt(present_tso[k])}")

    history_summary(data.get("history", {}))
    print()
    return {
        "run": run_name(path),
        "f1_tso": tm.get("f1_tso"),
        "f1_macro": cm.get("f1_score_macro"),
        "bal_acc": cm.get("balanced_accuracy"),
        "acc": cm.get("accuracy"),
        "test_loss": tm.get("loss"),
        "dur_h": tm.get("mean_pred_tso_duration_hours"),
        "segs": tm.get("mean_pred_tso_segment_count"),
    }


def print_comparison(rows):
    cols = [("run", 34), ("f1_tso", 9), ("f1_macro", 9), ("bal_acc", 9),
            ("acc", 9), ("test_loss", 10), ("dur_h", 8), ("segs", 7)]
    print("=" * 78)
    print("COMPARISON")
    print("=" * 78)
    print("".join(name.ljust(w) for name, w in cols))
    print("-" * sum(w for _, w in cols))
    for r in rows:
        line = ""
        for name, w in cols:
            v = r.get(name)
            s = (str(v)[: w - 1]) if name == "run" else fmt(v)
            line += s.ljust(w)
        print(line)


def main():
    ap = argparse.ArgumentParser(description="Inspect Deep TSO results_iter_*.joblib")
    ap.add_argument("paths", nargs="*", help="joblib files, globs, or run/results dirs")
    args = ap.parse_args()
    paths = args.paths or [DEFAULT_ROOT]

    files = find_files(paths)
    if not files:
        print(f"No results_iter_*.joblib found under: {paths}")
        return 1

    rows = []
    for f in files:
        try:
            row = inspect(f)
            if row:
                rows.append(row)
        except Exception as e:
            print(f"  ERROR reading {f}: {type(e).__name__}: {e}\n")

    if len(rows) > 1:
        print_comparison(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
