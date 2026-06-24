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

LOFO/LOSO cross-validation writes one results_iter_*.joblib PER held-out fold
into the SAME run folder. For those runs the headline number is the FOLD
AGGREGATE — mean +/- std across folds (e.g. 4-fold LOFO). The aggregate block
also flags any arm that did not complete all expected folds (a crashed fold
would otherwise be silently averaged over fewer than 4).

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
        is_glob = any(c in p for c in "*?[")
        # Expand globs first; a glob like ".../deep_tso_phase1_*" matches the RUN
        # DIRECTORIES, so each match still has to be recursed into.
        candidates = glob.glob(p, recursive=True) if is_glob else [p]
        for c in candidates:
            if os.path.isdir(c):
                out += glob.glob(os.path.join(c, "**", "results_iter_*.joblib"), recursive=True)
            elif os.path.isfile(c):
                # keep literal file args as-is; from a glob keep only .joblib files
                if c.endswith(".joblib") or not is_glob:
                    out.append(c)
    return sorted(set(out))


def run_name(path):
    # .../<RUN>/training/predictions/results_iter_N.joblib  -> <RUN>
    d = os.path.dirname(os.path.dirname(os.path.dirname(path)))
    return os.path.basename(d) or path


def save_loss_curve(path, data):
    """Write a train/val loss curve (with best-epoch marker) into the run's
    learning_plots/ folder, from the saved history. Returns the PNG path or None.

    Post-hoc companion to training/train_tso_patch_h5.py's plot_tso_learning_curves
    (which only runs during training): regenerates the loss curve from any saved
    results_iter_*.joblib, including --test_only runs.
    """
    history = (data.get("history") or {}) if isinstance(data, dict) else {}
    tl = history.get("train_loss") or []
    vl = history.get("val_loss") or []
    if not tl and not vl:
        return None
    import matplotlib
    matplotlib.use("Agg")  # headless (Domino)
    import matplotlib.pyplot as plt

    # .../<RUN>/training/predictions/results_iter_N.joblib -> .../<RUN>/training/learning_plots/
    out_dir = os.path.join(os.path.dirname(os.path.dirname(path)), "learning_plots")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"loss_curve_iter_{data.get('iteration', 0)}.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if tl:
        ax.plot(range(1, len(tl) + 1), tl, label="train", linewidth=2, color="#1f77b4")
    if vl:
        ax.plot(range(1, len(vl) + 1), vl, label="val", linewidth=2, color="#ff7f0e")
    sel = history.get("val_selection_score") or vl
    finite = [(i, s) for i, s in enumerate(sel) if s is not None and np.isfinite(s)]
    if finite:
        best = min(finite, key=lambda t: t[1])[0]
        ax.axvline(best + 1, color="gray", linestyle="--", alpha=0.6, label=f"best epoch {best + 1}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(f"{run_name(path)} — loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


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


def inspect(path, save_plots=False):
    data = joblib.load(path)
    # LOFO/LOSO write one joblib per held-out fold into the SAME run folder, so
    # tag the display label with split_tag to keep folds distinguishable.
    tag = data.get("split_tag") if isinstance(data, dict) else None
    label = run_name(path) + (f" [{data.get('testing', 'cv')}:{tag}]" if tag else "")
    print("=" * 78)
    print(label)
    print(path)
    print("=" * 78)
    if not isinstance(data, dict):
        print(f"  top-level object is {type(data).__name__}, not the expected dict.")
        return None

    print(f"  iteration: {data.get('iteration')}" + (f"  held-out: {tag}" if tag else ""))
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

    history_summary(data.get("history", {}))
    if save_plots:
        png = save_loss_curve(path, data)
        print(f"  loss curve -> {png}" if png else "  loss curve -> (no train/val loss in history)")
    print()
    return {
        "run": label,
        "run_dir": run_name(path),   # untagged folder, for grouping folds
        "fold": tag,                 # held-out fold/subject tag (None for non-CV)
        "testing": data.get("testing"),
        "num_folds": data.get("num_folds"),   # expected fold count (None on older runs)
        "f1_tso": tm.get("f1_tso"),
        "f1_macro": cm.get("f1_score_macro"),
        "bal_acc": cm.get("balanced_accuracy"),
        "acc": cm.get("accuracy"),
        "test_loss": tm.get("loss"),
        "dur_h": tm.get("mean_pred_tso_duration_hours"),
        "segs": tm.get("mean_pred_tso_segment_count"),
        "gt_model_iou": tm.get("gt_model_iou"),
        "gt_model_f1": tm.get("gt_model_f1"),
        "gt_model_onset_mae": tm.get("gt_model_onset_mae_min"),
        "gt_model_offset_mae": tm.get("gt_model_offset_mae_min"),
    }


def print_comparison(rows):
    # run_dir + fold as separate columns so per-fold LOFO/LOSO rows stay
    # distinguishable (the long run-id otherwise truncated the fold tag away).
    # (key, short header, width) — short headers keep the row width scannable.
    cols = [("run_dir", "run_dir", 30), ("fold", "fold", 8),
            ("f1_tso", "f1_tso", 8), ("f1_macro", "f1_mac", 8),
            ("bal_acc", "bal_acc", 9), ("acc", "acc", 8),
            ("gt_model_iou", "gtIoU", 8), ("gt_model_f1", "gtF1", 7),
            ("gt_model_onset_mae", "gtOnsMAE", 10)]
    print("=" * 78)
    print("COMPARISON (per file; for LOFO/LOSO see FOLD AGGREGATES below)")
    print("=" * 78)
    print("".join(hdr[: w - 1].ljust(w) for _, hdr, w in cols))
    print("-" * sum(w for _, _, w in cols))
    for r in rows:
        line = ""
        for key, _, w in cols:
            s = fmt(r.get(key))
            if len(s) > w - 1:        # keep the front (arm name / fold tag live there)
                s = s[: w - 1]
            line += s.ljust(w)
        print(line)


CV_MODES = ("LOFO", "LOSO")


def print_fold_aggregates(rows):
    """Mean +/- std across folds, grouped by run folder.

    For LOFO/LOSO each held-out fold is a separate results_iter_*.joblib in the
    SAME run folder; the headline number is the fold-averaged metric. (Also
    covers random runs repeated over --training_iterations.)

    For LOFO/LOSO arms the block ALSO runs a completeness check: an arm that did
    not finish every expected fold is flagged "INCOMPLETE", so a crashed fold is
    not silently averaged over fewer than (e.g.) 4. Expected folds come from each
    file's recorded ``num_folds`` when present, else from the union of fold tags
    seen across all CV arms (folds are shared across arms in a LOFO sweep).
    """
    groups = {}
    order = []
    for r in rows:
        k = r.get("run_dir") or r.get("run")
        if k not in groups:
            groups[k] = []
            order.append(k)
        groups[k].append(r)

    # full fold set seen across all CV arms (LOFO uses the same folds per arm)
    all_cv_folds = sorted({r.get("fold") for r in rows
                           if r.get("testing") in CV_MODES and r.get("fold")})

    def is_cv(rs):
        return any(r.get("testing") in CV_MODES for r in rs)

    # Show every CV arm (so incomplete ones get flagged even at n==1); show
    # non-CV folders only when there is more than one repeat to average.
    shown = [(k, groups[k]) for k in order if is_cv(groups[k]) or len(groups[k]) > 1]
    if not shown:
        return

    metrics = [("f1_tso", "f1_tso"), ("f1_macro", "f1_macro"),
               ("bal_acc", "bal_acc"), ("acc", "accuracy"),
               ("gt_model_iou", "IoU vs GT"), ("gt_model_f1", "F1 vs GT"),
               ("gt_model_onset_mae", "onset MAE min"),
               ("gt_model_offset_mae", "offset MAE min")]
    print("=" * 78)
    print("FOLD AGGREGATES — mean +/- std across folds (per run folder)")
    print("=" * 78)
    for run_dir, rs in shown:
        present = sorted({r.get("fold") for r in rs if r.get("fold")})
        print(f"{run_dir}  (n={len(rs)}" + (f": {', '.join(present)}" if present else "") + ")")
        if is_cv(rs):
            nfs = [int(r["num_folds"]) for r in rs if r.get("num_folds")]
            exp_n = nfs[0] if nfs else (len(all_cv_folds) or len(rs))
            missing = [f for f in all_cv_folds if f not in present]
            if len(present) < exp_n or missing:
                msg = f"    !! INCOMPLETE: {len(present)}/{exp_n} folds"
                if missing:
                    msg += f" (missing {', '.join(missing)})"
                print(msg)
        for key, name in metrics:
            vals = [r.get(key) for r in rs]
            vals = [float(v) for v in vals
                    if isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v))]
            if vals:
                a = np.array(vals)
                print(f"    {name:16s} {a.mean():.4f} +/- {a.std():.4f}  (n={len(vals)})")
        print()


# (metric_key, display, higher_is_better) for the paired significance table
SIG_METRICS = [("gt_model_iou", "IoU vs GT", True), ("gt_model_f1", "F1 vs GT", True),
               ("gt_model_onset_mae", "onset MAE", False), ("f1_tso", "f1_tso", True)]


def _paired_p(deltas):
    """Two-sided paired-difference p-value. Uses scipy paired t-test if available,
    else a manual t-stat with a normal-approx p (flagged ~). Returns (p, approx)."""
    a = np.asarray(deltas, dtype=float)
    n = len(a)
    if n < 2 or a.std(ddof=1) == 0:
        return (float("nan"), False)
    t = a.mean() / (a.std(ddof=1) / np.sqrt(n))
    try:
        from scipy import stats
        return (float(2 * stats.t.sf(abs(t), df=n - 1)), False)
    except Exception:
        # normal approximation (rough at small n) — labelled approximate
        from math import erf
        return (float(2 * (1 - 0.5 * (1 + erf(abs(t) / np.sqrt(2))))), True)


def print_pairwise_significance(rows, baseline_sub):
    """Per-fold paired comparison of each CV arm vs the baseline arm.

    Folds are matched by tag (so the comparison is paired within subject-fold). For
    each metric prints the mean delta (arm - baseline), how many folds favor the arm,
    and a paired-difference p-value. With only ~4 LOFO folds a p-value is weak — the
    per-fold win count and mean delta are the honest headline; p is a guide, flagged
    ~ when scipy is unavailable (normal approx).
    """
    by_arm = {}
    for r in rows:
        if r.get("testing") not in CV_MODES:
            continue
        by_arm.setdefault(r.get("run_dir"), {})[r.get("fold")] = r
    if len(by_arm) < 2:
        return
    base_key = next((k for k in by_arm if baseline_sub in str(k)), None)
    if base_key is None:
        print("=" * 78)
        print(f"PAIRWISE SIGNIFICANCE — no baseline arm matched '{baseline_sub}'; skipped.")
        print("=" * 78 + "\n")
        return
    base = by_arm[base_key]
    print("=" * 78)
    print(f"PAIRWISE SIGNIFICANCE vs baseline '{base_key}' (paired by fold; n folds small)")
    print("=" * 78)
    for arm, folds in by_arm.items():
        if arm == base_key:
            continue
        shared = sorted(set(folds) & set(base))
        print(f"{arm}  (paired on {len(shared)} folds: {', '.join(shared)})")
        for key, name, higher in SIG_METRICS:
            deltas = []
            for f in shared:
                av, bv = folds[f].get(key), base[f].get(key)
                if isinstance(av, (int, float)) and isinstance(bv, (int, float)) \
                        and not (np.isnan(av) or np.isnan(bv)):
                    deltas.append(av - bv)
            if len(deltas) < 2:
                continue
            d = np.array(deltas)
            wins = int(np.sum(d > 0 if higher else d < 0))
            p, approx = _paired_p(d)
            better = "arm better" if (d.mean() > 0) == higher else "arm worse"
            ptxt = (f"p{'~' if approx else ''}={p:.3f}" if np.isfinite(p) else "p=n/a")
            print(f"    {name:12s} d={d.mean():+.4f}  {wins}/{len(d)} folds favor arm  "
                  f"{ptxt}  ({better})")
        print()


def main():
    ap = argparse.ArgumentParser(description="Inspect Deep TSO results_iter_*.joblib")
    ap.add_argument("paths", nargs="*", help="joblib files, globs, or run/results dirs")
    ap.add_argument("--plots", action="store_true",
                    help="Write a train/val loss curve PNG into each run's learning_plots/ folder.")
    ap.add_argument("--baseline", default="baseline",
                    help="Substring identifying the baseline run folder for the paired "
                         "significance table (default 'baseline').")
    args = ap.parse_args()
    paths = args.paths or [DEFAULT_ROOT]

    files = find_files(paths)
    if not files:
        print(f"No results_iter_*.joblib found under: {paths}")
        return 1

    rows = []
    for f in files:
        try:
            row = inspect(f, save_plots=args.plots)
            if row:
                rows.append(row)
        except Exception as e:
            print(f"  ERROR reading {f}: {type(e).__name__}: {e}\n")

    if len(rows) > 1:
        print_comparison(rows)
    if rows:
        print_fold_aggregates(rows)   # self-gates; flags incomplete CV sweeps
        print_pairwise_significance(rows, args.baseline)
    return 0


if __name__ == "__main__":
    sys.exit(main())
