"""Bout-level scratch-classification threshold sweep.

Reproduces the eval-time threshold analysis in the v1.0 tech spec
(§5.3.2, Figure 10): for a single trained variant, vary τ over [0,1],
re-derive ``pr1 = (pr1_probs >= τ)``, and report the F1-maximizing
threshold along with ROC / PR / F1-vs-τ curves.  The tech spec found
τ*=0.62 on the LOSO-pooled curve and rounded to τ=0.6 for the final
productionized model — our number will differ across runs / variants;
that's the point of having the sweep here.

This is purely eval-side: the trained model and its CSVs are not
modified.  Run as a follow-up to a successful LOFO/LOSO training run.

Usage::

    python3.11 evaluation/threshold_sweep.py \
        --variant ablation-full-no_pretrain-bs32-fold4

    # On a baseline (classical CSVs also carry pr1_probs):
    python3.11 evaluation/threshold_sweep.py \
        --variant baseline-mahadevan2021-loso

The script writes ``threshold_sweep_<variant>.csv`` (metric per τ) and
``threshold_sweep_<variant>.png`` (3-panel ROC / PR / F1 vs τ matching
Figure 10's layout) under ``--out_dir``.  Once τ* is chosen, regenerate
the paper table at that threshold via::

    python3.11 evaluation/prediction_analysis.py \
        --mode paper-ablation --pr1_threshold 0.6
"""

import argparse
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics

from evaluation.prediction_analysis import get_predictions


def _segment_level_df(dfp, pr1_threshold):
    """Apply the prediction_analysis.build_variant_results pipeline at a
    fixed τ and return the segment-aggregated DataFrame.

    Mirrors prediction_analysis.py:225–233 (mask-pool, length-normalised
    gt3/pr3 in seconds, hurdle gate using the re-thresholded pr1).
    """
    out = dfp.copy()
    if pr1_threshold is not None and 'pr1_probs' in out.columns:
        out['pr1'] = (out['pr1_probs'] >= float(pr1_threshold)).astype(int)

    out['pr3_2'] = out.groupby('segment')['pr2'].transform('sum')
    out['length'] = out.groupby('segment')['pr2'].transform('count')
    out['gt3'] = (out['gt3'] * out['length']) / 20
    out['pr3'] = (out['pr3'] * out['length']) / 20
    out['pr3_2'] = out['pr3_2'] / 20
    out.loc[out.pr1 == 0, ['pr3', 'pr3_2']] = 0
    out.loc[out.pr3 < 0, ['pr3']] = 0

    seg = out.groupby('segment').max(numeric_only=True)
    return seg


def _row_at_threshold(dfp, tau):
    """Compute pooled segment-level paper metrics at threshold τ."""
    seg = _segment_level_df(dfp, tau)
    if seg.gt1.nunique() < 2:
        return None
    return {
        'tau': tau,
        'n_segments': int(len(seg)),
        'F1': metrics.f1_score(seg.gt1, seg.pr1, zero_division=0) * 100,
        'Precision': metrics.precision_score(seg.gt1, seg.pr1, zero_division=0) * 100,
        'Recall': metrics.recall_score(seg.gt1, seg.pr1, zero_division=0) * 100,
        'BalancedAccuracy': metrics.balanced_accuracy_score(seg.gt1, seg.pr1) * 100,
        'R2': metrics.r2_score(seg.gt3, seg.pr3),
        'MAE': metrics.mean_absolute_error(seg.gt3, seg.pr3),
    }


def sweep(dfp, thresholds):
    rows = [_row_at_threshold(dfp, t) for t in thresholds]
    return pd.DataFrame([r for r in rows if r is not None])


def _plot_three_panel(seg_probs_df, sweep_df, tau_star, out_png):
    """Figure-10-style 3-panel: ROC, PR, F1 vs τ."""
    fpr, tpr, _ = metrics.roc_curve(seg_probs_df.gt1, seg_probs_df.pr1_probs)
    auroc = metrics.auc(fpr, tpr)
    prec, rec, _ = metrics.precision_recall_curve(
        seg_probs_df.gt1, seg_probs_df.pr1_probs,
    )
    auprc = metrics.auc(rec, prec)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    ax.plot(fpr, tpr, label=f'ROC (AUC={auroc:.3f})')
    ax.plot([0, 1], [0, 1], '--', color='gray', label='Chance')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(rec, prec, color='orange', label=f'PR (AUPRC={auprc:.3f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(sweep_df.tau, sweep_df.F1, color='green', label='F1 vs τ')
    ax.axvline(tau_star, color='red', linestyle='--',
               label=f'τ*={tau_star:.2f}, F1={sweep_df.F1.max():.2f}')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('F1 (%)')
    ax.set_title('F1 across Thresholds')
    ax.legend(loc='lower center')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    plt.close(fig)


def run(args):
    folder_path = args.folder_path or (
        f"/mnt/data/GENEActive-featurized/results/DL/"
        f"{os.path.basename(args.train_data.rstrip('/').rstrip('/raw'))}/"
    )
    print(f"[sweep] variant={args.variant!r}")
    print(f"[sweep] folder_path={folder_path}")

    dfp = get_predictions(folder_path, args.variant, df_rest=None,
                          filter_tso=True)
    if len(dfp) == 0:
        print(f"[sweep] no predictions under {os.path.join(folder_path, args.variant)}")
        return None
    if 'pr1_probs' not in dfp.columns:
        print("[sweep] CSV is missing pr1_probs — cannot sweep")
        return None

    # Segment-level probabilities (one row per bout) drive the ROC / PR
    # curves; the sklearn calls expect a single (label, score) array.
    seg_probs_df = (
        dfp.groupby('segment')
           .agg(gt1=('gt1', 'max'), pr1_probs=('pr1_probs', 'max'))
           .reset_index()
    )
    seg_probs_df = seg_probs_df[seg_probs_df.gt1.notna()]

    taus = np.round(np.arange(args.tau_min, args.tau_max + 1e-9, args.tau_step), 4)
    sweep_df = sweep(dfp, taus)
    if sweep_df.empty:
        print("[sweep] empty sweep — every τ produced a degenerate split")
        return None

    tau_star = float(sweep_df.loc[sweep_df.F1.idxmax(), 'tau'])
    print(f"\n[sweep] τ* (F1-max) = {tau_star:.2f}")
    print(f"[sweep] at τ*: F1={sweep_df.F1.max():.2f}  "
          f"P={sweep_df.loc[sweep_df.F1.idxmax(), 'Precision']:.2f}  "
          f"R={sweep_df.loc[sweep_df.F1.idxmax(), 'Recall']:.2f}  "
          f"R²={sweep_df.loc[sweep_df.F1.idxmax(), 'R2']:.3f}")

    # Also report a few reference thresholds.
    for ref in [0.5, 0.6, 0.62]:
        row = sweep_df.iloc[(sweep_df.tau - ref).abs().idxmin()]
        print(f"[sweep] at τ={row.tau:.2f}: F1={row.F1:.2f}  "
              f"P={row.Precision:.2f}  R={row.Recall:.2f}  "
              f"R²={row.R2:.3f}")

    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, f"threshold_sweep_{args.variant}.csv")
    out_png = os.path.join(args.out_dir, f"threshold_sweep_{args.variant}.png")
    sweep_df.to_csv(out_csv, index=False)
    _plot_three_panel(seg_probs_df, sweep_df, tau_star, out_png)
    print(f"\n[sweep] wrote {out_csv}")
    print(f"[sweep] wrote {out_png}")
    return sweep_df


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant', type=str, required=True,
                   help='training.output name of the run to sweep, e.g. '
                        '"ablation-full-no_pretrain-bs32-fold4" or '
                        '"baseline-mahadevan2021-loso".')
    p.add_argument('--train_data', type=str,
                   default='/mnt/data/Nocturnal-scratch/'
                           'geneactive_20hz_3s_b1s_production_train_van_new_enh_lth-rth/raw/',
                   help='Raw-data folder used by the training run '
                        '(matches data.input_folder in the YAML).')
    p.add_argument('--folder_path', type=str, default=None,
                   help='Override the derived results root '
                        '/mnt/data/GENEActive-featurized/results/DL/<dataset>/. '
                        'Useful for non-default deployments.')
    p.add_argument('--out_dir', type=str,
                   default='experiments/logs/threshold_sweep/',
                   help='Where the CSV + PNG outputs are written.')
    p.add_argument('--tau_min', type=float, default=0.01)
    p.add_argument('--tau_max', type=float, default=0.99)
    p.add_argument('--tau_step', type=float, default=0.01)
    args = p.parse_args()
    return run(args)


if __name__ == '__main__':
    main()
