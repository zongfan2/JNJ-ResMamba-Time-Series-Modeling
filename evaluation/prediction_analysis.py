"""
Prediction Analysis Script

This script loads trained model predictions and evaluates performance metrics
for scratch detection models on wearable sensor data.
"""
# import subprocess

# shell_script = '''
# sudo python3.11 -m pip install plotly seaborn shap torcheval openpyxl pingouin ruptures adjustText matplotlib
# '''
# result = subprocess.run(shell_script, shell=True, capture_output=True, text=True)

import os
import sys

# Allow running as `python evaluation/prediction_analysis.py` from the repo
# root: put the project root on sys.path so `data.loading` resolves.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from scipy.stats import pearsonr

from data.loading import load_data
from utils.common import create_folder


def get_predictions(folder_path, model, df_rest=None, filter_tso=False):
    """
    Load model predictions from CSV files and optionally merge with rest data.

    Args:
        folder_path: Base directory containing model prediction folders
        model: Name of the model subdirectory
        df_rest: Optional rest-frames DataFrame to concatenate (legacy
            per-patient analysis path). Paper-table modes pass ``None``
            and skip the concat — the expensive ``load_data`` call over
            the raw 410 files is only needed for full-night aggregation.
        filter_tso: If True, filter predictions to only include TSO (Time Segment of Interest)

    Returns:
        pd.DataFrame: Combined predictions with PID extracted from segment names
    """
    dfp_list = []
    model_path = os.path.join(folder_path, model)
    files = sorted(
        os.path.join(dp, f)
        for dp, dn, filenames in os.walk(model_path)
        for f in filenames
        if f.endswith('.csv')
    )

    for file in files:
        if "Y_test_Y_pred" in file:
            file_name = os.path.basename(file)
            subject = file_name.replace(".csv", "").split("_")[-1]
            df_temp = pd.read_csv(file)

            if filter_tso:
                df_temp = df_temp[df_temp.inTSO == True]

            if len(df_temp) > 0:
                df_temp["subject"] = subject
                dfp_list.append(df_temp)

    if len(dfp_list) > 0:
        dfp = pd.concat(dfp_list, ignore_index=True)
        if df_rest is not None and len(df_rest) > 0:
            dfp = pd.concat(
                [dfp, df_rest[df_rest.inTSO == True]], ignore_index=True,
            )
        dfp['PID'] = dfp.reset_index().segment.str.split('_', expand=True).iloc[:, 0]
    else:
        dfp = pd.DataFrame()

    return dfp


def get_metrics(df, suffix):
    """
    Calculate evaluation metrics for scratch detection predictions.

    Computes F1, recall, precision, ROC-AUC, balanced accuracy, and R2 scores.
    Also includes random baseline metrics for comparison.

    Args:
        df: DataFrame with ground truth (gt*) and prediction (pr*) columns
        suffix: String suffix to append to metric names

    Returns:
        pd.Series: Dictionary of metric names and values
    """
    segments_df = df.groupby('segment').max(1)
    segments_df['pr1_rand'] = np.random.choice([True, False], size=len(segments_df))
    segments_df['pr3_rand'] = np.random.rand(len(segments_df))

    return pd.Series({
        f'F1_1_{suffix}': metrics.f1_score(segments_df.gt1, segments_df.pr1, zero_division=0) * 100,
        f'F1-2_1_{suffix}': metrics.f1_score(segments_df.gt1, segments_df.pr1_2, zero_division=0) * 100,
        f'Recall_1_{suffix}': metrics.recall_score(segments_df.gt1, segments_df.pr1, zero_division=0) * 100,
        f'Precision_1_{suffix}': metrics.precision_score(segments_df.gt1, segments_df.pr1, zero_division=0) * 100,
        f'ROC_AUC_1_{suffix}': metrics.roc_auc_score(segments_df.gt1, segments_df.pr1) * 100,
        f'Balanced_Accuracy_1_{suffix}': metrics.balanced_accuracy_score(segments_df.gt1, segments_df.pr1) * 100,
        f'F1_2_{suffix}': metrics.f1_score(df.gt2, df.pr2, zero_division=0) * 100,
        f'R2_{suffix}': metrics.r2_score(segments_df.gt3, segments_df.pr3),
        f'R2_2{suffix}': metrics.r2_score(segments_df.gt3, segments_df.pr3_2),
        f'MAE_{suffix}': metrics.mean_absolute_error(segments_df.gt3, segments_df.pr3),
        f'F1_rand_{suffix}': metrics.f1_score(segments_df.gt1, segments_df.pr1_rand, zero_division=0) * 100,
        f'Recall_rand_{suffix}': metrics.recall_score(segments_df.gt1, segments_df.pr1_rand, zero_division=0) * 100,
        f'Precision_rand_{suffix}': metrics.precision_score(segments_df.gt1, segments_df.pr1_rand, zero_division=0) * 100,
        f'ROC_AUC_rand_{suffix}': metrics.roc_auc_score(segments_df.gt1, segments_df.pr1_rand) * 100,
        f'R2_rand_{suffix}': metrics.r2_score(segments_df.gt3, segments_df.pr3_rand)
    })


# ============================================================================
# Paper-table helpers
# ----------------------------------------------------------------------------
# These helpers produce the exact metrics reported in
# papers/deep_scratch/deepscratch.tex Tables 2 (main results) and 3
# (ablation results).
#
# Reported metrics per variant, aggregated as mean ± std over LOFO folds:
#   F1 [%], Precision [%], Recall [%], AUROC [%], R^2, MAE [s]
#
# Each LOFO run produces one `Y_test_Y_pred_<FOLD>.csv` per fold under
#   <results_folder>/<output_name>/training/predictions/
# where <output_name> is the `training.output` value from the variant's
# YAML. `get_predictions` tags each row with its fold via the `subject`
# column; we group on it to compute per-fold metrics.
# ============================================================================


PAPER_METRIC_COLS = ['F1', 'Precision', 'Recall', 'AUROC', 'R2', 'MAE']


def compute_paper_metrics(segments_df):
    """Compute the six paper-table metrics on a segment-level DataFrame."""
    return {
        'F1':        metrics.f1_score(segments_df.gt1, segments_df.pr1, zero_division=0) * 100,
        'Precision': metrics.precision_score(segments_df.gt1, segments_df.pr1, zero_division=0) * 100,
        'Recall':    metrics.recall_score(segments_df.gt1, segments_df.pr1, zero_division=0) * 100,
        'AUROC':     metrics.roc_auc_score(segments_df.gt1, segments_df.pr1) * 100,
        'R2':        metrics.r2_score(segments_df.gt3, segments_df.pr3),
        'MAE':       metrics.mean_absolute_error(segments_df.gt3, segments_df.pr3),
    }


def compute_per_fold_metrics(dfp):
    """Compute paper metrics per held-out fold.

    Rest-frame rows (added by ``get_predictions`` with no ``subject`` tag)
    are excluded so the metric sees only actual test segments.
    """
    dfp = dfp[dfp['subject'].notna()]
    rows = []
    for fold, fdf in dfp.groupby('subject'):
        seg = fdf.groupby('segment').max(1)
        if len(seg) < 2 or seg.gt1.nunique() < 2:
            print(f"  [paper] fold={fold}: skipped "
                  f"(n_segments={len(seg)}, gt_classes={seg.gt1.nunique()})")
            continue
        row = {'fold': fold, 'n_segments': len(seg)}
        row.update(compute_paper_metrics(seg))
        rows.append(row)
    return pd.DataFrame(rows)


def build_variant_results(variant_label, model_name, folder_path, df_rest):
    """Load a variant's LOFO predictions and return one summary row.

    Returns a dict with mean and std of each paper metric, plus the raw
    per-fold DataFrame for downstream inspection.
    """
    dfp = get_predictions(folder_path, model_name, df_rest, filter_tso=True)
    if len(dfp) == 0:
        print(f"  [paper] {variant_label!r}: no predictions found under "
              f"{os.path.join(folder_path, model_name)}")
        return None, None

    # Mirror the segment-level derived columns from main().
    dfp.loc[:, 'pr3_2'] = dfp.groupby('segment')['pr2'].transform('sum')
    dfp['pr1_2'] = 0
    dfp.loc[dfp['pr3_2'] > 20, 'pr1_2'] = 1
    dfp.loc[:, 'length'] = dfp.groupby('segment')['pr2'].transform('count')
    dfp['gt3'] = (dfp['gt3'] * dfp['length']) / 20
    dfp['pr3'] = (dfp['pr3'] * dfp['length']) / 20
    dfp.loc[dfp.pr1 == 0, ['pr3', 'pr3_2']] = 0
    dfp.loc[dfp.pr3 < 0, ['pr3']] = 0

    per_fold = compute_per_fold_metrics(dfp)
    if per_fold.empty:
        print(f"  [paper] {variant_label!r}: no usable folds")
        return None, None

    # --- Per-fold mean ± std (shows variability across folds) ---
    summary = {'variant': variant_label, 'n_folds': len(per_fold)}
    for col in PAPER_METRIC_COLS:
        summary[f'{col}_mean'] = per_fold[col].mean()
        summary[f'{col}_std'] = per_fold[col].std(ddof=1) if len(per_fold) > 1 else 0.0

    # --- Overall / pooled metrics (all folds concatenated, single score) ---
    dfp_valid = dfp[dfp['subject'].notna()]
    seg_all = dfp_valid.groupby('segment').max(numeric_only=True)
    if len(seg_all) >= 2 and seg_all.gt1.nunique() >= 2:
        pooled = compute_paper_metrics(seg_all)
        for col in PAPER_METRIC_COLS:
            summary[f'{col}_pooled'] = pooled[col]

    return summary, per_fold


def _fmt(mean, std, digits=2):
    """Format a mean ± std cell with the paper's precision."""
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def to_latex_ablation(df, out_path, full_label='Full'):
    """Emit a LaTeX snippet matching papers/deep_scratch tab:ablation_results.

    Columns: F1, ΔF1, Precision, Recall, R² — covering both classification
    and regression tasks in a compact single-column-friendly format.
    """
    full_row = df[df['variant'].str.lower().str.contains(full_label.lower())]
    full_f1 = full_row['F1_mean'].iloc[0] if not full_row.empty else df['F1_mean'].iloc[0]

    lines = [
        r"% Auto-generated by evaluation/prediction_analysis.py",
        r"\begin{tabular}{lccccc}",
        r"    \toprule",
        r"    \multirow{2}{*}{Variant} & \multicolumn{4}{c}{Classification} & Regression \\",
        r"    \cmidrule(lr){2-5} \cmidrule(lr){6-6}",
        r"    & F1 & $\Delta$F1 & Precision & Recall & R\textsuperscript{2} \\",
        r"    \midrule",
    ]
    for _, row in df.iterrows():
        delta = row['F1_mean'] - full_f1
        delta_str = "---" if abs(delta) < 1e-9 else f"{delta:+.2f}"
        f1_str = _fmt(row['F1_mean'], row['F1_std'])
        prec_str = _fmt(row['Precision_mean'], row['Precision_std'])
        rec_str = _fmt(row['Recall_mean'], row['Recall_std'])
        r2_str = _fmt(row['R2_mean'], row['R2_std'], digits=3)
        lines.append(
            f"    {row['variant']} & {f1_str} & {delta_str} "
            f"& {prec_str} & {rec_str} & {r2_str} \\\\"
        )
    lines += [r"    \bottomrule", r"\end{tabular}"]
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  [paper] ablation LaTeX written to {out_path}")


def to_latex_pretrain_ablation(df, out_path, baseline_label='w/o pretrain'):
    """Emit a LaTeX snippet for the pretraining ablation table.

    Same 5-column layout as ``to_latex_ablation`` (F1, ΔF1, Precision,
    Recall, R²) but with ΔF1 computed against the no-pretrain baseline.
    A ``\\midrule`` separates fine-tuned rows from frozen-encoder rows.
    """
    baseline_row = df[df['variant'].str.lower().str.contains(baseline_label.lower())]
    baseline_f1 = (baseline_row['F1_mean'].iloc[0]
                   if not baseline_row.empty else df['F1_mean'].iloc[0])

    lines = [
        r"% Auto-generated by evaluation/prediction_analysis.py",
        r"\begin{tabular}{lccccc}",
        r"    \toprule",
        r"    \multirow{2}{*}{Variant} & \multicolumn{4}{c}{Classification} & Regression \\",
        r"    \cmidrule(lr){2-5} \cmidrule(lr){6-6}",
        r"    & F1 & $\Delta$F1 & Precision & Recall & R\textsuperscript{2} \\",
        r"    \midrule",
    ]
    freeze_started = False
    for _, row in df.iterrows():
        # Insert midrule before first "Freeze" row
        if not freeze_started and 'freeze' in row['variant'].lower():
            freeze_started = True
            lines.append(r"    \midrule")

        delta = row['F1_mean'] - baseline_f1
        delta_str = "---" if abs(delta) < 1e-9 else f"{delta:+.2f}"
        f1_str = _fmt(row['F1_mean'], row['F1_std'])
        prec_str = _fmt(row['Precision_mean'], row['Precision_std'])
        rec_str = _fmt(row['Recall_mean'], row['Recall_std'])
        r2_str = _fmt(row['R2_mean'], row['R2_std'], digits=3)
        lines.append(
            f"    {row['variant']} & {f1_str} & {delta_str} "
            f"& {prec_str} & {rec_str} & {r2_str} \\\\"
        )
    lines += [r"    \bottomrule", r"\end{tabular}"]
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  [paper] pretrain-ablation LaTeX written to {out_path}")


def to_latex_main_results(df, out_path):
    """Emit a LaTeX snippet matching papers/deep_scratch tab:main_results."""
    lines = [
        r"% Auto-generated by evaluation/prediction_analysis.py",
        r"\begin{tabular}{lcccccc}",
        r"    \toprule",
        r"    \multirow{2}{*}{Model} & \multicolumn{4}{c}{Classification} & \multicolumn{2}{c}{Regression} \\",
        r"    \cmidrule(lr){2-5} \cmidrule(lr){6-7}",
        r"    & F1 & Precision & Recall & AUROC & R\textsuperscript{2} & MAE (s) \\",
        r"    \midrule",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"    {row['variant']} "
            f"& {_fmt(row['F1_mean'], row['F1_std'])} "
            f"& {_fmt(row['Precision_mean'], row['Precision_std'])} "
            f"& {_fmt(row['Recall_mean'], row['Recall_std'])} "
            f"& {_fmt(row['AUROC_mean'], row['AUROC_std'])} "
            f"& {_fmt(row['R2_mean'], row['R2_std'], digits=3)} "
            f"& {_fmt(row['MAE_mean'], row['MAE_std'])} \\\\"
        )
    lines += [r"    \bottomrule", r"\end{tabular}"]
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  [paper] main-results LaTeX written to {out_path}")


def generate_paper_tables(variants, folder_path, df_rest, out_dir,
                          full_label='Full', table_kind='ablation'):
    """Build a paper-ready table from {variant_label: output_name} dict.

    Args:
        variants: ordered dict mapping the row label used in the paper
            (e.g. "Full", "w/o Mask Head") to the ``training.output`` name
            for that run (e.g. "ablation-full-no_pretrain-bs32-fold4").
        folder_path: absolute path of the dataset's DL results root, e.g.
            ``/mnt/data/GENEActive-featurized/results/DL/<dataset_name>/``.
        df_rest: DataFrame loaded via ``load_dataset`` for the same dataset.
        out_dir: directory where CSV + LaTeX snippets will be written.
        full_label: which row to compute ΔF1 from (ablation only).
        table_kind: 'ablation', 'pretrain', or 'main' — picks the LaTeX
            template.  For 'pretrain', ``full_label`` is used as the
            baseline label (default: the no-pretrain row).

    Returns:
        (summary_df, per_fold_dict) — summary_df has one row per variant
        with mean+std of each paper metric; per_fold_dict maps variant
        label → per-fold DataFrame for downstream analysis.
    """
    os.makedirs(out_dir, exist_ok=True)
    summary_rows = []
    per_fold_dict = {}
    for label, model_name in variants.items():
        print(f"\n[paper] variant={label!r}  output={model_name!r}")
        summary, per_fold = build_variant_results(
            label, model_name, folder_path, df_rest,
        )
        if summary is None:
            continue
        summary_rows.append(summary)
        per_fold_dict[label] = per_fold
        # Show per-fold mean vs pooled for quick comparison
        f1_mean = summary.get('F1_mean', 0)
        f1_pooled = summary.get('F1_pooled', 0)
        print(f"  F1: mean-of-folds={f1_mean:.2f}%  pooled={f1_pooled:.2f}%")

    if not summary_rows:
        print("[paper] no variants produced metrics — nothing to write.")
        return None, per_fold_dict

    summary_df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(out_dir, f'{table_kind}_table.csv')
    summary_df.to_csv(csv_path, index=False)
    print(f"[paper] summary CSV written to {csv_path}")

    # Optional per-fold dump for reproducibility and error-bar computation.
    folds_path = os.path.join(out_dir, f'{table_kind}_per_fold.csv')
    pd.concat(
        [df.assign(variant=label) for label, df in per_fold_dict.items()],
        ignore_index=True,
    ).to_csv(folds_path, index=False)
    print(f"[paper] per-fold CSV written to {folds_path}")

    tex_path = os.path.join(out_dir, f'{table_kind}_table.tex')
    if table_kind == 'ablation':
        to_latex_ablation(summary_df, tex_path, full_label=full_label)
    elif table_kind == 'pretrain':
        to_latex_pretrain_ablation(summary_df, tex_path,
                                   baseline_label=full_label)
    elif table_kind == 'main':
        to_latex_main_results(summary_df, tex_path)

    # Print final summary table: per-fold mean±std AND pooled (all folds
    # concatenated) for every paper metric.  Emitted as two compact tables
    # so each column stays readable.
    _precision = {'F1': 2, 'Precision': 2, 'Recall': 2, 'AUROC': 2, 'R2': 3, 'MAE': 3}

    def _fmt_mean_std(row, col):
        m, s = row.get(f'{col}_mean'), row.get(f'{col}_std')
        if pd.isna(m) or pd.isna(s):
            return "N/A"
        p = _precision[col]
        return f"{m:.{p}f}±{s:.{p}f}"

    def _fmt_pooled(row, col):
        v = row.get(f'{col}_pooled')
        if pd.isna(v):
            return "N/A"
        p = _precision[col]
        return f"{v:.{p}f}"

    variant_w = max(len('Variant'), *(len(r['variant']) for _, r in summary_df.iterrows())) + 2
    col_w = 18

    def _print_table(title, formatter):
        total_w = variant_w + col_w * len(PAPER_METRIC_COLS) + 2
        print("\n" + "=" * total_w)
        print(f"  {title}")
        print("=" * total_w)
        header = f"{'Variant':<{variant_w}}" + "".join(f"{col:>{col_w}}" for col in PAPER_METRIC_COLS)
        print(header)
        print("-" * total_w)
        for _, row in summary_df.iterrows():
            cells = "".join(f"{formatter(row, col):>{col_w}}" for col in PAPER_METRIC_COLS)
            print(f"{row['variant']:<{variant_w}}" + cells)
        print("=" * total_w)

    _print_table(f"PAPER TABLE SUMMARY ({table_kind}) — per-fold mean ± std", _fmt_mean_std)
    _print_table(f"PAPER TABLE SUMMARY ({table_kind}) — pooled (all folds concatenated)", _fmt_pooled)

    return summary_df, per_fold_dict


def plot_confusion_matrix(y_test, y_pred, model, output_filepath=""):
    """
    Plot confusion matrix for binary classification results.

    Args:
        y_test: Ground truth labels
        y_pred: Predicted labels
        model: Model name for plot title
        output_filepath: Optional path to save the figure

    Raises:
        Exception: If plotting fails
    """
    try:
        labels = [False, True]
        cm = metrics.confusion_matrix(y_test, y_pred, labels=labels)
        df_cm = pd.DataFrame(cm, index=labels, columns=labels)

        plt.figure(figsize=(5, 4))
        sns.heatmap(df_cm, annot=True, fmt="g", cmap="Blues",
                    xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title(f"Confusion Matrix using {model}")
        plt.tight_layout()

        if output_filepath:
            plt.savefig(output_filepath, dpi=300)
        plt.show()

    except Exception as e:
        raise Exception(f"Failed to plot confusion matrix. Exception: {e}")


def plot_scatter(df, x, y, x_label, y_label, title, path="",
                 metric='icc', log_scale=False):
    """Create scatter plot with agreement analysis and regression line.

    Args:
        df: DataFrame containing the data.
        x: Column name for x-axis (reference / ground truth).
        y: Column name for y-axis (predicted).
        x_label: Label for x-axis.
        y_label: Label for y-axis.
        title: Plot title.
        path: Optional path to save the figure.
        metric: ``'icc'`` for ICC(3,1) with 95 % CI (paper style) or
            ``'pearson'`` for Pearson R / R² (legacy style).
        log_scale: If True, relabel ticks as 2^tick (legacy behaviour for
            log-transformed data).  Default False.
    """
    gt = df[x]
    pr = df[y]

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle(title, fontsize=16)
    ax.scatter(gt, pr, color='darkblue', alpha=0.3)

    # Regression and identity lines
    slope, intercept = np.polyfit(gt, pr, 1)
    x_sorted = np.sort(gt)
    ax.plot(x_sorted, slope * x_sorted + intercept, color='red', label='Fit line')
    ax.plot([0, max(gt)], [0, max(gt)], color='black', linestyle='--',
            label='Identity line')

    # Agreement statistic
    if metric == 'icc':
        icc_val, pval, ci_lo, ci_hi = _compute_icc(gt.values, pr.values)
        pval_text = '<0.001' if pval < 0.001 else f'{pval:.3f}'
        ax.set_title(
            f"ICC={icc_val:.2f}; p-value={pval_text}; "
            f"CI95%=[{ci_lo:.2f} {ci_hi:.2f}];",
            ha="center", fontweight='bold', color='red')
    else:
        corr, pval = pearsonr(gt, pr)
        r_squared = corr ** 2
        pval_text = '<0.001' if pval < 0.001 else f'{pval:.3f}'
        ax.set_title(f"R={corr:.2f}; R\u00b2={r_squared:.2f}; "
                     f"p-value={pval_text}",
                     ha="center", fontweight='bold', color='red')

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if log_scale:
        x_ticks, _ = plt.xticks()
        y_ticks, _ = plt.yticks()
        plt.xticks(ticks=x_ticks, labels=[round(2 ** t, 1) for t in x_ticks])
        plt.yticks(ticks=y_ticks, labels=[round(2 ** t, 1) for t in y_ticks])

    ax.legend()
    ax.grid(True)

    if path:
        plt.savefig(path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close(fig)


def _compute_icc(gt, pr):
    """Compute ICC(3,1) — two-way mixed, single measures, consistency.

    Returns (icc, p_value, ci_lower, ci_upper).
    Uses the F-test formulation from Shrout & Fleiss (1979).
    """
    n = len(gt)
    k = 2  # two raters: reference and prediction
    data = np.column_stack([gt, pr])
    mean_subjects = data.mean(axis=1)
    grand_mean = data.mean()

    # Sums of squares
    ss_between = k * np.sum((mean_subjects - grand_mean) ** 2)
    ss_within = np.sum((data - mean_subjects[:, None]) ** 2)
    ss_total = np.sum((data - grand_mean) ** 2)
    ss_error = ss_within  # residual within subjects

    ms_between = ss_between / (n - 1)
    ms_error = ss_error / (n * (k - 1))

    # ICC(3,1) = (MS_between - MS_error) / (MS_between + (k-1)*MS_error)
    icc = (ms_between - ms_error) / (ms_between + (k - 1) * ms_error)

    # F-test
    f_val = ms_between / ms_error if ms_error > 0 else np.inf
    from scipy.stats import f as f_dist
    df1 = n - 1
    df2 = n * (k - 1)
    p_value = 1.0 - f_dist.cdf(f_val, df1, df2)

    # 95% CI (Shrout & Fleiss, 1979)
    f_lo = f_dist.ppf(0.025, df1, df2)
    f_hi = f_dist.ppf(0.975, df1, df2)
    ci_lower = (f_val / f_hi - 1) / (f_val / f_hi + k - 1)
    ci_upper = (f_val / f_lo - 1) / (f_val / f_lo + k - 1)

    return float(icc), float(p_value), float(ci_lower), float(ci_upper)


def plot_with_corr(x, y, **kwargs):
    """
    Helper function for facet grid plotting with correlation annotation.

    Args:
        x: X-axis data
        y: Y-axis data
        **kwargs: Additional keyword arguments
    """
    try:
        corr, _ = pearsonr(x, y)
        plt.text(0.1, 0.9, f'Corr: {corr:.2f}',
                transform=plt.gca().transAxes, fontsize=12, color='red')

        line_color = 'green' if corr > 0.1 else 'red'
        sns.regplot(x=x, y=y, scatter_kws={'s': 100},
                   line_kws={'color': line_color, 'linestyle': '--'}, ci=None)
    except Exception:
        pass

    plt.grid(True)


def plot_scatter_participants(df, x, y, PID, x_label, y_label, title, path=""):
    """
    Create faceted scatter plots for each participant.

    Args:
        df: DataFrame containing the data
        x: Column name for x-axis
        y: Column name for y-axis
        PID: Column name for participant ID
        x_label: Label for x-axis
        y_label: Label for y-axis
        title: Plot title
        path: Optional path to save the figure
    """
    g = sns.FacetGrid(df, col=PID, col_wrap=6, height=4, sharey=False, sharex=False)
    g.map(plot_with_corr, x, y)
    g.set_axis_labels(x_label, y_label)

    if path:
        plt.savefig(path, dpi=300)
    else:
        plt.show()


def plot_bland_altman(pr, gt, x_label, y_label, title, path=""):
    """
    Create Bland-Altman plot for agreement analysis.

    Args:
        pr: Predicted values
        gt: Ground truth values
        x_label: Label for x-axis (mean)
        y_label: Label for y-axis (difference)
        title: Plot title
        path: Optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle(title, fontsize=16)

    x = (pr + gt) / 2
    y = (pr - gt)

    plt.scatter(x, y, alpha=0.5)

    mean_diff = np.mean(y)
    std_diff = np.std(y)
    LLOA = mean_diff - 1.96 * std_diff
    ULOA = mean_diff + 1.96 * std_diff

    plt.axhline(mean_diff, color='red', linestyle='--', label='Bias')
    plt.axhline(ULOA, color='blue', linestyle='--', label='LOA')
    plt.axhline(LLOA, color='blue', linestyle='--')

    plt.title(f"Bias={round(mean_diff, 2)}; LLOA={round(LLOA, 2)}; ULOA={round(ULOA, 2)}",
              ha="right")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)

    if path:
        plt.savefig(path, dpi=300)
    else:
        plt.show()


def plot_nightly_agreement(gt, pr, metric_label, unit, out_dir,
                           prefix='scratch'):
    """Generate paired scatter + Bland-Altman PDFs for a nightly metric.

    Saves ``scatter_{prefix}_{metric_label}.pdf`` and
    ``perf_{prefix}_{metric_label}.pdf`` to *out_dir*, matching the
    paper figure layout (scatter on left, Bland-Altman on right).

    Args:
        gt: array-like of reference values (one per night).
        pr: array-like of predicted values.
        metric_label: short label used in filenames and axis text, e.g.
            ``'frequency'`` or ``'duration'``.
        unit: axis unit string, e.g. ``'events/h'`` or ``'seconds'``.
        out_dir: directory where PDFs are written.
        prefix: filename prefix (default ``'scratch'``).
    """
    import os
    os.makedirs(out_dir, exist_ok=True)
    gt = np.asarray(gt, dtype=float)
    pr = np.asarray(pr, dtype=float)
    ref_label = f'Reference {prefix} {metric_label} ({unit})'
    pred_label = f'Predicted {prefix} {metric_label} ({unit})'
    title = (f'Scatter plot of predicted vs reference: '
             f'Normalized {prefix} {metric_label}')

    # Scatter plot (ICC)
    _df = pd.DataFrame({'gt': gt, 'pr': pr})
    scatter_path = os.path.join(out_dir,
                                f'scatter_{prefix}_{metric_label}.pdf')
    plot_scatter(_df, 'gt', 'pr', ref_label, pred_label, title,
                 path=scatter_path, metric='icc')

    # Bland-Altman
    ba_title = title.replace('Scatter plot', 'Bland-Altman')
    mean_label = f'Mean {prefix} {metric_label} ({unit})'
    diff_label = f'Difference in {prefix} {metric_label} ({unit})'
    ba_path = os.path.join(out_dir,
                           f'perf_{prefix}_{metric_label}.pdf')
    plot_bland_altman(pr, gt, mean_label, diff_label, ba_title,
                      path=ba_path)
    print(f'  [paper] nightly agreement plots written to {out_dir}/')


def load_dataset(input_data_folder, energy_threshold=5):
    """
    Load or create dataset from data folder.

    Args:
        input_data_folder: Path to raw data folder
        energy_threshold: Energy threshold for data filtering

    Returns:
        tuple: (df_rest, dataset_name) - processed dataframe and dataset name
    """
    dataset_name = os.path.basename(input_data_folder.rstrip("/raw"))
    processed_data_folder = os.path.join(input_data_folder.rstrip("/raw"), "processed/")

    create_folder([processed_data_folder])

    # Load or create data image
    data_image_path = os.path.join(processed_data_folder,
                                   f"{dataset_name}_df_rest_th{energy_threshold}.parquet.gzip")

    if os.path.exists(data_image_path):
        print(f"Data image file found for {dataset_name}! Using existing image.")
        df_rest = pd.read_parquet(data_image_path)
    else:
        print(f"Data image file does not exist for {dataset_name}. Creating a new one...")
        df_rest = load_data(input_data_folder, energy_th=energy_threshold,
                           filter_type='non-motion', remove_outbed=False)
        df_rest.to_parquet(data_image_path, index=False)

    # Handle duplicate 'position' column if exists
    if 'position' in df_rest.columns and df_rest.columns.duplicated().any():
        # Keep only the first 'position' column
        cols = df_rest.columns.tolist()
        # Find all position indices
        position_indices = [i for i, col in enumerate(cols) if col == 'position']
        if len(position_indices) > 1:
            # Drop duplicate position columns (keep first)
            df_rest = df_rest.iloc[:, [i for i in range(len(cols)) if i not in position_indices[1:]]]

    # Rename columns to standard format
    rename_mapping = {}
    if 'position_segment' in df_rest.columns:
        # If position already exists, drop position_segment instead of renaming
        if 'position' in df_rest.columns:
            df_rest.drop(columns=['position_segment'], inplace=True)
        else:
            rename_mapping['position_segment'] = 'position'

    if 'segment_scratch' in df_rest.columns:
        rename_mapping['segment_scratch'] = 'gt1'
    if 'scratch' in df_rest.columns:
        rename_mapping['scratch'] = 'gt2'
    if 'scratch_duration' in df_rest.columns:
        rename_mapping['scratch_duration'] = 'gt3'

    if rename_mapping:
        df_rest.rename(columns=rename_mapping, inplace=True)

    df_rest.loc[:, ['pr1', 'pr2', 'pr3']] = [0, 0, 0]

    return df_rest, dataset_name


def print_dataset_statistics(df_rest, dataset_name):
    """
    Print segment statistics for a dataset.

    Args:
        df_rest: DataFrame with processed data
        dataset_name: Name of the dataset
    """
    df_segment = df_rest.groupby('segment', as_index=False).agg('max')
    th = 60
    sf = 20  # sampling frequency

    print(f"\n=== Dataset: {dataset_name} ===")
    print(f"Retained segment prevalence: "
          f"{df_segment[(df_segment.gt1==True) & (df_segment.scratch_count>th)].shape[0] / df_segment[df_segment.scratch_count>th].shape[0]:.4f}, "
          f"count: {df_segment[df_segment.scratch_count>th].shape[0]}, "
          f"average duration: {df_segment[df_segment.scratch_count>th].groupby('segment')['scratch_count'].agg('max').mean()/sf:.3f} s")

    print(f"Removed segment prevalence: "
          f"{df_segment[(df_segment.gt1==True) & (df_segment.scratch_count<=th)].shape[0] / df_segment[df_segment.scratch_count<=th].shape[0]:.4f}, "
          f"count: {df_segment[df_segment.scratch_count<=th].shape[0]}, "
          f"average duration: {df_segment[df_segment.scratch_count<=th].groupby('segment')['scratch_count'].agg('max').mean()/sf:.3f} s")

    print(f"FN due to filter: "
          f"{df_segment[(df_segment.gt1==True) & (df_segment.scratch_count<=th)].shape[0] * 100 / df_segment[(df_segment.gt1==True)].shape[0]:.4f}%")


def main():
    """Main execution function for model evaluation and analysis.

    Two modes, selected with ``--mode``:

      * ``paper-ablation`` (default): builds the Deep Scratch ablation table
        (papers/deep_scratch tab:ablation_results) from the nine ablation
        runs produced by ``experiments/run_ablation.sh``.
      * ``paper-main``: builds the main-results table (baselines + ours).
      * ``legacy``: preserves the original dataset-sweep + per-patient
        analysis flow from before the paper-table rewrite.
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='paper-ablation',
                        choices=['paper-ablation', 'paper-main', 'legacy'])
    parser.add_argument('--train_data', type=str,
                        default='/mnt/data/Nocturnal-scratch/geneactive_20hz_3s_b1s_production_train_van_new_enh_lth-rth/raw/',
                        help='Raw data folder used by the ablation runs '
                             '(matches data.input_folder in the ablation YAMLs).')
    parser.add_argument('--out_dir', type=str,
                        default='experiments/logs/paper_tables/',
                        help='Where CSV + LaTeX table snippets are written.')
    cli_args, _ = parser.parse_known_args()

    if cli_args.mode == 'paper-ablation':
        return run_paper_ablation(cli_args)
    if cli_args.mode == 'paper-main':
        return run_paper_main_results(cli_args)
    return run_legacy(cli_args)


# ----------------------------------------------------------------------------
# Paper-ablation driver
# ----------------------------------------------------------------------------
# Variant labels here match the row names in
# papers/deep_scratch/deepscratch.tex Table 3 (tab:ablation_results).
# The output-folder names (values) match the ``training.output`` field in
# experiments/configs/ablation/ablation_*.yaml.
# ----------------------------------------------------------------------------

ABLATION_VARIANTS = {
    'Ours (w/o pretrain)':      'ablation-full-no_pretrain-bs32-fold4',
    # Pretraining variants — runs pending; uncomment once the jobs finish:
    # 'Ours (w/ DINO pretrain)':  'ablation-full-dino_pretrain-bs32-fold4',
    # 'Ours (w/ MAE pretrain)':   'ablation-full-mae_pretrain-bs32-fold4',
    # Fine-tuning strategy sub-ablation — uncomment once the runs finish:
    # 'Freeze Encoder (DINO)':            'ablation-freeze_encoder_dino-bs32-fold4',
    # 'Freeze Encoder + Decoder (DINO)':  'ablation-freeze_encdec_dino-bs32-fold4',
    # 'Freeze Encoder (MAE)':             'ablation-freeze_encoder_mae-bs32-fold4',
    # 'Freeze Encoder + Decoder (MAE)':   'ablation-freeze_encdec_mae-bs32-fold4',
    'w/o Mask Head':            'ablation-no_mask_head-bs32-fold4',
    'w/o Mamba Encoder':        'ablation-no_mamba-bs32-fold4',
    'w/o ResNet Mapping':       'ablation-no_resnet-bs32-fold4',
    'w/o Cross-Attention':      'ablation-no_cross_attn-bs32-fold4',
    # Re-running after the size-1-batch BN failure — uncomment when done:
    # 'w/o Balanced Sampling':    'ablation-no_balanced-bs32-fold4',
    'Single-Task (Cls Only)':   'ablation-cls_only-bs32-fold4',
    # Baseline comparisons — LOSO outputs (uncomment once runs finish):
    # 'ResNet1D':                 'baseline-resnet1d-bs32-loso',
    # 'MTCNA2':                   'baseline-mtcna2-bs32-loso',
    # 'PatchTST':                 'baseline-patchtst-bs32-loso',
    # 'EfficientUNet':            'baseline-efficientunet-bs32-loso',
    # 'Conv1DTS':                 'baseline-conv1dts-bs32-loso',
    # 'ViT1D':                    'baseline-vit1d-bs32-loso',
    # 'BiLSTM':                   'baseline-bilstm-bs32-loso',
    # 'Mahadevan2021 (XGB)':      'baseline-mahadevan2021-loso',
    # 'Ji2023 (pragmatic)':       'baseline-ji2023-loso',
}


def run_paper_ablation(args):
    """Build the ablation-study table from the nine ablation runs."""
    print("=" * 80)
    print("PAPER TABLE: Deep Scratch Ablation Study")
    print("=" * 80)

    # Paper metrics only need per-segment predictions from
    # ``Y_test_Y_pred_*.csv`` — no raw-frame data required.
    dataset_name = os.path.basename(args.train_data.rstrip('/').rstrip('/raw'))
    folder_path = (
        f"/mnt/data/GENEActive-featurized/results/DL/{dataset_name}/"
    )
    print(f"Dataset:     {dataset_name}")
    print(f"Looking for runs under: {folder_path}")

    summary_df, _ = generate_paper_tables(
        variants=ABLATION_VARIANTS,
        folder_path=folder_path,
        df_rest=None,
        out_dir=args.out_dir,
        full_label='Ours (w/o pretrain)',  # ΔF1 reference row
        table_kind='ablation',
    )
    if summary_df is not None:
        cols = ['variant', 'n_folds'] + [
            f'{m}_{s}' for m in PAPER_METRIC_COLS for s in ('mean', 'std')
        ]
        print("\nAblation summary:")
        print(summary_df[cols].to_string(index=False))
    return summary_df


# ----------------------------------------------------------------------------
# Paper main-results driver
# ----------------------------------------------------------------------------
# Baselines + ours on the same 4-fold LOFO split.  Populate the output
# names as the baseline runs finish — defaults below match the naming
# convention used by train_scratch.py's results_folder_name.
# ----------------------------------------------------------------------------

MAIN_RESULTS_VARIANTS = {
    'ResNet1D':      'ns_detect-resnet1d-bs32-fold4',
    'MTCNA2':        'ns_detect-mtcna2-bs32-fold4',
    'PatchTST':      'ns_detect-patchtst-bs32-fold4',
    'EfficientUNet': 'ns_detect-efficientunet-bs32-fold4',
    'Conv1DTS':      'ns_detect-conv1dts-bs32-fold4',
    r'\model{} (Ours)': 'ablation-full-dino_pretrain-bs32-fold4',
}


def run_paper_main_results(args):
    """Build the main-results table from baselines + our best variant."""
    print("=" * 80)
    print("PAPER TABLE: Main Results (Baselines + Ours)")
    print("=" * 80)

    dataset_name = os.path.basename(args.train_data.rstrip('/').rstrip('/raw'))
    folder_path = (
        f"/mnt/data/GENEActive-featurized/results/DL/{dataset_name}/"
    )
    print(f"Dataset:     {dataset_name}")
    print(f"Looking for runs under: {folder_path}")

    summary_df, _ = generate_paper_tables(
        variants=MAIN_RESULTS_VARIANTS,
        folder_path=folder_path,
        df_rest=None,
        out_dir=args.out_dir,
        table_kind='main',
    )
    if summary_df is not None:
        cols = ['variant', 'n_folds'] + [
            f'{m}_{s}' for m in PAPER_METRIC_COLS for s in ('mean', 'std')
        ]
        print("\nMain-results summary:")
        print(summary_df[cols].to_string(index=False))
    return summary_df


# ----------------------------------------------------------------------------
# Legacy multi-dataset sweep (preserved from the pre-paper version)
# ----------------------------------------------------------------------------

def run_legacy(args):
    """Original behavior: iterate datasets_config + models_config."""

    datasets_config = {
        'dataset2': '/mnt/data/Nocturnal-scratch/geneactive_20hz_3s_b1s_production_test_scratch/test_tso_preprocess/raw',
        'dataset3': '/mnt/data/Nocturnal-scratch/geneactive_20hz_3s_b1s_production_test_scratch/DeepTSO-JNJ/raw',
    }
    folder_path = "/mnt/data/GENEActive-featurized/results/DL/DeepTSO-JNJ/"

    models_config = {
        'ns_detect-mbatsm-bs=32-param16-DeepTSO-UKB': {
            'results_folder': folder_path,
            'dataset': 'dataset2',
        },
        'ns_detect-mbatsm-bs=32-param16-DeepTSO-JNJ': {
            'results_folder': folder_path,
            'dataset': 'dataset3',
        },
    }

    energy_threshold = 5

    # ========================================================================
    # LOAD DATASETS
    # ========================================================================

    print("="*80)
    print("LOADING DATASETS")
    print("="*80)

    datasets = {}
    dataset_names = {}

    for dataset_key, data_folder in datasets_config.items():
        print(f"\nLoading {dataset_key} from {data_folder}")
        df_rest, dataset_name = load_dataset(data_folder, energy_threshold)
        datasets[dataset_key] = df_rest
        dataset_names[dataset_key] = dataset_name
        print_dataset_statistics(df_rest, dataset_name)

    # ========================================================================
    # EVALUATE MODELS
    # ========================================================================

    print("\n" + "="*80)
    print("EVALUATING MODELS")
    print("="*80)

    results_list = []

    for model_name, model_config in models_config.items():
        print(f"\n{'='*80}")
        print(f"Processing model: {model_name}")
        print(f"{'='*80}")

        results_folder = model_config['results_folder']
        dataset_key = model_config['dataset']

        # Validate dataset exists
        if dataset_key not in datasets:
            print(f"ERROR: Dataset '{dataset_key}' not found for model '{model_name}'. Skipping.")
            continue

        df_rest = datasets[dataset_key]
        dataset_name = dataset_names[dataset_key]
        window = os.path.basename(os.path.normpath(results_folder))

        print(f"Dataset: {dataset_name}")
        print(f"Results folder: {results_folder}")

        try:
            results_list_subjects = []
            dfp = get_predictions(results_folder, model_name, df_rest, filter_tso=True)

            if len(dfp) == 0:
                print(f"WARNING: No predictions found for model '{model_name}'. Skipping.")
                continue

            # Add predictions based on observation level predictions (pr2)
            dfp.loc[:, 'pr3_2'] = dfp.groupby('segment')['pr2'].transform('sum')
            dfp['pr1_2'] = 0
            dfp.loc[dfp['pr3_2'] > 20, 'pr1_2'] = 1

            # Calculate sequence lengths and adjust predictions
            dfp.loc[:, 'length'] = dfp.groupby('segment')['pr2'].transform('count')
            dfp['gt3'] = (dfp['gt3'] * dfp['length']) / 20
            dfp['pr3'] = (dfp['pr3'] * dfp['length']) / 20
            dfp['pr3_2'] = dfp['pr3_2'] / 20

            # Clean up predictions
            dfp.loc[dfp.pr1 == 0, ['pr3', 'pr3_2']] = 0
            dfp.loc[dfp.pr3 < 0, ['pr3']] = 0

            dfp["window"] = window
            dfp["model"] = model_name
            dfp["dataset"] = dataset_name

            # Calculate global and per-patient metrics
            g = get_metrics(dfp, 'g')
            p_subjects = dfp.groupby('PID').apply(get_metrics, 'p')
            p = p_subjects.mean()
            results_list_subjects.append(p_subjects)

            # Combine results with dataset info
            result_entry = pd.concat([
                pd.Series({'model': model_name, 'dataset': dataset_name}),
                g,
                p
            ], axis=0)
            results_list.append(result_entry)

            # Plot confusion matrix
            segments_dfp = dfp.groupby('segment').max(1)
            plot_confusion_matrix(segments_dfp.gt1, segments_dfp.pr1,
                                f"{model_name}_{dataset_name}")

            print(f"Successfully evaluated model '{model_name}' on dataset '{dataset_name}'")

        except Exception as e:
            print(f"ERROR when processing model '{model_name}': {e}")
            import traceback
            traceback.print_exc()

    # ========================================================================
    # COMBINE AND DISPLAY RESULTS
    # ========================================================================

    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)

    if results_list:
        results_df = pd.DataFrame(results_list)
        results_df = results_df.sort_values('F1_1_g', ascending=False)

        print("\nModel Performance Summary:")
        print(results_df[['model', 'dataset', 'F1_1_g', 'F1_1_p', 'Recall_1_g',
                         'Precision_1_g', 'ROC_AUC_1_g', 'Balanced_Accuracy_1_g']])

        # Save results to CSV with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_output_path = f"model_performance_summary_{timestamp}.csv"
        results_df.to_csv(results_output_path, index=False)
        print(f"\nResults saved to: {results_output_path}")

        return results_df
    else:
        print("No results to display.")
        return None


if __name__ == "__main__":
    main()
