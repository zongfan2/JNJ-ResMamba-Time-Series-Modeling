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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from scipy.stats import pearsonr

from Helpers.DL_helpers import load_data, create_folder


def get_predictions(folder_path, model, df_rest, filter_tso=False):
    """
    Load model predictions from CSV files and merge with rest data.

    Args:
        folder_path: Base directory containing model prediction folders
        model: Name of the model subdirectory
        df_rest: Rest data DataFrame to concatenate with predictions
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
        dfp = pd.concat([dfp, df_rest[df_rest.inTSO == True]], ignore_index=True)
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
        f'F1_rand_{suffix}': metrics.f1_score(segments_df.gt1, segments_df.pr1_rand, zero_division=0) * 100,
        f'Recall_rand_{suffix}': metrics.recall_score(segments_df.gt1, segments_df.pr1_rand, zero_division=0) * 100,
        f'Precision_rand_{suffix}': metrics.precision_score(segments_df.gt1, segments_df.pr1_rand, zero_division=0) * 100,
        f'ROC_AUC_rand_{suffix}': metrics.roc_auc_score(segments_df.gt1, segments_df.pr1_rand) * 100,
        f'R2_rand_{suffix}': metrics.r2_score(segments_df.gt3, segments_df.pr3_rand)
    })


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


def plot_scatter(df, x, y, x_label, y_label, title, path=""):
    """
    Create scatter plot with correlation analysis and regression line.

    Args:
        df: DataFrame containing the data
        x: Column name for x-axis (ground truth)
        y: Column name for y-axis (predictions)
        x_label: Label for x-axis
        y_label: Label for y-axis
        title: Plot title
        path: Optional path to save the figure
    """
    gt = df[x]
    pr = df[y]

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle(title, fontsize=16)
    plt.scatter(gt, pr, color='darkblue', alpha=0.3)
    plt.xscale('linear')
    plt.yscale('linear')

    # Calculate regression line
    slope, intercept = np.polyfit(gt, pr, 1)
    y_fit = slope * gt + intercept

    # Calculate Pearson correlation and R-squared
    corr, pval = pearsonr(gt, pr)
    r_squared = corr ** 2

    # Plot regression and identity lines
    plt.plot(gt, y_fit, color='red', label='Fit line')
    plt.plot([0, max(gt)], [0, max(gt)], color='black', linestyle='--', label='Identity line')

    # Format p-value
    pval_text = '<0.001' if pval < 0.001 else f'{pval:.3f}'

    plt.title(f"R={corr:.2f}; R²={r_squared:.2f}; p-value={pval_text}",
              ha="center", fontweight='bold', color='red')
    plt.ylabel(y_label)
    plt.xlabel(x_label)

    # Transform ticks if needed
    x_ticks, _ = plt.xticks()
    y_ticks, _ = plt.yticks()
    new_x_ticks = [round(2 ** tick, 1) for tick in x_ticks]
    new_y_ticks = [round(2 ** tick, 1) for tick in y_ticks]
    plt.xticks(ticks=x_ticks, labels=new_x_ticks)
    plt.yticks(ticks=y_ticks, labels=new_y_ticks)

    plt.legend()
    plt.grid(True)

    if path:
        plt.savefig(path, dpi=300)
    else:
        plt.show()


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
    """Main execution function for model evaluation and analysis."""

    # ========================================================================
    # CONFIGURATION: Define datasets and model-dataset mappings
    # ========================================================================

    # Define all datasets to load
    # Format: {dataset_key: data_folder_path}
    datasets_config = {
        # 'dataset1': '/mnt/data/Nocturnal-scratch/gene_20hz_3s_b1s_st18_ori_van_nw60nomerg_gap120_slp60_ORtp25_enha/raw',
        'dataset2': '/mnt/data/Nocturnal-scratch/geneactive_20hz_3s_b1s_production_test_scratch/test_tso_preprocess/raw',
        'dataset3': '/mnt/data/Nocturnal-scratch/geneactive_20hz_3s_b1s_production_test_scratch/DeepTSO-JNJ/raw'
    }
    folder_path = "/mnt/data/GENEActive-featurized/results/DL/DeepTSO-JNJ/"
    # folder_path2 = ""

    # Define model configurations
    # Format: {model_name: {'results_folder': path, 'dataset': dataset_key}}
    models_config = {

        'ns_detect-mbatsm-bs=32-param16-DeepTSO-UKB': {
            # 'results_folder': '/mnt/data/GENEActive-featurized/results/DL/gene_20hz_3s_b1s_st18_ori_van_nw60nomerg_gap120_slp60_ORtp25_enh/',
            'results_folder': folder_path,
            'dataset': 'dataset2',
        },
        # Add more models as needed:
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

