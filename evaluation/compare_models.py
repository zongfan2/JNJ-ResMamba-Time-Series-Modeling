"""
Script to compare predictions from different models.

Compares predictions from the same case under two folders and organizes them
based on accuracy difference threshold.
"""

import os
import re
from pathlib import Path
from typing import Dict, Tuple, List
import argparse
from PIL import Image


def parse_filename(filename: str) -> Tuple[str, float]:
    """
    Parse filename to extract case ID and accuracy.

    Args:
        filename: e.g., 'batch_predictions_5_US10013009_0_2023-02-23_20260121_065118_Acc=88.06.png'

    Returns:
        Tuple of (case_id, accuracy)
        case_id: e.g., 'US10013009_0_2023-02-23' (without timestamp)
        accuracy: e.g., 88.06
    """
    # Extract accuracy - match digits with optional decimal point
    acc_match = re.search(r'Acc=(\d+(?:\.\d+)?)', filename)
    if not acc_match:
        return None, None

    accuracy = float(acc_match.group(1))

    # Extract case ID - everything between batch_predictions_X_ and _Acc=
    # Pattern: batch_predictions_\d+_(.+?)_Acc=
    case_match = re.search(r'batch_predictions_\d+_(.+?)_Acc=', filename)
    if not case_match:
        return None, None

    full_case_id = case_match.group(1)

    # Remove timestamp suffix (format: _YYYYMMDD_HHMMSS)
    # Keep only: SUBJECT_WRIST_DATE (e.g., US10013009_0_2023-02-23)
    # Remove the timestamp pattern _\d{8}_\d{6} from the end
    case_id = re.sub(r'_\d{8}_\d{6}$', '', full_case_id)

    return case_id, accuracy


def merge_images_horizontal(img1_path: str, img2_path: str, output_path: str,
                           model1_name: str, model2_name: str,
                           acc1: float, acc2: float, spacing: int = 20):
    """
    Merge two images horizontally with labels.

    Args:
        img1_path: Path to first image
        img2_path: Path to second image
        output_path: Output path for merged image
        model1_name: Name of first model
        model2_name: Name of second model
        acc1: Accuracy of first model
        acc2: Accuracy of second model
        spacing: Pixel spacing between images (default: 20)
    """
    # Load images
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)

    # Get dimensions
    width1, height1 = img1.size
    width2, height2 = img2.size

    # Create new image with spacing
    total_width = width1 + width2 + spacing
    max_height = max(height1, height2)

    # Create white background
    merged = Image.new('RGB', (total_width, max_height), 'white')

    # Paste images (center vertically if heights differ)
    y_offset1 = (max_height - height1) // 2
    y_offset2 = (max_height - height2) // 2

    merged.paste(img1, (0, y_offset1))
    merged.paste(img2, (width1 + spacing, y_offset2))

    # Save merged image
    merged.save(output_path, dpi=(300, 300))


def collect_predictions(folder_path: str) -> Dict[str, Tuple[str, float]]:
    """
    Collect all prediction files from folder and subfolder (low_acc).

    Args:
        folder_path: Path to the val folder

    Returns:
        Dictionary mapping case_id to (file_path, accuracy)
    """
    predictions = {}
    folder = Path(folder_path)

    if not folder.exists():
        print(f"Warning: Folder {folder_path} does not exist")
        return predictions

    # Check main folder and low_acc subfolder
    search_paths = [folder]
    low_acc_folder = folder / 'low_acc'
    if low_acc_folder.exists():
        search_paths.append(low_acc_folder)

    for search_path in search_paths:
        for file in search_path.glob('*.png'):
            case_id, accuracy = parse_filename(file.name)
            if case_id is not None:
                # Store the full path and accuracy
                predictions[case_id] = (str(file), accuracy)

    return predictions


def compare_predictions(folder1: str, folder2: str, output_folder: str,
                       threshold: float = 10.0, model1_name: str = "model1",
                       model2_name: str = "model2"):
    """
    Compare predictions from two model folders.

    Args:
        folder1: Path to first model's val folder
        folder2: Path to second model's val folder
        output_folder: Output folder for comparison results
        threshold: Accuracy difference threshold (default 10%)
        model1_name: Name for first model (for output naming)
        model2_name: Name for second model (for output naming)
    """
    print("Collecting predictions from folder 1...")
    predictions1 = collect_predictions(folder1)
    print(f"Found {len(predictions1)} predictions in folder 1")

    print("Collecting predictions from folder 2...")
    predictions2 = collect_predictions(folder2)
    print(f"Found {len(predictions2)} predictions in folder 2")

    # Find common case IDs
    common_cases = set(predictions1.keys()) & set(predictions2.keys())
    print(f"\nFound {len(common_cases)} common cases")

    if len(common_cases) == 0:
        print("No common cases found. Exiting.")
        return

    # Create output folders
    same_folder = Path(output_folder) / 'similar_accuracy'
    diff_folder = Path(output_folder) / 'different_accuracy'
    same_folder.mkdir(parents=True, exist_ok=True)
    diff_folder.mkdir(parents=True, exist_ok=True)

    # Compare and organize
    similar_count = 0
    different_count = 0

    comparison_results = []

    for case_id in sorted(common_cases):
        file1, acc1 = predictions1[case_id]
        file2, acc2 = predictions2[case_id]

        acc_diff = abs(acc1 - acc2)

        comparison_results.append({
            'case_id': case_id,
            'model1_acc': acc1,
            'model2_acc': acc2,
            'acc_diff': acc_diff,
            'file1': file1,
            'file2': file2
        })

        # Determine output folder
        if acc_diff < threshold:
            target_folder = same_folder
            similar_count += 1
        else:
            target_folder = diff_folder
            different_count += 1

        # Merge images horizontally with descriptive name
        merged_filename = f"{case_id}_{model1_name}={acc1:.2f}_vs_{model2_name}={acc2:.2f}_diff={acc_diff:.2f}.png"
        merged_path = target_folder / merged_filename

        merge_images_horizontal(
            img1_path=file1,
            img2_path=file2,
            output_path=str(merged_path),
            model1_name=model1_name,
            model2_name=model2_name,
            acc1=acc1,
            acc2=acc2
        )

    # Generate summary report
    summary_path = Path(output_folder) / 'comparison_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("PREDICTION COMPARISON SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model 1 folder: {folder1}\n")
        f.write(f"Model 2 folder: {folder2}\n")
        f.write(f"Accuracy difference threshold: {threshold}%\n\n")
        f.write(f"Total predictions in Model 1: {len(predictions1)}\n")
        f.write(f"Total predictions in Model 2: {len(predictions2)}\n")
        f.write(f"Common cases found: {len(common_cases)}\n\n")
        f.write(f"Similar accuracy (diff < {threshold}%): {similar_count}\n")
        f.write(f"Different accuracy (diff >= {threshold}%): {different_count}\n\n")
        f.write("=" * 80 + "\n")
        f.write("DETAILED RESULTS\n")
        f.write("=" * 80 + "\n\n")

        # Sort by accuracy difference (descending)
        comparison_results.sort(key=lambda x: x['acc_diff'], reverse=True)

        f.write(f"{'Case ID':<50} {'Model1 Acc':<12} {'Model2 Acc':<12} {'Diff':<8} {'Category'}\n")
        f.write("-" * 100 + "\n")

        for result in comparison_results:
            category = "Similar" if result['acc_diff'] < threshold else "Different"
            f.write(f"{result['case_id']:<50} {result['model1_acc']:>10.2f}% "
                   f"{result['model2_acc']:>10.2f}% {result['acc_diff']:>6.2f}% "
                   f"{category}\n")

    print(f"\n{'='*80}")
    print("COMPARISON COMPLETE")
    print(f"{'='*80}")
    print(f"Similar accuracy cases: {similar_count}")
    print(f"Different accuracy cases: {different_count}")
    print(f"\nResults saved to: {output_folder}")
    print(f"  - Similar accuracy: {same_folder}")
    print(f"  - Different accuracy: {diff_folder}")
    print(f"  - Summary report: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare predictions from different models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python compare_models.py \\
    --folder1 model1_output/training/debug_predictions/val \\
    --folder2 model2_output/training/debug_predictions/val \\
    --output comparison_results \\
    --threshold 10.0 \\
    --model1_name mba4tso \\
    --model2_name patchtst
        """
    )

    parser.add_argument('--folder1', type=str, required=True,
                       help='Path to first model val folder')
    parser.add_argument('--folder2', type=str, required=True,
                       help='Path to second model val folder')
    parser.add_argument('--output', type=str, required=True,
                       help='Output folder for comparison results')
    parser.add_argument('--threshold', type=float, default=10.0,
                       help='Accuracy difference threshold in percentage (default: 10.0)')
    parser.add_argument('--model1_name', type=str, default='model1',
                       help='Name for first model (default: model1)')
    parser.add_argument('--model2_name', type=str, default='model2',
                       help='Name for second model (default: model2)')

    args = parser.parse_args()

    compare_predictions(
        folder1=args.folder1,
        folder2=args.folder2,
        output_folder=args.output,
        threshold=args.threshold,
        model1_name=args.model1_name,
        model2_name=args.model2_name
    )


if __name__ == '__main__':
    main()
