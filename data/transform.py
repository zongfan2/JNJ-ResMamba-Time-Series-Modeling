#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert Parquet Files to HuggingFace Dataset Format for UK Biobank Data

This script converts UK Biobank parquet.gzip files into HuggingFace datasets
that use memory-mapped files for efficient loading during training.

File naming convention:
    Processed_{eid}_{eid}_{instance}_{array_idx}_{field}_{date}.parquet.gzip
    Example: Processed_3075523_3075523_90001_0_0_2015-01-25.parquet.gzip
    Where the middle number (3075523) is the subject ID

Usage:
    python transform.py \
        --ukb_v2_folder /path/to/ukb_v2 \
        --ukb_v2_nonAD_folder /path/to/ukb_v2_nonAD \
        --output_dataset /path/to/output/hf_dataset \
        --n_nonAD_subjects 2600
"""

import os
import sys
import glob
import re
import argparse
import random
from datetime import datetime
import numpy as np
import pandas as pd


def extract_subject_id_from_filename(filename):
    """
    Extract subject ID from UK Biobank filename.

    Format: Processed_{eid}_{eid}_{instance}_{array_idx}_{field}_{date}.parquet.gzip
    Example: Processed_3075523_3075523_90001_0_0_2015-01-25.parquet.gzip

    Returns:
        str: Subject ID (e.g., '3075523')
    """
    basename = os.path.basename(filename).replace('.parquet.gzip', '')
    parts = basename.split('_')

    if len(parts) >= 3 and parts[0] == 'Processed':
        # The subject ID is the second part (index 1)
        return parts[1]

    return None


def get_subjects_from_folder(folder_path):
    """
    Get unique subject IDs from all parquet files in a folder.

    Args:
        folder_path: Path to folder containing parquet.gzip files

    Returns:
        dict: {subject_id: [list of file paths for that subject]}
    """
    files = glob.glob(os.path.join(folder_path, "*.parquet.gzip"))

    subject_files = {}
    for file in files:
        subject_id = extract_subject_id_from_filename(file)
        if subject_id:
            if subject_id not in subject_files:
                subject_files[subject_id] = []
            subject_files[subject_id].append(file)

    return subject_files


def convert_parquet_to_hf_dataset(selected_files, output_dataset_path, max_seq_length=86400, writer_batch_size=100):
    """
    Convert selected parquet files to a HuggingFace dataset.
    Uses generator with streaming writes for memory efficiency.

    Args:
        selected_files: List of parquet file paths to process
        output_dataset_path: Path where the HF dataset will be saved
        max_seq_length: Maximum sequence length in seconds (default: 86400 = 24 hours)
        writer_batch_size: Number of samples to accumulate before writing to disk (default: 100)
                          Lower values use less memory but are slower
    """
    try:
        from datasets import Dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")

    print("="*80)
    print("Converting parquet files to HuggingFace Dataset format")
    print("="*80)

    start_time = datetime.now()

    samples_per_second = 20  # 20Hz sampling rate
    max_samples = max_seq_length * samples_per_second

    # Define a generator function to yield data in batches (memory efficient)
    def data_generator():
        for file_idx, file in enumerate(selected_files):
            if file_idx % 100 == 0:
                print(f"Processing file {file_idx+1}/{len(selected_files)}: {os.path.basename(file)}")

            basename = os.path.basename(file).replace('.parquet.gzip', '')

            # Parse biobank filename: Processed_{eid}_{eid}_{instance}_{array_idx}_{field}_{date}
            parts = basename.split('_')
            if len(parts) < 7:
                print(f"Skipping invalid biobank filename: {basename}")
                continue

            subject = parts[1]  # Subject ID
            day = parts[-1]     # Date
            wrist = "unknown"   # Biobank doesn't specify wrist

            # Load parquet file
            try:
                df = pd.read_parquet(file)
            except Exception as e:
                print(f"Error loading {file}: {e}")
                continue

            # Truncate if needed
            if len(df) > max_samples:
                df = df.iloc[:max_samples].copy()

            # Create segment name
            segment_name = f"{subject}_{wrist}_{day}"

            # Prepare data dict
            data_dict = {
                'segment': segment_name,
                'subject': subject,
                'wrist': wrist,
                'day': day,
                'num_samples': len(df),
            }

            # Add sensor columns as lists
            for col in ['x', 'y', 'z', 'temperature', 'time_cyclic']:
                if col in df.columns:
                    data_dict[col] = df[col].tolist()
                else:
                    data_dict[col] = [0.0] * len(df)  # Placeholder

            # Add label columns
            for col in ['predictTSO', 'non-wear', 'other']:
                if col in df.columns:
                    data_dict[col] = df[col].tolist()
                else:
                    data_dict[col] = [0] * len(df)  # Default to 0

            yield data_dict
            del df, data_dict

    print(f"\nCreating HuggingFace Dataset from generator (streaming to disk)...")
    print(f"Total files to process: {len(selected_files)}")
    print("Using streaming writes to avoid OOM...")

    # Create dataset from generator with streaming writes
    # writer_batch_size controls how many samples to accumulate before writing
    # Smaller = less memory, but slower. Recommended: 100-1000
    print(f"Writer batch size: {writer_batch_size} (lower = less memory, slower)")

    dataset = Dataset.from_generator(
        data_generator,
        writer_batch_size=writer_batch_size,  # Write to disk every N samples
        keep_in_memory=False                  # Don't keep entire dataset in RAM
    )

    print(f"Dataset created with {len(dataset)} segments")
    print(f"Dataset features: {dataset.features}")

    # Save to disk (this enables memory-mapped access)
    print(f"\nSaving dataset to: {output_dataset_path}")
    dataset.save_to_disk(output_dataset_path)

    # Print statistics
    total_time = datetime.now() - start_time
    print(f"\n{'='*80}")
    print(f"Conversion complete!")
    print(f"  Total segments: {len(dataset)}")
    print(f"  Output path: {output_dataset_path}")
    print(f"  Total time: {total_time}")
    print(f"  Dataset size on disk: {get_folder_size(output_dataset_path):.2f} GB")
    print(f"{'='*80}")


def get_folder_size(folder_path):
    """Get total size of folder in GB"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total_size += os.path.getsize(fp)
    return total_size / (1024**3)  # Convert to GB


def main():
    parser = argparse.ArgumentParser(
        description='Convert UK Biobank parquet files to HuggingFace dataset format'
    )
    parser.add_argument(
        '--ukb_v2_folder',
        type=str,
        required=True,
        help='Path to ukb_v2 folder (all subjects will be included)'
    )
    parser.add_argument(
        '--ukb_v2_nonAD_folder',
        type=str,
        required=True,
        help='Path to ukb_v2_nonAD folder (will sample N subjects)'
    )
    parser.add_argument(
        '--output_dataset',
        type=str,
        required=True,
        help='Path where HF dataset will be saved'
    )
    parser.add_argument(
        '--n_nonAD_subjects',
        type=int,
        default=2600,
        help='Number of subjects to sample from ukb_v2_nonAD (default: 2600)'
    )
    parser.add_argument(
        '--max_seq_length',
        type=int,
        default=86400,
        help='Maximum sequence length in seconds (default: 86400 = 24 hours)'
    )
    parser.add_argument(
        '--random_seed',
        type=int,
        default=42,
        help='Random seed for subject sampling (default: 42)'
    )
    parser.add_argument(
        '--writer_batch_size',
        type=int,
        default=100,
        help='Number of samples to accumulate before writing to disk (default: 100). Lower this if OOM occurs.'
    )

    args = parser.parse_args()

    print("="*80)
    print("UK Biobank Data to HuggingFace Dataset Conversion")
    print("="*80)
    print(f"ukb_v2 folder: {args.ukb_v2_folder}")
    print(f"ukb_v2_nonAD folder: {args.ukb_v2_nonAD_folder}")
    print(f"Output dataset: {args.output_dataset}")
    print(f"Non-AD subjects to sample: {args.n_nonAD_subjects}")
    print(f"Random seed: {args.random_seed}")
    print("="*80)
    print()

    # Set random seed for reproducibility
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    # Step 1: Get all subjects from ukb_v2
    print("Step 1: Scanning ukb_v2 folder...")
    ukb_v2_subjects = get_subjects_from_folder(args.ukb_v2_folder)
    print(f"  Found {len(ukb_v2_subjects)} unique subjects")
    print(f"  Total files: {sum(len(files) for files in ukb_v2_subjects.values())}")

    # Step 2: Get subjects from ukb_v2_nonAD and sample N subjects
    print("\nStep 2: Scanning ukb_v2_nonAD folder...")
    ukb_v2_nonAD_subjects = get_subjects_from_folder(args.ukb_v2_nonAD_folder)
    print(f"  Found {len(ukb_v2_nonAD_subjects)} unique subjects")
    print(f"  Total files: {sum(len(files) for files in ukb_v2_nonAD_subjects.values())}")

    # Sample N subjects from nonAD
    if len(ukb_v2_nonAD_subjects) < args.n_nonAD_subjects:
        print(f"\n  WARNING: Requested {args.n_nonAD_subjects} subjects but only {len(ukb_v2_nonAD_subjects)} available")
        print(f"  Using all {len(ukb_v2_nonAD_subjects)} subjects from ukb_v2_nonAD")
        sampled_nonAD_subjects = list(ukb_v2_nonAD_subjects.keys())
    else:
        print(f"\n  Randomly sampling {args.n_nonAD_subjects} subjects from ukb_v2_nonAD...")
        all_nonAD_subjects = list(ukb_v2_nonAD_subjects.keys())
        sampled_nonAD_subjects = random.sample(all_nonAD_subjects, args.n_nonAD_subjects)

    print(f"  Selected {len(sampled_nonAD_subjects)} subjects from ukb_v2_nonAD")

    # Step 3: Combine file lists
    print("\nStep 3: Combining file lists...")
    all_selected_files = []

    # Add all files from ukb_v2
    for subject, files in ukb_v2_subjects.items():
        all_selected_files.extend(files)

    # Add files from sampled nonAD subjects
    for subject in sampled_nonAD_subjects:
        all_selected_files.extend(ukb_v2_nonAD_subjects[subject])

    print(f"  Total files selected: {len(all_selected_files)}")
    print(f"  Total subjects: {len(ukb_v2_subjects) + len(sampled_nonAD_subjects)}")
    print(f"    - From ukb_v2: {len(ukb_v2_subjects)} subjects")
    print(f"    - From ukb_v2_nonAD: {len(sampled_nonAD_subjects)} subjects")

    # Save subject list for reference
    subject_list_file = os.path.join(os.path.dirname(args.output_dataset), "selected_subjects.txt")
    os.makedirs(os.path.dirname(args.output_dataset), exist_ok=True)

    with open(subject_list_file, 'w') as f:
        f.write("# Selected Subjects for HuggingFace Dataset\n")
        f.write(f"# Generated: {datetime.now()}\n")
        f.write(f"# Random seed: {args.random_seed}\n\n")
        f.write(f"# ukb_v2 subjects ({len(ukb_v2_subjects)}):\n")
        for subject in sorted(ukb_v2_subjects.keys()):
            f.write(f"ukb_v2,{subject}\n")
        f.write(f"\n# ukb_v2_nonAD subjects ({len(sampled_nonAD_subjects)}):\n")
        for subject in sorted(sampled_nonAD_subjects):
            f.write(f"ukb_v2_nonAD,{subject}\n")

    print(f"\nSubject list saved to: {subject_list_file}")

    # Step 4: Convert to HuggingFace dataset
    print("\nStep 4: Converting to HuggingFace dataset...")
    try:
        convert_parquet_to_hf_dataset(
            selected_files=all_selected_files,
            output_dataset_path=args.output_dataset,
            max_seq_length=args.max_seq_length,
            writer_batch_size=args.writer_batch_size
        )

        print("\n" + "="*80)
        print("SUCCESS! Dataset conversion complete.")
        print("="*80)
        print("\nNext steps:")
        print("1. In your training script, load the dataset:")
        print("   from data.loading import load_hf_dataset_tso_patch")
        print(f"   dataset = load_hf_dataset_tso_patch('{args.output_dataset}')")
        print()
        print("2. Access samples during training (memory-efficient):")
        print("   for i in range(len(dataset)):")
        print("       sample = dataset[i]")
        print("       x_data = np.array(sample['x'])")
        print("       # ... process sample")
        print("="*80)

    except Exception as e:
        print(f"\nERROR during conversion: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
