# -*- coding: utf-8 -*-
"""
Convert Parquet Files to H5 Format for Fast Training
=====================================================

This script converts raw parquet files to a preprocessed H5 file that can be
loaded much faster during training. All preprocessing (scaling, time encoding,
padding) is done once here and saved to H5.

Usage:
    python convert_parquet_to_h5.py \
        --input_folder /path/to/parquet/files \
        --output_h5 /path/to/output.h5 \
        --additional_folder /path/to/additional/parquet \
        --scaler_path /path/to/scaler.joblib \
        --max_seq_length 86400 \
        --val_size 0.1 \
        --balance_folders
"""


import subprocess

import joblib

shell_script = '''
cd munge/predictive_modeling
sudo python3.11 -m pip install -r requirements-tso.txt
sudo python3.11 -m pip install -e .
sudo python3.11 -m pip install optuna==4.3.0 seaborn ray TensorboardX torcheval ruptures mamba-ssm[causal-conv1d]==2.2.2
'''
result = subprocess.run(shell_script, shell=True, capture_output=True, text=True)

import os
import sys
import argparse
import glob
import h5py
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from tqdm import tqdm
import random
import gc


def parse_biobank_filename(file):
    """Parse biobank filename to extract metadata."""
    basename = os.path.basename(file).replace('.parquet.gzip', '')
    parts = basename.split('_')

    if len(parts) >= 7 and parts[0] == 'Processed':
        current_subject = parts[1]
        day = parts[-1]
        instance_info = '_'.join(parts[2:-1])
    else:
        current_subject = parts[1] if len(parts) > 1 else 'unknown'
        day = parts[-1] if len(parts) > 0 else 'unknown'
        instance_info = '_'.join(parts[2:-1]) if len(parts) > 3 else ''

    return current_subject, day, instance_info


def generate_time_cyclic(timestamps, use_sincos=True):
    """
    Generate cyclic time encoding from timestamps.

    Args:
        timestamps: Pandas datetime series
        use_sincos: If True, return (sin, cos) pair for unique encoding.
                   If False, return only sin (backward compatible).

    Returns:
        If use_sincos=True: (time_sin, time_cos) - 2D encoding
        If use_sincos=False: time_sin only - 1D encoding (original behavior)

    Note:
        Sin+Cos encoding provides unique representation for all times:
        - 00:00 (midnight) -> (sin=0.0, cos=1.0)
        - 06:00 (morning)  -> (sin=1.0, cos=0.0)
        - 12:00 (noon)     -> (sin=0.0, cos=-1.0)  [Different from midnight!]
        - 18:00 (evening)  -> (sin=-1.0, cos=0.0)
    """
    if not pd.api.types.is_datetime64_any_dtype(timestamps):
        timestamps = pd.to_datetime(timestamps)

    hour = timestamps.dt.hour
    minute = timestamps.dt.minute

    # Time as fraction of day [0, 1)
    time_fraction = (hour + minute / 60) / 24

    # Generate sin component
    time_sin = np.sin(2 * np.pi * time_fraction)

    if use_sincos:
        # Generate cos component for unique 2D encoding
        time_cos = np.cos(2 * np.pi * time_fraction)
        return time_sin.values, time_cos.values
    else:
        # Backward compatible: return only sin
        return time_sin.values


def load_and_preprocess_segment(file, scaler, max_samples, samples_per_second=20, use_sincos=True):
    """
    Load a single parquet file and preprocess it.

    Args:
        file: Path to parquet file
        scaler: StandardScaler for x, y, z columns
        max_samples: Maximum number of samples to load
        samples_per_second: Sampling rate (default 20Hz)
        use_sincos: If True, generate sin+cos time encoding (6 channels total).
                   If False, generate only sin (5 channels, backward compatible).

    Returns:
        dict with keys: x, y, z, temperature, time_sin, [time_cos], predictTSO,
                       non_wear, segment, num_samples
    """
    try:
        # Parse filename
        current_subject, day, instance_info = parse_biobank_filename(file)

        # Load parquet file
        df = pd.read_parquet(file)

        # Truncate if needed
        if len(df) > max_samples:
            df = df.iloc[:max_samples]

        # Apply scaler to x, y, z columns if provided
        if scaler is not None:
            columns_to_scale = ['x', 'y', 'z']
            cols_present = [col for col in columns_to_scale if col in df.columns]
            if cols_present:
                df[cols_present] = scaler.transform(df[cols_present])

        # Generate time encoding from timestamp
        if 'timestamp' in df.columns:
            time_encoding = generate_time_cyclic(df['timestamp'], use_sincos=use_sincos)
            if use_sincos:
                df['time_sin'], df['time_cos'] = time_encoding
            else:
                df['time_sin'] = time_encoding
        else:
            # Fallback: generate based on sample position
            time_vals = np.arange(len(df)) / (samples_per_second * 3600 * 24)
            df['time_sin'] = np.sin(2 * np.pi * time_vals)
            if use_sincos:
                df['time_cos'] = np.cos(2 * np.pi * time_vals)

        # Determine wrist value
        wrist_value = int(df['wrist'].iloc[0]) if 'wrist' in df.columns and df['wrist'].iloc[0] in [0, 1] else 0

        # Create segment identifier
        segment_name = f"{current_subject}_{wrist_value}_{instance_info}_{day}"

        # Extract required columns as numpy arrays
        result = {
            'x': df['x'].values.astype(np.float32) if 'x' in df.columns else np.zeros(len(df), dtype=np.float32),
            'y': df['y'].values.astype(np.float32) if 'y' in df.columns else np.zeros(len(df), dtype=np.float32),
            'z': df['z'].values.astype(np.float32) if 'z' in df.columns else np.zeros(len(df), dtype=np.float32),
            'temperature': df['temperature'].values.astype(np.float32) if 'temperature' in df.columns else np.zeros(len(df), dtype=np.float32),
            'time_sin': df['time_sin'].values.astype(np.float32),
            'predictTSO': df['predictTSO'].values.astype(np.int8) if 'predictTSO' in df.columns else np.zeros(len(df), dtype=np.int8),
            'non_wear': df['non-wear'].values.astype(np.int8) if 'non-wear' in df.columns else np.zeros(len(df), dtype=np.int8),
            'segment': segment_name,
            'num_samples': len(df)
        }

        # Add time_cos if using sin+cos encoding
        if use_sincos:
            result['time_cos'] = df['time_cos'].values.astype(np.float32)

        return result

    except Exception as e:
        print(f"Error loading {file}: {e}")
        return None


def convert_parquet_to_h5(input_folder, output_h5, additional_folder=None,
                          scaler_path=None, max_seq_length=86400,
                          balance_folders=True, split_seed=42, use_sincos=True):
    """
    Convert parquet files to H5 format with preprocessing.

    Args:
        input_folder: Path to primary parquet folder
        output_h5: Path to output H5 file
        additional_folder: Path to second parquet folder (optional)
        scaler_path: Path to pretrained scaler (.joblib)
        max_seq_length: Maximum sequence length in seconds
        balance_folders: Balance file counts between folders
        split_seed: Random seed for file sampling
        use_sincos: If True, use sin+cos time encoding (6 channels: x,y,z,temp,time_sin,time_cos).
                   If False, use sin only (5 channels: x,y,z,temp,time_sin). Default: True
    """
    print(f"\n{'='*80}")
    print("Converting Parquet Files to H5 Format")
    print(f"{'='*80}\n")

    # Load scaler if provided
    scaler = None
    if scaler_path is not None:
        print(f"Loading scaler from: {scaler_path}")
        scaler = joblib.load(scaler_path)
        print(f"Scaler loaded: {type(scaler).__name__}\n")

    # Find parquet files
    parquet_files_folder1 = sorted(glob.glob(os.path.join(input_folder, "*.parquet*")))

    if len(parquet_files_folder1) == 0:
        raise ValueError(f"No parquet files found in {input_folder}")

    print(f"Found {len(parquet_files_folder1)} files in {input_folder}")

    # Load from additional folder if provided
    parquet_files = parquet_files_folder1
    if additional_folder is not None:
        parquet_files_folder2 = sorted(glob.glob(os.path.join(additional_folder, "*.parquet*")))

        if len(parquet_files_folder2) == 0:
            print(f"WARNING: No parquet files found in {additional_folder}")
        else:
            print(f"Found {len(parquet_files_folder2)} files in {additional_folder}")

            # Balance folders if requested
            if balance_folders:
                min_count = min(len(parquet_files_folder1), len(parquet_files_folder2))
                print(f"\nBalancing folders: Using {min_count} files from each")

                rng = random.Random(split_seed)
                parquet_files_folder1 = rng.sample(parquet_files_folder1, min_count) if len(parquet_files_folder1) > min_count else parquet_files_folder1
                parquet_files_folder2 = rng.sample(parquet_files_folder2, min_count) if len(parquet_files_folder2) > min_count else parquet_files_folder2

                print(f"  Folder 1: {len(parquet_files_folder1)} files")
                print(f"  Folder 2: {len(parquet_files_folder2)} files")

            parquet_files = parquet_files_folder1 + parquet_files_folder2
            print(f"Total files: {len(parquet_files)}\n")

    samples_per_second = 20
    max_samples = max_seq_length * samples_per_second

    # Determine number of channels based on time encoding method
    num_channels = 6 if use_sincos else 5

    print(f"Max sequence length: {max_seq_length}s = {max_samples:,} samples @ 20Hz")
    print(f"Time encoding: {'Sin+Cos (6 channels)' if use_sincos else 'Sin only (5 channels)'}")
    print(f"Processing {len(parquet_files)} files...\n")

    start_time = datetime.now()

    # Use max_samples as max_len directly (no need to scan files)
    # All files will be truncated to this length anyway
    max_len = max_samples
    valid_files = parquet_files  # Assume all files are valid; errors handled during processing
    num_segments = len(valid_files)

    if num_segments == 0:
        raise ValueError("No parquet files found!")

    print(f"\nTotal files to process: {num_segments}")
    print(f"Maximum segment length: {max_len:,} samples (= {max_seq_length}s @ 20Hz)")
    print(f"Note: Files will be processed incrementally to minimize memory usage\n")

    # Create H5 file with pre-allocated datasets
    print(f"Creating H5 file: {output_h5}")
    os.makedirs(os.path.dirname(output_h5), exist_ok=True)

    with h5py.File(output_h5, 'w') as h5f:
        # Store metadata
        h5f.attrs['num_segments'] = num_segments
        h5f.attrs['max_seq_length'] = max_seq_length
        h5f.attrs['samples_per_second'] = samples_per_second
        h5f.attrs['max_len'] = max_len
        h5f.attrs['num_channels'] = num_channels
        h5f.attrs['use_sincos'] = use_sincos
        h5f.attrs['creation_date'] = str(datetime.now())
        if scaler_path:
            h5f.attrs['scaler_path'] = scaler_path

        # Create datasets with chunking
        chunk_size = (1, min(1200, max_len), num_channels)

        ds_X = h5f.create_dataset(
            'X',
            shape=(num_segments, max_len, num_channels),
            dtype=np.float32,
            chunks=chunk_size,
            compression='gzip',
            compression_opts=4
        )

        ds_Y = h5f.create_dataset(
            'Y',
            shape=(num_segments, max_len, 2),
            dtype=np.int8,
            chunks=(1, min(1200, max_len), 2),
            compression='gzip',
            compression_opts=4
        )

        ds_lens = h5f.create_dataset(
            'seq_lengths',
            shape=(num_segments,),
            dtype=np.int32
        )

        dt = h5py.string_dtype(encoding='utf-8')
        ds_segments = h5f.create_dataset(
            'segment_names',
            shape=(num_segments,),
            dtype=dt
        )

        # Process and write each file incrementally
        print("\nProcessing and writing data to H5...")
        idx = 0
        failed_files = 0

        for file in tqdm(valid_files, desc="Writing to H5"):
            segment_data = load_and_preprocess_segment(file, scaler, max_samples, samples_per_second, use_sincos=use_sincos)

            if segment_data is None:
                failed_files += 1
                continue

            seg_len = segment_data['num_samples']

            # Stack channels based on time encoding method
            if use_sincos:
                # 6 channels: x, y, z, temperature, time_sin, time_cos
                X_seg = np.stack([
                    segment_data['x'],
                    segment_data['y'],
                    segment_data['z'],
                    segment_data['temperature'],
                    segment_data['time_sin'],
                    segment_data['time_cos']
                ], axis=1).astype(np.float32)  # [seq_len, 6]
            else:
                # 5 channels: x, y, z, temperature, time_sin
                X_seg = np.stack([
                    segment_data['x'],
                    segment_data['y'],
                    segment_data['z'],
                    segment_data['temperature'],
                    segment_data['time_sin']
                ], axis=1).astype(np.float32)  # [seq_len, 5]

            # Pad to max_len
            if seg_len < max_len:
                padding = np.zeros((max_len - seg_len, num_channels), dtype=np.float32)
                X_seg = np.vstack([X_seg, padding])

            ds_X[idx] = X_seg

            # Stack labels: predictTSO, non_wear
            Y_seg = np.stack([
                segment_data['predictTSO'],
                segment_data['non_wear']
            ], axis=1).astype(np.int8)  # [seq_len, 2]

            # Pad to max_len
            if seg_len < max_len:
                padding = np.zeros((max_len - seg_len, 2), dtype=np.int8)
                Y_seg = np.vstack([Y_seg, padding])

            ds_Y[idx] = Y_seg

            # Store length and segment name
            ds_lens[idx] = seg_len
            ds_segments[idx] = segment_data['segment']

            idx += 1

            # Free memory and force garbage collection every 50 files
            del segment_data, X_seg, Y_seg
            if idx % 50 == 0:
                gc.collect()

        # Trim datasets if some files failed
        actual_segments = idx
        if actual_segments < num_segments:
            print(f"\nTrimming H5 datasets: {actual_segments} valid segments (skipped {failed_files} failed files)")
            ds_X.resize((actual_segments, max_len, 5))
            ds_Y.resize((actual_segments, max_len, 2))
            ds_lens.resize((actual_segments,))
            ds_segments.resize((actual_segments,))
            h5f.attrs['num_segments'] = actual_segments

    total_time = datetime.now() - start_time
    file_size_mb = os.path.getsize(output_h5) / (1024 * 1024)

    print(f"\n{'='*80}")
    print("Conversion Complete!")
    print(f"{'='*80}")
    print(f"Output file: {output_h5}")
    print(f"File size: {file_size_mb:.2f} MB")
    print(f"Total segments written: {actual_segments if 'actual_segments' in locals() else num_segments}")
    if 'failed_files' in locals() and failed_files > 0:
        print(f"Failed files: {failed_files}")
    print(f"Total time: {total_time}")
    print(f"{'='*80}\n")


def split_h5_train_val(h5_file, val_size=0.1, shuffle=True, seed=42):
    """
    Create train/val split indices for H5 file.

    Args:
        h5_file: Path to H5 file
        val_size: Validation set fraction
        shuffle: Whether to shuffle before splitting
        seed: Random seed

    Returns:
        dict with 'train' and 'val' index arrays
    """
    with h5py.File(h5_file, 'r') as h5f:
        num_segments = h5f.attrs['num_segments']

    # Create indices
    indices = np.arange(num_segments)

    if shuffle:
        rng = np.random.RandomState(seed)
        rng.shuffle(indices)

    # Split
    val_count = int(num_segments * val_size)
    train_indices = indices[val_count:]
    val_indices = indices[:val_count]

    return {
        'train': train_indices,
        'val': val_indices
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert Parquet files to H5 format')
    parser.add_argument('--input_folder', type=str, required=True, help='Path to primary parquet folder')
    parser.add_argument('--output_h5', type=str, required=True, help='Path to output H5 file')
    parser.add_argument('--additional_folder', type=str, default=None, help='Path to second parquet folder')
    parser.add_argument('--scaler_path', type=str, default=None, help='Path to pretrained scaler (.joblib)')
    parser.add_argument('--max_seq_length', type=int, default=86400, help='Max sequence length in seconds (default: 86400 = 24h)')
    parser.add_argument('--balance_folders', action='store_true', help='Balance file counts between folders')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling')
    parser.add_argument('--val_size', type=float, default=0.1, help='Validation set size (default: 0.1)')
    parser.add_argument('--create_split', action='store_true', help='Create train/val split file')
    parser.add_argument('--use_sincos', type=bool, default=True,
                        help='Use sin+cos time encoding (6 channels) if True, sin only (5 channels) if False. Default: True')

    args = parser.parse_args()

    # Convert parquet to H5
    convert_parquet_to_h5(
        input_folder=args.input_folder,
        output_h5=args.output_h5,
        additional_folder=args.additional_folder,
        scaler_path=args.scaler_path,
        max_seq_length=args.max_seq_length,
        balance_folders=args.balance_folders,
        split_seed=args.seed,
        use_sincos=args.use_sincos
    )

    # Create train/val split if requested
    if args.create_split:
        print("\nCreating train/val split...")
        split_indices = split_h5_train_val(
            h5_file=args.output_h5,
            val_size=args.val_size,
            shuffle=True,
            seed=args.seed
        )

        split_file = args.output_h5.replace('.h5', '_split.npz')
        np.savez(split_file, **split_indices)

        print(f"Train indices: {len(split_indices['train'])}")
        print(f"Val indices: {len(split_indices['val'])}")
        print(f"Split saved to: {split_file}\n")
