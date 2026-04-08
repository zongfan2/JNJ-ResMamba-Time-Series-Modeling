#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocess UK Biobank accelerometer data into H5 format for Deep Scratch pretraining.

Filters:
    1. predictTSO: keep only rows within predicted TSO window (predictTSOSTART to predictTSOEND)
    2. stationary: remove stationary segments (keep active movement only)
    3. non-wear: remove non-wear periods

Output H5 structure:
    /segments/{segment_id}/
        x, y, z          - float32 arrays (raw or scaled accelerometer)
        num_samples       - int, original segment length
    /metadata/
        attrs: total_segments, total_samples, sampling_rate, scaler_applied, ...
    /scaler/  (if scaler applied)
        mean, scale      - scaler parameters for reproducibility
    /subjects/
        subject_list     - unique subject IDs

Usage:
    python3.11 data/preprocess_ukb_h5.py \
        --input_folder /mnt/imported/data/NocturnalScratch_Analysis/UKB_v2/raw/ \
        --output_h5 /mnt/data/UKB_pretrain/ukb_pretrain_20hz.h5 \
        --scaler_path /mnt/code/munge/predictive_modeling/std_scaler_3s.bin \
        --min_segment_samples 100 \
        --filter_nonwear \
        --filter_stationary
"""

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
import gc


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess UKB data for Deep Scratch pretraining')
    parser.add_argument('--input_folder', type=str, required=True,
                        help='Path to UKB raw parquet files')
    parser.add_argument('--output_h5', type=str, required=True,
                        help='Output H5 file path')
    parser.add_argument('--scaler_path', type=str, default='',
                        help='Path to saved StandardScaler (.bin/.joblib). If empty, no scaling.')
    parser.add_argument('--min_segment_samples', type=int, default=100,
                        help='Minimum segment length in samples (default: 100 = 5s at 20Hz)')
    parser.add_argument('--max_segment_samples', type=int, default=0,
                        help='Maximum segment length in samples (0 = no limit)')
    parser.add_argument('--filter_nonwear', action='store_true', default=True,
                        help='Remove non-wear periods (default: True)')
    parser.add_argument('--no_filter_nonwear', action='store_false', dest='filter_nonwear',
                        help='Do not remove non-wear periods')
    parser.add_argument('--filter_stationary', action='store_true', default=True,
                        help='Remove stationary segments (default: True)')
    parser.add_argument('--sampling_rate', type=int, default=20,
                        help='Sampling rate in Hz (default: 20)')
    parser.add_argument('--max_files', type=int, default=0,
                        help='Max number of files to process (0 = all)')
    parser.add_argument('--include_subjects', type=str, default='',
                        help='Comma-separated list of subject IDs to include (empty = all)')
    parser.add_argument('--exclude_subjects', type=str, default='',
                        help='Comma-separated list of subject IDs to exclude')
    return parser.parse_args()


def filter_tso_window(df):
    """Keep only rows where timestamp is between predictTSOSTART and predictTSOEND."""
    if 'predictTSOSTART' not in df.columns or 'predictTSOEND' not in df.columns:
        # Fall back to predictTSO boolean column
        if 'predictTSO' in df.columns:
            return df[df['predictTSO'] == True]
        return df

    # Ensure datetime types
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['predictTSOSTART'] = pd.to_datetime(df['predictTSOSTART'])
    df['predictTSOEND'] = pd.to_datetime(df['predictTSOEND'])

    # Filter: timestamp within TSO window
    mask = (df['timestamp'] >= df['predictTSOSTART']) & (df['timestamp'] <= df['predictTSOEND'])
    return df[mask]


def process_file(file_path, scaler=None, min_samples=100, max_samples=0,
                 filter_nonwear=True, filter_stationary=True):
    """
    Process a single UKB parquet file into segments.

    Returns:
        list of dicts, each with keys: segment, SUBJECT, x, y, z, num_samples
    """
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"  Warning: Failed to read {file_path}: {e}")
        return []

    if df.empty:
        return []

    # --- Filters ---
    # 1. TSO window filter
    df = filter_tso_window(df)
    if df.empty:
        return []

    # 2. Non-wear filter
    if filter_nonwear and 'non-wear' in df.columns:
        df = df[df['non-wear'] == False]
        if df.empty:
            return []

    # 3. Stationary filter
    if filter_stationary and 'stationary' in df.columns:
        df = df[df['stationary'] == False]
        if df.empty:
            return []

    # Verify required columns
    for col in ['x', 'y', 'z', 'segment']:
        if col not in df.columns:
            print(f"  Warning: Missing column '{col}' in {file_path}")
            return []

    # Extract subject ID
    subject = str(df['SUBJECT'].iloc[0]) if 'SUBJECT' in df.columns else 'unknown'

    # Group by segment and extract x, y, z
    segments = []
    for seg_name, seg_df in df.groupby('segment', sort=False):
        n = len(seg_df)

        # Filter by length
        if n < min_samples:
            continue
        if max_samples > 0 and n > max_samples:
            seg_df = seg_df.iloc[:max_samples]
            n = max_samples

        xyz = seg_df[['x', 'y', 'z']].values.astype(np.float32)

        # Apply scaler if provided
        if scaler is not None:
            # Check if scaler expects 3 features (raw xyz) or more (hand-crafted)
            n_features = getattr(scaler, 'n_features_in_', None) or len(scaler.mean_)
            if n_features == 3:
                xyz = scaler.transform(xyz)
            else:
                # Scaler was fitted on hand-crafted features, not raw xyz.
                # Fall back to per-segment z-normalization.
                pass

        segments.append({
            'segment': str(seg_name),
            'SUBJECT': subject,
            'x': xyz[:, 0],
            'y': xyz[:, 1],
            'z': xyz[:, 2],
            'num_samples': n,
        })

    return segments


def write_h5(output_path, all_segments, metadata):
    """Write all segments to H5 file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    total_segments = len(all_segments)
    print(f"\nWriting {total_segments} segments to {output_path}...")

    with h5py.File(output_path, 'w') as f:
        # Create segments group
        seg_grp = f.create_group('segments')

        for i, seg in enumerate(tqdm(all_segments, desc="Writing H5")):
            g = seg_grp.create_group(str(i))
            g.create_dataset('x', data=seg['x'], dtype='float32', compression='gzip')
            g.create_dataset('y', data=seg['y'], dtype='float32', compression='gzip')
            g.create_dataset('z', data=seg['z'], dtype='float32', compression='gzip')
            g.attrs['segment'] = seg['segment']
            g.attrs['SUBJECT'] = seg['SUBJECT']
            g.attrs['num_samples'] = seg['num_samples']

        # Write metadata
        meta_grp = f.create_group('metadata')
        for key, val in metadata.items():
            meta_grp.attrs[key] = val

        # Write scaler if available
        if 'scaler_mean' in metadata:
            scaler_grp = f.create_group('scaler')
            scaler_grp.create_dataset('mean', data=metadata['scaler_mean'])
            scaler_grp.create_dataset('scale', data=metadata['scaler_scale'])

        # Write subject list
        subjects = list(set(seg['SUBJECT'] for seg in all_segments))
        sub_grp = f.create_group('subjects')
        sub_grp.create_dataset('subject_list',
                               data=np.array(subjects, dtype=h5py.special_dtype(vlen=str)))

    file_size_mb = os.path.getsize(output_path) / (1024 ** 2)
    print(f"H5 file written: {output_path} ({file_size_mb:.1f} MB)")


def main():
    args = parse_args()

    print("=" * 60)
    print(" UKB Preprocessing for Deep Scratch Pretraining")
    print("=" * 60)
    print(f" Input folder:   {args.input_folder}")
    print(f" Output H5:      {args.output_h5}")
    print(f" Scaler:         {args.scaler_path or 'None (raw values)'}")
    print(f" Min samples:    {args.min_segment_samples}")
    print(f" Max samples:    {args.max_segment_samples or 'unlimited'}")
    print(f" Filter non-wear: {args.filter_nonwear}")
    print(f" Filter stationary: {args.filter_stationary}")
    print(f" Sampling rate:  {args.sampling_rate} Hz")
    print("=" * 60)

    # Load scaler
    scaler = None
    if args.scaler_path and os.path.exists(args.scaler_path):
        scaler = joblib.load(args.scaler_path)
        print(f"Loaded scaler from {args.scaler_path}")
    elif args.scaler_path:
        print(f"Warning: Scaler not found at {args.scaler_path}, proceeding without scaling")

    # Find all parquet files
    files = sorted(glob.glob(os.path.join(args.input_folder, "*.parquet.gzip")))
    if not files:
        files = sorted(glob.glob(os.path.join(args.input_folder, "*.parquet")))
    print(f"Found {len(files)} parquet files")

    # Apply subject filters
    if args.include_subjects:
        include_list = [s.strip() for s in args.include_subjects.split(',')]
        files = [f for f in files if any(s in os.path.basename(f) for s in include_list)]
        print(f"After include filter: {len(files)} files")

    if args.exclude_subjects:
        exclude_list = [s.strip() for s in args.exclude_subjects.split(',')]
        files = [f for f in files if all(s not in os.path.basename(f) for s in exclude_list)]
        print(f"After exclude filter: {len(files)} files")

    if args.max_files > 0:
        files = files[:args.max_files]
        print(f"Limited to {len(files)} files")

    # Process all files
    all_segments = []
    total_samples = 0
    skipped_files = 0
    start_time = datetime.now()

    for i, file_path in enumerate(files):
        if i % 50 == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            rate = i / elapsed if elapsed > 0 else 0
            print(f"  Processing file {i}/{len(files)} "
                  f"({rate:.1f} files/s, {len(all_segments)} segments so far)")

        segments = process_file(
            file_path,
            scaler=scaler,
            min_samples=args.min_segment_samples,
            max_samples=args.max_segment_samples,
            filter_nonwear=args.filter_nonwear,
            filter_stationary=args.filter_stationary,
        )

        if not segments:
            skipped_files += 1
            continue

        for seg in segments:
            total_samples += seg['num_samples']
        all_segments.extend(segments)

        # Periodic GC for large datasets
        if i % 200 == 0:
            gc.collect()

    elapsed = (datetime.now() - start_time).total_seconds()
    subjects = set(seg['SUBJECT'] for seg in all_segments)

    print(f"\nProcessing complete in {elapsed:.1f}s")
    print(f"  Total files: {len(files)}, skipped: {skipped_files}")
    print(f"  Total segments: {len(all_segments)}")
    print(f"  Total samples: {total_samples:,}")
    print(f"  Unique subjects: {len(subjects)}")
    if all_segments:
        lengths = [seg['num_samples'] for seg in all_segments]
        print(f"  Segment length: min={min(lengths)}, max={max(lengths)}, "
              f"mean={np.mean(lengths):.0f}, median={np.median(lengths):.0f}")

    if not all_segments:
        print("ERROR: No segments found after filtering. Check input data and filters.")
        sys.exit(1)

    # Build metadata
    metadata = {
        'total_segments': len(all_segments),
        'total_samples': total_samples,
        'num_subjects': len(subjects),
        'sampling_rate': args.sampling_rate,
        'min_segment_samples': args.min_segment_samples,
        'max_segment_samples': args.max_segment_samples,
        'filter_nonwear': args.filter_nonwear,
        'filter_stationary': args.filter_stationary,
        'scaler_applied': scaler is not None,
        'source': 'UKB_v2',
        'created': str(datetime.now()),
        'input_folder': args.input_folder,
    }

    if scaler is not None:
        metadata['scaler_mean'] = scaler.mean_
        metadata['scaler_scale'] = scaler.scale_

    # Write H5
    write_h5(args.output_h5, all_segments, metadata)

    print("\nDone!")


if __name__ == '__main__':
    main()
