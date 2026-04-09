# -*- coding: utf-8 -*-
"""
Load preprocessed UKB H5 data for self-supervised pretraining.

The H5 file is produced by data/preprocess_ukb_h5.py and stores variable-length
(x, y, z) segments. This loader returns a DataFrame compatible with the
batch_generator / add_padding pipeline used in train_scratch.py.

Memory-efficient: reads segment lengths first, pre-allocates arrays, then
fills in one pass. Avoids creating millions of small DataFrames.
"""

import h5py
import numpy as np
import pandas as pd
from datetime import datetime


def load_ukb_pretrain_h5(file_path, max_segments=0):
    """
    Load UKB pretraining H5 into a DataFrame with columns expected by
    batch_generator and add_padding: x, y, z, segment, SUBJECT, position_segment.

    Since UKB has no scratch labels, dummy labels are added:
        scratch = 0, segment_scratch = False, scratch_duration = 0.0

    Args:
        file_path: Path to UKB H5 file
        max_segments: Maximum segments to load (0 = all)

    Returns:
        df: DataFrame with per-sample rows, grouped by 'segment'
        metadata: dict of H5 metadata attributes
    """
    print(f"Loading UKB pretrain H5: {file_path}")
    start_time = datetime.now()

    metadata = {}

    with h5py.File(file_path, 'r') as f:
        # Read metadata
        for key, val in f['metadata'].attrs.items():
            metadata[key] = val

        segs_group = f['segments']
        total_in_file = len(segs_group)
        total = min(total_in_file, max_segments) if max_segments > 0 else total_in_file
        print(f"  Total segments in H5: {total_in_file}, loading: {total}")

        # --- Pass 1: read lengths to pre-allocate ---
        print("  Pass 1: reading segment lengths...")
        lengths = np.empty(total, dtype=np.int64)
        for i in range(total):
            lengths[i] = segs_group[str(i)]['x'].shape[0]
            if (i + 1) % 500000 == 0:
                print(f"    Scanned {i + 1}/{total} segments...")

        total_samples = int(lengths.sum())
        print(f"  Total samples: {total_samples:,}")

        # --- Pre-allocate arrays ---
        x_arr = np.empty(total_samples, dtype=np.float32)
        y_arr = np.empty(total_samples, dtype=np.float32)
        z_arr = np.empty(total_samples, dtype=np.float32)
        seg_arr = np.empty(total_samples, dtype=object)
        subj_arr = np.empty(total_samples, dtype=object)
        pos_arr = np.empty(total_samples, dtype=np.int32)

        # --- Pass 2: fill arrays ---
        print("  Pass 2: reading data...")
        offset = 0
        for i in range(total):
            g = segs_group[str(i)]
            n = lengths[i]
            end = offset + n

            x_arr[offset:end] = g['x'][()]
            y_arr[offset:end] = g['y'][()]
            z_arr[offset:end] = g['z'][()]
            seg_arr[offset:end] = g.attrs['segment']
            subj_arr[offset:end] = g.attrs['SUBJECT']
            pos_arr[offset:end] = np.arange(1, n + 1, dtype=np.int32)

            offset = end

            if (i + 1) % 100000 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                print(f"    Loaded {i + 1}/{total} segments "
                      f"({offset:,} samples, {elapsed:.0f}s)")

    # --- Build DataFrame in one shot ---
    print("  Building DataFrame...")
    df = pd.DataFrame({
        'x': x_arr,
        'y': y_arr,
        'z': z_arr,
        'segment': pd.Categorical(seg_arr),
        'SUBJECT': pd.Categorical(subj_arr),
        'position_segment': pos_arr,
        'scratch': np.zeros(total_samples, dtype=np.int8),
        'segment_scratch': np.zeros(total_samples, dtype=bool),
        'scratch_duration': np.zeros(total_samples, dtype=np.float32),
    })

    # Free temporary arrays
    del x_arr, y_arr, z_arr, seg_arr, subj_arr, pos_arr

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"  Loaded {total} segments, {len(df):,} samples in {elapsed:.1f}s")
    print(f"  Unique subjects: {df['SUBJECT'].nunique()}")
    print(f"  Memory: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

    return df, metadata
