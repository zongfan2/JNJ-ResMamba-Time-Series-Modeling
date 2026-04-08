# -*- coding: utf-8 -*-
"""
Load preprocessed UKB H5 data for self-supervised pretraining.

The H5 file is produced by data/preprocess_ukb_h5.py and stores variable-length
(x, y, z) segments. This loader returns a DataFrame compatible with the
batch_generator / add_padding pipeline used in train_scratch.py.
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

    frames = []
    metadata = {}

    with h5py.File(file_path, 'r') as f:
        # Read metadata
        for key, val in f['metadata'].attrs.items():
            metadata[key] = val

        total = len(f['segments'])
        if max_segments > 0:
            total = min(total, max_segments)

        print(f"  Total segments in H5: {len(f['segments'])}, loading: {total}")

        for i in range(total):
            g = f['segments'][str(i)]
            x = g['x'][()]
            y = g['y'][()]
            z = g['z'][()]
            n = len(x)
            segment_name = g.attrs['segment']
            subject = g.attrs['SUBJECT']

            seg_df = pd.DataFrame({
                'x': x,
                'y': y,
                'z': z,
                'segment': segment_name,
                'SUBJECT': subject,
                'position_segment': np.arange(1, n + 1),
                # Dummy labels for pretraining (no scratch annotations in UKB)
                'scratch': np.zeros(n, dtype=np.int8),
                'segment_scratch': np.zeros(n, dtype=bool),
                'scratch_duration': np.zeros(n, dtype=np.float32),
            })
            frames.append(seg_df)

            if (i + 1) % 1000 == 0:
                print(f"  Loaded {i + 1}/{total} segments...")

    df = pd.concat(frames, ignore_index=True)
    elapsed = (datetime.now() - start_time).total_seconds()

    print(f"  Loaded {total} segments, {len(df):,} samples in {elapsed:.1f}s")
    print(f"  Unique subjects: {df['SUBJECT'].nunique()}")
    print(f"  Memory: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

    return df, metadata
