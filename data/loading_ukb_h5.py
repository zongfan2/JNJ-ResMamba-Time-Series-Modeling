# -*- coding: utf-8 -*-
"""
UKB pretraining data loader.

The UKB H5 file produced by ``data/preprocess_ukb_h5.py`` already stores each
accelerometer window as its own group — ``f['segments'][str(i)]`` with ``x``,
``y``, ``z`` datasets and ``segment`` / ``SUBJECT`` attributes. There are no
labels, no aggregation, and no cross-segment operations needed for DINO
pretraining.

The previous implementation flattened every segment into a single giant
pandas DataFrame with a Categorical ``segment`` column so it could reuse the
JNJ ``batch_generator`` / ``add_padding_pretrain`` pipeline. That pipeline was
designed for CSV/Parquet inputs where per-sample rows must be grouped back
into segments — it does not fit UKB, and its Categorical/groupby internals
were the root cause of the recent OOM bugs.

This module replaces that path with a straight ``torch.utils.data.Dataset``
that reads segments from H5 lazily (one h5py handle per DataLoader worker)
and a ``collate_ukb_batch`` that pads a batch into ``[B, L, 3]`` with an
explicit length tensor. No DataFrame, no Categorical, no groupby.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional, Sequence

import h5py
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class UKBPretrainDataset(Dataset):
    """Lazy UKB pretraining dataset backed by an H5 file.

    Each item is a float32 tensor ``[L_i, 3]`` of (x, y, z) acceleration for
    one segment, optionally z-score normalised with a pre-fitted scaler.

    The H5 file is opened lazily per worker inside ``__getitem__`` so that
    ``torch.utils.data.DataLoader(num_workers>0)`` works correctly. Only a
    small in-memory index (segment lengths) is built in ``__init__`` — no
    signal data is loaded eagerly.
    """

    def __init__(
        self,
        h5_path: str,
        indices: Optional[Sequence[int]] = None,
        scaler=None,
        max_seq_len: Optional[int] = None,
        dtype: np.dtype = np.float32,
        verbose: bool = True,
    ):
        """
        Args:
            h5_path:      Path to UKB H5 file produced by preprocess_ukb_h5.py.
            indices:      Optional list of segment indices to include. If
                          None, every segment in the H5 is used. Used to
                          implement train/val splits without copying data.
            scaler:       Optional pre-fitted ``sklearn`` StandardScaler-like
                          object with ``.transform()``. Applied per segment
                          inside ``__getitem__``. If ``None``, raw (x, y, z)
                          are returned unchanged.
            max_seq_len:  If set, any segment longer than this is truncated
                          to ``max_seq_len`` samples inside ``__getitem__``.
                          This is a cheap safety cap; padding to the batch
                          max is handled by ``collate_ukb_batch``.
            dtype:        Output numpy dtype (float32 recommended).
            verbose:      Print index-build progress.
        """
        super().__init__()
        self.h5_path = h5_path
        self.scaler = scaler
        self.max_seq_len = max_seq_len
        self.dtype = dtype

        # Sanity-check the scaler shape once, cheaply.
        if scaler is not None:
            n_expected = getattr(scaler, 'n_features_in_', None)
            if n_expected is not None and n_expected != 3:
                raise ValueError(
                    f"UKB pretraining uses 3 channels (x, y, z) but scaler "
                    f"was fitted on {n_expected} features."
                )

        if verbose:
            print(f"UKBPretrainDataset: opening {h5_path}")
        start = datetime.now()

        # Worker-local h5py handle cache. Populated lazily in __getitem__ so
        # that handles are NOT shared across DataLoader worker processes.
        self._h5: Optional[h5py.File] = None

        # Build a lightweight index. Only segment lengths are cached — at
        # ~8 bytes per segment this is <10 MB for 1.3M segments.
        with h5py.File(h5_path, 'r') as f:
            segs_group = f['segments']
            total_in_file = len(segs_group)

            if indices is None:
                indices = np.arange(total_in_file, dtype=np.int64)
            else:
                indices = np.asarray(indices, dtype=np.int64)

            self.indices = indices
            n = len(indices)
            self.lengths = np.empty(n, dtype=np.int64)
            for i, seg_idx in enumerate(indices):
                self.lengths[i] = segs_group[str(int(seg_idx))]['x'].shape[0]
                if verbose and (i + 1) % 500_000 == 0:
                    print(f"  indexed {i + 1:,}/{n:,} segments")

        if verbose:
            total_samples = int(self.lengths.sum())
            elapsed = (datetime.now() - start).total_seconds()
            print(
                f"  indexed {n:,} segments "
                f"({total_samples:,} samples) in {elapsed:.1f}s"
            )
            print(
                f"  length stats: "
                f"min={int(self.lengths.min())}, "
                f"max={int(self.lengths.max())}, "
                f"mean={self.lengths.mean():.1f}, "
                f"median={int(np.median(self.lengths))}"
            )

    # -- pickle safety: drop the open h5 handle so workers re-open fresh ----
    def __getstate__(self):
        state = self.__dict__.copy()
        state['_h5'] = None
        return state

    def _ensure_handle(self) -> h5py.File:
        if self._h5 is None:
            # Each DataLoader worker opens its own handle. SWMR is not
            # required for read-only multi-process access.
            self._h5 = h5py.File(self.h5_path, 'r')
        return self._h5

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> torch.Tensor:
        seg_idx = int(self.indices[i])
        h5 = self._ensure_handle()
        g = h5['segments'][str(seg_idx)]

        # Read raw channels into a [L, 3] float32 array. Reading the three
        # datasets separately matches how the H5 was written; stacking once
        # here avoids any intermediate pandas allocation.
        x = np.asarray(g['x'][()], dtype=self.dtype)
        y = np.asarray(g['y'][()], dtype=self.dtype)
        z = np.asarray(g['z'][()], dtype=self.dtype)
        arr = np.stack([x, y, z], axis=1)  # [L, 3]

        if self.max_seq_len is not None and arr.shape[0] > self.max_seq_len:
            arr = arr[: self.max_seq_len]

        if self.scaler is not None:
            # StandardScaler.transform expects 2-D; arr is already [L, 3].
            arr = self.scaler.transform(arr).astype(self.dtype, copy=False)

        return torch.from_numpy(arr)


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def collate_ukb_batch(batch, max_seq_len: Optional[int] = None,
                      padding_value: float = 0.0):
    """Pad a list of ``[L_i, 3]`` tensors into ``[B, L_max, 3]``.

    Args:
        batch:         List of tensors returned by ``UKBPretrainDataset``.
        max_seq_len:   Optional cap — the padded length is
                       ``min(max(L_i), max_seq_len)``. Any segment longer
                       than ``max_seq_len`` is truncated.
        padding_value: Fill value for padding positions.

    Returns:
        padded:  ``FloatTensor [B, L_max, 3]``
        lengths: ``LongTensor  [B]`` — original (post-truncation) lengths
    """
    # Truncate individual items first so pad_sequence never allocates more
    # than max_seq_len along the time dimension.
    if max_seq_len is not None:
        batch = [
            t[:max_seq_len] if t.shape[0] > max_seq_len else t
            for t in batch
        ]

    lengths = torch.tensor([t.shape[0] for t in batch], dtype=torch.long)
    padded = pad_sequence(batch, batch_first=True,
                          padding_value=padding_value)  # [B, L_max, 3]

    # Optional hard cap on the padded length — redundant given the
    # per-item truncation above, but cheap and keeps the contract explicit.
    if max_seq_len is not None and padded.shape[1] > max_seq_len:
        padded = padded[:, :max_seq_len, :]
        lengths = torch.clamp(lengths, max=max_seq_len)

    return padded, lengths


def make_ukb_collate_fn(max_seq_len: Optional[int] = None,
                        padding_value: float = 0.0):
    """Return a closure suitable for ``DataLoader(collate_fn=...)``."""

    def _fn(batch):
        return collate_ukb_batch(batch, max_seq_len=max_seq_len,
                                 padding_value=padding_value)

    return _fn


# ---------------------------------------------------------------------------
# Index helpers
# ---------------------------------------------------------------------------

def read_ukb_h5_metadata(h5_path: str) -> dict:
    """Read the ``/metadata`` attribute block from a UKB H5 file."""
    meta = {}
    with h5py.File(h5_path, 'r') as f:
        if 'metadata' in f:
            for key, val in f['metadata'].attrs.items():
                meta[key] = val
    return meta


def list_ukb_segment_indices(h5_path: str,
                             max_segments: int = 0) -> np.ndarray:
    """Return a numpy array of segment indices to use.

    Args:
        h5_path:      Path to UKB H5 file.
        max_segments: If >0, truncate to the first ``max_segments`` indices.
                      Useful for quick smoke tests.
    """
    with h5py.File(h5_path, 'r') as f:
        total_in_file = len(f['segments'])
    total = (
        min(total_in_file, max_segments) if max_segments > 0 else total_in_file
    )
    return np.arange(total, dtype=np.int64)


def split_indices_by_segment(indices: np.ndarray, val_fraction: float = 0.1,
                             seed: int = 42):
    """Split segment indices into train / val at the segment level.

    Segments are independent units for DINO, so a simple random split is
    sufficient — no need for subject-aware splitting during pretraining.
    """
    from sklearn.model_selection import train_test_split
    train_idx, val_idx = train_test_split(
        indices, test_size=val_fraction, random_state=seed, shuffle=True
    )
    return np.asarray(train_idx), np.asarray(val_idx)
