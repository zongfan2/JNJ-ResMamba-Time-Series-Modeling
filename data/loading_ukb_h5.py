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
# Length-index cache
# ---------------------------------------------------------------------------
#
# Pass 1 of the old loader did 1.3M HDF5 metadata ops (group lookup + dataset
# open + shape read). On networked Domino storage that's 10+ minutes. The
# result never changes until the H5 is re-preprocessed, so we cache it.
#
# Three tiers, in order of preference:
#
#   1. ``/metadata/segment_lengths`` — a 1-D int64 dataset inside the H5
#      itself. Populated by ``bake_ukb_length_index_into_h5`` (one-shot).
#      Loading it is a single HDF5 read — effectively free.
#
#   2. Sidecar ``<h5_path>.lengths.npz`` — a numpy archive next to the H5
#      that stores ``lengths`` plus the H5's mtime at scan time for cache
#      invalidation. Written automatically the first time tier 3 runs.
#
#   3. Full scan. Slow, but runs at most once per H5 file. Afterwards the
#      sidecar takes over until the H5 is rewritten.
#
# All three tiers return an int64 array of length ``len(f['segments'])``
# covering EVERY segment in the file. The ``indices`` subset passed to
# ``UKBPretrainDataset.__init__`` is applied afterwards as a cheap slice.


def _sidecar_cache_path(h5_path: str) -> str:
    return h5_path + ".lengths.npz"


def _scan_segment_lengths(h5_file: h5py.File, verbose: bool = True) -> np.ndarray:
    """Brute-force scan: open every segment group and read its length."""
    segs_group = h5_file['segments']
    n = len(segs_group)
    lengths = np.empty(n, dtype=np.int64)
    if verbose:
        print(f"  scanning {n:,} segment lengths (first-run only)...")
    start = datetime.now()
    for i in range(n):
        lengths[i] = segs_group[str(i)]['x'].shape[0]
        if verbose and (i + 1) % 100_000 == 0:
            elapsed = (datetime.now() - start).total_seconds()
            rate = (i + 1) / max(elapsed, 1e-6)
            eta = (n - (i + 1)) / max(rate, 1e-6)
            print(f"    scanned {i + 1:,}/{n:,} "
                  f"({rate:,.0f} seg/s, ETA {eta:.0f}s)")
    if verbose:
        elapsed = (datetime.now() - start).total_seconds()
        print(f"  scan complete in {elapsed:.1f}s")
    return lengths


def _load_or_build_length_index(h5_path: str,
                                verbose: bool = True) -> tuple:
    """Return ``(lengths, source)`` for ALL segments in ``h5_path``.

    ``source`` is one of ``"h5-embedded"``, ``"sidecar-cache"``, or
    ``"scan"`` — purely informational so logs make the cache tier visible.
    """
    # Tier 1: embedded dataset inside the H5.
    try:
        with h5py.File(h5_path, 'r') as f:
            if 'metadata' in f and 'segment_lengths' in f['metadata']:
                lengths = np.asarray(f['metadata/segment_lengths'][()],
                                     dtype=np.int64)
                if verbose:
                    print(f"  length index loaded from "
                          f"/metadata/segment_lengths ({len(lengths):,} entries)")
                return lengths, "h5-embedded"
    except Exception as e:
        if verbose:
            print(f"  WARNING: could not probe embedded index: {e}")

    # Tier 2: sidecar .npz cache.
    cache_path = _sidecar_cache_path(h5_path)
    if os.path.exists(cache_path):
        try:
            h5_mtime = os.path.getmtime(h5_path)
            with np.load(cache_path) as cache:
                cached_mtime = float(cache['h5_mtime'][0])
                lengths = np.asarray(cache['lengths'], dtype=np.int64)
            if abs(cached_mtime - h5_mtime) < 1.0:
                if verbose:
                    print(f"  length index loaded from sidecar cache "
                          f"{cache_path} ({len(lengths):,} entries)")
                return lengths, "sidecar-cache"
            elif verbose:
                print(
                    f"  sidecar cache {cache_path} is stale "
                    f"(h5 mtime {h5_mtime:.0f} vs cached {cached_mtime:.0f}); "
                    f"rescanning"
                )
        except Exception as e:
            if verbose:
                print(f"  WARNING: sidecar cache unreadable ({e}); rescanning")

    # Tier 3: scan and write sidecar.
    with h5py.File(h5_path, 'r') as f:
        lengths = _scan_segment_lengths(f, verbose=verbose)

    try:
        h5_mtime = os.path.getmtime(h5_path)
        np.savez(cache_path,
                 lengths=lengths,
                 h5_mtime=np.array([h5_mtime], dtype=np.float64))
        if verbose:
            print(f"  wrote sidecar cache to {cache_path}")
    except Exception as e:
        if verbose:
            print(f"  WARNING: failed to write sidecar cache "
                  f"to {cache_path}: {e}")

    return lengths, "scan"


def bake_ukb_length_index_into_h5(h5_path: str, verbose: bool = True) -> int:
    """Embed per-segment lengths directly into the H5 file.

    Writes an int64 1-D dataset at ``/metadata/segment_lengths``. After this
    has been run once on a given H5 file, ``UKBPretrainDataset`` loads the
    length index in a single HDF5 read (tier 1).

    This opens the H5 in read/write mode, so it must NOT be called while
    training jobs hold read handles on the same file. Run it once after
    ``preprocess_ukb_h5.py`` completes, or as a one-shot maintenance step.

    Args:
        h5_path: Path to the UKB H5 file.
        verbose: Print progress messages.

    Returns:
        Number of segments indexed.
    """
    if verbose:
        print(f"bake_ukb_length_index_into_h5: {h5_path}")

    # Prefer the sidecar cache if it already exists — that saves the scan.
    lengths = None
    source = None
    cache_path = _sidecar_cache_path(h5_path)
    if os.path.exists(cache_path):
        try:
            h5_mtime = os.path.getmtime(h5_path)
            with np.load(cache_path) as cache:
                if abs(float(cache['h5_mtime'][0]) - h5_mtime) < 1.0:
                    lengths = np.asarray(cache['lengths'], dtype=np.int64)
                    source = "sidecar-cache"
        except Exception:
            lengths = None

    if lengths is None:
        with h5py.File(h5_path, 'r') as f:
            lengths = _scan_segment_lengths(f, verbose=verbose)
        source = "scan"

    if verbose:
        print(f"  using length index from {source} "
              f"({len(lengths):,} entries)")

    # Now write it into the H5. Open in append mode to avoid disturbing
    # existing groups.
    with h5py.File(h5_path, 'a') as f:
        if 'metadata' not in f:
            f.create_group('metadata')
        meta = f['metadata']
        if 'segment_lengths' in meta:
            del meta['segment_lengths']
        meta.create_dataset(
            'segment_lengths',
            data=lengths,
            dtype='int64',
            compression='gzip',
            compression_opts=4,
        )
        meta['segment_lengths'].attrs['n_segments'] = int(len(lengths))
        meta['segment_lengths'].attrs['total_samples'] = int(lengths.sum())
        meta['segment_lengths'].attrs['built_at'] = datetime.now().isoformat()

    if verbose:
        print(f"  wrote /metadata/segment_lengths ({len(lengths):,} int64s)")
    return int(len(lengths))


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
        self._segments_group = None  # cached /segments group handle

        # Three-tier length index, fastest to slowest:
        #   1. /metadata/segment_lengths embedded inside the H5  -> instant
        #   2. sidecar .lengths.npz next to the H5               -> <1 s
        #   3. scan all groups and cache to sidecar              -> minutes
        # The cache always covers ALL segments in the file so it stays valid
        # across different train/val index subsets.
        all_lengths, source = _load_or_build_length_index(h5_path,
                                                          verbose=verbose)
        total_in_file = int(all_lengths.shape[0])

        if indices is None:
            indices = np.arange(total_in_file, dtype=np.int64)
        else:
            indices = np.asarray(indices, dtype=np.int64)
            if indices.size and (indices.min() < 0
                                 or indices.max() >= total_in_file):
                raise IndexError(
                    f"indices out of range for H5 with {total_in_file} "
                    f"segments: [{int(indices.min())}, {int(indices.max())}]"
                )

        self.indices = indices
        self.lengths = all_lengths[indices]
        n = len(indices)

        if verbose:
            total_samples = int(self.lengths.sum())
            elapsed = (datetime.now() - start).total_seconds()
            print(
                f"  indexed {n:,} segments "
                f"({total_samples:,} samples) in {elapsed:.1f}s "
                f"via {source}"
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
        state['_segments_group'] = None
        return state

    def _ensure_handle(self):
        """Lazily open the H5 and cache the /segments group handle.

        Caching the group handle saves one dict-like lookup per
        ``__getitem__`` call — cheap individually, but with batch*epochs in
        the millions it adds up to several seconds per epoch.
        """
        if self._h5 is None:
            # Each DataLoader worker opens its own handle. SWMR is not
            # required for read-only multi-process access.
            #
            # rdcc_nbytes: per-dataset chunk cache. The default (1 MB) is
            # tiny relative to our segment sizes; bumping it to 16 MB lets
            # each worker keep recently-read chunks in memory, which helps
            # when segments share underlying chunks on disk.
            self._h5 = h5py.File(
                self.h5_path,
                'r',
                rdcc_nbytes=16 * 1024 * 1024,
                rdcc_nslots=100_003,
            )
            self._segments_group = self._h5['segments']
        return self._segments_group

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> torch.Tensor:
        seg_idx = int(self.indices[i])
        segs = self._ensure_handle()
        g = segs[str(seg_idx)]

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

    Prefers the embedded length-index dataset for a cheap segment count,
    falls back to ``len(f['segments'])`` which walks the group B-tree.

    Args:
        h5_path:      Path to UKB H5 file.
        max_segments: If >0, truncate to the first ``max_segments`` indices.
                      Useful for quick smoke tests.
    """
    with h5py.File(h5_path, 'r') as f:
        # Cheap path: read the count from the embedded index attribute.
        if 'metadata' in f and 'segment_lengths' in f['metadata']:
            ds = f['metadata/segment_lengths']
            total_in_file = int(
                ds.attrs.get('n_segments', ds.shape[0])
            )
        else:
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
