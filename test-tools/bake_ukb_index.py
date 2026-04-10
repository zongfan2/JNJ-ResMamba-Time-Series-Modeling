#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-shot maintenance tool: bake the UKB segment length index into an H5 file.

Pass 1 of ``UKBPretrainDataset`` has to know the length of every segment so
it can build batches and compute stats. The cheapest way to provide that is
to store the lengths as a 1-D int64 dataset at ``/metadata/segment_lengths``
inside the H5 file itself — after that, every training job loads the entire
index in a single HDF5 read (~10 ms for 1.3M segments) instead of walking
1.3M groups.

This script is idempotent: re-running it overwrites the embedded dataset.
It is safe against the sidecar cache — if a fresh sidecar already exists,
it's reused so we don't pay the scan cost twice.

The H5 file must NOT be open for writing anywhere else while this runs.

Usage:
    python3 test-tools/bake_ukb_index.py /path/to/ukb_pretrain_20hz.h5

    # Force a rescan even if the sidecar cache looks fresh:
    python3 test-tools/bake_ukb_index.py /path/to/ukb.h5 --force-rescan

    # Quiet mode (errors only):
    python3 test-tools/bake_ukb_index.py /path/to/ukb.h5 --quiet

Exit codes:
    0   success
    1   user error (bad path, unreadable file, etc.)
    2   H5-level error (file corrupt, read-only, missing /segments, ...)
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime


# Make the repo importable regardless of where the script is invoked from.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Bake the per-segment length index into a UKB pretraining H5 "
            "file as /metadata/segment_lengths. Run once after "
            "preprocess_ukb_h5.py so every future training job skips the "
            "slow Pass 1 scan."
        )
    )
    p.add_argument(
        "h5_path",
        type=str,
        help="Path to the UKB pretraining H5 file.",
    )
    p.add_argument(
        "--force-rescan",
        action="store_true",
        help=(
            "Delete the sidecar .lengths.npz first so the scan runs from "
            "scratch. Use this if you suspect the cached lengths are wrong."
        ),
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output (errors are still printed).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    verbose = not args.quiet

    h5_path = os.path.abspath(args.h5_path)

    if not os.path.exists(h5_path):
        print(f"error: H5 file not found: {h5_path}", file=sys.stderr)
        return 1
    if not os.access(h5_path, os.R_OK):
        print(f"error: H5 file not readable: {h5_path}", file=sys.stderr)
        return 1
    if not os.access(h5_path, os.W_OK):
        print(
            f"error: H5 file is not writable: {h5_path}\n"
            f"       The bake step opens the file in append mode to add\n"
            f"       /metadata/segment_lengths. Move the file to a writable\n"
            f"       location or run from an account with write access.",
            file=sys.stderr,
        )
        return 1

    # Optional: force a fresh scan by deleting the sidecar cache first.
    if args.force_rescan:
        try:
            from data.loading_ukb_h5 import _sidecar_cache_path
        except ImportError as e:
            print(f"error: could not import UKB loader: {e}", file=sys.stderr)
            return 1
        cache_path = _sidecar_cache_path(h5_path)
        if os.path.exists(cache_path):
            if verbose:
                print(f"removing stale sidecar: {cache_path}")
            try:
                os.remove(cache_path)
            except OSError as e:
                print(
                    f"warning: could not remove sidecar {cache_path}: {e}",
                    file=sys.stderr,
                )

    try:
        from data.loading_ukb_h5 import bake_ukb_length_index_into_h5
    except ImportError as e:
        print(
            f"error: could not import bake_ukb_length_index_into_h5: {e}\n"
            f"       Make sure PROJECT_ROOT={_PROJECT_ROOT} is correct and\n"
            f"       that data/loading_ukb_h5.py is up to date.",
            file=sys.stderr,
        )
        return 1

    if verbose:
        print("=" * 60)
        print(" bake_ukb_index")
        print("=" * 60)
        print(f"  h5_path:      {h5_path}")
        print(f"  force rescan: {args.force_rescan}")
        print(f"  started at:   {datetime.now().isoformat(timespec='seconds')}")
        print("-" * 60)

    start = datetime.now()
    try:
        n = bake_ukb_length_index_into_h5(h5_path, verbose=verbose)
    except KeyError as e:
        print(
            f"error: H5 file is missing an expected group or dataset: {e}\n"
            f"       Expected /segments group populated by "
            f"preprocess_ukb_h5.py.",
            file=sys.stderr,
        )
        return 2
    except OSError as e:
        print(
            f"error: H5 I/O error while baking index: {e}",
            file=sys.stderr,
        )
        return 2

    elapsed = (datetime.now() - start).total_seconds()

    if verbose:
        print("-" * 60)
        print(f"  indexed {n:,} segments in {elapsed:.1f}s")
        print(f"  wrote /metadata/segment_lengths to {h5_path}")
        print(
            "  future UKBPretrainDataset opens will hit the "
            "'h5-embedded' tier."
        )
        print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
