#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check which columns a folder of (UKB) parquet files contains, before building a
Deep TSO training H5 with training/convert_h5.py.

WHY: convert_h5.py silently writes all-zero labels when `predictTSO` / `non-wear`
are missing, producing a label-free H5 that trains to "no TSO". This scans a
sample of files and reports whether the required label + signal columns are
present, plus any candidate multi-annotator columns (for Phase-2 consensus).

Usage:
    python test-tools/check_parquet_columns.py \
        --input_folder /mnt/imported/data/NocturnalScratch_Analysis/UKB_v2/raw/ \
        --max_files 25

Exit code: 0 if predictTSO AND non-wear are present in every sampled file,
else 1 (so it can gate a pipeline).
"""

import argparse
import glob
import os
import sys

# Columns convert_h5.load_and_preprocess_segment relies on.
REQUIRED_LABELS = ["predictTSO", "non-wear"]
SIGNAL_COLUMNS = ["x", "y", "z", "temperature", "timestamp", "wrist"]
TSO_WINDOW_COLUMNS = ["predictTSOSTART", "predictTSOEND"]
# Substrings that hint at additional traditional-algorithm tracks (Phase-2).
ANNOTATOR_HINTS = ["sadeh", "cole", "kripke", "vanhees", "van_hees", "hdcza",
                   "oakley", "scripps", "annot"]


def get_columns(path):
    """Return the column names of a parquet file as cheaply as possible."""
    try:
        import pyarrow.parquet as pq
        return list(pq.read_schema(path))  # ParquetFile schema -> field names
    except Exception:
        import pandas as pd
        return list(pd.read_parquet(path).columns)


def find_files(folder):
    files = []
    for pat in ("*.parquet.gzip", "*.parquet", "*.pq"):
        files.extend(glob.glob(os.path.join(folder, pat)))
    return sorted(set(files))


def main():
    parser = argparse.ArgumentParser(description="Inspect parquet columns for Deep TSO conversion")
    parser.add_argument("--input_folder", required=True, help="Folder of parquet files")
    parser.add_argument("--max_files", type=int, default=25, help="How many files to sample")
    args = parser.parse_args()

    files = find_files(args.input_folder)
    if not files:
        print(f"ERROR: no parquet files found under {args.input_folder!r}")
        return 2

    sample = files[: args.max_files]
    print(f"Found {len(files)} parquet files; sampling {len(sample)}.\n")

    present_in_all = None
    present_in_any = set()
    failures = []
    for f in sample:
        try:
            cols = set(get_columns(f))
        except Exception as e:
            failures.append((os.path.basename(f), str(e)))
            continue
        present_in_any |= cols
        present_in_all = cols if present_in_all is None else (present_in_all & cols)

    if present_in_all is None:
        print("ERROR: could not read any sampled file:")
        for name, err in failures:
            print(f"  {name}: {err}")
        return 2

    def status(col):
        if col in present_in_all:
            return "ALL"
        if col in present_in_any:
            return "SOME"
        return "MISSING"

    print("Required label columns (convert_h5 -> Y):")
    for c in REQUIRED_LABELS:
        print(f"  {c:18s} {status(c)}")
    print("\nSignal columns:")
    for c in SIGNAL_COLUMNS:
        print(f"  {c:18s} {status(c)}")
    print("\nTSO-window columns (informational):")
    for c in TSO_WINDOW_COLUMNS:
        print(f"  {c:18s} {status(c)}")

    annotator_cols = sorted(
        c for c in present_in_any
        if any(h in c.lower() for h in ANNOTATOR_HINTS)
    )
    print("\nCandidate annotator columns (Phase-2 --annotator_columns):")
    print("  " + (", ".join(annotator_cols) if annotator_cols else "(none detected)"))

    if failures:
        print(f"\nWARNING: {len(failures)} file(s) could not be read (e.g. {failures[0][0]}).")

    labels_ok = all(c in present_in_all for c in REQUIRED_LABELS)
    print("\n" + "=" * 60)
    if labels_ok:
        print("OK: predictTSO and non-wear present in all sampled files.")
        print("    convert_h5.py will produce real per-minute labels.")
        return 0
    missing = [c for c in REQUIRED_LABELS if c not in present_in_all]
    print(f"WARNING: missing/partial label column(s): {missing}")
    print("    convert_h5.py will fall back to ALL-ZERO labels for those,")
    print("    yielding a label-free H5. Fix the source data before training.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
