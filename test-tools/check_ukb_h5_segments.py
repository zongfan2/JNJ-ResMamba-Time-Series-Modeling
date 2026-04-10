# -*- coding: utf-8 -*-
"""
Diagnostic script: scan a UKB pretraining H5 file and flag suspiciously
long segments.

The DINO pretraining pipeline expects per-segment lengths on the order of
~1221 samples (61s @ 20 Hz). Earlier runs hit CUDA OOMs consistent with
segments containing tens of millions of samples — strongly suggesting a
bug in the preprocessing script that wrote the H5 (e.g., missed a segment
boundary, concatenated an entire subject recording into one segment, or
used the wrong sampling rate).

What this script does
---------------------
1. Opens the H5 file and walks every segment in f['segments'].
2. Collects per-segment length, subject ID, and segment ID.
3. Prints length distribution stats (min / max / mean / percentiles).
4. Flags segments longer than several thresholds:
     - PROD_CAP (1221)     : anything above this gets truncated by the loader
     - MODERATE (5000)     : ~4x prod; unusual but possible if resampled
     - LARGE    (100_000)  : almost certainly a preprocessing bug
     - HUGE     (1_000_000): definitely a preprocessing bug
5. Prints the top-N longest segments with their subject/segment IDs.
6. Saves the raw (x, y, z) data of the top-N worst offenders to CSVs so
   you can eyeball what preprocessing produced.

Usage
-----
    python3 test-tools/check_ukb_h5_segments.py \\
        --h5 /mnt/data/GENEActive-featurized/results/DL/UKB_v2/ukb_pretrain_20hz.h5 \\
        --output-dir /mnt/data/.../diagnostics/ \\
        --top-n 5 \\
        --max-rows-per-csv 200000

The --max-rows-per-csv flag is a safety cap so a single suspicious segment
doesn't produce a multi-GB CSV file.
"""

import argparse
import os
from datetime import datetime

import h5py
import numpy as np
import pandas as pd


# --- Thresholds (samples) ---------------------------------------------------
PROD_CAP = 1221        # production sequence length cap used during training
MODERATE = 5_000       # ~4x prod cap: unusual but plausible
LARGE    = 100_000     # definitely suspicious; ~83 min @ 20Hz
HUGE     = 1_000_000   # absolutely broken; ~14 hours @ 20Hz in one segment


def parse_args():
    p = argparse.ArgumentParser(
        description="Scan a UKB pretraining H5 file for suspicious segments.")
    p.add_argument("--h5", required=True,
                   help="Path to the UKB pretrain H5 file.")
    p.add_argument("--output-dir", default=None,
                   help="Directory to save example CSVs and the length "
                        "report. Defaults to <h5-parent>/diagnostics/.")
    p.add_argument("--top-n", type=int, default=5,
                   help="Number of worst-offender segments to dump to CSV.")
    p.add_argument("--max-rows-per-csv", type=int, default=200_000,
                   help="Safety cap on rows per example CSV. If a segment "
                        "has more rows, only the first N are written.")
    p.add_argument("--max-segments", type=int, default=0,
                   help="Limit the scan to the first N segments (0 = all). "
                        "Useful for a quick smoke test.")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.h5):
        raise FileNotFoundError(f"H5 file not found: {args.h5}")

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.h5), "diagnostics")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Scanning H5 file: {args.h5}")
    print(f"Output directory: {args.output_dir}")
    print(f"Started at:       {datetime.now()}")
    print("-" * 70)

    start = datetime.now()

    with h5py.File(args.h5, "r") as f:
        # --- Metadata ---
        if "metadata" in f:
            print("H5 metadata:")
            for key, val in f["metadata"].attrs.items():
                print(f"  {key}: {val}")
            print()

        if "segments" not in f:
            raise KeyError("H5 file has no top-level 'segments' group.")

        segs_group = f["segments"]
        total_in_file = len(segs_group)
        total = (min(total_in_file, args.max_segments)
                 if args.max_segments > 0 else total_in_file)
        print(f"Total segments in file: {total_in_file:,}")
        print(f"Scanning:               {total:,}")
        print()

        # --- Pass 1: read lengths and identifying metadata ---
        print("Pass 1: reading segment lengths...")
        lengths   = np.empty(total, dtype=np.int64)
        seg_ids   = np.empty(total, dtype=object)
        subj_ids  = np.empty(total, dtype=object)
        index_ids = np.empty(total, dtype=np.int64)  # position in h5

        for i in range(total):
            g = segs_group[str(i)]
            lengths[i]   = g["x"].shape[0]
            seg_ids[i]   = g.attrs.get("segment", f"<missing_{i}>")
            subj_ids[i]  = g.attrs.get("SUBJECT", "<missing>")
            index_ids[i] = i

            if (i + 1) % 200_000 == 0:
                elapsed = (datetime.now() - start).total_seconds()
                print(f"  scanned {i + 1:,}/{total:,} segments ({elapsed:.0f}s)")

        elapsed = (datetime.now() - start).total_seconds()
        print(f"  done in {elapsed:.1f}s")
        print()

        # --- Length distribution ---
        print("Segment length statistics:")
        print(f"  count:  {total:,}")
        print(f"  min:    {int(lengths.min()):,}")
        print(f"  max:    {int(lengths.max()):,}")
        print(f"  mean:   {lengths.mean():,.1f}")
        print(f"  median: {int(np.median(lengths)):,}")
        print(f"  std:    {lengths.std():,.1f}")
        for p in (50, 75, 90, 95, 99, 99.9, 99.99):
            print(f"  p{p:>5}: {int(np.percentile(lengths, p)):,}")
        print()

        # --- Threshold counts ---
        print("Threshold flags:")
        for name, threshold in [
            ("PROD_CAP", PROD_CAP),
            ("MODERATE", MODERATE),
            ("LARGE",    LARGE),
            ("HUGE",     HUGE),
        ]:
            mask = lengths > threshold
            n = int(mask.sum())
            pct = 100.0 * n / total
            print(f"  > {threshold:>10,} ({name:8s}): "
                  f"{n:>10,} segments  ({pct:6.3f}%)")
        print()

        # --- Top-N worst offenders ---
        top_n = min(args.top_n, total)
        # argpartition for efficiency on large arrays, then sort those N
        top_idx = np.argpartition(-lengths, top_n - 1)[:top_n]
        top_idx = top_idx[np.argsort(-lengths[top_idx])]

        print(f"Top {top_n} longest segments:")
        print(f"  {'rank':>4}  {'h5_index':>10}  {'length':>12}  "
              f"{'subject':<24}  {'segment_id'}")
        for rank, idx in enumerate(top_idx, start=1):
            print(f"  {rank:>4}  {int(index_ids[idx]):>10,}  "
                  f"{int(lengths[idx]):>12,}  {str(subj_ids[idx]):<24}  "
                  f"{seg_ids[idx]}")
        print()

        # --- Build and save length report ---
        report_path = os.path.join(args.output_dir, "segment_length_report.csv")
        report_df = pd.DataFrame({
            "h5_index":   index_ids,
            "subject":    subj_ids,
            "segment_id": seg_ids,
            "length":     lengths,
        })
        # Sort by length descending so the report itself surfaces problems.
        report_df = report_df.sort_values("length", ascending=False)
        report_df.to_csv(report_path, index=False)
        print(f"Full length report saved: {report_path}")
        print(f"  ({len(report_df):,} rows, one per segment)")
        print()

        # --- Dump raw data of top-N longest segments ---
        print(f"Dumping raw (x, y, z) data for top {top_n} segments...")
        for rank, idx in enumerate(top_idx, start=1):
            g = segs_group[str(int(index_ids[idx]))]
            n_full = int(lengths[idx])
            n_save = min(n_full, args.max_rows_per_csv)

            x = g["x"][:n_save]
            y = g["y"][:n_save]
            z = g["z"][:n_save]

            sample_df = pd.DataFrame({
                "sample_index": np.arange(n_save, dtype=np.int64),
                "x": x,
                "y": y,
                "z": z,
            })
            # Metadata columns for context
            sample_df["subject"]     = str(subj_ids[idx])
            sample_df["segment_id"]  = str(seg_ids[idx])
            sample_df["h5_index"]    = int(index_ids[idx])
            sample_df["full_length"] = n_full
            sample_df["saved_rows"]  = n_save
            sample_df["truncated"]   = n_full > n_save

            # Safe filename (segment_id may contain slashes / spaces)
            safe_seg = str(seg_ids[idx]).replace("/", "_").replace(" ", "_")
            fname = (f"rank{rank:02d}_len{n_full}_"
                     f"subj{subj_ids[idx]}_seg{safe_seg}.csv")
            out_path = os.path.join(args.output_dir, fname)
            sample_df.to_csv(out_path, index=False)

            trunc_note = (f"  [truncated from {n_full:,}]"
                          if n_full > n_save else "")
            print(f"  rank {rank}: {n_save:,} rows{trunc_note}")
            print(f"    -> {out_path}")
        print()

    # --- Final verdict ---
    print("=" * 70)
    n_over_prod = int((lengths > PROD_CAP).sum())
    n_over_huge = int((lengths > HUGE).sum())
    if n_over_huge > 0:
        print(f"VERDICT: BROKEN. {n_over_huge:,} segments exceed {HUGE:,} "
              f"samples. This is almost certainly a bug in "
              f"data/preprocess_ukb_h5.py — check segment boundary logic "
              f"and sampling rate assumptions.")
    elif n_over_prod > 0:
        pct = 100.0 * n_over_prod / total
        print(f"VERDICT: {n_over_prod:,} ({pct:.2f}%) segments exceed the "
              f"production cap of {PROD_CAP}. These will be silently "
              f"truncated by the loader, which is safe but wastes data.")
    else:
        print(f"VERDICT: HEALTHY. All segments fit within the production "
              f"cap of {PROD_CAP} samples.")
    print("=" * 70)
    print(f"Finished at: {datetime.now()}")


if __name__ == "__main__":
    main()
