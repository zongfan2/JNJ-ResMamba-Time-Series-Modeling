#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare a UKB parquet folder against the noprod parquet folder to decide whether
"train on UKB (predictTSO only) / test on noprod (has inTSO)" is valid — i.e.
the two share the same signal channels, the right label columns, and compatible
UNITS / value ranges / sampling. The device differs (UKB=Axivity AX3,
noprod=GENEActive), so column names matching is NOT enough; units (g vs mg),
a missing temperature channel, or a different rate would break cross-dataset use.

Run on Domino (data is not local):
    python test-tools/check_ukb_vs_noprod.py \
        --ukb_folder    /mnt/imported/data/NocturnalScratch_Analysis/UKB_v2/raw/ \
        --noprod_folder /path/to/noprod/parquet/ \
        --gt_column inTSO --max_files 20

Exit 0 if cross-dataset training is viable as-is; 1 if there are blockers/warnings.
"""
import argparse
import glob
import os
import sys

import numpy as np

# Columns convert_h5.load_and_preprocess_segment uses.
SIGNAL = ["x", "y", "z", "temperature"]          # -> model channels (+ sin/cos derived)
TRAIN_LABELS = ["predictTSO", "non-wear"]         # needed in the TRAIN set (UKB)
TIME_COL = "timestamp"


def find_files(folder):
    files = []
    for pat in ("*.parquet.gzip", "*.parquet", "*.pq"):
        files.extend(glob.glob(os.path.join(folder, pat)))
    return sorted(set(files))


def columns(path):
    import pyarrow.parquet as pq
    return [str(n) for n in pq.read_schema(path).names]


def inspect_folder(folder, max_files):
    """Return dict: file count, union/intersection of columns, per-signal stats
    (min/max/mean/%nan), rows/file, and an estimated sampling rate (Hz)."""
    import pandas as pd
    files = find_files(folder)
    out = {"folder": folder, "n_files": len(files), "files_ok": 0,
           "cols_all": None, "cols_any": set(), "stats": {}, "rows": [], "rate_hz": None}
    if not files:
        return out
    sample = files[:max_files]
    acc = {c: {"min": np.inf, "max": -np.inf, "sum": 0.0, "n": 0, "nan": 0} for c in SIGNAL}
    rates = []
    for f in sample:
        try:
            cols = set(columns(f))
        except Exception:
            continue
        out["files_ok"] += 1
        out["cols_any"] |= cols
        out["cols_all"] = cols if out["cols_all"] is None else (out["cols_all"] & cols)
        # read only what we need for stats (signal + time), cheaply
        want = [c for c in SIGNAL + [TIME_COL] if c in cols]
        try:
            df = pd.read_parquet(f, columns=want)
        except Exception:
            continue
        out["rows"].append(len(df))
        for c in SIGNAL:
            if c in df.columns:
                v = df[c].to_numpy(dtype="float64", na_value=np.nan)
                finite = v[np.isfinite(v)]
                acc[c]["nan"] += int(np.isnan(v).sum())
                acc[c]["n"] += v.size
                if finite.size:
                    acc[c]["min"] = min(acc[c]["min"], float(finite.min()))
                    acc[c]["max"] = max(acc[c]["max"], float(finite.max()))
                    acc[c]["sum"] += float(finite.sum())
        # sampling rate from timestamp deltas (seconds)
        if TIME_COL in df.columns and len(df) > 100:
            t = pd.to_datetime(df[TIME_COL], errors="coerce").astype("int64", copy=False)
            dt = np.diff(t.to_numpy()) / 1e9  # ns -> s
            dt = dt[(dt > 0) & np.isfinite(dt)]
            if dt.size:
                rates.append(1.0 / float(np.median(dt)))
    for c in SIGNAL:
        a = acc[c]
        if a["n"] and np.isfinite(a["min"]):
            finite_n = a["n"] - a["nan"]
            out["stats"][c] = {"min": a["min"], "max": a["max"],
                               "mean": (a["sum"] / finite_n) if finite_n else float("nan"),
                               "pct_nan": 100.0 * a["nan"] / a["n"]}
    if rates:
        out["rate_hz"] = float(np.median(rates))
    return out


def fmt(x, nd=3):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) and np.isfinite(x) else "—"


def main():
    ap = argparse.ArgumentParser(description="UKB vs noprod parquet compatibility check")
    ap.add_argument("--ukb_folder",
                    default="/mnt/imported/data/NocturnalScratch_Analysis/UKB_v2/raw/")
    ap.add_argument("--noprod_folder",
                    default="/mnt/data/Nocturnal-scratch/geneactive_20hz_3s_b1s_production_train_van_new_enh_lth-rth/raw/")
    ap.add_argument("--gt_column", default="inTSO", help="GT TSO column expected in noprod (eval).")
    ap.add_argument("--max_files", type=int, default=20)
    args = ap.parse_args()

    print("Sampling UKB ...");    ukb = inspect_folder(args.ukb_folder, args.max_files)
    print("Sampling noprod ...");  nop = inspect_folder(args.noprod_folder, args.max_files)
    print()
    for tag, d in (("UKB", ukb), ("noprod", nop)):
        if not d["n_files"]:
            print(f"ERROR: no parquet files under {d['folder']!r} ({tag}).")
            return 2
        print(f"{tag:7s}: {d['n_files']} files, read {d['files_ok']}/{args.max_files} sampled; "
              f"rows/file~{int(np.median(d['rows'])) if d['rows'] else '—'}; "
              f"rate~{fmt(d['rate_hz'],2)} Hz")
    ua, na = ukb["cols_any"], nop["cols_any"]

    print("\n=== columns (presence) ===")
    print(f"  {'column':16s} {'UKB':>6s} {'noprod':>8s}")
    for c in SIGNAL + TRAIN_LABELS + [args.gt_column, "predictTSOSTART", "predictTSOEND", TIME_COL]:
        print(f"  {c:16s} {('yes' if c in ua else 'NO'):>6s} {('yes' if c in na else 'NO'):>8s}")
    only_ukb = sorted(ua - na); only_nop = sorted(na - ua)
    print(f"\n  only in UKB    : {', '.join(only_ukb) if only_ukb else '(none)'}")
    print(f"  only in noprod : {', '.join(only_nop) if only_nop else '(none)'}")

    print("\n=== signal value ranges (units check — device differs!) ===")
    print(f"  {'channel':12s} {'UKB min..max (mean, %nan)':38s} {'noprod min..max (mean, %nan)'}")
    for c in SIGNAL:
        us, ns = ukb["stats"].get(c), nop["stats"].get(c)
        ustr = f"{fmt(us['min'])}..{fmt(us['max'])} ({fmt(us['mean'])}, {fmt(us['pct_nan'],1)}%)" if us else "—"
        nstr = f"{fmt(ns['min'])}..{fmt(ns['max'])} ({fmt(ns['mean'])}, {fmt(ns['pct_nan'],1)}%)" if ns else "—"
        print(f"  {c:12s} {ustr:38s} {nstr}")

    # ---- verdict ----
    blockers, warnings = [], []
    for c in TRAIN_LABELS:
        if c not in ua:
            blockers.append(f"UKB missing train label '{c}' (cannot train).")
    for c in ("x", "y", "z"):
        if c not in ua: blockers.append(f"UKB missing signal '{c}'.")
        if c not in na: blockers.append(f"noprod missing signal '{c}'.")
    if args.gt_column not in na:
        blockers.append(f"noprod missing GT column '{args.gt_column}' (cannot evaluate vs inTSO).")
    # temperature channel mismatch (model uses 6 channels incl. temperature)
    if ("temperature" in ua) != ("temperature" in na):
        warnings.append("temperature present in only ONE dataset -> 6-channel mismatch. "
                        "Either drop temperature (retrain both) or zero-fill the missing side.")
    elif "temperature" not in ua and "temperature" not in na:
        warnings.append("temperature absent in BOTH -> fine, but build the H5 as 5-channel/zeroed consistently.")
    # accel UNIT mismatch (g vs mg): compare |max| magnitude of x
    ux, nx = ukb["stats"].get("x"), nop["stats"].get("x")
    if ux and nx:
        umag = max(abs(ux["min"]), abs(ux["max"])); nmag = max(abs(nx["min"]), abs(nx["max"]))
        if nmag > 0 and (umag / nmag > 8 or nmag / umag > 8):
            warnings.append(f"accel 'x' magnitude differs ~{umag/nmag:.0f}x (UKB |x|max~{fmt(umag,1)} "
                            f"vs noprod~{fmt(nmag,1)}) -> likely a UNIT mismatch (g vs mg). Rescale before H5.")
    # sampling rate (convert_h5 resamples to 20 Hz, so this is informational)
    if ukb["rate_hz"] and nop["rate_hz"] and abs(ukb["rate_hz"] - nop["rate_hz"]) > 1.0:
        warnings.append(f"native sampling rates differ (UKB~{fmt(ukb['rate_hz'],1)}Hz vs "
                        f"noprod~{fmt(nop['rate_hz'],1)}Hz). convert_h5 resamples to 20Hz, so usually OK.")

    print("\n" + "=" * 70)
    if blockers:
        print("NOT compatible as-is — BLOCKERS:")
        for b in blockers: print(f"  [X] {b}")
    else:
        print("Core requirements OK: UKB has predictTSO+non-wear+x,y,z; "
              f"noprod has x,y,z+{args.gt_column}.")
    if warnings:
        print("Warnings to resolve before/at H5 build:")
        for w in warnings: print(f"  [!] {w}")
    if not blockers and not warnings:
        print("\nFully compatible: train-UKB / test-noprod can proceed with the same H5 build.")
    return 1 if (blockers or warnings) else 0


if __name__ == "__main__":
    sys.exit(main())
