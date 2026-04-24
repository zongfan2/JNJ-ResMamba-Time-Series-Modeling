# -*- coding: utf-8 -*-
"""tsfresh feature extractor — Xing et al. 2024 (MDPI Sensors 24(11):3364).

Paper 2's feature-engineering pipeline: `tsfresh.ComprehensiveFCParameters`
computed on x/y/z + FFT (frequency-domain) features, producing ~2353
features per 3-s window, then z-scored.  We mirror that here:

  * Group the input DataFrame by `segment` (our 3-s equivalent).
  * For each segment, stack x/y/z into a long-format tsfresh frame
    (columns: ``id``, ``time``, ``value``).
  * Run ``extract_features`` with ``ComprehensiveFCParameters`` per channel.
  * Optionally add FFT magnitude features on the three axes (paper step).
  * Impute non-finite values with the per-column training median
    (``median_value_imputation`` in the paper's wording).

Output matches :func:`baselines_classical.features.extract_features_batch`
so it slots into ``training/train_classical.py`` without reshaping.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Lazy imports so projects that never touch mdpi2024_fe don't need tsfresh.
# ---------------------------------------------------------------------------


def _try_import_tsfresh():
    try:
        from tsfresh import extract_features as _extract
        from tsfresh.feature_extraction import ComprehensiveFCParameters
        return _extract, ComprehensiveFCParameters
    except ImportError:
        pass
    import subprocess
    import sys
    print("[mdpi2024_fe] tsfresh not found — attempting `pip install tsfresh` ...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "tsfresh"],
            check=True,
        )
        from tsfresh import extract_features as _extract
        from tsfresh.feature_extraction import ComprehensiveFCParameters
        return _extract, ComprehensiveFCParameters
    except Exception as exc:
        print(f"[mdpi2024_fe] tsfresh install failed: {exc}")
        return None, None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fft_magnitude_stats(seq: np.ndarray) -> dict:
    """Lightweight FFT features, named to avoid colliding with tsfresh output."""
    if seq.shape[0] < 4:
        return {"fft_peak_freq": 0.0, "fft_peak_mag": 0.0, "fft_mean_mag": 0.0}
    from scipy.fft import rfft
    mag = np.abs(rfft(seq - seq.mean()))
    if mag.size <= 1:
        return {"fft_peak_freq": 0.0, "fft_peak_mag": 0.0, "fft_mean_mag": 0.0}
    peak = int(np.argmax(mag[1:])) + 1
    return {
        "fft_peak_freq": float(peak),
        "fft_peak_mag": float(mag[peak]),
        "fft_mean_mag": float(mag.mean()),
    }


# ---------------------------------------------------------------------------
# Batch extractor — matches extract_features_batch's return tuple
# ---------------------------------------------------------------------------


def extract_tsfresh_features_batch(
    df: pd.DataFrame,
    seg_col: str = "segment",
    xyz_cols=("x", "y", "z"),
    n_jobs: int = 0,
    disable_progressbar: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Return (X, y, dur, seg_ids, feature_names) for the Xing 2024 FE pipeline.

    Args:
        df: frame-level DataFrame with ``segment``, ``x``, ``y``, ``z``,
            ``segment_scratch``, ``scratch_duration`` columns.
        seg_col: segment-id column name.
        xyz_cols: the three accelerometer columns.
        n_jobs: tsfresh parallelism (0 = off, which is usually fastest for
                small segments because of process-startup overhead).

    Returns:
        X: [N_segments, F] feature matrix (F ≈ 2350+ depending on tsfresh).
        y: [N_segments] binary scratch label.
        dur: [N_segments] scratch duration.
        seg_ids: [N_segments] str ids (aligned with X rows).
        feature_names: list[str] of length F (column names).
    """
    tsfresh_extract, ComprehensiveFCParameters = _try_import_tsfresh()
    if tsfresh_extract is None:
        raise ImportError(
            "mdpi2024_fe baseline requires tsfresh. Install via `pip install tsfresh`."
        )

    # Build long-format tsfresh frame, one row per (segment, channel, time).
    # tsfresh expects id = unique-per-series, so we use f"{seg}__{axis}".
    long_parts = []
    y_list, dur_list, seg_list = [], [], []
    for seg_id, seq in df.groupby(seg_col, sort=False, observed=True):
        if len(seq) < 4:
            continue
        for axis in xyz_cols:
            sub = pd.DataFrame({
                "id": f"{seg_id}__{axis}",
                "time": np.arange(len(seq)),
                "value": seq[axis].to_numpy(dtype=np.float32),
            })
            long_parts.append(sub)
        y_list.append(int(seq["segment_scratch"].any()))
        dur_list.append(float(seq["scratch_duration"].max()))
        seg_list.append(str(seg_id))

    if not long_parts:
        return (
            np.zeros((0, 0), dtype=np.float32),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float32),
            np.array([], dtype=object),
            [],
        )

    long_df = pd.concat(long_parts, ignore_index=True)
    settings = ComprehensiveFCParameters()
    ts_feat = tsfresh_extract(
        long_df,
        column_id="id",
        column_sort="time",
        column_value="value",
        default_fc_parameters=settings,
        n_jobs=n_jobs,
        disable_progressbar=disable_progressbar,
    )
    # ts_feat index is {seg}__{axis}.  Pivot back to one row per segment by
    # joining axis-tagged feature names.
    ts_feat.index = ts_feat.index.astype(str)
    idx_parts = ts_feat.index.str.rsplit("__", n=1, expand=True)
    ts_feat["__seg__"] = idx_parts[0]
    ts_feat["__axis__"] = idx_parts[1]
    wide = ts_feat.pivot(index="__seg__", columns="__axis__")
    wide.columns = [f"{col}_{axis}" for col, axis in wide.columns]

    # Align row order with seg_list (tsfresh reorders).
    wide = wide.reindex(seg_list)

    # FFT peak features per axis (paper adds FFT to tsfresh output).
    fft_rows = []
    for seg_id, seq in df.groupby(seg_col, sort=False, observed=True):
        if len(seq) < 4:
            continue
        row = {"__seg__": str(seg_id)}
        for axis in xyz_cols:
            stats = _fft_magnitude_stats(seq[axis].to_numpy(dtype=np.float32))
            for k, v in stats.items():
                row[f"{k}_{axis}"] = v
        fft_rows.append(row)
    fft_df = pd.DataFrame(fft_rows).set_index("__seg__").reindex(seg_list)

    X_df = pd.concat([wide, fft_df], axis=1)
    # Median-imputation (paper step).
    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    medians = X_df.median(axis=0, skipna=True)
    X_df = X_df.fillna(medians).fillna(0.0)

    X = X_df.to_numpy(dtype=np.float32)
    return (
        X,
        np.asarray(y_list, dtype=np.int64),
        np.asarray(dur_list, dtype=np.float32),
        np.asarray(seg_list, dtype=object),
        list(X_df.columns),
    )
