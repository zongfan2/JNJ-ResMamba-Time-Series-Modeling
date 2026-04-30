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
    """Return (extract_fn, settings_map) or (None, None) on failure."""
    try:
        from tsfresh import extract_features as _extract
        from tsfresh.feature_extraction import (
            ComprehensiveFCParameters,
            EfficientFCParameters,
            MinimalFCParameters,
        )
        settings = {
            "comprehensive": ComprehensiveFCParameters,
            "efficient": EfficientFCParameters,
            "minimal": MinimalFCParameters,
        }
        return _extract, settings
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
        from tsfresh.feature_extraction import (
            ComprehensiveFCParameters,
            EfficientFCParameters,
            MinimalFCParameters,
        )
        settings = {
            "comprehensive": ComprehensiveFCParameters,
            "efficient": EfficientFCParameters,
            "minimal": MinimalFCParameters,
        }
        return _extract, settings
    except Exception as exc:
        print(f"[mdpi2024_fe] tsfresh install failed: {exc}")
        return None, None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# Cap the auto-resolved worker count.  tsfresh forks workers via
# `multiprocessing.Pool`, and on big Linux boxes (Domino's 100+ cores) every
# fork dirties pages of the shared long_df (1–3 GB per axis at LOSO scale),
# so the working set explodes — we OOM'd a 224 GB box with cpu_count() workers.
# Cap at 4 by default (per-axis processing already removes the costliest
# pivot, so we don't need many workers); user can override via
# overrides.tsfresh_n_jobs in YAML.  Setting to 0 forces serial mode.
_TSFRESH_DEFAULT_MAX_WORKERS = 4


def _resolve_tsfresh_n_jobs(n_jobs: int | None,
                            max_workers: int = _TSFRESH_DEFAULT_MAX_WORKERS) -> int:
    """Translate sklearn-style ``n_jobs`` into a tsfresh-safe integer.

    - ``None`` or negative → ``min(cpu_count, max_workers)`` (auto, capped).
    - ``0`` → 0 (tsfresh's "no multiprocessing" mode).
    - positive → passed through as-is (user override; not capped).
    """
    if n_jobs is None or int(n_jobs) < 0:
        import os
        return min(os.cpu_count() or 1, max_workers)
    return int(n_jobs)


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
    n_jobs: int = -1,
    disable_progressbar: bool = False,
    feature_set_type: str = "comprehensive",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Return (X, y, dur, seg_ids, feature_names) for the Xing 2024 FE pipeline.

    Args:
        df: frame-level DataFrame with ``segment``, ``x``, ``y``, ``z``,
            ``segment_scratch``, ``scratch_duration`` columns.
        seg_col: segment-id column name.
        xyz_cols: the three accelerometer columns.
        n_jobs: tsfresh parallelism.  ``-1`` (default) uses all CPU cores,
                ``0`` disables multiprocessing (very slow on LOSO pools).
        disable_progressbar: hide tsfresh's per-series progress bar.
        feature_set_type: one of {"comprehensive", "efficient", "minimal"}.
            - "comprehensive" ≈ 794 features / axis (paper default, slowest)
            - "efficient"     ≈ 600 features / axis, skips CPU-heavy ones (~3× faster)
            - "minimal"       ≈ 10 features / axis (use for smoke tests)

    Returns:
        X: [N_segments, F] feature matrix.
        y: [N_segments] binary scratch label.
        dur: [N_segments] scratch duration.
        seg_ids: [N_segments] str ids (aligned with X rows).
        feature_names: list[str] of length F (column names).
    """
    tsfresh_extract, settings_map = _try_import_tsfresh()
    if tsfresh_extract is None:
        raise ImportError(
            "mdpi2024_fe baseline requires tsfresh. Install via `pip install tsfresh`."
        )
    if feature_set_type not in settings_map:
        raise ValueError(
            f"feature_set_type must be one of {list(settings_map)}, got {feature_set_type!r}"
        )
    SettingsCls = settings_map[feature_set_type]

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
    # Convert id to Categorical: per-row storage shrinks from ~80-byte Python
    # strings to small int codes (10–50× memory savings on the long_df,
    # critical when tsfresh forks N workers and each gets a copy).
    long_df["id"] = long_df["id"].astype("category")
    settings = SettingsCls()
    n_series = len(seg_list) * len(xyz_cols)
    resolved_n_jobs = _resolve_tsfresh_n_jobs(n_jobs)
    print(f"  [tsfresh] feature_set={feature_set_type} n_jobs={n_jobs}"
          f"{f' (resolved={resolved_n_jobs})' if resolved_n_jobs != n_jobs else ''} "
          f"n_series={n_series} (segments × axes) "
          f"rows={len(long_df):,}")
    ts_feat = tsfresh_extract(
        long_df,
        column_id="id",
        column_sort="time",
        column_value="value",
        default_fc_parameters=settings,
        n_jobs=resolved_n_jobs,
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


# ---------------------------------------------------------------------------
# Window-level extractor — Xing 2024 windowing protocol
# ---------------------------------------------------------------------------


def extract_tsfresh_features_from_windows(
    X_win: np.ndarray,
    fs: int = 20,
    n_jobs: int = -1,
    disable_progressbar: bool = False,
    feature_set_type: str = "comprehensive",
) -> tuple[np.ndarray, list[str]]:
    """tsfresh feature extraction on a stack of fixed-length 3-s windows.

    Mirrors :func:`extract_tsfresh_features_batch` but takes pre-tiled
    window tensors (shape ``[N_windows, win_len, 3]``) instead of a
    bout-level DataFrame.  Each row of the returned matrix is a single
    3-s window, matching the protocol of Xing 2024 (window=3 s,
    ComprehensiveFCParameters across x/y/z + FFT peaks per axis).

    Args:
        X_win: [N, win_len, 3] float window tensor (raw accelerometer).
        fs:    sampling rate, only used for FFT peak features.
        n_jobs / disable_progressbar / feature_set_type: as in the
                bout-level extractor.

    Returns:
        X:     [N, F] feature matrix aligned 1:1 with ``X_win``.
        feature_names: list[str] of length F.
    """
    tsfresh_extract, settings_map = _try_import_tsfresh()
    if tsfresh_extract is None:
        raise ImportError(
            "mdpi2024_fe baseline requires tsfresh. Install via `pip install tsfresh`."
        )
    if feature_set_type not in settings_map:
        raise ValueError(
            f"feature_set_type must be one of {list(settings_map)}, got {feature_set_type!r}"
        )
    SettingsCls = settings_map[feature_set_type]

    if X_win.ndim != 3 or X_win.shape[2] != 3:
        raise ValueError(f"X_win must be [N, L, 3]; got {X_win.shape}")
    n_windows, win_len, _ = X_win.shape
    if n_windows == 0:
        return np.zeros((0, 0), dtype=np.float32), []

    axes = ("x", "y", "z")
    settings = SettingsCls()
    resolved_n_jobs = _resolve_tsfresh_n_jobs(n_jobs)

    # PER-AXIS extraction.  The single-pass approach (one long_df with
    # 1.5 M (window × axis) series) forces a final `pivot` from
    # (n_series, n_features) to (n_windows, n_features × n_axes), and
    # pandas pivot transiently allocates 3–5× the data size — at LOSO
    # scale that means 30–50 GB even with a float32 final output, which
    # is what was OOM-ing the 224 GB box.  Running tsfresh once per axis
    # leaves us in window-major order from the start, so no pivot is
    # needed; we just rename columns to "{feat}_{axis}" and concatenate
    # along the column dimension.  Per-axis peak ~ tsfresh output
    # (n_windows × n_features × 4 bytes float32) plus the per-axis
    # long_df (n_windows × win_len × 12 bytes).
    print(f"  [tsfresh/window] feature_set={feature_set_type} n_jobs={n_jobs}"
          f"{f' (resolved={resolved_n_jobs})' if resolved_n_jobs != n_jobs else ''} "
          f"n_windows={n_windows:,} per_axis_rows={n_windows * win_len:,}")
    win_ids = [f"w{i}" for i in range(n_windows)]
    axis_frames: list[pd.DataFrame] = []
    feature_names_first: list[str] | None = None
    for axis_idx, axis in enumerate(axes):
        # Build a per-axis long_df with `n_windows` series of length win_len.
        series_codes = np.repeat(np.arange(n_windows, dtype=np.int64), win_len)
        time_idx = np.tile(np.arange(win_len, dtype=np.int32), n_windows)
        # Contiguous slice along the axis dimension; flatten in (window, t) order.
        values = np.ascontiguousarray(X_win[:, :, axis_idx]).reshape(-1).astype(np.float32, copy=False)
        id_cat = pd.Categorical.from_codes(series_codes, categories=win_ids)
        long_df = pd.DataFrame({"id": id_cat, "time": time_idx, "value": values})
        del series_codes, time_idx, values
        print(f"  [tsfresh/window] axis={axis} series={n_windows:,} rows={len(long_df):,}")
        ts_feat = tsfresh_extract(
            long_df,
            column_id="id",
            column_sort="time",
            column_value="value",
            default_fc_parameters=settings,
            n_jobs=resolved_n_jobs,
            disable_progressbar=disable_progressbar,
        )
        del long_df
        # Cast immediately to float32 so we don't carry a 2× tax through the
        # remaining 2 axes + concat.  Reorder rows to canonical win_ids order
        # (tsfresh may reshuffle when categorical-coded).
        ts_feat.index = ts_feat.index.astype(str)
        ts_feat = ts_feat.reindex(win_ids)
        ts_feat = ts_feat.astype(np.float32, copy=False)
        ts_feat.columns = [f"{c}_{axis}" for c in ts_feat.columns]
        if feature_names_first is None:
            feature_names_first = list(ts_feat.columns)
        axis_frames.append(ts_feat)
        # Best-effort GC between axes — releases the parent's COW pages
        # the workers dirtied so the next axis can fork cleanly.
        import gc
        gc.collect()

    # Concatenate axis-wise.  Float32 throughout so peak ≈ 3× per-axis output.
    wide = pd.concat(axis_frames, axis=1, copy=False)
    del axis_frames

    # Per-axis FFT peak features.  Vectorised over windows for speed.
    fft_data: dict[str, np.ndarray] = {}
    for axis_idx, axis in enumerate(axes):
        # Compute peak / mean magnitude per window in numpy (small, fast).
        sig = X_win[:, :, axis_idx].astype(np.float32)
        sig = sig - sig.mean(axis=1, keepdims=True)
        from scipy.fft import rfft  # type: ignore
        mag = np.abs(rfft(sig, axis=1))
        # skip DC bin
        if mag.shape[1] > 1:
            peak_idx = np.argmax(mag[:, 1:], axis=1) + 1
            peak_mag = mag[np.arange(mag.shape[0]), peak_idx]
            peak_freq = peak_idx.astype(np.float32)
        else:
            peak_mag = np.zeros(n_windows, dtype=np.float32)
            peak_freq = np.zeros(n_windows, dtype=np.float32)
        mean_mag = mag.mean(axis=1).astype(np.float32)
        fft_data[f"fft_peak_freq_{axis}"] = peak_freq
        fft_data[f"fft_peak_mag_{axis}"] = peak_mag.astype(np.float32)
        fft_data[f"fft_mean_mag_{axis}"] = mean_mag

    fft_df = pd.DataFrame(fft_data, index=win_ids).astype(np.float32)
    X_df = pd.concat([wide, fft_df], axis=1, copy=False)
    del wide, fft_df, fft_data

    # Median imputation (paper step).  Done on the fp32 frame.
    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    medians = X_df.median(axis=0, skipna=True)
    X_df = X_df.fillna(medians).fillna(0.0)
    X = X_df.to_numpy(dtype=np.float32, copy=False)
    feature_names = list(X_df.columns)
    del X_df
    import gc
    gc.collect()
    return X, feature_names
