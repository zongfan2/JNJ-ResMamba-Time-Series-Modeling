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
    chunk_size: int = 50000,
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

    import gc
    axes = ("x", "y", "z")
    settings = SettingsCls()
    resolved_n_jobs = _resolve_tsfresh_n_jobs(n_jobs)

    # CHUNKED + PER-AXIS + PREALLOCATED extraction.
    #
    # On a 40 GB host, even per-axis-without-pivot was tight: each axis
    # built a (n_windows, ~794) float32 frame (~1.5 GB at LOSO scale), and
    # accumulating 3 of them plus the concat transient + the still-alive
    # df_train (~2 GB) + worker COW pages tipped past 40 GB.
    #
    # Strategy:
    #   1.  Preallocate ONE float32 output array of final shape and write
    #       chunks directly into the right rows/columns.  Never accumulate
    #       per-axis DataFrames.
    #   2.  Process each axis × chunk separately so peak transient
    #       memory is bounded by (chunk_size × n_features × 4) + tsfresh's
    #       per-worker working set.
    #   3.  Collect FFT features in one vectorised numpy pass at the end
    #       — they're tiny (9 columns total).
    chunk_size = max(1, int(chunk_size))
    n_chunks = (n_windows + chunk_size - 1) // chunk_size
    print(f"  [tsfresh/window] feature_set={feature_set_type} n_jobs={n_jobs}"
          f"{f' (resolved={resolved_n_jobs})' if resolved_n_jobs != n_jobs else ''} "
          f"n_windows={n_windows:,} chunk_size={chunk_size:,} ({n_chunks} chunks/axis × {len(axes)} axes)")

    feature_names_per_axis: list[str] | None = None
    out_array: np.ndarray | None = None  # preallocated once we know n_features
    n_feats_per_axis = 0

    for axis_idx, axis in enumerate(axes):
        for ci, chunk_start in enumerate(range(0, n_windows, chunk_size)):
            chunk_end = min(chunk_start + chunk_size, n_windows)
            chunk_n = chunk_end - chunk_start

            # Per-chunk long_df: small (chunk_size × win_len rows).
            series_codes = np.repeat(np.arange(chunk_n, dtype=np.int64), win_len)
            time_idx = np.tile(np.arange(win_len, dtype=np.int32), chunk_n)
            values = np.ascontiguousarray(
                X_win[chunk_start:chunk_end, :, axis_idx]
            ).reshape(-1).astype(np.float32, copy=False)
            chunk_win_ids = [f"w{chunk_start + i}" for i in range(chunk_n)]
            id_cat = pd.Categorical.from_codes(series_codes, categories=chunk_win_ids)
            long_df = pd.DataFrame({"id": id_cat, "time": time_idx, "value": values})
            del series_codes, time_idx, values

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
            ts_feat.index = ts_feat.index.astype(str)
            ts_feat = ts_feat.reindex(chunk_win_ids)

            if feature_names_per_axis is None:
                # First chunk discovers the canonical feature schema.  Settings
                # are deterministic so subsequent chunks produce the same set.
                feature_names_per_axis = list(ts_feat.columns)
                n_feats_per_axis = len(feature_names_per_axis)
                # Final shape: per-axis tsfresh features ×3 + 9 FFT columns
                # (peak_freq/peak_mag/mean_mag per axis).
                out_array = np.zeros(
                    (n_windows, n_feats_per_axis * len(axes) + 3 * len(axes)),
                    dtype=np.float32,
                )
                print(f"  [tsfresh/window] preallocated output: shape={out_array.shape} "
                      f"size={out_array.nbytes / 1e9:.2f} GB float32")
            else:
                # Defensive reorder (should be a no-op when settings are constant).
                ts_feat = ts_feat[feature_names_per_axis]

            # Write chunk into the preallocated array.  to_numpy avoids
            # an intermediate DataFrame copy when ts_feat is already aligned.
            chunk_arr = ts_feat.to_numpy(dtype=np.float32, copy=False)
            del ts_feat
            chunk_arr[~np.isfinite(chunk_arr)] = np.nan
            col_start = axis_idx * n_feats_per_axis
            col_end = col_start + n_feats_per_axis
            out_array[chunk_start:chunk_end, col_start:col_end] = chunk_arr
            del chunk_arr

            print(f"  [tsfresh/window] axis={axis} chunk {ci+1}/{n_chunks} "
                  f"({chunk_start:,}..{chunk_end:,})")
            gc.collect()

    assert out_array is not None and feature_names_per_axis is not None

    # Vectorised FFT features over all windows × all axes at once.
    from scipy.fft import rfft  # type: ignore
    sig = X_win.astype(np.float32, copy=False)
    sig = sig - sig.mean(axis=1, keepdims=True)
    mag = np.abs(rfft(sig, axis=1)).astype(np.float32)  # (n_windows, n_freq, n_axes)
    n_freq = mag.shape[1]
    if n_freq > 1:
        peak_idx = np.argmax(mag[:, 1:, :], axis=1) + 1  # (n_windows, n_axes)
        peak_mag = np.take_along_axis(mag, peak_idx[:, None, :], axis=1)[:, 0, :]
        peak_freq = peak_idx.astype(np.float32)
    else:
        peak_freq = np.zeros((n_windows, len(axes)), dtype=np.float32)
        peak_mag = np.zeros((n_windows, len(axes)), dtype=np.float32)
    mean_mag = mag.mean(axis=1).astype(np.float32)
    del sig, mag

    fft_col_start = n_feats_per_axis * len(axes)
    out_array[:, fft_col_start : fft_col_start + len(axes)] = peak_freq
    out_array[:, fft_col_start + len(axes) : fft_col_start + 2 * len(axes)] = peak_mag
    out_array[:, fft_col_start + 2 * len(axes) : fft_col_start + 3 * len(axes)] = mean_mag

    # Median imputation across the full output array.
    nan_mask = np.isnan(out_array)
    if nan_mask.any():
        medians = np.nanmedian(out_array, axis=0).astype(np.float32)
        medians[np.isnan(medians)] = 0.0
        rows, cols = np.where(nan_mask)
        out_array[rows, cols] = medians[cols]
    np.nan_to_num(out_array, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    feature_names = (
        [f"{c}_{a}" for a in axes for c in feature_names_per_axis]
        + [f"fft_peak_freq_{a}" for a in axes]
        + [f"fft_peak_mag_{a}" for a in axes]
        + [f"fft_mean_mag_{a}" for a in axes]
    )
    gc.collect()
    return out_array, feature_names
