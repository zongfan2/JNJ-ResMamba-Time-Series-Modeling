# -*- coding: utf-8 -*-
"""Sliding-window tiler for paper-faithful classical baselines.

Mahadevan 2021, Ji 2023 and Xing 2024 all train and evaluate on fixed 3-s
windows rather than the full motion bouts. This module slices each bout
into 3-s windows so the classical pipelines reproduce their protocols:

  * train stride 1.5 s (50% overlap) — Mahadevan + Ji "to maximize data
    availability" / "to increase the sample size".
  * inference stride 3.0 s (non-overlap) — both papers' deployment rule;
    needed so that ``total_scratch_duration_seconds = 3.0 × n_pos_windows``.

Window labels follow Ji 2023: a window is positive if more than 1 s
(20 frames at 20 Hz) of frame-level scratch falls inside it. Mahadevan's
rule (any annotated scratch annotation ≥3 s) is recoverable by setting
``label_min_scratch_seconds`` to a small positive value.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_windows(
    df: pd.DataFrame,
    *,
    fs: int = 20,
    win_seconds: float = 3.0,
    stride_seconds: float = 1.5,
    label_min_scratch_seconds: float = 1.0,
    seg_col: str = "segment",
    xyz_cols=("x", "y", "z"),
    label_col: str = "scratch",
    extra_cols=("PID", "FOLD"),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Tile each bout in ``df`` into fixed-length windows.

    Args:
        df: frame-level DataFrame from data.loading.load_data.
        fs: sampling rate in Hz (20 for our dataset).
        win_seconds: window length (3 s in all three reference papers).
        stride_seconds: 1.5 for training (50% overlap), 3.0 for inference.
        label_min_scratch_seconds: a window is labelled positive if it
            contains more than this many seconds of frame-level scratch.
            Ji 2023 uses 1.0 s on a 3-s window; set to 0.0 for any-overlap.
        seg_col / xyz_cols / label_col: column names in ``df``.
        extra_cols: per-bout grouping columns to propagate (PID, FOLD,
            etc.). Each window inherits the bout's value verbatim.

    Returns:
        X:        [N, win_len, 3] float32 raw signal per window.
        y:        [N] int64 binary label (Ji rule).
        dur:      [N] float32 scratch seconds inside the window (0..win_seconds).
        seg_ids:  [N] str bout id per window — used to aggregate predictions
                  back to bout level.
        win_ids:  [N] str window id ("{seg}__w{k}") — useful for joins / debug.
        extras:   {col: np.ndarray[N]} for each name in ``extra_cols``.
    """
    win_len = int(round(win_seconds * fs))
    stride = max(1, int(round(stride_seconds * fs)))
    label_min_frames = float(label_min_scratch_seconds) * fs

    X_list, y_list, dur_list = [], [], []
    seg_list, win_list = [], []
    extras: dict[str, list] = {c: [] for c in extra_cols if c in df.columns}

    grouped = df.groupby(seg_col, sort=False, observed=True)
    for seg_id, seq in grouped:
        L = len(seq)
        if L < win_len:
            continue  # bouts shorter than the window are skipped
        xyz = seq.loc[:, list(xyz_cols)].to_numpy(dtype=np.float32)
        scratch = seq[label_col].to_numpy(dtype=np.float32)
        seg_str = str(seg_id)
        # Per-bout extras: take the first frame's value (constant within bout).
        extras_row = {c: seq[c].iloc[0] for c in extras}

        # Slide windows: starts = 0, stride, 2*stride, ... up to L-win_len.
        last_start = L - win_len
        starts = np.arange(0, last_start + 1, stride, dtype=np.int64)
        for k, s in enumerate(starts):
            e = s + win_len
            X_list.append(xyz[s:e])
            scr_frames = float(scratch[s:e].sum())
            dur_list.append(scr_frames / fs)
            y_list.append(int(scr_frames > label_min_frames))
            seg_list.append(seg_str)
            win_list.append(f"{seg_str}__w{k}")
            for c, v in extras_row.items():
                extras[c].append(v)

    if not X_list:
        empty_arr = np.zeros((0, win_len, 3), dtype=np.float32)
        return (
            empty_arr,
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float32),
            np.array([], dtype=object),
            np.array([], dtype=object),
            {c: np.array([], dtype=object) for c in extras},
        )

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.asarray(y_list, dtype=np.int64)
    dur = np.asarray(dur_list, dtype=np.float32)
    seg_ids = np.asarray(seg_list, dtype=object)
    win_ids = np.asarray(win_list, dtype=object)
    extras_out = {c: np.asarray(v, dtype=object) for c, v in extras.items()}
    return X, y, dur, seg_ids, win_ids, extras_out


def aggregate_window_predictions_to_bout(
    seg_ids: np.ndarray,
    win_pred: np.ndarray,
    win_prob: np.ndarray | None = None,
    *,
    win_seconds: float = 3.0,
    duration_estimator: str = "count",
    win_dur_pred: np.ndarray | None = None,
    min_pos_windows: int = 1,
) -> tuple[dict, dict, dict]:
    """Pool per-window predictions to per-bout endpoints.

    Args:
        seg_ids: bout id per window (output of ``make_windows``, infer stride).
        win_pred: [N] int 0/1 predicted window label.
        win_prob: [N] float optional predicted positive probability.
        win_seconds: window length used for the count rule.
        duration_estimator:
            - "count": Mahadevan/Ji rule. ``pr3 = win_seconds × n_pos_windows``.
              Equivalent to "total scratch duration during this bout."
            - "regressor": sum the per-window regressor output (passed in
              ``win_dur_pred``) over positive windows. Useful when a
              two-stage hurdle regressor is available.
        win_dur_pred: required when ``duration_estimator="regressor"``.
        min_pos_windows: bout flips positive once this many windows
            (contiguous positives are not enforced — Mahadevan's "any run"
            counts a single positive window as an event).

    Returns:
        seg_to_pr1, seg_to_pr1_prob, seg_to_pr3 — dicts keyed by seg_id.
    """
    seg_ids = np.asarray(seg_ids)
    win_pred = np.asarray(win_pred).astype(int)
    if win_prob is None:
        win_prob = win_pred.astype(float)
    win_prob = np.asarray(win_prob, dtype=float)

    if duration_estimator not in ("count", "regressor"):
        raise ValueError(f"duration_estimator must be 'count' or 'regressor' (got {duration_estimator!r})")
    if duration_estimator == "regressor":
        if win_dur_pred is None:
            raise ValueError("duration_estimator='regressor' requires win_dur_pred")
        win_dur_pred = np.asarray(win_dur_pred, dtype=float)

    seg_to_pr1: dict = {}
    seg_to_pr1_prob: dict = {}
    seg_to_pr3: dict = {}
    # groupby preserving first-seen order
    df = pd.DataFrame({"seg": seg_ids, "pred": win_pred, "prob": win_prob})
    if duration_estimator == "regressor":
        df["dur"] = win_dur_pred
    for seg_id, sub in df.groupby("seg", sort=False, observed=True):
        n_pos = int(sub["pred"].sum())
        seg_to_pr1[seg_id] = int(n_pos >= min_pos_windows)
        seg_to_pr1_prob[seg_id] = float(sub["prob"].max())
        if duration_estimator == "count":
            seg_to_pr3[seg_id] = float(win_seconds) * n_pos
        else:
            seg_to_pr3[seg_id] = float(np.clip(sub.loc[sub["pred"] == 1, "dur"].sum(), 0.0, None))
    return seg_to_pr1, seg_to_pr1_prob, seg_to_pr3
