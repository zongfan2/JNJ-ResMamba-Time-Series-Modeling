from __future__ import annotations

from collections import defaultdict

import numpy as np


def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
    segments = []
    start = None
    for i, value in enumerate(mask.astype(bool)):
        if value and start is None:
            start = i
        elif not value and start is not None:
            segments.append((start, i))
            start = None
    if start is not None:
        segments.append((start, len(mask)))
    return segments


def extract_tso_interval(pred_classes: np.ndarray, *, timestep_minutes: float = 1.0) -> dict:
    tso_class = 1 if np.max(pred_classes) <= 1 else 2
    segments = _segments(pred_classes == tso_class)
    if not segments:
        return {
            "onset_minute": np.nan,
            "offset_minute": np.nan,
            "duration_hours": 0.0,
            "segment_count": 0,
        }
    longest = max(segments, key=lambda pair: pair[1] - pair[0])
    duration_steps = longest[1] - longest[0]
    return {
        "onset_minute": float(longest[0] * timestep_minutes),
        "offset_minute": float(longest[1] * timestep_minutes),
        "duration_hours": float(duration_steps * timestep_minutes / 60.0),
        "segment_count": len(segments),
    }


def interval_agreement(pred_classes: np.ndarray, gt_classes: np.ndarray,
                       *, timestep_minutes: float = 1.0) -> dict:
    """Compare predicted vs GT TSO intervals (longest contiguous block each side).

    onset/offset MAE and duration error are defined only when BOTH sides have a
    TSO interval (else NaN). IoU is 0 when exactly one side has an interval, NaN
    when neither does.
    """
    pred = extract_tso_interval(np.asarray(pred_classes), timestep_minutes=timestep_minutes)
    gt = extract_tso_interval(np.asarray(gt_classes), timestep_minutes=timestep_minutes)
    pred_has = pred["segment_count"] > 0
    gt_has = gt["segment_count"] > 0
    out = {
        "pred_has_tso": bool(pred_has),
        "gt_has_tso": bool(gt_has),
        "onset_mae_min": np.nan,
        "offset_mae_min": np.nan,
        "duration_err_h": np.nan,
        "iou": np.nan,
    }
    if pred_has and gt_has:
        out["onset_mae_min"] = abs(pred["onset_minute"] - gt["onset_minute"])
        out["offset_mae_min"] = abs(pred["offset_minute"] - gt["offset_minute"])
        out["duration_err_h"] = pred["duration_hours"] - gt["duration_hours"]
        inter = max(0.0, min(pred["offset_minute"], gt["offset_minute"])
                    - max(pred["onset_minute"], gt["onset_minute"]))
        union = ((pred["offset_minute"] - pred["onset_minute"])
                 + (gt["offset_minute"] - gt["onset_minute"]) - inter)
        out["iou"] = float(inter / union) if union > 0 else 0.0
    elif pred_has != gt_has:
        out["iou"] = 0.0
    return out


def cross_night_consistency(intervals: list[dict]) -> dict:
    by_subject = defaultdict(list)
    for item in intervals:
        by_subject[item["subject"]].append(item)

    onset_stds = []
    offset_stds = []
    duration_stds = []
    for subject_intervals in by_subject.values():
        if len(subject_intervals) < 2:
            continue
        onset_stds.append(float(np.nanstd([x["onset_minute"] for x in subject_intervals])))
        offset_stds.append(float(np.nanstd([x["offset_minute"] for x in subject_intervals])))
        duration_stds.append(float(np.nanstd([x["duration_hours"] for x in subject_intervals])))

    return {
        "subjects_with_multiple_nights": len(onset_stds),
        "mean_onset_std_minutes": float(np.nanmean(onset_stds)) if onset_stds else np.nan,
        "mean_offset_std_minutes": float(np.nanmean(offset_stds)) if offset_stds else np.nan,
        "mean_duration_std_hours": float(np.nanmean(duration_stds)) if duration_stds else np.nan,
    }
