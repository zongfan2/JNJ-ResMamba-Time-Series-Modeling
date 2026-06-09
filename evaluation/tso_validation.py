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
