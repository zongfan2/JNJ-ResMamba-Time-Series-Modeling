import numpy as np


def test_extract_tso_interval_binary_sequence():
    from evaluation.tso_validation import extract_tso_interval

    result = extract_tso_interval(np.array([0, 0, 1, 1, 1, 0]), timestep_minutes=1.0)
    assert result["onset_minute"] == 2
    assert result["offset_minute"] == 5
    assert result["duration_hours"] == 3 / 60
    assert result["segment_count"] == 1


def test_cross_night_consistency_groups_by_subject():
    from evaluation.tso_validation import cross_night_consistency

    intervals = [
        {"subject": "S1", "onset_minute": 100, "offset_minute": 500, "duration_hours": 6.67, "segment_count": 1},
        {"subject": "S1", "onset_minute": 110, "offset_minute": 505, "duration_hours": 6.58, "segment_count": 1},
        {"subject": "S2", "onset_minute": 200, "offset_minute": 600, "duration_hours": 6.67, "segment_count": 1},
    ]
    metrics = cross_night_consistency(intervals)
    assert metrics["subjects_with_multiple_nights"] == 1
    assert metrics["mean_onset_std_minutes"] == 5.0
