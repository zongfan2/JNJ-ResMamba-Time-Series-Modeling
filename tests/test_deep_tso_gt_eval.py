import numpy as np


def test_interval_agreement_identical():
    from evaluation.tso_validation import interval_agreement
    a = np.array([0, 0, 1, 1, 1, 0])
    r = interval_agreement(a, a, timestep_minutes=1.0)
    assert r["pred_has_tso"] and r["gt_has_tso"]
    assert r["onset_mae_min"] == 0.0 and r["offset_mae_min"] == 0.0
    assert r["iou"] == 1.0 and r["duration_err_h"] == 0.0


def test_interval_agreement_shifted():
    from evaluation.tso_validation import interval_agreement
    pred = np.array([0, 0, 1, 1, 1, 0, 0])   # [2,5)
    gt = np.array([0, 0, 0, 1, 1, 1, 0])     # [3,6)
    r = interval_agreement(pred, gt, timestep_minutes=1.0)
    assert r["onset_mae_min"] == 1.0 and r["offset_mae_min"] == 1.0
    assert abs(r["iou"] - 0.5) < 1e-9
    assert r["duration_err_h"] == 0.0


def test_interval_agreement_one_missing():
    from evaluation.tso_validation import interval_agreement
    r = interval_agreement(np.array([0, 0, 0, 0]), np.array([0, 1, 1, 0]))
    assert r["pred_has_tso"] is False and r["gt_has_tso"] is True
    assert r["iou"] == 0.0
    assert np.isnan(r["onset_mae_min"]) and np.isnan(r["duration_err_h"])


def test_interval_agreement_neither():
    from evaluation.tso_validation import interval_agreement
    z = np.array([0, 0, 0])
    r = interval_agreement(z, z)
    assert r["pred_has_tso"] is False and r["gt_has_tso"] is False
    assert np.isnan(r["iou"]) and np.isnan(r["onset_mae_min"])
