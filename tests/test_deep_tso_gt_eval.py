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


def test_add_padding_returns_pad_y_gt():
    import torch
    from data.padding import add_padding_tso_patch_h5

    class _DS:
        num_channels = 6
        def __init__(self, with_gt):
            self.with_gt = with_gt
            self._x = __import__("numpy").zeros((120, 6), "float32")
            self._y = __import__("numpy").zeros((120, 2), "int8"); self._y[:60, 0] = 1
            self._g = __import__("numpy").zeros((120,), "int8"); self._g[:60] = 1  # GT minute0
        def __getitem__(self, i):
            d = {"X": self._x, "Y": self._y, "seq_length": 120,
                 "segment": "S%d_0_d" % i, "segment_id": i}
            if self.with_gt:
                d["Y_gt"] = self._g
            return d

    out = add_padding_tso_patch_h5(_DS(True), [0], torch.device("cpu"),
                                   max_seq_len=1440, patch_size=60, num_channels=6)
    assert len(out) == 6
    pad_Y_gt = out[5]
    assert pad_Y_gt is not None and tuple(pad_Y_gt.shape) == (1, 2)
    assert pad_Y_gt[0, 0].item() == 1 and pad_Y_gt[0, 1].item() == 0

    out2 = add_padding_tso_patch_h5(_DS(False), [0], torch.device("cpu"),
                                    max_seq_len=1440, patch_size=60, num_channels=6)
    assert len(out2) == 6 and out2[5] is None
