# -*- coding: utf-8 -*-
"""Runtime wrapper for classical (Mahadevan / Ji / MDPI) baselines.

NS-pipeline.py was written around the DL ``MBA_tsm`` model and ``run_model()``
loop in :mod:`Helpers.helpers`.  Classical baselines trained via
``training/train_classical.py`` live in a joblib bundle (sklearn / LightGBM /
XGBoost / a torch CNN wrapped in a sklearn-style adapter) and operate on
3-s windows rather than full bouts.  This wrapper bridges the two: given a
joblib bundle and a pre-processed motion DataFrame, it returns predictions
in the same ``{segment, position, pr1, pr1_probs, pr2, pr2_probs, pr3}``
schema the pipeline's downstream merge expects.

The wrapper is deliberately stateless beyond what's saved in the bundle.
Anything inference-specific that varies across architectures (windowing
params, feature standardisation, regressor log-target, feature selection)
must be persisted into the bundle by ``training/train_classical.py``; this
file refuses to guess.
"""
from __future__ import annotations

import os
import sys

import joblib
import numpy as np
import pandas as pd

# Make ``baselines_classical`` importable as a TOP-LEVEL name even though
# the package now lives at ``Helpers/baselines_classical/`` (it was moved
# in 2026-05-18).  Two reasons we keep the top-level alias rather than
# switching to ``from Helpers.baselines_classical.X import ...``:
#
#   1. joblib bundles pickled by training/train_classical.py reference
#      ``baselines_classical.cnn_classifier.WindowedCNNClassifier`` by its
#      full dotted module path.  Pickle resolves that string against
#      sys.modules at load time, so the name must point at the same
#      package the bundle was trained against — otherwise unpickling
#      breaks with ``ModuleNotFoundError: baselines_classical``.
#   2. Keeps this file portable: the research tree's top-level
#      ``baselines_classical/`` and the production copy stay
#      interchangeable, so a bundle trained anywhere drops in.
_HELPERS_ROOT = os.path.dirname(os.path.abspath(__file__))
if _HELPERS_ROOT not in sys.path:
    sys.path.insert(0, _HELPERS_ROOT)

from baselines_classical.features import (  # noqa: E402
    extract_features,
    extract_features_from_windows,
    feature_names,
)
from baselines_classical.windowing import (  # noqa: E402
    aggregate_window_predictions_to_bout,
    make_windows,
)


CLASSICAL_ARCHS = ("mahadevan2021", "ji2023", "mdpi2024_fe", "mdpi2024_cnn")


def is_classical_arch(name: str) -> bool:
    return str(name) in CLASSICAL_ARCHS


class ClassicalBackend:
    """Drop-in replacement for the DL model in ``NS-pipeline.py``.

    Loads a joblib bundle produced by ``training/train_classical.py`` and
    exposes a single :meth:`predict` method that mimics the output of
    :func:`Helpers.helpers.run_model` for a single motion-segmented input.

    The bundle is expected to contain at minimum::

        {
            "clf":           fitted sklearn-compatible classifier,
            "reg":           fitted regressor or None (count-rule pr3),
            "feature_idx":   np.ndarray of selected feature indices (mahadevan)
                             or np.arange(n_features) for others,
            "windowing":     {win_seconds, infer_stride_seconds,
                              label_min_scratch_seconds, duration_estimator,
                              min_pos_windows}  OR  None  (bout-level mode),
            "arch":          one of CLASSICAL_ARCHS,
            "feature_set":   "default" | "tsfresh" | "mdpi2024"  (optional),
            "feature_mu":    optional [F] float32 array — train-fold mean,
            "feature_sd":    optional [F] float32 array — train-fold std,
            "log_target":    bool — regressor was fit on log1p(duration),
        }
    """

    def __init__(self, bundle_path: str, fs: int = 20, verbose: bool = True):
        self.bundle_path = bundle_path
        self.bundle = joblib.load(bundle_path)
        self.fs = int(fs)
        self.verbose = bool(verbose)

        self.clf = self.bundle["clf"]
        self.reg = self.bundle.get("reg", None)
        self.feature_idx = self.bundle.get("feature_idx", None)
        self.windowing = self.bundle.get("windowing", None)
        self.arch = self.bundle.get("arch", None)
        self.feature_set = str(self.bundle.get("feature_set", "default")).lower()
        self.feature_mu = self.bundle.get("feature_mu", None)
        self.feature_sd = self.bundle.get("feature_sd", None)
        self.log_target = bool(self.bundle.get("log_target", True))

        if self.arch is None:
            raise ValueError(
                f"Classical bundle {bundle_path} missing required 'arch' key. "
                f"Retrain with the updated training/train_classical.py."
            )
        if self.arch not in CLASSICAL_ARCHS:
            raise ValueError(f"Unknown classical arch: {self.arch!r}")
        if self.verbose:
            wmode = "windowed" if self.windowing else "bout-level"
            reg_kind = type(self.reg).__name__ if self.reg is not None else "none"
            print(
                f"[classical] loaded {self.arch} from {bundle_path}  "
                f"(mode={wmode}, feature_set={self.feature_set}, "
                f"reg={reg_kind})"
            )

    # ------------------------------------------------------------------
    # PyTorch-shaped no-ops so the existing NS-pipeline call site
    # (``with torch.no_grad(): model.eval(); run_model(...)``) just works.
    # ------------------------------------------------------------------

    def eval(self):
        return self

    def to(self, device):
        return self

    # ------------------------------------------------------------------
    # Public inference entry point
    # ------------------------------------------------------------------

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run inference on a motion-segmented frame.

        Args:
            df: per-frame DataFrame with columns ``segment``, ``x``, ``y``,
                ``z`` (HPF inside features.py — do NOT pass z-scored x/y/z).
                Additional columns are ignored.

        Returns:
            DataFrame with columns ``segment, position, pr1, pr1_probs,
            pr2, pr2_probs, pr3``.  Frame-level ``pr2`` is broadcast from
            ``pr1`` since classical baselines emit one decision per bout.
            ``pr3`` is a fraction of the bout (in [0, 1]) so the existing
            ``pr3 = pr3 * segment_duration`` step downstream converts it
            to seconds — matching the DL contract.
        """
        if len(df) == 0:
            return self._empty_predictions(df)

        # make_windows reads a label column (defaults to 'scratch') purely to
        # compute window-level positivity for training; in production we don't
        # carry labels, so populate a zero column.  We do this on a shallow
        # copy to avoid mutating the caller's frame.
        if "scratch" not in df.columns:
            df = df.assign(scratch=0)

        if self.windowing is not None:
            return self._predict_windowed(df)
        return self._predict_bout_level(df)

    # ------------------------------------------------------------------
    # Windowed inference  (paper-faithful path: mahadevan / ji / mdpi)
    # ------------------------------------------------------------------

    def _predict_windowed(self, df: pd.DataFrame) -> pd.DataFrame:
        win = self.windowing
        win_seconds = float(win.get("win_seconds", 3.0))
        infer_stride_seconds = float(win.get("infer_stride_seconds", win_seconds))
        label_min_scratch_seconds = float(win.get("label_min_scratch_seconds", 1.0))
        duration_estimator = str(win.get("duration_estimator", "count")).lower()
        min_pos_windows = int(win.get("min_pos_windows", 1))

        Xw, _, _, seg_ids, _, _ = make_windows(
            df,
            fs=self.fs,
            win_seconds=win_seconds,
            stride_seconds=infer_stride_seconds,
            label_min_scratch_seconds=label_min_scratch_seconds,
            label_col="scratch",
            extra_cols=(),
        )
        if len(Xw) == 0:
            return self._empty_predictions(df)

        # Feature extraction.
        if self.arch == "mdpi2024_cnn":
            X = Xw  # CNN consumes raw [N, T, 3] tensors directly.
        elif self.feature_set in ("tsfresh", "mdpi2024"):
            from baselines_classical.tsfresh_features import (  # noqa: E402
                extract_tsfresh_features_from_windows,
            )
            X, _ = extract_tsfresh_features_from_windows(Xw, fs=self.fs)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            X = extract_features_from_windows(Xw, fs=self.fs, hpf=True)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        X = self._postprocess_features(X)

        # Classifier predictions.
        pr1 = self.clf.predict(X).astype(int)
        if hasattr(self.clf, "predict_proba"):
            pr1_prob = self.clf.predict_proba(X)[:, 1].astype(float)
        else:
            pr1_prob = pr1.astype(float)

        # Regressor predictions (only used when duration_estimator='regressor').
        reg_raw = None
        if self.reg is not None and duration_estimator == "regressor":
            reg_raw = self.reg.predict(X).astype(float)
            if self.log_target:
                reg_raw = np.expm1(reg_raw)
            reg_raw = np.clip(reg_raw, 0.0, None)

        seg_to_pr1, seg_to_pr1_prob, seg_to_pr3_seconds = aggregate_window_predictions_to_bout(
            seg_ids,
            pr1,
            pr1_prob,
            win_seconds=win_seconds,
            duration_estimator=duration_estimator,
            win_dur_pred=reg_raw,
            min_pos_windows=min_pos_windows,
        )

        # Convert seconds → bout fraction.  The pipeline multiplies pr3 by
        # ``segment_duration`` (seconds) downstream to recover seconds, so
        # passing back a fraction matches the DL contract.
        bout_total_seconds = (
            df.groupby("segment", sort=False, observed=True).size() / self.fs
        ).to_dict()
        seg_to_pr3 = {
            seg: float(np.clip(secs / max(1e-6, bout_total_seconds.get(seg, 1e-6)), 0.0, 1.0))
            for seg, secs in seg_to_pr3_seconds.items()
        }

        return self._broadcast_to_frames(df, seg_to_pr1, seg_to_pr1_prob, seg_to_pr3)

    # ------------------------------------------------------------------
    # Bout-level inference  (non-paper protocol; rarely deployed)
    # ------------------------------------------------------------------

    def _predict_bout_level(self, df: pd.DataFrame) -> pd.DataFrame:
        X_list, seg_list = [], []
        for seg_id, seq in df.groupby("segment", sort=False, observed=True):
            xyz = seq.loc[:, ["x", "y", "z"]].to_numpy(dtype=np.float32)
            if xyz.shape[0] < 4:
                continue
            X_list.append(extract_features(xyz, fs=self.fs, hpf=True))
            seg_list.append(str(seg_id))
        if not X_list:
            return self._empty_predictions(df)
        X = np.vstack(X_list).astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X = self._postprocess_features(X)

        pr1 = self.clf.predict(X).astype(int)
        if hasattr(self.clf, "predict_proba"):
            pr1_prob = self.clf.predict_proba(X)[:, 1].astype(float)
        else:
            pr1_prob = pr1.astype(float)

        if self.reg is not None:
            pr3_seconds = self.reg.predict(X).astype(float)
            if self.log_target:
                pr3_seconds = np.expm1(pr3_seconds)
            pr3_seconds = np.clip(pr3_seconds, 0.0, None)
        else:
            pr3_seconds = np.zeros(len(seg_list), dtype=float)

        bout_total_seconds = (
            df.groupby("segment", sort=False, observed=True).size() / self.fs
        ).to_dict()
        seg_to_pr1 = dict(zip(seg_list, pr1))
        seg_to_pr1_prob = dict(zip(seg_list, pr1_prob))
        seg_to_pr3 = {
            seg: float(np.clip(secs / max(1e-6, bout_total_seconds.get(seg, 1e-6)), 0.0, 1.0))
            for seg, secs in zip(seg_list, pr3_seconds)
        }
        return self._broadcast_to_frames(df, seg_to_pr1, seg_to_pr1_prob, seg_to_pr3)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _postprocess_features(self, X: np.ndarray) -> np.ndarray:
        """Apply train-fold standardisation + feature selection if recorded.

        CNN inputs are raw [N, T, 3] tensors — skipped.
        """
        if X.ndim != 2:
            return X
        if self.feature_mu is not None and self.feature_sd is not None:
            mu = np.asarray(self.feature_mu, dtype=np.float32)
            sd = np.asarray(self.feature_sd, dtype=np.float32)
            sd = np.where(sd < 1e-8, 1.0, sd)
            X = ((X - mu) / sd).astype(np.float32)
        if self.feature_idx is not None and len(self.feature_idx) > 0:
            X = X[:, np.asarray(self.feature_idx)]
        return X

    def _broadcast_to_frames(
        self,
        df: pd.DataFrame,
        seg_to_pr1: dict,
        seg_to_pr1_prob: dict,
        seg_to_pr3: dict,
    ) -> pd.DataFrame:
        """Build a per-frame prediction DataFrame matching run_model output."""
        seg_keys = df["segment"]
        # Bridges str/non-str segment key lookups — make_windows stringifies seg ids.
        seg_str = seg_keys.astype(str)
        pr1 = seg_str.map(seg_to_pr1)
        if pr1.isna().any():
            pr1 = pr1.fillna(seg_keys.map(seg_to_pr1))
        pr1 = pr1.fillna(0).astype(int).to_numpy()

        pr1_probs = seg_str.map(seg_to_pr1_prob)
        if pr1_probs.isna().any():
            pr1_probs = pr1_probs.fillna(seg_keys.map(seg_to_pr1_prob))
        pr1_probs = pr1_probs.fillna(0.0).astype(float).to_numpy()

        pr3 = seg_str.map(seg_to_pr3)
        if pr3.isna().any():
            pr3 = pr3.fillna(seg_keys.map(seg_to_pr3))
        pr3 = pr3.fillna(0.0).astype(float).to_numpy()

        position = (
            df.groupby("segment", sort=False, observed=True).cumcount().to_numpy() + 1
        )
        return pd.DataFrame(
            {
                "segment": seg_keys.to_numpy(),
                "position": position,
                "pr1": pr1,
                "pr1_probs": pr1_probs,
                # Broadcast pr1 to pr2 — classical baselines don't predict
                # per-timestep masks, so every frame in a positive bout is
                # labelled positive (and every frame in a negative bout, zero).
                "pr2": pr1,
                "pr2_probs": pr1_probs,
                "pr3": pr3,
            }
        )

    def _empty_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        position = (
            df.groupby("segment", sort=False, observed=True).cumcount().to_numpy() + 1
        )
        n = len(df)
        return pd.DataFrame(
            {
                "segment": df["segment"].to_numpy(),
                "position": position,
                "pr1": np.zeros(n, dtype=int),
                "pr1_probs": np.zeros(n, dtype=float),
                "pr2": np.zeros(n, dtype=int),
                "pr2_probs": np.zeros(n, dtype=float),
                "pr3": np.zeros(n, dtype=float),
            }
        )
