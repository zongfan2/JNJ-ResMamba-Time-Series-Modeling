# -*- coding: utf-8 -*-
"""
Hand-crafted time/frequency features for classical scratch-detection baselines.

References
----------
- Mahadevan et al. 2021 (npj Digital Medicine): RF on 36 time/frequency features
  computed from SVM / PC1 / PC2 of a 0.25 Hz HPF-filtered 3-axis signal.
- Ji et al. 2023 (npj Digital Medicine): same physics-based features form the
  bulk of their interpretable feature set (TDA and DL features added on top;
  omitted in this "pragmatic" replication).

The exact 36 features are not enumerated in either paper.  The set below is a
faithful reconstruction from the named feature groups (mean cross rate, SPARC,
jerk, dominant frequency, plus standard descriptive statistics) and matches
conventions used elsewhere in the human-activity-recognition literature.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal as sp_signal
from scipy.fft import rfft, rfftfreq


# ---------------------------------------------------------------------------
# Signal preprocessing (shared by both papers, accel-only branch)
# ---------------------------------------------------------------------------


def highpass_filter(x: np.ndarray, fs: int = 20, cutoff: float = 0.25) -> np.ndarray:
    """First-order Butterworth IIR high-pass filter along axis 0.

    Removes the ~0 Hz gravity component, matching both papers' accel-only
    preprocessing step.
    """
    if x.shape[0] < 9:
        # filtfilt needs enough samples; short segments fall back to mean-subtract
        return x - x.mean(axis=0, keepdims=True)
    b, a = sp_signal.butter(1, cutoff / (fs / 2.0), btype="highpass")
    return sp_signal.filtfilt(b, a, x, axis=0)


def derive_signals(xyz: np.ndarray) -> np.ndarray:
    """Compute [SVM, PC1, PC2] from a 3-axis window.

    Args:
        xyz: [L, 3] array of x/y/z samples within a single window (already HPF'd).

    Returns:
        [L, 3] array of derived signals, in order: SVM, PC1, PC2.
    """
    svm = np.sqrt((xyz ** 2).sum(axis=1))

    # PCA within the window.  3D → 2 principal axes.  If the window is too
    # short or rank-deficient we fall back to (x, y) to stay numerically safe.
    centered = xyz - xyz.mean(axis=0, keepdims=True)
    try:
        # eigendecomposition is cheaper than full SVD for 3x3
        cov = (centered.T @ centered) / max(1, centered.shape[0] - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]  # largest first
        pc = centered @ eigvecs[:, order[:2]]
        pc1, pc2 = pc[:, 0], pc[:, 1]
    except np.linalg.LinAlgError:
        pc1, pc2 = centered[:, 0], centered[:, 1]

    return np.stack([svm, pc1, pc2], axis=1)  # [L, 3]


# ---------------------------------------------------------------------------
# Per-signal feature block (12 features × 3 signals = 36 features)
# ---------------------------------------------------------------------------

_FEATURE_NAMES = (
    "mean",
    "std",
    "min",
    "max",
    "ptp",
    "rms",
    "skew",
    "kurt",
    "mean_cross_rate",
    "dom_freq",
    "spectral_entropy",
    "sparc",
)


def _mean_cross_rate(x: np.ndarray) -> float:
    """Number of mean-crossings per sample.  High for scratch (rapid wobble)."""
    centered = x - x.mean()
    sign = np.sign(centered)
    # ignore zero-valued samples when counting sign flips
    sign[sign == 0] = 1
    crossings = np.sum(np.diff(sign) != 0)
    return crossings / max(1, len(x) - 1)


def _dominant_frequency(x: np.ndarray, fs: int) -> float:
    """Frequency bin with the largest FFT-magnitude (ignoring DC)."""
    if len(x) < 2:
        return 0.0
    mag = np.abs(rfft(x - x.mean()))
    freqs = rfftfreq(len(x), d=1.0 / fs)
    if len(mag) <= 1:
        return 0.0
    # skip DC bin
    peak = int(np.argmax(mag[1:])) + 1
    return float(freqs[peak])


def _spectral_entropy(x: np.ndarray) -> float:
    """Normalised Shannon entropy of the power spectrum."""
    if len(x) < 2:
        return 0.0
    psd = np.abs(rfft(x - x.mean())) ** 2
    psd_sum = psd.sum()
    if psd_sum <= 0:
        return 0.0
    p = psd / psd_sum
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(len(p))) if len(p) > 1 else 0.0


def _sparc(x: np.ndarray, fs: int, fc: float = 10.0) -> float:
    """Spectral Arc Length — a smoothness metric (Balasubramanian 2012).

    More negative values == smoother signal.  We follow the standard
    definition: integrate the arc-length of the normalised magnitude
    spectrum up to a cutoff frequency `fc`.
    """
    n = len(x)
    if n < 4:
        return 0.0
    # zero-pad to next power of 2 for stable FFT
    nfft = 1 << int(np.ceil(np.log2(max(n, 2))))
    mag = np.abs(rfft(x - x.mean(), n=nfft))
    if mag.max() <= 0:
        return 0.0
    mag = mag / mag.max()
    freqs = rfftfreq(nfft, d=1.0 / fs)
    mask = freqs <= fc
    if mask.sum() < 2:
        return 0.0
    f = freqs[mask]
    m = mag[mask]
    df = np.diff(f) / fc  # normalise x-axis to [0, 1]
    dm = np.diff(m)
    return float(-np.sum(np.sqrt(df ** 2 + dm ** 2)))


def _skew(x: np.ndarray) -> float:
    m = x.mean()
    s = x.std()
    if s <= 1e-12:
        return 0.0
    return float(((x - m) ** 3).mean() / s ** 3)


def _kurt(x: np.ndarray) -> float:
    m = x.mean()
    s = x.std()
    if s <= 1e-12:
        return 0.0
    return float(((x - m) ** 4).mean() / s ** 4 - 3.0)


def _signal_features(x: np.ndarray, fs: int) -> list[float]:
    """Compute all 12 features for a single 1-D signal."""
    return [
        float(x.mean()),
        float(x.std()),
        float(x.min()),
        float(x.max()),
        float(x.max() - x.min()),
        float(np.sqrt((x ** 2).mean())),
        _skew(x),
        _kurt(x),
        _mean_cross_rate(x),
        _dominant_frequency(x, fs),
        _spectral_entropy(x),
        _sparc(x, fs),
    ]


def feature_names(prefixes=("svm", "pc1", "pc2")) -> list[str]:
    names: list[str] = []
    for p in prefixes:
        names.extend(f"{p}_{name}" for name in _FEATURE_NAMES)
    return names


def extract_features(xyz: np.ndarray, fs: int = 20, hpf: bool = True) -> np.ndarray:
    """Full per-window feature vector.

    Args:
        xyz: [L, 3] raw accelerometer window (x, y, z).
        fs:  Sampling frequency (Hz).  Default 20 Hz (our dataset).
        hpf: Apply the 0.25 Hz high-pass filter before deriving signals.
    """
    if hpf:
        xyz = highpass_filter(xyz, fs=fs)
    derived = derive_signals(xyz)  # [L, 3]  cols: SVM, PC1, PC2
    feats: list[float] = []
    for i in range(derived.shape[1]):
        feats.extend(_signal_features(derived[:, i], fs))
    return np.asarray(feats, dtype=np.float32)


def extract_raw_segments_batch(
    df: pd.DataFrame,
    seg_col: str = "segment",
    xyz_cols=("x", "y", "z"),
    max_len: int = 1200,
    hpf: bool = True,
    fs: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Return raw (HPF-filtered) segment tensors padded/truncated to ``max_len``.

    Segment order matches :func:`extract_features_batch`, so the handcrafted
    features and any DL-derived features can be concatenated without a realign.

    Returns:
        raw:      [N_segments, max_len, 3] float32 array.
        seg_ids:  [N_segments] str identifier.
    """
    raw_list, seg_list = [], []
    for seg_id, seq in df.groupby(seg_col, sort=False, observed=True):
        xyz = seq.loc[:, list(xyz_cols)].to_numpy(dtype=np.float32)
        if xyz.shape[0] < 4:
            continue
        if hpf:
            xyz = highpass_filter(xyz, fs=fs).astype(np.float32)
        L = xyz.shape[0]
        if L < max_len:
            xyz = np.concatenate(
                [xyz, np.zeros((max_len - L, xyz.shape[1]), dtype=np.float32)], axis=0
            )
        elif L > max_len:
            xyz = xyz[:max_len]
        raw_list.append(xyz)
        seg_list.append(str(seg_id))
    if not raw_list:
        return np.zeros((0, max_len, 3), dtype=np.float32), np.array([], dtype=object)
    return np.stack(raw_list, axis=0), np.asarray(seg_list, dtype=object)


def extract_features_batch(
    df: pd.DataFrame,
    seg_col: str = "segment",
    xyz_cols=("x", "y", "z"),
    fs: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorise feature extraction over all segments in `df`.

    Returns:
        X:        [N_segments, 36] feature matrix.
        y:        [N_segments] binary scratch label (from `segment_scratch`).
        dur:      [N_segments] scratch duration (from `scratch_duration`).
        seg_ids:  [N_segments] segment identifier (str).
    """
    X_list, y_list, dur_list, seg_list = [], [], [], []
    grouped = df.groupby(seg_col, sort=False, observed=True)
    for seg_id, seq in grouped:
        xyz = seq.loc[:, list(xyz_cols)].to_numpy(dtype=np.float32)
        if xyz.shape[0] < 4:
            continue
        X_list.append(extract_features(xyz, fs=fs, hpf=True))
        y_list.append(int(seq["segment_scratch"].any()))
        dur_list.append(float(seq["scratch_duration"].max()))
        seg_list.append(str(seg_id))
    if not X_list:
        empty = np.zeros((0, 36), dtype=np.float32)
        return empty, np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32), np.array([], dtype=object)
    return (
        np.vstack(X_list),
        np.asarray(y_list, dtype=np.int64),
        np.asarray(dur_list, dtype=np.float32),
        np.asarray(seg_list, dtype=object),
    )


def extract_features_from_windows(
    X_win: np.ndarray,
    fs: int = 20,
    hpf: bool = True,
) -> np.ndarray:
    """Compute the 36 hand-crafted features per window.

    Args:
        X_win: [N, win_len, 3] raw signal tensor produced by
               ``baselines_classical.windowing.make_windows``.
        fs:   sampling rate.  Default 20 Hz.
        hpf:  apply 0.25 Hz high-pass before deriving SVM/PC1/PC2.

    Returns:
        X: [N, 36] float32 feature matrix.  Row order matches ``X_win``.
    """
    if X_win.ndim != 3 or X_win.shape[2] != 3:
        raise ValueError(f"X_win must be [N, L, 3]; got {X_win.shape}")
    if X_win.shape[0] == 0:
        return np.zeros((0, 36), dtype=np.float32)
    feats = np.empty((X_win.shape[0], 36), dtype=np.float32)
    for i in range(X_win.shape[0]):
        feats[i] = extract_features(X_win[i], fs=fs, hpf=hpf)
    return feats
