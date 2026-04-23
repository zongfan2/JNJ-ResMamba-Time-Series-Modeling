# -*- coding: utf-8 -*-
"""Small CNN and BiLSTM DL-feature extractors for Paper 2 (Ji et al. 2023).

Each model is trained end-to-end to classify scratch for the current fold;
the 5-dim penultimate-layer activation is then used as a feature vector
alongside the 36 handcrafted time/frequency features and fed to LightGBM.

Architecture (faithful to the paper's description):
  - CNN: 2 × (Conv1d → BN → ReLU → MaxPool), global average pool, then a
         3-layer MLP with output dimensions (16, 5, 1).  Penultimate = 5.
  - BiLSTM: single bidirectional LSTM, mean-pool over time, then the same
            3-layer MLP (16, 5, 1).  Penultimate = 5.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Architectures
# ---------------------------------------------------------------------------


class SmallCNN(nn.Module):
    """2 × (Conv-BN-Pool) + MLP(16, 5, 1).  Penultimate = 5-dim."""

    input_layout: str = "BCL"

    def __init__(self, in_channels: int = 3, penultimate_dim: int = 5):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(4)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, penultimate_dim)   # penultimate
        self.fc3 = nn.Linear(penultimate_dim, 1)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, L] → [B, penultimate_dim]."""
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.global_pool(x).squeeze(-1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc3(self.features(x))


class SmallBiLSTM(nn.Module):
    """BiLSTM + MLP(16, 5, 1).  Penultimate = 5-dim.

    Uses a mask-aware mean pool so that right-padded zero tails do not
    dilute the temporal aggregate (critical when segments have variable
    length and were right-padded to a common `dl_max_len`).
    """

    input_layout: str = "BLC"

    def __init__(self, in_channels: int = 3, hidden_size: int = 32,
                 penultimate_dim: int = 5):
        super().__init__()
        self.lstm = nn.LSTM(
            in_channels, hidden_size,
            batch_first=True, bidirectional=True,
        )
        self.fc1 = nn.Linear(hidden_size * 2, 16)
        self.fc2 = nn.Linear(16, penultimate_dim)   # penultimate
        self.fc3 = nn.Linear(penultimate_dim, 1)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, C] → [B, penultimate_dim]."""
        out, _ = self.lstm(x)  # [B, L, 2*H]
        # Mask: True for timesteps with any non-zero channel (real data).
        # Falls back to uniform mean if a sample is all-zero (unlikely).
        valid = (x.abs().sum(dim=-1) > 0).float()           # [B, L]
        counts = valid.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]
        pooled = (out * valid.unsqueeze(-1)).sum(dim=1) / counts
        x = F.relu(self.fc1(pooled))
        return self.fc2(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc3(self.features(x))


# ---------------------------------------------------------------------------
# Training + feature extraction
# ---------------------------------------------------------------------------


def _prepare_input(x: np.ndarray, layout: str) -> torch.Tensor:
    """Convert [B, L, C] numpy → torch tensor in the layout the model expects."""
    t = torch.from_numpy(x).float()
    if layout == "BCL":
        return t.permute(0, 2, 1).contiguous()
    return t  # BLC


def _group_val_split(n: int, groups, val_frac: float, rng):
    """Subject-grouped train/val split: hold out whole subjects in val.

    Fallback to a random split if ``groups`` is None or has a single unique
    value (which would collapse the split).
    """
    if groups is None:
        idx = rng.permutation(n)
        n_val = max(1, int(n * val_frac))
        return idx[n_val:], idx[:n_val]
    groups = np.asarray(groups)
    unique = np.unique(groups)
    if len(unique) < 2:
        idx = rng.permutation(n)
        n_val = max(1, int(n * val_frac))
        return idx[n_val:], idx[:n_val]
    n_val_groups = max(1, int(round(len(unique) * val_frac)))
    perm = rng.permutation(len(unique))
    val_groups = set(unique[perm[:n_val_groups]].tolist())
    mask_val = np.array([g in val_groups for g in groups])
    return np.where(~mask_val)[0], np.where(mask_val)[0]


def train_dl_model(
    model: nn.Module,
    X: np.ndarray,      # [N, L, C]
    y: np.ndarray,      # [N] binary
    groups=None,        # [N] subject ids (LOSO-aware val split)
    device: str = "cpu",
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    val_frac: float = 0.2,
    patience: int = 5,
    verbose: bool = False,
    seed: int = 42,
) -> nn.Module:
    """Train a DL feature model with class-balanced BCE + early stopping.

    When ``groups`` is provided, the inner train/val split is done at the
    group (subject) level so that early stopping mirrors the outer LOSO
    generalization setting and does not leak within-subject segments
    across train/val.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    tr_idx, val_idx = _group_val_split(len(X), groups, val_frac, rng)

    X_tr_np, y_tr_np = X[tr_idx], y[tr_idx]
    X_va_np, y_va_np = X[val_idx], y[val_idx]

    # Keep train/val tensors on CPU — a full train pool can be multi-GB
    # (LOSO pools easily have >10k segments × 1200 × 3).  We move mini-batches
    # to GPU on demand and free them immediately afterwards.
    X_tr = _prepare_input(X_tr_np, model.input_layout)
    y_tr = torch.from_numpy(y_tr_np).float()
    X_va = _prepare_input(X_va_np, model.input_layout)
    y_va = torch.from_numpy(y_va_np).float()

    train_loader = DataLoader(
        TensorDataset(X_tr, y_tr),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device != "cpu"),
    )
    val_loader = DataLoader(
        TensorDataset(X_va, y_va),
        batch_size=batch_size,        # chunk val too — cuDNN LSTM allocates
        shuffle=False,                # workspace ∝ batch_size, not segment len
        num_workers=0,
        pin_memory=(device != "cpu"),
    )

    n_pos = max(1, int(y_tr_np.sum()))
    n_neg = max(1, len(y_tr_np) - n_pos)
    pos_weight = torch.tensor([n_neg / n_pos], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    waited = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad()
            logits = model(xb).squeeze(-1)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_sum, val_count = 0.0, 0
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                logits = model(xb).squeeze(-1)
                val_sum += criterion(logits, yb).item() * xb.shape[0]
                val_count += xb.shape[0]
            val_loss = val_sum / max(1, val_count)
        if verbose:
            print(f"    dl epoch={epoch:02d}  val_loss={val_loss:.4f}")
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            waited = 0
        else:
            waited += 1
            if waited >= patience:
                break

    model.load_state_dict(best_state)
    # Always emit a one-line summary so it's easy to tell if training
    # actually converged (or early-stopped at random init).
    print(
        f"  [dl] {type(model).__name__:<12s} done  "
        f"best_val_loss={best_val:.4f}  "
        f"epochs_run={epoch + 1}  n_tr={len(tr_idx)} n_va={len(val_idx)}"
    )
    return model


@torch.no_grad()
def extract_penultimate(
    model: nn.Module, X: np.ndarray, device: str = "cpu", batch_size: int = 256,
) -> np.ndarray:
    """Return [N, penultimate_dim] features from a trained DL model."""
    model.to(device).eval()
    out_chunks = []
    for i in range(0, len(X), batch_size):
        xb = _prepare_input(X[i:i + batch_size], model.input_layout).to(device)
        out_chunks.append(model.features(xb).cpu().numpy())
    if not out_chunks:
        return np.zeros((0, 5), dtype=np.float32)
    return np.concatenate(out_chunks, axis=0).astype(np.float32)


def train_and_extract_dl_features(
    X_raw_tr: np.ndarray,
    y_tr: np.ndarray,
    X_raw_te: np.ndarray,
    *,
    groups_tr=None,       # subject ids of training segments (LOSO-aware val split)
    oof_folds: int = 0,   # 0 = naive (leaky training features); >=2 = OOF stacking
    device: str = "cpu",
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    seed: int = 42,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Train SmallCNN + SmallBiLSTM and return penultimate features.

    Data leakage controls:
      - ``groups_tr`` (required for LOSO correctness): subject ids for each
        training segment.  Used to hold whole subjects out in the DL
        early-stopping validation split — mirrors the outer LOSO setting.
      - ``oof_folds`` (0 | 2+): if 0, training features are extracted from the
        same model that trained on them (faster but features encode their own
        labels).  If >=2, use group-K-fold OOF stacking: for each inner fold,
        train DL on K-1 subject groups and extract features on the held-out
        group.  The final training features are leak-free.  The test features
        still come from a model trained on all training data.
    """
    def _free_cuda():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _fit_standardizer(X):
        """Per-channel mean/std computed over non-zero (real) samples only.

        Right-padded zero tails are excluded so the standardiser reflects
        the distribution of actual accelerometer data, not the padding.
        """
        valid = (np.abs(X).sum(axis=-1) > 0)  # [N, L] bool
        # reshape to [T, C] over valid points per channel
        flat = X.reshape(-1, X.shape[-1])
        mask = valid.reshape(-1)
        real = flat[mask]
        if real.size == 0:
            real = flat  # fallback (shouldn't happen)
        mean = real.mean(axis=0).astype(np.float32)
        std = real.std(axis=0).astype(np.float32)
        std = np.where(std < 1e-6, 1.0, std)
        return mean, std

    def _standardize(X, mean, std):
        """Apply z-score; preserve zero padding (pad stays 0 after scaling)."""
        valid = (np.abs(X).sum(axis=-1) > 0)  # [N, L]
        out = (X - mean) / std
        out[~valid] = 0.0
        return out.astype(np.float32)

    def _train_pair(X, y, groups, seed):
        cnn = SmallCNN(in_channels=X.shape[-1])
        bilstm = SmallBiLSTM(in_channels=X.shape[-1])
        train_dl_model(cnn, X, y, groups=groups, device=device, epochs=epochs,
                       batch_size=batch_size, lr=lr, seed=seed, verbose=verbose)
        _free_cuda()
        train_dl_model(bilstm, X, y, groups=groups, device=device, epochs=epochs,
                       batch_size=batch_size, lr=lr, seed=seed + 1, verbose=verbose)
        _free_cuda()
        return cnn, bilstm

    def _extract_pair(cnn, bilstm, X):
        out = np.concatenate([
            extract_penultimate(cnn, X, device=device),
            extract_penultimate(bilstm, X, device=device),
        ], axis=1)
        _free_cuda()
        return out

    # ---- Test-set features: trained on ALL training data, standardised by
    # training-pool channel stats.
    mean_full, std_full = _fit_standardizer(X_raw_tr)
    X_tr_std = _standardize(X_raw_tr, mean_full, std_full)
    X_te_std = _standardize(X_raw_te, mean_full, std_full)
    if verbose:
        print(f"  [dl] train channel mean={mean_full.tolist()} std={std_full.tolist()}")
        print("  [dl] training final CNN+BiLSTM on full training data ...")
    cnn_full, bilstm_full = _train_pair(X_tr_std, y_tr, groups_tr, seed)
    te = _extract_pair(cnn_full, bilstm_full, X_te_std)

    # ---- Training-set features: naive or OOF
    if oof_folds and oof_folds >= 2 and groups_tr is not None:
        from sklearn.model_selection import GroupKFold
        gkf = GroupKFold(n_splits=oof_folds)
        tr = np.zeros((len(X_raw_tr), 10), dtype=np.float32)
        if verbose:
            print(f"  [dl] OOF features over {oof_folds} group folds ...")
        for fold_i, (tr_idx, va_idx) in enumerate(
            gkf.split(X_raw_tr, y_tr, groups=groups_tr)
        ):
            if verbose:
                print(f"    [dl-oof] fold {fold_i + 1}/{oof_folds} "
                      f"train={len(tr_idx)}  holdout={len(va_idx)}")
            # Each OOF fold must re-fit its own standardiser on its own
            # training subjects — otherwise the holdout's statistics leak in.
            mean_k, std_k = _fit_standardizer(X_raw_tr[tr_idx])
            X_in = _standardize(X_raw_tr[tr_idx], mean_k, std_k)
            X_hold = _standardize(X_raw_tr[va_idx], mean_k, std_k)
            cnn_k, bilstm_k = _train_pair(
                X_in, y_tr[tr_idx],
                np.asarray(groups_tr)[tr_idx],
                seed + 10 * (fold_i + 1),
            )
            tr[va_idx] = _extract_pair(cnn_k, bilstm_k, X_hold)
    else:
        tr = _extract_pair(cnn_full, bilstm_full, X_tr_std)

    return tr, te
