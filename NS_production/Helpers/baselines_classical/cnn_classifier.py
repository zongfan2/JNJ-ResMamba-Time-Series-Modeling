# -*- coding: utf-8 -*-
"""sklearn-style adapter wrapping the Xing 2024 ConvNormPool CNN.

Lets ``training/train_classical.py`` plug the Xing et al. 2024 (MDPI Sensors
24(11):3364) deep-learning baseline into the same windowed pipeline used by
the classical (XGBoost/LightGBM/GBM) baselines — Mahadevan/Ji-style 3-s
windows + count-rule aggregation.

Paper-faithful training recipe:
  - Encoder: 3 × ConvNormPool (Conv1d → BN → Swish → MaxPool), hidden=128,
    kernel=5, pool=2.
  - Head: GlobalAvgPool → 2 FC → 1 binary logit (single-task; the
    multi-task heads from ``models/mdpi2024_cnn.py`` are dropped here so the
    CNN matches the paper's window classification objective).
  - Loss: focal loss (γ=2 by default).
  - Sampling: 1:1 class-balanced via ``WeightedRandomSampler``.
  - Optimiser: Adam(lr=1e-3, weight_decay=1e-3), StepLR(step=10, γ=0.1).
  - 200 epochs, batch=16.

The resulting object exposes ``.fit / .predict / .predict_proba`` so it slots
into ``train_classical.py``'s existing classifier path with no further
changes — bout-level pr3 still comes from the count rule, hurdle gate still
applies, CSV schema is unchanged.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


# ---------------------------------------------------------------------------
# Encoder — single-task variant of MDPI2024CNN (classification head only).
# ---------------------------------------------------------------------------


class _BinaryConvNormPoolCNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        hidden_size: int = 128,
        kernel_size: int = 5,
        pool_size: int = 2,
        num_blocks: int = 3,
        fc_dim: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        padding = kernel_size // 2

        def block(in_c: int, out_c: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv1d(in_c, out_c, kernel_size, padding=padding),
                nn.BatchNorm1d(out_c),
                nn.SiLU(),  # Swish
                nn.MaxPool1d(pool_size),
            )

        layers = [block(in_channels, hidden_size)]
        for _ in range(num_blocks - 1):
            layers.append(block(hidden_size, hidden_size))
        self.encoder = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(hidden_size, fc_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(fc_dim, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C] → [B, C, T] for Conv1d.
        h = x.permute(0, 2, 1)
        h = self.encoder(h)
        h = self.gap(h).squeeze(-1)
        h = F.silu(self.fc1(h))
        h = self.dropout(h)
        return self.classifier(h).squeeze(-1)  # [B]


def _binary_focal_loss_with_logits(
    logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    pt = torch.where(targets > 0.5, p, 1.0 - p)
    return ((1.0 - pt) ** gamma * bce).mean()


# ---------------------------------------------------------------------------
# sklearn-style adapter
# ---------------------------------------------------------------------------


class WindowedCNNClassifier:
    """Window-level binary CNN with sklearn-compatible interface.

    Args mirror the Xing 2024 (MDPI Sensors) recipe; defaults are paper values.
    Inputs to ``fit`` / ``predict`` / ``predict_proba`` must be 3-D float arrays
    of shape ``[N, win_len, in_channels]`` — exactly what
    ``baselines_classical/windowing.py::make_windows`` returns.
    """

    def __init__(
        self,
        hidden_size: int = 128,
        kernel_size: int = 5,
        pool_size: int = 2,
        num_blocks: int = 3,
        fc_dim: int = 128,
        dropout: float = 0.0,
        epochs: int = 200,
        batch_size: int = 16,
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        lr_step: int = 10,
        lr_gamma: float = 0.1,
        focal_gamma: float = 2.0,
        device: str | None = None,
        random_state: int = 42,
        verbose: bool = True,
    ):
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.num_blocks = num_blocks
        self.fc_dim = fc_dim
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_step = lr_step
        self.lr_gamma = lr_gamma
        self.focal_gamma = focal_gamma
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.random_state = random_state
        self.verbose = verbose
        self.model_: _BinaryConvNormPoolCNN | None = None
        self.in_channels_: int | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight=None) -> "WindowedCNNClassifier":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if X.ndim != 3:
            raise ValueError(
                f"WindowedCNNClassifier expects X of shape [N, T, C]; got {X.shape}"
            )
        self.in_channels_ = int(X.shape[2])

        device = torch.device(self.device)
        self.model_ = _BinaryConvNormPoolCNN(
            in_channels=self.in_channels_,
            hidden_size=self.hidden_size,
            kernel_size=self.kernel_size,
            pool_size=self.pool_size,
            num_blocks=self.num_blocks,
            fc_dim=self.fc_dim,
            dropout=self.dropout,
        ).to(device)

        # Paper convention: 1:1 class-balanced sampling each epoch.  Per-class
        # weights produce equal expected positive/negative counts per draw;
        # ``replacement=True`` matches the paper's resampling behaviour.
        n_pos = max(1, int(y.sum()))
        n_neg = max(1, len(y) - n_pos)
        weights = np.where(y > 0.5, 1.0 / n_pos, 1.0 / n_neg).astype(np.float64)
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(weights),
            num_samples=len(weights),
            replacement=True,
        )
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        loader = DataLoader(ds, batch_size=self.batch_size, sampler=sampler)

        opt = torch.optim.Adam(
            self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        sch = torch.optim.lr_scheduler.StepLR(
            opt, step_size=self.lr_step, gamma=self.lr_gamma
        )

        log_every = max(1, self.epochs // 10)
        self.model_.train()
        for epoch in range(self.epochs):
            running, n_batches = 0.0, 0
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = self.model_(xb)
                loss = _binary_focal_loss_with_logits(
                    logits, yb, gamma=self.focal_gamma
                )
                opt.zero_grad()
                loss.backward()
                opt.step()
                running += float(loss.item())
                n_batches += 1
            sch.step()
            if self.verbose and ((epoch + 1) % log_every == 0 or epoch == 0):
                lr_now = sch.get_last_lr()[0]
                print(
                    f"    [cnn] epoch {epoch + 1:3d}/{self.epochs}  "
                    f"loss={running / max(1, n_batches):.4f}  lr={lr_now:.2e}"
                )
        return self

    @torch.no_grad()
    def _logits(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("WindowedCNNClassifier is not fitted")
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 3:
            raise ValueError(
                f"WindowedCNNClassifier expects X of shape [N, T, C]; got {X.shape}"
            )
        device = next(self.model_.parameters()).device
        self.model_.eval()
        out = []
        bs = max(1, self.batch_size * 4)
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i : i + bs]).to(device)
            out.append(self.model_(xb).detach().cpu().numpy())
        if not out:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(out, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = self._logits(X)
        # Numerically stable sigmoid.
        p = np.where(
            logits >= 0,
            1.0 / (1.0 + np.exp(-logits)),
            np.exp(logits) / (1.0 + np.exp(logits)),
        ).astype(np.float64)
        return np.stack([1.0 - p, p], axis=1)

    # joblib uses pickle; PyTorch modules pickle fine on CPU.  Move there
    # before serialising so the saved file is portable.
    def __getstate__(self):
        state = self.__dict__.copy()
        if self.model_ is not None:
            state["model_"] = self.model_.cpu()
        return state
