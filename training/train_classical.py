# -*- coding: utf-8 -*-
"""Classical (feature-engineering + ML) baselines for Deep Scratch.

Implements two literature baselines that sit on top of the same LOFO
data-loading pipeline as train_scratch.py, and emits predictions in the
same ``Y_test_Y_pred_test_subject_<FOLD>.csv`` format so they land directly
in evaluation/prediction_analysis.py.

Supported classifiers (set via ``model.architecture`` in the YAML):
  - ``mahadevan2021``  Random Forest (50 trees) on 26-of-36 RFECV-selected
                       features.  Paper 1 (Mahadevan et al., npj Digit. Med.
                       2021).  Feature set: SVM/PC1/PC2 × 12 time/freq stats.
  - ``ji2023``         LightGBM with ``scale_pos_weight`` on the same 36
                       features.  Paper 2 (Ji et al., npj Digit. Med. 2023) —
                       pragmatic replication that omits the TDA and DL
                       feature blocks.

Usage:
    python3.11 training/train_classical.py \
        --config experiments/configs/ablation/ablation_baseline_mahadevan2021.yaml
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Pip-preamble (mirrors training/train_scratch.py).  Ensures the classical
# baselines' deps (xgboost, lightgbm) are present on Domino where the base
# env doesn't include them.  Quiet — failures fall through to a sklearn
# HistGradientBoosting fallback in build_classifier / build_regressor.
# ---------------------------------------------------------------------------
_shell_script = '''
cd munge/predictive_modeling 2>/dev/null || true
sudo python3.11 -m pip install --quiet xgboost lightgbm tsfresh 2>/dev/null || \
python3.11 -m pip install --quiet xgboost lightgbm tsfresh 2>/dev/null || true
'''
try:
    subprocess.run(_shell_script, shell=True, capture_output=True, text=True, timeout=180)
except Exception:
    pass

import joblib
import numpy as np
import pandas as pd
import yaml

# Make the project root importable regardless of launch dir.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data.loading import load_data  # noqa: E402
from utils.common import create_folder  # noqa: E402

from baselines_classical.features import (  # noqa: E402
    extract_features_batch,
    extract_raw_segments_batch,
    feature_names,
)


# ---------------------------------------------------------------------------
# Config / CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Classical ML baselines for Deep Scratch.")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--single_fold", type=str, default="",
                   help="Restrict LOFO loop to one fold (e.g. FOLD4).")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Classifier factories
# ---------------------------------------------------------------------------


def build_classifier(arch: str, overrides: dict):
    """Return a fitted-on-demand sklearn-compatible classifier."""
    if arch == "mahadevan2021":
        # XGBoost (per user request, replacing paper 1's RandomForest).
        xgb = _try_import_xgboost()
        if xgb is not None:
            return xgb.XGBClassifier(
                n_estimators=int(overrides.get("n_estimators", 300)),
                learning_rate=float(overrides.get("learning_rate", 0.05)),
                max_depth=int(overrides.get("max_depth", 6)),
                min_child_weight=float(overrides.get("min_child_weight", 1.0)),
                reg_lambda=float(overrides.get("reg_lambda", 1.0)),
                subsample=float(overrides.get("subsample", 0.9)),
                colsample_bytree=float(overrides.get("colsample_bytree", 0.9)),
                random_state=int(overrides.get("random_state", 42)),
                n_jobs=int(overrides.get("n_jobs", -1)),
                eval_metric="logloss",
                tree_method="hist",
            )
        from sklearn.ensemble import HistGradientBoostingClassifier
        print("[classical] xgboost unavailable — falling back to "
              "sklearn.HistGradientBoostingClassifier.")
        return HistGradientBoostingClassifier(
            max_iter=int(overrides.get("n_estimators", 300)),
            learning_rate=float(overrides.get("learning_rate", 0.05)),
            max_depth=None if int(overrides.get("max_depth", -1)) < 0 else int(overrides["max_depth"]),
            min_samples_leaf=int(overrides.get("min_child_samples", 20)),
            l2_regularization=float(overrides.get("reg_lambda", 1.0)),
            random_state=int(overrides.get("random_state", 42)),
        )
    elif arch == "ji2023":
        lgb = _try_import_lightgbm()
        if lgb is not None:
            # YAML `scale_pos_weight: null` means "auto" — set at fit time in
            # run_cv from the train-fold neg/pos ratio.  Use 1.0 as the init
            # placeholder; run_cv overwrites before fit.
            spw_override = overrides.get("scale_pos_weight")
            spw_init = float(spw_override) if spw_override is not None else 1.0
            return lgb.LGBMClassifier(
                n_estimators=int(overrides.get("n_estimators", 500)),
                learning_rate=float(overrides.get("learning_rate", 0.05)),
                num_leaves=int(overrides.get("num_leaves", 31)),
                max_depth=int(overrides.get("max_depth", -1)),
                min_child_samples=int(overrides.get("min_child_samples", 20)),
                reg_lambda=float(overrides.get("reg_lambda", 0.0)),
                scale_pos_weight=spw_init,
                random_state=int(overrides.get("random_state", 42)),
                n_jobs=int(overrides.get("n_jobs", -1)),
                verbose=-1,
            )
        # Fallback: sklearn's histogram-GBDT — closest always-available analog
        # to LightGBM.  Class imbalance is handled via sample_weight at fit time
        # (see run_lofo); HistGradientBoosting doesn't expose scale_pos_weight.
        from sklearn.ensemble import HistGradientBoostingClassifier
        print("[classical] lightgbm unavailable — falling back to "
              "sklearn.HistGradientBoostingClassifier.")
        return HistGradientBoostingClassifier(
            max_iter=int(overrides.get("n_estimators", 500)),
            learning_rate=float(overrides.get("learning_rate", 0.05)),
            max_leaf_nodes=int(overrides.get("num_leaves", 31)),
            max_depth=None if int(overrides.get("max_depth", -1)) < 0 else int(overrides["max_depth"]),
            min_samples_leaf=int(overrides.get("min_child_samples", 20)),
            l2_regularization=float(overrides.get("reg_lambda", 0.0)),
            random_state=int(overrides.get("random_state", 42)),
        )
    elif arch == "mdpi2024_fe":
        # Xing et al. 2024 feature-engineering arm — sklearn GradientBoosting
        # on ~2353 tsfresh features.  Paper uses the straight sklearn GBM
        # (not XGBoost/LightGBM), so we match that.
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(
            n_estimators=int(overrides.get("n_estimators", 200)),
            learning_rate=float(overrides.get("learning_rate", 0.05)),
            max_depth=int(overrides.get("max_depth", 3)),
            subsample=float(overrides.get("subsample", 1.0)),
            random_state=int(overrides.get("random_state", 42)),
        )
    else:
        raise ValueError(f"Unknown classical baseline architecture: {arch}")


def _try_import_lightgbm():
    """Try importing lightgbm; attempt a one-shot pip install on miss."""
    try:
        import lightgbm as lgb
        return lgb
    except ImportError:
        pass
    import subprocess
    print("[classical] lightgbm not found — attempting `pip install lightgbm` ...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "lightgbm"],
            check=True,
        )
        import lightgbm as lgb  # noqa: F401
        return lgb
    except Exception as exc:
        print(f"[classical] lightgbm install failed: {exc}")
        return None


def _try_import_xgboost():
    """Try importing xgboost; attempt a one-shot pip install on miss."""
    try:
        import xgboost as xgb
        return xgb
    except ImportError:
        pass
    import subprocess
    print("[classical] xgboost not found — attempting `pip install xgboost` ...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "xgboost"],
            check=True,
        )
        import xgboost as xgb  # noqa: F401
        return xgb
    except Exception as exc:
        print(f"[classical] xgboost install failed: {exc}")
        return None


def build_regressor(arch: str, overrides: dict):
    """Return a duration regressor matching the architecture family.

    mahadevan2021 → XGBRegressor  (with HistGradientBoosting fallback)
    ji2023        → LGBMRegressor (with HistGradientBoosting fallback)
    """
    if arch == "mahadevan2021":
        xgb = _try_import_xgboost()
        if xgb is not None:
            return xgb.XGBRegressor(
                n_estimators=int(overrides.get("reg_n_estimators", overrides.get("n_estimators", 300))),
                learning_rate=float(overrides.get("reg_learning_rate", overrides.get("learning_rate", 0.05))),
                max_depth=int(overrides.get("reg_max_depth", overrides.get("max_depth", 6))),
                reg_lambda=float(overrides.get("reg_lambda", 1.0)),
                subsample=float(overrides.get("subsample", 0.9)),
                colsample_bytree=float(overrides.get("colsample_bytree", 0.9)),
                random_state=int(overrides.get("random_state", 42)),
                n_jobs=int(overrides.get("n_jobs", -1)),
                objective="reg:squarederror",
                tree_method="hist",
            )
    elif arch == "ji2023":
        lgb = _try_import_lightgbm()
        if lgb is not None:
            return lgb.LGBMRegressor(
                n_estimators=int(overrides.get("reg_n_estimators", overrides.get("n_estimators", 500))),
                learning_rate=float(overrides.get("reg_learning_rate", overrides.get("learning_rate", 0.05))),
                num_leaves=int(overrides.get("num_leaves", 31)),
                max_depth=int(overrides.get("max_depth", -1)),
                min_child_samples=int(overrides.get("min_child_samples", 20)),
                reg_lambda=float(overrides.get("reg_lambda", 0.0)),
                random_state=int(overrides.get("random_state", 42)),
                n_jobs=int(overrides.get("n_jobs", -1)),
                verbose=-1,
            )
    elif arch == "mdpi2024_fe":
        # Xing et al. 2024 — sklearn GradientBoostingRegressor for duration.
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(
            n_estimators=int(overrides.get("reg_n_estimators", overrides.get("n_estimators", 200))),
            learning_rate=float(overrides.get("reg_learning_rate", overrides.get("learning_rate", 0.05))),
            max_depth=int(overrides.get("reg_max_depth", overrides.get("max_depth", 3))),
            random_state=int(overrides.get("random_state", 42)),
        )
    else:
        raise ValueError(f"Unknown classical baseline architecture: {arch}")
    # Fallback: sklearn HistGradientBoostingRegressor
    from sklearn.ensemble import HistGradientBoostingRegressor
    print("[classical] boosted-tree regressor unavailable — falling back to "
          "sklearn.HistGradientBoostingRegressor.")
    return HistGradientBoostingRegressor(
        max_iter=int(overrides.get("reg_n_estimators", overrides.get("n_estimators", 300))),
        learning_rate=float(overrides.get("reg_learning_rate", overrides.get("learning_rate", 0.05))),
        max_depth=None if int(overrides.get("max_depth", -1)) < 0 else int(overrides["max_depth"]),
        min_samples_leaf=int(overrides.get("min_child_samples", 20)),
        l2_regularization=float(overrides.get("reg_lambda", 0.0)),
        random_state=int(overrides.get("random_state", 42)),
    )


# ---------------------------------------------------------------------------
# Feature selection (Paper 1) — RFECV with decision-tree estimator → 26 feats
# ---------------------------------------------------------------------------


def select_features_rfecv(X: np.ndarray, y: np.ndarray, target_n: int = 26) -> np.ndarray:
    """Return the column indices of the top-`target_n` features via RFECV.

    Following Paper 1 (Mahadevan 2021): recursive feature elimination with a
    DecisionTree estimator; they report 26/36 selected.  We use RFE with a
    fixed target to keep runtime bounded and deterministic.
    """
    from sklearn.feature_selection import RFE
    from sklearn.tree import DecisionTreeClassifier

    estimator = DecisionTreeClassifier(random_state=42)
    n_feats = X.shape[1]
    target = min(target_n, n_feats)
    rfe = RFE(estimator=estimator, n_features_to_select=target, step=1)
    rfe.fit(X, y)
    return np.where(rfe.support_)[0]


# ---------------------------------------------------------------------------
# Class balancing (Paper 1) — random subsample the majority class to 50:50
# ---------------------------------------------------------------------------


def balance_classes(
    X: np.ndarray, y: np.ndarray, dur: np.ndarray, seg_ids: np.ndarray, rng: np.random.Generator
):
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return X, y, dur, seg_ids
    n = min(len(pos_idx), len(neg_idx))
    keep = np.concatenate([
        rng.choice(pos_idx, size=n, replace=False),
        rng.choice(neg_idx, size=n, replace=False),
    ])
    rng.shuffle(keep)
    return X[keep], y[keep], dur[keep], seg_ids[keep]


# ---------------------------------------------------------------------------
# LOFO loop
# ---------------------------------------------------------------------------


def _build_predictions_df(
    df_test: pd.DataFrame,
    seg_to_pr1: dict,
    seg_to_pr1_prob: dict,
    seg_to_pr3: dict,
) -> pd.DataFrame:
    """Assemble a test_predictions DataFrame matching train_scratch.py schema.

    Columns: segment, position, inTSO, predictTSO, gt1, gt2, gt3,
             pr1, pr1_probs, pr2, pr2_probs, pr3.

    Classical baselines predict at the segment level only, so we broadcast
    per-segment predictions to every frame in the segment.  ``pr2`` is a
    broadcast of ``pr1`` so the per-frame F1 metric is well-defined.
    """
    out = pd.DataFrame({
        "segment":     df_test["segment"].astype(str).values,
        "position":    df_test.groupby("segment").cumcount().values + 1,
        "inTSO":       df_test.get("inTSO", pd.Series(True, index=df_test.index)).values,
        "predictTSO":  df_test.get("predictTSO", df_test.get("inTSO", pd.Series(True, index=df_test.index))).values,
        "gt1":         df_test["segment_scratch"].astype(int).values,
        "gt2":         df_test["scratch"].astype(int).values,
        "gt3":         df_test["scratch_duration"].astype(float).values,
    })
    out["pr1"]       = out["segment"].map(seg_to_pr1).fillna(0).astype(int)
    out["pr1_probs"] = out["segment"].map(seg_to_pr1_prob).fillna(0.0).astype(float)
    out["pr2"]       = out["pr1"]
    out["pr2_probs"] = out["pr1_probs"]
    out["pr3"]       = out["segment"].map(seg_to_pr3).fillna(0.0).astype(float)
    return out


def run_cv(cfg: dict, args) -> None:
    """Cross-validated classical ML baseline.

    Supports ``testing: LOFO`` (group by ``FOLD`` — 4 splits) and
    ``testing: LOSO`` (group by ``PID`` — one split per subject).
    """
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})

    input_folder = data_cfg["input_folder"]
    arch = model_cfg["architecture"]
    overrides = model_cfg.get("overrides") or {}
    output_name = train_cfg["output"]
    testing = str(train_cfg.get("testing", "LOFO")).upper()
    if testing not in ("LOFO", "LOSO"):
        raise ValueError(f"training.testing must be LOFO or LOSO (got {testing!r})")
    group_col = "FOLD" if testing == "LOFO" else "PID"

    # ---- Output layout mirrors train_scratch.py ----
    dataset_name = os.path.basename(input_folder.rstrip("/raw"))
    results_folder = Path(
        f"/mnt/data/GENEActive-featurized/results/DL/{dataset_name}/{output_name}/"
    )
    predictions_folder = results_folder / "training" / "predictions"
    models_folder = results_folder / "training" / "model_weights"
    logs_folder = results_folder / "training" / "training_logs"
    create_folder([str(predictions_folder), str(models_folder), str(logs_folder)])

    single_fold = args.single_fold or train_cfg.get("single_fold") or ""

    # ---- Load data once; split per group ----
    print(f"[classical] loading data from {input_folder}  (CV={testing})")
    # filter_type=None disables the built-in segment-length/energy filter in
    # load_data; classical baselines train on all segments (motion and
    # non-motion) so the LOSO test set isn't silently pre-filtered.
    df = load_data(input_folder, filter_type=None)
    df[group_col] = df[group_col].astype(str)
    all_groups = sorted(df[group_col].unique())
    groups = [single_fold] if single_fold else all_groups
    print(f"[classical] {len(groups)} CV split(s) grouped by '{group_col}'")

    rng = np.random.default_rng(int(overrides.get("random_state", 42)))
    feat_names = feature_names()

    for fold in groups:
        print(f"\n[classical] ==== {group_col}={fold} ====  arch={arch}")
        t0 = time.time()
        df_train = df[df[group_col] != fold]
        df_test = df[df[group_col] == fold]
        if df_test.empty:
            print(f"  [skip] {group_col}={fold} has no test rows")
            continue

        # ---- Feature extraction ----
        feature_set = str(overrides.get("feature_set", "default")).lower()
        if feature_set in ("tsfresh", "mdpi2024"):
            from baselines_classical.tsfresh_features import extract_tsfresh_features_batch
            ts_jobs = int(overrides.get("tsfresh_n_jobs", -1))  # -1 = all CPUs
            ts_set = str(overrides.get("tsfresh_feature_set", "comprehensive")).lower()
            ts_progress = bool(overrides.get("tsfresh_progress", True))
            print(f"  extracting tsfresh features on {df_train['segment'].nunique()} train segments ...")
            X_tr, y_tr, dur_tr, seg_tr, tsf_names = extract_tsfresh_features_batch(
                df_train,
                n_jobs=ts_jobs,
                feature_set_type=ts_set,
                disable_progressbar=not ts_progress,
            )
            print(f"  extracting tsfresh features on {df_test['segment'].nunique()} test  segments ...")
            X_te, y_te, dur_te, seg_te, _ = extract_tsfresh_features_batch(
                df_test,
                n_jobs=ts_jobs,
                feature_set_type=ts_set,
                disable_progressbar=not ts_progress,
            )
            feat_names_fold = tsf_names
        else:
            print(f"  extracting features on {df_train['segment'].nunique()} train segments ...")
            X_tr, y_tr, dur_tr, seg_tr = extract_features_batch(df_train)
            print(f"  extracting features on {df_test['segment'].nunique()} test  segments ...")
            X_te, y_te, dur_te, seg_te = extract_features_batch(df_test)
            feat_names_fold = feat_names
        if X_tr.size == 0 or X_te.size == 0:
            print("  [skip] empty feature matrix")
            continue

        # ---- Clean up NaNs (e.g. constant segments) ----
        X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0)
        X_te = np.nan_to_num(X_te, nan=0.0, posinf=0.0, neginf=0.0)

        # ---- Standardize features (Xing 2024 paper step; safe for others) ----
        if bool(overrides.get("standardize_features", feature_set in ("tsfresh", "mdpi2024"))):
            mu = X_tr.mean(axis=0, keepdims=True)
            sd = X_tr.std(axis=0, keepdims=True)
            sd = np.where(sd < 1e-8, 1.0, sd)
            X_tr = ((X_tr - mu) / sd).astype(np.float32)
            X_te = ((X_te - mu) / sd).astype(np.float32)

        # ---- Optional: DL-derived features (Paper 2) ----
        use_dl = bool(overrides.get("use_dl_features", False))
        if use_dl:
            from baselines_classical.dl_features import train_and_extract_dl_features
            try:
                import torch as _torch
                dl_device = "cuda" if _torch.cuda.is_available() else "cpu"
            except ImportError:
                dl_device = "cpu"
            dl_max_len = int(overrides.get("dl_max_len", 1200))
            dl_epochs = int(overrides.get("dl_epochs", 30))
            dl_bs = int(overrides.get("dl_batch_size", 128))
            dl_lr = float(overrides.get("dl_learning_rate", 1e-3))
            oof_folds = int(overrides.get("dl_oof_folds", 5))
            print(f"  [dl] extracting raw segments (max_len={dl_max_len}) for DL features on {dl_device} ...")
            raw_tr, raw_seg_tr = extract_raw_segments_batch(df_train, max_len=dl_max_len)
            raw_te, raw_seg_te = extract_raw_segments_batch(df_test,  max_len=dl_max_len)
            # Realign raw tensors to the feature-batch segment order (defensive).
            if not np.array_equal(raw_seg_tr, seg_tr):
                order = pd.Series(np.arange(len(raw_seg_tr)), index=raw_seg_tr).reindex(seg_tr).values
                raw_tr = raw_tr[order]
            if not np.array_equal(raw_seg_te, seg_te):
                order = pd.Series(np.arange(len(raw_seg_te)), index=raw_seg_te).reindex(seg_te).values
                raw_te = raw_te[order]
            # Subject ids per segment — required for subject-grouped DL val
            # split and OOF-stacking (avoids within-subject leakage).
            pid_by_seg = df_train.groupby("segment", sort=False, observed=True)["PID"].first().to_dict()
            groups_tr = np.asarray([pid_by_seg.get(s) for s in seg_tr])
            print(f"  [dl] subject-grouped val split: {len(np.unique(groups_tr))} unique PIDs; "
                  f"oof_folds={oof_folds if oof_folds >= 2 else 0}")
            dl_tr, dl_te = train_and_extract_dl_features(
                raw_tr, y_tr.astype(np.int64), raw_te,
                groups_tr=groups_tr,
                oof_folds=oof_folds if oof_folds >= 2 else 0,
                device=dl_device, epochs=dl_epochs,
                batch_size=dl_bs, lr=dl_lr,
                seed=int(overrides.get("random_state", 42)),
                verbose=bool(overrides.get("dl_verbose", False)),
            )
            X_tr = np.concatenate([X_tr, dl_tr], axis=1).astype(np.float32)
            X_te = np.concatenate([X_te, dl_te], axis=1).astype(np.float32)
            print(f"  [dl] appended 10 DL features → X_tr {X_tr.shape}, X_te {X_te.shape}")

        # ---- Balance classes (Paper 1 convention; Paper 2 uses scale_pos_weight) ----
        if arch == "mahadevan2021":
            X_tr, y_tr, dur_tr, seg_tr = balance_classes(X_tr, y_tr, dur_tr, seg_tr, rng)
            print(f"  balanced train set: n={len(y_tr)}  pos={int(y_tr.sum())}")

        # ---- Feature selection (Paper 1 only; Paper 2 uses all 36 by default) ----
        if arch == "mahadevan2021":
            target_n = int(overrides.get("num_features", 26))
            keep = select_features_rfecv(X_tr, y_tr, target_n=target_n)
            X_tr = X_tr[:, keep]
            X_te = X_te[:, keep]
            selected = [feat_names_fold[i] for i in keep]
            print(f"  selected {len(keep)} features")
        else:
            keep = np.arange(X_tr.shape[1])

        # ---- Fit classifier ----
        clf = build_classifier(arch, overrides)
        clf_fit_kwargs = {}
        if arch == "ji2023":
            n_pos = max(1, int(y_tr.sum()))
            n_neg = max(1, len(y_tr) - n_pos)
            override_spw = overrides.get("scale_pos_weight")
            spw = float(override_spw) if override_spw is not None else n_neg / n_pos
            if hasattr(clf, "scale_pos_weight"):
                clf.scale_pos_weight = spw
            else:
                clf_fit_kwargs["sample_weight"] = np.where(y_tr == 1, spw, 1.0)
        print(f"  fitting {type(clf).__name__} (classifier) ...")
        clf.fit(X_tr, y_tr, **clf_fit_kwargs)

        # ---- Fit duration regressor (two-stage hurdle model) ----
        # `scratch_duration` is zero-inflated: every non-scratch segment is
        # exactly 0, a smaller positive tail for scratch segments.  A naive
        # regressor trained on all segments predicts ≈ 0 (matches the mode),
        # giving R^2 ≈ 0.  Train the regressor ONLY on positive segments —
        # it now learns E[duration | scratch=1], which has a well-behaved
        # non-degenerate distribution — and gate its output by the
        # classifier's probability:  pr3 = p_scratch * E[duration | scratch].
        # Optional log1p transform further symmetrises the positive tail.
        hurdle = bool(overrides.get("reg_hurdle", True))
        log_target = bool(overrides.get("reg_log_target", True))

        reg = build_regressor(arch, overrides)
        if hurdle and (y_tr == 1).any():
            pos_mask = (y_tr == 1)
            X_reg = X_tr[pos_mask]
            y_reg = dur_tr[pos_mask].astype(np.float32)
            n_reg = int(pos_mask.sum())
            print(f"  fitting {type(reg).__name__} (hurdle duration regressor) "
                  f"on {n_reg} positive-only segments; log_target={log_target}")
        else:
            X_reg = X_tr
            y_reg = dur_tr.astype(np.float32)
            print(f"  fitting {type(reg).__name__} (duration regressor) "
                  f"on {len(y_reg)} segments (hurdle disabled)")

        y_fit = np.log1p(y_reg) if log_target else y_reg
        reg.fit(X_reg, y_fit)

        # ---- Predict on held-out fold ----
        pr1 = clf.predict(X_te).astype(int)
        if hasattr(clf, "predict_proba"):
            pr1_prob = clf.predict_proba(X_te)[:, 1]
        else:
            pr1_prob = pr1.astype(float)

        reg_raw = reg.predict(X_te).astype(float)
        if log_target:
            reg_raw = np.expm1(reg_raw)
        reg_raw = np.clip(reg_raw, 0.0, None)  # duration is non-negative
        # NOTE: do NOT multiply by pr1_prob here.  evaluation/prediction_analysis.py
        # applies a hard gate `dfp.loc[dfp.pr1 == 0, 'pr3'] = 0` during metric
        # computation, so a soft probability gate here would double-attenuate
        # (pr3 = p * reg * [pr1==1]) and systematically under-scale predictions
        # for positive segments — directly hurting R^2.  Write the raw regressor
        # output; prediction_analysis does the hurdle step during scoring.
        pr3 = reg_raw

        seg_to_pr1      = dict(zip(seg_te, pr1))
        seg_to_pr1_prob = dict(zip(seg_te, pr1_prob))
        seg_to_pr3      = dict(zip(seg_te, pr3))

        # ---- Build per-frame CSV ----
        out_df = _build_predictions_df(df_test, seg_to_pr1, seg_to_pr1_prob, seg_to_pr3)
        out_path = predictions_folder / f"Y_test_Y_pred_test_subject_{fold}.csv"
        out_df.to_csv(out_path, index=False)

        # ---- Persist models (small, cheap) ----
        joblib.dump(
            {"clf": clf, "reg": reg, "feature_idx": keep, "feature_names": feat_names},
            models_folder / f"{arch}_test_subject_{fold}.joblib",
        )

        # ---- Log a quick segment-level sanity metric (F1, AUROC, R^2) ----
        try:
            from sklearn.metrics import f1_score, roc_auc_score, r2_score
            f1 = f1_score(y_te, pr1, zero_division=0)
            auc = roc_auc_score(y_te, pr1_prob) if len(set(y_te)) > 1 else float("nan")
            r2 = r2_score(dur_te, pr3) if len(dur_te) > 1 and np.var(dur_te) > 0 else float("nan")
            print(f"  test F1={f1*100:.2f}  AUROC={auc*100:.2f}  R2={r2:.3f}  "
                  f"n_test_segments={len(y_te)}  pos_rate={y_te.mean():.3f}  "
                  f"elapsed={time.time()-t0:.1f}s")
            with open(logs_folder / f"sanity_{group_col}_{fold}.txt", "w") as fp:
                fp.write(f"{group_col}={fold} F1={f1*100:.3f} AUROC={auc*100:.3f} "
                         f"R2={r2:.3f} n_test_segments={len(y_te)}\n")
        except Exception as exc:
            print(f"  [warn] sanity metric failed: {exc}")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    run_cv(cfg, args)


if __name__ == "__main__":
    main()
