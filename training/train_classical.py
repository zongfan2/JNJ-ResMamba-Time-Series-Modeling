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
import sys
import time
from pathlib import Path

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
        from sklearn.ensemble import RandomForestClassifier
        n_trees = int(overrides.get("n_estimators", 50))
        return RandomForestClassifier(
            n_estimators=n_trees,
            random_state=overrides.get("random_state", 42),
            n_jobs=overrides.get("n_jobs", -1),
            class_weight=overrides.get("class_weight", None),
        )
    elif arch == "ji2023":
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise ImportError(
                "Paper 2 baseline requires lightgbm. Install via `pip install lightgbm`."
            ) from exc
        return lgb.LGBMClassifier(
            n_estimators=int(overrides.get("n_estimators", 500)),
            learning_rate=float(overrides.get("learning_rate", 0.05)),
            num_leaves=int(overrides.get("num_leaves", 31)),
            max_depth=int(overrides.get("max_depth", -1)),
            min_child_samples=int(overrides.get("min_child_samples", 20)),
            reg_lambda=float(overrides.get("reg_lambda", 0.0)),
            scale_pos_weight=float(overrides.get("scale_pos_weight", 1.0)),
            random_state=int(overrides.get("random_state", 42)),
            n_jobs=int(overrides.get("n_jobs", -1)),
            verbose=-1,
        )
    else:
        raise ValueError(f"Unknown classical baseline architecture: {arch}")


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


def run_lofo(cfg: dict, args) -> None:
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})

    input_folder = data_cfg["input_folder"]
    arch = model_cfg["architecture"]
    overrides = model_cfg.get("overrides") or {}
    output_name = train_cfg["output"]

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

    # ---- Load data once; split per fold ----
    print(f"[classical] loading data from {input_folder}")
    df = load_data(input_folder)
    df["FOLD"] = df["FOLD"].astype(str)
    all_folds = sorted(df["FOLD"].unique())
    folds = [single_fold] if single_fold else all_folds

    rng = np.random.default_rng(int(overrides.get("random_state", 42)))
    feat_names = feature_names()

    for fold in folds:
        print(f"\n[classical] ==== fold {fold} ====  arch={arch}")
        t0 = time.time()
        df_train = df[df["FOLD"] != fold]
        df_test = df[df["FOLD"] == fold]
        if df_test.empty:
            print(f"  [skip] fold {fold} has no test rows")
            continue

        # ---- Feature extraction ----
        print(f"  extracting features on {df_train['segment'].nunique()} train segments ...")
        X_tr, y_tr, dur_tr, seg_tr = extract_features_batch(df_train)
        print(f"  extracting features on {df_test['segment'].nunique()} test  segments ...")
        X_te, y_te, dur_te, seg_te = extract_features_batch(df_test)
        if X_tr.size == 0 or X_te.size == 0:
            print("  [skip] empty feature matrix")
            continue

        # ---- Clean up NaNs (e.g. constant segments) ----
        X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0)
        X_te = np.nan_to_num(X_te, nan=0.0, posinf=0.0, neginf=0.0)

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
            selected = [feat_names[i] for i in keep]
            print(f"  selected {len(keep)} features: {selected}")
        else:
            keep = np.arange(X_tr.shape[1])

        # ---- Regression target (mean scratch duration among scratch segments) ----
        # Used as a simple surrogate for pr3 so R^2/MAE are defined.
        pos_mean_dur = dur_tr[y_tr == 1].mean() if (y_tr == 1).any() else 0.0

        # ---- Fit classifier ----
        clf = build_classifier(arch, overrides)
        # Paper 2: set scale_pos_weight from data if not supplied
        if arch == "ji2023" and overrides.get("scale_pos_weight") is None:
            n_pos = max(1, int(y_tr.sum()))
            n_neg = max(1, len(y_tr) - n_pos)
            clf.scale_pos_weight = n_neg / n_pos
        print(f"  fitting {type(clf).__name__} ...")
        clf.fit(X_tr, y_tr)

        # ---- Predict on held-out fold ----
        pr1 = clf.predict(X_te).astype(int)
        if hasattr(clf, "predict_proba"):
            pr1_prob = clf.predict_proba(X_te)[:, 1]
        else:
            pr1_prob = pr1.astype(float)
        pr3 = pr1_prob * pos_mean_dur

        seg_to_pr1      = dict(zip(seg_te, pr1))
        seg_to_pr1_prob = dict(zip(seg_te, pr1_prob))
        seg_to_pr3      = dict(zip(seg_te, pr3))

        # ---- Build per-frame CSV ----
        out_df = _build_predictions_df(df_test, seg_to_pr1, seg_to_pr1_prob, seg_to_pr3)
        out_path = predictions_folder / f"Y_test_Y_pred_test_subject_{fold}.csv"
        out_df.to_csv(out_path, index=False)

        # ---- Persist model (small, cheap) ----
        joblib.dump(
            {"clf": clf, "feature_idx": keep, "feature_names": feat_names,
             "pos_mean_dur": float(pos_mean_dur)},
            models_folder / f"{arch}_test_subject_{fold}.joblib",
        )

        # ---- Log a quick segment-level F1 for sanity ----
        try:
            from sklearn.metrics import f1_score, roc_auc_score
            f1 = f1_score(y_te, pr1, zero_division=0)
            auc = roc_auc_score(y_te, pr1_prob) if len(set(y_te)) > 1 else float("nan")
            print(f"  test F1={f1*100:.2f}  AUROC={auc*100:.2f}  "
                  f"n_test_segments={len(y_te)}  pos_rate={y_te.mean():.3f}  "
                  f"elapsed={time.time()-t0:.1f}s")
            with open(logs_folder / f"sanity_fold_{fold}.txt", "w") as fp:
                fp.write(f"fold={fold} F1={f1*100:.3f} AUROC={auc*100:.3f} "
                         f"n_test_segments={len(y_te)}\n")
        except Exception as exc:
            print(f"  [warn] sanity metric failed: {exc}")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    run_lofo(cfg, args)


if __name__ == "__main__":
    main()
