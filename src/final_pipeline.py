"""Shared FINAL model pipeline (strict, no leakage).

  - Single stratified split: train/val/test (70/15/15) on labelled rows.
  - Preprocessor fitted ONLY on the training folds.
  - XGBoost trained on training data with the tuned hyperparameters.
  - Threshold selected on VALIDATION (never test).
  - `evaluate()` runs the untouched test set exactly once.

Exposes functions used by threshold-optimisation, final-evaluation, SHAP,
error-analysis and robustness steps - all sharing the SAME fitted model.
"""
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier

import pipeline as P
import encoding as ENC

SUBSET = "Model_C_top15"


def get_config():
    with open(os.path.join(P.OUT, "best_params.json")) as f:
        cfg = json.load(f)
    return cfg["params"]


def build_pre_and_X(train_df, cols):
    pre, _, _ = ENC.make_preprocessor(
        [c for c in cols if c in ENC.NUMERICAL_COLS],
        [c for c in cols if c in ENC.CATEGORICALS])
    Xtr = pre.fit_transform(train_df[cols])
    return pre, Xtr


def train_xgb(X, y, params=None, seed=P.SEED):
    params = params or get_config()
    model = XGBClassifier(
        **params, eval_metric="logloss", tree_method="hist",
        random_state=seed, verbosity=0, use_label_encoder=False,
        n_jobs=-1)
    model.fit(X, y)
    return model


def make_split_sets():
    """Return (train, val, test) dataframes plus cols and encoded tensors.

    The preprocessor is fit on train only; val/test are transformed.
    Also returns the model object (fitted on train).
    """
    df = P.feature_engineer(P.load_raw())
    train, val, test = P.make_split(df)
    cols = P.load_chosen_subset(SUBSET)

    pre, Xtr = build_pre_and_X(train, cols)
    Xval = pre.transform(val[cols])
    Xte = pre.transform(test[cols])

    ytr = train["target"].astype(int).values
    yval = val["target"].astype(int).values
    yte = test["target"].astype(int).values

    model = train_xgb(Xtr, ytr)
    return (train, val, test), cols, pre, model, (Xtr, Xval, Xte), (ytr, yval, yte)


def validate_thresholds(model, Xval, yval, lo=0.10, hi=0.90, step=0.02):
    """Scan thresholds on the validation set; return list of metric dicts."""
    prob = model.predict_proba(Xval)[:, 1]
    rows = []
    for t in np.round(np.arange(lo, hi + 1e-9, step), 3):
        m = P.full_metrics(yval, prob, threshold=t)
        m["threshold"] = t
        rows.append(m)
    return prob, rows


def select_threshold(rows, metric="MCC"):
    """Choose threshold maximising a given metric (on validation)",
    with fallback for ties."""
    best = max(rows, key=lambda r: (r[metric], r["F1"], r["sensitivity"]))
    return best["threshold"]
