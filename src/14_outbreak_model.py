"""Step 14 - District/upazila/division-level measles OUTBREAK prediction.

Builds on the existing shared pipeline but aggregates confirmed-measles cases
into spatial-unit x epi-week cells and predicts whether a cell is an
'outbreak' cell (>= OUTBREAK_MIN confirmed cases in that unit-week).

Levels modelled:
  - DIVISION  (9 units)
  - DISTRICT  (77 units)
  - UPZAMUNC  (460+ units)  <-- primary focus

Because the surveillance data all fall within a single year (2026, epi-weeks
1-22), there is NO year-to-year chronology -> true multi-year forecasting (as in
district-level retrospective climate studies) is not possible from these data.
Instead this models CROSS-SECTIONAL outbreak-status per unit-week, using:

  static per-unit features   (mean burden, active-weeks, region)
  within-year temporal feats (epi-week seasonality, prior-week lag, rolling sum)

Outputs (in output/outbreak/ and figures/outbreak/):
  - cell_features_*.csv      per-unit-week feature matrix + label
  - baseline_comparison.csv  8 classifiers x level (5-fold CV)
  - best_params_*.json
  - upazila_ensemble_result.json   combined best model at upazila level
  - roc per level + upazila ensemble ROC figure
"""
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, \
    HistGradientBoostingClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             confusion_matrix, roc_curve)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import pipeline as P
import encoding as ENC

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTBREAK_MIN = 2          # confirmed measles cases in a unit-week to call 'outbreak'
MAX_WEEK = 22             # weeks present in 2026 (1..22)
LEVELS = ["DIVISION", "DISTRICT", "UPZMUNCC"]
PRIMARY = "UPZMUNCC"

OUT_DIR = os.path.join(P.OUT, "outbreak")
FIG_DIR = os.path.join(P.FIG, "outbreak")
for d in (OUT_DIR, FIG_DIR):
    os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Aggregate confirmed-measles -> unit x week cells with outbreak label
# ---------------------------------------------------------------------------
def build_cells(df, unit_col, min_cases=OUTBREAK_MIN):
    """Return a per-unit-per-week matrix of case counts and outbreak label."""
    m = df[df["CLASS"] == P.MEASLES_CLASS].copy()
    d = pd.to_datetime(m["DNOT"], errors="coerce")
    m["wk"] = d.dt.isocalendar().week.astype(int)
    m["yr"] = d.dt.year.astype(int)

    counts = (m.groupby([unit_col, "yr", "wk"])
                .size().reset_index(name="cases"))
    # Build a dense grid of ALL units x all weeks (including zero-case cells)
    units = m[unit_col].dropna().unique()
    grid = pd.MultiIndex.from_product(
        [units, [2026], list(range(1, MAX_WEEK + 1))],
        names=[unit_col, "yr", "wk"])
    full = (pd.DataFrame(index=grid)
              .reset_index()
              .merge(counts, on=[unit_col, "yr", "wk"], how="left"))
    full["cases"] = full["cases"].fillna(0).astype(int)
    full["OUTBREAK"] = (full["cases"] >= min_cases).astype(int)
    full["prev_cases"] = full.groupby(unit_col)["cases"].shift(1).fillna(0)
    full["cases_mean"] = full["cases"].groupby(full[unit_col]).transform("mean")
    full["active_weeks"] = (full["cases"] > 0).groupby(
        full[unit_col]).transform("sum")
    return full


# ---------------------------------------------------------------------------
# 2. Feature matrix assembly for a given level
# ---------------------------------------------------------------------------
def feature_matrix_for_level(df, unit_col):
    """Combine static + temporal features into a model-ready frame.

    IMPORTANT (leakage control): only features computable at prediction time are
    kept. Static per-unit aggregates of the WHOLE-period confirmed-case count are
    deliberately excluded (they are just the label aggregated and would leak).
    Only strictly-past temporal features (lag1/lag2/rolling) + season + region
    are used, mirroring a realistic early-warning setting.
    """
    cells = build_cells(df, unit_col)

    # region identity (helps the model borrow signal across units)
    feats = cells
    if unit_col != "DIVISION" and unit_col in df.columns:
        div_map = df.drop_duplicates(unit_col)[[unit_col, "DIVISION"]]
        div_map.columns = [unit_col, "division_lvl"]
        feats = feats.merge(div_map, on=unit_col, how="left")

    # temporal / lag / rolling features (past weeks only -> no leakage)
    feats = feats.sort_values([unit_col, "wk"]).reset_index(drop=True)
    feats["lag1"] = feats.groupby(unit_col)["cases"].shift(1).fillna(0)
    feats["lag2"] = feats.groupby(unit_col)["cases"].shift(2).fillna(0)
    feats["roll3"] = (feats.groupby(unit_col)["cases"].shift(1)
                      .fillna(0).groupby(feats[unit_col])
                      .transform(lambda s: s.rolling(3, min_periods=1).sum()))
    feats["prev_active"] = (feats["lag1"] > 0).astype(int)
    feats["wk_sin"] = np.sin(2 * np.pi * feats["wk"] / MAX_WEEK)
    feats["wk_cos"] = np.cos(2 * np.pi * feats["wk"] / MAX_WEEK)

    return feats


# ---------------------------------------------------------------------------
# 4. Baseline model comparison (5-fold CV) at every level
# ---------------------------------------------------------------------------
def make_models():
    return {
        "LogisticRegression": LogisticRegression(max_iter=3000, random_state=P.SEED),
        "DecisionTree": DecisionTreeClassifier(max_depth=6, random_state=P.SEED),
        "NaiveBayes": GaussianNB(),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=P.SEED, n_jobs=-1),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=300, random_state=P.SEED, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=P.SEED),
        "XGBoost": XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                                 subsample=0.8, colsample_bytree=0.8,
                                 eval_metric="logloss", random_state=P.SEED,
                                 use_label_encoder=False, verbosity=0),
        "LightGBM": LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                   num_leaves=15, random_state=P.SEED, verbose=-1),
        "CatBoost": CatBoostClassifier(iterations=300, learning_rate=0.05, depth=5,
                                       random_state=P.SEED, verbose=0,
                                       allow_writing_files=False),
    }


def cross_evaluate(models, X, y, cv):
    rows = []
    for name, est in models.items():
        try:
            scores = {"roc_auc": [], "pr_auc": [], "acc": [], "f1": [],
                      "sens": [], "spec": [], "mcc": []}
            for tr, va in cv.split(X, y):
                est.fit(X[tr], y[tr])
                prob = est.predict_proba(X[va])[:, 1] \
                    if hasattr(est, "predict_proba") else est.decision_function(X[va])
                if name == "LinearSVC":
                    prob = 1 / (1 + np.exp(-prob))
                pred = (prob >= 0.5).astype(int)
                cm = confusion_matrix(y[va], pred)
                tn, fp, fn, tp = cm.ravel()
                scores["roc_auc"].append(roc_auc_score(y[va], prob))
                scores["pr_auc"].append(average_precision_score(y[va], prob))
                scores["acc"].append((tp + tn) / max(1, tp + tn + fp + fn))
                scores["sens"].append(tp / max(1, tp + fn))
                scores["spec"].append(tn / max(1, tn + fp))
                prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
                scores["f1"].append(2 * prec * rec / max(1e-9, prec + rec))
                scores["mcc"].append((tp * tn - fp * fn) / max(1e-9, (
                    (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5))
            rows.append({"model": name,
                         "roc_auc": np.mean(scores["roc_auc"]),
                         "pr_auc": np.mean(scores["pr_auc"]),
                         "accuracy": np.mean(scores["acc"]),
                         "sensitivity": np.mean(scores["sens"]),
                         "specificity": np.mean(scores["spec"]),
                         "f1": np.mean(scores["f1"]),
                         "mcc": np.mean(scores["mcc"])})
        except Exception as e:
            rows.append({"model": name, "roc_auc": np.nan, "pr_auc": np.nan,
                         "accuracy": np.nan, "sensitivity": np.nan,
                         "specificity": np.nan, "f1": np.nan, "mcc": np.nan,
                         "error": str(e)})
    return pd.DataFrame(rows)


NUMERIC_FEATURES = ["wk", "lag1", "lag2", "roll3", "prev_active",
                    "wk_sin", "wk_cos"]
CATEGORICAL_FEATURES = ["division_lvl"]


def assemble_Xy(feats, unit_col):
    cols = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES
            if c in feats.columns]
    Xfeats = feats[cols].copy()
    y = feats["OUTBREAK"].astype(int).values
    return Xfeats, y, cols


# ---------------------------------------------------------------------------
# 5. Upazila combined/ensemble model
# ---------------------------------------------------------------------------
def run_ensemble(X, y, cols, feats):
    """Simple-average GBM ensemble at upazila level (XGB + CatBoost + LightGBM),
    with correct out-of-fold validation for threshold selection and a final
    fit on all data for the combined model."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=P.SEED)
    # out-of-fold probabilities
    oof = np.zeros(len(y))
    names = ["XGBoost", "CatBoost", "LightGBM"]
    for tr, va in cv.split(X, y):
        probs = []
        for n in names:
            bm = make_models()[n]
            bm.fit(X[tr], y[tr])
            probs.append(bm.predict_proba(X[va])[:, 1])
        oof[va] = np.mean(probs, axis=0)

    # threshold selection on OOF to maximise accuracy
    best_t, best_acc = 0.5, -1
    for t in np.round(np.arange(0.30, 0.70, 0.01), 2):
        acc = ((oof >= t) == y).mean()
        if acc > best_acc:
            best_acc, best_t = acc, t

    # final models fit on all data
    models = {n: make_models()[n] for n in names}
    for n, m in models.items():
        m.fit(X, y)

    return models, best_t, best_acc, oof


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------
def main():
    df = P.load_raw()
    print("=" * 70)
    print("OUTBREAK-LEVEL PREDICTION (unit x epi-week, 2026)")
    print(f"Outbreak definition: >= {OUTBREAK_MIN} confirmed cases in a unit-week")
    print("=" * 70)

    all_level_results = []
    trained = {}
    for unit_col in LEVELS:
        print(f"\n--- LEVEL: {unit_col} ---")
        feats = feature_matrix_for_level(df, unit_col)
        Xfeats, y, cols = assemble_Xy(feats, unit_col)
        feats.to_csv(os.path.join(OUT_DIR, f"cell_{unit_col}.csv"),
                     index=False)

        # keep only rows with known features
        mask = Xfeats.isna().any(axis=1)
        X = Xfeats[~mask].reset_index(drop=True)
        yy = y[~mask]

        print(f"  units={feats[unit_col].nunique()}  cells={len(yy)}  "
              f"outbreak_cells={(yy == 1).sum()} "
              f"({100 * yy.mean():.1f}%)")

        # one-hot encode categoricals (division_lvl)
        X_enc = pd.get_dummies(X, columns=[c for c in CATEGORICAL_FEATURES
                                           if c in X.columns],
                               drop_first=False)
        X_enc = X_enc.astype(float)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=P.SEED)
        models = make_models()
        res = cross_evaluate(models, X_enc.values, yy, cv)
        res["level"] = unit_col
        all_level_results.append(res)
        print(res.round(4).to_string(index=False))
        trained[unit_col] = (X_enc, yy, cols, feats)

    comparison = pd.concat(all_level_results, ignore_index=True)
    comparison.to_csv(os.path.join(OUT_DIR, "baseline_comparison.csv"),
                      index=False)
    print("\nSaved -> output/outbreak/baseline_comparison.csv")

    # ---- ROC per level (best baseline) ----
    for unit_col in LEVELS:
        X_enc, yy, cols, feats = trained[unit_col]
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=P.SEED)
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        base = {"xgb": make_models()["XGBoost"]}
        tprs, aucs, mean_fpr = [], [], np.linspace(0, 1, 100)
        for tr, va in cv.split(X_enc.values, yy):
            base["xgb"].fit(X_enc.values[tr], yy[tr])
            prob = base["xgb"].predict_proba(X_enc.values[va])[:, 1]
            fpr, tpr, _ = roc_curve(yy[va], prob)
            aucs.append(roc_auc_score(yy[va], prob))
            tprs.append(np.interp(mean_fpr, fpr, tpr)); tprs[-1][0] = 0.0
        mtpr = np.mean(tprs, axis=0); mtpr[-1] = 1.0
        ax.plot(mean_fpr, mtpr, lw=2,
                label=f"XGBoost (AUC={np.mean(aucs):.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.set_title(f"ROC - {unit_col} Outbreak Level")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, f"roc_{unit_col}.png"), dpi=150)
        plt.close(fig)

    # ---- PRIMARY: upazila combined/ensemble best model ----
    print("\n" + "=" * 70)
    print(f"PRIMARY: COMBINED BEST MODEL @ {PRIMARY} LEVEL")
    print("=" * 70)
    X_enc, yy, cols, feats = trained[PRIMARY]
    models, best_t, oof_acc, oof_probs = run_ensemble(X_enc.values, yy, cols, feats)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=P.SEED)
    tprs, aucs, mean_fpr = [], [], np.linspace(0, 1, 100)
    for tr, va in cv.split(X_enc.values, yy):
        probs = []
        for n in ["XGBoost", "CatBoost", "LightGBM"]:
            m = make_models()[n]
            m.fit(X_enc.values[tr], yy[tr])
            probs.append(m.predict_proba(X_enc.values[va])[:, 1])
        pv = np.mean(probs, axis=0)
        fpr, tpr, _ = roc_curve(yy[va], pv)
        aucs.append(roc_auc_score(yy[va], pv))
        tprs.append(np.interp(mean_fpr, fpr, tpr)); tprs[-1][0] = 0.0
    mtpr = np.mean(tprs, axis=0); mtpr[-1] = 1.0

    # full data OOF AUC/PR
    oof_auc = roc_auc_score(yy, oof_probs)
    oof_prauc = average_precision_score(yy, oof_probs)
    oof_pred = (oof_probs >= best_t).astype(int)
    cm = confusion_matrix(yy, oof_pred)
    tn, fp, fn, tp = cm.ravel()
    sens = tp / max(1, tp + fn); spec = tn / max(1, tn + fp)
    prec = tp / max(1, tp + fp); f1 = 2 * prec * sens / max(1e-9, prec + sens)
    mcc = (tp * tn - fp * fn) / max(1e-9, (
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot(mean_fpr, mtpr, lw=2, label=f"Ensemble (AUC={np.mean(aucs):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(f"ROC - {PRIMARY} Combined (XGB+CatBoost+LightGBM)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"upazila_ensemble_roc.png"), dpi=150)
    plt.close(fig)

    result = {
        "level": PRIMARY,
        "outbreak_definition": f">= {OUTBREAK_MIN} confirmed cases / unit-week",
        "n_units": int(feats[PRIMARY].nunique()),
        "n_cells": int(len(yy)),
        "outbreak_cells": int(yy.sum()),
        "ensemble_models": ["XGBoost", "CatBoost", "LightGBM"],
        "threshold": float(best_t),
        "metrics_oof": {
            "accuracy": oof_acc, "ROC_AUC": float(oof_auc),
            "PR_AUC": float(oof_prauc), "sensitivity": float(sens),
            "specificity": float(spec), "precision": float(prec),
            "F1": float(f1), "MCC": float(mcc),
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        },
        "features": cols,
        "note": ("All data fall in 2026 (weeks 1-22). This is a cross-sectional "
                 "outbreak-status model per unit-week, NOT a multi-year "
                 "temporal forecast (no year-to-year lag available)."),
    }
    with open(os.path.join(OUT_DIR, "upazila_ensemble_result.json"), "w") as f:
        json.dump(result, f, indent=2, default=float)

    print(f"\nThreshold (locked on OOF, max accuracy): t={best_t}  "
          f"OOF acc={oof_acc:.4f}")
    print(f"OOF ROC-AUC={oof_auc:.4f}  PR-AUC={oof_prauc:.4f}")
    print(f"TN={tn} FP={fp} FN={fn} TP={tp}  sens={sens:.3f} spec={spec:.3f} "
          f"F1={f1:.3f} MCC={mcc:.3f}")
    print("\nSaved -> output/outbreak/upazila_ensemble_result.json, "
          "figures/outbreak/upazila_ensemble_roc.png")


if __name__ == "__main__":
    main()
