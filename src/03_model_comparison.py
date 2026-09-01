"""Step 3 - Baseline model comparison.

Compares LogisticRegression, RandomForest, ExtraTrees, XGBoost, LightGBM,
CatBoost, HistGradientBoosting and LinearSVC using Stratified 5-fold CV on the
training data, for the chosen feature subset, across class-imbalance strategies.

Saves: model_comparison.csv and a latex-friendly table.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, \
    HistGradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import roc_auc_score, average_precision_score, \
    confusion_matrix, roc_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import pipeline as P
import encoding as ENC

SUBSET = "Model_C_top15"


def make_models(scale_pos_weight=None, class_weight=None):
    """Return dict of name -> estimator (wrapped). CatBoost/LGBM/XGB handle
    imbalance natively via scale_pos_weight."""
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=3000, class_weight=class_weight, random_state=P.SEED),
        "RandomForest": RandomForestClassifier(
            n_estimators=400, class_weight=class_weight, random_state=P.SEED,
            n_jobs=-1),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=400, class_weight=class_weight, random_state=P.SEED,
            n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            random_state=P.SEED),
        "LinearSVC": LinearSVC(max_iter=5000, random_state=P.SEED),
        "XGBoost": XGBClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            scale_pos_weight=scale_pos_weight, random_state=P.SEED,
            use_label_encoder=False, verbosity=0),
        "LightGBM": LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=15,
            scale_pos_weight=scale_pos_weight, random_state=P.SEED,
            verbose=-1),
        "CatBoost": CatBoostClassifier(
            iterations=400, learning_rate=0.05, depth=5,
            scale_pos_weight=scale_pos_weight, random_state=P.SEED,
            verbose=0, allow_writing_files=False),
    }
    return models


def cross_evaluate(models, X, y, cv):
    """Run 5-fold CV returning per-model metric means. X and y are numpy."""
    rows = []
    for name, est in models.items():
        try:
            scores = {"roc_auc": [], "pr_auc": [], "acc": [], "f1": [],
                      "sens": [], "spec": [], "mcc": []}
            for tr, va in cv.split(X, y):
                est.fit(X[tr], y[tr])
                prob = est.predict_proba(X[va])[:, 1] \
                    if hasattr(est, "predict_proba") else \
                    est.decision_function(X[va])
                if name == "LinearSVC":
                    prob = 1 / (1 + np.exp(-prob))
                pred = (prob >= 0.5).astype(int)
                cm = confusion_matrix(y[va], pred)
                tn, fp, fn, tp = cm.ravel()
                scores["roc_auc"].append(roc_auc_score(y[va], prob))
                scores["pr_auc"].append(average_precision_score(y[va], prob))
                scores["acc"].append((tp + tn) / (tp + tn + fp + fn))
                scores["sens"].append(tp / max(1, tp + fn))
                scores["spec"].append(tn / max(1, tn + fp))
                prec = tp / max(1, tp + fp)
                rec = tp / max(1, tp + fn)
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
            print(f"  !! {name} failed: {e}")
    return pd.DataFrame(rows)


def plot_roc_per_model(models, X, y, cv, tag=""):
    """Fit each model once on the CV train folds, average the ROC curve over
    the test folds, then save an individual ROC plot per model."""
    os.makedirs(os.path.join(P.FIG, "roc_per_model"), exist_ok=True)
    mean_fpr = np.linspace(0, 1, 100)
    for name, est in models.items():
        tprs = []
        fold_fprs = []
        fold_tprs = []
        aucs = []
        try:
            for tr, va in cv.split(X, y):
                est.fit(X[tr], y[tr])
                prob = est.predict_proba(X[va])[:, 1] \
                    if hasattr(est, "predict_proba") else \
                    est.decision_function(X[va])
                if name == "LinearSVC":
                    prob = 1 / (1 + np.exp(-prob))
                fpr, tpr, _ = roc_curve(y[va], prob)
                aucs.append(roc_auc_score(y[va], prob))
                fold_fprs.append(fpr)
                fold_tprs.append(tpr)
                tprs.append(np.interp(mean_fpr, fpr, tpr))
                tprs[-1][0] = 0.0
            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = np.mean(aucs)
            fig, ax = plt.subplots(figsize=(5.5, 5.5))
            for ffpr, ftpr in zip(fold_fprs, fold_tprs):
                ax.plot(ffpr, ftpr, alpha=0.15, color="gray", lw=1)
            ax.plot(mean_fpr, mean_tpr, color="blue", lw=2,
                    label=f"mean ROC (AUC={mean_auc:.3f})")
            ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(f"ROC Curve - {name}{tag}")
            ax.legend(loc="lower right")
            fig.tight_layout()
            fname = f"{name.replace(' ', '_')}_roc.png"
            fig.savefig(os.path.join(P.FIG, "roc_per_model", fname), dpi=150)
            plt.close(fig)
            print(f"  saved roc_per_model/{fname}  (CV AUC={mean_auc:.3f})")
        except Exception as e:
            print(f"  !! {name} ROC failed: {e}")


def plot_roc_combined(models, X, y, cv, tag=""):
    """Overlay every model's mean CV ROC curve on a single figure."""
    mean_fpr = np.linspace(0, 1, 100)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    for name, est in models.items():
        tprs = []
        aucs = []
        try:
            for tr, va in cv.split(X, y):
                est.fit(X[tr], y[tr])
                prob = est.predict_proba(X[va])[:, 1] \
                    if hasattr(est, "predict_proba") else \
                    est.decision_function(X[va])
                if name == "LinearSVC":
                    prob = 1 / (1 + np.exp(-prob))
                fpr, tpr, _ = roc_curve(y[va], prob)
                aucs.append(roc_auc_score(y[va], prob))
                tprs.append(np.interp(mean_fpr, fpr, tpr))
                tprs[-1][0] = 0.0
            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = np.mean(aucs)
            ax.plot(mean_fpr, mean_tpr, lw=2,
                    label=f"{name} (AUC={mean_auc:.3f})")
        except Exception as e:
            print(f"  !! {name} combined ROC failed: {e}")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Combined ROC Comparison{tag}")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fname = "roc_combined.png" + (f"_{tag}" if tag else "")
    fig.savefig(os.path.join(P.FIG, "roc_per_model", fname), dpi=150)
    plt.close(fig)
    print(f"\nSaved combined ROC -> figures/roc_per_model/{fname}")


def main():
    df = P.feature_engineer(P.load_raw())
    label = df.dropna(subset=["target"]).copy()
    y = label["target"].astype(int).values
    cols = P.load_chosen_subset(SUBSET)

    # Encode once for the coarse model-comparison screen. NOTE: this screen only
    # ranks models; the final pipeline re-fits preprocessing strictly on train.
    pre, _, _ = ENC.make_preprocessor(
        [c for c in cols if c in ENC.NUMERICAL_COLS],
        [c for c in cols if c in ENC.CATEGORICALS])
    X = pre.fit_transform(label[cols])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=P.SEED)

    # baseline prevalence
    print(f"Using subset: {SUBSET}  |  features={len(cols)}")
    print(f"Train set size (labelled) = {len(y)}  "
          f"positive={y.sum()} ({100*y.mean():.1f}%)")

    print("\n===== CLASS IMBALANCE STRATEGY SCREEN (XGBoost, 5-fold CV) =====")
    spw = (y == 0).sum() / y.sum()   # negative/positive
    strategies = {
        "no_weight": None,
        "moderate_scale_pos_weight_1.5": 1.5,
        "scale_pos_weight_ratio": round(spw, 3),
        "scale_pos_weight_2.0x": 2.0,
    }
    strat_rows = []
    for name, sp in strategies.items():
        m = {"XGBoost": make_models(scale_pos_weight=sp)["XGBoost"]}
        r = cross_evaluate(m, X, y, cv)
        r["strategy"] = name
        strat_rows.append(r)
    strat_df = pd.concat(strat_rows, ignore_index=True)
    strat_df = strat_df[["strategy", "roc_auc", "pr_auc", "accuracy",
                         "sensitivity", "specificity", "f1", "mcc"]]
    print(strat_df.to_string(index=False))
    strat_df.to_csv(os.path.join(P.OUT, "imbalance_strategy.csv"), index=False)

    print("\n===== BASELINE MODEL COMPARISON (5-fold CV) =====")
    models = make_models(class_weight="balanced")
    result = cross_evaluate(models, X, y, cv)
    result = result.sort_values("roc_auc", ascending=False)
    print(result.round(4).to_string(index=False))
    result.to_csv(os.path.join(P.OUT, "model_comparison.csv"), index=False)
    print("\nSaved -> output/model_comparison.csv")

    print("\n===== ROC CURVES (per model, 5-fold CV) =====")
    plot_roc_per_model(models, X, y, cv)
    plot_roc_combined(models, X, y, cv)
    print("Saved -> figures/roc_per_model/")


if __name__ == "__main__":
    main()
