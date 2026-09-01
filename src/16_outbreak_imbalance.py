"""Step 16 - Outbreak model: class-imbalance improvements.

Compares three approaches to fix the low sensitivity (27%) of the Step-14
upazila outbreak ensemble:

  1. Threshold sweep  — no retraining, just lower the decision threshold
  2. Class weights     — retrain XGB/CatBoost/LightGBM with scale_pos_weight
  3. SMOTE oversampling — synthetic minority oversampling before training

All evaluated with the SAME 5-fold stratified CV and out-of-fold (OOF)
probabilities to avoid overfitting.  The best approach is then compared
against the original Step-14 baseline.
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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             confusion_matrix, roc_curve, f1_score)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import pipeline as P

# Optional: SMOTE
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    print("WARNING: imbalanced-learn not installed. SMOTE experiments skipped.")
    print("  Install with: pip install imbalanced-learn")

import pipeline as P2
import encoding as ENC

OUT_DIR = os.path.join(P.OUT, "outbreak")
FIG_DIR = os.path.join(P.FIG, "outbreak")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Configuration (same as Step 14)
# ---------------------------------------------------------------------------
OUTBREAK_MIN = 2
MAX_WEEK = 22
NUMERIC_FEATURES = ["wk", "lag1", "lag2", "roll3", "prev_active",
                    "wk_sin", "wk_cos"]
CATEGORICAL_FEATURES = ["division_lvl"]


# ---------------------------------------------------------------------------
# Rebuild the same feature matrix as Step 14 (inline copy)
# ---------------------------------------------------------------------------
def build_cells(df, unit_col, min_cases=OUTBREAK_MIN):
    m = df[df["CLASS"] == P.MEASLES_CLASS].copy()
    d = pd.to_datetime(m["DNOT"], errors="coerce")
    m["wk"] = d.dt.isocalendar().week.astype(int)
    m["yr"] = d.dt.year.astype(int)
    counts = (m.groupby([unit_col, "yr", "wk"])
              .size().reset_index(name="cases"))
    units = m[unit_col].dropna().unique()
    grid = pd.MultiIndex.from_product(
        [units, [2026], list(range(1, MAX_WEEK + 1))],
        names=[unit_col, "yr", "wk"])
    full = (pd.DataFrame(index=grid).reset_index()
            .merge(counts, on=[unit_col, "yr", "wk"], how="left"))
    full["cases"] = full["cases"].fillna(0).astype(int)
    full["OUTBREAK"] = (full["cases"] >= min_cases).astype(int)
    full["prev_cases"] = full.groupby(unit_col)["cases"].shift(1).fillna(0)
    full["cases_mean"] = full["cases"].groupby(full[unit_col]).transform("mean")
    full["active_weeks"] = (full["cases"] > 0).groupby(
        full[unit_col]).transform("sum")
    return full


def make_models():
    return {
        "XGBoost": XGBClassifier(n_estimators=300, learning_rate=0.05,
                                 max_depth=4, subsample=0.8,
                                 colsample_bytree=0.8, eval_metric="logloss",
                                 random_state=P.SEED, use_label_encoder=False,
                                 verbosity=0),
        "LightGBM": LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                   num_leaves=15, random_state=P.SEED, verbose=-1),
        "CatBoost": CatBoostClassifier(iterations=300, learning_rate=0.05,
                                       depth=5, random_state=P.SEED, verbose=0,
                                       allow_writing_files=False),
    }


def _get_data():
    """Load raw, build upazila features, return (feats, X_enc, y, cols)."""
    df = P.load_raw()
    unit_col = "UPZMUNCC"
    cells = build_cells(df, unit_col)

    # division map
    div_map = df.drop_duplicates(unit_col)[[unit_col, "DIVISION"]]
    div_map.columns = [unit_col, "division_lvl"]
    feats = cells.merge(div_map, on=unit_col, how="left")

    feats = feats.sort_values([unit_col, "wk"]).reset_index(drop=True)
    feats["lag1"] = feats.groupby(unit_col)["cases"].shift(1).fillna(0)
    feats["lag2"] = feats.groupby(unit_col)["cases"].shift(2).fillna(0)
    feats["roll3"] = (feats.groupby(unit_col)["cases"].shift(1)
                      .fillna(0).groupby(feats[unit_col])
                      .transform(lambda s: s.rolling(3, min_periods=1).sum()))
    feats["prev_active"] = (feats["lag1"] > 0).astype(int)
    feats["wk_sin"] = np.sin(2 * np.pi * feats["wk"] / MAX_WEEK)
    feats["wk_cos"] = np.cos(2 * np.pi * feats["wk"] / MAX_WEEK)

    cols = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES
            if c in feats.columns]
    Xfeats = feats[cols].copy()
    y = feats["OUTBREAK"].astype(int).values

    # keep only rows with known features
    mask = Xfeats.isna().any(axis=1)
    X = Xfeats[~mask].reset_index(drop=True)
    yy = y[~mask]

    # one-hot encode categoricals
    X_enc = pd.get_dummies(X, columns=[c for c in CATEGORICAL_FEATURES
                                        if c in X.columns], drop_first=False)
    X_enc = X_enc.astype(float)

    return feats, X_enc, yy, cols


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def eval_oof(oof_probs, y, threshold):
    """Compute metrics from OOF probabilities at a given threshold."""
    pred = (oof_probs >= threshold).astype(int)
    cm = confusion_matrix(y, pred)
    tn, fp, fn, tp = cm.ravel()
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    sens = tp / max(1, tp + fn)
    spec = tn / max(1, tn + fp)
    prec = tp / max(1, tp + fp)
    f1 = 2 * prec * sens / max(1e-9, prec + sens)
    mcc = (tp * tn - fp * fn) / max(1e-9, (
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5)
    auc = roc_auc_score(y, oof_probs)
    prauc = average_precision_score(y, oof_probs)
    return {
        "accuracy": acc, "sensitivity": sens, "specificity": spec,
        "precision": prec, "F1": f1, "MCC": mcc,
        "ROC_AUC": auc, "PR_AUC": prauc,
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def threshold_sweep(oof_probs, y, thresholds=None):
    """Scan thresholds, return list of (threshold, metrics) sorted by F1."""
    if thresholds is None:
        thresholds = np.round(np.arange(0.10, 0.80, 0.02), 2)
    rows = []
    for t in thresholds:
        m = eval_oof(oof_probs, y, t)
        m["threshold"] = float(t)
        rows.append(m)
    return rows


# ---------------------------------------------------------------------------
# Approach 1: Threshold sweep (no retraining)
# ---------------------------------------------------------------------------
def approach_threshold(X_enc, y):
    """Use original Step-14 models to get OOF probabilities, then sweep."""
    print("\n--- Approach 1: Threshold Sweep (no retraining) ---")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=P.SEED)
    oof = np.zeros(len(y))
    for tr, va in cv.split(X_enc.values, y):
        probs = []
        for n in ["XGBoost", "CatBoost", "LightGBM"]:
            m = make_models()[n]
            m.fit(X_enc.values[tr], y[tr])
            probs.append(m.predict_proba(X_enc.values[va])[:, 1])
        oof[va] = np.mean(probs, axis=0)

    rows = threshold_sweep(oof, y)
    # find best by F1 and by sensitivity@acc>=0.80
    best_f1 = max(rows, key=lambda r: r["F1"])
    best_sens = max(
        [r for r in rows if r["accuracy"] >= 0.80],
        key=lambda r: r["sensitivity"], default=best_f1)
    return oof, rows, best_f1, best_sens


# ---------------------------------------------------------------------------
# Approach 2: Class weights (scale_pos_weight)
# ---------------------------------------------------------------------------
def approach_class_weights(X_enc, y):
    """Retrain with scale_pos_weight = neg/pos to penalise missed outbreaks."""
    print("\n--- Approach 2: Class Weights (scale_pos_weight) ---")
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    spw = n_neg / max(1, n_pos)
    print(f"  scale_pos_weight = {spw:.2f}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=P.SEED)
    oof = np.zeros(len(y))
    for tr, va in cv.split(X_enc.values, y):
        probs = []
        for n in ["XGBoost", "CatBoost", "LightGBM"]:
            m = make_models()[n]
            # set class weight
            if hasattr(m, "scale_pos_weight"):
                m.set_params(scale_pos_weight=spw)
            elif hasattr(m, "class_weight"):
                m.set_params(class_weight="balanced")
            m.fit(X_enc.values[tr], y[tr])
            probs.append(m.predict_proba(X_enc.values[va])[:, 1])
        oof[va] = np.mean(probs, axis=0)

    rows = threshold_sweep(oof, y)
    best_f1 = max(rows, key=lambda r: r["F1"])
    best_sens = max(
        [r for r in rows if r["accuracy"] >= 0.80],
        key=lambda r: r["sensitivity"], default=best_f1)
    return oof, rows, best_f1, best_sens


# ---------------------------------------------------------------------------
# Approach 3: SMOTE oversampling
# ---------------------------------------------------------------------------
def approach_smote(X_enc, y):
    """SMOTE oversampling of minority class before training."""
    if not HAS_SMOTE:
        print("\n--- Approach 3: SMOTE --- SKIPPED (imbalanced-learn not installed)")
        return None, None, None, None

    print("\n--- Approach 3: SMOTE Oversampling ---")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=P.SEED)
    oof = np.zeros(len(y))
    for tr, va in cv.split(X_enc.values, y):
        # Apply SMOTE on training fold only
        smote = SMOTE(random_state=P.SEED, k_neighbors=5)
        X_res, y_res = smote.fit_resample(X_enc.values[tr], y[tr])
        print(f"  Fold: {len(y[tr])} -> {len(y_res)} "
              f"(+{len(y_res) - len(y[tr])} synthetic)")

        probs = []
        for n in ["XGBoost", "CatBoost", "LightGBM"]:
            m = make_models()[n]
            m.fit(X_res, y_res)
            probs.append(m.predict_proba(X_enc.values[va])[:, 1])
        oof[va] = np.mean(probs, axis=0)

    rows = threshold_sweep(oof, y)
    best_f1 = max(rows, key=lambda r: r["F1"])
    best_sens = max(
        [r for r in rows if r["accuracy"] >= 0.80],
        key=lambda r: r["sensitivity"], default=best_f1)
    return oof, rows, best_f1, best_sens


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("OUTBREAK MODEL: CLASS-IMBALANCE IMPROVEMENTS")
    print(f"Outbreak definition: >= {OUTBREAK_MIN} confirmed cases / unit-week")
    print("=" * 70)

    feats, X_enc, y, cols = _get_data()
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    print(f"\nData: {len(y)} cells, {n_pos} outbreak ({100*n_pos/len(y):.1f}%), "
          f"{n_neg} non-outbreak ({100*n_neg/len(y):.1f}%)")

    # Run all three approaches
    oof1, rows1, best_f1_1, best_sens_1 = approach_threshold(X_enc, y)
    oof2, rows2, best_f1_2, best_sens_2 = approach_class_weights(X_enc, y)
    oof3, rows3, best_f1_3, best_sens_3 = approach_smote(X_enc, y)

    # Baseline (original Step-14 threshold = 0.54)
    baseline = eval_oof(oof1, y, 0.54)
    baseline["threshold"] = 0.54

    # Summary table
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    summary = []
    rows_summary = [
        ("Baseline (t=0.54)", baseline),
        (f"Threshold sweep (t={best_f1_1['threshold']:.2f}, best F1)", best_f1_1),
        (f"Threshold sweep (t={best_sens_1['threshold']:.2f}, sens@acc>=0.80)", best_sens_1),
        (f"Class weights (t={best_f1_2['threshold']:.2f}, best F1)", best_f1_2),
        (f"Class weights (t={best_sens_2['threshold']:.2f}, sens@acc>=0.80)", best_sens_2),
    ]
    if oof3 is not None:
        rows_summary.append(
            (f"SMOTE (t={best_f1_3['threshold']:.2f}, best F1)", best_f1_3))
        rows_summary.append(
            (f"SMOTE (t={best_sens_3['threshold']:.2f}, sens@acc>=0.80)", best_sens_3))

    for label, m in rows_summary:
        summary.append({
            "approach": label,
            "threshold": m["threshold"],
            "accuracy": m["accuracy"],
            "sensitivity": m["sensitivity"],
            "specificity": m["specificity"],
            "precision": m["precision"],
            "F1": m["F1"],
            "MCC": m["MCC"],
            "ROC_AUC": m["ROC_AUC"],
            "FP": m["FP"],
            "FN": m["FN"],
        })
        print(f"\n  {label}")
        print(f"    acc={m['accuracy']:.4f}  sens={m['sensitivity']:.4f}  "
              f"spec={m['specificity']:.4f}  F1={m['F1']:.4f}  MCC={m['MCC']:.4f}")
        print(f"    ROC-AUC={m['ROC_AUC']:.4f}  FP={m['FP']}  FN={m['FN']}")

    # Save results
    df_out = pd.DataFrame(summary)
    df_out.to_csv(os.path.join(OUT_DIR, "imbalance_experiment.csv"), index=False)

    # ROC curves comparison
    fig, ax = plt.subplots(figsize=(7, 7))
    for oof, label, color in [
        (oof1, "Baseline + threshold sweep", "#2196F3"),
        (oof2, "Class weights", "#FF5722"),
        (oof3, "SMOTE", "#4CAF50"),
    ]:
        if oof is None:
            continue
        fpr, tpr, _ = roc_curve(y, oof)
        auc = roc_auc_score(y, oof)
        ax.plot(fpr, tpr, lw=2, color=color, label=f"{label} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    ax.set_title("Outbreak Model - ROC Comparison\n(Upazila Ensemble, OOF)", fontsize=12)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "imbalance_roc_comparison.png"), dpi=150)
    plt.close(fig)

    # Sensitivity vs Threshold plot
    fig, ax = plt.subplots(figsize=(7, 5))
    for rows_data, label, color in [
        (rows1, "Baseline ensemble", "#2196F3"),
        (rows2, "Class weights", "#FF5722"),
        (rows3, "SMOTE", "#4CAF50"),
    ]:
        if rows_data is None:
            continue
        ts = [r["threshold"] for r in rows_data]
        sens = [r["sensitivity"] for r in rows_data]
        acc = [r["accuracy"] for r in rows_data]
        ax.plot(ts, sens, lw=2, color=color, label=f"{label} (sensitivity)")
        ax.plot(ts, acc, lw=1.5, color=color, linestyle="--",
                alpha=0.5, label=f"{label} (accuracy)")

    ax.axhline(y=0.80, color="gray", linestyle=":", alpha=0.5, label="80% target")
    ax.set_xlabel("Decision Threshold", fontsize=11)
    ax.set_ylabel("Metric Value", fontsize=11)
    ax.set_title("Sensitivity & Accuracy vs Threshold\n(Upazila Outbreak Ensemble, OOF)", fontsize=12)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    ax.set_xlim(0.10, 0.80)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "imbalance_sensitivity_vs_threshold.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # F1 vs Threshold
    fig, ax = plt.subplots(figsize=(7, 5))
    for rows_data, label, color in [
        (rows1, "Baseline ensemble", "#2196F3"),
        (rows2, "Class weights", "#FF5722"),
        (rows3, "SMOTE", "#4CAF50"),
    ]:
        if rows_data is None:
            continue
        ts = [r["threshold"] for r in rows_data]
        f1s = [r["F1"] for r in rows_data]
        ax.plot(ts, f1s, lw=2, color=color, label=label)

    ax.set_xlabel("Decision Threshold", fontsize=11)
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.set_title("F1 Score vs Threshold\n(Upazila Outbreak Ensemble, OOF)", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(0.10, 0.80)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "imbalance_f1_vs_threshold.png"), dpi=150)
    plt.close(fig)

    print(f"\nSaved:")
    print(f"  output/outbreak/imbalance_experiment.csv")
    print(f"  figures/outbreak/imbalance_roc_comparison.png")
    print(f"  figures/outbreak/imbalance_sensitivity_vs_threshold.png")
    print(f"  figures/outbreak/imbalance_f1_vs_threshold.png")


if __name__ == "__main__":
    main()
