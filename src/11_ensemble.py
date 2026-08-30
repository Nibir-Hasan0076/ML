"""Step 11 - Accuracy improvement: SIMPLE-AVERAGE GBM ensemble.

The individual models (XGBoost, CatBoost, LightGBM) are near-equally strong;
simple probability averaging is fully legitimate (no target leakage) and proved
more robust than weight-search AUC optimisation + isotonic calibration (which
collapsed onto a single model and degraded the operating point).

Protocol (no leakage):
  - same fixed stratified train/val/test split, preprocessor fit on train only
  - train the 3 GBMs on TRAIN
  - average their probabilities
  - lock the threshold that maximises ACCURACY on the VALIDATION set
  - evaluate EXACTLY ONCE on the untouched TEST set
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
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             roc_auc_score, average_precision_score,
                             brier_score_loss)
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

import pipeline as P
import final_pipeline as FP

import joblib


def main():
    sets, cols, pre, single_model, X, y = FP.make_split_sets()
    (train, val, test) = sets
    (Xtr, Xval, Xte) = X
    (ytr, yval, yte) = y

    print("=" * 70)
    print("IMPROVED MODEL: simple-average GBM ensemble (XGB+CatBoost+LightGBM)")
    print("=" * 70)

    cfg = FP.get_config()
    models = {
        "XGBoost": XGBClassifier(
            **cfg, eval_metric="logloss", tree_method="hist",
            random_state=P.SEED, verbosity=0, use_label_encoder=False, n_jobs=-1),
        "CatBoost": CatBoostClassifier(
            iterations=600, learning_rate=0.05, depth=6,
            scale_pos_weight=cfg["scale_pos_weight"],
            random_state=P.SEED, verbose=0, allow_writing_files=False),
        "LightGBM": LGBMClassifier(
            n_estimators=600, learning_rate=0.05, num_leaves=31,
            scale_pos_weight=cfg["scale_pos_weight"],
            random_state=P.SEED, verbose=-1),
    }

    val_prob, te_prob = {}, {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        val_prob[name] = m.predict_proba(Xval)[:, 1]
        te_prob[name] = m.predict_proba(Xte)[:, 1]
        print(f"  trained {name}: val_AUC="
              f"{roc_auc_score(yval, val_prob[name]):.4f}")

    pv_avg = np.mean([val_prob[n] for n in models], axis=0)
    pt_avg = np.mean([te_prob[n] for n in models], axis=0)
    print(f"  ensemble val_AUC={roc_auc_score(yval, pv_avg):.4f}")

    # ---- Lock threshold on VALIDATION by accuracy ----
    best_t, best_acc = 0.5, -1
    for t in np.round(np.arange(0.30, 0.70, 0.01), 2):
        pred = (pv_avg >= t).astype(int)
        cm = confusion_matrix(yval, pred)
        tn, fp, fn, tp = cm.ravel()
        acc = (tp + tn) / (tp + tn + fp + fn)
        if acc > best_acc:
            best_acc, best_t = acc, t
    print(f"\nThreshold locked on validation (max accuracy): t={best_t}, "
          f"val_acc={best_acc:.4f}")

    # ---- Evaluate ONCE on test ----
    pred = (pt_avg >= best_t).astype(int)
    cm = confusion_matrix(yte, pred)
    tn, fp, fn, tp = cm.ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
    auc = roc_auc_score(yte, pt_avg)
    prauc = average_precision_score(yte, pt_avg)
    sens = tp / (tp + fn); spec = tn / (tn + fp)
    prec = tp / (tp + fp); npv = tn / (tn + fn)
    f1 = 2 * prec * sens / (prec + sens)
    bal = (sens + spec) / 2
    mcc = (tp * tn - fp * fn) / max(1e-9, (
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5)
    brier = brier_score_loss(yte, pt_avg)

    print("\n" + "=" * 70)
    print("ENSEMBLE TEST EVALUATION (once, threshold=%.2f)" % best_t)
    print("=" * 70)
    print(f"  TN={tn} FP={fp} FN={fn} TP={tp}")
    for k, v in [("accuracy", acc), ("ROC-AUC", auc), ("PR-AUC", prauc),
                 ("sensitivity", sens), ("specificity", spec),
                 ("precision", prec), ("NPV", npv), ("F1", f1),
                 ("balanced_accuracy", bal), ("MCC", mcc), ("brier", brier)]:
        print(f"  {k:18}: {v:.4f}")

    # ---- Baseline single XGBoost for comparison (threshold by validation acc) ----
    _, rows_s = FP.validate_thresholds(single_model, Xval, yval)
    t0, best0 = 0.44, -1
    for r in rows_s:
        if r["accuracy"] > best0:
            best0, t0 = r["accuracy"], r["threshold"]
    p0 = single_model.predict_proba(Xte)[:, 1]
    pred0 = (p0 >= t0).astype(int)
    cm0 = confusion_matrix(yte, pred0)
    tn0, fp0, fn0, tp0 = cm0.ravel()
    acc0 = (tp0 + tn0) / (tp0 + tn0 + fp0 + fn0)
    print("\n" + "-" * 70)
    print("BASELINE single XGBoost (threshold by validation accuracy):")
    print(f"  threshold={t0:.2f}  test_acc={acc0:.4f}  test_AUC="
          f"{roc_auc_score(yte, p0):.4f}  FP={fp0} FN={fn0}")
    print("-" * 70)
    print(f"ACCURACY IMPROVEMENT: {acc - acc0:+.4f} (+{(acc - acc0)*100:+.2f} pp)")

    # ---- Save artefacts ----
    result = {
        "model": "simple_average_ensemble_XGB_CatBoost_LightGBM",
        "threshold": float(best_t),
        "metrics": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
                    "accuracy": acc, "ROC_AUC": auc, "PR_AUC": prauc,
                    "sensitivity": sens, "specificity": spec, "precision": prec,
                    "NPV": npv, "F1": f1, "balanced_accuracy": bal, "MCC": mcc,
                    "brier": brier},
        "baseline_acc": float(acc0), "baseline_threshold": float(t0),
    }
    with open(os.path.join(P.OUT, "ensemble_result.json"), "w") as f:
        json.dump(result, f, indent=2, default=float)

    joblib.dump({"models": models, "weights": [1/3, 1/3, 1/3],
                 "preprocessor": pre, "threshold": float(best_t)},
                os.path.join(P.MODEL, "ensemble_pipeline.joblib"))

    # predictions CSV (test)
    pred_df = pd.DataFrame({
        "true_label": test["target"].values, "pred_prob": pt_avg,
        "pred_label": pred})
    for c in ["age", "sex", "ccc", "doses_mcv", "fever_duration",
              "rash_duration", "travel", "division"]:
        if c in test.columns:
            pred_df[c] = test[c].values
    pred_df.to_csv(os.path.join(P.OUT, "predictions_ensemble_test.csv"),
                   index=False)

    # confusion matrix plot
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(confusion_matrix=cm,
                           display_labels=["Not Measles", "Measles"]).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Ensemble Confusion Matrix (thr={best_t})")
    fig.tight_layout()
    fig.savefig(os.path.join(P.FIG, "ensemble_confusion_matrix.png"), dpi=150)
    plt.close(fig)

    print("\nSaved -> output/ensemble_result.json, "
          "output/predictions_ensemble_test.csv, "
          "model/ensemble_pipeline.joblib, figures/ensemble_confusion_matrix.png")


if __name__ == "__main__":
    main()
