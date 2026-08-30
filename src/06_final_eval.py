"""Step 6 - Final evaluation on the untouched TEST set (exactly once).

Generates confusion matrix, ROC, PR and calibration curves, saves the trained
model + preprocessing pipeline + selected feature list + predictions CSV and a
final evaluation report JSON.
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
from sklearn.metrics import (
    roc_curve, roc_auc_score, average_precision_score, precision_recall_curve,
    confusion_matrix, ConfusionMatrixDisplay, brier_score_loss)
from sklearn.calibration import CalibrationDisplay

import final_pipeline as FP
import pipeline as P

import joblib


def main():
    sets, cols, pre, model, X, y = FP.make_split_sets()
    (train, val, test), _, _, _, (Xtr, Xval, Xte), (ytr, yval, yte) = \
        sets, cols, pre, model, X, y

    with open(os.path.join(P.OUT, "final_threshold.json")) as f:
        thr = json.load(f)["threshold"]

    prob_test = model.predict_proba(Xte)[:, 1]
    pred_test = (prob_test >= thr).astype(int)
    cm = confusion_matrix(yte, pred_test)

    print("=" * 70)
    print("FINAL TEST EVALUATION (untouched test set, threshold=%.2f)" % thr)
    print("=" * 70)
    metrics = P.full_metrics(yte, prob_test, threshold=thr)
    print("Confusion Matrix:")
    print("          Pred Neg   Pred Pos")
    print(f"True Neg  {metrics['TN']:>9}  {metrics['FP']:>9}")
    print(f"True Pos  {metrics['FN']:>9}  {metrics['TP']:>9}")
    print("\nMetric table:")
    order = ["TN", "FP", "FN", "TP", "accuracy", "ROC_AUC", "PR_AUC",
             "sensitivity", "specificity", "precision", "NPV", "F1",
             "balanced_accuracy", "MCC", "FPR", "FNR"]
    for k in order:
        print(f"  {k:18}: {metrics[k]:.4f}"
              if isinstance(metrics[k], float) else f"  {k:18}: {metrics[k]}")

    # Brier score (calibration)
    brier = brier_score_loss(yte, prob_test)
    print(f"  brier_score         : {brier:.4f}")

    # --------------------------- SAVE ARTIFACTS ---------------------------
    joblib.dump(model, os.path.join(P.MODEL, "final_xgb_model.joblib"))
    joblib.dump(pre, os.path.join(P.MODEL, "preprocessor.joblib"))
    pd.Series(cols).to_csv(os.path.join(P.OUT, "final_feature_list.csv"),
                           index=False, header=False)
    with open(os.path.join(P.OUT, "final_metrics.json"), "w") as f:
        json.dump({**metrics, "brier": brier, "threshold": thr}, f,
                  indent=2, default=float)

    pred_df = pd.DataFrame({
        "true_label": test["target"].values,
        "pred_prob": prob_test,
        "pred_label": pred_test,
    })
    # attach a few clinical columns for later error analysis
    for c in ["age", "sex", "ccc", "doses_mcv", "fever_duration",
              "rash_duration", "travel", "division"]:
        if c in test.columns:
            pred_df[c] = test[c].values
    pred_df.to_csv(os.path.join(P.OUT, "predictions_test.csv"), index=False)
    print("\nSaved: model/final_xgb_model.joblib, preprocessor.joblib,"
          "\noutput/final_feature_list.csv, final_metrics.json,"
          " predictions_test.csv")

    # --------------------------- PLOTS ---------------------------
    # Confusion matrix
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(confusion_matrix=cm,
                           display_labels=["Not Measles", "Measles"]).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Test Confusion Matrix (thr={thr})")
    fig.tight_layout()
    fig.savefig(os.path.join(P.FIG, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    # ROC
    fpr, tpr, _ = roc_curve(yte, prob_test)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot(fpr, tpr, lw=2, label=f"XGBoost (AUC={roc_auc_score(yte, prob_test):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (test)"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(P.FIG, "roc_curve.png"), dpi=150)
    plt.close(fig)

    # PR curve
    prec, rec, _ = precision_recall_curve(yte, prob_test)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot(rec, prec, lw=2,
            label=f"XGBoost (PR-AUC={average_precision_score(yte, prob_test):.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve (test)"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(P.FIG, "pr_curve.png"), dpi=150)
    plt.close(fig)

    # Calibration curve
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    CalibrationDisplay.from_predictions(yte, prob_test, n_bins=10, ax=ax)
    ax.set_title("Calibration Curve (test)")
    fig.tight_layout(); fig.savefig(os.path.join(P.FIG, "calibration_curve.png"), dpi=150)
    plt.close(fig)

    print("\nPlots saved to figures/ : confusion_matrix.png, roc_curve.png,"
          " pr_curve.png, calibration_curve.png")


if __name__ == "__main__":
    main()
