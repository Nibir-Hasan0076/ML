"""Step 13 - FINAL operating point for missed-measles priority.

User decision: MISSED MEASLES is the worst error (worse than a false alarm).
So we want to push sensitivity very high, but NOT so far that accuracy collapses
and false alarms explode (user also dislikes FPs).

Rule locked on VALIDATION only, evaluated ONCE on the untouched TEST set:

  Choose the threshold that MAXIMISES SENSITIVITY subject to
  ACCURACY staying >= a floor AND FP not ballooning beyond a cap.

This finds the lowest-FN threshold that keeps the model reasonably accurate,
instead of chasing 100% sensitivity at the cost of 400 false alarms.

Leakage-safe: preprocessor fit on train only; threshold chosen on val.
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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier
import joblib

import pipeline as P
import final_pipeline as FP


def main():
    print("=" * 70)
    print("FINAL MODEL - missed-measles priority (FN-min, keep accuracy)")
    print("=" * 70)

    sets, cols, pre, single_model, X, y = FP.make_split_sets()
    (train, val, test) = sets
    (Xtr, Xval, Xte) = X
    (ytr, yval, yte) = y

    cfg = FP.get_config()
    models = {
        "XGBoost": XGBClassifier(
            **cfg, eval_metric="logloss", tree_method="hist",
            random_state=P.SEED, verbosity=0, use_label_encoder=False, n_jobs=-1),
    }
    try:
        from catboost import CatBoostClassifier
        from lightgbm import LGBMClassifier
        models["CatBoost"] = CatBoostClassifier(
            iterations=600, learning_rate=0.05, depth=6,
            scale_pos_weight=cfg["scale_pos_weight"],
            random_state=P.SEED, verbose=0, allow_writing_files=False)
        models["LightGBM"] = LGBMClassifier(
            n_estimators=600, learning_rate=0.05, num_leaves=31,
            scale_pos_weight=cfg["scale_pos_weight"],
            random_state=P.SEED, verbose=-1)
    except ImportError:
        pass

    val_prob, te_prob = {}, {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        val_prob[name] = m.predict_proba(Xval)[:, 1]
        te_prob[name] = m.predict_proba(Xte)[:, 1]

    pv_avg = np.mean([val_prob[n] for n in models], axis=0)
    pt_avg = np.mean([te_prob[n] for n in models], axis=0)

    # ---- Lock threshold on VALIDATION --------------------------------
    # For every threshold compute (sensitivity, accuracy, FP, FN) on validation.
    # Pick the LOWEST-FN threshold whose accuracy is >= acc_floor.
    ACC_FLOOR = 0.72   # keep accuracy reasonable (baseline ceiling ~0.79)
    FP_CAP = 160       # don't let false alarms explode either

    best_t = None
    for t in np.round(np.arange(0.10, 0.90, 0.01), 2):
        pred = (pv_avg >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(yval, pred).ravel()
        acc = (tp + tn) / (tp + tn + fp + fn)
        sens = tp / (tp + fn)
        if acc >= ACC_FLOOR and fp <= FP_CAP:
            if best_t is None or fn < best_t[1] or (
                    fn == best_t[1] and acc > best_t[2]):
                best_t = (t, fn, acc, sens)
    if best_t is None:
        best_t = (0.50, np.inf, 0.0, 0.0)
        print("  WARNING: no threshold met floors; fell back to 0.50")
    sel_t, sel_fn, sel_acc, sel_sens = best_t
    print(f"\nThreshold locked on VALIDATION: t={sel_t:.2f} "
          f"(FN={int(sel_fn)}, acc={sel_acc:.3f}, sens={sel_sens:.3f})")

    # ---- Evaluate ONCE on test ---------------------------------------
    pred = (pt_avg >= sel_t).astype(int)
    cm = confusion_matrix(yte, pred)
    tn, fp, fn, tp = cm.ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
    sens = tp / (tp + fn); spec = tn / (tn + fp)
    prec = tp / (tp + fp); npv = tn / (tn + fn)
    f1 = 2 * prec * sens / (prec + sens)
    bal = (sens + spec) / 2
    mcc = (tp * tn - fp * fn) / max(1e-9, (
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5)

    print("\n" + "=" * 70)
    print("FINAL TEST EVALUATION (once, threshold=%.2f)" % sel_t)
    print("=" * 70)
    print(f"  TN={tn} FP={fp} FN={fn} TP={tp}")
    for k, v in [("accuracy", acc), ("sensitivity", sens), ("specificity", spec),
                 ("precision", prec), ("NPV", npv), ("F1", f1),
                 ("balanced_accuracy", bal), ("MCC", mcc)]:
        print(f"  {k:20}: {v:.4f}")
    print(f"\n  Missed measles (FN) on test: {fn}")

    # ---- Compare with old accuracy-max operating point ----------------
    pred_old = (pt_avg >= 0.42).astype(int)  # previous ensemble threshold
    cml = confusion_matrix(yte, pred_old)
    tno, fpo, fno, tpo = cml.ravel()
    print("\n  PREVIOUS (accuracy-max, t=0.42): "
          f"acc={acc if False else (tpo+tno)/(tpo+tno+fpo+fno):.3f} "
          f"FP={fpo} FN={fno}")

    # ---- Save ----------------------------------------------------------
    result = {
        "model": "simple_average_ensemble_XGB_CatBoost_LightGBM",
        "rule": "min_FN_subject_to_acc_floor_and_FP_cap",
        "acc_floor": ACC_FLOOR, "fp_cap": FP_CAP,
        "threshold": float(sel_t),
        "metrics": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
                    "accuracy": acc, "sensitivity": sens, "specificity": spec,
                    "precision": prec, "NPV": npv, "F1": f1,
                    "balanced_accuracy": bal, "MCC": mcc},
        "previous_accuracy_max": {"threshold": 0.42, "FP": int(fpo),
                                  "FN": int(fno),
                                  "accuracy": (tpo+tno)/(tpo+tno+fpo+fno)},
    }
    with open(os.path.join(P.OUT, "final_model_missed_measles.json"), "w") as f:
        json.dump(result, f, indent=2, default=float)
    joblib.dump({"models": models, "weights": [1/len(models)]*len(models),
                 "preprocessor": pre, "threshold": float(sel_t)},
                os.path.join(P.MODEL, "final_model_missed_measles.joblib"))

    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(cm, display_labels=["Not Measles", "Measles"]).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Final Confusion Matrix (thr={sel_t:.2f})")
    fig.tight_layout()
    fig.savefig(os.path.join(P.FIG, "final_fp_fn_confusion.png"), dpi=150)
    plt.close(fig)

    print("\nSaved -> output/final_model_missed_measles.json, "
          "model/final_model_missed_measles.joblib, "
          "figures/final_fp_fn_confusion.png")


if __name__ == "__main__":
    main()
