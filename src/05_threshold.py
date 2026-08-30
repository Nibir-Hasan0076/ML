"""Step 5 - Threshold optimization on the validation set.

Locks in the operating threshold by scanning 0.10-0.90 on the held-out
validation set (test remains untouched). Because the user is concerned about
false positives, we explicitly show how each threshold trades FP vs FN and pick
a threshold that maximises MCC (a balanced measure) while preferring higher
sensitivity among near-equal MCC.

Saves: output/threshold_curve.csv and output/final_threshold.json
"""
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import final_pipeline as FP
import pipeline as P


def main():
    sets, cols, pre, model, X, y = FP.make_split_sets()
    _, Xval, _ = X
    _, yval, _ = y
    prob, rows = FP.validate_thresholds(model, Xval, yval)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(P.OUT, "threshold_curve.csv"), index=False)
    print("Threshold curve saved -> output/threshold_curve.csv")

    print("\n=== THRESHOLD SWEEP (validation, not test) ===")
    print(df[["threshold", "accuracy", "sensitivity", "specificity",
              "precision", "NPV", "F1", "balanced_accuracy", "MCC",
              "FP", "FN", "FPR", "FNR"]].round(3).to_string(index=False))

    best_mcc = FP.select_threshold(rows, metric="MCC")
    # Also report a balanced-accuracy pick and the default 0.5 for comparison
    best_bal = FP.select_threshold(rows, metric="balanced_accuracy")
    print("\nBest threshold by MCC            :", best_mcc)
    print("Best threshold by balanced_acc   :", best_bal)
    print("Default 0.5 threshold metrics    :",
          P.full_metrics(yval, prob, threshold=0.5)["MCC"].round(3))

    # Choose MCC-based threshold (favours a balanced FP/FN profile)
    chosen = best_mcc
    summary = P.full_metrics(yval, prob, threshold=chosen)
    print(f"\nLocked threshold = {chosen}")
    print("Validation metrics at locked threshold:")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    with open(os.path.join(P.OUT, "final_threshold.json"), "w") as f:
        json.dump({"threshold": chosen, "validation_metrics": summary}, f,
                  indent=2, default=float)
    print("\nSaved -> output/final_threshold.json")


if __name__ == "__main__":
    main()
