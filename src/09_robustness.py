"""Step 9 - Robustness check.

Trains the final XGBoost model repeatedly across multiple stratified splits
(different random seeds) and reports mean +/- std of the key metrics so we can
see whether the performance is stable and not dependent on one lucky split.
Uses the locked threshold from the earlier step.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import pipeline as P
import encoding as ENC
import final_pipeline as FP

SEEDS = [42, 7, 123, 2024, 999]


def run_seed(df, cols, thr, seed):
    label = df.dropna(subset=["target"]).copy()
    y = label["target"].astype(int)
    train, test = train_test_split(label, test_size=0.30, stratify=y,
                                   random_state=seed)
    val, te = train_test_split(test, test_size=0.50,
                               stratify=test["target"].astype(int),
                               random_state=seed)
    pre, Xtr = FP.build_pre_and_X(train, cols)
    Xval = pre.transform(val[cols]); Xte = pre.transform(te[cols])
    model = FP.train_xgb(Xtr, train["target"].astype(int).values, seed=seed)
    # threshold re-tuned on this run's validation
    _, rows = FP.validate_thresholds(model, Xval, val["target"].astype(int).values)
    t = FP.select_threshold(rows, metric="MCC")
    prob = model.predict_proba(Xte)[:, 1]
    m = P.full_metrics(te["target"].astype(int).values, prob, threshold=t)
    m["threshold"] = t
    return m


def main():
    with open(os.path.join(P.OUT, "final_threshold.json")) as f:
        thr = json.load(f)["threshold"]
    cols = P.load_chosen_subset(FP.SUBSET)
    df = P.feature_engineer(P.load_raw())

    print(f"ROBUSTNESS CHECK: {len(SEEDS)} random-seed stratified "
          f"80/20 splits, threshold re-tuned per split, test=30% holdout")
    metrics_of_interest = ["accuracy", "ROC_AUC", "PR_AUC", "sensitivity",
                           "specificity", "precision", "F1", "MCC"]
    records = []
    for seed in SEEDS:
        m = run_seed(df, cols, thr, seed)
        records.append({k: m[k] for k in metrics_of_interest})
        print(f"  seed={seed}: AUC={m['ROC_AUC']:.3f} PR={m['PR_AUC']:.3f} "
              f"Sens={m['sensitivity']:.3f} Spec={m['specificity']:.3f} "
              f"MCC={m['MCC']:.3f}")

    res = pd.DataFrame(records)
    agg = pd.DataFrame({
        "mean": res.mean(),
        "std": res.std(),
    })
    print("\nMean +/- Std across seeds:")
    print(agg.round(4).to_string())
    agg.to_csv(os.path.join(P.OUT, "robustness.csv"))
    print("\nSaved -> output/robustness.csv")


if __name__ == "__main__":
    main()
