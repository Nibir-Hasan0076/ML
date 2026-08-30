"""Step 8 - Error analysis on the TEST set.

Identifies false positives (predicted Measles, actually Not Measles) and false
negatives (predicted Not Measles, actually Measles) and characterises their
clinical profile (age group, sex, vaccination, symptom durations, geography).
Saves -> output/error_analysis.csv + a summary table.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import numpy as np
import pandas as pd

import final_pipeline as FP
import pipeline as P


def main():
    pred_df = pd.read_csv(os.path.join(P.OUT, "predictions_test.csv"))
    with open(os.path.join(P.OUT, "final_threshold.json")) as f:
        thr = json.load(f)["threshold"]

    pred_df["bucket"] = np.select(
        [(pred_df["true_label"] == 0) & (pred_df["pred_label"] == 1),
         (pred_df["true_label"] == 1) & (pred_df["pred_label"] == 0)],
        ["False Positive", "False Negative"],
        default="Correct",
    )

    fp = pred_df[pred_df["bucket"] == "False Positive"]
    fn = pred_df[pred_df["bucket"] == "False Negative"]
    tp = pred_df[(pred_df["true_label"] == 1) & (pred_df["pred_label"] == 1)]
    tn = pred_df[(pred_df["true_label"] == 0) & (pred_df["pred_label"] == 0)]

    print("=" * 70)
    print(f"ERROR ANALYSIS (test set, threshold={thr})")
    print("=" * 70)
    print(f"False Positives (pred Measles, actually Not Measles): {len(fp)}")
    print(f"False Negatives (pred Not Measles, actually Measles): {len(fn)}")

    def profile(sub, name):
        print(f"\n--- {name} (n={len(sub)}) ---")
        if len(sub) == 0:
            return
        age_groups = pd.cut(sub["age"], [-1, 1, 5, 12, 18, 120],
                            labels=["<1y", "1-5y", "6-12y", "13-18y", ">18y"])
        print("Age group:", age_groups.value_counts().to_string())
        print("Sex:", sub["sex"].value_counts().to_string())
        print("CCC (prodrome triad):", sub["ccc"].value_counts().to_string())
        print("Measles vaccine doses:", sub["doses_mcv"].value_counts().to_string())
        print("Travel:", sub["travel"].value_counts().to_string())
        print("Median fever_duration:", sub["fever_duration"].median(),
              "| median rash_duration:", sub["rash_duration"].median())
        print("Divisions:", sub["division"].value_counts().to_string())

    profile(fp, "FALSE POSITIVES")
    profile(fn, "FALSE NEGATIVES")
    profile(tp, "TRUE POSITIVES (reference)")
    profile(tn, "TRUE NEGATIVES (reference)")

    # Save per-case error records with rounded prob
    out = pred_df[pred_df["bucket"] != "Correct"].copy()
    out["pred_prob"] = out["pred_prob"].round(3)
    out.to_csv(os.path.join(P.OUT, "error_analysis.csv"), index=False)
    print("\nSaved -> output/error_analysis.csv")

    summary = {
        "n_fp": int(len(fp)), "n_fn": int(len(fn)),
        "fp_median_age": float(fp["age"].median()) if len(fp) else None,
        "fn_median_age": float(fn["age"].median()) if len(fn) else None,
        "fp_unvaccinated_pct":
            float((fp["doses_mcv"] == 0).mean()) if len(fp) else None,
        "fn_unvaccinated_pct":
            float((fn["doses_mcv"] == 0).mean()) if len(fn) else None,
        "fp_ccc_yes_pct": float((fp["ccc"] == "YES").mean()) if len(fp) else None,
        "fn_ccc_yes_pct": float((fn["ccc"] == "YES").mean()) if len(fn) else None,
    }
    with open(os.path.join(P.OUT, "error_analysis_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved -> output/error_analysis_summary.json")


if __name__ == "__main__":
    main()
