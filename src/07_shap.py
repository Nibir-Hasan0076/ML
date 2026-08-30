"""Step 7 - SHAP interpretability for the final XGBoost model.

Computes SHAP values on the untouched TEST set using the same fitted
preprocessing pipeline, and generates:
  - global SHAP importance bar
  - beeswarm summary (feature -> direction of influence)
  - representative individual predictions
Saves -> figures/shap_bar.png, figures/shap_beeswarm.png, SHAP summary CSV.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

import final_pipeline as FP
import pipeline as P
import encoding as ENC

import joblib


def main():
    sets, cols, pre, model, X, y = FP.make_split_sets()
    _, _, test = sets
    _, Xval, Xte = X
    _, _, yte = y

    # feature names after one-hot encoding
    feat_names = [n.replace("num__", "").replace("cat__", "") or n
                  for n in pre.get_feature_names_out()]

    print("Computing SHAP values on test set ...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(Xte)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # ---- Global importance (mean |SHAP|) ----
    mean_abs = np.abs(shap_values).mean(axis=0)
    imp = pd.Series(mean_abs, index=feat_names).sort_values(ascending=False)
    imp.to_csv(os.path.join(P.OUT, "shap_importance.csv"))
    print("\nTop features by mean|SHAP|:")
    print(imp.head(20).round(4).to_string())

    # ---- Bar plot (top 20) ----
    plt.figure(figsize=(9, 7))
    top = imp.head(20).iloc[::-1]
    plt.barh(top.index, top.values)
    plt.xlabel("mean |SHAP value|")
    plt.title("Global SHAP feature importance (top 20)")
    plt.tight_layout()
    plt.savefig(os.path.join(P.FIG, "shap_bar.png"), dpi=150)
    plt.close()

    # ---- Beeswarm with compact feature names ----
    plt.figure(figsize=(11, 9))
    shap.summary_plot(shap_values, Xte, feature_names=feat_names,
                      max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(P.FIG, "shap_beeswarm.png"), dpi=150)
    plt.close()

    print("\nSaved -> figures/shap_bar.png, shap_beeswarm.png,"
          " output/shap_importance.csv")

    # ---- Representative individual predictions ----
    # highest-prob true measles, lowest-prob false negative, a clear negative
    prob_test = model.predict_proba(Xte)[:, 1]
    yte_np = np.asarray(yte)
    rows = []
    def row(name, i):
        rows.append({
            "case": name,
            "index": int(i),
            "true_label": int(yte_np[i]),
            "prob": round(float(prob_test[i]), 3),
            "predicted": "Measles" if prob_test[i] >= 0.44 else "Not Measles",
        })
    # most confident measles
    pos_idx = np.where(yte_np == 1)[0]
    row("Most-confident Measles", pos_idx[np.argmax(prob_test[pos_idx])])
    # least confident among true measles (hard FN-zone)
    fn_zone = pos_idx[prob_test[pos_idx] < 0.44]
    if len(fn_zone):
        row("Least-confident true Measles (FN risk)",
            fn_zone[np.argmin(prob_test[fn_zone])])
    # most confident non-measles
    neg_idx = np.where(yte_np == 0)[0]
    row("Most-confident Not Measles", neg_idx[np.argmin(prob_test[neg_idx])])
    # a clearly-misclassified FP with high prob
    fp_idx = np.where((yte_np == 0) & (prob_test >= 0.44))[0]
    if len(fp_idx):
        row("High-prob False Positive", fp_idx[np.argmax(prob_test[fp_idx])])

    print("\nRepresentative individual predictions (test set):")
    rep = pd.DataFrame(rows)
    print(rep.to_string(index=False))

    # SHAP waterplots for the first two representatives
    Xte_dense = Xte.toarray() if hasattr(Xte, "toarray") else np.asarray(Xte)
    for _, r in rep.head(2).iterrows():
        i = int(r["index"])
        plt.figure(figsize=(8, 8))
        shap.plots.waterfall(shap.Explanation(
            values=shap_values[i, :],
            base_values=explainer.expected_value,
            data=Xte_dense[i, :], feature_names=feat_names), show=False,
            max_display=15)
        plt.title(f"{r['case']} | prob={r['prob']}")
        plt.tight_layout()
        safe = r["case"].replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        plt.savefig(os.path.join(P.FIG, f"shap_waterfall_{safe}.png"), dpi=150)
        plt.close()
    print("Saved representative waterfall plots.")


if __name__ == "__main__":
    main()
