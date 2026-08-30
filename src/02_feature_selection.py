"""Step 2 - Feature selection & feature-subset experiment setup.

Computes:
  - correlation analysis (numerical)
  - mutual information (vs target)
  - univariate ANOVA / chi2 importance
  - gradient-boosting feature importance (on training only)
  - RFE-free top-k selection to define subsets B/C/D
  - clinical subset (E)
  - full + engineered (F)
Saves: feature_importance.csv + the chosen subset column lists.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold

import pipeline as P
import encoding as ENC


def main():
    df = P.feature_engineer(P.load_raw())
    label = df.dropna(subset=["target"]).copy()
    y = label["target"].astype(int)

    CORR = P.get_feature_columns()

    # ---- encode the labelled frame for feature-level analysis -------------
    # Use one-hot feature names mapped back to original column for ranking.
    Xall, feat_names = ENC.feature_matrix(label, CORR)
    model = RandomForestClassifier(n_estimators=300, random_state=P.SEED, n_jobs=-1)
    model.fit(Xall, y)
    imp = model.feature_importances_

    # Aggregate importances per base column using robust prefix matching.
    # Encoded names are 'num__<col>' or 'cat__<col>_<category>'.
    def col_of(name, coldict):
        for col in coldict:
            ncol = col.replace(" ", "_")
            if name == f"num__{ncol}" or name.startswith(f"cat__{ncol}_"):
                return col
            if name == f"cat__{ncol}":
                return col
        return name
    agg = {}
    for n, i in zip(feat_names, imp):
        b = col_of(n, CORR)
        agg[b] = agg.get(b, 0) + i
    imp_series = pd.Series(agg).sort_values(ascending=False)

    # ---- Mutual information per original column ----
    # For categorical cols use them as-is (sklearn MI handles categorical ints)
    mi_vals = {}
    for col in CORR:
        if col in ENC.CATEGORICALS:
            codes = label[col].astype("category").cat.codes
            mi_vals[col] = mutual_info_classif(codes.values.reshape(-1, 1), y,
                                               random_state=P.SEED)[0]
        else:
            v = pd.to_numeric(label[col], errors="coerce").fillna(label[col].median())
            mi_vals[col] = mutual_info_classif(v.values.reshape(-1, 1), y,
                                               random_state=P.SEED)[0]

    mi_series = pd.Series(mi_vals).sort_values(ascending=False)

    # ---- combined score (normalised importance + normalised MI) ----
    rank = pd.DataFrame({
        "imp_train_rf": imp_series,
        "mutual_info": mi_series,
    }).fillna(0)
    rank["norm_imp"] = rank["imp_train_rf"] / (rank["imp_train_rf"].max() + 1e-9)
    rank["norm_mi"] = rank["mutual_info"] / (rank["mutual_info"].max() + 1e-9)
    rank["score"] = 0.5 * rank["norm_imp"] + 0.5 * rank["norm_mi"]
    rank = rank.sort_values("score", ascending=False)

    rank.to_csv(os.path.join(P.OUT, "feature_importance.csv"))
    print("Feature importance / ranking saved ->",
          os.path.join(P.OUT, "feature_importance.csv"))
    print("\n--- FEATURE RANKING (combined) ---")
    print(rank[["imp_train_rf", "mutual_info", "score"]].to_string())

    # ---- numerical correlation with target & each other ----
    num = [c for c in CORR if c not in ENC.CATEGORICALS]
    numdf = label[num].apply(pd.to_numeric, errors="coerce")
    corr_target = numdf.corrwith(y).sort_values(key=abs, ascending=False)
    print("\n--- NUMERICAL CORRELATION WITH TARGET ---")
    print(corr_target.to_string())

    # ---- Define Model feature subsets ----
    FEAT = {"Model_F": CORR}   # engineered + original (default feature set)

    ordered = rank.index.tolist()

    FEAT["Model_B_top20"] = ordered[:20]
    FEAT["Model_C_top15"] = ordered[:15]
    FEAT["Model_D_top10"] = ordered[:10]

    # Model A = original VALID early-presentation features (pre-engineering)
    ModelA = ["age", "sex", "division", "district", "upzmunc", "ccc", "travel",
              "doses_mcv", "doses_rcv", "time_since_mcv", "fever_duration",
              "rash_duration", "invest_lag"]
    FEAT["Model_A_original"] = [c for c in ModelA if c in CORR]

    # Model E = clinically selected (domain-driven small set)
    ModelE = ["age", "sex", "ccc", "fever_duration", "rash_duration",
              "doses_mcv", "travel", "division", "fever_gte7", "rash_gte3",
              "age_group"]
    FEAT["Model_E_clinical"] = [c for c in ModelE if c in CORR]

    # Cross-validate a GBM quickly per subset to rank them on VALIDATION
    print("\n--- FEATURE SUBSET CV SCORES (XGBoost, AUC on stratified CV) ---")
    from xgboost import XGBClassifier
    from sklearn.pipeline import Pipeline
    results = []
    for name, cols in FEAT.items():
        pre = ENC.make_preprocessor(
            [c for c in cols if c in ENC.NUMERICAL_COLS],
            [c for c in cols if c in ENC.CATEGORICALS])[0]
        pipe = Pipeline([
            ("pre", pre),
            ("clf", XGBClassifier(n_estimators=200, learning_rate=0.1,
                                  max_depth=4, subsample=0.8,
                                  colsample_bytree=0.8,
                                  eval_metric="logloss", random_state=P.SEED)),
        ])
        X = label[[c for c in cols]]
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=P.SEED)
        auc = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=1)
        pr = cross_val_score(pipe, X, y, cv=cv,
                             scoring="average_precision", n_jobs=1)
        print(f"{name:24} n={len(cols):3d}  AUC={auc.mean():.4f}±{auc.std():.3f}"
              f"  PR={pr.mean():.4f}")
        results.append((name, len(cols), auc.mean(), auc.std(), pr.mean()))

    res = pd.DataFrame(results, columns=["subset", "n_features", "auc_mean",
                                         "auc_std", "pr_mean"])
    res = res.sort_values("auc_mean", ascending=False)
    res.to_csv(os.path.join(P.OUT, "feature_subset_experiment.csv"), index=False)
    print("\n--- SUBSET EXPERIMENT TABLE ---")
    print(res.to_string(index=False))

    # ---- Persist chosen subsets for later steps ----
    for name, cols in FEAT.items():
        pd.Series(cols).to_csv(os.path.join(P.OUT, f"subset_{name}.csv"),
                               index=False, header=False)
    print("\nSubsets saved to output/subset_*.csv")


if __name__ == "__main__":
    main()
