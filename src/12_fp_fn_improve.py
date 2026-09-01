"""FP/FN improvement experiment - reduce BOTH error types while keeping accuracy.

Two levers, both evaluated on the untouched test set (threshold always locked
on validation, preprocessor fit on train only -> no leakage):

  Lever 1  DECISION RULE: instead of locking threshold on "max accuracy", scan
           each candidate rule and compare how accuracy / FP / FN change.
  Lever 2  FEATURES: compare the current 15-feature subset vs the full
           engineered feature set to see if AUC (and thus BOTH error types)
           can genuinely improve.

Output is a comparison table saved to output/fp_fn_improvement.json
"""
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix, roc_auc_score,
                             average_precision_score)

import pipeline as P
import final_pipeline as FP
import encoding as ENC

from xgboost import XGBClassifier

SUBSET = "Model_C_top15"


def eval_at(X, y, model, t):
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= t).astype(int)
    cm = confusion_matrix(y, pred)
    tn, fp, fn, tp = cm.ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    mcc = (tp * tn - fp * fn) / max(1e-9, (
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5)
    f1 = 2 * (tp / max(1, tp + fp)) * sens / max(1e-9, (
        (tp / max(1, tp + fp)) + sens))
    return {"acc": acc, "sens": sens, "spec": spec, "mcc": mcc, "f1": f1,
            "FP": int(fp), "FN": int(fn), "TP": int(tp), "TN": int(tn),
            "threshold": t}


def lock_threshold(pval, yval, crit="acc", fp_cost=1.0, fn_cost=1.0):
    """Scan thresholds on validation, pick best by the chosen criterion."""
    best = None
    for t in np.round(np.arange(0.10, 0.90, 0.01), 2):
        m = eval_at(pval[:, None], yval, TMP, t) if False else None
        pred = (pval >= t).astype(int)
        cm = confusion_matrix(yval, pred)
        tn, fp, fn, tp = cm.ravel()
        acc = (tp + tn) / (tp + tn + fp + fn)
        sens = tp / (tp + fn)
        spec = tn / (tn + fp)
        bal = (sens + spec) / 2
        if crit == "acc":
            score = acc
        elif crit == "bal":
            score = bal
        elif crit == "cost":
            # penalise FNs (missed measles) more than FPs
            score = acc - (fn_cost * fn / max(1, fn + tp)
                           - fp_cost * fp / max(1, fp + tn))
        elif crit == "mcc":
            denom = max(1e-9, ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5)
            score = (tp * tn - fp * fn) / denom
        row = {"threshold": t, "score": score, "bal": bal, "acc": acc,
               "sens": sens, "spec": spec, "FP": int(fp), "FN": int(fn)}
        if best is None or score > best["score"]:
            best = row
    return best


def main():
    print("=" * 70)
    print("FP/FN IMPROVEMENT EXPERIMENT (strict split, threshold on val only)")
    print("=" * 70)

    df = P.feature_engineer(P.load_raw())
    train, val, test = P.make_split(df)

    variants = {
        "subset_top15": P.load_chosen_subset(SUBSET),
        "full_features": P.get_feature_columns(),
    }

    results = {}
    auc_rows = []
    for name, cols in variants.items():
        print(f"\n--- Feature variant: {name} ({len(cols)} cols) ---")
        pre = ENC.make_preprocessor(
            [c for c in cols if c in ENC.NUMERICAL_COLS],
            [c for c in cols if c in ENC.CATEGORICALS])[0]
        Xtr = pre.fit_transform(train[cols])
        Xval = pre.transform(val[cols])
        Xte = pre.transform(test[cols])
        ytr = train["target"].astype(int).values
        yval = val["target"].astype(int).values
        yte = test["target"].astype(int).values

        cfg = FP.get_config()
        model = XGBClassifier(**cfg, eval_metric="logloss", tree_method="hist",
                              random_state=P.SEED, verbosity=0,
                              use_label_encoder=False, n_jobs=-1)
        model.fit(Xtr, ytr)

        prob_val = model.predict_proba(Xval)[:, 1]
        prob_te = model.predict_proba(Xte)[:, 1]

        auc_val = roc_auc_score(yval, prob_val)
        auc_te = roc_auc_score(yte, prob_te)
        auc_rows.append((name, auc_val, auc_te))
        print(f"  val_AUC={auc_val:.4f}  test_AUC={auc_te:.4f}")

        # ---- Lock threshold on validation under different criteria ----
        crit_rows = []
        for crit, kw in [("acc", {}), ("bal", {}),
                         ("mcc", {}), ("cost", {"fp_cost": 1.0, "fn_cost": 3.0})]:
            def scan(p, y, criterion, fp_c=1.0, fn_c=1.0):
                best = None
                for t in np.round(np.arange(0.10, 0.90, 0.01), 2):
                    pred = (p >= t).astype(int)
                    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
                    acc = (tp + tn) / (tp + tn + fp + fn)
                    sens = tp / (tp + fn); spec = tn / (tn + fp)
                    bal = (sens + spec) / 2
                    if criterion == "acc": score = acc
                    elif criterion == "bal": score = bal
                    elif criterion == "mcc":
                        denom = max(1e-9, ((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn))**0.5)
                        score = (tp*tn - fp*fn) / denom
                    else:
                        score = acc - (fn_c*fn/max(1, fn+tp) - fp_c*fp/max(1, fp+tn))
                    row = {"threshold": t, "score": score, "acc": acc,
                           "sens": sens, "spec": spec, "FP": int(fp), "FN": int(fn)}
                    if best is None or score > best["score"]:
                        best = row
                return best
            b = scan(prob_val, yval, crit, kw.get("fp_cost", 1.0),
                     kw.get("fn_cost", 1.0))
            # evaluate chosen threshold on TEST (once)
            te = eval_at(Xte, yte, model, b["threshold"])
            te["rule"] = f"{crit}{'' if crit!='cost' else '(FN3x)'}"
            te["val_acc"] = b["acc"]
            crit_rows.append(te)

        results[name] = crit_rows
        print("  " + " | ".join(f"{r['rule']}: acc={r['acc']:.3f} "
                                f"FP={r['FP']} FN={r['FN']}" for r in crit_rows))

    # ---- summary ----
    print("\n" + "=" * 70)
    print("THRESHOLD-RULE COMPARISON (locked on val, evaluated once on test)")
    print("=" * 70)
    rows_out = []
    for name, crit_rows in results.items():
        for r in crit_rows:
            rows_out.append({
                "features": name, "rule": r["rule"], "threshold": r["threshold"],
                "accuracy": r["acc"], "sensitivity": r["sens"],
                "specificity": r["spec"], "MCC": r["mcc"], "F1": r["f1"],
                "FPs": r["FP"], "FNs": r["FN"], "TPs": r["TP"], "TNs": r["TN"]})
    df_out = pd.DataFrame(rows_out)
    pd.set_option("display.width", 200)
    print(df_out.to_string(index=False))

    out = {"features_auc": [{"features": n, "val_AUC": v, "test_AUC": t}
                            for n, v, t in auc_rows],
           "rules": rows_out}
    with open(os.path.join(P.OUT, "fp_fn_improvement.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    print("\nSaved -> output/fp_fn_improvement.json")


if __name__ == "__main__":
    main()
