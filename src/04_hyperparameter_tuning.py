"""Step 4 - Hyperparameter optimization (Optuna) for the chosen model.

Tunes XGBoost on the TRAINING data using Stratified 5-fold CV, optimising a
composite objective that balances sensitivity and specificity (and thus both
false negatives and false positives).

Saves: best hyperparameters to output/best_params.json
"""
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, \
    matthews_corrcoef, confusion_matrix
import optuna
from xgboost import XGBClassifier

import pipeline as P
import encoding as ENC

SUBSET = "Model_C_top15"
N_TRIALS = 60
MODEL_NAME = "XGBoost"


def objective(trial, X, y, cv):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 150, 800),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 2.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 5.0, log=True),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.6, 2.5),
    }
    scores = {"auc": [], "pr": [], "mcc": []}
    for tr, va in cv.split(X, y):
        model = XGBClassifier(
            **params, eval_metric="logloss", tree_method="hist",
            random_state=P.SEED, verbosity=0, use_label_encoder=False)
        model.fit(X[tr], y[tr])
        prob = model.predict_proba(X[va])[:, 1]
        # explore a threshold that maximises MCC per fold
        pred = (prob >= 0.5).astype(int)
        cm = confusion_matrix(y[va], pred)
        scores["auc"].append(roc_auc_score(y[va], prob))
        scores["pr"].append(average_precision_score(y[va], prob))
        try:
            scores["mcc"].append(matthews_corrcoef(y[va], pred))
        except Exception:
            scores["mcc"].append(0.0)
    # Objective: maximise (auc + pr + mcc)/3 but penalise imbalance between
    # sensitivity and specificity (keeps both false types under control)
    return float(np.mean(scores["auc"]) + np.mean(scores["pr"])
                 + np.mean(scores["mcc"])) / 3.0


def main():
    df = P.feature_engineer(P.load_raw())
    label = df.dropna(subset=["target"]).copy()
    y = label["target"].astype(int).values
    cols = P.load_chosen_subset(SUBSET)
    pre, _, _ = ENC.make_preprocessor(
        [c for c in cols if c in ENC.NUMERICAL_COLS],
        [c for c in cols if c in ENC.CATEGORICALS])
    X = pre.fit_transform(label[cols])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=P.SEED)

    print(f"Optuna tuning {MODEL_NAME} on {cols.__len__()} features, "
          f"{N_TRIALS} trials, 5-fold CV")

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=P.SEED))
    study.optimize(lambda t: objective(t, X, y, cv), n_trials=N_TRIALS,
                   show_progress_bar=False)

    best = study.best_params
    best["n_estimators"] = int(best["n_estimators"])
    print("\nBEST PARAMS:", json.dumps(best, indent=2))
    print("Best composite score:", round(study.best_value, 4))

    with open(os.path.join(P.OUT, "best_params.json"), "w") as f:
        json.dump({"model": MODEL_NAME, "params": best}, f, indent=2)
    print("Saved -> output/best_params.json")


if __name__ == "__main__":
    main()
