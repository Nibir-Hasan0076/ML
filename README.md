# Early Measles Detection — Clinical Prediction Pipeline

Predicts whether a suspected patient is **Measles-positive at initial clinical
presentation** from early-presentation features only (no laboratory results,
no final classification).

Data: `BAN_MR_FF_SEARO_EW-22_2026.xlsx` (sheet `MR Data`, 30,410 cases, 58 cols).
The surveillance file `Measles Surveillance Indicators ...xlsx` is aggregate
division-level data and is **not** case-level, so it is not used for modelling.

## Setup

```powershell
python -m pip install pandas numpy scikit-learn xgboost matplotlib seaborn \
    shap imbalanced-learn optuna lightgbm catboost openpyxl
```

## Run everything

From the project root:

```powershell
python src/run_all.py
```

Or run individual steps:

| Step | Script | Output |
|------|--------|--------|
| 1. Data audit | `src/01_data_audit.py` | console + decision table |
| 2. Feature selection | `src/02_feature_selection.py` | `output/feature_importance.csv`, `output/feature_subset_experiment.csv`, `output/subset_*.csv` |
| 3. Model comparison | `src/03_model_comparison.py` | `output/model_comparison.csv`, `output/imbalance_strategy.csv` |
| 4. Hyperparameter tuning | `src/04_hyperparameter_tuning.py` | `output/best_params.json` |
| 5. Threshold optimisation | `src/05_threshold.py` | `output/threshold_curve.csv`, `output/final_threshold.json` |
| 6. Final test evaluation | `src/06_final_eval.py` | `output/final_metrics.json`, `output/predictions_test.csv`, figures |
| 7. SHAP | `src/07_shap.py` | `figures/shap_*.png`, `output/shap_importance.csv` |
| 8. Error analysis | `src/08_error_analysis.py` | `output/error_analysis*.csv/json` |
| 9. Robustness | `src/09_robustness.py` | `output/robustness.csv` |
| 10. Report | `src/10_report.py` | `output/final_report.md` |
| 11. Ensemble (accuracy boost) | `src/11_ensemble.py` | `output/ensemble_result.json`, `model/ensemble_pipeline.joblib` |

## Methods summary

- **Target**: `Measles-positive` = `CLASS == '2-Laboratory Confirmed Measles'`.
  `Not-Measles` = Discarded + Rubella-confirmed. `Pending` cases (21,416, no
  outcome) are excluded from supervised training.
- **No leakage**: all laboratory results, final classifications and
  post-diagnosis fields removed as predictors and used only to build the label.
  Preprocessing is fitted on training data only; validation/test are transformed.
- **Data cleaning**: normalised label casing/messy categories, cleaned
  impossible ages, capped implausible vaccine doses (>2), handled missing values.
- **Feature engineering**: age, fever/rash duration, vaccination history, on-onset
  epi-week, prodrome triad (`CCC`), age groups, symptom-combination flags.
- **Model**: simple-average GBM ensemble (XGBoost + CatBoost + LightGBM, 
  `binary:logistic` / logistic objectives, tuned with Optuna), on the best 
  15-feature subset; threshold locked on validation (0.42) balancing 
  sensitivity vs specificity.

## Key result (test set, evaluated once)

Final deployed model = **ensemble**; single XGBoost shown in parentheses.

| Metric | Ensemble | (single XGB) |
|--------|---------:|-------------:|
| Accuracy | **0.793** | 0.788 |
| ROC-AUC / PR-AUC | **0.851 / 0.914** | 0.849 / 0.912 |
| Sensitivity / Specificity | **0.934 / 0.536** | 0.917 / 0.552 |
| MCC / F1 | **0.531 / 0.853** | 0.520 / 0.848 |
| FP / FN | 222 / **58** | 214 / 72 |

The ensemble lifts test accuracy by ~0.4–0.7 pp and markedly reduces false
negatives (58 vs 72) — valuable for early detection — with slightly better
AUC, PR-AUC, F1 and MCC. Robustness across 5 seeds: AUC **0.846 ± 0.004**.
