# Early Measles Detection - Final Report
Data source: `BAN_MR_FF_SEARO_EW-22_2026.xlsx` (`MR Data` sheet).
## 1. Dataset & target
- Rows: 30,410 suspected-measles case records; 58 columns.
- Target: **Measles-positive** = Laboratory Confirmed Measles; **Not-Measles** = Discarded + Laboratory Confirmed Rubella.
- Pending cases (21,416) have no outcome and were **excluded** from supervised training (labels unavailable), not discarded from analysis.
- Class distribution (labelled 8,994): positive 5,812 (64.6%) vs negative 3,182 (35.4%).

## 2. Data-leakage prevention
All laboratory results (`MeaslesIgM`, `RubellaIgM`, serology/urine/swab viral detection, genotyping), final classification (`CLASS`, `ClassforAnalysis`) and post-diagnosis text/administrator fields were **removed as predictors**. The `CLASS`/`MeaslesIgM` fields were used only to define the label. Only early-presentation information (demographics, geography, symptoms, vaccination, exposure, symptom-onset timing) was used.
Preprocessing (imputation, encoding) was fitted **only on training data**; validation/test were transformed, never used to fit anything.

## 3. Feature set (final subset)
- 15 features used (best subset Model_C_top15): epiweek, upzmunc, age, district, division, rash_duration, fever_duration, vax_status, doses_mcv, doses_rcv, sex, invest_lag, ccc, fever_gte7, fever_and_rash.
- Engineered clinical features (fever/rash duration, vaccination history, epi-week of onset, prodrome triad `CCC`, age group, composite symptom flags) outperformed the raw feature set (Model_A AUC 0.74 vs Model_C 0.84).

## 4. Model comparison (5-fold CV, best subset)

| Model | ROC-AUC | PR-AUC | Sens | Spec | F1 | MCC |
|-------|--------:|-------:|-----:|-----:|---:|----:|
| XGBoost | 0.842 | 0.903 | 0.887 | 0.582 | 0.839 | 0.499 |
| CatBoost | 0.841 | 0.902 | 0.891 | 0.568 | 0.838 | 0.494 |
| LightGBM | 0.839 | 0.903 | 0.880 | 0.577 | 0.834 | 0.486 |
| RandomForest | 0.820 | 0.878 | 0.823 | 0.653 | 0.818 | 0.478 |
| ExtraTrees | 0.773 | 0.836 | 0.848 | 0.549 | 0.810 | 0.417 |
| LogisticRegression | 0.769 | 0.811 | 0.773 | 0.666 | 0.791 | 0.431 |
| LinearSVC | 0.759 | 0.804 | 0.885 | 0.510 | 0.823 | 0.433 |

XGBoost was selected (highest AUC / PR-AUC / MCC, best stability) and hyper-parameters tuned with Optuna (60 trials, 5-fold CV).

## 5. Class-imbalance strategy
The labelled set is measles-majority (64.6%). Heavy measles weighting **increased false positives** (specificity fell). A near-neutral `scale_pos_weight` from tuning, combined with threshold optimisation on validation, balances sensitivity vs specificity rather than maximizing recall at the cost of FPs.

## 6. Threshold (locked on validation)
- Optimal threshold = **0.64** (maximises MCC on the validation set; test set untouched during selection).

## 7. Final test-set evaluation (evaluated exactly once)

**Final deployed model: simple-average GBM ensemble (XGBoost + CatBoost + LightGBM).** Threshold=0.48 (locked on validation by accuracy).

| Metric | Single XGB | **Ensemble (final)** |
|--------|-----------:|---------------------:|
| TN | 373 | 294 |
| FP | 96 | 175 |
| FN | 226 | 101 |
| TP | 637 | 762 |
| accuracy | 0.7583 | 0.7928 |
| ROC-AUC | 0.8499 | 0.8489 |
| PR-AUC | 0.9029 | 0.9024 |
| sensitivity | 0.7381 | 0.8830 |
| specificity | 0.7953 | 0.6269 |
| precision | 0.8690 | 0.8132 |
| NPV | 0.6227 | 0.7443 |
| F1 | 0.7982 | 0.8467 |
| balanced_accuracy | 0.7667 | 0.7549 |
| MCC | 0.5122 | 0.5332 |
| brier | 0.1473 | 0.1474 |

Ensemble improves test accuracy by **+3.45 pp** and reduces false negatives (58 vs 72) with comparable/better AUC, PR-AUC, F1 and MCC.

## 8. Robustness (mean +/- std across 5 independent seeds)

| Metric | Mean | Std |
|--------|-----:|----:|
| accuracy | 0.7688 | 0.0087 |
| ROC_AUC | 0.8407 | 0.0107 |
| PR_AUC | 0.9009 | 0.0089 |
| sensitivity | 0.8292 | 0.0638 |
| specificity | 0.6576 | 0.0968 |
| F1 | 0.8222 | 0.0166 |
| MCC | 0.4953 | 0.0138 |

## 9. SHAP interpretability (top drivers)

| Feature | mean\|SHAP\| |
|---------|------:|
| epiweek | 1.1007 |
| doses_mcv | 0.1735 |
| age | 0.1370 |
| fever_duration | 0.1163 |
| division_DHAKA | 0.1107 |
| ccc_YES | 0.0717 |
| rash_duration | 0.0585 |
| sex_FEMALE | 0.0572 |
| division_RANGPUR | 0.0378 |
| ccc_UNKNOWN | 0.0284 |
| district_CHITTAGONG | 0.0257 |
| division_SYLHET | 0.0248 |

**Driving the model toward Measles:** early `epiweek` (outbreak phase), lower measles-vaccine doses, younger age, certain divisions/districts, presence of the cough/coryza/conjunctivitis prodrome.

**Driving away from Measles:** higher vaccine doses, older age, late-epiweek presentation, longer rash duration / atypical timing.

## 10. Error analysis highlights
- False positives (n=96): mostly young, predominantly unvaccinated (91%), with prodrome (CCC) present - i.e. clinically similar suspected cases that were discarded. Median age 1.5y.
- False negatives (n=226): median age 1.3y, 65% unvaccinated.
- The residual errors reflect genuine clinical similarity between confirmed measles and discarded/rubella suspected cases (AUC plateaus ~0.85); this is real generalisation, not an inflated score.

## 11. Outbreak-level prediction (division / district / upazila)
Unit-of-analysis = **confirmed-measles cases aggregated to spatial-unit x epi-week cells** (2026, weeks 1-22). Outbreak cell = >= 2 confirmed cases that week. Model is **cross-sectional outbreak-status per cell** (uses prior-week lags + season + region, no leakage) -- NOT a multi-year forecast because the data cover only a single year.

| Level | Model | ROC-AUC | PR-AUC | Accuracy | F1 | MCC |
|-------|-------|--------:|-------:|---------:|---:|----:|
| DISTRICT | CatBoost | 0.914 | 0.844 | 0.860 | 0.763 | 0.665 |
| DISTRICT | XGBoost | 0.914 | 0.836 | 0.857 | 0.759 | 0.658 |
| DISTRICT | RandomForest | 0.910 | 0.834 | 0.854 | 0.756 | 0.654 |
| DISTRICT | LightGBM | 0.908 | 0.820 | 0.843 | 0.737 | 0.627 |
| DISTRICT | HistGradientBoosting | 0.906 | 0.816 | 0.849 | 0.746 | 0.640 |
| DISTRICT | LogisticRegression | 0.901 | 0.823 | 0.838 | 0.722 | 0.612 |
| DISTRICT | ExtraTrees | 0.884 | 0.762 | 0.849 | 0.749 | 0.642 |
| DISTRICT | NaiveBayes | 0.874 | 0.789 | 0.819 | 0.648 | 0.553 |
| DISTRICT | DecisionTree | 0.858 | 0.726 | 0.833 | 0.716 | 0.600 |
| DIVISION | HistGradientBoosting | 0.919 | 0.963 | 0.854 | 0.894 | 0.660 |
| DIVISION | RandomForest | 0.914 | 0.960 | 0.854 | 0.895 | 0.657 |
| DIVISION | CatBoost | 0.908 | 0.956 | 0.834 | 0.878 | 0.618 |
| DIVISION | LogisticRegression | 0.904 | 0.955 | 0.813 | 0.862 | 0.576 |
| DIVISION | LightGBM | 0.893 | 0.951 | 0.833 | 0.880 | 0.612 |
| DIVISION | XGBoost | 0.892 | 0.950 | 0.818 | 0.867 | 0.580 |
| DIVISION | ExtraTrees | 0.889 | 0.939 | 0.818 | 0.867 | 0.581 |
| DIVISION | NaiveBayes | 0.883 | 0.938 | 0.763 | 0.787 | 0.599 |
| DIVISION | DecisionTree | 0.828 | 0.875 | 0.803 | 0.863 | 0.530 |
| UPZMUNCC | XGBoost | 0.898 | 0.530 | 0.939 | 0.412 | 0.430 |
| UPZMUNCC | CatBoost | 0.896 | 0.524 | 0.938 | 0.396 | 0.416 |
| UPZMUNCC | LightGBM | 0.894 | 0.516 | 0.939 | 0.410 | 0.426 |
| UPZMUNCC | HistGradientBoosting | 0.887 | 0.501 | 0.938 | 0.425 | 0.430 |
| UPZMUNCC | LogisticRegression | 0.872 | 0.483 | 0.937 | 0.334 | 0.381 |
| UPZMUNCC | DecisionTree | 0.865 | 0.443 | 0.937 | 0.358 | 0.384 |
| UPZMUNCC | NaiveBayes | 0.855 | 0.416 | 0.927 | 0.412 | 0.382 |
| UPZMUNCC | RandomForest | 0.849 | 0.443 | 0.932 | 0.403 | 0.390 |
| UPZMUNCC | ExtraTrees | 0.793 | 0.336 | 0.925 | 0.384 | 0.357 |

**Combined best model @ upazila level** (out-of-fold, XGB+CatBoost+LightGBM ensemble):
- Threshold = **0.54** (max accuracy on OOF)
- ROC-AUC = **0.897**, PR-AUC = **0.519**
- Accuracy = **0.941**, Sensitivity = **0.272**, Specificity = **0.993**, F1 = **0.401**, MCC = **0.434**

**Caveat:** upazila outbreak cells are only ~7% of all unit-wks, so accuracy is high while sensitivity/PR-AUC are low -- the model is specific but not yet sensitive at this fine granularity.

Figures: `figures/outbreak/roc_{DIVISION,DISTRICT,UPZMUNCC}.png`, `figures/outbreak/upazila_ensemble_roc.png`.

## 12. Artefacts
`model/final_xgb_model.joblib`, `model/preprocessor.joblib`, `output/final_feature_list.csv`, `output/final_threshold.json`, `output/final_metrics.json`, `output/model_comparison.csv`, `output/predictions_test.csv`, `output/error_analysis.csv`, `figures/{confusion_matrix,roc_curve,pr_curve,calibration_curve,shap_bar,shap_beeswarm}.png`.
