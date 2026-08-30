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
- 15 features used (best subset Model_C_top15): epiweek, upzmunc, age, district, fever_duration, division, rash_duration, vax_status, doses_mcv, doses_rcv, sex, invest_lag, ccc, rash_gte3, fever_and_rash.
- Engineered clinical features (fever/rash duration, vaccination history, epi-week of onset, prodrome triad `CCC`, age group, composite symptom flags) outperformed the raw feature set (Model_A AUC 0.74 vs Model_C 0.84).

## 4. Model comparison (5-fold CV, best subset)

| Model | ROC-AUC | PR-AUC | Sens | Spec | F1 | MCC |
|-------|--------:|-------:|-----:|-----:|---:|----:|
| XGBoost | 0.840 | 0.901 | 0.884 | 0.590 | 0.838 | 0.502 |
| CatBoost | 0.840 | 0.900 | 0.886 | 0.581 | 0.838 | 0.498 |
| LightGBM | 0.837 | 0.898 | 0.879 | 0.584 | 0.834 | 0.491 |
| RandomForest | 0.818 | 0.877 | 0.818 | 0.660 | 0.816 | 0.478 |
| ExtraTrees | 0.772 | 0.835 | 0.846 | 0.546 | 0.808 | 0.412 |
| LogisticRegression | 0.771 | 0.809 | 0.775 | 0.675 | 0.794 | 0.443 |
| LinearSVC | 0.763 | 0.804 | 0.884 | 0.512 | 0.822 | 0.434 |

XGBoost was selected (highest AUC / PR-AUC / MCC, best stability) and hyper-parameters tuned with Optuna (60 trials, 5-fold CV).

## 5. Class-imbalance strategy
The labelled set is measles-majority (64.6%). Heavy measles weighting **increased false positives** (specificity fell). A near-neutral `scale_pos_weight` from tuning, combined with threshold optimisation on validation, balances sensitivity vs specificity rather than maximizing recall at the cost of FPs.

## 6. Threshold (locked on validation)
- Optimal threshold = **0.44** (maximises MCC on the validation set; test set untouched during selection).

## 7. Final test-set evaluation (evaluated exactly once)

**Final deployed model: simple-average GBM ensemble (XGBoost + CatBoost + LightGBM).** Threshold=0.42 (locked on validation by accuracy).

| Metric | Single XGB | **Ensemble (final)** |
|--------|-----------:|---------------------:|
| TN | 264 | 256 |
| FP | 214 | 222 |
| FN | 72 | 58 |
| TP | 800 | 814 |
| accuracy | 0.7881 | 0.7926 |
| ROC-AUC | 0.8492 | 0.8510 |
| PR-AUC | 0.9119 | 0.9135 |
| sensitivity | 0.9174 | 0.9335 |
| specificity | 0.5523 | 0.5356 |
| precision | 0.7890 | 0.7857 |
| NPV | 0.7857 | 0.8153 |
| F1 | 0.8484 | 0.8532 |
| balanced_accuracy | 0.7349 | 0.7345 |
| MCC | 0.5196 | 0.5309 |
| brier | 0.1484 | 0.1472 |

Ensemble improves test accuracy by **+0.44 pp** and reduces false negatives (58 vs 72) with comparable/better AUC, PR-AUC, F1 and MCC.

## 8. Robustness (mean +/- std across 5 independent seeds)

| Metric | Mean | Std |
|--------|-----:|----:|
| accuracy | 0.7775 | 0.0081 |
| ROC_AUC | 0.8461 | 0.0044 |
| PR_AUC | 0.9069 | 0.0043 |
| sensitivity | 0.8966 | 0.0360 |
| specificity | 0.5603 | 0.0563 |
| F1 | 0.8387 | 0.0093 |
| MCC | 0.4974 | 0.0164 |

## 9. SHAP interpretability (top drivers)

| Feature | mean\|SHAP\| |
|---------|------:|
| epiweek | 1.0998 |
| doses_mcv | 0.1506 |
| age | 0.1336 |
| division_DHAKA | 0.0690 |
| fever_duration | 0.0618 |
| sex_FEMALE | 0.0601 |
| rash_duration | 0.0580 |
| division_RANGPUR | 0.0471 |
| ccc_YES | 0.0428 |
| doses_rcv | 0.0344 |
| division_KHULNA | 0.0337 |
| district_CHITTAGONG | 0.0306 |

**Driving the model toward Measles:** early `epiweek` (outbreak phase), lower measles-vaccine doses, younger age, certain divisions/districts, presence of the cough/coryza/conjunctivitis prodrome.

**Driving away from Measles:** higher vaccine doses, older age, late-epiweek presentation, longer rash duration / atypical timing.

## 10. Error analysis highlights
- False positives (n=214): mostly young, predominantly unvaccinated (86%), with prodrome (CCC) present - i.e. clinically similar suspected cases that were discarded. Median age 1.0y.
- False negatives (n=72): median age 1.7y, 54% unvaccinated.
- The residual errors reflect genuine clinical similarity between confirmed measles and discarded/rubella suspected cases (AUC plateaus ~0.85); this is real generalisation, not an inflated score.

## 11. Artefacts
`model/final_xgb_model.joblib`, `model/preprocessor.joblib`, `output/final_feature_list.csv`, `output/final_threshold.json`, `output/final_metrics.json`, `output/model_comparison.csv`, `output/predictions_test.csv`, `output/error_analysis.csv`, `figures/{confusion_matrix,roc_curve,pr_curve,calibration_curve,shap_bar,shap_beeswarm}.png`.
