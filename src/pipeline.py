"""Shared pipeline for the Early Measles Detection project.

Handles:
  - loading the raw Excel dataset
  - target construction (Measles-positive vs Not-Measles, excluding Pending)
  - leakage feature removal (lab results, final classification, etc.)
  - cleaning messy clinical values
  - clinically-sensible feature engineering
  - stratified train/val/test split
  - a library of metric helpers shared across steps
"""
import json
import os

import numpy as np
import pandas as pd

SEED = 42
SHEET = "Sheet1"

# ---------------------------------------------------------------------------
# Resolve the raw data path automatically so the same code runs in
# PyCharm (Windows) and Google Colab (Drive / content) without edits.
#
# Priority:
#   1. MEASLES_DATA environment variable if set and the file exists
#   2. candidates resolved relative to the current working directory
#   3. the original hardcoded Windows path as a last resort
# ---------------------------------------------------------------------------
_CANDIDATES = [
    "BAN_MR_FF_SEARO_EW-22_2026.xlsx",          # Colab: file directly in /content
    os.path.join("drive", "MyDrive", "archive",
                 "BAN_MR_FF_SEARO_EW-22_2026.xlsx"),
    os.path.join("archive", "BAN_MR_FF_SEARO_EW-22_2026.xlsx"),
    os.path.join(os.path.dirname(os.path.dirname(__file__)),
                 "BAN_MR_FF_SEARO_EW-22_2026.xlsx"),
    r"E:\archive\BAN_MR_FF_SEARO_EW-22_2026.xlsx",  # Windows fallback
]


def _resolve_data_path():
    env = os.environ.get("MEASLES_DATA")
    if env and os.path.exists(env):
        return env
    for cand in _CANDIDATES:
        if os.path.exists(cand):
            return cand
    # No candidate found -> return the env value (or the Windows path) so
    # pandas raises a clear, correct error instead of a hardcoded one.
    return env if env else _CANDIDATES[-1]


DATA_PATH = _resolve_data_path()

OUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
FIG = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")
MODEL = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
for d in (OUT, FIG, MODEL):
    os.makedirs(d, exist_ok=True)

MEASLES_CLASS = "2-Laboratory Confirmed Measles"
NON_MEASLES_CLASSES = ["6-Discarded", "4-Laboratory Confirmed Rubella"]
PENDING_CLASSES = ["7-Pending"]

# ---------------------------------------------------------------------------
# Leakage / non-early-presentation columns -> removed as predictors
# ---------------------------------------------------------------------------
LEAKAGE_COLUMNS = [
    # Lab serology (MeaslesIgM is effectively the target; RubellaIgM and all lab
    # specimen/result metadata only exist after a specimen is collected)
    "MeaslesIgM", "RubellaIgM",
    "DateSpecSero", "DateSeroSent", "DateSeroRec", "SpecIDSero",
    "DateMeaIgMResult", "DateRubIgMResult",
    # Urine viral detection / genotyping
    "DateSpecUrine", "DateUrineSent", "DateUrineRec", "SpecIDUrine",
    "MeaVirDetectUrine", "GenotypeMeaUrine", "DateMeaGenoResultUrine",
    "RubVirDetectUrine", "GenotypeRubUrine", "DateRubGenoResultUrine",
    # Throat swab viral detection / genotyping
    "DateSpecSwab", "DateSwabSent", "DateSwabRec", "SpecIDSwab",
    "MeaVirDetectSwab", "GenotypeMeaSwab", "DateMeaGenoResultSwab",
    "RubVirDetectSwab", "GenotypeRubSwab", "DateRubGenoResultSwab",
    # Final classification = the target itself (never a predictor)
    "CLASS", "ClassforAnalysis",
    # Post-diagnosis / administrative-investigator text
    "Comment", "CASE_INV", "CASE_INV_DESIG",
    # PII / identifiers (not legitimate predictors in early detection)
    "NAME", "F_NAME",
]

LABEL_COLS = ["CLASS", "MeaslesIgM"]


def load_raw():
    df = pd.read_excel(DATA_PATH, sheet_name=SHEET, dtype={})
    df.columns = [str(c).strip() for c in df.columns]
    return df


def build_target(df):
    """Map the final measles classification to a binary target.

    Positive = Laboratory Confirmed Measles
    Negative = Discarded or Laboratory Confirmed Rubella (both rule out measles)
    Pending  = no outcome -> EXCLUDED from supervised learning.
    """
    t = pd.Series(np.nan, index=df.index, dtype=float)
    t[df["CLASS"].isin([MEASLES_CLASS])] = 1
    t[df["CLASS"].isin(NON_MEASLES_CLASSES)] = 0
    return t


# ---------------------------------------------------------------------------
# Clinical / demographic categorical columns kept as predictors (early
# presentation). Normalisation of messy labels is done here.
# ---------------------------------------------------------------------------

def _norm_division(s):
    return s.astype(str).str.strip().str.upper()


def _norm_sex(s):
    s = s.astype(str).str.strip().str.upper()
    s = s.replace({"M": "MALE", "F": "FEMALE"})
    return s.map(lambda x: x if x in ("MALE", "FEMALE") else "UNKNOWN")


def _norm_ccc(s):
    s = s.astype(str).str.strip()
    s = s.replace({"1-Yes": "YES", "2-No": "NO", "9-Unknown": "UNKNOWN"})
    return s.map(lambda x: x if x in ("YES", "NO", "UNKNOWN") else "UNKNOWN")


def _norm_travel(s):
    s = s.astype(str).str.strip().str.upper()
    s = s.map({"1-YES": "YES", "2-NO": "NO", "9-UNKNOWN": "UNKNOWN"})
    return s.fillna("NO").map(lambda x: x if x in ("YES", "NO", "UNKNOWN") else "NO")


def clean_doses(s):
    """Doses cannot exceed a sane bound (>=1 dose per year of life, cap at 2
    for both MCV/RCV since routine schedule is 2 doses). Values >2 are
    implausible surveillance-entry errors -> treat as 2 (documented)."""
    s = pd.to_numeric(pd.Series(s), errors="coerce")
    s = s.clip(lower=0, upper=2)
    return s.fillna(0).astype(int)


def clean_age(Ageyear, DOB):
    """Age in years from Ageyear, validated against DOB when present.

    Ageyear contains impossible values (>1000) and negatives. When Ageyear is
    clearly pathological we fall back to DOB-derived age; extreme outliers are
    winsorised at 90 years (a reasonable upper bound for clinical data)."""
    age = pd.to_numeric(pd.Series(Ageyear), errors="coerce")
    dob = pd.to_datetime(DOB, errors="coerce")
    # DoB is ~62% missing. Reference date approximates the reporting window.
    ref = pd.Timestamp("2026-05-29")
    dob_age = (ref - dob).dt.days / 365.25
    # Use Ageyear normally; where Ageyear is impossible, use DOB-derived age
    valid_ageyear = age.between(0, 100)
    age = age.where(valid_ageyear, dob_age)
    # Where still NaN (no DOB and bad Ageyear), fall back to overall median
    med = age.median()
    age = age.fillna(med)
    # Winsorise to a clinically plausible range
    age = age.clip(lower=0, upper=90)
    return age.values.astype(float)


def feature_engineer(df):
    """Return a clean DataFrame of early-presentation features + the target.

    Everything here is available at initial clinical presentation.
    """
    df = df.copy()

    # ---- Timing features (administrative & clinical, available at intake) --
    DOnsetF = pd.to_datetime(df["DOnsetF"], errors="coerce")
    DOnsetR = pd.to_datetime(df["DOnsetR"], errors="coerce")
    DNOT = pd.to_datetime(df["DNOT"], errors="coerce")
    DOI = pd.to_datetime(df["DOI"], errors="coerce")

    fever_dur = (DNOT - DOnsetF).dt.days
    rash_dur = (DNOT - DOnsetR).dt.days
    invest_lag = (DOI - DNOT).dt.days

    # ---- Vaccination history ----
    DosesMCV = clean_doses(df["DosesMCV"])
    DosesRCV = clean_doses(df["DosesRCV"])
    DateLastMCV = pd.to_datetime(df["DateLastMCV"], errors="coerce")
    # time since last MCV (years) available at presentation if a dose was recorded
    time_since_mcv = (DNOT - DateLastMCV).dt.days / 365.25
    time_since_mcv = time_since_mcv.where(time_since_mcv.between(0, 60))

    # ---- Cutaneous / mucosal presentation ----
    # (fever & rash onset are present for all classified suspected cases, but we
    # keep the *duration* which carries discriminating signal)
    ccc = _norm_ccc(df["CCC"])

    feats = pd.DataFrame({
        "age": clean_age(df["Ageyear"], df["DOB"]),
        "sex": _norm_sex(df["SEX"]),
        "division": _norm_division(df["DIVISION"]),
        "district": df["DISTRICT"].astype(str).str.strip(),
        "upzmunc": df["UPZMUNCC"].astype(str).str.strip(),
        "ccc": ccc,
        "travel": _norm_travel(df["TravelHistory"]),
        # vaccination (doses known at intake from records / mother's report)
        "doses_mcv": DosesMCV,
        "doses_rcv": DosesRCV,
        "time_since_mcv": time_since_mcv.values,
        # clinical timing
        "fever_duration": fever_dur.values,
        "rash_duration": rash_dur.values,
        "invest_lag": invest_lag.values,
        # notification periodicity
        "epiweek": DNOT.dt.isocalendar().week.astype(float).values,
    })

    # Combined vaccination status (any measles vaccine recorded)
    feats["vax_status"] = np.select(
        [DosesMCV >= 1, DosesMCV == 0],
        ["VACCINATED", "UNVACCINATED"],
        default="UNKNOWN",
    )

    # ---- Engineered clinical composites (early-presentation) ----
    feats["age_group"] = np.select(
        [feats["age"] < 1,
         feats["age"].between(1, 9),
         feats["age"].between(10, 17),
         feats["age"] >= 18],
        ["infant", "child", "adolescent", "adult"],
        default="child",
    )
    feats["fever_gte7"] = (feats["fever_duration"] >= 7).astype(int)
    feats["rash_gte3"] = (feats["rash_duration"] >= 3).astype(int)
    feats["ccc_yes"] = (feats["ccc"] == "YES").astype(int)
    # Clinical combinations reflecting the measles prodrome
    feats["fever_cough_coryza_conj"] = feats["ccc_yes"] * feats["fever_gte7"]
    feats["fever_and_rash"] = feats["fever_gte7"] * feats["rash_gte3"]

    # target
    feats["target"] = build_target(df)

    return feats


def get_feature_columns():
    """Names of all predictive (non-target) columns, in a stable order."""
    return [
        "age", "sex", "division", "district", "upzmunc", "ccc", "travel",
        "doses_mcv", "doses_rcv", "time_since_mcv",
        "fever_duration", "rash_duration", "invest_lag", "epiweek",
        "vax_status", "age_group",
        "fever_gte7", "rash_gte3", "ccc_yes",
        "fever_cough_coryza_conj", "fever_and_rash",
    ]


def load_chosen_subset(name="Model_C_top15"):
    """Read a saved feature-subset column list."""
    path = os.path.join(OUT, f"subset_{name}.csv")
    cols = pd.read_csv(path, header=None)[0].tolist()
    return [c for c in cols if c in get_feature_columns()]


def make_split(df):
    """Stratified 70 / 15 / 15 split on label-presence rows, fixed seed."""
    from sklearn.model_selection import train_test_split

    labeled = df.dropna(subset=["target"]).copy()
    y = labeled["target"].astype(int)
    Xy = labeled

    train, temp = train_test_split(
        Xy, test_size=0.30, stratify=y, random_state=SEED)
    y_train = train["target"].astype(int)
    val, test = train_test_split(
        temp, test_size=0.50, stratify=temp["target"].astype(int),
        random_state=SEED)

    return train, val, test


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------
def metrics_from_cm(cm):
    tn, fp, fn, tp = cm.ravel()
    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    sens = tp / max(1, (tp + fn))
    spec = tn / max(1, (tn + fp))
    prec = tp / max(1, (tp + fp))
    npv = tn / max(1, (tn + fn))
    f1 = 2 * prec * sens / max(1e-9, (prec + sens))
    bal_acc = (sens + spec) / 2
    denom = max(1e-9, ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5)
    mcc = (tp * tn - fp * fn) / denom
    fpr = fp / max(1, (fp + tn))
    fnr = fn / max(1, (fn + tp))
    return {
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "accuracy": acc, "sensitivity": sens, "specificity": spec,
        "precision": prec, "NPV": npv, "F1": f1,
        "balanced_accuracy": bal_acc, "MCC": mcc,
        "FPR": fpr, "FNR": fnr,
    }


def full_metrics(y_true, y_prob, threshold=0.5):
    from sklearn.metrics import roc_auc_score, average_precision_score
    y_pred = (y_prob >= threshold).astype(int)
    cm = np.array([[np.sum((y_true == 0) & (y_pred == 0)),
                    np.sum((y_true == 0) & (y_pred == 1))],
                   [np.sum((y_true == 1) & (y_pred == 0)),
                    np.sum((y_true == 1) & (y_pred == 1))]])
    m = metrics_from_cm(cm)
    m["ROC_AUC"] = roc_auc_score(y_true, y_prob)
    m["PR_AUC"] = average_precision_score(y_true, y_prob)
    return m


def save_json(obj, name):
    path = os.path.join(OUT, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=float)
    return path
