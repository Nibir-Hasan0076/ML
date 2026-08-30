"""Split dataset into stratified 80% train / 20% test and save as CSV.

Works in both Windows and Google Colab.

Colab usage:
    from google.colab import drive
    drive.mount('/content/drive')
    import os
    os.environ['MEASLES_DATA'] = '/content/drive/MyDrive/measles/BAN_MR_FF_SEARO_EW-22_2026.xlsx'
    !python '/content/.../split_train_test.py'

Outputs (before feature engineering, so you can engineer on training only):
    measles_train.csv   (80%, labelled, stratified)
    measles_test.csv    (20%, labelled, stratified)

Only the cases with a definitive label (Laboratory Confirmed Measles, Discarded,
Laboratory Confirmed Rubella) are included. Pending cases (no outcome) are
excluded because they cannot provide a supervised label.
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from sklearn.model_selection import train_test_split

import pipeline as P


# Columns to carry forward so the two files are complete enough for later
# feature engineering. These are the raw, early-presentation + target columns.
CARRY = [
    # target (computed)
    "target",
    # demographics
    "SEX", "Ageyear", "DOB",
    # geography
    "DIVISION", "DISTRICT", "UPZMUNCC",
    # vaccination / exposure
    "DosesMCV", "DosesRCV", "DateLastMCV",
    "TravelHistory", "DateTravelFrom", "DateTravelTo",
    # clinical signs & timing (available at initial presentation)
    "CCC", "DOnsetF", "DOnsetR", "DNOT", "DOI",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=P.DATA_PATH)
    ap.add_argument("--out", default=".")
    ap.add_argument("--seed", type=int, default=P.SEED)
    ap.add_argument("--test-size", type=float, default=0.20)
    args = ap.parse_args()

    print("Loading:", args.data)
    df = P.load_raw()

    # Build the target exactly like the main pipeline
    label = df.dropna(subset=["CLASS"]).copy()
    label["target"] = P.build_target(label)
    # keep only definitive labels (drop Pending)
    label = label.dropna(subset=["target"]).copy()
    label["target"] = label["target"].astype(int)

    print(f"Labelled cases available: {len(label)}")
    print("Class distribution:")
    print(label["target"].value_counts().to_string(),
          f"  (positive rate {label['target'].mean():.3f})")
    print(f"Pending (excluded): {len(df) - len(label)}")

    C = [c for c in CARRY if c in label.columns]
    labeled = label[C].copy()

    # Stratified 80/20 train/test split (NaN 'target' already dropped)
    strat = labeled["target"]
    train, test = train_test_split(
        labeled, test_size=args.test_size, stratify=strat,
        random_state=args.seed)

    train_path = os.path.join(args.out, "measles_train.csv")
    test_path = os.path.join(args.out, "measles_test.csv")
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    print(f"\nSaved train -> {train_path}  ({len(train)} rows)")
    print(f"Saved test  -> {test_path}  ({len(test)} rows)")
    print(f"Train pos rate {train['target'].mean():.3f} | "
          f"Test pos rate {test['target'].mean():.3f}")


if __name__ == "__main__":
    main()
