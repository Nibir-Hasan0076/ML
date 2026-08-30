"""Step 1 - Dataset audit: dump structure, missingness, duplicates, target."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 250)

import pipeline as P


def main():
    df = P.load_raw()
    print("=" * 70)
    print("STEP 1: DATASET AUDIT")
    print("=" * 70)

    print("\nDATASET SHAPE         :", df.shape)
    print("ROWS (case records)    :", df.shape[0])
    print("COLUMNS                :", df.shape[1])
    print("\nCOLUMN NAMES (all):")
    for c in df.columns:
        print("   -", c)

    print("\n--- DATA TYPES ---")
    print(df.dtypes.to_string())

    print("\n--- MISSING VALUES (by %) ---")
    miss = (df.isna().mean() * 100).sort_values(ascending=False)
    print(miss.round(1).to_string())

    print("\n--- DUPLICATED ROWS ---")
    print("Exact duplicate rows:", df.duplicated().sum())

    print("\n--- TARGET IDENTIFICATION ---")
    print("Final classification column 'CLASS' value counts:")
    print(df["CLASS"].value_counts(dropna=False).to_string())
    print("\n'ClassforAnalysis' (mirror of CLASS):")
    print(df["ClassforAnalysis"].value_counts(dropna=False).to_string())
    print("\nLab 'MeaslesIgM' (used to confirm):")
    print(df["MeaslesIgM"].value_counts(dropna=False).to_string())

    labeled = df[df["CLASS"].isin([P.MEASLES_CLASS] + P.NON_MEASLES_CLASSES)]
    print("\n--- TARGET CONSTRUCTION ---")
    print("Measles-positive (Lab Confirmed Measles):",
          (labeled["CLASS"] == P.MEASLES_CLASS).sum())
    print("Not-Measles (Discarded + Rubella):",
          labeled["CLASS"].isin(P.NON_MEASLES_CLASSES).sum())
    print("Pending (no outcome, EXCLUDED from supervised training):",
          df["CLASS"].isin(P.PENDING_CLASSES).sum())

    print("\n--- UNIQUE VALUES / CATEGORICAL SUMMARY ---")
    for c in ["SEX", "CCC", "TravelHistory", "DosesMCV", "DosesRCV", "DIVISION"]:
        print(f"\n[{c}]")
        print(df[c].value_counts(dropna=False).to_string())

    print("\n--- NUMERICAL SUMMARY (Ageyear, fevers/doses) ---")
    print(df["Ageyear"].describe().to_string())
    print("\nImplausible ages: Ageyear>100:", (df["Ageyear"] > 100).sum(),
          "| Ageyear<0:", (df["Ageyear"] < 0).sum())

    print("\n--- LEAKAGE / NON-EARLY-PRESENTATION COLUMN DECISION TABLE ---")
    from pipeline import LEAKAGE_COLUMNS
    table_rows = []
    for c in df.columns:
        if c in P.LABEL_COLS:
            table_rows.append((c, "Remove", "Final outcome / lab target derivative - is the label", "No"))
        elif c in LEAKAGE_COLUMNS:
            table_rows.append((c, "Remove", "Lab result / post-diagnosis information not available at initial presentation", "No"))
        elif c in ["YEAR", "DNOT", "DOI", "DOnsetF", "DOnsetR", "DateLastMCV", "DateLastRCV"]:
            table_rows.append((c, "Keep (derived)", "Timing/vaccination info available at intake; used to build safe features", "Yes"))
        elif c in ["DIVISION", "DISTRICT", "UPZMUNCC", "SEX", "DOB", "Ageyear",
                   "DosesMCV", "DosesRCV", "TravelHistory", "DateTravelFrom",
                   "DateTravelTo", "TravelAddress", "CCC"]:
            table_rows.append((c, "Keep (raw)", "Demographics / geography / symptoms / exposure at presentation", "Yes"))
        elif c in ["VILL_MAHAL", "UNION_WARD", "FACILITY"]:
            table_rows.append((c, "Remove (adjacent)", "High-cardinality location identifiers - minimal clinical signal, near-constant per case", "Yes"))
        else:
            table_rows.append((c, "Remove", "Unused / duplicate / identifier", "n/a"))

    print(f"{'Feature':34}{'Keep/Remove':14}{'Reason':70}")
    print("-" * 120)
    for f, kr, why, avail in table_rows:
        print(f"{f:34}{kr:14}{why:70}")


if __name__ == "__main__":
    main()
