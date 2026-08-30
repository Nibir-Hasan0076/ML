"""Run the complete Early Measles Detection pipeline end-to-end.

Usage from the project root:
    python src/run_all.py
"""
import os
import subprocess
import sys

SRC = os.path.dirname(os.path.abspath(__file__))


def run(step, name):
    print("\n" + "#" * 70)
    print(f"# {name}")
    print("#" * 70)
    r = subprocess.run([sys.executable, os.path.join(SRC, step)],
                       capture_output=True, text=True)
    print(r.stdout)
    if r.returncode != 0:
        print("ERROR in", step)
        print(r.stderr)
        sys.exit(r.returncode)


if __name__ == "__main__":
    run("01_data_audit.py", "1. DATASET AUDIT")
    run("02_feature_selection.py", "2. FEATURE SELECTION & SUBSETS")
    run("03_model_comparison.py", "3. BASELINE MODEL COMPARISON")
    run("04_hyperparameter_tuning.py", "4. HYPERPARAMETER TUNING (Optuna)")
    run("05_threshold.py", "5. THRESHOLD OPTIMISATION")
    run("06_final_eval.py", "6. FINAL TEST EVALUATION")
    run("07_shap.py", "7. SHAP INTERPRETABILITY")
    run("08_error_analysis.py", "8. ERROR ANALYSIS")
    run("09_robustness.py", "9. ROBUSTNESS CHECK")
    run("11_ensemble.py", "11. IMPROVED MODEL - SIMPLE-AVERAGE ENSEMBLE")
    run("10_report.py", "10. FINAL REPORT & SUMMARY")
    print("\nALL STEPS COMPLETE. See output/final_report.md")
