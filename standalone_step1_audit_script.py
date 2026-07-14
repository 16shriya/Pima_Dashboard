"""
STEP 1 — DATA AUDIT (run before any EDA/modeling)
Produces the exact numbers referenced in the playbook:
  1. Zero/missing quantification per column
  2. Missingness mechanism check (is missing-ness itself predictive?)
  3. Outlier / impossible value flags
  4. Duplicate row check
  5. Class balance
  6. Correlation matrix (computed on NaN-safe data)
"""
import pandas as pd
import numpy as np
from scipy import stats
import json

df = pd.read_csv("pima_diabetes.csv")

ZERO_INVALID_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

audit = {}

# ---------- 1. Zero / missingness quantification ----------
zero_report = {}
for col in ZERO_INVALID_COLS:
    n_zero = int((df[col] == 0).sum())
    pct = round(100 * n_zero / len(df), 2)
    zero_report[col] = {"n_zero": n_zero, "pct_zero": pct}
audit["zero_report"] = zero_report

# Replace invalid zeros with NaN for everything downstream
df_clean = df.copy()
for col in ZERO_INVALID_COLS:
    df_clean.loc[df_clean[col] == 0, col] = np.nan

# ---------- 2. Missingness mechanism (MAR vs MNAR signal) ----------
# For each column with missingness, create indicator and test association
# with Outcome (logistic-style via t-test on Outcome) and with other observed features.
missingness_mechanism = {}
for col in ZERO_INVALID_COLS:
    n_missing = df_clean[col].isna().sum()
    if n_missing == 0:
        continue
    indicator = df_clean[col].isna().astype(int)

    # Association with Outcome (is missingness itself predictive of the label?)
    grp0 = indicator[df_clean["Outcome"] == 0]
    grp1 = indicator[df_clean["Outcome"] == 1]
    # point-biserial correlation between missing-indicator and Outcome
    r_outcome, p_outcome = stats.pointbiserialr(indicator, df_clean["Outcome"])

    # Association with other observed features (Age, BMI if not this col, Pregnancies)
    assoc_with_features = {}
    for other in ["Age", "Pregnancies"]:
        r, p = stats.pointbiserialr(indicator, df_clean[other])
        assoc_with_features[other] = {"r": round(r, 3), "p": round(p, 4)}

    missingness_mechanism[col] = {
        "n_missing": int(n_missing),
        "pct_missing": round(100 * n_missing / len(df), 2),
        "corr_with_outcome_r": round(r_outcome, 3),
        "corr_with_outcome_p": round(p_outcome, 4),
        "assoc_with_observed_features": assoc_with_features,
        "likely_mechanism": (
            "MNAR-leaning (missingness itself predicts Outcome)"
            if p_outcome < 0.05
            else "MAR-consistent (no significant direct link to Outcome)"
        ),
    }
audit["missingness_mechanism"] = missingness_mechanism

# ---------- 3. Outlier / impossible value flags ----------
outlier_flags = {}
bounds = {
    "BMI": (12, 60),
    "Age": (18, 90),
    "Pregnancies": (0, 15),
    "Glucose": (40, 300),
    "BloodPressure": (30, 140),
}
for col, (low, high) in bounds.items():
    n_out = int(((df[col] < low) | (df[col] > high)).sum())
    outlier_flags[col] = {"bounds": [low, high], "n_flagged": n_out}
audit["outlier_flags"] = outlier_flags

# ---------- 4. Duplicate rows ----------
audit["duplicate_rows"] = int(df.duplicated().sum())

# ---------- 5. Class balance ----------
vc = df["Outcome"].value_counts(normalize=True).round(4)
audit["class_balance"] = {"Outcome=0": float(vc.get(0, 0)), "Outcome=1": float(vc.get(1, 0))}

# ---------- 6. Correlation matrix (on cleaned/NaN data, pairwise complete) ----------
corr = df_clean.drop(columns=["Outcome"]).corr(method="pearson").round(3)
audit["correlation_matrix"] = corr.to_dict()

# Save results
with open("audit_results.json", "w") as f:
    json.dump(audit, f, indent=2)

df_clean.to_csv("pima_diabetes_nan_encoded.csv", index=False)

print(json.dumps(audit, indent=2))
