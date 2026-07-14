"""
STEP 1 — DATA AUDIT
Reusable functions: zero/missing quantification, missingness-mechanism testing,
outlier flagging, duplicate check, class balance, correlation matrix.
"""
import pandas as pd
import numpy as np
from scipy import stats

ZERO_INVALID_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

OUTLIER_BOUNDS = {
    "BMI": (12, 60),
    "Age": (18, 90),
    "Pregnancies": (0, 15),
    "Glucose": (40, 300),
    "BloodPressure": (30, 140),
}


def load_raw(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def encode_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Replace physiologically-impossible zeros with NaN. Does NOT touch Pregnancies (0 is valid)."""
    out = df.copy()
    for col in ZERO_INVALID_COLS:
        out.loc[out[col] == 0, col] = np.nan
    return out


def zero_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ZERO_INVALID_COLS:
        n_zero = int((df[col] == 0).sum())
        rows.append({
            "Feature": col,
            "N Zero (=missing)": n_zero,
            "% of Dataset": round(100 * n_zero / len(df), 2),
        })
    return pd.DataFrame(rows).sort_values("% of Dataset", ascending=False).reset_index(drop=True)


def missingness_mechanism_report(df_clean: pd.DataFrame) -> pd.DataFrame:
    """
    For each column with missingness, test whether the missingness INDICATOR
    is significantly associated with Outcome (MNAR-leaning signal) or with
    other observed features (MAR-consistent signal).
    """
    rows = []
    for col in ZERO_INVALID_COLS:
        n_missing = int(df_clean[col].isna().sum())
        if n_missing == 0:
            continue
        indicator = df_clean[col].isna().astype(int)
        r_outcome, p_outcome = stats.pointbiserialr(indicator, df_clean["Outcome"])
        r_age, p_age = stats.pointbiserialr(indicator, df_clean["Age"])
        r_preg, p_preg = stats.pointbiserialr(indicator, df_clean["Pregnancies"])

        mechanism = (
            "MNAR-leaning (missingness predicts Outcome directly)"
            if p_outcome < 0.05
            else "MAR-consistent (no direct Outcome link; may relate to observed covariates)"
        )

        rows.append({
            "Feature": col,
            "% Missing": round(100 * n_missing / len(df_clean), 2),
            "corr(missing, Outcome)": round(r_outcome, 3),
            "p-value (Outcome)": round(p_outcome, 4),
            "corr(missing, Age)": round(r_age, 3),
            "p-value (Age)": round(p_age, 4),
            "corr(missing, Pregnancies)": round(r_preg, 3),
            "p-value (Pregnancies)": round(p_preg, 4),
            "Likely mechanism": mechanism,
        })
    return pd.DataFrame(rows)


def outlier_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, (low, high) in OUTLIER_BOUNDS.items():
        n_out = int(((df[col] < low) | (df[col] > high)).sum())
        rows.append({
            "Feature": col,
            "Plausible range": f"{low}–{high}",
            "N flagged outside range": n_out,
        })
    return pd.DataFrame(rows)


def duplicate_count(df: pd.DataFrame) -> int:
    return int(df.duplicated().sum())


def class_balance(df: pd.DataFrame) -> pd.DataFrame:
    vc = df["Outcome"].value_counts(normalize=True).round(4) * 100
    return pd.DataFrame({
        "Outcome": ["0 (No Diabetes)", "1 (Diabetes)"],
        "Percentage": [vc.get(0, 0), vc.get(1, 0)],
        "Count": [int((df["Outcome"] == 0).sum()), int((df["Outcome"] == 1).sum())],
    })


def correlation_matrix(df_clean: pd.DataFrame) -> pd.DataFrame:
    return df_clean.drop(columns=["Outcome"]).corr(method="pearson").round(3)


def run_full_audit(path: str):
    """Convenience wrapper returning all audit artifacts at once."""
    raw = load_raw(path)
    clean = encode_missing(raw)
    return {
        "raw": raw,
        "clean_nan_encoded": clean,
        "zero_report": zero_report(raw),
        "missingness_mechanism": missingness_mechanism_report(clean),
        "outliers": outlier_report(raw),
        "duplicates": duplicate_count(raw),
        "class_balance": class_balance(raw),
        "correlation": correlation_matrix(clean),
    }


if __name__ == "__main__":
    results = run_full_audit("data/pima_diabetes.csv")
    print("Zero report:\n", results["zero_report"], "\n")
    print("Missingness mechanism:\n", results["missingness_mechanism"], "\n")
    print("Outliers:\n", results["outliers"], "\n")
    print("Duplicates:", results["duplicates"], "\n")
    print("Class balance:\n", results["class_balance"], "\n")
    print("Correlation matrix:\n", results["correlation"])
