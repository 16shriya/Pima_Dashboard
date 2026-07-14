"""
STEP 3 — LEAKAGE-SAFE PREPROCESSING
Builds a sklearn Pipeline that fits MICE (IterativeImputer) and StandardScaler
INSIDE each CV fold only on training data, then transforms validation data.
This is what prevents the data-leakage error common in naive Pima pipelines.
"""
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa: F401 (required to unlock IterativeImputer)
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

ZERO_INVALID_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def encode_missing(X):
    """Replace 0 with NaN for the 5 physiologically-invalid columns. Expects a DataFrame."""
    X = X.copy()
    for col in ZERO_INVALID_COLS:
        if col in X.columns:
            X.loc[X[col] == 0, col] = np.nan
    return X


def build_mice_pipeline(model, random_state=42):
    """
    Returns a Pipeline: MICE imputation -> StandardScaler -> model.
    Fit this ONLY on the training fold; call .transform via the same pipeline
    on the validation fold (sklearn Pipeline handles this correctly by design).
    """
    mice = IterativeImputer(
        estimator=BayesianRidge(),
        max_iter=15,
        random_state=random_state,
        sample_posterior=False,
    )
    pipeline = Pipeline([
        ("mice_imputer", mice),
        ("scaler", StandardScaler()),
        ("model", model),
    ])
    return pipeline


def build_median_pipeline(model):
    """Baseline comparator: median imputation (what the original paper used)."""
    pipeline = Pipeline([
        ("median_imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model),
    ])
    return pipeline
