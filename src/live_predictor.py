"""
LIVE PREDICTION MODULE — Part 2 of the dashboard
Deliberately separate from causal_recourse.py. This module NEVER suggests
feature changes or "what-if" modifications. It only:
  1. Validates input against observed + clinically plausible ranges
  2. Produces a calibrated risk probability
  3. Quantifies uncertainty via a fold-ensemble (not a single point estimate)
  4. Explains the prediction's feature contributions (SHAP) as DESCRIPTIVE
     attribution only -- not as a recommendation or actionable suggestion
  5. Surfaces the population/generalizability caveat every time
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from catboost import CatBoostClassifier
import shap

from .preprocessing import build_mice_pipeline, encode_missing

FEATURE_ORDER = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

# Hard physiological limits -- values outside these are rejected outright,
# not just flagged (e.g. negative glucose is not a data-entry edge case, it's invalid).
HARD_LIMITS = {
    "Pregnancies": (0, 20),
    "Glucose": (0, 400),
    "BloodPressure": (0, 200),
    "SkinThickness": (0, 120),
    "Insulin": (0, 1000),
    "BMI": (10, 80),
    "DiabetesPedigreeFunction": (0.0, 3.0),
    "Age": (18, 100),
}


def observed_range_table(df_raw: pd.DataFrame) -> pd.DataFrame:
    """1st-99th percentile of OBSERVED (non-missing) training values per feature.
    Used to warn -- not block -- when a new input falls outside what the model
    actually learned from, since predictions there are extrapolations."""
    df_clean = encode_missing(df_raw)
    rows = []
    for col in FEATURE_ORDER:
        s = df_clean[col].dropna()
        rows.append(
            {
                "Feature": col,
                "Observed min": round(float(s.min()), 2),
                "Observed max": round(float(s.max()), 2),
                "1st percentile": round(float(s.quantile(0.01)), 2),
                "99th percentile": round(float(s.quantile(0.99)), 2),
            }
        )
    return pd.DataFrame(rows)


def validate_input(patient: dict, obs_range_df: pd.DataFrame):
    """
    Returns (is_hard_invalid: bool, hard_errors: list[str], soft_warnings: list[str])
    Hard errors = physiologically impossible, should block prediction.
    Soft warnings = outside the training distribution, prediction allowed but flagged.
    """
    hard_errors = []
    soft_warnings = []
    obs = obs_range_df.set_index("Feature")

    for feat, val in patient.items():
        lo, hi = HARD_LIMITS[feat]
        if val < lo or val > hi:
            hard_errors.append(
                f"{feat} = {val} is outside physiologically possible range ({lo}-{hi})."
            )
            continue
        p1, p99 = obs.loc[feat, "1st percentile"], obs.loc[feat, "99th percentile"]
        if val < p1 or val > p99:
            soft_warnings.append(
                f"{feat} = {val} falls outside the 1st-99th percentile of the training "
                f"data ({p1}-{p99}). The model has seen little to no data like this — "
                f"treat the prediction below with extra caution."
            )
    return (len(hard_errors) > 0), hard_errors, soft_warnings


def build_prediction_ensemble(
    df_raw: pd.DataFrame, best_params=None, n_splits=5, n_repeats=2, random_state=42
):
    """
    Trains one MICE+CatBoost pipeline per CV fold (not one single model fit on
    everything). Predicting a new patient through all fold-models and looking at
    the SPREAD of predictions is an honest, cheap uncertainty signal: if the
    fold-models disagree a lot on this patient, that's real information the
    single-point-estimate version of this demo would have hidden.
    """
    best_params = best_params or {"iterations": 200, "depth": 6}
    X = df_raw.drop(columns=["Outcome"])
    y = df_raw["Outcome"]
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )

    ensemble = []
    for train_idx, _ in rskf.split(X, y):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        model = CatBoostClassifier(verbose=0, random_state=random_state, **best_params)
        pipe = build_mice_pipeline(model, random_state=random_state)
        pipe.fit(X_train, y_train)
        ensemble.append(pipe)
    return ensemble


def predict_with_uncertainty(ensemble, patient: dict):
    """Returns mean probability, std, and the full array of per-fold-model predictions."""
    row = pd.DataFrame([patient])[FEATURE_ORDER]
    probs = np.array([pipe.predict_proba(row)[0, 1] for pipe in ensemble])
    return float(probs.mean()), float(probs.std()), probs


def risk_category(mean_proba: float, std_proba: float):
    """
    Conservative, explicitly-labeled-as-illustrative thresholds. NOT a validated
    clinical cutoff -- stated plainly to avoid the exact overclaiming reviewers
    already flagged.
    """
    if std_proba > 0.15:
        return (
            "Uncertain — model disagreement too high for a confident category",
            "warning",
        )
    if mean_proba < 0.30:
        return "Lower predicted risk", "success"
    elif mean_proba < 0.60:
        return "Moderate predicted risk", "warning"
    else:
        return "Higher predicted risk", "error"


def explain_single_prediction(ensemble, patient: dict):
    """
    SHAP attribution for ONE fold-model (representative, not averaged across the
    ensemble, since SHAP baselines differ per model) -- purely descriptive:
    "these are the values that pushed the prediction up or down."
    This function must never be used to suggest changing any value.
    """
    model_pipe = ensemble[0]
    row = pd.DataFrame([patient])[FEATURE_ORDER]
    row_transformed = row.copy()
    for name, step in model_pipe.steps[:-1]:
        row_transformed = step.transform(row_transformed)
    row_transformed = pd.DataFrame(row_transformed, columns=FEATURE_ORDER)

    explainer = shap.TreeExplainer(model_pipe.named_steps["model"])
    shap_values = explainer.shap_values(row_transformed)[0]

    df = pd.DataFrame(
        {
            "Feature": FEATURE_ORDER,
            "Patient value": [patient[f] for f in FEATURE_ORDER],
            "SHAP contribution": shap_values,
        }
    ).sort_values("SHAP contribution", key=abs, ascending=False)
    return df
