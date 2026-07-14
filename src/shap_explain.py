"""
STEP 5 — SHAP EXPLAINABILITY
Fits a final CatBoost model on MICE-imputed full data, computes SHAP values
for global (summary) and local (single-patient waterfall) interpretation.
"""
import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier
from .preprocessing import build_mice_pipeline


def fit_final_model(X, y, params=None, random_state=42):
    params = params or {}
    model = CatBoostClassifier(verbose=0, random_state=random_state, **params)
    pipe = build_mice_pipeline(model, random_state=random_state)
    pipe.fit(X, y)
    return pipe


def compute_shap_values(pipe, X):
    """Returns (explainer, shap_values, X_transformed_for_display)."""
    # Transform X through imputer+scaler (everything except the final model step)
    X_transformed = X.copy()
    for name, step in pipe.steps[:-1]:
        X_transformed = step.transform(X_transformed)
    X_transformed = pd.DataFrame(X_transformed, columns=X.columns, index=X.index)

    model = pipe.named_steps["model"]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed)
    return explainer, shap_values, X_transformed


def global_importance_table(shap_values, feature_names):
    mean_abs = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({"Feature": feature_names, "Mean |SHAP value|": mean_abs})
    return df.sort_values("Mean |SHAP value|", ascending=False).reset_index(drop=True)
