"""
STEP 6 — CAUSAL-AWARE COUNTERFACTUAL RECOURSE
Directly answers Reviewer #2: "physiological variables are deeply interconnected;
reducing BMI will organically impact blood pressure and insulin sensitivity."

Approach:
  1. Define a minimal causal DAG from domain knowledge + the correlation audit
     (BMI -> BloodPressure, BMI -> Insulin, Glucose -> Insulin).
  2. Fit simple linear structural equations for each edge (child ~ parent) on
     the MICE-completed data.
  3. When searching for a counterfactual by perturbing a root cause (BMI and/or
     Glucose), PROPAGATE the expected downstream shift to children rather than
     holding them fixed (which is what naive independent-perturbation recourse does).
  4. Compare "independent perturbation" vs "causal propagated" recourse side by side.

Immutable features (Age, DiabetesPedigreeFunction, Pregnancies) are never perturbed.
"""
import numpy as np
import pandas as pd
# from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
# from scipy.optimize import differential_evolution
from scipy.optimize import minimize

IMMUTABLE = ["Age", "DiabetesPedigreeFunction", "Pregnancies"]
MUTABLE_ROOTS = ["Glucose", "BMI"]  # features we directly intervene on
CAUSAL_EDGES = {
    # child_feature: [parent_features]
    "BloodPressure": ["BMI"],
    "Insulin": ["BMI", "Glucose"],
    "SkinThickness": ["BMI"],
}

# Medically-informed bounds (approx. physiological plausible ranges), used instead
# of arbitrary manual deltas -- derived from empirical percentiles + clinical ranges.
FEASIBLE_BOUNDS = {
    "Glucose": (70, 200),
    "BMI": (18.5, 45),
}
# Physiological limits for causal descendants
PHYSIO_LIMITS = {
    "BloodPressure": (40, 140),  # mmHg
    "SkinThickness": (0, 80),  # mm
    "Insulin": (0, 600),  # μU/mL
}


def fit_structural_equations(df_complete: pd.DataFrame):
    """
    Fit nonlinear structural equations using
    HistGradientBoostingRegressor.
    """

    equations = {}

    for child, parents in CAUSAL_EDGES.items():

        X = df_complete[parents].values
        y = df_complete[child].values

        reg = HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=200, random_state=42
        )

        reg.fit(X, y)

        equations[child] = {"parents": parents, "model": reg}

    return equations


def propagate(patient: dict, new_roots: dict, equations: dict) -> dict:
    """
    Propagate downstream causal effects while enforcing
    physiologically plausible ranges.
    """
    modified = dict(patient)
    modified.update(new_roots)

    for child, eq in equations.items():
        parents = eq["parents"]
        reg = eq["model"]

        # Prediction after intervention
        parent_vals_new = np.array([[modified[p] for p in parents]])
        predicted_child = reg.predict(parent_vals_new)[0]

        # Prediction before intervention
        parent_vals_old = np.array([[patient[p] for p in parents]])
        predicted_child_old = reg.predict(parent_vals_old)[0]

        # Causal shift
        delta = predicted_child - predicted_child_old

        modified[child] = patient[child] + delta

        # -------------------------------
        # Physiological clipping
        # -------------------------------
        if child in PHYSIO_LIMITS:
            lo, hi = PHYSIO_LIMITS[child]
            modified[child] = float(np.clip(modified[child], lo, hi))

    return modified


def _predict_proba_single(pipe, patient_dict, feature_order):
    row = pd.DataFrame([patient_dict])[feature_order]
    return pipe.predict_proba(row)[0, 1]


def independent_recourse(pipe, patient: dict, feature_order, target_proba=0.3, seed=42):
    """
    Naive baseline: perturb Glucose & BMI only, hold every other feature (including
    causally-dependent ones) fixed at original values. This mirrors what the
    original paper's unconstrained-independence approach does.
    Uses differential evolution (derivative-free) since tree-model predictions
    are piecewise-constant and gradient-based search stalls at the start point.
    """
    x0 = np.array([patient["Glucose"], patient["BMI"]])
    bounds = [FEASIBLE_BOUNDS["Glucose"], FEASIBLE_BOUNDS["BMI"]]

    def objective(x):
        candidate = dict(patient)
        candidate["Glucose"], candidate["BMI"] = x
        proba = _predict_proba_single(pipe, candidate, feature_order)
        validity_loss = max(0, proba - target_proba) ** 2
        distance = ((x[0] - x0[0]) / 130) ** 2 + ((x[1] - x0[1]) / 26) ** 2
        return validity_loss * 5 + distance

    # res = differential_evolution(
    #     objective,
    #     bounds,
    #     seed=seed,
    #     maxiter=20,
    #     popsize=6,
    #     tol=1e-3,
    #     polish=False,
    # )
    res = minimize(

        objective,

        x0,

        method="Powell",

        bounds=bounds,

        options={

            "maxiter":80,

            "xtol":1e-2,

            "ftol":1e-2,

        },

    )
    final = dict(patient)
    final["Glucose"], final["BMI"] = res.x
    final_proba = _predict_proba_single(pipe, final, feature_order)
    return final, final_proba


from scipy.optimize import minimize


def causal_recourse(pipe, patient: dict, equations, feature_order, target_proba=0.3):

    x0 = np.array([patient["Glucose"], patient["BMI"]])

    bounds = [
        FEASIBLE_BOUNDS["Glucose"],
        FEASIBLE_BOUNDS["BMI"],
    ]

    def objective(x):

        new_roots = {
            "Glucose": x[0],
            "BMI": x[1],
        }

        candidate = propagate(
            patient,
            new_roots,
            equations,
        )

        proba = _predict_proba_single(
            pipe,
            candidate,
            feature_order,
        )

        validity_loss = (
            max(
                0,
                proba - target_proba,
            )
            ** 2
        )

        distance = ((x[0] - x0[0]) / 130) ** 2 + ((x[1] - x0[1]) / 26) ** 2

        return validity_loss * 5 + distance

    res = minimize(
        objective,
        x0=x0,
        method="Powell",
        bounds=bounds,
        options={
            "maxiter": 40,
            "xtol": 1e-2,
            "ftol": 1e-2,
        },
    )

    new_roots = {
        "Glucose": res.x[0],
        "BMI": res.x[1],
    }

    final = propagate(
        patient,
        new_roots,
        equations,
    )

    final_proba = _predict_proba_single(
        pipe,
        final,
        feature_order,
    )

    return final, final_proba
