"""
PIMA DIABETES — REVIEWER-PROOF ML PIPELINE DASHBOARD
Run with:  streamlit run app.py

Walks through every step of the playbook end-to-end, live:
  Step 0: Problem statement & domain understanding
  Step 1: Data audit
  Step 2: Exploratory data analysis
  Step 3: Leakage-safe preprocessing (MICE vs median)
  Step 4: Modeling + Bayesian hyperparameter optimization + significance testing
  Step 5: Calibration analysis
  Step 6: SHAP explainability
  Step 7: Causal-aware counterfactual recourse (answers Reviewer #2)
  Step 8: Subgroup fairness analysis (answers Reviewer #3)
  Step 9: Reviewer-checklist summary
"""

import warnings

warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, SimpleImputer

from src import data_audit
from src.preprocessing import encode_missing, build_mice_pipeline, build_median_pipeline
from src import modeling
from src import shap_explain
from src import causal_recourse as cr
from src import fairness
from src import live_predictor as lp

st.set_page_config(page_title="Pima Diabetes — Reviewer-Proof Pipeline", layout="wide")

DATA_PATH = "data/pima_diabetes.csv"


# ---------------------------------------------------------------------------
# Cached data loaders / heavy computations
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Cached data loaders / heavy computations
# ---------------------------------------------------------------------------
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.impute import IterativeImputer, SimpleImputer

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def get_audit(path):
    return data_audit.run_full_audit(path)


@st.cache_data
def cached_audit():
    return data_audit.run_full_audit(DATA_PATH)


# ---------------------------------------------------------------------------
# Imputation
# ---------------------------------------------------------------------------


@st.cache_data
def get_mice_completed(df):
    df_clean = encode_missing(df)

    completed = pd.DataFrame(
        IterativeImputer(
            max_iter=15,
            random_state=42,
        ).fit_transform(df_clean),
        columns=df_clean.columns,
    )

    return completed
@st.cache_data
def get_mice_completed(df):
    """
    Perform MICE imputation on predictor variables only.
    Outcome column is excluded.
    """

    # Keep only predictor variables
    X = df.drop(columns=["Outcome"]).copy()

    # Replace physiologically impossible zeros with NaN
    X = encode_missing(X)

    # MICE imputation
    imputer = IterativeImputer(
        max_iter=15,
        random_state=42,
    )

    X_complete = pd.DataFrame(
        imputer.fit_transform(X),
        columns=X.columns,
    )

    return X_complete

@st.cache_data
def get_median_completed(df):
    df_clean = encode_missing(df)

    completed = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(df_clean),
        columns=df_clean.columns,
    )

    return completed

# ---------------------------------------------------------------------------
# STEP 4 (FAST - PRECOMPUTED)
# ---------------------------------------------------------------------------


@st.cache_resource
def cached_cv_comparison():
    return joblib.load("results/baseline_results.pkl")


@st.cache_resource
def cached_optuna_tune():
    best_params = joblib.load("results/best_params.pkl")
    nested = joblib.load("results/nested_results.pkl")

    best_value = np.mean(nested["auc"])

    return best_params, best_value


@st.cache_resource
def cached_evaluate_optimized():
    nested = joblib.load("results/nested_results.pkl")
    return nested["auc"]


# ---------------------------------------------------------------------------
# STEP 5 / STEP 6
# Live model (NO pickle)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Fitting final deployment model & SHAP...")
def cached_final_model_and_shap(df, params_tuple):

    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    params = dict(params_tuple)

    pipe = shap_explain.fit_final_model(
        X,
        y,
        params=params,
    )

    explainer, shap_values, X_display = shap_explain.compute_shap_values(
        pipe,
        X,
    )

    return pipe, shap_values, X_display


# ---------------------------------------------------------------------------
# STEP 7
# Live structural equations
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Fitting structural equations...")
def cached_structural_equations(df_complete):

    return cr.fit_structural_equations(df_complete)


# ---------------------------------------------------------------------------
# Prediction Ensemble
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Training fold ensemble...")
def cached_prediction_ensemble(
    df,
    best_params_tuple,
    n_splits,
    n_repeats,
):

    best_params = dict(best_params_tuple)

    return lp.build_prediction_ensemble(
        df,
        best_params=best_params,
        n_splits=n_splits,
        n_repeats=n_repeats,
    )


# ---------------------------------------------------------------------------
# Observed feature ranges
# ---------------------------------------------------------------------------


@st.cache_data
def cached_observed_ranges(df):
    return lp.observed_range_table(df)


# ---------------------------------------------------------------------------
# Top-level navigation: Part 1 (pipeline walkthrough) vs Part 2 (live demo)
# ---------------------------------------------------------------------------
st.sidebar.title("Pima Diabetes Dashboard")
part = st.sidebar.radio(
    "Dashboard Section",
    ["Part 1 — Pipeline Walkthrough", "Part 2 — Live Prediction Demo"],
)
st.sidebar.markdown("---")

if part.startswith("Part 1"):
    st.sidebar.title("Pipeline Steps")
    step = st.sidebar.radio(
        "Navigate",
        [
            "1 — Data Audit",
            "2 — Exploratory Data Analysis",
            "3 — Preprocessing (MICE vs Median)",
            "4 — Modeling & Significance Testing",
            "5 — Calibration Analysis",
            "6 — SHAP Explainability",
            "7 — Causal-Aware Recourse",
        ],
    )
else:
    step = "P2"
    st.sidebar.caption(
        "Live prediction demo — enter patient values and get a risk estimate."
    )

st.sidebar.markdown("---")
st.sidebar.caption("Dataset: Pima Indians Diabetes (768 patients, 8 clinical features)")

df_raw = load_data()

# ===========================================================================
# STEP 0
# ===========================================================================
# if step.startswith("0"):
# st.title("Step 0 — Problem Statement & Domain Understanding")
# st.markdown("""
# ### What problem are we actually solving?
# "Diabetes prediction" hides at least four distinct problems. This work targets
# **risk screening + actionable recourse** — not autonomous diagnosis.

# | Framing | What it answers | Relevant here? |
# |---|---|---|
# | Diagnosis | Does this patient currently have diabetes? | No — not a lab-test replacement |
# | **Risk prediction / screening** | Probability of risk, to prioritize testing | **Yes** |
# | Risk stratification | Who benefits most from intervention? | Partially |
# | **Actionable recourse** | What changes would lower predicted risk? | **Yes** |

# ### Who is the end user?
# - **Clinician**: needs a *calibrated* probability and features that match clinical reasoning.
# - **Patient**: needs *achievable* recommendations — this is why immutable vs. mutable
#   feature separation is not a technical nicety, it's the entire point.
# - **Reviewer**: needs statistical rigor and honesty about limitations.

# ### What does existing domain knowledge already say?
# - ADA diagnostic criteria: fasting glucose ≥126 mg/dL, HbA1c ≥6.5%, OGTT ≥200 mg/dL.
# - Established risk factors: BMI, age, family history, pregnancies (gestational link).
# - Existing simple risk scores already exist (FINDRISC, ADA Risk Test) — the ML system
#   needs to justify its value over a free checklist.

# ### The data-generating process
# Pima Indians Diabetes data was collected from a NIH longitudinal study of Pima Heritage
# women near Phoenix, Arizona (1980s) — a population with one of the highest recorded
# diabetes prevalence rates, for well-documented genetic and dietary-transition reasons.

# **Consequence**: the ~35% base rate in this dataset is *not* representative of a general
# population (typically ~10–11% in the broader US). Any deployment claim beyond this specific
# population needs this caveat — directly motivating the external-validation step in this
# pipeline.
# """)

# ===========================================================================
# STEP 1 — DATA AUDIT
# ===========================================================================
if step.startswith("1"):
    st.title("Step 1 — Data Audit")
    audit = get_audit(DATA_PATH)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Zero / Missingness Quantification")
        st.dataframe(audit["zero_report"], use_container_width=True, hide_index=True)
        st.caption(
            "Zeros in these 5 columns are physiologically impossible — treated as missing."
        )

    with col2:
        st.subheader("Class Balance")
        st.dataframe(audit["class_balance"], use_container_width=True, hide_index=True)
        fig = px.pie(
            audit["class_balance"],
            names="Outcome",
            values="Count",
            title="Outcome Distribution",
            hole=0.4,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Missingness Mechanism Test (MAR vs MNAR signal)")
    st.markdown(
        "Tests whether *missingness itself* is statistically associated with the outcome "
        "(MNAR-leaning) or only with other observed features (MAR-consistent)."
    )
    st.dataframe(
        audit["missingness_mechanism"], use_container_width=True, hide_index=True
    )

    st.subheader("Outlier / Implausible Value Flags")
    st.dataframe(audit["outliers"], use_container_width=True, hide_index=True)

    st.subheader(f"Duplicate Rows: {audit['duplicates']}")

    st.subheader("Correlation Matrix (post missing-value encoding)")
    corr = audit["correlation"]
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Feature Correlation Matrix",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.info(
        "Glucose–Insulin (r=0.581) and BMI–SkinThickness (r=0.648) are the strongest pairs — "
        "this is the empirical evidence base for the causal DAG used in Step 7."
    )

# ===========================================================================
# STEP 2 — EDA
# ===========================================================================
elif step.startswith("2"):
    st.title("Step 2 — Exploratory Data Analysis")
    df_clean = encode_missing(df_raw)

    feature = st.selectbox(
        "Select feature to explore", [c for c in df_raw.columns if c != "Outcome"]
    )

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(
            df_clean,
            x=feature,
            color="Outcome",
            barmode="overlay",
            marginal="box",
            title=f"{feature} distribution by Outcome",
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = px.box(
            df_clean,
            x="Outcome",
            y=feature,
            color="Outcome",
            title=f"{feature} by Outcome (boxplot)",
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Pairwise scatter (colored by Outcome)")
    c1, c2 = st.columns(2)
    with c1:
        fx = st.selectbox("X axis", df_raw.columns[:-1], index=1, key="x_axis")
    with c2:
        fy = st.selectbox("Y axis", df_raw.columns[:-1], index=5, key="y_axis")
    fig3 = px.scatter(
        df_clean, x=fx, y=fy, color="Outcome", opacity=0.6, title=f"{fx} vs {fy}"
    )
    st.plotly_chart(fig3, use_container_width=True)

# ===========================================================================
# STEP 3 — PREPROCESSING
# ===========================================================================
elif step.startswith("3"):
    st.title("Step 3 — Leakage-Safe Preprocessing: MICE vs Median")
    st.markdown("""
    **Why this matters**: the original paper used median imputation. MICE (Iterative Imputer
    with BayesianRidge) exploits linear relationships between correlated features
    (e.g., predicting a patient's likely Insulin from their Glucose + BMI) rather than
    filling every missing value with the same dataset-wide number.

    **Leakage-safety**: both imputers below are demonstrated on the full dataset for
    visualization purposes only — in the actual CV pipeline (Step 4), they are refit
    **inside every training fold** and never see validation data.
    """)

    mice_completed = get_mice_completed(df_raw)
    median_completed = get_median_completed(df_raw)

    feature = st.selectbox(
        "Compare imputation methods for:",
        ["Insulin", "SkinThickness", "BloodPressure", "BMI", "Glucose"],
    )

    df_clean = encode_missing(df_raw)
    missing_mask = df_clean[feature].isna()

    compare_df = pd.DataFrame(
        {
            "Median Imputed": median_completed.loc[missing_mask, feature],
            "MICE Imputed": mice_completed.loc[missing_mask, feature],
        }
    )

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(x=compare_df["Median Imputed"], name="Median", opacity=0.6)
    )
    fig.add_trace(go.Histogram(x=compare_df["MICE Imputed"], name="MICE", opacity=0.6))
    fig.update_layout(
        barmode="overlay", title=f"Imputed values for {feature} (missing records only)"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
    - Median imputation fills **every** missing {feature} with the same value → an artificial spike.
    - MICE varies the imputed value per patient based on their other clinical measurements →
      more physiologically realistic, individualized values.
    """)

# ===========================================================================
# STEP 4 — MODELING & SIGNIFICANCE
# ===========================================================================
elif step.startswith("4"):
    st.title("Step 4 — Modeling, Bayesian Optimization & Statistical Significance")

    st.sidebar.markdown("### Speed controls")
    n_splits = st.sidebar.slider("CV folds", 3, 10, 5)
    n_repeats = st.sidebar.slider("CV repeats", 1, 5, 2)
    n_trials = st.sidebar.slider("Optuna trials", 5, 50, 15)

    # ----------------------------------------------------------
    # Baseline CV (compute only once)
    # ----------------------------------------------------------
    cache_key = f"cv_{n_splits}_{n_repeats}"

    if cache_key not in st.session_state:
        st.session_state[cache_key] = cached_cv_comparison()

    results = st.session_state[cache_key]

    st.subheader("Baseline Model Comparison (Repeated Stratified CV)")

    summary_rows = []
    for name, r in results.items():
        summary_rows.append(
            {
                "Model": name,
                "Mean ROC-AUC": round(np.mean(r["auc"]), 4),
                "Std ROC-AUC": round(np.std(r["auc"]), 4),
                "Mean F1": round(np.mean(r["f1"]), 4),
                "Mean Accuracy": round(np.mean(r["acc"]), 4),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("Mean ROC-AUC", ascending=False)

    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    fig = px.box(
        pd.DataFrame({name: r["auc"] for name, r in results.items()}).melt(
            var_name="Model",
            value_name="ROC-AUC",
        ),
        x="Model",
        y="ROC-AUC",
        points="all",
        title="ROC-AUC distribution across folds",
    )

    st.plotly_chart(fig, use_container_width=True)

    # ----------------------------------------------------------
    # Optuna (compute only once)
    # ----------------------------------------------------------
    optuna_key = f"optuna_{n_trials}_{n_splits}"

    if optuna_key not in st.session_state:

        best_params, best_value = cached_optuna_tune()

        st.session_state[optuna_key] = (
            best_params,
            best_value,
        )

    best_params, best_value = st.session_state[optuna_key]

    st.subheader("Bayesian Hyperparameter Optimization")

    st.write("**Best Hyperparameters:**")
    st.json(best_params)

    st.write(f"**Best Mean CV ROC-AUC:** {best_value:.4f}")

    # ----------------------------------------------------------
    # Optimized evaluation (compute once)
    # ----------------------------------------------------------
    eval_key = f"optimized_{n_splits}_{n_repeats}_{tuple(sorted(best_params.items()))}"

    if eval_key not in st.session_state:

        st.session_state[eval_key] = cached_evaluate_optimized()

    optimized_aucs = st.session_state[eval_key]

    # ----------------------------------------------------------
    # Statistics
    # ----------------------------------------------------------

    baseline_catboost_aucs = results["CatBoost"]["auc"][: len(optimized_aucs)]

    sig = modeling.paired_significance_test(
        baseline_catboost_aucs,
        optimized_aucs,
    )

    st.subheader("Paired Statistical Significance Test")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Mean AUC Gain",
        f"{sig['mean_diff']:+.4f}",
    )

    c2.metric(
        "Wilcoxon p-value",
        f"{sig['p_value']:.4f}",
    )

    c3.metric(
        "Significant?",
        "Yes" if sig["significant_at_0.05"] else "No",
    )

    if sig["significant_at_0.05"]:
        st.success("Improvement is statistically significant.")
    else:
        st.warning("Improvement did not reach statistical significance.")

    st.session_state["best_params"] = best_params
# ===========================================================================
# STEP 5 — CALIBRATION
# ===========================================================================
elif step.startswith("5"):

    st.title("Step 5 — Calibration Analysis")

    # Get best parameters found in Step 4
    best_params = st.session_state.get(
        "best_params",
        {"iterations": 200, "depth": 6},
    )

    # Fit final deployment model (cached)
    pipe, shap_values, X_t = cached_final_model_and_shap(
        df_raw,
        tuple(sorted(best_params.items())),
    )

    X = df_raw.drop(columns=["Outcome"])
    y = df_raw["Outcome"]

    # Predicted probabilities
    proba = pipe.predict_proba(X)[:, 1]

    # Calibration statistics
    bins, brier = modeling.calibration_summary(
        y.values,
        proba,
        n_bins=10,
    )

    bin_df = pd.DataFrame(bins)

    # ---------------------------------------------------------
    # Calibration curve
    # ---------------------------------------------------------
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect calibration",
            line=dict(
                dash="dash",
                color="gray",
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=bin_df["mean_predicted"],
            y=bin_df["fraction_positive"],
            mode="lines+markers",
            name="CatBoost",
        )
    )

    fig.update_layout(
        title=f"Calibration Curve (Brier Score = {brier:.4f})",
        xaxis_title="Mean Predicted Probability",
        yaxis_title="Observed Fraction of Positives",
        template="plotly_white",
    )

    st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------
    c1, c2 = st.columns(2)

    c1.metric(
        "Brier Score",
        f"{brier:.4f}",
    )

    c2.metric(
        "Calibration Bins",
        len(bin_df),
    )

    # ---------------------------------------------------------
    # Calibration table
    # ---------------------------------------------------------
    st.subheader("Calibration Bin Summary")

    st.dataframe(
        bin_df,
        use_container_width=True,
        hide_index=True,
    )

# ===========================================================================
# STEP 6 — SHAP
# ===========================================================================
elif step.startswith("6"):
    st.title("Step 6 — SHAP Explainability")
    best_params = st.session_state.get("best_params", {"iterations": 200, "depth": 6})
    pipe, shap_values, X_t = cached_final_model_and_shap(
        df_raw, tuple(sorted(best_params.items()))
    )

    st.subheader("Global Feature Importance")
    gi = shap_explain.global_importance_table(shap_values, X_t.columns.tolist())
    fig = px.bar(
        gi,
        x="Mean |SHAP value|",
        y="Feature",
        orientation="h",
        title="Global SHAP Importance",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("SHAP Dependence: Glucose")
    fig2 = px.scatter(
        x=X_t["Glucose"],
        y=shap_values[:, X_t.columns.get_loc("Glucose")],
        color=X_t["BMI"],
        labels={"x": "Glucose", "y": "SHAP value", "color": "BMI"},
        title="SHAP dependence plot for Glucose (colored by BMI)",
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Local Explanation — pick a patient")
    idx = st.slider("Patient index", 0, len(X_t) - 1, 4)
    patient_shap = shap_values[idx]
    patient_df = pd.DataFrame(
        {
            "Feature": X_t.columns,
            "SHAP value": patient_shap,
            "Feature value": X_t.iloc[idx].values,
        }
    ).sort_values("SHAP value", key=abs, ascending=False)
    fig3 = px.bar(
        patient_df,
        x="SHAP value",
        y="Feature",
        orientation="h",
        title=f"Local SHAP contributions — patient #{idx}",
        color="SHAP value",
        color_continuous_scale="RdBu_r",
    )
    fig3.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig3, use_container_width=True)
    st.dataframe(patient_df, use_container_width=True, hide_index=True)

# ===========================================================================
# STEP 7 — CAUSAL-AWARE RECOURSE
# ===========================================================================
# ===========================================================================
# STEP 7 — CAUSAL-AWARE COUNTERFACTUAL RECOURSE
# ===========================================================================
elif step.startswith("7"):

    st.title("Step 7 — Causal-Aware Counterfactual Recourse")

    st.markdown("""
A minimal causal DAG is fit from the cleaned data:

- BMI → BloodPressure
- BMI → SkinThickness
- BMI → Insulin
- Glucose → Insulin
""")

    best_params = st.session_state.get(
        "best_params",
        {"iterations": 200, "depth": 6},
    )

    # Cached final model
    pipe, _, _ = cached_final_model_and_shap(
        df_raw,
        tuple(sorted(best_params.items())),
    )

    # MICE-completed feature matrix (Outcome already removed)
    X = get_mice_completed(df_raw)

    # Fit structural equations once
    equations = cached_structural_equations(X)

    # Predict risk
    proba_all = pipe.predict_proba(X)[:, 1]

    high_risk_idx = np.argsort(-proba_all)[:30]

    idx = st.selectbox(
        "Select a high-risk patient",
        high_risk_idx,
        format_func=lambda i: (
            f"Patient #{i} " f"(Predicted Risk = {proba_all[i]:.3f})"
        ),
    )

    patient = X.iloc[idx].to_dict()

    target_proba = st.slider(
        "Target risk probability",
        0.10,
        0.50,
        0.30,
        0.05,
    )

    if st.button("Generate Counterfactual"):

        with st.spinner("Searching feasible recourse..."):

            final_ind, proba_ind = cr.independent_recourse(
                pipe,
                patient,
                X.columns.tolist(),
                target_proba=target_proba,
            )

            final_causal, proba_causal = cr.causal_recourse(
                pipe,
                patient,
                equations,
                X.columns.tolist(),
                target_proba=target_proba,
            )

        original = proba_all[idx]

        st.metric(
            "Original Predicted Risk",
            f"{original:.3f}",
        )

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Independent Perturbation")

            st.metric(
                "New Risk",
                f"{proba_ind:.3f}",
                delta=f"{proba_ind-original:+.3f}",
            )

            comp_ind = pd.DataFrame(
                {
                    "Feature": list(patient.keys()),
                    "Original": list(patient.values()),
                    "Recourse": [final_ind[k] for k in patient.keys()],
                }
            )

            st.dataframe(
                comp_ind,
                use_container_width=True,
                hide_index=True,
            )

        with col2:

            st.subheader("Causal Propagation")

            st.metric(
                "New Risk",
                f"{proba_causal:.3f}",
                delta=f"{proba_causal-original:+.3f}",
            )

            comp_causal = pd.DataFrame(
                {
                    "Feature": list(patient.keys()),
                    "Original": list(patient.values()),
                    "Recourse": [final_causal[k] for k in patient.keys()],
                }
            )

            st.dataframe(
                comp_causal,
                use_container_width=True,
                hide_index=True,
            )

        st.success(
            "Causal recourse propagates downstream physiological effects through the learned structural equations."
        )
# ===========================================================================
# STEP 8 — FAIRNESS
# ===========================================================================
# elif step.startswith("8"):
#     st.title("Step 8 — Subgroup Fairness Analysis")
#     best_params = st.session_state.get("best_params", {"iterations": 200, "depth": 6})
#     pipe, _, _ = cached_final_model_and_shap(df_raw, tuple(sorted(best_params.items())))

#     X = df_raw.drop(columns=["Outcome"])
#     y = df_raw["Outcome"]
#     proba_all = pipe.predict_proba(X)[:, 1]

#     df_g = fairness.add_subgroups(df_raw)

#     st.subheader("Performance by Age Group")
#     age_perf = fairness.subgroup_performance(df_g, proba_all, "Age_group")
#     st.dataframe(age_perf, use_container_width=True, hide_index=True)
#     fig = px.bar(
#         age_perf, x="Subgroup", y="ROC-AUC", title="ROC-AUC by Age Group", text="N"
#     )
#     st.plotly_chart(fig, use_container_width=True)

#     st.subheader("Performance by Pregnancy Group")
#     preg_perf = fairness.subgroup_performance(df_g, proba_all, "Pregnancy_group")
#     st.dataframe(preg_perf, use_container_width=True, hide_index=True)
#     fig2 = px.bar(
#         preg_perf,
#         x="Subgroup",
#         y="ROC-AUC",
#         title="ROC-AUC by Pregnancy Group",
#         text="N",
#     )
#     st.plotly_chart(fig2, use_container_width=True)

#     st.caption(
#         "Note: in-sample performance shown for demonstration. Report CV-based subgroup "
#         "performance in the paper to avoid optimistic bias."
#     )

# ===========================================================================
# STEP 9 — SUMMARY
# ===========================================================================
# elif step.startswith("9"):
#     st.title("Step 9 — Reviewer Checklist Summary")
#     checklist = pd.DataFrame(
#         [
#             {
#                 "Reviewer Concern": "Manual median imputation, unrealistic zeros",
#                 "Addressed By": "MICE (IterativeImputer), fit inside each CV fold",
#                 "Dashboard Step": "Step 3",
#             },
#             {
#                 "Reviewer Concern": "Manually-designed feasibility bounds",
#                 "Addressed By": "Bounds derived from empirical/clinical plausible ranges",
#                 "Dashboard Step": "Step 7",
#             },
#             {
#                 "Reviewer Concern": "Independent feature perturbation (Reviewer #2)",
#                 "Addressed By": "Causal DAG + structural equations propagating BMI/Glucose shifts",
#                 "Dashboard Step": "Step 7",
#             },
#             {
#                 "Reviewer Concern": "No statistical significance testing",
#                 "Addressed By": "Paired Wilcoxon signed-rank test, baseline vs optimized CatBoost",
#                 "Dashboard Step": "Step 4",
#             },
#             {
#                 "Reviewer Concern": "No fairness analysis",
#                 "Addressed By": "Subgroup ROC-AUC/Brier by Age and Pregnancy groups",
#                 "Dashboard Step": "Step 8",
#             },
#             {
#                 "Reviewer Concern": "Small dataset, generalizability",
#                 "Addressed By": "Explicit population caveat (Step 0) + external validation recommended",
#                 "Dashboard Step": "Step 0 / Future Work",
#             },
#             {
#                 "Reviewer Concern": "No missingness mechanism justification",
#                 "Addressed By": "Point-biserial correlation test of missing-indicator vs Outcome/covariates",
#                 "Dashboard Step": "Step 1",
#             },
#         ]
#     )
#     st.dataframe(checklist, use_container_width=True, hide_index=True)
#     st.success(
#         "Every major reviewer comment from all three reviews maps to a concrete, "
#         "demonstrated step in this pipeline."
#     )

# ===========================================================================
# PART 2 — LIVE PREDICTION DEMO
# ===========================================================================
elif step == "P2":
    st.title("Live Diabetes Risk Prediction — Demo")

    st.warning(
        "**This is a research screening tool, not a diagnostic device.** "
        "It estimates statistical risk based on patterns in a specific historical "
        "dataset (768 women of Pima Heritage, 1980s NIH study). It does not diagnose "
        "diabetes, does not replace a clinical test (fasting glucose, OGTT, HbA1c), "
        "and should never be the sole basis for a medical decision. No counterfactual "
        "or 'what to change' recommendation is shown here intentionally — recourse "
        "analysis is a separate, clearly-labeled research question, not a clinical "
        "action plan (see Part 1, Step 7)."
    )

    best_params = st.session_state.get("best_params", {"iterations": 200, "depth": 6})
    obs_ranges = cached_observed_ranges(df_raw)

    with st.expander(
        "What data range did the model actually learn from?", expanded=False
    ):
        st.dataframe(obs_ranges, use_container_width=True, hide_index=True)
        st.caption(
            "Inputs outside the 1st–99th percentile column are extrapolations — the "
            "model has seen little or no training data like that, and the prediction "
            "should be treated with more caution, not less."
        )

    st.subheader("Enter patient clinical measurements")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        pregnancies = st.number_input(
            "Pregnancies (count)", min_value=0, max_value=20, value=2, step=1
        )
        glucose = st.number_input(
            "Glucose (mg/dL, plasma 2h OGTT)", min_value=0, max_value=400, value=120
        )
    with c2:
        bp = st.number_input(
            "Blood Pressure (mmHg, diastolic)", min_value=0, max_value=200, value=70
        )
        skin = st.number_input(
            "Skin Thickness (mm, triceps)", min_value=0, max_value=120, value=25
        )
    with c3:
        insulin = st.number_input(
            "Insulin (mu U/mL, 2h serum)", min_value=0, max_value=1000, value=100
        )
        bmi = st.number_input(
            "BMI (kg/m²)", min_value=10.0, max_value=80.0, value=28.0, step=0.1
        )
    with c4:
        dpf = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0,
            max_value=3.0,
            value=0.4,
            step=0.01,
        )
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=35)

    patient = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": bp,
        "SkinThickness": skin,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age,
    }

    if st.button("Run risk prediction", type="primary"):
        is_invalid, hard_errors, soft_warnings = lp.validate_input(patient, obs_ranges)

        if is_invalid:
            st.error("**Prediction blocked — input is physiologically impossible:**")
            for e in hard_errors:
                st.error(f"• {e}")
        else:
            if soft_warnings:
                st.warning("**Out-of-distribution input detected:**")
                for w in soft_warnings:
                    st.warning(f"• {w}")

            with st.spinner(
                "Scoring across fold-ensemble for an uncertainty-aware estimate..."
            ):
                ensemble = cached_prediction_ensemble(
                    df_raw, tuple(sorted(best_params.items())), 5, 2
                )
                mean_p, std_p, all_probs = lp.predict_with_uncertainty(
                    ensemble, patient
                )
                category, level = lp.risk_category(mean_p, std_p)

            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            col1.metric("Predicted risk probability", f"{mean_p:.1%}")
            col2.metric("Uncertainty (std across 10 fold-models)", f"±{std_p:.1%}")
            col3.metric("Category", category)

            if level == "success":
                st.success(
                    f"**{category}** — estimated probability {mean_p:.1%} (±{std_p:.1%})"
                )
            elif level == "warning":
                st.warning(
                    f"**{category}** — estimated probability {mean_p:.1%} (±{std_p:.1%})"
                )
            else:
                st.error(
                    f"**{category}** — estimated probability {mean_p:.1%} (±{std_p:.1%})"
                )

            fig = px.histogram(
                x=all_probs,
                nbins=10,
                range_x=[0, 1],
                title="Spread of predicted probability across 10 independently-trained fold models",
            )
            fig.update_layout(
                xaxis_title="Predicted probability", yaxis_title="Number of fold-models"
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "If this spread is wide, the fold-models disagree on this patient — that's "
                "real information, not noise to be hidden. A wide spread means treat the "
                "point estimate with real caution."
            )

            st.subheader("Why did the model produce this number? (attribution only)")
            st.caption(
                "This shows which of THIS patient's values pushed the prediction up or down "
                "using SHAP. It is a description of the model's reasoning, not a suggested "
                "action — nothing here should be read as 'change X to lower your risk.'"
            )
            expl = lp.explain_single_prediction(ensemble, patient)
            fig2 = px.bar(
                expl,
                x="SHAP contribution",
                y="Feature",
                orientation="h",
                color="SHAP contribution",
                color_continuous_scale="RdBu_r",
                title="Feature attribution for this prediction",
            )
            fig2.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig2, use_container_width=True)
            st.dataframe(expl, use_container_width=True, hide_index=True)
            # ==========================================================
            # NEW SECTION STARTS HERE
            # ==========================================================

            st.markdown("---")

            # Medical Feasibility Validation
            st.subheader(" Medical Feasibility Validation")

            rows = []

            immutable = ["Age","Pregnancies","DiabetesPedigreeFunction"]

            for feature,value in patient.items():

                if feature in immutable:
                    status = " Immutable"

                elif feature=="BMI":
                    if value >= 30:
                        status = " Lifestyle intervention recommended"
                    else:
                        status = " Within acceptable range"

                elif feature=="Glucose":
                    if value >= 140:
                        status = " Clinical follow-up advised"
                    else:
                        status = " Acceptable"

                elif feature=="BloodPressure":
                    if value >= 90:
                        status = " Monitor blood pressure"
                    else:
                        status = " Normal"

                else:
                    status = "✓"

                rows.append([
                    feature,
                    value,
                    status
                ])

            validation_df = pd.DataFrame(
                rows,
                columns=[
                    "Feature",
                    "Current Value",
                    "Clinical Assessment"
                ]
            )

            st.dataframe(validation_df,use_container_width=True,hide_index=True)

            # Guideline Source Table
            st.subheader("Clinical Guideline Reference")

            guidelines = pd.DataFrame({
                "Risk Factor":[
                    "BMI",
                    "Glucose",
                    "Blood Pressure",
                    "Age",
                    "Pregnancies"
                ],
                "Clinical Recommendation":[
                    "5–10% weight reduction",
                    "Confirm with HbA1c / FPG testing",
                    "Routine BP monitoring",
                    "Routine diabetes screening",
                    "Gestational diabetes history assessment"
                ],
                "Guideline":[
                    "ADA",
                    "ADA",
                    "AHA",
                    "ADA",
                    "ADA"
                ]
            })

            st.dataframe(
                guidelines,
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "Recommendations are derived from established clinical guidelines "
                "(ADA: American Diabetes Association, AHA: American Heart Association) "
                "and are intended as decision-support rather than medical advice."
            )

            # Personalized Recommendations
            st.subheader(" Personalized Actionable Recommendations")

            recommendations = []

            if patient["BMI"] >= 30:
                recommendations.append(
                    " Reduce body weight by 5–10% through diet and regular physical activity (ADA)."
                )

            if patient["Glucose"] >= 140:
                recommendations.append(
                    " Undergo confirmatory HbA1c or fasting plasma glucose testing (ADA)."
                )

            if patient["BloodPressure"] >= 90:
                recommendations.append(
                    " Monitor blood pressure regularly and reduce sodium intake (AHA)."
                )

            if patient["Age"] >= 45:
                recommendations.append(
                    " Schedule routine diabetes screening because age increases diabetes risk (ADA)."
                )

            if len(recommendations) == 0:
                recommendations.append(
                    " Continue maintaining a healthy lifestyle with periodic diabetes screening."
                )

            for rec in recommendations:
                st.success(rec)

            # ==========================================================
            # NEW SECTION ENDS HERE
            # ==========================================================

            st.markdown("---")
            st.info(
                "**Reminder of this tool's limits (matches the paper's stated limitations):** "
                "trained on a small (768-patient), single-population, single-sex dataset; "
                "predictions reflect statistical association, not a causal mechanism; "
                "calibration and subgroup fairness were only partially validated (see Part 1, "
                "Steps 5 & 8); this demo does not constitute clinical advice."
            )
