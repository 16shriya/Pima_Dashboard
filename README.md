# Pima Diabetes — Reviewer-Proof Pipeline Dashboard

An interactive Streamlit dashboard that walks through the full pipeline live —
built to demonstrate every fix requested by the three conference reviewers.

## How to run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the dashboard
streamlit run app.py
```

Opens at `http://localhost:8501`. Use the sidebar to navigate between the 10 steps
(Step 0 through Step 9). For live presentation, the **Modeling** page (Step 4) has
sidebar sliders to reduce CV folds / Optuna trials if you need faster reruns during
a demo — increase them again for the "real" numbers before the panel sees them.

## Project structure

```
pima_dashboard/
├── app.py                     # Main Streamlit dashboard (all 10 steps)
├── data/
│   └── pima_diabetes.csv      # Standard Pima Indians Diabetes dataset (768 rows)
├── src/
│   ├── data_audit.py          # Step 1: zero/missingness quantification, MAR/MNAR test,
│   │                            outlier flags, duplicates, class balance, correlation
│   ├── preprocessing.py       # Step 3: leakage-safe MICE / median imputation pipelines
│   ├── modeling.py            # Step 4: baseline CV, Optuna Bayesian tuning,
│   │                            Wilcoxon significance test, calibration summary
│   ├── shap_explain.py        # Step 6: SHAP global + local explainability
│   ├── causal_recourse.py     # Step 7: causal-DAG-aware counterfactual recourse
│   │                            (answers Reviewer #2's independent-perturbation critique)
│   └── fairness.py            # Step 8: subgroup (Age/Pregnancy) performance analysis
└── requirements.txt
```
