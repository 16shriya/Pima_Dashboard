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

## What each step demonstrates (mapped to reviewer comments)

| Step | What it shows                                                               | Reviewer comment answered                           |
| ---- | --------------------------------------------------------------------------- | --------------------------------------------------- |
| 0    | Problem framing, end users, domain grounding                                | (context-setting)                                   |
| 1    | Real zero/missingness counts, MAR/MNAR statistical test, correlation matrix | Rev #2/#3: manual constraints not grounded in data  |
| 2    | Interactive EDA (histograms, boxplots, scatter)                             | (supporting evidence)                               |
| 3    | MICE vs. median imputation, side-by-side                                    | Rev #1: stronger preprocessing rigor                |
| 4    | Repeated stratified CV, Optuna tuning, **paired Wilcoxon test**             | Rev #3: no statistical significance testing         |
| 5    | Calibration curve + Brier score                                             | Rev #1: probabilistic reliability                   |
| 6    | SHAP global + local explanations                                            | (core interpretability claim)                       |
| 7    | **Causal-DAG-propagated recourse vs. naive independent perturbation**       | Rev #2: features are physiologically interconnected |
| 8    | Subgroup ROC-AUC/Brier by Age and Pregnancy group                           | Rev #3: fairness analysis across demographic groups |
| 9    | Full checklist mapping table                                                | Summary for panel Q&A                               |

## Notes for the panel presentation

- All numbers are computed **live** from the actual CSV, not hardcoded — you can
  change the dataset file and everything recalculates.
- Step 4's significance test may show a **non-significant** p-value for the
  Bayesian-optimization gain — this is expected given Pima's small size (768 rows)
  and is worth stating honestly to the panel rather than hidden; it directly
  demonstrates rigor rather than weakness.
- Step 7 is the most important addition relative to the original manuscript —
  it's a working, if simplified, answer to the sharpest reviewer critique
  (Reviewer #2's causal-independence concern).
- Calibration/fairness numbers shown are in-sample for interactive speed; for the
  camera-ready numbers, swap in cross-validated out-of-fold predictions
  (the CV infrastructure in `modeling.py` already supports this with minor changes).
