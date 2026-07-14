"""
STEP 4 — MODELING, HYPERPARAMETER OPTIMIZATION, STATISTICAL SIGNIFICANCE
Repeated stratified CV for baseline models, Bayesian optimization (Optuna) for
CatBoost, and a paired Wilcoxon signed-rank test comparing baseline vs optimized
AUC across folds (directly answers the "no significance testing" reviewer gap).
"""
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import wilcoxon
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import optuna

from .preprocessing import build_mice_pipeline

optuna.logging.set_verbosity(optuna.logging.WARNING)


def get_baseline_models(random_state=42):
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=random_state),
        "XGBoost": XGBClassifier(
            eval_metric="logloss", random_state=random_state, verbosity=0
        ),
        "CatBoost": CatBoostClassifier(
            verbose=0, random_state=random_state
        ),
    }


def run_cv_comparison(X, y, n_splits=10, n_repeats=3, random_state=42, progress_callback=None):
    """
    Runs repeated stratified CV for each baseline model using the MICE pipeline
    (fit only on training folds -> leakage-safe).
    Returns: dict of {model_name: {"auc": [...], "f1": [...], "acc": [...]}}
    """
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    models = get_baseline_models(random_state)
    results = {name: {"auc": [], "f1": [], "acc": []} for name in models}

    total_folds = n_splits * n_repeats
    fold_i = 0
    for train_idx, test_idx in rskf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        for name, model in models.items():
            pipe = build_mice_pipeline(model, random_state=random_state)
            pipe.fit(X_train, y_train)
            proba = pipe.predict_proba(X_test)[:, 1]
            pred = pipe.predict(X_test)
            results[name]["auc"].append(roc_auc_score(y_test, proba))
            results[name]["f1"].append(f1_score(y_test, pred))
            results[name]["acc"].append(accuracy_score(y_test, pred))

        fold_i += 1
        if progress_callback:
            progress_callback(fold_i, total_folds)

    return results


def optuna_tune_catboost(X, y, n_trials=25, n_splits=5, random_state=42, progress_callback=None):
    """
    Bayesian hyperparameter optimization (Optuna's TPE sampler) for CatBoost.
    Objective: maximize mean ROC-AUC across stratified CV folds (leakage-safe MICE pipeline).
    """
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=1, random_state=random_state)
    trial_counter = {"n": 0}

    def objective(trial):
        params = {
            "depth": trial.suggest_int("depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "iterations": trial.suggest_int("iterations", 100, 500),
        }
        model = CatBoostClassifier(verbose=0, random_state=random_state, **params)
        aucs = []
        for train_idx, test_idx in rskf.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            pipe = build_mice_pipeline(model, random_state=random_state)
            pipe.fit(X_train, y_train)
            proba = pipe.predict_proba(X_test)[:, 1]
            aucs.append(roc_auc_score(y_test, proba))

        trial_counter["n"] += 1
        if progress_callback:
            progress_callback(trial_counter["n"], n_trials)
        return float(np.mean(aucs))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, study.best_value, study


def evaluate_optimized_model(X, y, best_params, n_splits=10, n_repeats=3, random_state=42):
    """Re-run repeated CV with the tuned hyperparameters to get a fold-level AUC distribution
    directly comparable (same fold scheme) to the baseline CatBoost run."""
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    aucs = []
    for train_idx, test_idx in rskf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model = CatBoostClassifier(verbose=0, random_state=random_state, **best_params)
        pipe = build_mice_pipeline(model, random_state=random_state)
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        aucs.append(roc_auc_score(y_test, proba))
    return aucs


def paired_significance_test(baseline_aucs, optimized_aucs):
    """
    Paired Wilcoxon signed-rank test (nonparametric, appropriate for small paired
    fold-level samples where normality can't be assumed).
    """
    baseline_aucs = np.array(baseline_aucs)
    optimized_aucs = np.array(optimized_aucs)
    diff = optimized_aucs - baseline_aucs
    if np.all(diff == 0):
        return {"statistic": None, "p_value": 1.0, "mean_diff": 0.0}
    stat, p = wilcoxon(baseline_aucs, optimized_aucs)
    return {
        "statistic": float(stat),
        "p_value": float(p),
        "mean_diff": float(np.mean(diff)),
        "significant_at_0.05": bool(p < 0.05),
    }


def calibration_summary(y_true, y_proba, n_bins=10):
    """Returns bin-level calibration data + overall Brier score."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(y_proba, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        rows.append({
            "bin_center": (bins[b] + bins[b + 1]) / 2,
            "mean_predicted": float(np.mean(y_proba[mask])),
            "fraction_positive": float(np.mean(y_true[mask])),
            "n_samples": int(mask.sum()),
        })
    brier = float(brier_score_loss(y_true, y_proba))
    return rows, brier
