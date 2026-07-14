"""
STEP 4 — NESTED CROSS-VALIDATION
Fixes the tuning-leakage bug present in the original notebook, where Optuna's
objective ran cross_val_score(pipeline, X, y, ...) on the FULL dataset -- meaning
hyperparameters were chosen with knowledge of data that later became the "held out"
test set for calibration/reporting.

The correct structure (nested CV):
    for each OUTER fold (train_outer, test_outer):
        run Optuna, tuning hyperparameters using ONLY an INNER CV on train_outer
        fit the best model on the full train_outer (leakage-safe MICE pipeline)
        evaluate ONCE on test_outer (never touched during tuning)
    -> the collected test_outer scores are an unbiased performance estimate

This is more expensive than a single CV loop but it is the statistically correct
way to report "tuned model performance" without optimistic bias.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import optuna

from .preprocessing import build_mice_pipeline
from .config import RANDOM_STATE, OUTER_SPLITS, OUTER_REPEATS, INNER_SPLITS, N_OPTUNA_TRIALS

optuna.logging.set_verbosity(optuna.logging.WARNING)


def get_baseline_models(random_state=RANDOM_STATE):
    """Untuned baselines -- no hyperparameter search, so no nested CV needed for these;
    a single leakage-safe (fold-internal MICE) CV loop is statistically valid."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=random_state),
        "XGBoost": XGBClassifier(eval_metric="logloss", random_state=random_state, verbosity=0),
        "CatBoost": CatBoostClassifier(verbose=0, random_state=random_state),
    }


def run_baseline_cv(X, y, n_splits=OUTER_SPLITS, n_repeats=OUTER_REPEATS, random_state=RANDOM_STATE):
    """Repeated stratified CV for untuned baseline models. MICE fit inside each fold."""
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    models = get_baseline_models(random_state)
    results = {name: {"auc": [], "f1": [], "acc": []} for name in models}

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
    return results


def _inner_tune(X_train_outer, y_train_outer, n_trials, n_inner_splits, random_state, trial_seed_offset=0):
    """Optuna search using ONLY the outer-training data, with its own inner CV.
    This never sees the outer-test fold."""
    inner_cv = StratifiedKFold(n_splits=n_inner_splits, shuffle=True,
                                random_state=random_state + trial_seed_offset)

    def objective(trial):
        params = {
            "depth": trial.suggest_int("depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "iterations": trial.suggest_int("iterations", 100, 400),
        }
        model = CatBoostClassifier(verbose=0, random_state=random_state, **params)
        pipe = build_mice_pipeline(model, random_state=random_state)
        scores = cross_val_score(pipe, X_train_outer, y_train_outer, cv=inner_cv, scoring="roc_auc")
        return float(scores.mean())

    sampler = optuna.samplers.TPESampler(seed=random_state)  # SEEDED -> reproducible
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def run_nested_cv_catboost(X, y, n_outer_splits=OUTER_SPLITS, n_outer_repeats=OUTER_REPEATS,
                            n_inner_splits=INNER_SPLITS, n_trials=N_OPTUNA_TRIALS,
                            random_state=RANDOM_STATE, progress_callback=None):
    """
    The statistically valid replacement for the notebook's single-loop Optuna tuning.
    Returns: list of per-outer-fold AUC (unbiased), and the list of best_params chosen
    per fold (useful to see how stable the search is across folds).
    """
    outer_cv = RepeatedStratifiedKFold(n_splits=n_outer_splits, n_repeats=n_outer_repeats,
                                        random_state=random_state)
    outer_aucs, outer_f1s, fold_params = [], [], []

    total = n_outer_splits * n_outer_repeats
    for i, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        X_train_outer, X_test_outer = X.iloc[train_idx], X.iloc[test_idx]
        y_train_outer, y_test_outer = y.iloc[train_idx], y.iloc[test_idx]

        best_params = _inner_tune(X_train_outer, y_train_outer, n_trials, n_inner_splits,
                                   random_state, trial_seed_offset=i)

        model = CatBoostClassifier(verbose=0, random_state=random_state, **best_params)
        pipe = build_mice_pipeline(model, random_state=random_state)
        pipe.fit(X_train_outer, y_train_outer)

        proba = pipe.predict_proba(X_test_outer)[:, 1]
        pred = pipe.predict(X_test_outer)
        outer_aucs.append(roc_auc_score(y_test_outer, proba))
        outer_f1s.append(f1_score(y_test_outer, pred))
        fold_params.append(best_params)

        if progress_callback:
            progress_callback(i + 1, total)

    return {"auc": outer_aucs, "f1": outer_f1s, "fold_params": fold_params}


def fit_final_deployment_model(X, y, n_trials=N_OPTUNA_TRIALS, n_splits=5, random_state=RANDOM_STATE):
    """
    Builds ONE final model for downstream explainability/recourse/calibration artifacts.
    IMPORTANT: this model's own CV score during tuning is NOT a valid performance claim
    (it's tuned on all available data). The unbiased performance number for the paper
    comes from run_nested_cv_catboost() above. This function only produces a concrete,
    reproducible artifact to explain and to build the recourse system on.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def objective(trial):
        params = {
            "depth": trial.suggest_int("depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "iterations": trial.suggest_int("iterations", 100, 400),
        }
        model = CatBoostClassifier(verbose=0, random_state=random_state, **params)
        pipe = build_mice_pipeline(model, random_state=random_state)
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
        return float(scores.mean())

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    final_model = CatBoostClassifier(verbose=0, random_state=random_state, **study.best_params)
    final_pipe = build_mice_pipeline(final_model, random_state=random_state)
    final_pipe.fit(X, y)
    return final_pipe, study.best_params
