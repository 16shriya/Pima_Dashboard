"""
STEP 5 — CALIBRATION ANALYSIS
Fixes the original notebook's approach (train once on 80%, evaluate calibration on
the remaining 20% holdout -- valid but wastes data and gives a single noisy estimate
on ~154 patients). Instead, uses cross_val_predict to get genuine out-of-fold
probabilities for EVERY patient, using the leakage-safe MICE pipeline internally.
This uses all 768 patients for a more stable calibration curve / Brier score while
remaining fully out-of-fold (no patient's probability was produced by a model that
saw that patient during training).
"""
import numpy as np
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import brier_score_loss
from catboost import CatBoostClassifier

from .preprocessing import build_mice_pipeline
from .config import RANDOM_STATE


def out_of_fold_probabilities(X, y, params, n_splits=10, random_state=RANDOM_STATE):
    model = CatBoostClassifier(verbose=0, random_state=random_state, **params)
    pipe = build_mice_pipeline(model, random_state=random_state)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    return proba


def calibration_summary(y_true, y_proba, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.clip(np.digitize(y_proba, bins) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        rows.append({
            "bin_center": (bins[b] + bins[b + 1]) / 2,
            "mean_predicted": float(np.mean(y_proba[mask])),
            "fraction_positive": float(np.mean(np.array(y_true)[mask])),
            "n_samples": int(mask.sum()),
        })
    brier = float(brier_score_loss(y_true, y_proba))
    # Expected Calibration Error
    ece = sum((r["n_samples"] / len(y_proba)) * abs(r["fraction_positive"] - r["mean_predicted"])
              for r in rows)
    return rows, brier, ece
