"""
PLOTTING HELPERS
Fixes the original notebook's bug where every plt.savefig() ran in a cell AFTER
plt.show() had already closed the figure, producing blank PNGs (verified by
reproduction). The fix: every function below saves the figure to disk BEFORE
calling plt.show(), inside the same function call, so there is no cell boundary
between "figure exists" and "figure is saved."
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

from .config import FIGURES_DIR


def _save_and_show(fig, filename, dpi=300, close=True):
    path = Path(FIGURES_DIR) / filename
    fig.savefig(path, dpi=dpi, bbox_inches="tight")  # SAVE FIRST
    plt.show()                                        # THEN show (no-op outside a notebook/inline backend)
    if close:
        plt.close(fig)
    return path, fig


def plot_histograms(df, filename="feature_histograms.png", close=True):
    fig = df.hist(figsize=(15, 10))[0][0].figure
    plt.tight_layout()
    return _save_and_show(fig, filename, close=close)


def plot_correlation_matrix(df_clean, filename="correlation_matrix.png", close=True):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df_clean.drop(columns=["Outcome"]).corr(), annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Feature Correlation Matrix")
    return _save_and_show(fig, filename, close=close)


def plot_outcome_distribution(df, filename="outcome_distribution.png", close=True):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.countplot(x=df["Outcome"], ax=ax)
    ax.set_title("Diabetes Outcome Distribution")
    return _save_and_show(fig, filename, close=close)


def plot_calibration_curve(bin_df, brier, ece, filename="calibration_curve.png", close=True):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
    ax.plot(bin_df["mean_predicted"], bin_df["fraction_positive"], marker="o", label="Model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"Calibration Curve (Brier={brier:.4f}, ECE={ece:.4f})")
    ax.legend()
    return _save_and_show(fig, filename, close=close)


def plot_shap_summary(shap_values, X_display, filename="shap_summary.png", close=True):
    import shap
    fig = plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_display, feature_names=X_display.columns, show=False)
    fig = plt.gcf()
    return _save_and_show(fig, filename, close=close)


def plot_shap_bar(shap_values, X_display, filename="shap_bar.png", close=True):
    import shap
    fig = plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_display, feature_names=X_display.columns,
                       plot_type="bar", show=False)
    fig = plt.gcf()
    return _save_and_show(fig, filename, close=close)


def plot_shap_dependence(feature, shap_values, X_display, filename=None, close=True):
    import shap
    filename = filename or f"shap_dependence_{feature}.png"
    fig = plt.figure(figsize=(8, 6))
    shap.dependence_plot(feature, shap_values, X_display, feature_names=X_display.columns, show=False)
    fig = plt.gcf()
    return _save_and_show(fig, filename, close=close)


def plot_recourse_comparison(original_prob, independent_prob, causal_prob,
                              filename="recourse_comparison.png", close=True):
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = ["Original", "Independent\nrecourse", "Causal\nrecourse"]
    values = [original_prob, independent_prob, causal_prob]
    ax.bar(labels, values, color=["#888888", "#e74c3c", "#2ecc71"])
    ax.set_ylabel("Predicted Diabetes Risk")
    ax.set_title("Risk Reduction: Independent vs. Causal-Aware Recourse")
    ax.set_ylim(0, 1)
    return _save_and_show(fig, filename, close=close)
