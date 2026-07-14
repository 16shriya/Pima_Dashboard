"""
STATISTICAL SIGNIFICANCE TESTING
Paired Wilcoxon signed-rank test -- appropriate for small paired per-fold samples
where normality can't be safely assumed. Pairing requires the SAME fold scheme for
both models being compared (this module assumes that; the pipeline scripts enforce it
by using the same RepeatedStratifiedKFold random_state for both baseline and nested CV).
"""
import numpy as np
from scipy.stats import wilcoxon, shapiro


def paired_significance_test(scores_a, scores_b, alpha=0.05):
    """
    scores_a, scores_b: equal-length arrays of paired per-fold metric values
    (e.g. baseline_auc[i] and optimized_auc[i] must come from the SAME fold i).
    """
    a = np.array(scores_a)
    b = np.array(scores_b)
    assert len(a) == len(b), "Paired test requires equal-length, fold-aligned arrays."

    diff = b - a
    # Report normality check for transparency (informs whether Wilcoxon vs paired t-test
    # is the more appropriate choice) -- Wilcoxon is used regardless since n is small.
    if len(diff) >= 3 and np.std(diff) > 0:
        _, shapiro_p = shapiro(diff)
    else:
        shapiro_p = None

    if np.all(diff == 0):
        return {"statistic": None, "p_value": 1.0, "mean_diff": 0.0,
                "significant": False, "shapiro_p_on_diffs": shapiro_p}

    stat, p = wilcoxon(a, b)
    return {
        "statistic": float(stat),
        "p_value": float(p),
        "mean_diff": float(np.mean(diff)),
        "std_diff": float(np.std(diff)),
        "significant": bool(p < alpha),
        "shapiro_p_on_diffs": shapiro_p,
        "n_pairs": len(a),
    }


def confidence_interval(scores, confidence=0.95):
    """Normal-approximation CI for a metric across CV folds. For n<30 this is an
    approximation; report alongside the raw fold values, not as a substitute for them."""
    from scipy import stats as st
    a = np.array(scores)
    mean = a.mean()
    sem = st.sem(a)
    interval = sem * st.t.ppf((1 + confidence) / 2, len(a) - 1)
    return {"mean": float(mean), "ci_low": float(mean - interval), "ci_high": float(mean + interval)}
