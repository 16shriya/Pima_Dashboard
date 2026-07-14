"""
STEP 7 — SUBGROUP FAIRNESS ANALYSIS
Stratifies model performance (AUC, calibration/Brier) by Age tertiles and
Pregnancies groups -- the demographic proxies available in this dataset.
Directly answers Reviewer #3's fairness-analysis request.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss


def add_subgroups(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Age_group"] = pd.qcut(out["Age"], q=3, labels=["Younger", "Middle", "Older"])
    out["Pregnancy_group"] = pd.cut(
        out["Pregnancies"], bins=[-1, 0, 3, 100], labels=["0", "1-3", "4+"]
    )
    return out


def subgroup_performance(df_with_groups: pd.DataFrame, y_proba: np.ndarray, group_col: str) -> pd.DataFrame:
    df = df_with_groups.copy()
    df["_proba"] = y_proba
    rows = []
    for grp, sub in df.groupby(group_col, observed=True):
        if sub["Outcome"].nunique() < 2:
            auc = np.nan
        else:
            auc = roc_auc_score(sub["Outcome"], sub["_proba"])
        brier = brier_score_loss(sub["Outcome"], sub["_proba"])
        rows.append({
            "Subgroup": grp,
            "N": len(sub),
            "ROC-AUC": round(auc, 3) if not np.isnan(auc) else "n/a (single class)",
            "Brier Score": round(brier, 3),
            "Positive rate": round(sub["Outcome"].mean(), 3),
        })
    return pd.DataFrame(rows)
