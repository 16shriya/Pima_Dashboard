"""
CENTRAL CONFIG — single source of truth for every random seed, path, and constant.
Every module imports from here. This is what makes the whole pipeline reproducible:
change RANDOM_STATE once, rerun, and every downstream number is deterministic.
"""
from pathlib import Path

RANDOM_STATE = 42

# Paths (relative to project root)
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "pima_diabetes.csv"
FIGURES_DIR = ROOT_DIR / "results" / "figures"
TABLES_DIR = ROOT_DIR / "results" / "tables"
MODELS_DIR = ROOT_DIR / "results" / "models"

for d in [FIGURES_DIR, TABLES_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Columns where 0 is a missing-value sentinel, not a true physiological zero
ZERO_INVALID_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
IMMUTABLE_FEATURES = ["Age", "DiabetesPedigreeFunction", "Pregnancies"]
TARGET_COL = "Outcome"

# Nested CV settings (defaults chosen for a reasonable runtime on 768 rows;
# scale up N_OPTUNA_TRIALS and OUTER_SPLITS for a final production run)
OUTER_SPLITS = 5
OUTER_REPEATS = 2
INNER_SPLITS = 3
N_OPTUNA_TRIALS = 20

# Feasible physiological bounds for counterfactual recourse search
FEASIBLE_BOUNDS = {
    "Glucose": (70, 200),
    "BMI": (18.5, 45),
}

CAUSAL_EDGES = {
    # child_feature: [parent_features] -- derived from the correlation audit
    "BloodPressure": ["BMI"],
    "Insulin": ["BMI", "Glucose"],
    "SkinThickness": ["BMI"],
}
