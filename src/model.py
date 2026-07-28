"""
Model pipeline definitions.

We start with Logistic Regression (baseline, same as the starter notebook).
Leaves room for adding LightGBM/XGBoost as the next iteration.
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from . import config


def build_logreg_pipeline(class_weight=None):
    """
    Baseline Logistic Regression pipeline.

    Note on class_weight: using "balanced" can help AUC but distorts
    the output probabilities, which hurts Log Loss. Since the competition
    metric weights Log Loss at 60%, try both variants (None vs "balanced")
    and compare the combined score, not just AUC.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight=class_weight,
            solver="liblinear",
            random_state=config.RANDOM_STATE,
        )),
    ])


# --- Placeholder for the next model to try ---
# def build_lightgbm_pipeline():
#     from lightgbm import LGBMClassifier
#     return LGBMClassifier(random_state=config.RANDOM_STATE)
