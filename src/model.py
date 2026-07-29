"""
Model pipeline definitions.

We start with Logistic Regression (baseline, same as the starter notebook).
Leaves room for adding LightGBM/XGBoost as the next iteration.
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

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


def build_lightgbm_pipeline(class_weight=None):
    """
    LightGBM baseline, as an alternative to Logistic Regression.

    Source: LightGBM/XGBoost ensembles were the winning approach in both the
    Amex Default Prediction and Home Credit Default Risk competitions (see
    RESOURCES.md) -- same problem shape as this one (time-windowed customer
    features + financial risk target). Tree-based models don't need feature
    scaling, so no StandardScaler here (unlike build_logreg_pipeline).
    """
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        class_weight=class_weight,
        random_state=config.RANDOM_STATE,
        verbose=-1,
    )

def build_lightgbm_calibrated_pipeline(class_weight=None, method="sigmoid", cv=3):
    """
    LightGBM wrapped in probability calibration. 

    Source: Niculescu-Mizil & Caruana, "Predicting Good Probabilities with
    Supervised Learning" (ICML 2005) -- shows boosted trees push probability
    mass away from 0 and 1 (good ranking, poorly calibrated output), and
    that Platt scaling (sigmoid) corrects this. See RESOURCES.md. Log Loss
    carries 60% of this competition's score (src/evaluate.py), so this
    directly targets that weight without affecting ranking (ROC-AUC).
    """
    return CalibratedClassifierCV(
        estimator=build_lightgbm_pipeline(class_weight=class_weight),
        method=method,
        cv=cv,
    )
