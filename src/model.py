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


def build_lightgbm_pipeline(class_weight=None, n_estimators=100, learning_rate=0.1, num_leaves=31,
                             min_data_in_leaf=20, feature_fraction=1.0, bagging_fraction=1.0,
                             bagging_freq=0, lambda_l1=0.0, lambda_l2=0.0):
    """
    LightGBM baseline, as an alternative to Logistic Regression.

    Source: LightGBM/XGBoost ensembles were the winning approach in both the
    Amex Default Prediction and Home Credit Default Risk competitions (see
    RESOURCES.md) -- same problem shape as this one (time-windowed customer
    features + financial risk target). Tree-based models don't need feature
    scaling, so no StandardScaler here (unlike build_logreg_pipeline).

    n_estimators/learning_rate/num_leaves default to LightGBM's own library
    defaults. See config.TUBED_LGBM_PARAMS for the values chosen by random
    search (LightGBM docs "Parameters Tuning"; Bergstra & Bengio, JMLR 2012
    -- see RESOURCES.md).

    min_data_in_leaf/feature_fraction/bagging_fraction/bagging_freq/
    lambda_l1/lambda_l2 are LightGBM's overfitting-control knobs (LightGBM
    docs "Parameters Tuning", "Deal with over-fitting" section -- see
    RESOURCES.md), left at their library defaults here so existing callers
    are unaffected; see the Step 13 regularization search in
    notebooks/01_eda_trends.ipynb for tuned values.
    """
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        class_weight=class_weight,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_data_in_leaf=min_data_in_leaf,
        feature_fraction=feature_fraction,
        bagging_fraction=bagging_fraction,
        bagging_freq=bagging_freq,
        lambda_l1=lambda_l1,
        lambda_l2=lambda_l2,
        random_state=config.RANDOM_STATE,
        verbose=-1,
    )

def build_lightgbm_calibrated_pipeline(class_weight=None, method="sigmoid", cv=3,
                                       n_estimators=100, learning_rate=0.1, num_leaves=31,
                                       min_data_in_leaf=20, feature_fraction=1.0,
                                       bagging_fraction=1.0, bagging_freq=0,
                                       lambda_l1=0.0, lambda_l2=0.0):
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
        estimator=build_lightgbm_pipeline(
            class_weight=class_weight,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            min_data_in_leaf=min_data_in_leaf,
            feature_fraction=feature_fraction,
            bagging_fraction=bagging_fraction,
            bagging_freq=bagging_freq,
            lambda_l1=lambda_l1,
            lambda_l2=lambda_l2,
            ),
        method=method,
        cv=cv,
    )


def build_ensemble_pipeline(class_weight=None, method="sigmoid", cv=3, weights=None):
    """
    Simple soft-voting average of calibrated LightGBM + calibrated XGBoost.

    Source: LightGBM/XGBoost ensembles were the winning approach in the
    Amex Default Prediction and Home Credit Default Risk competitions (see
    RESOURCES.md). XGBoost is calibrated the same way as LightGBM
    (Niculescu-Mizil & Caruana, ICML 2005 -- boosted trees in general push
    probability mass away from 0/1) so both inputs to the vote are
    comparably calibrated. VotingClassifier(voting="soft", weights=None)
    computes a plain average of predict_proba across estimators (sklearn
    docs -- Voting Classifier, RESOURCES.md) -- matches the "simple
    average of probabilities" approach decided for this round.

    An optional `weights=[lgbm_weight, xgb_weight]` shifts the average
    toward one model -- see config.TUNED_ENSEMBLE_WEIGHTS (if adopted) or
    the negative-result note in README.md for the grid search that
    validated this.

    XGBoost uses library defaults (no tuning yet) -- consistent with how
    LightGBM was first evaluated before a separate tuning round.
    """
    from xgboost import XGBClassifier
    from sklearn.ensemble import VotingClassifier

    lgbm = build_lightgbm_calibrated_pipeline(
        class_weight=class_weight, method=method, cv=cv, **config.TUNED_LGBM_PARAMS
    )
    xgb = CalibratedClassifierCV(
        estimator=XGBClassifier(random_state=config.RANDOM_STATE),
        method=method,
        cv=cv,
    )
    return VotingClassifier(estimators=[("lgbm", lgbm), ("xgb", xgb)], voting="soft", weights=weights)


def build_xgboost_calibrated_pipeline(method="sigmoid", cv=3,
                                       n_estimators=100, learning_rate=0.3, max_depth=6):
    """
    Calibrated XGBoost, evaluated standalone (not as part of the ensemble).

    Source: XGBoost official "Notes on Parameter Tuning" docs (RESOURCES.md)
    -- n_estimators/learning_rate/max_depth are the primary tuning knobs,
    analogous to LightGBM's n_estimators/learning_rate/num_leaves. Calibrated
    the same way as LightGBM (Niculescu-Mizil & Caruana, ICML 2005) so this
    round's baseline and trials are all comparably calibrated.

    Defaults (n_estimators=100, learning_rate=0.3, max_depth=6) are
    XGBoost's own library defaults, so existing callers keep working
    unchanged if these aren't passed. See config.TUNED_XGBOOST_PARAMS (if
    adopted) for the random search's winning values.
    """
    from xgboost import XGBClassifier

    return CalibratedClassifierCV(
        estimator=XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=config.RANDOM_STATE,
        ),
        method=method,
        cv=cv,
    )
