"""
Model evaluation using the competition's actual metric:
    score = 0.6 * log_loss + 0.4 * (1 - roc_auc)

Note: Zindi's leaderboard typically reports a score where LOWER is better
for log loss and HIGHER is better for AUC. Here we build a combined score
where lower = better, so experiments can be compared consistently.
Always check the competition's "Evaluation" page for exactly how the two
metrics are combined before relying on this number to make decisions.
"""

from sklearn.metrics import log_loss, roc_auc_score

from . import config

import numpy as np
from sklearn.model_selection import StratifiedKFold


def combined_score(y_true, y_pred_prob) -> dict:
    """
    Compute log loss, AUC, and a weighted combined score.
    Returns a dict with all three metrics for easy logging.
    """
    ll = log_loss(y_true, y_pred_prob)
    auc = roc_auc_score(y_true, y_pred_prob)

    # Combined score: lower is better (low log loss + low (1-auc))
    score = config.LOGLOSS_WEIGHT * ll + config.AUC_WEIGHT * (1 - auc)

    return {
        "log_loss": ll,
        "roc_auc": auc,
        "combined_score": score,
    }


def print_scores(scores: dict, label: str = "Validation"):
    print(f"--- {label} ---")
    print(f"Log Loss      : {scores['log_loss']:.5f}")
    print(f"ROC-AUC       : {scores['roc_auc']:.5f}")
    print(f"Combined score: {scores['combined_score']:.5f} (lower is better)")

def cross_validate_score(pipeline_builder, X, y, n_splits=5, random_state=None) -> dict:
    """
    Stratified K-Fold cross-validation of the competition's combined score. 

    Source: the Home Credit Default Risk 1st place solution used stratified
    K-fold CV to get a robust score estimate before ensembling (see
    RESOURCES.md) -- more reliable than a single train/val split, especially
    with an imbalanced target like this dataset's (~15% positive rate).

    pipeline_builder must be a zero-argument callable returning a fresh,
    unfitted pipeline (e.g. `lambda: model.build_lightgbm_pipeline()`), so
    each fold trains its own model from scratch.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        pipeline = pipeline_builder()
        pipeline.fit(X_train, y_train)
        val_pred_prob = pipeline.predict_proba(X_val)[:, 1]

        scores = combined_score(y_val, val_pred_prob)
        print_scores(scores, label=f"Fold {fold}/{n_splits}")
        fold_scores.append(scores)
    summary = {}
    for key in ("log_loss", "roc_auc", "combined_score"):
        values = [s[key] for s in fold_scores]
        summary[f"{key}_mean"] = np.mean(values)
        summary[f"{key}_std"] = np.std(values)
    return summary

def print_cv_summary(summary: dict):
    print("--- Cross-validation summary ---")
    print(f"Log Loss      : {summary['log_loss_mean']:.5f} +/- {summary['log_loss_std']:.5f}")
    print(f"ROC-AUC       : {summary['roc_auc_mean']:.5f} +/- {summary['roc_auc_std']:.5f}")
    print(f"Combined score: {summary['combined_score_mean']:.5f} +/- {summary['combined_score_std']:.5f} (lower is better)")