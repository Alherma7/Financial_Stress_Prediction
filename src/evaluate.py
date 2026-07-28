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
