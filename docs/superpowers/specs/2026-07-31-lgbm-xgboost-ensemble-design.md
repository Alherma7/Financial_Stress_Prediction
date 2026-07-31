# Design: LightGBM + XGBoost ensemble (soft voting)

Date: 2026-07-31
Status: Approved

## Goal

Test a simple probability-average ensemble of the tuned/calibrated
LightGBM pipeline and a calibrated XGBoost model, targeting the
competition's combined score (0.6 x Log Loss + 0.4 x ROC-AUC). This is
the next candidate in the project's backlog after `class_weight="balanced"`
was tried and rejected (see `README.md` Progress log and
`memory/project_next_session_class_weight.md`) -- that result established
the model already handles the dataset's 85/15 class imbalance well
without extra correction.

## Scope

Only a simple (unweighted) soft-voting average of two models'
`predict_proba` output, via `sklearn.ensemble.VotingClassifier(voting="soft")`.
Both underlying models are calibrated with `CalibratedClassifierCV`
before voting, for the same reason LightGBM alone was calibrated (see
`RESOURCES.md`, Niculescu-Mizil & Caruana). XGBoost is tried with its
library defaults, not tuned -- consistent with how LightGBM was first
evaluated (defaults) before a separate tuning round; per this project's
practice (Home Credit lesson: feature engineering/ensembling matters
more than tuning), tuning XGBoost is deferred unless this first pass
shows promise.

Not in scope (deferred, user decision during brainstorming):
- Weighted averaging (searching for an optimal weight between the two
  models) -- only plain unweighted averaging is tested this round.
- Stacking (a meta-model over out-of-fold predictions) -- more complex,
  higher overfitting risk on this dataset's size (40k rows), and not
  needed to answer "does ensembling help at all" first.
- Tuning XGBoost's own hyperparameters.

## Sources

- **American Express -- Default Prediction / Home Credit Default Risk
  winning solutions** (already in `RESOURCES.md`): LightGBM/XGBoost/CatBoost
  ensembles were the winning approach in both competitions -- same problem
  shape as this one. Justifies trying an ensemble at all.
- **Niculescu-Mizil & Caruana, "Predicting Good Probabilities with
  Supervised Learning"** (ICML 2005, already in `RESOURCES.md`): shows
  boosted trees in general -- not specifically LightGBM -- push probability
  mass away from 0 and 1. Applies equally to XGBoost, hence calibrating
  it too before averaging, so both inputs to the vote are comparably
  well-calibrated probabilities.
- **scikit-learn docs -- Voting Classifier**
  (https://scikit-learn.org/stable/modules/ensemble.html#voting-classifier):
  verified via fetch on 2026-07-31. Confirms `voting="soft"` with
  `weights=None` (default) computes the plain average of each
  classifier's predicted probabilities (argmax of the summed/averaged
  probabilities) -- the exact "simple average of probabilities" mechanism
  decided during brainstorming. New addition to `RESOURCES.md`.

## Design

### 1. Install XGBoost

`xgboost` is listed in `requirements.txt` but is **not actually installed**
in the current environment (verified via `pip show xgboost` -- not found).
The user runs `pip install xgboost` before the notebook experiment; this
is the only step in this round that requires an internet connection.

### 2. New pipeline builder in `src/model.py`

```python
def build_ensemble_pipeline(class_weight=None, method="sigmoid", cv=3):
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
    return VotingClassifier(estimators=[("lgbm", lgbm), ("xgb", xgb)], voting="soft")
```

Reuses `build_lightgbm_calibrated_pipeline()` rather than duplicating its
LightGBM+calibration logic (DRY). `class_weight` defaults to `None` since
that's the established best setting (see the rejected `"balanced"`
experiment); the parameter is kept so a future experiment could still
override it without changing this function.

### 3. Notebook experiment (Step 9)

In `notebooks/01_eda_trends.ipynb`, a new step following the Step 8
pattern: run `evaluate.cross_validate_score()` on
`model.build_ensemble_pipeline()` over the current production feature
set, and print `combined_score_mean/std`, `log_loss_mean/std`,
`roc_auc_mean/std` next to the documented baseline (0.20752 ± 0.00332,
single tuned LightGBM, `class_weight=None`).

Handed to the user as a chat code block for manual pasting (notebook
experiment code stays hands-on per
`memory/feedback_manual_code_insertion.md`); everything else in this
round (`RESOURCES.md`, `README.md`, `src/model.py`, `src/config.py`,
`src/train.py`, git commits) is edited directly by Claude once the
result is known.

### 4. Validation and wiring gate

Same gate as every previous round: only wire `build_ensemble_pipeline()`
into `src/train.py` if the combined score improves or at least holds on
**both** Log Loss and ROC-AUC vs. the 0.20752 ± 0.00332 baseline.

- **If it passes:** `src/train.py` switches its two
  `model.build_lightgbm_calibrated_pipeline(...)` call sites to
  `model.build_ensemble_pipeline()`.
- **If it doesn't pass:** document as a negative result in `README.md`'s
  Progress log (same style as the `class_weight="balanced"` and
  min/max/net-flow entries), leave `src/train.py` unchanged, and note
  feature engineering as the next candidate (the only one left in the
  backlog after this).

## Out of scope (deferred)

- Weighted averaging / weight search.
- Stacking with a meta-model.
- Tuning XGBoost's hyperparameters.
- Adding CatBoost (mentioned in `RESOURCES.md`'s competition writeups but
  not an existing dependency -- would need a new install decision of its
  own, separate from this round which only adds the already-listed
  `xgboost`).
