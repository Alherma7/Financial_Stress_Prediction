# Design: Weight the LightGBM + XGBoost ensemble toward LightGBM

Date: 2026-07-31
Status: Approved

## Goal

Test whether weighting the LightGBM + XGBoost soft-voting ensemble toward
LightGBM recovers the regression seen with an unweighted 50/50 average
(combined score 0.21489 ± 0.00330, worse than the single-LightGBM
baseline 0.20752 ± 0.00332 by ~2x the noise band -- see `README.md`
Progress log and
`docs/superpowers/specs/2026-07-31-lgbm-xgboost-ensemble-design.md`).
Hypothesis: the untuned XGBoost is weaker than the tuned/calibrated
LightGBM, so an unweighted average drags the result toward the weaker
model; weighting more heavily toward LightGBM should move the combined
score back toward (or past) the baseline.

## Scope

A small fixed grid of LightGBM/XGBoost weight ratios: 90/10, 80/20,
70/30, 60/40 -- four new points, evaluated with the existing
`evaluate.cross_validate_score()`. Compared against two points that are
**already known and not recomputed**: 100/0 (single tuned LightGBM,
0.20752 ± 0.00332, documented in `README.md`) and 50/50 (the unweighted
ensemble, 0.21489 ± 0.00330, documented in `README.md`) -- both were
already measured in prior rounds on the same feature set and CV setup, so
re-running them here would waste compute (each ensemble point trains two
calibrated models across 5 CV folds).

Not in scope: XGBoost hyperparameter tuning (still deferred, per the
ensemble design spec's "Out of scope"), a finer grid (5% steps) --
deferred unless this coarser grid shows a promising region worth
narrowing in on, and any weighting mechanism other than
`VotingClassifier`'s built-in `weights` parameter.

## Sources

No new sources needed. The scikit-learn Voting Classifier docs
(https://scikit-learn.org/stable/modules/ensemble.html#voting-classifier),
already added to `RESOURCES.md` in the previous round, include the
weighted-average formula and the `weights` parameter example -- this
round only exercises a parameter that source already documents.

## Design

### 1. Extend `model.build_ensemble_pipeline()`

Add a `weights=None` parameter, passed straight through to
`VotingClassifier`. Default unchanged (`None` = equal weights), so the
existing signature and any prior calls keep working:

```python
def build_ensemble_pipeline(class_weight=None, method="sigmoid", cv=3, weights=None):
    ...
    return VotingClassifier(estimators=[("lgbm", lgbm), ("xgb", xgb)], voting="soft", weights=weights)
```

### 2. Notebook grid search (Step 10)

In `notebooks/01_eda_trends.ipynb`, a new step loops over
`lgbm_weight in [0.9, 0.8, 0.7, 0.6]`, calling
`evaluate.cross_validate_score()` on
`model.build_ensemble_pipeline(weights=[lgbm_weight, 1 - lgbm_weight])`
for each. Results are collected into the same kind of results table used
in Step 7's hyperparameter search, seeded with two rows for the
already-documented endpoints (100/0 -> 0.20752 ± 0.00332; 50/50 ->
0.21489 ± 0.00330) so the full 0.5-1.0 range is visible without
recomputing those two.

### 3. Validation and wiring gate

Same gate as every previous round: take the best point across all six
rows (four new + two documented) and compare it to the 0.20752 ± 0.00332
baseline. Only wire into `src/train.py` if it improves or at least holds
on **both** Log Loss and ROC-AUC.

- **If it passes:** `src/train.py`'s two pipeline-builder call sites
  switch to `model.build_ensemble_pipeline(weights=[<best_lgbm_weight>, <best_xgb_weight>])`,
  and `config.py` gets a `TUNED_ENSEMBLE_WEIGHTS` constant with the
  winning ratio.
- **If it doesn't pass (i.e. the 100/0 point -- plain LightGBM -- remains
  best):** document as a negative result. This would mean ensembling with
  an untuned XGBoost doesn't help at any weight tried, and the remaining
  untried options become: tuning XGBoost before ensembling, or feature
  engineering.

## Out of scope (deferred)

- XGBoost hyperparameter tuning.
- A finer weight grid (5% steps).
- Non-linear or learned blending (stacking) -- still deferred per the
  original ensemble design spec.
