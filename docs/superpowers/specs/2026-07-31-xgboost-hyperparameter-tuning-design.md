# Design: XGBoost hyperparameter tuning (random search)

Date: 2026-07-31
Status: Approved

## Goal

Tune XGBoost's `n_estimators`, `learning_rate`, and `max_depth` via random
search, as the next candidate after three rejected experiments this same
day (`class_weight="balanced"`, unweighted ensemble, weighted ensemble --
all documented in `README.md`'s Progress log and
`memory/project_next_session_class_weight.md`). The weighted-ensemble
round showed the combined score degrades monotonically as XGBoost's
weight in the blend increases, at every ratio tried -- consistent with
XGBoost (library defaults) being simply weaker than the tuned/calibrated
LightGBM on this dataset. Tuning XGBoost on its own merits might close
that gap enough to make ensembling worth revisiting later.

## Scope

Only `n_estimators`, `learning_rate`, `max_depth` -- the direct XGBoost
analogs of the 3 parameters already tuned for LightGBM
(`num_leaves` -> `max_depth`, since XGBoost grows trees depth-wise by
default, unlike LightGBM's leaf-wise growth). Does not add
`min_child_weight`, `gamma`, `subsample`, or `colsample_bytree` -- all
mentioned in XGBoost's own tuning docs but out of scope for this round
(user decision during brainstorming), matching how the LightGBM round
also limited itself to 3 parameters.

**This round evaluates XGBoost standalone**, not the ensemble: the
random search baseline is XGBoost with library defaults (calibrated,
same-run), not the production 0.20752 LightGBM baseline. Retesting
ensembling with the newly-tuned XGBoost is explicitly deferred to a
future round (user decision) -- this round's only wiring decision is
whether to save the winning hyperparameters to `config.py` for that
future round to consume.

## Sources

- **XGBoost official docs, "Notes on Parameter Tuning"**
  (https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html):
  verified via fetch on 2026-07-31. Identifies `max_depth`,
  `min_child_weight`, `gamma` as the primary model-complexity/overfitting
  controls, `subsample`/`colsample_bytree` as randomness-based
  regularization, and the `eta` (`learning_rate`) vs `num_round`
  (`n_estimators`) trade-off ("reduce stepsize eta ... increase num_round
  when you do so") -- the same trade-off already tuned for LightGBM. New
  addition to `RESOURCES.md`.
- **Bergstra & Bengio, "Random Search for Hyper-Parameter Optimization"**
  (JMLR 2012, already in `RESOURCES.md` from the LightGBM tuning round):
  justifies random over grid search for the same compute budget. Not
  re-added, just re-used.

## Design

### 1. New pipeline builder in `src/model.py`

```python
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
```

Kept separate from `build_ensemble_pipeline()` (which still hardcodes
XGBoost's library defaults internally) -- this round doesn't change the
ensemble function; a future round would update `build_ensemble_pipeline()`
to accept XGBoost hyperparameters if/when ensembling is revisited.

### 2. Random search loop (notebook, Step 11)

Same shape as the LightGBM round (`notebooks/01_eda_trends.ipynb` Step 7):
20 trials, seeded `numpy.random.RandomState(config.RANDOM_STATE)`, scored
with the existing `evaluate.cross_validate_score()` on the current
production feature set (baseline one-hot + trend features):

- `n_estimators`: sampled from `[100, 200, 300, 500, 800]` (same choices
  as the LightGBM search)
- `learning_rate`: log-uniform between 0.01 and 0.3 (same range as
  LightGBM -- multiplicative effect, so uniform-in-log-space)
- `max_depth`: sampled from `[3, 4, 5, 6, 8, 10]` -- centered on XGBoost's
  own default (6), extending both shallower and deeper

A same-run baseline (library defaults: `n_estimators=100,
learning_rate=0.3, max_depth=6`) is computed first, same pattern as
Step 7. No refined second-round search is planned upfront (unlike
Step 7's Step 7b) -- if the broad search suggests a promising region
worth narrowing in on, that can be a follow-up decision after seeing the
results, not committed to now.

### 3. Decision

If the best trial improves on (or holds, on both Log Loss and ROC-AUC)
the same-run XGBoost-defaults baseline, save the winning values as
`config.TUNED_XGBOOST_PARAMS` in `src/config.py` -- a candidate input for
a future ensembling-retest round, not wired into `src/train.py` (XGBoost
alone was never the production model). If it doesn't improve, document as
a negative result in `README.md`; no `config.py` addition.

## Out of scope (deferred)

- `min_child_weight`, `gamma`, `subsample`, `colsample_bytree`.
- A refined/narrowed second search round (only if the broad search's
  results suggest it's worth it -- not planned now).
- Retesting the LightGBM+XGBoost ensemble with the tuned XGBoost --
  separate future round, consuming `config.TUNED_XGBOOST_PARAMS` if this
  round adopts it.
- Early-stopping-based selection of `n_estimators` -- same reason as the
  LightGBM round: no eval-set infrastructure in the training pipeline yet.
