# Design: LightGBM hyperparameter tuning (random search)

Date: 2026-07-30
Status: Approved

## Goal

Complete the remaining "Tune LightGBM hyperparameters (`n_estimators`,
`learning_rate`, `num_leaves`) instead of the library defaults" item in
`README.md`'s Next steps, targeting the competition's combined score
(0.6 x Log Loss + 0.4 x ROC-AUC).

## Scope

Only the 3 hyperparameters already named in the README backlog item:
`n_estimators`, `learning_rate`, `num_leaves`. Does not add `max_depth`,
`min_data_in_leaf`, or other LightGBM parameters -- that would be scope
creep beyond the approved backlog item, and can be a future round.

## Sources

- **LightGBM official docs, "Parameters Tuning"**
  (https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html):
  `num_leaves` should stay below `2^max_depth`; `learning_rate` and
  `num_iterations` (`n_estimators`) trade off against each other (lower
  learning rate needs more iterations); early stopping with a validation
  set is the recommended way to choose `num_iterations`. This project
  does not have early-stopping infrastructure yet, so `n_estimators` is
  tuned directly as a hyperparameter instead, per the backlog item's
  explicit request.
- **Bergstra & Bengio, "Random Search for Hyper-Parameter Optimization"**
  (JMLR 2012, https://www.jmlr.org/papers/v13/bergstra12a.html): random
  search finds equally good or better hyperparameters than grid search
  for the same compute budget, because only a subset of hyperparameters
  matters per dataset and that subset varies -- justifies random sampling
  over an exhaustive grid for this 3-parameter search.

Both sources are new additions to `RESOURCES.md` (not previously cited
there) and will be added as part of implementation.

## Design

### 1. Extend `src/model.py`

`build_lightgbm_pipeline()` and `build_lightgbm_calibrated_pipeline()` gain
three new keyword arguments: `n_estimators=100, learning_rate=0.1,
num_leaves=31` -- these defaults are exactly LightGBM's own library
defaults, so existing callers (including `src/train.py`) keep working
unchanged if the new arguments aren't passed.

### 2. Random search loop (notebook)

In `notebooks/01_eda_trends.ipynb`, a new step samples `N_TRIALS` (default
20) hyperparameter combinations using a seeded `numpy.random.RandomState`
(seed = `config.RANDOM_STATE`, for reproducibility):

- `n_estimators`: sampled from `[100, 200, 300, 500, 800]`
- `learning_rate`: log-uniform between 0.01 and 0.3 (multiplicative effect,
  so uniform-in-log-space avoids over-sampling large values)
- `num_leaves`: sampled from `[15, 31, 63, 127, 255]`

Each combination is scored with the existing
`evaluate.cross_validate_score()`, using
`model.build_lightgbm_calibrated_pipeline()` directly (the actual
production pipeline) so the search objective matches the real competition
metric, not an uncalibrated proxy. With ~40k rows this is fast enough for
20 trials; if it proves too slow in practice, `N_TRIALS` can be lowered,
or the search can run on the uncalibrated pipeline
(`build_lightgbm_pipeline`) with calibration applied only to the final
chosen hyperparameters.

Results (params + `combined_score_mean/std`, `log_loss_mean/std`,
`roc_auc_mean/std` per trial) are collected into a list of dicts.

### 3. Visualize search results

Convert the trial results to a `pandas.DataFrame`. Produce 3 scatter plots
(one per hyperparameter vs. `combined_score_mean`, with `combined_score_std`
as error bars), highlighting the best trial in a different color/marker.
Since this is random search (not an exhaustive grid), these show the noisy
marginal relationship between each hyperparameter and the score rather
than a clean elbow curve -- but still make it possible to eyeball whether
the score plateaus or worsens in some region.

### 4. Config

`config.TUNED_LGBM_PARAMS = {"n_estimators": ..., "learning_rate": ...,
"num_leaves": ...}` -- filled in with the best trial's actual values after
running the search (not before; no placeholder values in committed code).

### 5. Validation and wiring

Same gate as the previous feature-engineering round: compare the best
trial's combined score against the documented baseline (0.21333 ± 0.00420
if the previous trend-feature round wasn't wired in, since it wasn't
adopted -- see `README.md` Progress section). Only wire
`config.TUNED_LGBM_PARAMS` into `src/train.py` if it improves or at least
does not regress on both Log Loss and ROC-AUC; otherwise, document the
negative/neutral result in `README.md` and leave the library defaults in
place, same as the prior round.

## Out of scope (deferred)

- Tuning additional LightGBM parameters (`max_depth`, `min_data_in_leaf`,
  regularization terms).
- Early-stopping-based selection of `n_estimators` (would require adding
  eval-set support to the training pipeline -- separate future item).
- Bayesian/TPE search (e.g. Optuna) -- would add a new dependency; random
  search reuses only already-installed libraries (numpy, the project's own
  `evaluate.cross_validate_score()`).
