# Design: LightGBM regularization tuning (fix confirmed overfitting)

Date: 2026-07-31
Status: Documented, implementation deferred to next session

## Goal

Close the overfitting confirmed in `notebooks/01_eda_trends.ipynb` Step 12:
mean train combined score **0.07672** vs. mean val **0.20752**, gap +0.125
to +0.136, consistent across all 5 CV folds (not noise) -- without
regressing the validation combined score currently documented for
`config.TUNED_LGBM_PARAMS` (**0.20752 ± 0.00332**).

## Scope

Only the regularization knobs the Step 7 tuning round explicitly left out
(see
`docs/superpowers/specs/2026-07-30-lightgbm-hyperparameter-tuning-design.md`
"Out of scope"): `min_data_in_leaf`, `feature_fraction`,
`bagging_fraction` + `bagging_freq`, `lambda_l1`, `lambda_l2`, and
optionally capping `max_depth`. `n_estimators`/`learning_rate`/`num_leaves`
stay fixed at `config.TUNED_LGBM_PARAMS`'s already-adopted values
(600 / 0.041141 / 31) -- one round at a time, same discipline as every
prior tuning round in this project.

## Sources

- **LightGBM official docs, "Parameters Tuning"** (see `RESOURCES.md`) --
  its "Deal with over-fitting" section names exactly these knobs as the
  overfitting controls (as opposed to the `n_estimators`/`learning_rate`/
  `num_leaves` trio Step 7 already tuned).
- **Bergstra & Bengio, JMLR 2012** (see `RESOURCES.md`) -- random search
  over this multi-parameter space, same justification already used for
  Step 7 (LightGBM) and Step 11 (XGBoost).
- **Abhishek Thakur, "Approaching (Almost) Any Machine Learning Problem"**
  (see `RESOURCES.md`) -- train-vs-val gap as the overfitting diagnostic;
  already implemented in Step 12 and reused here as a second gate.

## Design

1. Notebook: new "Step 13" random search (`N_TRIALS`, e.g. 20), sampling:
   - `min_data_in_leaf`: e.g. `[10, 20, 50, 100, 200]`
   - `feature_fraction`: uniform(0.5, 1.0)
   - `bagging_fraction`: uniform(0.5, 1.0), with `bagging_freq` fixed
     (e.g. 1 or 5) whenever `bagging_fraction < 1`
   - `lambda_l1`, `lambda_l2`: log-uniform(1e-3, 10), including 0 as a
     possible value (no penalty)

   `n_estimators`/`learning_rate`/`num_leaves` fixed at
   `config.TUNED_LGBM_PARAMS` for every trial.

2. For each trial, reuse `evaluate.cross_validate_score()` for the
   validation combined score/Log Loss/ROC-AUC (same 5 folds as always),
   **and** reuse Step 12's per-fold train-vs-val loop to check whether
   the trial actually shrinks the gap, not just produces a different
   validation number by luck.

3. Same-run baseline: `config.TUNED_LGBM_PARAMS` with library defaults for
   these new knobs (i.e. the current Step 7/12 result), computed in the
   same run -- not the stale README number, per this project's standing
   rule.

4. Gate (same spirit as every prior round, extended with the gap check):
   adopt into `config.TUNED_LGBM_PARAMS` only if the trial:
   - improves or holds the CV combined score on **both** Log Loss and
     ROC-AUC, **and**
   - meaningfully reduces the train-val gap (more than one fold's noise
     band; a trial that closes the gap by validating worse is not a
     win, and a trial with an unchanged gap means regularization isn't
     the right lever here -- revisit feature engineering or more data
     instead).

5. If adopted: extend `config.TUNED_LGBM_PARAMS` with the winning values,
   log the before/after numbers in README.md. If it doesn't clear the
   gate: log the negative/neutral result in README.md exactly like the
   four prior rejected rounds (`class_weight`, both ensembles, standalone
   XGBoost tuning) -- code stays committed but unused.

## Out of scope (deferred)

- Re-tuning `n_estimators`/`learning_rate`/`num_leaves` jointly with these
  new knobs (blows up the search space; revisit only if this round's best
  trial plateaus short of closing the gap).
- Early-stopping via a held-out eval set (needs pipeline changes beyond
  this round, per Step 7's original scope note).
- Feature reduction/selection as an alternative overfitting fix (untried
  candidate; a separate round if regularization alone doesn't close
  enough of the gap).

## Status

Documented 2026-07-31, end of session -- **not yet implemented**. Next
session should start here: draft the Step 13 notebook cell (random
search + reused Step 12 gap check) as a chat code block, per this
project's workflow (notebook experiment code is pasted/run by the user;
Claude wires the result into `config.py`/`README.md` once the outcome is
known).
