# XGBoost Hyperparameter Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tune XGBoost's `n_estimators`, `learning_rate`, and `max_depth` via random search, evaluated standalone (calibrated XGBoost alone, not the ensemble), and save the winning hyperparameters to `config.py` if they improve on a same-run XGBoost-defaults baseline — as an input for a future ensembling-retest round, not wired into `src/train.py` this round.

**Architecture:** `src/model.py` gains one new function, `build_xgboost_calibrated_pipeline()`, calibrated the same way as LightGBM. The notebook gets a new self-contained cell (Step 11) that runs a 20-trial random search, same shape as the LightGBM tuning round's Step 7.

**Tech Stack:** Python, xgboost, scikit-learn (`CalibratedClassifierCV`), numpy, pandas. No new dependencies. No unit test framework — verification is a synthetic-data smoke check plus the notebook's real cross-validated comparison.

## Global Constraints

- **Manual code insertion** (`memory/feedback_manual_code_insertion.md`): only notebook cells (`notebooks/01_eda_trends.ipynb`) are handed to the user as chat text for pasting/running. Everything else — `src/model.py`, `src/config.py`, `README.md`, `RESOURCES.md`, and all git commits (including the notebook's, once the user confirms they ran it) — is edited and committed directly by Claude.
- **Scope**: only `n_estimators`, `learning_rate`, `max_depth` — no `min_child_weight`, `gamma`, `subsample`, `colsample_bytree`, no refined second search round planned upfront. See `docs/superpowers/specs/2026-07-31-xgboost-hyperparameter-tuning-design.md` "Out of scope".
- **This round evaluates XGBoost standalone**: the comparison baseline is XGBoost with library defaults (calibrated, same-run), not the production 0.20752 LightGBM baseline. No `src/train.py` changes in this plan — XGBoost alone was never the production model.
- **Gate**: only add `config.TUNED_XGBOOST_PARAMS` if the best trial improves or at least holds on **both** Log Loss and ROC-AUC vs. the same-run XGBoost-defaults baseline. Otherwise document as negative, no `config.py` change.
- **Sources**: XGBoost official "Notes on Parameter Tuning" docs (new, added in Task 2) and Bergstra & Bengio JMLR 2012 (already in `RESOURCES.md` from the LightGBM round, reused not re-added).
- English only in all project content.

---

### Task 1: Add `build_xgboost_calibrated_pipeline()` to `src/model.py`

**Files:**
- Modify: `src/model.py` (append after `build_ensemble_pipeline()`)

**Interfaces:**
- Consumes: `CalibratedClassifierCV` (already imported), `config.RANDOM_STATE` (pre-existing).
- Produces: `build_xgboost_calibrated_pipeline(method="sigmoid", cv=3, n_estimators=100, learning_rate=0.3, max_depth=6)` — returns an unfitted `CalibratedClassifierCV` wrapping an `XGBClassifier`, with `.fit(X, y)` / `.predict_proba(X)`, consumed by Task 3 (notebook).

- [ ] **Step 1: Add the function**

Append to `src/model.py`:

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

- [ ] **Step 2: Verify with a synthetic-data smoke check**

```powershell
@'
import numpy as np
import pandas as pd
from src import model

default_pipe = model.build_xgboost_calibrated_pipeline()
default_params = default_pipe.estimator.get_params()
assert default_params["n_estimators"] == 100
assert default_params["learning_rate"] == 0.3
assert default_params["max_depth"] == 6

custom_pipe = model.build_xgboost_calibrated_pipeline(n_estimators=300, learning_rate=0.05, max_depth=4)
custom_params = custom_pipe.estimator.get_params()
assert custom_params["n_estimators"] == 300
assert custom_params["learning_rate"] == 0.05
assert custom_params["max_depth"] == 4

rng = np.random.RandomState(0)
X = pd.DataFrame(rng.rand(200, 5), columns=[f"f{i}" for i in range(5)])
y = pd.Series(rng.randint(0, 2, size=200))
custom_pipe.fit(X, y)
proba = custom_pipe.predict_proba(X)
assert proba.shape == (200, 2)

print("build_xgboost_calibrated_pipeline OK")
'@ | python -
```

Expected: prints `build_xgboost_calibrated_pipeline OK` with no errors.

- [ ] **Step 3: Commit**

```powershell
git add src/model.py
git commit -m "Add build_xgboost_calibrated_pipeline for standalone XGBoost tuning"
```

---

### Task 2: Add the XGBoost tuning-docs source to `RESOURCES.md`

Independent of Task 3's outcome — no dependency on the experiment's result.

**Files:**
- Modify: `RESOURCES.md` (new entry under "Papers and official documentation"; new row in the "How this applies to this repo's pipeline" table)

- [ ] **Step 1: Add the source entry**

Insert after the Voting Classifier entry added in the previous round:

```markdown
- **XGBoost official docs — Notes on Parameter Tuning**
  https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html
  Why: identifies `max_depth`, `min_child_weight`, `gamma` as the primary
  model-complexity/overfitting controls, `subsample`/`colsample_bytree` as
  randomness-based regularization, and the `eta` (`learning_rate`) vs
  `num_round` (`n_estimators`) trade-off -- the same trade-off already
  tuned for LightGBM. Justifies tuning `n_estimators`, `learning_rate`,
  `max_depth` for XGBoost via random search
  (`model.build_xgboost_calibrated_pipeline()`), analogous to
  `config.TUNED_LGBM_PARAMS`.
```

- [ ] **Step 2: Add the table row**

```markdown
| XGBoost tuning knobs (official XGBoost Parameter Tuning docs) | `src/model.py::build_xgboost_calibrated_pipeline()` |
```

- [ ] **Step 3: Commit**

```powershell
git add RESOURCES.md
git commit -m "Add source for XGBoost hyperparameter tuning"
```

---

### Task 3: Run the random search in the notebook

**Files:**
- Modify: `notebooks/01_eda_trends.ipynb` (new cells after the Step 10 cells added in the previous round)

**Interfaces:**
- Consumes: `model.build_xgboost_calibrated_pipeline(n_estimators, learning_rate, max_depth)` (Task 1), `evaluate.cross_validate_score()`, `config.TOP_TREND_FAMILIES/CV_FOLDS/RANDOM_STATE` (pre-existing).
- Produces: `results_df` (pandas DataFrame, one row per trial plus the same-run baseline row), read by the user and reported back for Task 4's gate decision.

- [ ] **Step 1: Hand over the markdown cell**

```markdown
## Step 11 — Tune XGBoost hyperparameters (random search)

**Source:** XGBoost's official "Notes on Parameter Tuning" docs (see
RESOURCES.md) identify `max_depth` (model complexity), and the
`learning_rate`/`n_estimators` trade-off, as primary tuning targets --
the same shape already tuned for LightGBM (Bergstra & Bengio, JMLR 2012,
also in RESOURCES.md, justifies random over grid search).

This evaluates XGBoost **standalone** (calibrated, library defaults vs.
random search), not the ensemble -- the point is improving XGBoost on its
own merits first; a future round would retest ensembling with these
tuned hyperparameters. Random search over n_estimators, learning_rate,
max_depth, scored with the existing `evaluate.cross_validate_score()` on
the current production feature set.
```

- [ ] **Step 2: Hand over the search loop code cell**

```python
import numpy as np

feature_cols = features.get_feature_columns(train, config.ID_COL, config.TARGET)
X_search, _ = features.encode_features(train, test, feature_cols)
trend_search = features.build_monthly_trend_features(train, config.TOP_TREND_FAMILIES)
X_search = pd.concat([X_search, trend_search], axis=1)
X_search, _ = features.impute_missing(X_search, X_search.copy())
y = train[config.TARGET]

# Same-run baseline: XGBoost library defaults on this exact feature set.
baseline_cv = evaluate.cross_validate_score(
    lambda: model.build_xgboost_calibrated_pipeline(),
    X_search, y, n_splits=config.CV_FOLDS, random_state=config.RANDOM_STATE, verbose=False,
)
print("Baseline (XGBoost library defaults, current feature set):")
evaluate.print_cv_summary(baseline_cv)

N_TRIALS = 20
rng = np.random.RandomState(config.RANDOM_STATE)
n_estimators_choices = [100, 200, 300, 500, 800]
max_depth_choices = [3, 4, 5, 6, 8, 10]

trial_results = [{
    "n_estimators": 100, "learning_rate": 0.3, "max_depth": 6, "is_baseline": True,
    **baseline_cv,
}]

for trial in range(N_TRIALS):
    n_estimators = int(rng.choice(n_estimators_choices))
    learning_rate = float(10 ** rng.uniform(np.log10(0.01), np.log10(0.3)))
    max_depth = int(rng.choice(max_depth_choices))

    cv_summary = evaluate.cross_validate_score(
        lambda: model.build_xgboost_calibrated_pipeline(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
        ),
        X_search, y, n_splits=config.CV_FOLDS, random_state=config.RANDOM_STATE, verbose=False,
    )
    print(f"Trial {trial + 1}/{N_TRIALS}: n_estimators={n_estimators}, "
          f"learning_rate={learning_rate:.4f}, max_depth={max_depth} "
          f"-> combined_score_mean={cv_summary['combined_score_mean']:.5f}")

    trial_results.append({
        "n_estimators": n_estimators, "learning_rate": learning_rate,
        "max_depth": max_depth, "is_baseline": False,
        **cv_summary,
    })

results_df = pd.DataFrame(trial_results).sort_values("combined_score_mean").reset_index(drop=True)
print("\nTop 5 trials (including baseline) by combined score:")
print(results_df.head(5)[["n_estimators", "learning_rate", "max_depth", "is_baseline",
                           "combined_score_mean", "log_loss_mean", "roc_auc_mean"]])
```

Note for the user: 20 trials x 5 CV folds x internal 3-fold calibration means this cell is comparably slow to (or slower than) the LightGBM tuning round's Step 7 — expect several minutes.

- [ ] **Step 3: Hand over the best-trial detail cell**

```python
print(results_df.iloc[0][["n_estimators", "learning_rate", "max_depth",
                           "combined_score_mean", "combined_score_std",
                           "log_loss_mean", "log_loss_std",
                           "roc_auc_mean", "roc_auc_std"]])
```

- [ ] **Step 4: User pastes, runs, and reports back**

Only Cell 1 (imports/data loading) needs to have run first — this cell rebuilds `X_search`/`y` itself. The user reports back the baseline row's numbers and the full best-trial detail (including `_std` values).

- [ ] **Step 5: Claude commits the notebook**

Once the user confirms the cells ran successfully:

```powershell
git add notebooks/01_eda_trends.ipynb
git commit -m "Tune XGBoost hyperparameters via random search (standalone)"
```

---

### Task 4: Decide and document the result

Use Task 3's actual reported numbers to pick **exactly one** of the two variants below.

**Files:**
- Modify: `README.md` (Progress section)
- Modify: `src/config.py` (only if adopted)

- [ ] **Step 1a: If the best trial improves or holds on both Log Loss and ROC-AUC vs. the same-run baseline** — append to `README.md`'s Progress section, with every `<...>` replaced by Task 3's real values:

```markdown
- Tuned XGBoost's `n_estimators`, `learning_rate`, and `max_depth` via
  random search (20 trials, see `notebooks/01_eda_trends.ipynb` Step 11 —
  sources: XGBoost "Notes on Parameter Tuning" docs, Bergstra & Bengio
  JMLR 2012, both in `RESOURCES.md`; design in
  `docs/superpowers/specs/2026-07-31-xgboost-hyperparameter-tuning-design.md`).
  Evaluated standalone (calibrated XGBoost only, not the ensemble).
  Same-run baseline (library defaults): combined score
  <BASELINE_COMBINED> ± <BASELINE_STD> (Log Loss <BASELINE_LL> ±
  <BASELINE_LL_STD>, ROC-AUC <BASELINE_AUC> ± <BASELINE_AUC_STD>). Best
  trial (`n_estimators=<BEST_N>`, `learning_rate=<BEST_LR>`,
  `max_depth=<BEST_MD>`): combined score <BEST_COMBINED> ± <BEST_STD>
  (Log Loss <BEST_LL> ± <BEST_LL_STD>, ROC-AUC <BEST_AUC> ±
  <BEST_AUC_STD>) — improves on both metrics, so saved as
  `config.TUNED_XGBOOST_PARAMS` (not wired into `src/train.py` — XGBoost
  alone was never the production model; this is an input for a future
  round retesting the LightGBM+XGBoost ensemble with a tuned XGBoost).
```

Then add to `src/config.py` (after `TUNED_LGBM_PARAMS`), with the real winning values:

```python
# --- Tuned XGBoost hyperparameters (standalone) ---
# Chosen by random search (see notebooks/01_eda_trends.ipynb Step 11 and
# RESOURCES.md: XGBoost "Notes on Parameter Tuning" docs, Bergstra &
# Bengio JMLR 2012). Combined score <BEST_COMBINED> vs <BASELINE_COMBINED>
# same-run XGBoost-defaults baseline. Not wired into src/train.py --
# saved as an input for a future ensembling-retest round
# (model.build_ensemble_pipeline() still uses XGBoost's library defaults
# internally).
TUNED_XGBOOST_PARAMS = {
    "n_estimators": <BEST_N>,
    "learning_rate": <BEST_LR>,
    "max_depth": <BEST_MD>,
}
```

- [ ] **Step 1b: If the best trial does not improve on both metrics** — append this bullet instead, with every `<...>` replaced by the real values:

```markdown
- Tuned XGBoost's `n_estimators`, `learning_rate`, and `max_depth` via
  random search (20 trials, see `notebooks/01_eda_trends.ipynb` Step 11 —
  sources: XGBoost "Notes on Parameter Tuning" docs, Bergstra & Bengio
  JMLR 2012, both in `RESOURCES.md`; design in
  `docs/superpowers/specs/2026-07-31-xgboost-hyperparameter-tuning-design.md`).
  Evaluated standalone (calibrated XGBoost only, not the ensemble).
  Same-run baseline (library defaults): combined score
  <BASELINE_COMBINED> ± <BASELINE_STD> (Log Loss <BASELINE_LL> ±
  <BASELINE_LL_STD>, ROC-AUC <BASELINE_AUC> ± <BASELINE_AUC_STD>). Best
  trial (`n_estimators=<BEST_N>`, `learning_rate=<BEST_LR>`,
  `max_depth=<BEST_MD>`): combined score <BEST_COMBINED> ± <BEST_STD>
  (Log Loss <BEST_LL> ± <BEST_LL_STD>, ROC-AUC <BEST_AUC> ±
  <BEST_AUC_STD>) — did not improve on <WHICH METRIC(S)>, so **not
  adopted**; no `config.py` change. Standalone XGBoost tuning doesn't
  close the gap with LightGBM enough (or at all) to justify revisiting
  ensembling; feature engineering remains the only untried candidate.
```

`src/config.py` stays unchanged in this branch.

- [ ] **Step 2: Commit**

```powershell
git add README.md
git commit -m "Document XGBoost hyperparameter tuning result"
```

If adopted (Step 1a), also stage `src/config.py`:

```powershell
git add README.md src/config.py
git commit -m "Save tuned XGBoost hyperparameters for a future ensembling round"
```
