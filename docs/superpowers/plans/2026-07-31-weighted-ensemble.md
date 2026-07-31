# Weighted LightGBM + XGBoost Ensemble Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `weights` parameter to `model.build_ensemble_pipeline()`, grid-search a handful of LightGBM/XGBoost weight ratios, and either wire the best one into `src/train.py` or document the round as a negative result, per the project's established gate.

**Architecture:** `VotingClassifier`'s existing `weights` parameter (already in use conceptually via the sklearn docs cited in `RESOURCES.md`) is exposed on `build_ensemble_pipeline()`. A new notebook cell (Step 10) loops over 4 new weight ratios, scored with the existing `evaluate.cross_validate_score()`, and compares them against 2 already-documented ratios (100/0 and 50/50) without recomputing those two.

**Tech Stack:** Python, scikit-learn (`VotingClassifier.weights`). No new dependencies. No unit test framework in this project — verification is a synthetic-data smoke check (same precedent as the two prior plans) plus the notebook's real cross-validated comparison.

## Global Constraints

- **Manual code insertion** (`memory/feedback_manual_code_insertion.md`): only notebook cells (`notebooks/01_eda_trends.ipynb`) are handed to the user as chat text for pasting/running. Everything else — `src/model.py`, `src/config.py`, `src/train.py`, `README.md`, and all git commits (including the notebook's, once the user confirms they ran it) — is edited and committed directly by Claude.
- **Gate**: only wire the ensemble into `src/train.py` if the best of the 6 weight ratios (4 new + 2 documented) improves or at least holds on **both** Log Loss and ROC-AUC vs. the baseline (0.20752 ± 0.00332, Log Loss 0.27039 ± 0.00291, ROC-AUC 0.88677 ± 0.00402). If the 100/0 point (plain LightGBM) remains best, document as negative.
- **Scope**: only the 4-point grid [0.9, 0.8, 0.7, 0.6] (LightGBM weight) plus the 2 already-documented endpoints. No finer grid, no XGBoost tuning, no stacking — see `docs/superpowers/specs/2026-07-31-weighted-ensemble-design.md` "Out of scope".
- **No new sources needed**: the scikit-learn Voting Classifier docs (already in `RESOURCES.md`) already cover the `weights` parameter used this round.
- English only in all project content.

---

### Task 1: Add `weights` to `model.build_ensemble_pipeline()`

**Files:**
- Modify: `src/model.py` (`build_ensemble_pipeline()`, currently the last function in the file)

**Interfaces:**
- Produces: `build_ensemble_pipeline(class_weight=None, method="sigmoid", cv=3, weights=None)` — same return type as before (unfitted `VotingClassifier`), now with an optional `weights` list `[lgbm_weight, xgb_weight]` passed straight through. `weights=None` (default) preserves the current equal-weighting behavior.
- Consumes: nothing new.

- [ ] **Step 1: Update the function signature and return statement**

In `src/model.py`, change:

```python
def build_ensemble_pipeline(class_weight=None, method="sigmoid", cv=3):
```

to:

```python
def build_ensemble_pipeline(class_weight=None, method="sigmoid", cv=3, weights=None):
```

and change the final line from:

```python
    return VotingClassifier(estimators=[("lgbm", lgbm), ("xgb", xgb)], voting="soft")
```

to:

```python
    return VotingClassifier(estimators=[("lgbm", lgbm), ("xgb", xgb)], voting="soft", weights=weights)
```

Also add one sentence to the docstring, right after the existing paragraph about `VotingClassifier(voting="soft", weights=None)`:

```python
    An optional `weights=[lgbm_weight, xgb_weight]` shifts the average
    toward one model -- see config.TUNED_ENSEMBLE_WEIGHTS (if adopted) or
    the negative-result note in README.md for the grid search that
    validated this.
```

- [ ] **Step 2: Verify with a synthetic-data smoke check**

```powershell
@'
import numpy as np
import pandas as pd
from src import model

rng = np.random.RandomState(0)
X = pd.DataFrame(rng.rand(200, 5), columns=[f"f{i}" for i in range(5)])
y = pd.Series(rng.randint(0, 2, size=200))

default_pipe = model.build_ensemble_pipeline()
assert default_pipe.weights is None

weighted_pipe = model.build_ensemble_pipeline(weights=[0.7, 0.3])
assert weighted_pipe.weights == [0.7, 0.3]

weighted_pipe.fit(X, y)
proba = weighted_pipe.predict_proba(X)
assert proba.shape == (200, 2)
assert np.allclose(proba.sum(axis=1), 1.0)

print("build_ensemble_pipeline weights OK")
'@ | python -
```

Expected: prints `build_ensemble_pipeline weights OK` with no errors.

- [ ] **Step 3: Commit**

```powershell
git add src/model.py
git commit -m "Add weights parameter to build_ensemble_pipeline"
```

---

### Task 2: Run the weight grid search in the notebook

**Files:**
- Modify: `notebooks/01_eda_trends.ipynb` (new cells after the Step 9 cells added in the previous round)

**Interfaces:**
- Consumes: `model.build_ensemble_pipeline(weights=[lgbm_weight, xgb_weight])` (Task 1), `evaluate.cross_validate_score()`, `config.TOP_TREND_FAMILIES/CV_FOLDS/RANDOM_STATE` (pre-existing).
- Produces: `weight_results_df` (pandas DataFrame, 6 rows: 4 new grid points + 2 documented endpoints), read by the user and reported back for Task 3's gate decision.

- [ ] **Step 1: Hand over the markdown cell**

```markdown
## Step 10 — Weight the ensemble toward LightGBM

**Hypothesis:** Step 9's unweighted 50/50 average (combined score
0.21489 ± 0.00330) was worse than plain LightGBM (0.20752 ± 0.00332) by
~2x the noise band -- likely because the untuned XGBoost drags the
average toward the weaker model. Weighting the vote more heavily toward
LightGBM should move the result back toward, or past, the baseline.

**Source:** same as Step 9 -- scikit-learn Voting Classifier docs (see
RESOURCES.md), which document the `weights` parameter used here.

Grid: LightGBM weight in [0.9, 0.8, 0.7, 0.6] (XGBoost weight =
1 - LightGBM weight), scored with the existing
`evaluate.cross_validate_score()`. The 100/0 (plain LightGBM) and 50/50
(Step 9's ensemble) points are already documented in `README.md` and are
NOT recomputed here -- they're added to the results table directly, to
avoid re-training two calibrated models per point for numbers we already
have.
```

- [ ] **Step 2: Hand over the grid search code cell**

```python
feature_cols = features.get_feature_columns(train, config.ID_COL, config.TARGET)
X_search, _ = features.encode_features(train, test, feature_cols)
trend_search = features.build_monthly_trend_features(train, config.TOP_TREND_FAMILIES)
X_search = pd.concat([X_search, trend_search], axis=1)
X_search, _ = features.impute_missing(X_search, X_search.copy())
y = train[config.TARGET]

# Already-documented endpoints (README.md Progress log) -- not recomputed.
weight_results = [
    {"lgbm_weight": 1.0, "xgb_weight": 0.0,
     "combined_score_mean": 0.20752, "combined_score_std": 0.00332,
     "log_loss_mean": 0.27039, "log_loss_std": 0.00291,
     "roc_auc_mean": 0.88677, "roc_auc_std": 0.00402},
    {"lgbm_weight": 0.5, "xgb_weight": 0.5,
     "combined_score_mean": 0.21489, "combined_score_std": 0.00330,
     "log_loss_mean": 0.28115, "log_loss_std": 0.00277,
     "roc_auc_mean": 0.88448, "roc_auc_std": 0.00443},
]

for lgbm_weight in [0.9, 0.8, 0.7, 0.6]:
    xgb_weight = 1 - lgbm_weight
    cv_summary = evaluate.cross_validate_score(
        lambda: model.build_ensemble_pipeline(weights=[lgbm_weight, xgb_weight]),
        X_search, y, n_splits=config.CV_FOLDS, random_state=config.RANDOM_STATE, verbose=False,
    )
    print(f"LightGBM weight={lgbm_weight:.1f}, XGBoost weight={xgb_weight:.1f} "
          f"-> combined_score_mean={cv_summary['combined_score_mean']:.5f}")
    weight_results.append({
        "lgbm_weight": lgbm_weight, "xgb_weight": xgb_weight,
        **cv_summary,
    })

weight_results_df = pd.DataFrame(weight_results).sort_values("combined_score_mean").reset_index(drop=True)
print("\nAll 6 points, sorted by combined score (lower is better):")
print(weight_results_df[["lgbm_weight", "xgb_weight", "combined_score_mean", "combined_score_std",
                          "log_loss_mean", "roc_auc_mean"]])
```

Note for the user: each of the 4 new grid points is as slow as Step 9's single ensemble run (two calibrated models x 5 CV folds), so this cell takes roughly 4x as long as Step 9.

- [ ] **Step 3: User pastes, runs, and reports back**

Only Cell 1 (imports/data loading) needs to have run first — this cell rebuilds `X_search`/`y` itself. The user reports back the full printed table (all 6 rows).

- [ ] **Step 4: Claude commits the notebook**

Once the user confirms the cells ran successfully:

```powershell
git add notebooks/01_eda_trends.ipynb
git commit -m "Grid-search LightGBM/XGBoost ensemble weights"
```

---

### Task 3: Decide and document the result

Use Task 2's actual reported table to pick **exactly one** of the two variants below.

**Files:**
- Modify: `README.md` (Progress section; Next steps if adopted)
- Modify: `src/config.py` (only if adopted)
- Modify: `src/train.py:37,42` (only if adopted)

- [ ] **Step 1a: If the best of the 6 points improves or holds on both Log Loss and ROC-AUC vs. the 100/0 baseline** — append to `README.md`'s Progress section, with every `<...>` replaced by Task 2's real values for the winning row:

```markdown
- Weighted the LightGBM + XGBoost ensemble toward LightGBM
  (`notebooks/01_eda_trends.ipynb` Step 10 — grid search over
  LightGBM/XGBoost weight ratios; source: scikit-learn Voting Classifier
  docs, already in `RESOURCES.md`; design in
  `docs/superpowers/specs/2026-07-31-weighted-ensemble-design.md`). Best
  ratio: LightGBM=<BEST_LGBM_WEIGHT>, XGBoost=<BEST_XGB_WEIGHT>: combined
  score <BEST_COMBINED> ± <BEST_STD> (Log Loss <BEST_LL> ± <BEST_LL_STD>,
  ROC-AUC <BEST_AUC> ± <BEST_AUC_STD>) vs. the documented baseline
  0.20752 ± 0.00332 (Log Loss 0.27039 ± 0.00291, ROC-AUC 0.88677 ±
  0.00402) — improves on both Log Loss and ROC-AUC, so wired into
  `src/train.py` (`config.TUNED_ENSEMBLE_WEIGHTS`).
```

and update the Next steps section:

```markdown
- [x] Weight the LightGBM + XGBoost ensemble toward LightGBM.
```

then add to `src/config.py` (after `TUNED_LGBM_PARAMS`), with the real winning weights:

```python
# --- Tuned ensemble weights ---
# Chosen by grid search (see notebooks/01_eda_trends.ipynb Step 10 and
# docs/superpowers/specs/2026-07-31-weighted-ensemble-design.md).
# Combined score <BEST_COMBINED> vs 0.20752 single-LightGBM baseline.
TUNED_ENSEMBLE_WEIGHTS = [<BEST_LGBM_WEIGHT>, <BEST_XGB_WEIGHT>]
```

and replace both `model.build_lightgbm_calibrated_pipeline(class_weight=None, **config.TUNED_LGBM_PARAMS)` call sites in `src/train.py` with `model.build_ensemble_pipeline(weights=config.TUNED_ENSEMBLE_WEIGHTS)`, then verify:

```powershell
python -m src.train
```

Expected: prints a CV summary close to Task 2's winning row, finishes without error, regenerates `submissions/submission.csv`.

- [ ] **Step 1b: If the 100/0 point (plain LightGBM) remains best** — append this bullet instead, with every `<...>` replaced by the real values of the best-performing weighted point (even though it didn't win):

```markdown
- Weighted the LightGBM + XGBoost ensemble toward LightGBM
  (`notebooks/01_eda_trends.ipynb` Step 10 — grid search over
  LightGBM/XGBoost weight ratios; source: scikit-learn Voting Classifier
  docs, already in `RESOURCES.md`; design in
  `docs/superpowers/specs/2026-07-31-weighted-ensemble-design.md`). Best
  weighted ratio tried, LightGBM=<BEST_LGBM_WEIGHT>: combined score
  <BEST_COMBINED> ± <BEST_STD> (Log Loss <BEST_LL> ± <BEST_LL_STD>,
  ROC-AUC <BEST_AUC> ± <BEST_AUC_STD>), still worse than plain LightGBM
  (0.20752 ± 0.00332, Log Loss 0.27039 ± 0.00291, ROC-AUC 0.88677 ±
  0.00402) — so **not adopted**; `src/train.py` keeps the single tuned
  LightGBM pipeline. Ensembling with this untuned XGBoost doesn't help at
  any weight tried in this grid. Remaining untried candidates: tuning
  XGBoost's hyperparameters before ensembling, or further feature
  engineering (customer-profile interactions, remaining 16 trend
  families).
```

`src/config.py` and `src/train.py` stay unchanged in this branch.

- [ ] **Step 2: Commit**

```powershell
git add README.md
git commit -m "Document weighted LightGBM+XGBoost ensemble grid search result"
```

If adopted (Step 1a), also stage the wiring:

```powershell
git add README.md src/config.py src/train.py
git commit -m "Wire weighted LightGBM+XGBoost ensemble into src/train.py"
```
