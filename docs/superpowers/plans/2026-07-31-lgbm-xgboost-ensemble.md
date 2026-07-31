# LightGBM + XGBoost Ensemble Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a soft-voting ensemble of the tuned/calibrated LightGBM pipeline and a calibrated XGBoost model (`model.build_ensemble_pipeline()`), validate it against the documented baseline (combined score 0.20752 ± 0.00332), and either wire it into `src/train.py` or document it as a negative result, per the project's established gate.

**Architecture:** `src/model.py` gains one new function, `build_ensemble_pipeline()`, that wraps the existing `build_lightgbm_calibrated_pipeline(**config.TUNED_LGBM_PARAMS)` and a new `CalibratedClassifierCV(XGBClassifier())` in `sklearn.ensemble.VotingClassifier(voting="soft")`. No other existing function changes. The notebook gets one new self-contained cell (Step 9) that scores it with the existing `evaluate.cross_validate_score()`.

**Tech Stack:** Python, scikit-learn (`VotingClassifier`, `CalibratedClassifierCV`), xgboost (new: not yet installed), lightgbm. No unit test framework in this project — verification is an ad hoc synthetic-data smoke check (same pattern as `docs/superpowers/plans/2026-07-30-lightgbm-hyperparameter-tuning.md` Task 1 Step 3) plus the notebook's real cross-validated comparison.

## Global Constraints

- **Manual code insertion, narrowed scope** (`memory/feedback_manual_code_insertion.md`, updated 2026-07-31): only notebook cells (`notebooks/01_eda_trends.ipynb`) are handed to the user as chat text for them to paste and run. Everything else — `src/model.py`, `src/config.py`, `src/train.py`, `README.md`, `RESOURCES.md`, and all git commits (including the notebook's, once the user confirms they ran it) — is edited and committed directly by Claude.
- **Gate**: only wire `build_ensemble_pipeline()` into `src/train.py` if Task 3's measured combined score improves or at least holds on **both** Log Loss and ROC-AUC vs. the baseline (0.20752 ± 0.00332, Log Loss 0.27039 ± 0.00291, ROC-AUC 0.88677 ± 0.00402). Otherwise document as negative and leave `src/train.py` unchanged.
- **Scope**: only unweighted (`weights=None`) soft voting between exactly two models. No stacking, no weight search, no XGBoost tuning, no CatBoost — see `docs/superpowers/specs/2026-07-31-lgbm-xgboost-ensemble-design.md` "Out of scope".
- **Sources**: every new technique traces to a source in `RESOURCES.md` — Amex/Home Credit ensembling writeups (already there), Niculescu-Mizil & Caruana (already there), and the scikit-learn Voting Classifier docs (new, added in Task 4).
- English only in all project content (code, comments, docs, notebook cells).

---

### Task 1: Install XGBoost

`xgboost` is listed in `requirements.txt` but is not actually installed in the current environment (`pip show xgboost` returns "Package(s) not found"). This is the only step in this round that needs an internet connection.

**Files:** none (environment change only).

- [ ] **Step 1: User installs the package**

```powershell
pip install xgboost
```

- [ ] **Step 2: Confirm it's importable**

```powershell
python -c "import xgboost; print(xgboost.__version__)"
```

Expected: prints a version number (e.g. `2.x.x`) with no `ModuleNotFoundError`.

---

### Task 2: Add `build_ensemble_pipeline()` to `src/model.py`

**Files:**
- Modify: `src/model.py` (append after `build_lightgbm_calibrated_pipeline()`, currently ending at line 83)

**Interfaces:**
- Consumes: `build_lightgbm_calibrated_pipeline(class_weight, method, cv, n_estimators, learning_rate, num_leaves)` (pre-existing, unchanged), `config.TUNED_LGBM_PARAMS`, `config.RANDOM_STATE` (pre-existing).
- Produces: `build_ensemble_pipeline(class_weight=None, method="sigmoid", cv=3)` — returns an unfitted `sklearn.ensemble.VotingClassifier` with `.fit(X, y)` / `.predict_proba(X)`, consumed by Task 3 (notebook) and Task 5 (conditional wiring into `src/train.py`).

- [ ] **Step 1: Add the function**

Append to `src/model.py`:

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

- [ ] **Step 2: Verify with a synthetic-data smoke check**

```powershell
@'
import numpy as np
import pandas as pd
from src import model

pipeline = model.build_ensemble_pipeline()
assert [name for name, _ in pipeline.estimators] == ["lgbm", "xgb"]
assert pipeline.voting == "soft"
assert pipeline.weights is None

rng = np.random.RandomState(0)
X = pd.DataFrame(rng.rand(200, 5), columns=[f"f{i}" for i in range(5)])
y = pd.Series(rng.randint(0, 2, size=200))

pipeline.fit(X, y)
proba = pipeline.predict_proba(X)
assert proba.shape == (200, 2)
assert np.allclose(proba.sum(axis=1), 1.0)

print("build_ensemble_pipeline OK")
'@ | python -
```

Expected: prints `build_ensemble_pipeline OK` with no errors.

- [ ] **Step 3: Commit**

```powershell
git add src/model.py
git commit -m "Add build_ensemble_pipeline: soft-voting LightGBM + XGBoost"
```

---

### Task 3: Run the ensemble experiment in the notebook

**Files:**
- Modify: `notebooks/01_eda_trends.ipynb` (new cells after the Step 8 cells added in the previous round)

**Interfaces:**
- Consumes: `model.build_ensemble_pipeline()` (Task 2), `evaluate.cross_validate_score()`, `config.TOP_TREND_FAMILIES/CV_FOLDS/RANDOM_STATE` (pre-existing).
- Produces: `ensemble_cv` (dict with `combined_score_mean/std`, `log_loss_mean/std`, `roc_auc_mean/std`), read by the user and reported back in chat for Task 5's gate decision.

- [ ] **Step 1: Hand over the markdown cell**

```markdown
## Step 9 — LightGBM + XGBoost ensemble (soft voting)

**Source:** LightGBM/XGBoost ensembles were the winning approach in the
Amex Default Prediction and Home Credit Default Risk competitions (see
RESOURCES.md). XGBoost is calibrated the same way as LightGBM
(Niculescu-Mizil & Caruana, ICML 2005) so both inputs to the vote are
comparably calibrated; `VotingClassifier(voting="soft")` with default
`weights=None` averages predict_proba across the two models (scikit-learn
docs -- Voting Classifier, see RESOURCES.md) -- the simple average
decided for this round. XGBoost uses library defaults (no tuning yet).

Compared against the documented baseline in `README.md`'s Progress
section (single tuned LightGBM): combined score 0.20752 ± 0.00332 (Log
Loss 0.27039 ± 0.00291, ROC-AUC 0.88677 ± 0.00402). Per this project's
gate, only adopted if it improves or holds on both Log Loss and ROC-AUC.
```

- [ ] **Step 2: Hand over the code cell**

```python
feature_cols = features.get_feature_columns(train, config.ID_COL, config.TARGET)
X_search, _ = features.encode_features(train, test, feature_cols)
trend_search = features.build_monthly_trend_features(train, config.TOP_TREND_FAMILIES)
X_search = pd.concat([X_search, trend_search], axis=1)
X_search, _ = features.impute_missing(X_search, X_search.copy())
y = train[config.TARGET]

ensemble_cv = evaluate.cross_validate_score(
    lambda: model.build_ensemble_pipeline(),
    X_search, y, n_splits=config.CV_FOLDS, random_state=config.RANDOM_STATE, verbose=False,
)
print("LightGBM + XGBoost ensemble (soft voting):")
evaluate.print_cv_summary(ensemble_cv)

baseline_combined = 0.20752
print(f"\nCombined score change vs documented baseline (single tuned LightGBM): "
      f"{baseline_combined - ensemble_cv['combined_score_mean']:+.5f} (positive = better)")
```

Note for the user: this cell is slower than Step 8's — it trains two calibrated models (LightGBM + XGBoost, each with its own internal 3-fold calibration split) across 5 CV folds, so expect several times longer than the single-model experiments.

- [ ] **Step 3: User pastes, runs, and reports back**

Only Cell 1 (imports/data loading) needs to have run first — this cell rebuilds `X_search`/`y` itself, same self-contained pattern as Steps 7-8. The user reports back the printed `combined_score_mean/std`, `log_loss_mean/std`, `roc_auc_mean/std`.

- [ ] **Step 4: Claude commits the notebook**

Once the user confirms the cells ran successfully:

```powershell
git add notebooks/01_eda_trends.ipynb
git commit -m "Test LightGBM + XGBoost soft-voting ensemble"
```

---

### Task 4: Add the scikit-learn Voting Classifier source to `RESOURCES.md`

Independent of Task 3's outcome — do this regardless of whether the experiment passes or fails the gate.

**Files:**
- Modify: `RESOURCES.md` (new entry under "Papers and official documentation"; new row in the "How this applies to this repo's pipeline" table)

- [ ] **Step 1: Add the source entry**

Insert after the `class_weight` glossary entry added in the previous round:

```markdown
- **scikit-learn docs — Voting Classifier**
  https://scikit-learn.org/stable/modules/ensemble.html#voting-classifier
  Why: confirms `VotingClassifier(voting="soft")` with `weights=None`
  (default) computes a plain average of each classifier's predicted
  probabilities — the exact "simple average of probabilities" mechanism
  used in `model.build_ensemble_pipeline()`.
```

- [ ] **Step 2: Add the table row**

```markdown
| Soft-voting ensemble (sklearn Voting Classifier docs) | `src/model.py::build_ensemble_pipeline()` |
```

- [ ] **Step 3: Commit**

```powershell
git add RESOURCES.md
git commit -m "Add source for the LightGBM + XGBoost ensemble"
```

---

### Task 5: Decide and document the result

Use Task 3's actual reported numbers to pick **exactly one** of the two variants below.

**Files:**
- Modify: `README.md` (Progress section; Next steps if adopted)
- Modify: `src/train.py:37,42` (only if adopted)

- [ ] **Step 1a: If it passes the gate (improves or holds on both Log Loss and ROC-AUC)** — append to `README.md`'s Progress section, with every `<...>` replaced by Task 3's real printed values:

```markdown
- Tried a soft-voting ensemble of the tuned LightGBM pipeline and a
  calibrated XGBoost model with library defaults
  (`notebooks/01_eda_trends.ipynb` Step 9 — sources: Amex/Home Credit
  ensembling writeups, Niculescu-Mizil & Caruana, scikit-learn Voting
  Classifier docs, all in `RESOURCES.md`; design in
  `docs/superpowers/specs/2026-07-31-lgbm-xgboost-ensemble-design.md`).
  Result: combined score <ENSEMBLE_COMBINED> ± <ENSEMBLE_STD> (Log Loss
  <ENSEMBLE_LL> ± <ENSEMBLE_LL_STD>, ROC-AUC <ENSEMBLE_AUC> ±
  <ENSEMBLE_AUC_STD>) vs. the documented baseline 0.20752 ± 0.00332 (Log
  Loss 0.27039 ± 0.00291, ROC-AUC 0.88677 ± 0.00402) — improves on both
  Log Loss and ROC-AUC, so wired into `src/train.py`
  (`model.build_ensemble_pipeline()`).
```

and update the Next steps section:

```markdown
- [x] Try ensembling LightGBM + XGBoost.
```

then wire it in — replace both `model.build_lightgbm_calibrated_pipeline(class_weight=None, **config.TUNED_LGBM_PARAMS)` call sites in `src/train.py` with `model.build_ensemble_pipeline()`, and run:

```powershell
python -m src.train
```

Expected: prints a CV summary close to Task 3's numbers, finishes without error, regenerates `submissions/submission.csv`.

- [ ] **Step 1b: If it fails the gate** — append this bullet instead, with every `<...>` replaced by the real values:

```markdown
- Tried a soft-voting ensemble of the tuned LightGBM pipeline and a
  calibrated XGBoost model with library defaults
  (`notebooks/01_eda_trends.ipynb` Step 9 — sources: Amex/Home Credit
  ensembling writeups, Niculescu-Mizil & Caruana, scikit-learn Voting
  Classifier docs, all in `RESOURCES.md`; design in
  `docs/superpowers/specs/2026-07-31-lgbm-xgboost-ensemble-design.md`).
  Result: combined score <ENSEMBLE_COMBINED> ± <ENSEMBLE_STD> (Log Loss
  <ENSEMBLE_LL> ± <ENSEMBLE_LL_STD>, ROC-AUC <ENSEMBLE_AUC> ±
  <ENSEMBLE_AUC_STD>) vs. the documented baseline 0.20752 ± 0.00332 (Log
  Loss 0.27039 ± 0.00291, ROC-AUC 0.88677 ± 0.00402) — worse on
  <WHICH METRIC(S)>, so **not adopted**; `src/train.py` keeps the single
  tuned LightGBM pipeline. Only feature engineering (customer-profile
  interactions, remaining 16 trend families) remains as an untried
  candidate.
```

`src/train.py` stays unchanged in this branch.

- [ ] **Step 2: Commit**

```powershell
git add README.md
git commit -m "Document LightGBM + XGBoost ensemble result"
```

If adopted (Step 1a), also stage `src/train.py`:

```powershell
git add README.md src/train.py
git commit -m "Wire LightGBM + XGBoost ensemble into src/train.py"
```
