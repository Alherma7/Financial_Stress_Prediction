# LightGBM Hyperparameter Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tune LightGBM's `n_estimators`, `learning_rate`, and `num_leaves` via random search (instead of library defaults), and validate the effect on the competition's combined score (0.6 x Log Loss + 0.4 x ROC-AUC).

**Architecture:** Extend `src/model.py`'s pipeline builders to accept the 3 hyperparameters (defaults unchanged). Run a seeded random search in the notebook, scoring each combination with the existing `evaluate.cross_validate_score()` against the *current* production feature set (computed fresh in this same run, not an old recorded number, since the feature set has changed since that number was written). Visualize the marginal relationship between each hyperparameter and the score. Wire the winning combination into `src/train.py` only if it improves or holds on both metrics.

**Tech Stack:** Python, lightgbm, scikit-learn, numpy, pandas, matplotlib. No new dependencies. No unit test framework (project has none) — verification uses small standalone synthetic-data snippets; real validation is the notebook's random search + comparison to a same-run baseline.

## Global Constraints

- Scope is limited to `n_estimators`, `learning_rate`, `num_leaves` — do not add `max_depth`, `min_data_in_leaf`, or regularization parameters (deferred, see spec).
- Every new technique must trace to a source already added to `RESOURCES.md`: LightGBM official "Parameters Tuning" docs, and Bergstra & Bengio (JMLR 2012) — see `docs/superpowers/specs/2026-07-30-lightgbm-hyperparameter-tuning-design.md`.
- `build_lightgbm_pipeline()`'s and `build_lightgbm_calibrated_pipeline()`'s new keyword arguments must default to LightGBM's own library defaults (`n_estimators=100, learning_rate=0.1, num_leaves=31`), so `src/train.py`'s current wiring keeps working unmodified until Task 6.
- The random search must compare against a baseline computed in the *same* notebook run, on the *same* feature set used for the search (not the older `0.21333` figure in `README.md`, which predates the trend-feature exploration round and no longer reflects `src/train.py`'s exact current output).
- Do not wire `config.TUNED_LGBM_PARAMS` into `src/train.py` (Task 6) unless the best trial's combined score improves (or at least does not regress on both Log Loss and ROC-AUC) versus that same-run baseline.

---

### Task 1: Extend `model.py` pipeline builders with tunable hyperparameters

**Files:**
- Modify: `src/model.py` (`build_lightgbm_pipeline()` lines 36-52, `build_lightgbm_calibrated_pipeline()` lines 54-69)

**Interfaces:**
- Produces: `build_lightgbm_pipeline(class_weight=None, n_estimators=100, learning_rate=0.1, num_leaves=31)` and `build_lightgbm_calibrated_pipeline(class_weight=None, method="sigmoid", cv=3, n_estimators=100, learning_rate=0.1, num_leaves=31)`.
- Consumes: nothing new.

- [ ] **Step 1: Update `build_lightgbm_pipeline()`**

Replace it in `src/model.py` with:

```python
def build_lightgbm_pipeline(class_weight=None, n_estimators=100, learning_rate=0.1, num_leaves=31):
    """
    LightGBM baseline, as an alternative to Logistic Regression.

    Source: LightGBM/XGBoost ensembles were the winning approach in both the
    Amex Default Prediction and Home Credit Default Risk competitions (see
    RESOURCES.md) -- same problem shape as this one (time-windowed customer
    features + financial risk target). Tree-based models don't need feature
    scaling, so no StandardScaler here (unlike build_logreg_pipeline).

    n_estimators/learning_rate/num_leaves default to LightGBM's own library
    defaults. See config.TUNED_LGBM_PARAMS for the values chosen by random
    search (LightGBM docs "Parameters Tuning"; Bergstra & Bengio, JMLR 2012
    -- see RESOURCES.md).
    """
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        class_weight=class_weight,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        random_state=config.RANDOM_STATE,
        verbose=-1,
    )
```

- [ ] **Step 2: Update `build_lightgbm_calibrated_pipeline()`**

Replace it with:

```python
def build_lightgbm_calibrated_pipeline(class_weight=None, method="sigmoid", cv=3,
                                        n_estimators=100, learning_rate=0.1, num_leaves=31):
    """
    LightGBM wrapped in probability calibration.

    Source: Niculescu-Mizil & Caruana, "Predicting Good Probabilities with
    Supervised Learning" (ICML 2005) -- shows boosted trees push probability
    mass away from 0 and 1 (good ranking, poorly calibrated output), and
    that Platt scaling (sigmoid) corrects this. See RESOURCES.md. Log Loss
    carries 60% of this competition's score (src/evaluate.py), so this
    directly targets that weight without affecting ranking (ROC-AUC).
    """
    return CalibratedClassifierCV(
        estimator=build_lightgbm_pipeline(
            class_weight=class_weight,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
        ),
        method=method,
        cv=cv,
    )
```

- [ ] **Step 3: Verify defaults and pass-through with a synthetic check**

Run in PowerShell (paste as one block):

```powershell
@'
from src import model

default_pipe = model.build_lightgbm_pipeline()
default_params = default_pipe.get_params()
assert default_params["n_estimators"] == 100
assert default_params["learning_rate"] == 0.1
assert default_params["num_leaves"] == 31

custom_pipe = model.build_lightgbm_pipeline(n_estimators=200, learning_rate=0.05, num_leaves=63)
custom_params = custom_pipe.get_params()
assert custom_params["n_estimators"] == 200
assert custom_params["learning_rate"] == 0.05
assert custom_params["num_leaves"] == 63

calibrated = model.build_lightgbm_calibrated_pipeline(n_estimators=300, learning_rate=0.2, num_leaves=15)
inner_params = calibrated.estimator.get_params()
assert inner_params["n_estimators"] == 300
assert inner_params["learning_rate"] == 0.2
assert inner_params["num_leaves"] == 15

print("model.py params OK")
'@ | python -
```

Expected: prints `model.py params OK`.

- [ ] **Step 4: Commit**

```powershell
git add src/model.py
git commit -m "Make n_estimators/learning_rate/num_leaves tunable in LightGBM pipeline builders"
```

---

### Task 2: Add a `verbose` flag to `evaluate.cross_validate_score()`

The random search in Task 3 will call `cross_validate_score()` ~20 times (5 folds each) — printing every fold's scores every time would flood the notebook output. Add an opt-out flag, defaulting to the current (verbose) behavior so no existing caller changes.

**Files:**
- Modify: `src/evaluate.py:44-76` (`cross_validate_score()`)

**Interfaces:**
- Produces: `cross_validate_score(pipeline_builder, X, y, n_splits=5, random_state=None, verbose=True) -> dict` (same return shape as before).
- Consumes: nothing new.

- [ ] **Step 1: Add the `verbose` parameter**

In `src/evaluate.py`, change the function signature and the print call:

```python
def cross_validate_score(pipeline_builder, X, y, n_splits=5, random_state=None, verbose=True) -> dict:
    """
    Stratified K-Fold cross-validation of the competition's combined score.

    Source: the Home Credit Default Risk 1st place solution used stratified
    K-fold CV to get a robust score estimate before ensembling (see
    RESOURCES.md) -- more reliable than a single train/val split, especially
    with an imbalanced target like this dataset's (~15% positive rate).

    pipeline_builder must be a zero-argument callable returning a fresh,
    unfitted pipeline (e.g. `lambda: model.build_lightgbm_pipeline()`), so
    each fold trains its own model from scratch.

    Set verbose=False to suppress the per-fold print_scores() output (e.g.
    when calling this many times in a hyperparameter search).
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
        if verbose:
            print_scores(scores, label=f"Fold {fold}/{n_splits}")
        fold_scores.append(scores)
    summary = {}
    for key in ("log_loss", "roc_auc", "combined_score"):
        values = [s[key] for s in fold_scores]
        summary[f"{key}_mean"] = np.mean(values)
        summary[f"{key}_std"] = np.std(values)
    return summary
```

- [ ] **Step 2: Verify existing behavior is unchanged and the flag works**

```powershell
@'
import pandas as pd
import numpy as np
from src import evaluate

y_true = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
y_pred = pd.Series([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.1, 0.9, 0.2, 0.8])

# default (verbose=True) must still work with no verbose arg passed
scores = evaluate.combined_score(y_true, y_pred)
evaluate.print_scores(scores)
print("combined_score/print_scores unaffected: OK")
'@ | python -
```

Expected: prints the score lines followed by `combined_score/print_scores unaffected: OK` (this checks the untouched functions still import correctly; `cross_validate_score`'s `verbose=False` path itself is exercised directly in Task 3).

- [ ] **Step 3: Commit**

```powershell
git add src/evaluate.py
git commit -m "Add verbose flag to cross_validate_score to support quiet hyperparameter search loops"
```

---

### Task 3: Random search loop in the notebook

**Files:**
- Modify: `notebooks/01_eda_trends.ipynb` (new cells after the Step 6 cells added in the previous round)

**Interfaces:**
- Consumes: `model.build_lightgbm_calibrated_pipeline()` (Task 1), `evaluate.cross_validate_score(..., verbose=False)` (Task 2), `features.get_feature_columns/encode_features/build_monthly_trend_features/impute_missing` (pre-existing), `config.TOP_TREND_FAMILIES/CV_FOLDS/RANDOM_STATE` (pre-existing).
- Produces: `results_df` (pandas DataFrame, one row per trial plus one row for the same-run default-hyperparameter baseline), consumed by Task 4 (visualization) and Task 5 (decision).

- [ ] **Step 1: Add a markdown cell**

```markdown
## Step 7 — Tune LightGBM hyperparameters (random search)

**Source:** LightGBM's official "Parameters Tuning" docs (see RESOURCES.md)
describe the num_leaves / learning_rate / num_iterations trade-offs;
Bergstra & Bengio (JMLR 2012, see RESOURCES.md) show random search matches
or beats grid search for the same compute budget, since only a subset of
hyperparameters matters and that subset varies by dataset.

Random search over n_estimators, learning_rate, num_leaves, scored with
the existing `evaluate.cross_validate_score()` on the *current* feature
set (baseline one-hot + trend features), using
`model.build_lightgbm_calibrated_pipeline()` directly so the search
target matches the real competition metric. Compared against a baseline
computed in this same cell run (not the older number in README.md, since
the feature set has changed since that number was recorded).
```

- [ ] **Step 2: Add the search loop code cell**

```python
import numpy as np

# Same feature set src/train.py currently builds by default.
feature_cols = features.get_feature_columns(train, config.ID_COL, config.TARGET)
X_search, _ = features.encode_features(train, test, feature_cols)
trend_search = features.build_monthly_trend_features(train, config.TOP_TREND_FAMILIES)
X_search = pd.concat([X_search, trend_search], axis=1)
X_search, _ = features.impute_missing(X_search, X_search.copy())
y = train[config.TARGET]

# Same-run baseline: LightGBM library defaults on this exact feature set.
baseline_cv = evaluate.cross_validate_score(
    lambda: model.build_lightgbm_calibrated_pipeline(class_weight=None),
    X_search, y, n_splits=config.CV_FOLDS, random_state=config.RANDOM_STATE, verbose=False,
)
print("Baseline (library defaults, current feature set):")
evaluate.print_cv_summary(baseline_cv)

N_TRIALS = 20
rng = np.random.RandomState(config.RANDOM_STATE)
n_estimators_choices = [100, 200, 300, 500, 800]
num_leaves_choices = [15, 31, 63, 127, 255]

trial_results = [{
    "n_estimators": 100, "learning_rate": 0.1, "num_leaves": 31, "is_baseline": True,
    **baseline_cv,
}]

for trial in range(N_TRIALS):
    n_estimators = int(rng.choice(n_estimators_choices))
    learning_rate = float(10 ** rng.uniform(np.log10(0.01), np.log10(0.3)))
    num_leaves = int(rng.choice(num_leaves_choices))

    cv_summary = evaluate.cross_validate_score(
        lambda: model.build_lightgbm_calibrated_pipeline(
            class_weight=None,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
        ),
        X_search, y, n_splits=config.CV_FOLDS, random_state=config.RANDOM_STATE, verbose=False,
    )
    print(f"Trial {trial + 1}/{N_TRIALS}: n_estimators={n_estimators}, "
          f"learning_rate={learning_rate:.4f}, num_leaves={num_leaves} "
          f"-> combined_score_mean={cv_summary['combined_score_mean']:.5f}")

    trial_results.append({
        "n_estimators": n_estimators, "learning_rate": learning_rate,
        "num_leaves": num_leaves, "is_baseline": False,
        **cv_summary,
    })

results_df = pd.DataFrame(trial_results).sort_values("combined_score_mean").reset_index(drop=True)
print("\nTop 5 trials (including baseline) by combined score:")
print(results_df.head(5)[["n_estimators", "learning_rate", "num_leaves", "is_baseline", "combined_score_mean", "log_loss_mean", "roc_auc_mean"]])
```

- [ ] **Step 3: Run the cell and record the numbers**

Restart the kernel first (so `model.py` and `evaluate.py` changes from Tasks 1-2 are picked up), run the notebook top to bottom, then run this new cell. Note the baseline row and the best trial row from the printed top-5 table.

- [ ] **Step 4: Commit**

```powershell
git add notebooks/01_eda_trends.ipynb
git commit -m "Add LightGBM random hyperparameter search to the notebook"
```

---

### Task 4: Visualize the search results

**Files:**
- Modify: `notebooks/01_eda_trends.ipynb` (new cell right after Task 3's)

**Interfaces:**
- Consumes: `results_df` (Task 3).

- [ ] **Step 1: Add the visualization cell**

```python
best_idx = results_df["combined_score_mean"].idxmin()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, param in zip(axes, ["n_estimators", "learning_rate", "num_leaves"]):
    ax.errorbar(
        results_df[param], results_df["combined_score_mean"],
        yerr=results_df["combined_score_std"], fmt="o", alpha=0.6,
        color="tab:blue", ecolor="tab:blue", label="trials",
    )
    ax.scatter(
        results_df.loc[[best_idx], param], results_df.loc[[best_idx], "combined_score_mean"],
        color="tab:red", s=120, zorder=5, label="best trial",
    )
    if param == "learning_rate":
        ax.set_xscale("log")
    ax.set_xlabel(param)
    ax.set_ylabel("combined_score_mean (lower is better)")
    ax.set_title(f"{param} vs combined score")
    ax.legend()
plt.tight_layout()
plt.show()

best_trial = results_df.loc[best_idx]
print("Best trial:")
print(best_trial[["n_estimators", "learning_rate", "num_leaves", "is_baseline",
                   "combined_score_mean", "combined_score_std",
                   "log_loss_mean", "roc_auc_mean"]])
```

Note: since this is random search (not an exhaustive grid), each point's neighbors differ in the *other* two hyperparameters too, so expect a noisy scatter rather than a clean elbow curve — the plot shows the marginal trend per parameter, not a controlled one-at-a-time sweep.

- [ ] **Step 2: Run and save the notebook**

Run the cell, inspect the 3 plots, then save the notebook (File → Save) so the plot output and `results_df` state are persisted.

- [ ] **Step 3: Commit**

```powershell
git add notebooks/01_eda_trends.ipynb
git commit -m "Visualize LightGBM hyperparameter search results per parameter"
```

---

### Task 5: Decide and document the result

Compare the best trial (Task 3/4) against the `is_baseline=True` row from the *same* run (not the old `0.21333` figure in `README.md`).

**Files:**
- Modify: `README.md` (Progress and Next steps sections)

- [ ] **Step 1: Update `README.md`'s Progress section**

Add a bullet after the trend-feature-exploration bullet added in the previous round, using the actual numbers from Task 3/4 (replace every `<...>` below with the real printed values):

```markdown
- Tuned LightGBM's `n_estimators`, `learning_rate`, and `num_leaves` via
  random search (20 trials, see `notebooks/01_eda_trends.ipynb` Step 7 --
  sources: LightGBM "Parameters Tuning" docs, Bergstra & Bengio JMLR 2012,
  both in `RESOURCES.md`; design in
  `docs/superpowers/specs/2026-07-30-lightgbm-hyperparameter-tuning-design.md`).
  Same-run baseline (library defaults, current feature set): combined
  score <BASELINE_COMBINED> +/- <BASELINE_STD> (Log Loss <BASELINE_LL>,
  ROC-AUC <BASELINE_AUC>). Best trial (`n_estimators=<N>`,
  `learning_rate=<LR>`, `num_leaves=<NL>`): combined score
  <BEST_COMBINED> +/- <BEST_STD> (Log Loss <BEST_LL>, ROC-AUC <BEST_AUC>).
  <ONE SENTENCE: state whether this improved on both metrics, and whether
  it was wired into src/train.py per Task 6's outcome.>
```

- [ ] **Step 2: Update the Next steps checkbox**

```markdown
- [x] Tune LightGBM hyperparameters (`n_estimators`, `learning_rate`,
      `num_leaves`) instead of the library defaults.
```

- [ ] **Step 3: Commit**

```powershell
git add README.md
git commit -m "Document LightGBM hyperparameter search results"
```

---

### Task 6: Wire the tuned hyperparameters into `src/train.py` (conditional)

Only do this task if the best trial from Task 3 improved the combined score, or at minimum did not regress on both Log Loss and ROC-AUC, versus the same-run baseline. If it didn't, stop after Task 5 — the search is still valuable information (documented), just not adopted, same as the previous round's outcome.

**Files:**
- Modify: `src/config.py` (add `TUNED_LGBM_PARAMS`)
- Modify: `src/train.py:37-43` (the pipeline-builder call sites)

**Interfaces:**
- Consumes: `model.build_lightgbm_calibrated_pipeline(**config.TUNED_LGBM_PARAMS)` (Task 1's new kwargs).

- [ ] **Step 1: Add the tuned params to `config.py`**

Using the real best-trial values from Task 3 (not placeholders):

```python
# --- Tuned LightGBM hyperparameters ---
# Chosen by random search (see notebooks/01_eda_trends.ipynb Step 7 and
# RESOURCES.md: LightGBM "Parameters Tuning" docs, Bergstra & Bengio JMLR
# 2012). Combined score <BEST_COMBINED> vs <BASELINE_COMBINED> baseline
# (same feature set, same run).
TUNED_LGBM_PARAMS = {
    "n_estimators": <N>,
    "learning_rate": <LR>,
    "num_leaves": <NL>,
}
```

- [ ] **Step 2: Wire it into `src/train.py`**

Replace the two `model.build_lightgbm_calibrated_pipeline(class_weight=None)` call sites in `src/train.py` with `model.build_lightgbm_calibrated_pipeline(class_weight=None, **config.TUNED_LGBM_PARAMS)`.

- [ ] **Step 3: Run the full pipeline to confirm it still works end-to-end**

```powershell
python -m src.train
```

Expected: prints the CV summary (should be close to Task 3's best-trial numbers) and finishes without error, regenerating `submissions/submission.csv`.

- [ ] **Step 4: Commit**

```powershell
git add src/config.py src/train.py
git commit -m "Wire tuned LightGBM hyperparameters into src/train.py"
```
