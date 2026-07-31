# LightGBM class_weight="balanced" Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test `class_weight="balanced"` on the tuned, calibrated LightGBM pipeline (`model.build_lightgbm_calibrated_pipeline(**config.TUNED_LGBM_PARAMS)`, the current default in `src/train.py`), and either wire it in or document it as a negative result, per the project's existing improve-on-both-metrics gate.

**Architecture:** A single new notebook cell (Step 8) runs `evaluate.cross_validate_score()` on the tuned pipeline with `class_weight="balanced"` instead of `None`, and compares the printed `combined_score_mean`/`log_loss_mean`/`roc_auc_mean` against the documented baseline (0.20752 ± 0.00332 / 0.27039 ± 0.00291 / 0.88677 ± 0.00402). No new functions are needed — `class_weight` is already a parameter on both pipeline builders. The result then branches: wire into `config.py`/`train.py` (pass) or document as negative (fail) — either way `RESOURCES.md` gets two new source entries and the stale project memory note gets corrected.

**Tech Stack:** Python, lightgbm, scikit-learn, pandas, numpy. No new dependencies, no unit test framework (project has none) — verification is the notebook's cross-validated comparison against the documented baseline.

## Global Constraints

- **Manual code insertion**: do not use Edit/Write/NotebookEdit to add code to `notebooks/01_eda_trends.ipynb`, `src/config.py`, or `src/train.py`. Every code/text change to those files is handed to the user as a chat code block; the user pastes it in, runs it, and reports back. (Exception already used in this project: design/plan docs under `docs/superpowers/` and the assistant's own memory files are written directly.)
- **Gate**: only wire `class_weight="balanced"` into `src/train.py` if Task 1's measured combined score improves or at least holds on **both** Log Loss and ROC-AUC vs. the baseline (0.20752 ± 0.00332, Log Loss 0.27039 ± 0.00291, ROC-AUC 0.88677 ± 0.00402). Otherwise stop after Task 2 (document negative, do not touch `config.py`/`train.py`).
- **Sources**: every new technique traces to a source in `RESOURCES.md` — scikit-learn glossary (`class_weight`) and arXiv 2606.29720 (Zewen Liu), correctly framed (see `docs/superpowers/specs/2026-07-31-lightgbm-class-weight-balanced-design.md` — the Liu paper does **not** compare class weighting to SMOTE; it's cited only as the reason to check calibration/Log Loss, not just AUC, when applying any imbalance-correction technique).
- English only in all project content (code, comments, docs, notebook cells).

---

### Task 1: Run the `class_weight="balanced"` experiment in the notebook

**Files:**
- Modify: `notebooks/01_eda_trends.ipynb` (new cells after the Step 7b cell added in the previous round)

**Interfaces:**
- Consumes: `model.build_lightgbm_calibrated_pipeline(class_weight, **kwargs)`, `evaluate.cross_validate_score(pipeline_builder, X, y, n_splits, random_state, verbose)`, `config.TUNED_LGBM_PARAMS`, `config.TOP_TREND_FAMILIES`, `config.CV_FOLDS`, `config.RANDOM_STATE` (all pre-existing, no changes).
- Produces: `balanced_cv` (dict with `combined_score_mean/std`, `log_loss_mean/std`, `roc_auc_mean/std`), read by the user and reported back in chat for Task 2's gate decision.

- [ ] **Step 1: Hand over the markdown cell**

Give the user this markdown cell to paste at the end of the notebook:

```markdown
## Step 8 — Try `class_weight="balanced"`

**Source:** scikit-learn glossary, `class_weight`
(https://scikit-learn.org/stable/glossary.html#term-class_weight) — the
`"balanced"` mode weights each sample as
`n_samples / (n_classes * np.bincount(y))`, so both classes contribute
equal total weight to the loss (linear models) or split criterion (trees),
correcting for the ~15% positive rate in this dataset. `class_weight` is
already a parameter on both pipeline builders in `src/model.py`, always
called with `None` so far.

Compared against the documented baseline in `README.md`'s Progress
section (the tuned, calibrated pipeline): combined score 0.20752 ± 0.00332
(Log Loss 0.27039 ± 0.00291, ROC-AUC 0.88677 ± 0.00402). Per this
project's gate, only adopted if it improves or holds on both Log Loss and
ROC-AUC.
```

- [ ] **Step 2: Hand over the code cell**

```python
feature_cols = features.get_feature_columns(train, config.ID_COL, config.TARGET)
X_search, _ = features.encode_features(train, test, feature_cols)
trend_search = features.build_monthly_trend_features(train, config.TOP_TREND_FAMILIES)
X_search = pd.concat([X_search, trend_search], axis=1)
X_search, _ = features.impute_missing(X_search, X_search.copy())
y = train[config.TARGET]

balanced_cv = evaluate.cross_validate_score(
    lambda: model.build_lightgbm_calibrated_pipeline(class_weight="balanced", **config.TUNED_LGBM_PARAMS),
    X_search, y, n_splits=config.CV_FOLDS, random_state=config.RANDOM_STATE, verbose=False,
)
print("class_weight='balanced' (tuned LightGBM params):")
evaluate.print_cv_summary(balanced_cv)

baseline_combined = 0.20752
print(f"\nCombined score change vs documented baseline (class_weight=None): "
      f"{baseline_combined - balanced_cv['combined_score_mean']:+.5f} (positive = better)")
```

- [ ] **Step 3: User pastes, runs, and reports back**

The user pastes both cells at the end of `notebooks/01_eda_trends.ipynb`, restarts the kernel or re-runs from the top (so `train`/`test`/`config` are in scope), runs them, and reports back the printed `combined_score_mean/std`, `log_loss_mean/std`, `roc_auc_mean/std`.

- [ ] **Step 4: User saves and commits the notebook**

```powershell
git add notebooks/01_eda_trends.ipynb
git commit -m "Test class_weight=balanced on the tuned LightGBM pipeline"
```

---

### Task 2: Add the two sources to `RESOURCES.md`

Independent of Task 1's outcome — do this regardless of whether the experiment passes or fails the gate.

**Files:**
- Modify: `RESOURCES.md` (new entries under "Papers and official documentation", and a new row in the "How this applies to this repo's pipeline" table)

- [ ] **Step 1: Hand over the new source entries**

Give the user this text to insert into `RESOURCES.md`, under the existing "Papers and official documentation" section (after the sklearn calibration entry):

```markdown
- **scikit-learn glossary — `class_weight`**
  https://scikit-learn.org/stable/glossary.html#term-class_weight
  Why: defines the `"balanced"` mode's exact mechanism —
  `n_samples / (n_classes * np.bincount(y))` — i.e. each sample is
  weighted inversely proportional to its class's frequency, so both
  classes contribute equal total weight to the loss (linear models) or
  split criterion (trees). Directly justifies passing
  `class_weight="balanced"` to `model.build_lightgbm_pipeline()` /
  `build_lightgbm_calibrated_pipeline()` as a correction for this
  dataset's ~15% positive rate.

- **Zewen Liu, "The Hidden Cost of Resampling: How Imbalance Correction
  Degrades Probability Calibration in Tree Ensembles"** (arXiv 2606.29720)
  https://arxiv.org/abs/2606.29720
  Why: studies how resampling techniques (SMOTE, under/over-sampling)
  degrade probability calibration in tree ensembles — SMOTE's cost is
  modest, undersampling's is worse and grows with imbalance severity;
  recommends post-hoc recalibration after resampling. Note: this paper
  does **not** compare class weighting to resampling directly — it's
  cited here only as the reason to check Log Loss (not just ROC-AUC) when
  applying *any* imbalance-correction technique, consistent with this
  project's existing improve-on-both-metrics gate.
```

- [ ] **Step 2: Hand over the table row**

Give the user this row to add to the "How this applies to this repo's pipeline" table at the end of `RESOURCES.md`:

```markdown
| `class_weight="balanced"` (sklearn glossary; calibration risk per Liu 2026) | `src/model.py::build_lightgbm_pipeline()` / `build_lightgbm_calibrated_pipeline()` |
```

- [ ] **Step 3: User pastes and commits**

```powershell
git add RESOURCES.md
git commit -m "Add sources for class_weight=balanced experiment"
```

---

### Task 3: Decide and document the result in `README.md`

Use Task 1's actual reported numbers to pick **exactly one** of the two variants below — do not paste both.

**Files:**
- Modify: `README.md` (Progress section; Next steps if adopted)

- [ ] **Step 1a: If it passes the gate (improves or holds on both Log Loss and ROC-AUC)** — hand over this bullet, with every `<...>` replaced by Task 1's real printed values, to append to the Progress section:

```markdown
- Tried `class_weight="balanced"` on the tuned LightGBM pipeline
  (`notebooks/01_eda_trends.ipynb` Step 8 — source: scikit-learn glossary
  `class_weight`, see `RESOURCES.md`; design in
  `docs/superpowers/specs/2026-07-31-lightgbm-class-weight-balanced-design.md`).
  Result: combined score <BALANCED_COMBINED> ± <BALANCED_STD> (Log Loss
  <BALANCED_LL> ± <BALANCED_LL_STD>, ROC-AUC <BALANCED_AUC> ±
  <BALANCED_AUC_STD>) vs. the documented baseline 0.20752 ± 0.00332 (Log
  Loss 0.27039 ± 0.00291, ROC-AUC 0.88677 ± 0.00402) — improves on both
  Log Loss and ROC-AUC, so wired into `src/train.py`
  (`config.TUNED_LGBM_CLASS_WEIGHT`).
```

- [ ] **Step 1b: If it fails the gate** — hand over this bullet instead, with every `<...>` replaced by the real values:

```markdown
- Tried `class_weight="balanced"` on the tuned LightGBM pipeline
  (`notebooks/01_eda_trends.ipynb` Step 8 — source: scikit-learn glossary
  `class_weight`, see `RESOURCES.md`; design in
  `docs/superpowers/specs/2026-07-31-lightgbm-class-weight-balanced-design.md`).
  Result: combined score <BALANCED_COMBINED> ± <BALANCED_STD> (Log Loss
  <BALANCED_LL> ± <BALANCED_LL_STD>, ROC-AUC <BALANCED_AUC> ±
  <BALANCED_AUC_STD>) vs. the documented baseline 0.20752 ± 0.00332 (Log
  Loss 0.27039 ± 0.00291, ROC-AUC 0.88677 ± 0.00402) — worse on
  <WHICH METRIC(S)>, so **not adopted**; `src/train.py` keeps
  `class_weight=None`. Next candidates: ensembling LightGBM + XGBoost, or
  further feature engineering (see `RESOURCES.md`).
```

- [ ] **Step 2: If adopted, also update the Next steps section** — add:

```markdown
- [x] Try `class_weight="balanced"` on the tuned LightGBM pipeline.
```

- [ ] **Step 3: User pastes and commits**

```powershell
git add README.md
git commit -m "Document class_weight=balanced experiment result"
```

---

### Task 4: Wire `class_weight="balanced"` into `config.py`/`train.py` (conditional — only if Task 1 passed the gate)

Skip this task entirely if Task 1 failed the gate — Task 3's Step 1b already documents that outcome and no code changes are needed.

**Files:**
- Modify: `src/config.py` (add `TUNED_LGBM_CLASS_WEIGHT`)
- Modify: `src/train.py:37,42` (the two pipeline-builder call sites)

**Interfaces:**
- Consumes: `model.build_lightgbm_calibrated_pipeline(class_weight=..., **config.TUNED_LGBM_PARAMS)` (pre-existing signature, no changes).

- [ ] **Step 1: Hand over the `config.py` addition**

```python
# --- Tuned LightGBM class_weight ---
# Validated in notebooks/01_eda_trends.ipynb Step 8 against the tuned
# baseline (combined score 0.20752 -> <BALANCED_COMBINED>, both Log Loss
# and ROC-AUC improved). Source: scikit-learn glossary `class_weight`
# (RESOURCES.md).
TUNED_LGBM_CLASS_WEIGHT = "balanced"
```

- [ ] **Step 2: Hand over the `train.py` change**

Replace both occurrences of `class_weight=None` in `src/train.py`'s two `model.build_lightgbm_calibrated_pipeline(...)` call sites with `class_weight=config.TUNED_LGBM_CLASS_WEIGHT`.

- [ ] **Step 3: User pastes and runs the full pipeline**

```powershell
python -m src.train
```

Expected: prints a CV summary close to Task 1's `balanced_cv` numbers, finishes without error, and regenerates `submissions/submission.csv`.

- [ ] **Step 4: User commits**

```powershell
git add src/config.py src/train.py
git commit -m "Wire class_weight=balanced into src/train.py"
```

---

### Task 5: Correct the stale project memory note

This is the assistant's own memory file, not project code — done directly, no user action needed.

- [ ] **Step 1: Update `memory/project_next_session_class_weight.md`**

Replace its content to remove the inaccurate claim that arXiv 2606.29720 shows class weighting outperforms SMOTE (it doesn't make that comparison — see `docs/superpowers/specs/2026-07-31-lightgbm-class-weight-balanced-design.md`), and record that the experiment was run and its outcome (fill in once Task 1's numbers are known).
