# Design: LightGBM `class_weight="balanced"`

Date: 2026-07-31
Status: Approved

## Goal

Try `class_weight="balanced"` on the tuned LightGBM pipeline -- the
cheapest untried lever identified after hyperparameter tuning converged
(see `README.md` Progress log: best trial 0.20752 ± 0.00332, refined
search only gained 0.0006 further, within noise). `class_weight` already
exists as a parameter in `src/model.py::build_lightgbm_pipeline()` and
`build_lightgbm_calibrated_pipeline()` but is always called with
`class_weight=None` everywhere in the codebase -- this validates whether
setting it to `"balanced"` actually helps.

## Scope

Single-parameter experiment, evaluated only on the pipeline that is
`src/train.py`'s actual default: `build_lightgbm_calibrated_pipeline()`
with `config.TUNED_LGBM_PARAMS`. Not testing the uncalibrated pipeline in
isolation -- the calibrated pipeline is what ships, so it's the only
comparison that matters for the wiring decision (user decision, see
brainstorming transcript).

Not in scope: custom per-class sample weights via
`sklearn.utils.class_weight.compute_class_weight` (more control than
needed for a single-lever experiment on an already-identified technique;
would be scope creep beyond what was decided).

## Sources

- **scikit-learn glossary, `class_weight`**
  (https://scikit-learn.org/stable/glossary.html#term-class_weight):
  defines the `"balanced"` mode as
  `n_samples / (n_classes * np.bincount(y))` -- i.e. each sample is
  weighted inversely proportional to its class's frequency, so both
  classes contribute equal total weight to the loss/split criterion. This
  is the actual mechanism being tested and is new to `RESOURCES.md`
  (same kind of official-docs citation already used there for
  `CalibratedClassifierCV`).
- **Zewen Liu, "The Hidden Cost of Resampling: How Imbalance Correction
  Degrades Probability Calibration in Tree Ensembles"** (arXiv 2606.29720):
  verified via fetch on 2026-07-31. Corrected framing vs. the prior
  session's memory note -- the paper does **not** compare class weighting
  to SMOTE/resampling. It studies resampling techniques (SMOTE, under/
  over-sampling) specifically and finds their calibration cost is real but
  method-dependent (SMOTE: modest ECE increase; undersampling: worse,
  growing with imbalance severity), recommending post-hoc recalibration
  after resampling. Cited here only as the reason to check Log Loss (not
  just ROC-AUC) when applying any imbalance-correction technique --
  consistent with this project's existing practice of gating on both
  metrics, not as evidence that `class_weight="balanced"` outperforms
  resampling.

Both sources are added to `RESOURCES.md` as part of implementation.

## Design

### 1. Notebook experiment (Step 8)

In `notebooks/01_eda_trends.ipynb`, a new step following the same pattern
as Step 7's tuning comparisons: run
`evaluate.cross_validate_score()` on
`model.build_lightgbm_calibrated_pipeline(class_weight="balanced", **config.TUNED_LGBM_PARAMS)`
and print `combined_score_mean/std`, `log_loss_mean/std`,
`roc_auc_mean/std`, next to the documented baseline (0.20752 ± 0.00332,
`class_weight=None`).

No new code needed in `src/model.py` -- `class_weight` is already a
parameter there; this only exercises it with a real value for the first
time.

### 2. Validation and wiring gate

Same gate as every previous round in this project: only wire in if the
combined score improves or at least holds on **both** Log Loss and
ROC-AUC vs. the 0.20752 ± 0.00332 baseline.

- **If it passes:** add `config.TUNED_LGBM_CLASS_WEIGHT = "balanced"` (or
  `None`, to keep both states explicit) to `src/config.py`, and pass it
  as `class_weight=config.TUNED_LGBM_CLASS_WEIGHT` in `src/train.py`'s
  pipeline construction.
- **If it doesn't pass:** document as a negative result in `README.md`'s
  Progress log (same style as the min/max/weighted-mean/net-flow entry),
  leave `src/train.py` unchanged (`class_weight=None` stays implicit/
  default), and note the next-in-line options already identified
  (ensembling LightGBM + XGBoost, or further feature engineering) as the
  following candidate.

### 3. Documentation updates

- `RESOURCES.md`: add the two sources above, plus a row in the "How this
  applies to this repo's pipeline" table.
- `README.md`: append the result (pass or fail) to the Progress log,
  following the existing entry style.
- Project memory: correct/replace
  `memory/project_next_session_class_weight.md`, which currently
  misattributes to the Liu paper a claim it doesn't make (that class
  weighting outperforms SMOTE).

## Out of scope (deferred)

- Custom sample weights (`compute_class_weight` / manual `sample_weight`
  arrays) -- only the built-in `"balanced"` string mode is tested.
- Testing `class_weight="balanced"` on the uncalibrated pipeline in
  isolation.
- Ensembling LightGBM + XGBoost and further feature engineering -- next
  candidates only if this experiment doesn't pan out, per
  `memory/project_next_session_class_weight.md`.
