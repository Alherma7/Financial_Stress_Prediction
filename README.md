# Liquidity / Financial Stress Prediction Challenge

Project for the Zindi challenge: predict whether a customer will experience
financial/liquidity stress in the next 30 days, based on 6 months of
mobile money transaction history.

## Project structure

```
liquidity-stress-project/
├── data/
│   ├── raw/            # Train.csv and Test.csv go here (not pushed to git)
│   └── processed/      # intermediate/generated datasets
├── notebooks/          # exploration notebooks (EDA, experiments)
├── src/                # reusable code, organized in modules
│   ├── config.py       # paths, column names, constants
│   ├── data.py         # data loading and submission saving
│   ├── features.py     # feature engineering (encoding + trend features)
│   ├── model.py        # model pipeline definitions
│   ├── evaluate.py      # combined metric (60% Log Loss + 40% AUC)
│   └── train.py         # end-to-end script: load -> train -> evaluate -> submission
├── models/              # saved trained models (.pkl, etc.)
├── submissions/         # generated submission files
├── requirements.txt
└── README.md
```

## Getting started

1. Create an environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Put `Train.csv` and `Test.csv` inside `data/raw/`.

3. Run the full baseline pipeline:
   ```bash
   python -m src.train
   ```
   This trains a Logistic Regression, prints validation Log Loss / AUC /
   combined score, and generates `submissions/submission.csv`.

4. To experiment freely, use a notebook in `notebooks/` and import
   functions from `src/` instead of rewriting the pipeline:
   ```python
   from src import config, data, features, model, evaluate
   ```

## Competition metric

Final score = 0.6 × Log Loss + 0.4 × ROC-AUC (see details in
`src/evaluate.py`). When comparing experiments, always use
`combined_score`, not just AUC — a model can rank customers well
(good AUC) but output poorly calibrated probabilities (bad Log Loss).

## Sources and strategy

Books, comparable Kaggle competitions, and reference repos used to define
the feature engineering and modeling approach: see [`RESOURCES.md`](RESOURCES.md).

## Final result (project closed 2026-08-10)

Closed after diminishing returns: five of the last six experiments
(regularization tuning, both ensemble variants, XGBoost tuning,
`class_weight="balanced"`) were rejected, and the one that did land
(`segment` x trend interactions) moved the combined score by less than
one standard deviation of CV noise. Further iteration isn't worth the
cost/benefit at this point -- see "Ideas not pursued" below for what's
left on the table if this is revisited.

Final production pipeline: `model.build_lightgbm_calibrated_pipeline()`
+ `config.TUNED_LGBM_PARAMS` on `features.build_production_features()`'s
319-column feature set (one-hot + top-13 trend families + segment x
trend interactions). Reproduced via `python -m src.train`:

- **Combined score: 0.20755 ± 0.00402** (Log Loss 0.27036 ± 0.00365,
  ROC-AUC 0.88668 ± 0.00466), 5-fold stratified CV.
- Final submission: `submissions/submission.csv` (30,000 rows, `ID` +
  `Target`, no nulls), generated from the model refit on 100% of
  `Train.csv`.
- Known limitation, not resolved: the train-vs-CV combined score gap is
  still ~+0.13 (Step 12/13) -- the model overfits each training fold
  significantly, but neither regularization (Step 13) nor the feature
  changes since have closed it. Doesn't block submission quality (CV
  score is still the relevant estimate of held-out performance), but
  would be the first thing to revisit if this project resumes.

### Ideas not pursued

Documented for a future session, in priority order (see project memory /
`docs/superpowers/specs/2026-08-10-segment-trend-interaction-design.md`
"Out of scope" for full rationale):

1. Expand `segment` interactions beyond `delta_m1_m6` to the other
   `_trend_stats()` outputs (mean_6m, std_6m, ratio_m1_mean, etc.).
2. `region` x trend interactions (weaker standalone signal than
   `segment`, same technique).
3. Feature reduction/selection, or early stopping via a held-out eval
   set, to actually address the unresolved overfitting gap.
4. Seed ensembling (variance reduction, not an overfitting fix -- lower
   priority than the above).

## Progress

EDA and feature engineering steps done so far live in
[`notebooks/01_eda_trends.ipynb`](notebooks/01_eda_trends.ipynb):

- Confirmed all 29 monthly variable families have all 6 months present
  (174 monthly columns total).
- Confirmed via `data_dictionary.csv` that **m1 is the most recent month and
  m6 is the oldest** (opposite of the naive assumption) — this flips the
  direction of any month-over-month trend calculation.
- Ranked all 29 families by the ROC-AUC of their raw `m1 - m6` delta against
  the target; `daily_avg_bal`, `deposit_*`, `withdraw_*` and `received_*` came
  out as the strongest signals, `mm_send_*`/`paybill_*` the weakest.
- Implemented `build_monthly_trend_features()` in `src/features.py` (delta,
  mean, std, and last-vs-mean ratio per family, for the top 13 families —
  see the function's docstring for the source technique).
- Validated the improvement: adding these trend features to the baseline
  Logistic Regression improved the validation combined score from **0.326
  to 0.310** (lower is better), with both Log Loss and AUC improving.
- Wired `build_monthly_trend_features()` (top 13 families, see
  `config.TOP_TREND_FAMILIES`) into `src/train.py`'s default pipeline.
  Running `python -m src.train` now reproduces the notebook's validation:
  combined score **0.30993**, Log Loss 0.36546, ROC-AUC 0.77337.
- Compared Logistic Regression vs LightGBM on the same (baseline + trend)
  feature set (`notebooks/01_eda_trends.ipynb`, Step 4) — LightGBM improved
  the combined score from 0.30993 to **0.22580** (Log Loss 0.29016, ROC-AUC
  0.87075), so it replaced Logistic Regression as the default model in
  `src/train.py` (`model.build_lightgbm_pipeline()`).
- Replaced the single train/validation split in `src/train.py` with
  Stratified K-Fold cross-validation (`evaluate.cross_validate_score()`,
  `config.CV_FOLDS = 5`) for a more robust score estimate, following the
  Home Credit Default Risk 1st place solution (see `RESOURCES.md`). Result:
  combined score **0.21829 ± 0.00529** (Log Loss 0.28360 ± 0.00482, ROC-AUC
  0.87967 ± 0.00630) — consistent with the single-split estimate, now with
  an uncertainty band.
- Wrapped LightGBM in `CalibratedClassifierCV` (sigmoid/Platt scaling), per
  Niculescu-Mizil & Caruana (ICML 2005) — boosted trees rank well but
  produce poorly calibrated probabilities, which hurts Log Loss (60% of the
  score). Now the default model in `src/train.py`
  (`model.build_lightgbm_calibrated_pipeline()`). Result: combined score
  **0.21333 ± 0.00420** (Log Loss 0.27683 ± 0.00359, ROC-AUC 0.88192 ±
  0.00532) — improves on the uncalibrated 0.21829 on both Log Loss and AUC.
- Explored min/max, recency-weighted mean (Home Credit "weighted moving
  averages" lesson), and net-flow (deposit + received - withdraw, Zheng &
  Casari interaction-feature technique) trend features on top of the
  existing top-13 trend families (`src/features.py::build_monthly_trend_features()`,
  `build_net_flow_features()` — see
  `docs/superpowers/specs/2026-07-30-additional-trend-features-design.md`).
  Validated in `notebooks/01_eda_trends.ipynb` (Step 6): combined score
  **0.21427 ± 0.00352** (Log Loss 0.27736 ± 0.00306, ROC-AUC 0.88035 ±
  0.00444) vs the documented baseline **0.21333 ± 0.00420** — slightly
  worse on both Log Loss and ROC-AUC, though the delta is smaller than one
  standard deviation of noise. Per this project's practice of only
  adopting changes that hold or improve on both metrics, **not wired into
  `src/train.py`**; the functions remain available in `src/features.py`
  for future experiments (e.g. isolating which of the three techniques
  helps vs. hurts, or revisiting after hyperparameter tuning) but are not
  part of the default pipeline.
- Tuned LightGBM's `n_estimators`, `learning_rate`, and `num_leaves` via
  random search (20 broad trials + 15 refined trials around the best
  region, see `notebooks/01_eda_trends.ipynb` Step 7 — sources: LightGBM
  "Parameters Tuning" docs, Bergstra & Bengio JMLR 2012, both in
  `RESOURCES.md`; design in
  `docs/superpowers/specs/2026-07-30-lightgbm-hyperparameter-tuning-design.md`).
  Same-run baseline (library defaults, current feature set): combined
  score 0.21308 ± 0.00488 (Log Loss 0.27639 ± 0.00435, ROC-AUC 0.88188 ±
  0.00578). Best trial (`n_estimators=600`, `learning_rate=0.0411`,
  `num_leaves=31`): combined score **0.20752 ± 0.00332** (Log Loss
  0.27039 ± 0.00291, ROC-AUC 0.88677 ± 0.00402) — improves on both Log
  Loss and ROC-AUC, so wired into `src/train.py`
  (`config.TUNED_LGBM_PARAMS`). The refined search only found a further
  0.0006 improvement over the broad search's best, within noise —
  tuning has converged on these 3 parameters; further gains likely need
  `class_weight="balanced"`, model ensembling (LightGBM + XGBoost, see
  RESOURCES.md), or more feature engineering, not more tuning.
- Tried `class_weight="balanced"` on the tuned LightGBM pipeline
  (`notebooks/01_eda_trends.ipynb` Step 8 — source: scikit-learn glossary
  `class_weight`, see `RESOURCES.md`; design in
  `docs/superpowers/specs/2026-07-31-lightgbm-class-weight-balanced-design.md`).
  Result: combined score 0.21061 ± 0.00376 (Log Loss 0.27371 ± 0.00345,
  ROC-AUC 0.88403 ± 0.00428) vs. the documented baseline 0.20752 ± 0.00332
  (Log Loss 0.27039 ± 0.00291, ROC-AUC 0.88677 ± 0.00402) — worse on both
  Log Loss and ROC-AUC (though the delta is smaller than one standard
  deviation of noise), so **not adopted**; `src/train.py` keeps
  `class_weight=None`. Next candidates: ensembling LightGBM + XGBoost, or
  further feature engineering (see `RESOURCES.md`).
- Tried a soft-voting ensemble of the tuned LightGBM pipeline and a
  calibrated XGBoost model with library defaults
  (`notebooks/01_eda_trends.ipynb` Step 9 — sources: Amex/Home Credit
  ensembling writeups, Niculescu-Mizil & Caruana, scikit-learn Voting
  Classifier docs, all in `RESOURCES.md`; design in
  `docs/superpowers/specs/2026-07-31-lgbm-xgboost-ensemble-design.md`).
  Result: combined score 0.21489 ± 0.00330 (Log Loss 0.28115 ± 0.00277,
  ROC-AUC 0.88448 ± 0.00443) vs. the documented baseline 0.20752 ±
  0.00332 (Log Loss 0.27039 ± 0.00291, ROC-AUC 0.88677 ± 0.00402) —
  worse on both Log Loss and ROC-AUC, by a larger margin than either
  round's noise band (about 2x the standard deviation, unlike the
  `class_weight` round) — so **not adopted**; `src/train.py` keeps the
  single tuned LightGBM pipeline. Likely cause: an unweighted 50/50
  average with an untuned XGBoost drags the ensemble toward the weaker
  model. Untried variants deferred for a future round: weighting the
  vote toward LightGBM, or tuning XGBoost before ensembling (see
  `docs/superpowers/specs/2026-07-31-lgbm-xgboost-ensemble-design.md`
  "Out of scope"); feature engineering (customer-profile interactions,
  remaining 16 trend families) is also still untried.
- Tuned XGBoost's `n_estimators`, `learning_rate`, and `max_depth` via
  random search, evaluated standalone (`notebooks/01_eda_trends.ipynb`
  Step 11 — sources: XGBoost "Notes on Parameter Tuning" docs, Bergstra &
  Bengio JMLR 2012, both in `RESOURCES.md`; design in
  `docs/superpowers/specs/2026-07-31-xgboost-hyperparameter-tuning-design.md`).
  Same-run baseline (library defaults): combined score 0.23380 ± 0.00420
  (Log Loss 0.30229 ± 0.00331, ROC-AUC 0.86895 ± 0.00611). Stopped early
  by the user after 11 of 20 trials, since the best partial trials
  (`n_estimators=800, learning_rate=0.0761, max_depth=4` → 0.22328, plus
  three others clustering around 0.223–0.225) were clearly not going to
  approach the single-LightGBM baseline (0.20752) or even the previously
  rejected ensemble attempts (0.20831–0.21489) — the search was already
  plateauing well short of closing that gap. **Not adopted**; no
  `config.py` change (no complete "best trial" to save, and it wouldn't
  be used regardless — ensembling is deprioritized following this
  result). Remaining untried candidate: feature engineering
  (customer-profile interactions — `segment` shows the strongest
  untapped categorical signal in the dataset, 19.0%/17.2%/12.6% target
  rate for HVC/LVC/MVC vs. a ~15% base rate — or the remaining 16 trend
  families).
- Weighted the LightGBM + XGBoost ensemble toward LightGBM
  (`notebooks/01_eda_trends.ipynb` Step 10 — grid search over
  LightGBM/XGBoost weight ratios; source: scikit-learn Voting Classifier
  docs, already in `RESOURCES.md`; design in
  `docs/superpowers/specs/2026-07-31-weighted-ensemble-design.md`). Best
  weighted ratio tried, LightGBM=0.9: combined score 0.20831 ± 0.00328
  (Log Loss 0.27166 ± 0.00283, ROC-AUC 0.88672 ± 0.00410), still worse
  than plain LightGBM (0.20752 ± 0.00332, Log Loss 0.27039 ± 0.00291,
  ROC-AUC 0.88677 ± 0.00402) — so **not adopted**; `src/train.py` keeps
  the single tuned LightGBM pipeline. The score degraded monotonically
  as XGBoost's weight increased (0.20831 → 0.20946 → 0.21094 → 0.21275 →
  0.21489 for LightGBM weights 0.9 → 0.8 → 0.7 → 0.6 → 0.5), so no weight
  in this grid helps — ensembling with this untuned XGBoost doesn't pay
  off at any ratio tried. Remaining untried candidates: tuning XGBoost's
  hyperparameters before ensembling, or further feature engineering
  (customer-profile interactions, remaining 16 trend families).
- Checked the tuned LightGBM pipeline for overfitting
  (`notebooks/01_eda_trends.ipynb` Step 12 — source: Abhishek Thakur,
  *Approaching (Almost) Any Machine Learning Problem*, see
  `RESOURCES.md`): for each of the same 5 CV folds, scored the fitted
  pipeline on its own training fold in addition to its validation fold.
  Result: mean train combined score **0.07672** vs. mean val **0.20752**
  (per-fold gap +0.125 to +0.136, consistent across all 5 folds — not
  noise). **Confirms significant overfitting** — the model nearly
  memorizes each training fold (train Log Loss/AUC close to a perfect
  fit) while validation performance matches the already-documented CV
  number. `config.TUNED_LGBM_PARAMS`'s random search (Step 7) only
  covered `n_estimators`/`learning_rate`/`num_leaves` and explicitly
  scoped out regularization knobs (see
  `docs/superpowers/specs/2026-07-30-lightgbm-hyperparameter-tuning-design.md`
  "Out of scope") — those are the next candidates to try (LightGBM
  "Parameters Tuning" docs' "Deal with over-fitting" section, see
  `RESOURCES.md`: smaller `num_leaves`/`max_depth`, `min_data_in_leaf`,
  `feature_fraction`/`bagging_fraction`, `lambda_l1`/`lambda_l2`).

- Tried regularization tuning to close the Step 12 overfitting gap
  (`notebooks/01_eda_trends.ipynb` Step 13 -- random search over
  `min_data_in_leaf`, `feature_fraction`, `bagging_fraction`/
  `bagging_freq`, `lambda_l1`, `lambda_l2`, with `n_estimators`/
  `learning_rate`/`num_leaves` fixed at `config.TUNED_LGBM_PARAMS`;
  sources: LightGBM "Parameters Tuning" docs "Deal with over-fitting"
  section, Bergstra & Bengio JMLR 2012, Abhishek Thakur -- all in
  `RESOURCES.md`; design in
  `docs/superpowers/specs/2026-07-31-lightgbm-regularization-design.md`).
  Same-run baseline (library defaults for these knobs): val combined
  score 0.20763, train combined score 0.07674, gap +0.13089 -- matches
  the already-documented Step 12 numbers. Stopped early by the user after
  13 of 20 trials: every trial fell into one of two failing patterns, and
  none cleared the gate (both hold/improve the val score **and**
  meaningfully shrink the gap). Most trials (e.g. Trial 2: val 0.21153,
  gap +0.12806; Trial 10: val 0.21321, gap +0.12426) validated worse than
  baseline while the gap barely moved. The trial with the largest gap
  reduction, Trial 9 (gap +0.10928, down ~0.022 from baseline), did so by
  validating clearly worse (val 0.21116) and pushing train score up to
  0.10188 -- regularizing into underfitting rather than closing the gap
  productively, exactly the failure mode the design doc's gate was
  written to reject. The single trial with a (marginally) better val
  score, Trial 5 (val 0.20722), only reduced the gap by ~0.005 -- smaller
  than the +0.011 fold-to-fold spread already seen in Step 12, i.e. not a
  meaningful reduction. **Not adopted**; `src/train.py` keeps
  `config.TUNED_LGBM_PARAMS` unchanged, `src/model.py`'s new
  `min_data_in_leaf`/`feature_fraction`/`bagging_fraction`/
  `bagging_freq`/`lambda_l1`/`lambda_l2` parameters stay at their library
  defaults for the production pipeline (added to `build_lightgbm_pipeline()`/
  `build_lightgbm_calibrated_pipeline()` to make this round possible, kept
  for future experiments). Conclusion: regularizing the existing feature
  set trades val score for train score along a fixed frontier without
  closing the gap -- the overfitting isn't primarily a "model has too much
  capacity for these features" problem, so the next candidate is changing
  what the model is fed rather than how much it's allowed to fit it.
  Deferred alternatives if this pattern recurs: feature reduction/
  selection, or early stopping via a held-out eval set (needs pipeline
  changes beyond this round).

- Added `segment` (HVC/LVC/MVC) x top trend family interaction features
  (`notebooks/01_eda_trends.ipynb` Step 14 -- source: Zheng & Casari,
  "Feature Engineering for Machine Learning", same interaction-feature
  technique already used for `build_net_flow_features()`; design in
  `docs/superpowers/specs/2026-08-10-segment-trend-interaction-design.md`).
  `features.build_segment_interaction_features()` multiplies each of the
  13 top trend families' `delta_m1_m6` by each of the 3 `segment`
  indicators (39 new columns). Same-run baseline (current production
  features): combined score 0.20763 (Log Loss 0.27051, ROC-AUC 0.88668).
  With interaction features: combined score **0.20755** (Log Loss
  **0.27036**, ROC-AUC 0.88668, held) -- improved Log Loss, held ROC-AUC,
  so **adopted** (margin is smaller than one standard deviation of CV
  noise, but per this project's gate, holding/improving on both metrics
  is what matters, not the size of the delta). Wired into
  `features.build_production_features()`, so `src/train.py` picks it up
  automatically. Production feature count: 319 columns (up from 280).

## Next steps (closed -- see "Ideas not pursued" above for anything remaining)

- [x] Inspect the actual columns in `Train.csv` (confirm the monthly
      m1_...m6_ prefixes) and implement `build_monthly_trend_features()`
      in `src/features.py`.
- [x] Wire `build_monthly_trend_features()` into `src/train.py`'s default
      pipeline (currently only used/validated in the notebook).
- [x] Try LightGBM/XGBoost as an alternative to Logistic Regression
      (placeholder already left in `src/model.py`).
- [x] Try probability calibration (`CalibratedClassifierCV`) to
      improve Log Loss.
- [x] Cross-validation (stratified K-Fold) instead of a single split,
      for a more robust estimate of the combined score.
- [x] Tune LightGBM hyperparameters (`n_estimators`, `learning_rate`,
      `num_leaves`) instead of the library defaults.
- [x] Explore additional feature engineering beyond the current top 13
      trend families (`config.TOP_TREND_FAMILIES`).
- [x] Fix the confirmed overfitting (train vs. CV gap +0.131, Step 12):
      tried LightGBM's regularization knobs (Step 13) -- no trial closed
      the gap without giving up val score. **Not adopted** (5th rejected
      experiment); see Progress log above.
- [x] Feature engineering: `segment` (HVC/LVC/MVC) x top trend family
      interaction features. Combined score 0.20763 -> **0.20755**.
      **Adopted** -- see Progress log above.
