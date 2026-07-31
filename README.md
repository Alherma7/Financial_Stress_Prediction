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

## Next steps (pending)

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
