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

Not yet done: `build_monthly_trend_features()` is implemented and validated
in the notebook, but **not yet wired into `src/train.py`'s default pipeline**
(see next steps below).

## Next steps (pending)

- [x] Inspect the actual columns in `Train.csv` (confirm the monthly
      m1_...m6_ prefixes) and implement `build_monthly_trend_features()`
      in `src/features.py`.
- [ ] Wire `build_monthly_trend_features()` into `src/train.py`'s default
      pipeline (currently only used/validated in the notebook).
- [ ] Try LightGBM/XGBoost as an alternative to Logistic Regression
      (placeholder already left in `src/model.py`).
- [ ] Try probability calibration (`CalibratedClassifierCV`) to
      improve Log Loss.
- [ ] Cross-validation (stratified K-Fold) instead of a single split,
      for a more robust estimate of the combined score.
