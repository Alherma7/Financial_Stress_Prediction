# Sources and references

Resources used to define the project strategy (see "Next steps" in `README.md`).
All links were verified as accessible on 2026-07-28.

## Competition

- **AI4EAC Liquidity Stress Early Warning Challenge (Zindi)**
  https://zindi.africa/competitions/liquidity-stress-early-warning-challenge
  Predict whether a mobile money customer will experience financial stress within 30 days,
  based on 6 months of history. Metric: 0.6 × Log Loss + 0.4 × (1 − ROC-AUC). Restricted to
  university students in the East African Community, $7,500 prize pool.
  Status: completed — no public discussions/solution notebooks were found for this specific
  competition at the time of review (the discussions tab did not load content via fetch;
  check directly in the browser in case there are threads with tips).

## Books

- **"Approaching (Almost) Any Machine Learning Problem"** — Abhishek Thakur (4x Kaggle Grandmaster).
  Repo: https://github.com/abhishekkrthakur/approachingalmost (~8.4k stars; includes environment
  configs and dataset references, not the full book).
  Why: it's the most direct manual for how to structure a tabular competition project
  (CV, encoding, feature engineering, ensembling) — the general approach of this repo
  (`src/config.py`, `src/features.py`, etc.) follows that same spirit.

- **"Feature Engineering for Machine Learning"** — Alice Zheng & Amanda Casari (O'Reilly, 2018).
  Repo: https://github.com/alicezheng/feature-engineering-book (~1.5k stars; notebooks only, no data).
  Why: relevant for completing `build_monthly_trend_features()` in `src/features.py`
  (numerical and interaction feature chapters).

## Comparable competitions (Kaggle) and what they offer

Same type of problem: time-windowed customer features + financial risk target
+ a metric combining ranking (AUC) and calibration (log loss).

- **American Express – Default Prediction (2022)**
  Key lesson: for each time-windowed variable (equivalent to our `m1_...m6_`),
  compute mean/std/min/max/last value, and especially **delta and ratio between the last
  value and the first/mean**. One write-up reported ~12% improvement from these
  time-aggregation features alone.
  Winning models: LightGBM/XGBoost/CatBoost ensembles, heavy stacking near the top.
  Reference repos:
  - https://github.com/nodematerial/Kaggle_amex (modular feature engineering/training/inference pipeline)
  - https://github.com/EfthimiosVlahos/AMEX-Credit-Default-Prediction (LogReg/RF/GBM comparison)

- **Home Credit Default Risk (2018)**
  Key lesson from 1st place: **feature engineering mattered more than tuning or stacking**.
  They used windowed aggregations restricted to specific periods plus weighted moving
  averages on time-based features, and forward feature selection via ridge regression
  before ensembling several LightGBM/XGBoost models with stratified K-fold CV.
  Reference repos:
  - https://github.com/kozodoi/Kaggle_Home_Credit (top-4%, LightGBM ensembles)
  - https://github.com/NoxMoon/home-credit-default-risk (17th/7198, feature engineering + GRU + 32-model ensemble)

- **Home Credit – Credit Risk Model Stability (2024)**
  A variant with a stability metric (penalizes model performance drifting over time) — less
  directly applicable here since our problem is a single snapshot, but useful for
  understanding temporal robustness if needed.

- **Similar domain (not confirmed whether they use monthly windowed features)**: on Zindi —
  *Xente Credit Scoring Challenge*, *African Credit Scoring Challenge*,
  *Mobile Money and Financial Inclusion in Tanzania Challenge*, *Expresso Churn Prediction Challenge*.

## How this applies to this repo's pipeline

| Source | Where it applies in this project |
|---|---|
| Trend/delta/ratio features (Amex, Home Credit) | `src/features.py::build_monthly_trend_features()` (pending) |
| Feature engineering > tuning (Home Credit) | Prioritize EDA + features before tuning hyperparameters |
| LightGBM/XGBoost ensembles | `src/model.py` (placeholder already left) |
| Probability calibration | `CalibratedClassifierCV`, already noted in README as pending |
| Stratified K-Fold | Replace the single split in `src/train.py` |
