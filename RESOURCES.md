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

## Papers and official documentation

- **"Predicting Good Probabilities with Supervised Learning"** — Niculescu-Mizil & Caruana
  (ICML 2005). https://www.cs.cornell.edu/~caruana/niculescu.scldbst.crc.rev4.pdf
  Why: empirically shows that boosted trees (like LightGBM/XGBoost) push probability mass
  away from 0 and 1, producing a characteristic sigmoid-shaped distortion in predicted
  probabilities — i.e. good ranking (AUC) but poorly calibrated output. They show Platt
  scaling (sigmoid) and Isotonic Regression correct this, and that calibrated boosted trees
  give the best probabilities of the methods they tested. Directly justifies using
  `CalibratedClassifierCV(method="sigmoid")` on top of `build_lightgbm_pipeline()` to
  improve Log Loss (60% of this competition's score) without hurting ROC-AUC.

- **scikit-learn: Probability calibration (user guide)**
  https://scikit-learn.org/stable/modules/calibration.html
  Why: official reference for `CalibratedClassifierCV`'s `sigmoid` (Platt) vs `isotonic`
  methods and the `cv` parameter (internal CV split used to fit the calibrator) — used
  directly in `src/model.py::build_lightgbm_calibrated_pipeline()`.

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

- **scikit-learn docs — Voting Classifier**
  https://scikit-learn.org/stable/modules/ensemble.html#voting-classifier
  Why: confirms `VotingClassifier(voting="soft")` with `weights=None`
  (default) computes a plain average of each classifier's predicted
  probabilities — the exact "simple average of probabilities" mechanism
  used in `model.build_ensemble_pipeline()`.

- **LightGBM official docs — Parameters Tuning**
  https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
  Why: describes the `num_leaves`/`learning_rate`/`num_iterations`
  trade-offs used to design the Step 7 random search
  (`config.TUNED_LGBM_PARAMS`). Its "Deal with over-fitting" section also
  lists the regularization knobs that round explicitly left untuned
  (smaller `num_leaves`/`max_depth`, `min_data_in_leaf`,
  `feature_fraction`/`bagging_fraction`, `lambda_l1`/`lambda_l2`,
  more training data) — the candidates for fixing the overfitting
  confirmed in `notebooks/01_eda_trends.ipynb` Step 12 (train vs. CV gap
  +0.131).

- **Bergstra & Bengio, "Random Search for Hyper-Parameter Optimization"**
  (JMLR 2012) https://www.jmlr.org/papers/v13/bergstra12a.html
  Why: shows random search matches or beats grid search for the same
  compute budget, since only a subset of hyperparameters matters per
  dataset and that subset varies — justifies random search (rather than
  an exhaustive grid) for both the LightGBM (Step 7) and XGBoost (Step 11)
  hyperparameter tuning rounds.

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
| LightGBM/XGBoost ensembles | `src/model.py::build_lightgbm_pipeline()` (done — now the default model in `src/train.py`, combined score 0.310 -> 0.226) |
| Probability calibration (Niculescu-Mizil & Caruana; sklearn docs) | `src/model.py::build_lightgbm_calibrated_pipeline()` |
| Stratified K-Fold | Replace the single split in `src/train.py` |
| `class_weight="balanced"` (sklearn glossary; calibration risk per Liu 2026) | `src/model.py::build_lightgbm_pipeline()` / `build_lightgbm_calibrated_pipeline()` |
| Soft-voting ensemble (sklearn Voting Classifier docs) | `src/model.py::build_ensemble_pipeline()` |
| XGBoost tuning knobs (official XGBoost Parameter Tuning docs) | `src/model.py::build_xgboost_calibrated_pipeline()` |
