# Design: Additional sourced trend features (min/max, weighted mean, net flow)

Date: 2026-07-30
Status: Approved

## Goal

Continue the "Explore additional feature engineering beyond the current top 13
trend families" item in `README.md`'s Next steps, in a way that directly
targets the competition metric (0.6 x Log Loss + 0.4 x ROC-AUC), using only
techniques already grounded in a source documented in `RESOURCES.md`.

## Scope

Applies only to the existing 13 top trend families
(`config.TOP_TREND_FAMILIES`) validated in
`notebooks/01_eda_trends.ipynb`. Does not extend trend features to the
remaining 16 families, and does not add customer-profile interactions --
both are deferred to a future round.

## Features

### 1. Min / max per family (extends `build_monthly_trend_features()`)

New columns per family: `{fam}_min_6m`, `{fam}_max_6m`.

Source: American Express - Default Prediction top solutions (see
`RESOURCES.md`, "Comparable competitions" section) -- explicitly lists
"mean/std/min/max/last value" as the aggregation set for time-windowed
variables. `build_monthly_trend_features()` currently computes mean, std,
last (`_delta_m1_m6` uses the most recent value), and ratio, but not min/max.

### 2. Recency-weighted mean per family (extends `build_monthly_trend_features()`)

New column per family: `{fam}_weighted_mean_6m`.

Source: Home Credit Default Risk 1st place solution (see `RESOURCES.md`,
"Comparable competitions" section) -- "weighted moving averages on
time-based features". The source does not specify exact weights, so this
design uses linearly decreasing weights over `monthly_prefixes`
(m1=6/21 ... m6=1/21, since m1 is most recent per `data_dictionary.csv`),
giving more weight to months closer to the 30-day prediction window. This
weighting scheme is a project-level implementation choice on top of a cited
technique, not a literal citation, and is documented as such in the
function's docstring.

### 3. Net flow features (new `build_net_flow_features()`)

For each month `m`, compute:

```
net_flow_m = m_deposit_total_value + m_received_total_value - m_withdraw_total_value
```

using `config.NET_FLOW_INFLOW_COLS = ["deposit_total_value", "received_total_value"]`
and `config.NET_FLOW_OUTFLOW_COLS = ["withdraw_total_value"]` (all three
already part of `config.TOP_TREND_FAMILIES`).

Then apply the same trend statistics (delta, mean, weighted mean, std, min,
max, ratio) used for individual families to this derived monthly series,
producing `net_flow_delta_m1_m6`, `net_flow_mean_6m`,
`net_flow_weighted_mean_6m`, `net_flow_std_6m`, `net_flow_min_6m`,
`net_flow_max_6m`, `net_flow_ratio_m1_mean`.

Source: Zheng & Casari, "Feature Engineering for Machine Learning" (see
`RESOURCES.md`), cited for numerical/interaction feature techniques --
net inflow minus outflow is an interaction feature across families rather
than within a single family, and maps directly onto the definition of
liquidity stress (spending capacity vs incoming funds).

## Implementation notes

- Extract a shared internal helper, e.g. `_trend_stats(monthly_values, name,
  monthly_prefixes) -> dict[str, pd.Series]`, computing delta/mean/weighted_mean/
  std/min/max/ratio from a DataFrame of monthly values. Both
  `build_monthly_trend_features()` (per family) and `build_net_flow_features()`
  (per derived net-flow series) call this helper, avoiding duplicating the
  7-statistic logic in two places.
- `build_monthly_trend_features()`'s public signature and existing output
  columns (delta, mean, std, ratio) do not change -- only min/max and
  weighted_mean are added per family, so existing callers keep working.
- `build_net_flow_features(df, monthly_prefixes=(...))` returns a DataFrame
  indexed like its input, following the same convention as
  `build_monthly_trend_features()`.

## Validation plan

Same pattern already used in this project for prior feature additions:

1. Add a new cell to `notebooks/01_eda_trends.ipynb` that builds the new
   features on top of the existing top-13 trend features and compares
   `evaluate.cross_validate_score()` (combined score, Log Loss, ROC-AUC)
   before/after, using the same calibrated LightGBM pipeline
   (`model.build_lightgbm_calibrated_pipeline()`) already the default.
2. Only if the combined score improves (or at least Log Loss / AUC don't
   regress), wire the new feature functions into `src/train.py`'s default
   pipeline, following the exact wiring pattern already used for
   `build_monthly_trend_features()`.
3. Update `README.md`'s "Progress" / "Next steps" sections with the result,
   same as prior entries.

No unit test framework exists in this project (`features.py`'s existing
functions have no unit tests, only notebook-based validation), so this
design does not introduce one -- consistent with current project convention.

## Out of scope (deferred)

- Extending trend features to the remaining 16 families.
- Customer-profile (`arpu`, `age`, `segment`, `earning_pattern`, `x_90_d_activity_rate`)
  interactions with trend features.
- Hyperparameter tuning of LightGBM (separate pending item in `README.md`).
