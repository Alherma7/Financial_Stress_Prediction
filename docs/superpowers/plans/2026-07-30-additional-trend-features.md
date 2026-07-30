# Additional Sourced Trend Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add min/max, recency-weighted mean, and net-flow trend features (all sourced in `RESOURCES.md`) on top of the existing 13 top trend families, and validate their effect on the competition's combined score (0.6 x Log Loss + 0.4 x ROC-AUC).

**Architecture:** Extract a shared `_trend_stats()` helper in `src/features.py` used by both the existing `build_monthly_trend_features()` (extended with min/max/weighted_mean) and a new `build_net_flow_features()` (inflow - outflow per month, same stats). Validate in the notebook before wiring into `src/train.py`.

**Tech Stack:** Python, pandas, numpy. No new dependencies. No unit test framework (project has none today) — verification uses small standalone synthetic-data snippets, and the real validation is a notebook cell comparing `evaluate.cross_validate_score()` before/after.

## Global Constraints

- Scope is limited to the 13 families in `config.TOP_TREND_FAMILIES` — do not extend trend features to the other 16 families (deferred, see spec).
- Every new technique must trace to a source already in `RESOURCES.md` (Amex for min/max, Home Credit 1st place for weighted mean, Zheng & Casari for net-flow interaction) — see `docs/superpowers/specs/2026-07-30-additional-trend-features-design.md`.
- `build_monthly_trend_features()`'s existing public signature and existing output columns (delta, mean, std, ratio) must not change — only new columns are added, so `src/train.py`'s current wiring keeps working unmodified until Task 6.
- `net_flow_*` columns use exactly `config.NET_FLOW_INFLOW_COLS = ["deposit_total_value", "received_total_value"]` and `config.NET_FLOW_OUTFLOW_COLS = ["withdraw_total_value"]`.
- Do not wire anything into `src/train.py` (Task 6) unless the notebook validation in Task 5 shows the combined score improves (or at least does not regress on both Log Loss and ROC-AUC).

---

### Task 1: Extract `_trend_stats()` helper (refactor, no behavior change)

**Files:**
- Modify: `src/features.py` (the `build_monthly_trend_features()` function, currently lines 54-86)

**Interfaces:**
- Produces: `_trend_stats(monthly_values: pd.DataFrame, monthly_prefixes: tuple) -> dict[str, pd.Series]`, keys: `"delta_m1_m6"`, `"mean_6m"`, `"std_6m"`, `"ratio_m1_mean"` (only these 4 for now — min/max/weighted_mean come in Tasks 2-3). `monthly_values` must have exactly `len(monthly_prefixes)` columns, ordered the same way (column 0 = most recent).
- Consumes: nothing new — pure refactor of existing logic.

This is a pure refactor: the output of `build_monthly_trend_features()` must be byte-for-byte identical before and after. Verify with a characterization check before touching the code, then again after.

- [ ] **Step 1: Write a characterization snippet and capture the "before" output**

Run this in a Python shell from the project root (`python`, then paste), **before making any changes**:

```python
import pandas as pd
from src import features

df = pd.DataFrame({
    "m1_deposit_total_value": [100.0, 50.0],
    "m2_deposit_total_value": [90.0, 40.0],
    "m3_deposit_total_value": [80.0, 30.0],
    "m4_deposit_total_value": [70.0, 20.0],
    "m5_deposit_total_value": [60.0, 10.0],
    "m6_deposit_total_value": [50.0, 0.0],
})

before = features.build_monthly_trend_features(df, ["deposit_total_value"])
before.to_csv("_before_refactor.csv")
print(before)
```

Expected output (so you can sanity-check by eye): for row 0, `deposit_total_value_delta_m1_m6` = 50.0 (100-50), `_mean_6m` = 75.0, `_ratio_m1_mean` = 100/75 ≈ 1.333.

- [ ] **Step 2: Refactor `build_monthly_trend_features()` to use the new helper**

Replace the body of `build_monthly_trend_features()` in `src/features.py` with:

```python
def _trend_stats(monthly_values: pd.DataFrame, monthly_prefixes) -> dict:
    """
    Shared trend statistics for a monthly value series (one column per
    entry in monthly_prefixes, ordered most-recent -> oldest, i.e. column 0
    corresponds to monthly_prefixes[0] = m1 = most recent).
    """
    most_recent = monthly_values.iloc[:, 0]
    oldest = monthly_values.iloc[:, -1]
    mean_6m = monthly_values.mean(axis=1)

    return {
        "delta_m1_m6": most_recent - oldest,
        "mean_6m": mean_6m,
        "std_6m": monthly_values.std(axis=1),
        "ratio_m1_mean": (most_recent / mean_6m).replace([np.inf, -np.inf], np.nan),
    }


def build_monthly_trend_features(df: pd.DataFrame, families: list, monthly_prefixes=("m1", "m2", "m3", "m4", "m5", "m6")) -> pd.DataFrame:
    """
    Build trend features per variable family across the 6-month window.

    Source: mean/std/last-value aggregation + delta and ratio between the last
    value and the mean, following the American Express - Default Prediction
    top solutions (see RESOURCES.md).

    Note: monthly_prefixes is ordered most-recent -> oldest (m1 = most
    recent, m6 = oldest, per data_dictionary.csv).
    """
    trend_df = pd.DataFrame(index=df.index)
    for fam in families:
        cols = [f"{m}_{fam}" for m in monthly_prefixes]
        monthly_values = df[cols]
        stats = _trend_stats(monthly_values, monthly_prefixes)
        for stat_name, series in stats.items():
            trend_df[f"{fam}_{stat_name}"] = series
    return trend_df
```

Keep the module's existing `import pandas as pd` and `import numpy as np` at the top of the file unchanged.

- [ ] **Step 3: Verify the refactor produced identical output**

Run in the same Python shell:

```python
import importlib
from src import features
importlib.reload(features)

after = features.build_monthly_trend_features(df, ["deposit_total_value"])
pd.testing.assert_frame_equal(before, after)
print("Refactor OK: output unchanged")
```

Expected: prints `Refactor OK: output unchanged` with no `AssertionError`. If it raises, compare `before` vs `after` column-by-column to find the mismatch before proceeding.

- [ ] **Step 4: Delete the scratch file and commit**

```bash
rm _before_refactor.csv
git add src/features.py
git commit -m "Extract _trend_stats() helper from build_monthly_trend_features (no behavior change)"
```

---

### Task 2: Add min/max trend stats (Amex source)

**Files:**
- Modify: `src/features.py` (`_trend_stats()` from Task 1)

**Interfaces:**
- Produces: `_trend_stats()` now also returns `"min_6m"` and `"max_6m"` keys. `build_monthly_trend_features()` output gains `{fam}_min_6m` and `{fam}_max_6m` columns for every family, automatically (no other code changes needed since it already iterates over `stats.items()`).
- Consumes: `_trend_stats()` from Task 1.

- [ ] **Step 1: Add min/max to `_trend_stats()`**

In `src/features.py`, update the `return` dict in `_trend_stats()`:

```python
    return {
        "delta_m1_m6": most_recent - oldest,
        "mean_6m": mean_6m,
        "std_6m": monthly_values.std(axis=1),
        "min_6m": monthly_values.min(axis=1),
        "max_6m": monthly_values.max(axis=1),
        "ratio_m1_mean": (most_recent / mean_6m).replace([np.inf, -np.inf], np.nan),
    }
```

Also update `build_monthly_trend_features()`'s docstring to mention min/max, citing the source (Amex "mean/std/min/max/last value", see `RESOURCES.md`).

- [ ] **Step 2: Verify on synthetic data**

Run in a Python shell:

```python
import importlib, pandas as pd
from src import features
importlib.reload(features)

df = pd.DataFrame({
    "m1_deposit_total_value": [100.0],
    "m2_deposit_total_value": [90.0],
    "m3_deposit_total_value": [80.0],
    "m4_deposit_total_value": [70.0],
    "m5_deposit_total_value": [60.0],
    "m6_deposit_total_value": [50.0],
})
out = features.build_monthly_trend_features(df, ["deposit_total_value"])
assert out["deposit_total_value_min_6m"].iloc[0] == 50.0
assert out["deposit_total_value_max_6m"].iloc[0] == 100.0
print("min/max OK")
```

Expected: prints `min/max OK`.

- [ ] **Step 3: Commit**

```bash
git add src/features.py
git commit -m "Add min/max trend stats to build_monthly_trend_features (Amex source, RESOURCES.md)"
```

---

### Task 3: Add recency-weighted mean (Home Credit source)

**Files:**
- Modify: `src/features.py` (`_trend_stats()`)

**Interfaces:**
- Produces: `_linear_recency_weights(n: int) -> np.ndarray` (module-level helper). `_trend_stats()` now also returns `"weighted_mean_6m"`. `build_monthly_trend_features()` output gains `{fam}_weighted_mean_6m` per family.
- Consumes: `_trend_stats()` from Tasks 1-2.

- [ ] **Step 1: Add the weight helper and wire it into `_trend_stats()`**

Add above `_trend_stats()` in `src/features.py`:

```python
def _linear_recency_weights(n: int) -> np.ndarray:
    """
    Linearly decreasing weights, index 0 (most recent month) heaviest,
    normalized to sum to 1. E.g. n=6 -> [6, 5, 4, 3, 2, 1] / 21.

    Source: Home Credit Default Risk 1st place solution used "weighted
    moving averages on time-based features" (see RESOURCES.md) without
    specifying exact weights; linear decay is this project's
    implementation choice, weighting months closer to the 30-day
    prediction window more heavily (monthly_prefixes[0] = m1 = most recent).
    """
    weights = np.arange(n, 0, -1, dtype=float)
    return weights / weights.sum()
```

Then update `_trend_stats()`'s body (add before the `return`):

```python
    weights = _linear_recency_weights(len(monthly_prefixes))
    weighted_mean_6m = (monthly_values * weights).sum(axis=1)
```

and add `"weighted_mean_6m": weighted_mean_6m,` to the returned dict.

- [ ] **Step 2: Verify on synthetic data**

Run in a Python shell:

```python
import importlib, pandas as pd, numpy as np
from src import features
importlib.reload(features)

df = pd.DataFrame({
    "m1_deposit_total_value": [100.0],
    "m2_deposit_total_value": [90.0],
    "m3_deposit_total_value": [80.0],
    "m4_deposit_total_value": [70.0],
    "m5_deposit_total_value": [60.0],
    "m6_deposit_total_value": [50.0],
})
out = features.build_monthly_trend_features(df, ["deposit_total_value"])

values = np.array([100.0, 90.0, 80.0, 70.0, 60.0, 50.0])
weights = np.arange(6, 0, -1) / np.arange(6, 0, -1).sum()
expected = (values * weights).sum()

assert abs(out["deposit_total_value_weighted_mean_6m"].iloc[0] - expected) < 1e-9
assert out["deposit_total_value_weighted_mean_6m"].iloc[0] > out["deposit_total_value_mean_6m"].iloc[0]
print("weighted mean OK:", out["deposit_total_value_weighted_mean_6m"].iloc[0], "vs plain mean", out["deposit_total_value_mean_6m"].iloc[0])
```

Expected: prints `weighted mean OK: 81.66... vs plain mean 75.0` (weighted mean pulled toward the more heavily-weighted recent (higher) values, so it's greater than the plain mean for this descending series).

- [ ] **Step 3: Commit**

```bash
git add src/features.py
git commit -m "Add recency-weighted mean trend stat (Home Credit source, RESOURCES.md)"
```

---

### Task 4: Add `build_net_flow_features()` (Zheng & Casari source)

**Files:**
- Modify: `src/config.py` (add `NET_FLOW_INFLOW_COLS`, `NET_FLOW_OUTFLOW_COLS`)
- Modify: `src/features.py` (add `build_net_flow_features()`, add `from . import config` import)

**Interfaces:**
- Consumes: `_trend_stats()` from Tasks 1-3 (now returning delta/mean/weighted_mean/std/min/max/ratio).
- Produces: `build_net_flow_features(df: pd.DataFrame, monthly_prefixes=("m1",...,"m6"), inflow_cols=None, outflow_cols=None) -> pd.DataFrame` with columns `net_flow_delta_m1_m6`, `net_flow_mean_6m`, `net_flow_weighted_mean_6m`, `net_flow_std_6m`, `net_flow_min_6m`, `net_flow_max_6m`, `net_flow_ratio_m1_mean`.

- [ ] **Step 1: Add config constants**

In `src/config.py`, after the `TOP_TREND_FAMILIES` block, add:

```python
# --- Net flow features ---
# Inflow minus outflow per month, using columns already in TOP_TREND_FAMILIES.
# Source: Zheng & Casari, "Feature Engineering for Machine Learning" (see
# RESOURCES.md), interaction features across variables -- net inflow vs
# outflow maps directly onto the definition of liquidity stress.
NET_FLOW_INFLOW_COLS = ["deposit_total_value", "received_total_value"]
NET_FLOW_OUTFLOW_COLS = ["withdraw_total_value"]
```

- [ ] **Step 2: Add `build_net_flow_features()`**

In `src/features.py`, add near the top: `from . import config` (after the existing `import numpy as np`). Then add this function after `build_monthly_trend_features()`:

```python
def build_net_flow_features(df: pd.DataFrame, monthly_prefixes=("m1", "m2", "m3", "m4", "m5", "m6"),
                             inflow_cols=None, outflow_cols=None) -> pd.DataFrame:
    """
    Net cash flow (inflow - outflow) per month, then the same trend
    statistics used for individual families (see _trend_stats).

    Source: Zheng & Casari, "Feature Engineering for Machine Learning"
    (see RESOURCES.md) -- an interaction feature across families rather
    than within one, capturing whether a customer is spending down more
    than they bring in, which is the definition of liquidity stress.
    """
    inflow_cols = inflow_cols if inflow_cols is not None else config.NET_FLOW_INFLOW_COLS
    outflow_cols = outflow_cols if outflow_cols is not None else config.NET_FLOW_OUTFLOW_COLS

    monthly_net_flow = pd.DataFrame(index=df.index)
    for m in monthly_prefixes:
        inflow = sum(df[f"{m}_{c}"] for c in inflow_cols)
        outflow = sum(df[f"{m}_{c}"] for c in outflow_cols)
        monthly_net_flow[m] = inflow - outflow

    stats = _trend_stats(monthly_net_flow, monthly_prefixes)
    return pd.DataFrame({f"net_flow_{name}": series for name, series in stats.items()}, index=df.index)
```

- [ ] **Step 3: Verify on synthetic data**

Run in a Python shell:

```python
import importlib, pandas as pd
from src import features
importlib.reload(features)

df = pd.DataFrame({
    "m1_deposit_total_value": [100.0], "m1_received_total_value": [20.0], "m1_withdraw_total_value": [30.0],
    "m2_deposit_total_value": [90.0],  "m2_received_total_value": [20.0], "m2_withdraw_total_value": [30.0],
    "m3_deposit_total_value": [80.0],  "m3_received_total_value": [20.0], "m3_withdraw_total_value": [30.0],
    "m4_deposit_total_value": [70.0],  "m4_received_total_value": [20.0], "m4_withdraw_total_value": [30.0],
    "m5_deposit_total_value": [60.0],  "m5_received_total_value": [20.0], "m5_withdraw_total_value": [30.0],
    "m6_deposit_total_value": [50.0],  "m6_received_total_value": [20.0], "m6_withdraw_total_value": [30.0],
})
out = features.build_net_flow_features(df)
# m1 net flow = 100 + 20 - 30 = 90; m6 net flow = 50 + 20 - 30 = 40
assert out["net_flow_delta_m1_m6"].iloc[0] == 50.0
assert set(out.columns) == {
    "net_flow_delta_m1_m6", "net_flow_mean_6m", "net_flow_weighted_mean_6m",
    "net_flow_std_6m", "net_flow_min_6m", "net_flow_max_6m", "net_flow_ratio_m1_mean",
}
print("net flow OK")
```

Expected: prints `net flow OK`.

- [ ] **Step 4: Commit**

```bash
git add src/config.py src/features.py
git commit -m "Add build_net_flow_features (Zheng & Casari interaction-feature source, RESOURCES.md)"
```

---

### Task 5: Validate combined score in the notebook

**Files:**
- Modify: `notebooks/01_eda_trends.ipynb` (add a new cell after the existing trend-feature validation cell)

**Interfaces:**
- Consumes: `features.build_monthly_trend_features()` (Tasks 1-3), `features.build_net_flow_features()` (Task 4), `evaluate.cross_validate_score()`, `model.build_lightgbm_calibrated_pipeline()`, `config.TOP_TREND_FAMILIES`, `config.CV_FOLDS`, `config.RANDOM_STATE` (all pre-existing).
- Produces: a combined-score comparison (before/after) that Task 6 depends on to decide whether to wire into `src/train.py`.

- [ ] **Step 1: Add a new notebook cell with the comparison**

Add this as a new cell at the end of `notebooks/01_eda_trends.ipynb` (adjust the import line if the notebook already has `from src import config, data, features, model, evaluate` in an earlier cell — don't duplicate it):

```python
from src import config, data, features, model, evaluate

train, test = data.load_raw_data()
feature_cols = features.get_feature_columns(train, config.ID_COL, config.TARGET)
X_encoded, _ = features.encode_features(train, test, feature_cols)

# Baseline: existing trend features only (current src/train.py behavior)
trend_only = features.build_monthly_trend_features(train, config.TOP_TREND_FAMILIES)
X_before = pd.concat([X_encoded, trend_only], axis=1)
X_before, _ = features.impute_missing(X_before, X_before.copy())
y = train[config.TARGET]

cv_before = evaluate.cross_validate_score(
    lambda: model.build_lightgbm_calibrated_pipeline(class_weight=None),
    X_before, y, n_splits=config.CV_FOLDS, random_state=config.RANDOM_STATE,
)
print("BEFORE (existing trend features only):")
evaluate.print_cv_summary(cv_before)

# After: existing trend features (now with min/max/weighted_mean) + net flow
net_flow = features.build_net_flow_features(train)
X_after = pd.concat([X_encoded, trend_only, net_flow], axis=1)
X_after, _ = features.impute_missing(X_after, X_after.copy())

cv_after = evaluate.cross_validate_score(
    lambda: model.build_lightgbm_calibrated_pipeline(class_weight=None),
    X_after, y, n_splits=config.CV_FOLDS, random_state=config.RANDOM_STATE,
)
print("AFTER (+ min/max + weighted_mean + net flow):")
evaluate.print_cv_summary(cv_after)
```

- [ ] **Step 2: Run the notebook cell and record the numbers**

Run all cells in `notebooks/01_eda_trends.ipynb` top to bottom (so `trend_only` reflects Tasks 1-3's new min/max/weighted_mean columns automatically, since `build_monthly_trend_features()`'s signature didn't change). Record the printed `combined_score`, Log Loss, and ROC-AUC for both BEFORE and AFTER.

- [ ] **Step 3: Commit the notebook**

```bash
git add notebooks/01_eda_trends.ipynb
git commit -m "Validate min/max, weighted_mean, and net-flow features against combined score"
```

---

### Task 6: Wire into `src/train.py` and update `README.md` (conditional on Task 5's result)

Only do this task if Task 5 showed the combined score improve, or at minimum not regress on both Log Loss and ROC-AUC. If it regressed, stop here and report the numbers instead — do not wire in features that made the score worse.

**Files:**
- Modify: `src/train.py:23-28` (the trend-features block)
- Modify: `README.md` (Progress and Next steps sections)

**Interfaces:**
- Consumes: `features.build_net_flow_features()` (Task 4), `features.build_monthly_trend_features()` (Tasks 1-3, unchanged call signature).

- [ ] **Step 1: Add net-flow features to the training pipeline**

In `src/train.py`, replace:

```python
    # 2b. Trend features (validated in notebooks/01_eda_trends.ipynb, cell 6:
    # improved combined score from 0.326 to 0.310)
    trend_train = features.build_monthly_trend_features(train, config.TOP_TREND_FAMILIES)
    trend_test = features.build_monthly_trend_features(test, config.TOP_TREND_FAMILIES)
    X_encoded = pd.concat([X_encoded, trend_train], axis=1)
    test_encoded = pd.concat([test_encoded, trend_test], axis=1)
```

with:

```python
    # 2b. Trend features (validated in notebooks/01_eda_trends.ipynb, cell 6:
    # improved combined score from 0.326 to 0.310; extended with min/max,
    # weighted_mean, and net-flow features, see cell 7 -- combined score
    # <FILL IN FROM TASK 5's AFTER RESULT>)
    trend_train = features.build_monthly_trend_features(train, config.TOP_TREND_FAMILIES)
    trend_test = features.build_monthly_trend_features(test, config.TOP_TREND_FAMILIES)
    net_flow_train = features.build_net_flow_features(train)
    net_flow_test = features.build_net_flow_features(test)
    X_encoded = pd.concat([X_encoded, trend_train, net_flow_train], axis=1)
    test_encoded = pd.concat([test_encoded, trend_test, net_flow_test], axis=1)
```

Replace `<FILL IN FROM TASK 5's AFTER RESULT>` with the actual combined score printed in Task 5, Step 2.

- [ ] **Step 2: Run the full pipeline to confirm it still works end-to-end**

```bash
python -m src.train
```

Expected: prints the CV summary (should match Task 5's AFTER numbers) and finishes without error, regenerating `submissions/submission.csv`.

- [ ] **Step 3: Update `README.md`**

In the "Progress" section, add a bullet after the existing calibration bullet, describing the new features and the before/after combined score from Task 5 (use the real numbers, not placeholders).

In the "Next steps (pending)" section, check off:

```markdown
- [x] Explore additional feature engineering beyond the current top 13
      trend families (`config.TOP_TREND_FAMILIES`).
```

- [ ] **Step 4: Commit**

```bash
git add src/train.py README.md
git commit -m "Wire min/max, weighted_mean, and net-flow features into src/train.py"
```
