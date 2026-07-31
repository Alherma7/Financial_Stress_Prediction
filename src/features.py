"""
Feature engineering.

This module separates two responsibilities:
1. encode_features(): basic encoding (one-hot + train/test alignment),
   same as the starter notebook.
2. build_monthly_trend_features(): PENDING completion once we confirm
   the actual monthly column names (m1_, m2_, ... m6_).

The idea behind build_monthly_trend_features is to capture signals such as:
- Is the average balance dropping month over month?
- Is transactional activity decreasing?
- Is the withdrawals/deposits ratio getting worse?
"""

import pandas as pd
import numpy as np
from . import config

def get_feature_columns(df: pd.DataFrame, id_col: str, target_col: str) -> list:
    """Return the feature columns, excluding ID and target."""
    return [c for c in df.columns if c not in [id_col, target_col]]


def encode_features(train: pd.DataFrame, test: pd.DataFrame, feature_cols: list):
    """
    One-hot encoding of categorical columns + alignment of columns
    between train and test. Equivalent to what the starter notebook does.
    """
    X_raw = train[feature_cols].copy()
    test_raw = test[feature_cols].copy()

    cat_cols = [c for c in feature_cols if train[c].dtype == "object"]

    X_encoded = pd.get_dummies(X_raw, columns=cat_cols, drop_first=True)
    test_encoded = pd.get_dummies(test_raw, columns=cat_cols, drop_first=True)

    X_encoded, test_encoded = X_encoded.align(
        test_encoded, join="left", axis=1, fill_value=0
    )

    return X_encoded, test_encoded


def impute_missing(X_encoded: pd.DataFrame, test_encoded: pd.DataFrame):
    """Impute missing values with the median computed on train."""
    for col in X_encoded.columns:
        median_value = X_encoded[col].median()
        X_encoded[col] = X_encoded[col].fillna(median_value)
        test_encoded[col] = test_encoded[col].fillna(median_value)
    return X_encoded, test_encoded
def _linear_recency_weights(n: int) -> np.ndarray:
    """
    Linearly decreasing weights, index 0 (most recent month) heaviest,
    normalized to sum to 1. E.g. n=6 -> [6, 5, 4, 3, 2, 1] / 21.

    Source: Home Credit Default Risk 1st place solution used "weighted
    moving averages on time-based features" (see RESOURCES.md) without
    specifying exact weights; linear decay is this project's 
    implementation choice, weighting months closer to the 30-day
    prediction window more heavily (monthly_prefixes[0] = m1 = most_recent).
    """
    weights = np.arange(n, 0, -1, dtype=float)
    return weights / weights.sum()

def _trend_stats(monthly_values: pd.DataFrame, monthly_prefixes) -> dict:
    """
    Shared trend statistics for a monthly values series (one column per
    entry in monthly_prefixes, ordered most-recent -> oldest, i.e. column 0
    corresponds to monthly_prefixes[0] = m1 = most recent).
    """
    most_recent = monthly_values.iloc[:, 0]
    oldest = monthly_values.iloc[:,-1]
    mean_6m = monthly_values.mean(axis=1)

    weights = _linear_recency_weights(len(monthly_prefixes))
    weighted_mean_6m = (monthly_values * weights).sum(axis=1)

    return {
        "delta_m1_m6": most_recent - oldest,
        "mean_6m": mean_6m,
        "std_6m": monthly_values.std(axis=1),
        "min_6m": monthly_values.min(axis=1),
        "max_6m": monthly_values.max(axis=1),
        "ratio_m1_mean": (most_recent / mean_6m).replace([np.inf, -np.inf], np.nan),
        "weighted_mean_6m": weighted_mean_6m
    }

def build_monthly_trend_features(df: pd.DataFrame, families: list, monthly_prefixes=("m1","m2","m3",
"m4","m5", "m6")) -> pd.DataFrame:
    """
    Build trend features per variable family across the 6-month window.

    Source: mean/std/last-value aggregation + delta and ratio between the last
    value and the mean, following the American Express - Default Prediction
    top solutions (see RESOURCES.md). The `_delta_m1_m6` feature was validated
    in notebooks/01_eda_trends.ipynb (Cell 4): ranking all 29 families by the
    ROC-AUC of this single delta against the target showed daily_avg_bal,
    deposit_*, withdraw_* and received_* as the strongest signals.

    Note: monthly_prefixes is ordered most-recent -> oldest (m1 = most recent,
    m6 = oldest, per data_dictionary.csv), so monthly_prefixes[0] is "most
    recent" and monthly_prefixes[-1] is "oldest".
    """
    trend_df = pd.DataFrame(index=df.index)
    for fam in families:
        cols = [f"{m}_{fam}" for m in monthly_prefixes]
        monthly_values = df[cols]
        stats = _trend_stats(monthly_values, monthly_prefixes)
        for stat_name, series in stats.items():
            trend_df[f"{fam}_{stat_name}"] = series
    return trend_df

def build_production_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    The current production feature set: one-hot encoding + the top trend
    families (config.TOP_TREND_FAMILIES), imputed with the train median.

    Combines encode_features() + build_monthly_trend_features() +
    impute_missing() -- the same three-step sequence used by both
    src/train.py and the notebook's validation cells -- so that sequence
    only needs to be written once. Callers who only need train features
    (e.g. a notebook cell scoring `train` alone) can pass `test_df` anyway
    and ignore the second return value.
    """
    feature_cols = get_feature_columns(train_df, config.ID_COL, config.TARGET)
    X_encoded, test_encoded = encode_features(train_df, test_df, feature_cols)

    trend_train = build_monthly_trend_features(train_df, config.TOP_TREND_FAMILIES)
    trend_test = build_monthly_trend_features(test_df, config.TOP_TREND_FAMILIES)
    X_encoded = pd.concat([X_encoded, trend_train], axis=1)
    test_encoded = pd.concat([test_encoded, trend_test], axis=1)

    return impute_missing(X_encoded, test_encoded)


def build_net_flow_features(df: pd.DataFrame, monthly_prefixes=("m1","m2","m3","m4","m5","m6"),
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
