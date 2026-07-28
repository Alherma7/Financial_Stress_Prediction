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


def build_monthly_trend_features(df: pd.DataFrame, families: list,  monthly_prefixes=("m1", "m2", "m3", "m4", "m5", "m6")) -> pd.DataFrame:
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

        most_recent = df[f"{monthly_prefixes[0]}_{fam}"]
        oldest = df[f"{monthly_prefixes[-1]}_{fam}"]
        mean_6m = monthly_values.mean(axis=1)
        std_6m = monthly_values.std(axis=1)

        trend_df[f"{fam}_delta_m1_m6"] = most_recent - oldest
        trend_df[f"{fam}_mean_6m"] = mean_6m
        trend_df[f"{fam}_std_6m"] = std_6m
         # guard against dividing by a zero mean (e.g. a customer with no
        # deposits at all across the window)
        trend_df[f"{fam}_ratio_m1_mean"] = (most_recent / mean_6m).replace([np.inf, -np.inf], np.nan)

    return trend_df
