"""
Main script: loads data, trains the baseline, evaluates, and generates the submission.

Usage:
    python -m src.train
"""


import pandas as pd

from . import config, data, features, model, evaluate


def main():
    # 1. Load data
    train, test = data.load_raw_data()
    sample_submission = data.make_sample_submission(test)

    # 2. Prepare features
    feature_cols = features.get_feature_columns(train, config.ID_COL, config.TARGET)
    X_encoded, test_encoded = features.encode_features(train, test, feature_cols)

    # 2b. Trend features (validated in notebooks/01_eda_trends.ipynb, cell 6:
    # improved combined score from 0.326 to 0.310)
    trend_train = features.build_monthly_trend_features(train, config.TOP_TREND_FAMILIES)
    trend_test = features.build_monthly_trend_features(test, config.TOP_TREND_FAMILIES)
    X_encoded = pd.concat([X_encoded, trend_train], axis=1)
    test_encoded = pd.concat([test_encoded, trend_test], axis=1)

    X_encoded, test_encoded = features.impute_missing(X_encoded, test_encoded)

    y = train[config.TARGET]

    # 3. Train/validation split
    # 3-5. Stratified K-Fold cross-validation (more robust than a single
    # split -- see RESOURCES.md, Home Credit 1st place solution)
    cv_summary = evaluate.cross_validate_score(
        lambda: model.build_lightgbm_calibrated_pipeline(class_weight=None, **config.TUNED_LGBM_PARAMS),
        X_encoded, y,
        n_splits=config.CV_FOLDS,
        random_state=config.RANDOM_STATE,
    )
    evaluate.print_cv_summary(cv_summary)

    # 6. Retrain on all the data and predict on test
    pipeline = model.build_lightgbm_calibrated_pipeline(class_weight=None, **config.TUNED_LGBM_PARAMS)
    pipeline.fit(X_encoded, y)
    assert list(test_encoded.columns) == list(X_encoded.columns), \
        "Train and test columns do not match after encoding."

    test_predictions = pipeline.predict_proba(test_encoded)[:, 1]

    # 7. Save submission
    submission = sample_submission.copy()
    submission[config.SUBMISSION_TARGET] = test_predictions
    data.save_submission(submission)


if __name__ == "__main__":
    main()
