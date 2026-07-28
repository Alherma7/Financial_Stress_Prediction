"""
Main script: loads data, trains the baseline, evaluates, and generates the submission.

Usage:
    python -m src.train
"""

from sklearn.model_selection import train_test_split

from . import config, data, features, model, evaluate


def main():
    # 1. Load data
    train, test = data.load_raw_data()
    sample_submission = data.make_sample_submission(test)

    # 2. Prepare features
    feature_cols = features.get_feature_columns(train, config.ID_COL, config.TARGET)
    X_encoded, test_encoded = features.encode_features(train, test, feature_cols)
    X_encoded, test_encoded = features.impute_missing(X_encoded, test_encoded)

    y = train[config.TARGET]

    # 3. Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X_encoded, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )

    # 4. Train baseline
    pipeline = model.build_logreg_pipeline(class_weight=None)
    pipeline.fit(X_train, y_train)

    # 5. Evaluate with the actual competition metric
    val_pred_prob = pipeline.predict_proba(X_val)[:, 1]
    scores = evaluate.combined_score(y_val, val_pred_prob)
    evaluate.print_scores(scores)

    # 6. Retrain on all the data and predict on test
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
