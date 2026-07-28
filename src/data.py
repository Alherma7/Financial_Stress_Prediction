"""
Raw data loading and submission-saving utilities.
"""

import pandas as pd

from . import config


def load_raw_data():
    """Load Train.csv and Test.csv from data/raw/."""
    train = pd.read_csv(config.TRAIN_PATH)
    test = pd.read_csv(config.TEST_PATH)
    return train, test


def make_sample_submission(test: pd.DataFrame) -> pd.DataFrame:
    """Create a submission template from the test IDs."""
    return pd.DataFrame({
        config.ID_COL: test[config.ID_COL],
        config.SUBMISSION_TARGET: 0.5,
    })


def save_submission(submission: pd.DataFrame, filename: str = "submission.csv"):
    """Save the submission file to submissions/."""
    out_path = config.SUBMISSIONS_DIR / filename
    submission.to_csv(out_path, index=False)
    print(f"Submission saved to: {out_path}")
    return out_path
