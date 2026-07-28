"""
Central project configuration.
Change paths, column names, and constants here in a single place.
"""

from pathlib import Path

# --- Paths ---
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
SUBMISSIONS_DIR = ROOT_DIR / "submissions"

TRAIN_PATH = DATA_RAW_DIR / "Train.csv"
TEST_PATH = DATA_RAW_DIR / "Test.csv"

# --- Key columns ---
ID_COL = "ID"
TARGET = "liquidity_stress_next_30d"       # actual target used for training/validation
SUBMISSION_TARGET = "Target"               # column name required by Zindi

# --- Reproducibility ---
RANDOM_STATE = 42
TEST_SIZE = 0.2  # validation split size

# --- Competition metric weights ---
# Final score = LOGLOSS_WEIGHT * log_loss + AUC_WEIGHT * (1 - auc)
# (see src/evaluate.py for details on how it's combined)
LOGLOSS_WEIGHT = 0.6
AUC_WEIGHT = 0.4
