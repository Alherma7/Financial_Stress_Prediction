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
CV_FOLDS = 5

# --- Competition metric weights ---
# Final score = LOGLOSS_WEIGHT * log_loss + AUC_WEIGHT * (1 - auc)
# (see src/evaluate.py for details on how it's combined)
LOGLOSS_WEIGHT = 0.6
AUC_WEIGHT = 0.4

# --- Trend features ---
# Top 13 families by trend-signal strength (|AUC - 0.5| of the raw m1-m6 delta),
# selected in notebooks/01_eda_trends.ipynb (Cell 5). Validated there: adding
# trend features for these families improved the validation combined score
# from 0.326 to 0.310. Technique: delta/mean/std/last-vs-mean ratio, following
# the American Express - Default Prediction approach (see RESOURCES.md).
TOP_TREND_FAMILIES = [
    "daily_avg_bal", "deposit_total_value", "deposit_volume", "deposit_agents",
    "deposit_highest_amount", "withdraw_volume", "received_total_value",
    "received_volume", "withdraw_highest_amount", "withdraw_total_value",
    "received_senders", "withdraw_agents", "received_highest_amount",
]

# --- Net flow features ---
# Inflow minus outflow per month, using columns already in TOP_TREND_FAMILIES.
# Source: Zheng & Casari, "Feature Engineering for Machine Learning" (see
# RESOURCES.md), interaction features across variables -- net inflow vs
# outflow maps directly onto the definition of liquidity stress.
NET_FLOW_INFLOW_COLS = ["deposit_total_value", "received_total_value"]
NET_FLOW_OUTFLOW_COLS = ["withdraw_total_value"]