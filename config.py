"""
Configuration file for wildfire prediction project.
Contains hyperparameters, paths, and model settings.
"""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories
for dir in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, FEATURES_DIR, MODELS_DIR, RESULTS_DIR]:
    dir.mkdir(exist_ok=True, parents=True)

# Data Configuration
TARGET_COLUMN = 'fire_occurred'
DATE_COLUMN = 'date'
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1

# Feature groups
BASE_WEATHER_FEATURES = [
    'temp_max', 'temp_min', 'temp_avg',
    'relative_humidity', 'wind_speed', 'precipitation'
]

ENVIRONMENTAL_FEATURES = [
    'ndvi', 'elevation', 'distance_to_road', 'latitude', 'longitude'
]

TEMPORAL_FEATURES = [
    'month', 'day_of_week', 'day_of_year',
    'month_sin', 'month_cos', 'day_of_year_sin', 'day_of_year_cos',
    'day_of_week_sin', 'day_of_week_cos', 'is_summer'
]

# Model hyperparameters
LOGISTIC_REGRESSION_PARAMS = {
    'C': 1.0,
    'max_iter': 1000,
    'random_state': RANDOM_STATE
}

RANDOM_FOREST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 15,
    'min_samples_split': 10,
    'min_samples_leaf': 4,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

XGBOOST_PARAMS = {
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_STATE,
    'eval_metric': 'logloss'
}

LIGHTGBM_PARAMS = {
    'n_estimators': 200,
    'max_depth': 7,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_STATE,
    'verbose': -1
}

# Evaluation metrics
CLASSIFICATION_METRICS = [
    'accuracy', 'precision', 'recall', 'f1',
    'roc_auc', 'average_precision'
]

# Optuna hyperparameter tuning
OPTUNA_N_TRIALS = 50
OPTUNA_TIMEOUT = 3600  # 1 hour

# SHAP configuration
SHAP_SAMPLE_SIZE = 100  # Number of samples for SHAP analysis
