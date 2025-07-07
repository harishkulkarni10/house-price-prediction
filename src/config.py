# src/config.py

import os

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "AmesHousing.csv")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed", "processed_data_with_features.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Model paths
LINEAR_MODEL_PATH = os.path.join(MODELS_DIR, "linear_regression.pkl")
RIDGE_MODEL_PATH = os.path.join(MODELS_DIR, "ridge_regression.pkl")
LASSO_MODEL_PATH = os.path.join(MODELS_DIR, "lasso_regression.pkl")

TREE_MODEL_PATHS = {
    'DecisionTree': "models/decision_tree.pkl",
    'RandomForest': "models/random_forest.pkl",
    'GradientBoosting': "models/gradient_boosting.pkl",
    'XGBoost': "models/xgboost.pkl",
    'LightGBM': "models/lightgbm.pkl"
}
