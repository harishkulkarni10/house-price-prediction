import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer

from src.config import PROCESSED_DATA_PATH, TREE_MODEL_PATHS
from src.evaluation import evaluate_model, print_evaluation

def train_and_save(model, name, X_train, X_test, y_train, y_test):
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = evaluate_model(y_test, y_pred)
    print_evaluation(metrics)

    model_path = TREE_MODEL_PATHS.get(name)
    if model_path:
        joblib.dump(model, model_path)
        print(f"Model saved to: {model_path}")
    else:
        print(f"Path not defined for model: {name}")

def main():
    df = pd.read_csv(PROCESSED_DATA_PATH)
    df = pd.get_dummies(df, drop_first=True)

    X = df.drop(columns=['SalePrice', 'PID', 'Order'], errors='ignore')
    y = df['SalePrice']

    # Impute missing values (required for models like GradientBoosting, XGBoost, etc.)
    imputer = SimpleImputer(strategy='median')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'DecisionTree': DecisionTreeRegressor(random_state=42),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, random_state=42),
        'LightGBM': LGBMRegressor(n_estimators=100, random_state=42)
    }

    for name, model in models.items():
        train_and_save(model, name, X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
