import pandas as pd
import numpy as np
import joblib

from src.data_loader import load_data, save_data
from src.preprocessing import (
    handle_missing_values,
    encode_categorical_variables,
    handle_skewed_features,
    handle_outliers,
    scale_numerical_features
)
from src.config import PROCESSED_DATA_PATH, TREE_MODEL_PATHS
from src.evaluation import evaluate_model, print_evaluation

from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from plot_linear_utils import (
    plot_linear_diagnostics,
    plot_lasso_coefficients,
    plot_feature_coefficients
)

from plot_tree_utils import plot_feature_importance

# --------------------------------------------------------------------------
# 1. Preprocess and Save Cleaned Data
# --------------------------------------------------------------------------
def preprocess_and_save():
    df = load_data("data/raw/AmesHousing.csv")
    df = handle_missing_values(df)
    df = encode_categorical_variables(df)
    df = handle_skewed_features(df)
    df = handle_outliers(df)
    df = scale_numerical_features(df)
    save_data(df, "data/processed/processed_data.csv")
    print("Processed data saved.")

# --------------------------------------------------------------------------
# 2. Evaluate Linear Models and Save Plots
# --------------------------------------------------------------------------
def evaluate_linear_models():
    df = pd.read_csv("data/processed/processed_data_with_features.csv")
    df = pd.get_dummies(df, drop_first=True)

    drop_cols = [col for col in ['SalePrice', 'PID', 'Order'] if col in df.columns]
    X = df.drop(columns=drop_cols)
    y = df['SalePrice']

    imputer = SimpleImputer(strategy='median')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    linear_models = {
        'LinearRegression': 'models/linear_regression.pkl',
        'RidgeRegression': 'models/ridge_regression.pkl',
        'LassoRegression': 'models/lasso_regression.pkl'
    }

    for name, path in linear_models.items():
        print(f"\n{name} Evaluation")
        model = joblib.load(path)
        y_pred = model.predict(X)

        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        print(f"RMSE: {rmse:.2f}")
        print(f"MAE : {mae:.2f}")
        print(f"R²  : {r2:.2f}")

        # Save feature plots
        plot_path = f"reports/coefficients_{name.lower()}.png"
        plot_feature_coefficients(X, model, title=f"{name} Coefficients", save_path=plot_path)

    # Lasso special plot for top features
    lasso_model = joblib.load(linear_models['LassoRegression'])
    plot_lasso_coefficients(X, lasso_model, save_path="reports/coefficients_lasso_top.png")

# --------------------------------------------------------------------------
# 3. Evaluate Tree Models and Save Feature Importances
# --------------------------------------------------------------------------
def evaluate_tree_models():
    df = pd.read_csv("data/processed/processed_data_with_features.csv")
    df = pd.get_dummies(df, drop_first=True)

    drop_cols = [col for col in ['SalePrice', 'PID', 'Order'] if col in df.columns]
    X = df.drop(columns=drop_cols)
    y = df['SalePrice']

    imputer = SimpleImputer(strategy='median')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    tree_model_names = ['DecisionTree', 'RandomForest', 'GradientBoosting', 'XGBoost', 'LightGBM']

    for name in tree_model_names:
        print(f"\n{name} Regression")
        model_path = TREE_MODEL_PATHS.get(name)
        if model_path:
            model = joblib.load(model_path)
            y_pred = model.predict(X)

            metrics = evaluate_model(y, y_pred)
            print_evaluation(metrics)

            # Save feature importance plot
            plot_path = f"reports/feature_importance_{name.lower()}.png"
            plot_feature_importance(model, X.columns, name, save_path=plot_path)
        else:
            print(f"Model not found: {name}")

# --------------------------------------------------------------------------
if __name__ == "__main__":
    preprocess_and_save()
    evaluate_linear_models()
    evaluate_tree_models()
