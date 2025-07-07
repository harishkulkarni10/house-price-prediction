# src/evaluation.py

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def evaluate_model(y_true, y_pred):
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    }

def print_evaluation(metrics):
    print(f"RMSE: {metrics['RMSE']:.2f}")
    print(f"MAE : {metrics['MAE']:.2f}")
    print(f"R²  : {metrics['R2']:.2f}")