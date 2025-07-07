
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.impute import SimpleImputer

from src.config import PROCESSED_DATA_PATH, LINEAR_MODEL_PATH, RIDGE_MODEL_PATH, LASSO_MODEL_PATH
from src.evaluation import evaluate_model, print_evaluation

# 1. Load processed data
df = pd.read_csv(PROCESSED_DATA_PATH)
df = pd.get_dummies(df, drop_first=True)

# 2. Define X and y
drop_cols = [col for col in ['SalePrice', 'PID', 'Order'] if col in df.columns]
X = df.drop(columns=drop_cols)
y = df['SalePrice']

# 3. Handle missing values
imputer = SimpleImputer(strategy='median')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Define models
models = {
    "Linear Regression": (LinearRegression(), LINEAR_MODEL_PATH),
    "Ridge Regression": (Ridge(alpha=1.0), RIDGE_MODEL_PATH),
    "Lasso Regression": (Lasso(alpha=0.01), LASSO_MODEL_PATH)
}

# 6. Train, evaluate, and save each model
for name, (model, path) in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = evaluate_model(y_test, y_pred)
    print_evaluation(name, metrics)

    joblib.dump(model, path)
    print(f"Model saved to: {path}")
