import pandas as pd
import numpy as np

# Handling missing values

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values based on column types and missing patterns.
    """

    # Fill LotFrontage using median by Neighborhood
    if "LotFrontage" in df.columns and "Neighborhood" in df.columns:
        df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(
            lambda x: x.fillna(x.median())
        )

    # Fill categorical columns with 'None' if >20% missing
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        if df[col].isnull().mean() > 0.2:
            df[col] = df[col].fillna("None")

    # Fill numeric columns with 0 if >20% missing
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    for col in num_cols:
        if df[col].isnull().mean() > 0.2:
            df[col] = df[col].fillna(0)

    # Fill remaining categorical columns with mode
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Fill remaining numeric columns with median
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    return df


# Encoding categorical variables

def encode_categorical_variables(df: pd.DataFrame) -> pd.DataFrame:
    # -----------------------------
    # 1. Ordinal Encoding
    # -----------------------------
    qual_mapping = {
        "Ex": 5,
        "Gd": 4,
        "TA": 3,
        "Fa": 2,
        "Po": 1,
        "None": 0
    }

    ord_features = [
        "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC","KitchenQual",
        "FireplaceQu", "GarageQual", "GarageCond", "PoolQC"]
    
    for col in ord_features:
        if col in df.columns:
            df[col] = df[col].map(qual_mapping)


    bsmt_exposure_map = {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'None': 0}
    if 'BsmtExposure' in df.columns:
        df['BsmtExposure'] = df['BsmtExposure'].map(bsmt_exposure_map)

    bsmt_finish_map = {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'None': 0}
    for col in ['BsmtFinType1', 'BsmtFinType2']:
        if col in df.columns:
            df[col] = df[col].map(bsmt_finish_map)

    # -----------------------------
    # 2. Binary Encoding
    # -----------------------------
    binary_map = {'Y': 1, 'N': 0, 'Yes': 1, 'No': 0, 'True': 1, 'False': 0, 'T': 1, 'F': 0}
    if 'CentralAir' in df.columns:
        df['CentralAir'] = df['CentralAir'].map(binary_map)

    # -----------------------------
    # 3. One-Hot Encoding
    # -----------------------------

    cat_cols = df.select_dtypes(include='object').columns
    pd.get_dummies(df, drop_first=True, columns=cat_cols)

    print("Categorical feature encoding is complete.")
    return df


# Handling Skewness

from scipy.stats import skew

def handle_skewed_features(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:

    numeric_feats = df.select_dtypes(include = ['int64', 'float64']).drop('SalePrice', axis=1, errors='ignore')

    skewness = numeric_feats.apply(lambda x: skew(x.dropna()))
    skewed_feats = skewness[skewness > 0.75].index

    print("Skewed features detected: ", len(skewed_feats))
    print(skewed_feats)
    print("Skewness values: ", skewness[skewed_feats])

    df[skewed_feats] = np.log1p(df[skewed_feats])

    return df


# Outlier Handling

def detect_outliers_iqr(df: pd.DataFrame, column: str, threshold: float = 1.5) -> pd.Series:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR

    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)

    return outliers

def cap_outliers(df, column): 
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df[column] = df[column].apply(lambda x: lower_bound if x < lower_bound else (upper_bound if x > upper_bound else x))
    return df

def handle_outliers(df):
    outlier_cols = ['Gr Liv Area', 'Lot Area', 'Total Bsmt SF', '1st Flr SF', 'Garage Area']

    for col in outlier_cols:
        df = cap_outliers(df, col)

    return df 


# Feature Scaling
from sklearn.preprocessing import StandardScaler

def scale_numerical_features(df):
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    if 'SalePrice' in numerical_cols:
        numerical_cols.remove('SalePrice')

    scaler = StandardScaler()

    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = handle_missing_values(df)
    df = encode_categorical_variables(df)
    df = handle_skewed_features(df)
    df = handle_outliers(df)
    df = scale_numerical_features(df)
    return df
