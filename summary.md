**House Price Prediction**

**Project Summary**

- This project aims to predict housing prices using the Ames Housing dataset, which contains a variety of features describing residential homes in Ames, Iowa. The goal was to develop a robust regression model that generalizes well and provides interpretable insights into what affects house prices the most.

**Dataset Overview**
Source: AmesHousing.csv
Size: ~1,460 rows × 80+ columns
Target Variable: SalePrice

- The dataset includes numeric and categorical variables covering zoning, lot size, building type, quality ratings, square footage, and more.

**Key Steps Performed**
**Data Preprocessing**

- Handled missing values using median imputation and logical domain knowledge
- Encoded categorical features using one-hot encoding
- Treated skewness using log transformation
- Scaled numerical values for linear models
- Detected and treated outliers based on domain context

**Model Building**

- Trained both linear and tree-based regression models:
- Linear Regression, Ridge, Lasso
- Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM
- Used train_test_split and imputation consistently across models

**Evaluation Metrics**

- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score

**Model Comparison**
**Linear models**

- Linear Regression: RMSE: 23,954.59, MAE: 15,634.68, R2 Score: 0.91
- Ridge Regression: RMSE: 25,274.21, MAE: 16,264.36, R2 Score: 0.90
- Lasso Regression: RMSE: 23,996.96, MAE: 15,694.97, R2 Score: 0.91

  **Tree models**

- Dcision Tree: RMSE: 17,751.69, MAE: 4,814.62, R2 Score: 0.95
- Random Forest: RMSE: 15,082.63, MAE: 7,870.78, R2 Score: 0.96
- Gradient Boosting: RMSE: 18,272.39, MAE: 12,301.59, R2 Score: 0.95
- XGBoost: RMSE: 11,548.11, MAE: 5,010.94, R2 Score: 0.98
- LightGBM: RMSE: 15,527.17, MAE: 8,580.94, R2 Score: 0.96

**Final Observations**

- XGBoost outperformed all other models in terms of RMSE and R².
- Linear models were interpretable but showed higher error.
- Tree-based models captured non-linear patterns more effectively.
- Features like GrLivArea, OverallQual, and GarageCars had significant influence on price.
