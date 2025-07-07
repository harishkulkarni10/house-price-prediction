import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import pandas as pd
import numpy as np
import os

def plot_linear_diagnostics(y, y_pred, residuals, df=None, feature=None, model_name="Linear Regression", save_path=None):
    plt.figure(figsize=(18, 10))

    # 1. Actual vs Predicted
    plt.subplot(2, 3, 1)
    sns.scatterplot(x=y, y=y_pred, alpha=0.6, color='teal')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], '--', color='red')
    plt.xlabel("Actual Sale Price")
    plt.ylabel("Predicted Sale Price")
    plt.title("Actual vs Predicted")

    # 2. Residuals vs Predicted
    plt.subplot(2, 3, 2)
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.5, color='orange')
    plt.axhline(0, linestyle='--', color='red')
    plt.xlabel("Predicted Sale Price")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Predicted")

    # 3. Histogram of Residuals
    plt.subplot(2, 3, 3)
    sns.histplot(residuals, kde=True, color='purple')
    plt.xlabel("Residuals")
    plt.title("Distribution of Residuals")

    # 4. QQ Plot (Normality of residuals)
    plt.subplot(2, 3, 4)
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title("QQ Plot of Residuals")

    # 5. Feature vs Target (like GrLivArea if passed)
    plt.subplot(2, 3, 5)
    if df is not None and feature is not None and feature in df.columns:
        sns.scatterplot(x=df[feature], y=y, alpha=0.5)
        sns.lineplot(x=df[feature], y=y_pred, color='green', linewidth=1)
        plt.xlabel(feature)
        plt.ylabel("SalePrice")
        plt.title(f"{feature} vs SalePrice")
    else:
        plt.text(0.3, 0.5, "Feature not available", fontsize=12)

    plt.tight_layout()
    plt.suptitle(f"{model_name} Diagnostic Plots", fontsize=16, y=1.02)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_lasso_coefficients(X, model, top_n=10, save_path=None):
    coef_df = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_
    })
    coef_df = coef_df[coef_df['Coefficient'] != 0] 
    coef_df = coef_df.sort_values(by='Coefficient', key=abs, ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=coef_df, x='Coefficient', y='Feature', palette='viridis')
    plt.title('Top Important Features (Lasso)')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_feature_coefficients(X, model, title='Top 20 Influential Features', top_n=20, save_path=None):
    coef_df = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_
    })
    coef_df['Abs_Coefficient'] = coef_df['Coefficient'].abs()
    coef_df = coef_df.sort_values(by='Abs_Coefficient', ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=coef_df, x='Coefficient', y='Feature', palette='coolwarm')
    plt.title(title)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()
