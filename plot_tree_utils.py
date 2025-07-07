import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

def plot_feature_importance(model, feature_names, model_name, top_n=20, save_path=None):
    importances = model.feature_importances_
    indices = importances.argsort()[::-1][:top_n]

    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_importances, y=top_features, palette='viridis')
    plt.title(f"Top {top_n} Feature Importances - {model_name}")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()
