from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_confusion_matrix(y_test, y_pred, model_name):
    """Plot confusion matrix."""
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


def feature_importance(model, X_columns):
    """Plot feature importance (for tree-based models)."""
    if hasattr(model, "feature_importances_"):
        importances = pd.DataFrame(
            model.feature_importances_, index=X_columns, columns=["Importance"]
        )
        importances = importances.sort_values("Importance", ascending=False)
        # print("\nTop Features:\n", importances.head(10))

        importances.head(10).plot(kind="barh", figsize=(10, 6), color="teal")
        plt.title("Top 10 Feature Importances")
        plt.xlabel("Importance Score")
        plt.ylabel("Features")
        plt.gca().invert_yaxis()
        plt.show()
