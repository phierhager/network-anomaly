from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)


def train_and_evaluate(X_train, X_test, y_train, y_test, model_name):
    """Train and evaluate a machine learning model."""
    # Choose model
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=100, random_state=42
        ),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=42
        ),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
    }

    if model_name not in models:
        raise ValueError(
            "Model not recognized. Supported: RandomForest, SVM, LogisticRegression, DecisionTree"
        )

    model = models[model_name]
    # print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    # print("\nClassification Report:\n", classification_report(y_test, y_pred))
    # print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    # print("Accuracy Score:", accuracy_score(y_test, y_pred))

    return model
