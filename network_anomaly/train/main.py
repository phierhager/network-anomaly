from data import load_dataset, preprocess_data
from sklearn.model_selection import train_test_split
from train import train_and_evaluate
from plot import plot_confusion_matrix, feature_importance
import joblib


def main(dataset_name, target_column, model_name):
    """Main function to orchestrate loading, preprocessing, training, and evaluation."""
    # Load dataset
    df = load_dataset(dataset_name)

    # Preprocess data
    X, y = preprocess_data(df, target_column)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Train and evaluate the model
    model = train_and_evaluate(X_train, X_test, y_train, y_test, model_name)

    # Plot confusion matrix
    y_pred = model.predict(X_test)
    plot_confusion_matrix(y_test, y_pred, model_name)

    # Plot feature importance (if applicable)
    feature_importance(model, df.drop(target_column, axis=1).columns)

    # Save the model
    joblib.dump(model, f"{model_name}_{dataset_name}_model.pkl")
    print(f"\nModel saved as '{model_name}_{dataset_name}_model.pkl'")


# Run the script
if __name__ == "__main__":
    # Example: Choose dataset, target column, and model
    dataset_name = "CICIDS2017"  # Replace with your dataset name
    target_column = "label"  # Replace with your target column
    model_name = "RandomForest"  # Replace with desired model name

    main(dataset_name, target_column, model_name)
