from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
from pathlib import Path


def load_dataset(dataset_path):
    """Load different datasets based on user input."""
    df = pd.read_parquet(dataset_path)
    dataset_name = dataset_path.split("/")[-1].split(".")[0]
    print(f"Loaded {dataset_name} dataset with shape: {df.shape}")
    return df


def preprocess_data(df, target_column):
    """Clean and preprocess the data for modeling."""
    # strip and rename columns
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace(" ", "_")
    df.rename(columns={"Label": "label"}, inplace=True)

    # Encode categorical features
    categorical_cols = df.select_dtypes(include=["category"]).columns
    for col in categorical_cols:
        if col != target_column:  # Exclude target column from feature encoding
            df[col] = LabelEncoder().fit_transform(df[col])

    # Separate features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Scale numeric features
    numeric_cols = X.select_dtypes(
        include=["int16", "int32", "int64", "float32"]
    ).columns
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    return X, y
