# Load Cardiovascular dataset
# synthetic_noise = pd.read_csv("data\synthetic_data\synthetic_data_with_noise.csv")
# synthetic_baseline = pd.read_csv("data\synthetic_data\synthetic_data_baseline.csv")
# synthetic_high_spars_high_redund = pd.read_csv("data\synthetic_data\synthetic_dataset_High_sparsity,_high_redundancy.csv")
# synthetic_high_spars_low_redund = pd.read_csv("data\synthetic_data\synthetic_dataset_High_sparsity,_low_redundancy.csv")
# synthetic_low_spars_high_redund = pd.read_csv("data\synthetic_data\synthetic_dataset_Low_sparsity,_high_redundancy.csv")
# synthetic_low_spars_low_redund = pd.read_csv("data\synthetic_data\synthetic_dataset_Low_sparsity,_low_redundancy.csv")


import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from frame.frame_selector import FRAMESelector
from typing import Dict

# Load synthetic datasets
def load_synthetic_datasets() -> Dict[str, pd.DataFrame]:
    """
    Loads all synthetic datasets from the predefined file paths.

    Returns:
        A dictionary mapping dataset names to pandas DataFrames.
    """
    return {
        "Synthetic_Noise": pd.read_csv("data\synthetic_data\synthetic_data_with_noise.csv"),
        "Synthetic_Baseline": pd.read_csv("data\synthetic_data\synthetic_dataset_Baseline_dataset.csv"),
        "High_Sparsity_High_Redundancy": pd.read_csv("data\synthetic_data\synthetic_dataset_High_sparsity,_high_redundancy.csv"),
        "High_Sparsity_Low_Redundancy": pd.read_csv("data\synthetic_data\synthetic_dataset_High_sparsity,_low_redundancy.csv"),
        "Low_Sparsity_High_Redundancy": pd.read_csv("data\synthetic_data\synthetic_dataset_Low_sparsity,_high_redundancy.csv"),
        "Low_Sparsity_Low_Redundancy": pd.read_csv("data\synthetic_data\synthetic_dataset_Low_sparsity,_low_redundancy.csv"),
    }

synthetic_datasets = load_synthetic_datasets()

for dataset_name, df in synthetic_datasets.items():
    print(f"\n=== Processing {dataset_name} ===")

    # Assume last column is the target
    X: pd.DataFrame = df.iloc[:, :-1]
    y: pd.Series = df.iloc[:, -1]

    # Handle missing values
    X.fillna(X.mean(), inplace=True)
    for col in X.select_dtypes(include=["object"]).columns:
        X[col].fillna(X[col].mode()[0], inplace=True)

    # Convert categorical variables if needed
    X = pd.get_dummies(X, drop_first=True)

    # Determine task type (classification or regression)
    if y.nunique() > 10:  # Assuming regression if more than 10 unique values
        model = LinearRegression()
        task_type = "Regression"
        num_features = 5
    else:
        model = LogisticRegression(max_iter=200)
        task_type = "Classification"
        num_features = 5

    # Ensure num_features is valid
    if num_features > X.shape[1]:
        raise ValueError(f"num_features ({num_features}) cannot be greater than number of features ({X.shape[1]}) in {dataset_name}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply FRAME Selector
    print(f"Running FRAME Feature Selection for {dataset_name} ({task_type})")
    frame_selector = FRAMESelector(model=model, num_features=num_features)
    X_train_selected: pd.DataFrame = frame_selector.fit_transform(X_train, y_train)

    # Print selected features
    print(f"Selected Features ({dataset_name} - {task_type}):", frame_selector.selected_features_)
    print(f"Transformed X_train shape: {X_train_selected.shape}")
