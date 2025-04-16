import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from frame.frame_selector import FRAMESelector

def load_parkinsons_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load and preprocess the Parkinson's dataset.

    Returns:
        Tuple of features (X) and target (y) as pandas DataFrame and Series.
    """
    parkinsons_df = pd.read_csv("data/pd_speech_features_parkinsons.csv", header=1)
    print(parkinsons_df.head(5))
    print(parkinsons_df.columns)  # Debugging step to check column names

    X = parkinsons_df.drop(columns=["class", "Unnamed: 0", "Unnamed: 1"], errors='ignore')
    y = parkinsons_df["class"]
    return X, y

def split_data(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the data into training and testing sets.

    Args:
        X: Feature dataframe.
        y: Target series.

    Returns:
        Tuple of X_train, X_test, y_train, y_test.
    """
    return train_test_split(X, y, test_size=0.2, random_state=42)

def run_frame_selector(X_train: pd.DataFrame, y_train: pd.Series, num_features: int = 5) -> tuple[FRAMESelector, pd.DataFrame]:
    """Apply FRAME feature selection on training data.

    Args:
        X_train: Training features.
        y_train: Training target.
        num_features: Number of top features to select.

    Returns:
        Tuple of fitted FRAMESelector instance and transformed X_train.
    """
    if num_features > X_train.shape[1]:
        raise ValueError(f"Number of features to select ({num_features}) cannot exceed total features ({X_train.shape[1]}).")

    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    frame_selector = FRAMESelector(model=model, num_features=num_features)
    X_train_selected = frame_selector.fit_transform(X_train, y_train)
    return frame_selector, X_train_selected

# Load and preprocess Parkinson's dataset
X_parkinsons, y_parkinsons = load_parkinsons_data()

# Split the dataset
X_train_parkinsons, X_test_parkinsons, y_train_parkinsons, y_test_parkinsons = split_data(X_parkinsons, y_parkinsons)

# Apply FRAME Selector for Parkinson's classification
print("\n=== Running FRAME Feature Selection for Parkinson's Classification ===")
frame_selector_parkinsons, X_train_selected_parkinsons = run_frame_selector(X_train_parkinsons, y_train_parkinsons, num_features=5)

print("Selected Features (Parkinson's Classification):", frame_selector_parkinsons.selected_features_)
print("Transformed X_train shape:", X_train_selected_parkinsons.shape)
