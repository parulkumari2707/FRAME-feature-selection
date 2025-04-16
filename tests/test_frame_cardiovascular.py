import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from frame.frame_selector import FRAMESelector
from typing import Tuple

# Load Cardiovascular dataset
cardio_df: pd.DataFrame = pd.read_csv("data/myocardial_infarction_data.csv")

def preprocess_cardio_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Preprocesses the cardiovascular dataset by handling missing values and separating features and target.

    Args:
        df (pd.DataFrame): Input dataframe containing features and target.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Processed features and target.
    """
    X: pd.DataFrame = df.drop(columns=["LET_IS"], errors='ignore')
    y: pd.Series = df["LET_IS"]

    # Handle missing values
    X.fillna(X.mean(), inplace=True)
    for col in X.select_dtypes(include=["object"]).columns:
        X[col].fillna(X[col].mode()[0], inplace=True)

    return X, y

def run_frame_on_cardio(X: pd.DataFrame, y: pd.Series, num_features: int = 5) -> None:
    """
    Applies the FRAME feature selector to the cardiovascular dataset and prints the selected features.

    Args:
        X (pd.DataFrame): Feature dataframe.
        y (pd.Series): Target series.
        num_features (int): Number of top features to select.

    Raises:
        ValueError: If num_features > number of features in X.
    """
    if num_features > X.shape[1]:
        raise ValueError(f"num_features ({num_features}) cannot be greater than number of features ({X.shape[1]})")

    # Split data
    X_train_cardio, X_test_cardio, y_train_cardio, y_test_cardio = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Apply FRAME Selector
    print("\n=== Running FRAME Feature Selection for Cardiovascular Classification ===")
    cardio_classifier_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    frame_selector_cardio = FRAMESelector(model=cardio_classifier_model, num_features=num_features)
    X_train_selected_cardio: pd.DataFrame = frame_selector_cardio.fit_transform(X_train_cardio, y_train_cardio)

    print("Selected Features (Cardiovascular Classification):", frame_selector_cardio.selected_features_)
    print("Transformed X_train shape:", X_train_selected_cardio.shape)

# Run processing and FRAME
X_cardio, y_cardio = preprocess_cardio_data(cardio_df)
run_frame_on_cardio(X_cardio, y_cardio)