import pytest
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.datasets import make_regression
from frame.frame_selector import FRAMESelector

# Generate synthetic regression data
X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)
# Convert numpy arrays to pandas DataFrame and Series for better compatibility
X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
y = pd.Series(y)

def test_frame_regression_initialization() -> None:
    """Test if FRAMESelector initializes correctly for regression."""
    model = XGBRegressor()
    # Set top_k to be less than the number of features (10)
    selector = FRAMESelector(model=model, num_features=5, top_k=8)
    assert selector.num_features == 5
    assert selector.top_k == 8
    assert isinstance(selector.model, XGBRegressor)

def test_frame_regression_fit() -> None:
    """Test if FRAMESelector selects the correct number of features for regression."""
    model = XGBRegressor()
    # Set top_k to be less than the number of features (10)
    selector = FRAMESelector(model=model, num_features=5, top_k=8)
    selector.fit(X, y)
    print("Selected Features (Regression):", selector.selected_features_)
    assert len(selector.selected_features_) == 5, f"Expected 5 features, got {len(selector.selected_features_)}"

def test_frame_regression_transform() -> None:
    """Test if transform method correctly reduces feature dimensions."""
    model = XGBRegressor()
    # Set top_k to be less than the number of features (10)
    selector = FRAMESelector(model=model, num_features=5, top_k=8)
    selector.fit(X, y)
    X_selected = selector.transform(X)
    print("Transformed X shape (Regression):", X_selected.shape)
    assert X_selected.shape[1] == 5, f"Expected 5 features, got {X_selected.shape[1]}"

def test_frame_regression_fit_transform() -> None:
    """Test if fit_transform method works as expected."""
    model = XGBRegressor()
    # Set top_k to be less than the number of features (10)
    selector = FRAMESelector(model=model, num_features=5, top_k=8)
    X_selected = selector.fit_transform(X, y)
    print("Selected Features after fit_transform (Regression):", selector.selected_features_)
    print("Transformed X shape after fit_transform:", X_selected.shape)
    assert X_selected.shape[1] == 5, f"Expected 5 features, got {X_selected.shape[1]}"

def test_frame_regression_with_dataframe() -> None:
    """Test if FRAMESelector works with pandas DataFrame as input."""
    model = XGBRegressor()
    selector = FRAMESelector(model=model, num_features=5, top_k=8)
    # X and y are already converted to DataFrame and Series at the top
    selector.fit(X, y)
    X_selected = selector.transform(X)
    assert isinstance(X_selected, pd.DataFrame)
    assert X_selected.shape[1] == 5