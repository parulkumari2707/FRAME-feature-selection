import pytest
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.datasets import make_regression, load_iris, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from frame.frame_selector import FRAMESelector
from typing import Tuple

# Generate synthetic regression data
X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)

def test_frame_regression_initialization() -> None:
    """Test if FRAMESelector initializes correctly for regression."""
    model = XGBRegressor()
    selector = FRAMESelector(model=model, num_features=5)
    assert selector.num_features == 5
    assert isinstance(selector.model, XGBRegressor)

def test_frame_regression_fit() -> None:
    """Test if FRAMESelector selects the correct number of features for regression."""
    model = XGBRegressor()
    selector = FRAMESelector(model=model, num_features=5)
    selector.fit(X, y)
    print("Selected Features (Regression):", selector.selected_features_)
    assert len(selector.selected_features_) == 5, f"Expected 5 features, got {len(selector.selected_features_)}"

def test_frame_regression_transform() -> None:
    """Test if transform method correctly reduces feature dimensions."""
    model = XGBRegressor()
    selector = FRAMESelector(model=model, num_features=5)
    selector.fit(X, y)
    X_selected = selector.transform(X)
    print("Transformed X shape (Regression):", X_selected.shape)
    assert X_selected.shape[1] == 5, f"Expected 5 features, got {X_selected.shape[1]}"

def test_frame_regression_fit_transform() -> None:
    """Test if fit_transform method works as expected."""
    model = XGBRegressor()
    selector = FRAMESelector(model=model, num_features=5)
    X_selected = selector.fit_transform(X, y)
    print("Selected Features after fit_transform (Regression):", selector.selected_features_)
    print("Transformed X shape after fit_transform:", X_selected.shape)
    assert X_selected.shape[1] == 5, f"Expected 5 features, got {X_selected.shape[1]}"

@pytest.fixture
def classification_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load and prepare classification data (Iris dataset)."""
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

@pytest.fixture
def regression_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load and prepare regression data (Diabetes dataset)."""
    diabetes = load_diabetes()
    X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    y = pd.Series(diabetes.target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def test_frame_classification(classification_data: Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]) -> None:
    """Test FRAME feature selection for classification."""
    X_train, X_test, y_train, y_test = classification_data
    classifier_model = LogisticRegression(max_iter=200)
    frame_selector = FRAMESelector(model=classifier_model, num_features=2)

    X_train_selected = frame_selector.fit_transform(X_train, y_train)

    # Print selected features
    print("\n=== Selected Features for Classification ===")
    print(frame_selector.selected_features_)

    # Assertions
    assert len(frame_selector.selected_features_) == 2  # Ensuring two features are selected
    assert X_train_selected.shape[1] == 2  # Checking transformed shape

def test_frame_regression_extended(regression_data: Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]) -> None:
    """Test FRAME feature selection for regression."""
    X_train, X_test, y_train, y_test = regression_data
    regressor_model = XGBRegressor(n_estimators=100, max_depth=3, random_state=42)
    frame_selector = FRAMESelector(model=regressor_model, num_features=3)

    X_train_selected = frame_selector.fit_transform(X_train, y_train)

    # Print selected features
    print("\n=== Selected Features for Regression ===")
    print(frame_selector.selected_features_)

    # Assertions
    assert len(frame_selector.selected_features_) == 3  # Ensuring three features are selected
    assert X_train_selected.shape[1] == 3  # Checking transformed shape
