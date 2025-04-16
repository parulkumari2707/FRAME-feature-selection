import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from frame.frame_selector import FRAMESelector
from typing import Tuple

# Create sample dataset
X, y = make_classification(n_samples=100, n_features=20, n_informative=10, random_state=42)
X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])  # Convert to DataFrame

def test_frame_initialization() -> None:
    """Test if FRAMESelector initializes correctly."""
    model = LogisticRegression()
    selector = FRAMESelector(model=model, num_features=5)

    assert selector.model == model
    assert selector.num_features == 5

def test_frame_fit() -> None:
    """Test if FRAMESelector selects the correct number of features."""
    model = LogisticRegression()
    selector = FRAMESelector(model=model, num_features=5)
    selector.fit(X, y)

    print("\n=== Selected Features ===")
    print(selector.selected_features_)

    assert len(selector.selected_features_) == 5
    assert all(isinstance(i, str) for i in selector.selected_features_)  # Features should be column names

def test_frame_transform() -> None:
    """Test if transform method correctly reduces feature dimensions."""
    model = LogisticRegression()
    selector = FRAMESelector(model=model, num_features=5)
    selector.fit(X, y)
    X_selected = selector.transform(X)

    print("\n=== Transformed X Shape ===")
    print(X_selected.shape)

    assert X_selected.shape[1] == 5  # Ensure feature count matches selected features

def test_frame_fit_transform() -> None:
    """Test if fit_transform method works as expected."""
    model = LogisticRegression()
    selector = FRAMESelector(model=model, num_features=5)
    X_selected = selector.fit_transform(X, y)

    print("\n=== Selected Features from Fit_Transform ===")
    print(selector.selected_features_)

    assert X_selected.shape[1] == 5  # Check if transformation applied correctly

''' 
def test_selected_features():
    """Test if selected_features_ attribute contains correct feature names."""
    model = LogisticRegression()
    selector = FRAMESelector(model=model, num_features=5)
    selector.fit(X, y)

    selected_features = selector.selected_features_
    assert len(selected_features) == 5
    assert all(isinstance(i, str) for i in selected_features)  # Ensure correct feature names
'''

## Either add get_support () here in test_frame_selector.py, frame_selector.py and usage.py or 
# use test_selected_feature instead or simply remove it
# for now I have removed it
