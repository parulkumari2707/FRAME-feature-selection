import pytest
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.datasets import make_classification
from frame.frame_selector import FRAMESelector

# Generate synthetic classification data
X, y = make_classification(n_samples=100, n_features=10, random_state=42)
X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
y = pd.Series(y)

def test_frame_classification_initialization() -> None:
    """Test if FRAMESelector initializes correctly for classification.
    Ensures that the number of features and model type are correctly assigned during initialization.
    """
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    selector = FRAMESelector(model=model, num_features=5, top_k=8)
    assert selector.num_features == 5
    assert selector.top_k == 8
    assert isinstance(selector.model, XGBClassifier)

def test_frame_classification_fit() -> None:
    """Test if FRAMESelector selects the correct number of features for classification.
    Ensures the fit method identifies exactly the number of features specified.
    """
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    selector = FRAMESelector(model=model, num_features=5, top_k=8)
    selector.fit(X, y)
    print("Selected Features:", selector.selected_features_)
    assert len(selector.selected_features_) == 5, f"Expected 5 features, got {len(selector.selected_features_)}"

def test_frame_classification_transform() -> None:
    """Test if transform method correctly reduces feature dimensions.
    Checks that the transform method returns a DataFrame with the expected number of selected features.
    """
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    selector = FRAMESelector(model=model, num_features=5, top_k=8)
    selector.fit(X, y)
    X_selected = selector.transform(X)
    print("Selected Features:", selector.selected_features_)
    assert X_selected.shape[1] == 5, f"Expected 5 features, got {X_selected.shape[1]}"

def test_frame_classification_fit_transform() -> None:
    """Test if fit_transform method works as expected.
    Combines fit and transform, verifying it outputs data with the correct number of features.
    """
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    selector = FRAMESelector(model=model, num_features=5, top_k=8)
    X_selected = selector.fit_transform(X, y)
    print("Selected Features:", selector.selected_features_)
    assert X_selected.shape[1] == 5, f"Expected 5 features, got {X_selected.shape[1]}"

def test_classification_feature_names() -> None:
    """Test that selected features preserve column names from original DataFrame."""
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    selector = FRAMESelector(model=model, num_features=5, top_k=8)
    selector.fit(X, y)
    
    # Verify selected features are a subset of original column names
    assert all(feature in X.columns for feature in selector.selected_features_)
    
    # Check transformed data has the correct column names
    X_selected = selector.transform(X)
    assert list(X_selected.columns) == selector.selected_features_