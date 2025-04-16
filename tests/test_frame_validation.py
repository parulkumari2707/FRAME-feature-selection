import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from frame.frame_selector import FRAMESelector

@pytest.fixture
def valid_data():
    # Generates valid data with 50 samples and 10 features
    X = pd.DataFrame(np.random.randn(50, 10), columns=[f'feature_{i}' for i in range(10)])
    y = pd.Series(np.random.randn(50))  # Ensure y is a pandas Series
    return X, y

def test_fit_y_dimension(valid_data):
    X, y = valid_data
    selector = FRAMESelector(model=GradientBoostingRegressor(), num_features=5, top_k=8)
    
    # Test with 2D y (should raise an error)
    y_2d = np.random.randn(50, 2)  # Create a truly 2D ndarray
    with pytest.raises((ValueError, TypeError)):
        selector.fit(X, y_2d)
    
    # Test with 1D pandas Series (valid case)
    selector.fit(X, y)  # This should work without error

def test_non_dataframe_inputs(valid_data):
    X, y = valid_data
    selector = FRAMESelector(model=GradientBoostingRegressor(), num_features=5, top_k=8)
    
    # Test numpy array inputs (should work by converting to DataFrame/Series)
    X_np = X.values
    y_np = y.values
    selector.fit(X_np, y_np)
    
    # Test invalid input type
    with pytest.raises(TypeError):
        selector.fit("not a dataframe", y)
    
    # Test dimension mismatch
    with pytest.raises(ValueError):
        selector.fit(X, y[:30])  # Mismatched dimensions

def test_transform_validation(valid_data):
    X, y = valid_data
    selector = FRAMESelector(model=GradientBoostingRegressor(), num_features=5, top_k=8)
    
    # First fit the selector
    selector.fit(X, y)
    
    # Test transform with numpy array
    X_np = X.values
    transformed_np = selector.transform(X_np)
    assert isinstance(transformed_np, pd.DataFrame)
    assert transformed_np.shape[1] == 5
    
    # Test transform with invalid input
    with pytest.raises(TypeError):
        selector.transform("not a dataframe")
    
    # Test transform without fitting first
    new_selector = FRAMESelector()
    with pytest.raises(RuntimeError):
        new_selector.transform(X)

def test_selected_features_after_fit(valid_data):
    X, y = valid_data
    selector = FRAMESelector(model=GradientBoostingRegressor(), num_features=5, top_k=8)
    
    # Check that selected_features_ doesn't exist before fit
    assert not hasattr(selector, "selected_features_") or not selector.selected_features_
    
    # After fitting, selected_features_ should exist and have the correct length
    selector.fit(X, y)
    assert hasattr(selector, "selected_features_")
    assert len(selector.selected_features_) == 5
    assert all(feature in X.columns for feature in selector.selected_features_)

def test_fit_transform(valid_data):
    X, y = valid_data
    selector = FRAMESelector(model=GradientBoostingRegressor(), num_features=5, top_k=8)
    
    # Test fit_transform
    result = selector.fit_transform(X, y)
    
    # Verify it's a DataFrame with the right dimensions
    assert isinstance(result, pd.DataFrame)
    assert result.shape[1] == 5
    assert result.shape[0] == X.shape[0]
    
    # Verify it's the same as doing fit and transform separately
    selector2 = FRAMESelector(model=GradientBoostingRegressor(), num_features=5, top_k=8)
    selector2.fit(X, y)
    result2 = selector2.transform(X)
    pd.testing.assert_frame_equal(result, result2)

def test_parameter_validation(valid_data):
    X, y = valid_data
    
    # Test top_k > X.shape[1]
    with pytest.raises(ValueError):
        selector = FRAMESelector(num_features=5, top_k=15)  # top_k > 10 features
        selector.fit(X, y)
    
    # Test num_features > top_k
    with pytest.raises(ValueError):
        selector = FRAMESelector(num_features=9, top_k=8)
        selector.fit(X, y)
    
    # Test valid parameters work
    selector = FRAMESelector(num_features=5, top_k=8)
    selector.fit(X, y)  # Should not raise an error