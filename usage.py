import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from frame.frame_selector import FRAMESelector
import numpy as np

# Load dataset (Use California Housing instead of Boston)
data = fetch_california_housing()
X, y = data.data, data.target

X = pd.DataFrame(X, columns=data.feature_names)

# Initialize FRAMESelector
frame = FRAMESelector(top_k=8, num_features=5)

# Apply feature selection
X_selected = frame.fit_transform(X, y)

# Print the selected features
print("Selected features:", frame.selected_features_)
