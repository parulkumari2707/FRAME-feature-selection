"""Synthetic-dataset smoke tests.

These exercise FRAME over the (gitignored) generated CSVs under
``data/synthetic_data/``. They skip cleanly when those files are absent so the
suite stays green on a fresh clone. See CLAUDE.md.
"""

import os

import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split

from frame.frame_selector import FRAMESelector

DATA_DIR = os.path.join("data", "synthetic_data")

DATASETS = {
    "Synthetic_Noise": "synthetic_data_with_noise.csv",
    "Synthetic_Baseline": "synthetic_dataset_Baseline_dataset.csv",
    "High_Sparsity_High_Redundancy": "synthetic_dataset_High_sparsity,_high_redundancy.csv",
    "High_Sparsity_Low_Redundancy": "synthetic_dataset_High_sparsity,_low_redundancy.csv",
    "Low_Sparsity_High_Redundancy": "synthetic_dataset_Low_sparsity,_high_redundancy.csv",
    "Low_Sparsity_Low_Redundancy": "synthetic_dataset_Low_sparsity,_low_redundancy.csv",
}


@pytest.mark.parametrize("name,filename", list(DATASETS.items()))
def test_frame_on_synthetic(name, filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        pytest.skip(f"dataset '{path}' not committed; see CLAUDE.md")

    df = pd.read_csv(path)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X = X.fillna(X.mean(numeric_only=True))
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = X[col].fillna(X[col].mode()[0])
    X = pd.get_dummies(X, drop_first=True)

    is_regression = y.nunique() > 10
    model = LinearRegression() if is_regression else LogisticRegression(max_iter=200)
    num_features = 5

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    selector = FRAMESelector(model=model, num_features=num_features, random_state=42)
    X_selected = selector.fit_transform(X_train, y_train)

    assert X_selected.shape[1] == num_features
