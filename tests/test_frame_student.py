"""Student-performance (regression) smoke test.

Runs only when the (gitignored) dataset CSV is available locally; skips cleanly
otherwise so the suite stays green on a fresh clone. See CLAUDE.md.
"""

import os

import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from frame.frame_selector import FRAMESelector

DATA_PATH = "data/student_data_student_performance.csv"
pytestmark = pytest.mark.skipif(
    not os.path.exists(DATA_PATH),
    reason=f"dataset '{DATA_PATH}' not committed; see CLAUDE.md",
)


def test_frame_on_student():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["G3"], errors="ignore")
    y = df["G3"]

    X = X.fillna(X.mean(numeric_only=True))
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = X[col].fillna(X[col].mode()[0])
    X = pd.get_dummies(X, drop_first=True)

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    selector = FRAMESelector(model=LinearRegression(), num_features=5, random_state=42)
    X_selected = selector.fit_transform(X_train, y_train)

    assert len(selector.selected_features_) == 5
    assert X_selected.shape[1] == 5
