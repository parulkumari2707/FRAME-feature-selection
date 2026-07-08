"""Cardiovascular (classification) smoke test.

Runs only when the (gitignored) dataset CSV is available locally; skips cleanly
otherwise so the suite stays green on a fresh clone. See CLAUDE.md.
"""

import os
from typing import Tuple

import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from frame.frame_selector import FRAMESelector

DATA_PATH = "data/myocardial_infarction_data.csv"
pytestmark = pytest.mark.skipif(
    not os.path.exists(DATA_PATH),
    reason=f"dataset '{DATA_PATH}' not committed; see CLAUDE.md",
)


def preprocess_cardio_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=["LET_IS"], errors="ignore")
    y = df["LET_IS"]
    X = X.fillna(X.mean(numeric_only=True))
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = X[col].fillna(X[col].mode()[0])
    return X, y


def test_frame_on_cardio():
    df = pd.read_csv(DATA_PATH)
    X, y = preprocess_cardio_data(df)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBClassifier(eval_metric="logloss", random_state=42)
    selector = FRAMESelector(model=model, num_features=5, random_state=42)
    X_selected = selector.fit_transform(X_train, y_train)

    assert len(selector.selected_features_) == 5
    assert X_selected.shape[1] == 5
