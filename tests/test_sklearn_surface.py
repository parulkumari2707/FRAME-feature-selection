"""Tier 0.3: FRAMESelector exposes the standard scikit-learn selector surface."""

import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from frame.frame_selector import FRAMESelector


@pytest.fixture
def clf_data():
    X, y = make_classification(
        n_samples=200, n_features=12, n_informative=6, n_redundant=2, random_state=0
    )
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    return X, pd.Series(y)


def test_records_input_metadata(clf_data):
    X, y = clf_data
    selector = FRAMESelector(num_features=3, top_k=6, random_state=1).fit(X, y)
    assert selector.n_features_in_ == 12
    assert list(selector.feature_names_in_) == list(X.columns)


def test_get_support_boolean_mask(clf_data):
    X, y = clf_data
    selector = FRAMESelector(num_features=3, top_k=6, random_state=1).fit(X, y)
    mask = selector.get_support()
    assert mask.dtype == bool
    assert mask.shape == (12,)
    assert mask.sum() == 3
    # Selected columns via the mask match selected_features_.
    assert list(X.columns[mask]) == selector.selected_features_


def test_get_support_indices(clf_data):
    X, y = clf_data
    selector = FRAMESelector(num_features=3, top_k=6, random_state=1).fit(X, y)
    idx = selector.get_support(indices=True)
    assert list(X.columns[idx]) == selector.selected_features_


def test_get_feature_names_out(clf_data):
    X, y = clf_data
    selector = FRAMESelector(num_features=3, top_k=6, random_state=1).fit(X, y)
    names = selector.get_feature_names_out()
    assert list(names) == selector.selected_features_


def test_works_inside_pipeline(clf_data):
    X, y = clf_data
    pipe = Pipeline(
        [
            ("select", FRAMESelector(num_features=3, top_k=6, random_state=1)),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert preds.shape[0] == X.shape[0]
