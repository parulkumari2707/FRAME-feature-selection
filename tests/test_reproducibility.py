"""Tier 0.4: random_state makes FRAME selection reproducible."""

import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from frame.frame_selector import FRAMESelector


@pytest.fixture
def clf_data():
    X, y = make_classification(
        n_samples=200,
        n_features=12,
        n_informative=6,
        n_redundant=2,
        random_state=0,
    )
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    return X, pd.Series(y)


@pytest.fixture
def reg_data():
    X, y = make_regression(n_samples=200, n_features=12, n_informative=6, noise=0.1, random_state=0)
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    return X, pd.Series(y)


def test_random_state_is_a_constructor_param():
    selector = FRAMESelector(random_state=42)
    assert selector.get_params()["random_state"] == 42


def test_default_classifier_selection_is_reproducible(clf_data):
    X, y = clf_data
    s1 = FRAMESelector(num_features=3, top_k=6, random_state=42).fit(X, y)
    s2 = FRAMESelector(num_features=3, top_k=6, random_state=42).fit(X, y)
    assert s1.selected_features_ == s2.selected_features_


def test_default_regressor_selection_is_reproducible(reg_data):
    X, y = reg_data
    s1 = FRAMESelector(num_features=3, top_k=6, random_state=7).fit(X, y)
    s2 = FRAMESelector(num_features=3, top_k=6, random_state=7).fit(X, y)
    assert s1.selected_features_ == s2.selected_features_
