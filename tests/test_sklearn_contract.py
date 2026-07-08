"""Tier 0.2/0.3: FRAMESelector honours the scikit-learn estimator contract."""

import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification
from frame.frame_selector import FRAMESelector


@pytest.fixture
def clf_data():
    X, y = make_classification(
        n_samples=200, n_features=12, n_informative=6, n_redundant=2, random_state=0
    )
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    return X, pd.Series(y)


def test_fit_does_not_mutate_model_param(clf_data):
    X, y = clf_data
    selector = FRAMESelector(num_features=3, top_k=6, random_state=1)
    selector.fit(X, y)
    # The constructor argument must be left untouched (sklearn contract).
    assert selector.model is None
    assert selector.get_params()["model"] is None


def test_fit_exposes_fitted_estimator(clf_data):
    X, y = clf_data
    selector = FRAMESelector(num_features=3, top_k=6, random_state=1).fit(X, y)
    # The actually-used estimator is exposed as a fitted attribute.
    assert hasattr(selector, "model_")
    assert selector.model_ is not None


def test_clone_produces_unfitted_equivalent(clf_data):
    X, y = clf_data
    selector = FRAMESelector(num_features=3, top_k=6, random_state=1).fit(X, y)
    fresh = clone(selector)
    assert fresh.get_params() == selector.get_params()
    assert not hasattr(fresh, "selected_features_")
    # A clone must be independently fittable.
    fresh.fit(X, y)
    assert fresh.selected_features_ == selector.selected_features_
