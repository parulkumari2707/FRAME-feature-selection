"""Tier 1 Foundation slice: auto num_features, scores, get_feature_scores, n_jobs, verbose."""

import pandas as pd
import pytest
from sklearn.datasets import make_classification
from frame.frame_selector import FRAMESelector


@pytest.fixture
def clf_data():
    X, y = make_classification(
        n_samples=200, n_features=12, n_informative=6, n_redundant=2, random_state=0
    )
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    return X, pd.Series(y)


def test_new_params_are_stored():
    s = FRAMESelector(n_jobs=-1, verbose=2, tol=0.01)
    params = s.get_params()
    assert params["n_jobs"] == -1
    assert params["verbose"] == 2
    assert params["tol"] == 0.01


def test_num_features_auto_selects_within_top_k(clf_data):
    X, y = clf_data
    s = FRAMESelector(num_features="auto", top_k=8, random_state=0).fit(X, y)
    k = len(s.selected_features_)
    assert 1 <= k <= 8


def test_higher_tol_selects_no_more_than_lower_tol(clf_data):
    X, y = clf_data
    loose = FRAMESelector(num_features="auto", top_k=8, tol=1e-6, random_state=0).fit(X, y)
    strict = FRAMESelector(num_features="auto", top_k=8, tol=0.05, random_state=0).fit(X, y)
    assert len(strict.selected_features_) <= len(loose.selected_features_)


def test_invalid_num_features_raises(clf_data):
    X, y = clf_data
    with pytest.raises(ValueError):
        FRAMESelector(num_features="half", top_k=8).fit(X, y)
    with pytest.raises(ValueError):
        FRAMESelector(num_features=-3, top_k=8).fit(X, y)


def test_default_call_unchanged_and_njobs_invariant(clf_data):
    X, y = clf_data
    base = FRAMESelector(top_k=8, random_state=0).fit(X, y)
    # default num_features=None -> n_features // 2
    assert len(base.selected_features_) == X.shape[1] // 2
    # n_jobs must not change the selection
    parallel = FRAMESelector(top_k=8, random_state=0, n_jobs=-1).fit(X, y)
    assert parallel.selected_features_ == base.selected_features_
