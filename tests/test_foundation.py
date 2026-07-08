"""Tier 1 Foundation slice: auto num_features, scores, get_feature_scores, n_jobs, verbose."""

import numpy as np
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


def test_ranking_is_permutation_with_selected_first(clf_data):
    X, y = clf_data
    s = FRAMESelector(num_features=4, top_k=8, random_state=0).fit(X, y)
    assert s.ranking_.shape == (X.shape[1],)
    assert sorted(s.ranking_.tolist()) == list(range(1, X.shape[1] + 1))
    selected_idx = [X.columns.get_loc(f) for f in s.selected_features_]
    assert sorted(s.ranking_[selected_idx].tolist()) == [1, 2, 3, 4]


def test_scores_aligned_and_finite(clf_data):
    X, y = clf_data
    s = FRAMESelector(num_features=4, top_k=8, random_state=0).fit(X, y)
    assert s.scores_.shape == (X.shape[1],)
    assert np.all(np.isfinite(s.scores_))


def test_informative_features_rank_better_than_noise():
    # 4 informative, 8 pure-noise features; informative are columns 0..3.
    X, y = make_classification(
        n_samples=300,
        n_features=12,
        n_informative=4,
        n_redundant=0,
        n_repeated=0,
        shuffle=False,
        random_state=0,
    )
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    y = pd.Series(y)
    s = FRAMESelector(num_features=4, top_k=8, random_state=0).fit(X, y)
    informative_ranks = s.ranking_[:4]
    noise_ranks = s.ranking_[4:]
    assert informative_ranks.max() < noise_ranks.min()


def test_get_feature_scores_sorted_series(clf_data):
    X, y = clf_data
    s = FRAMESelector(num_features=4, top_k=8, random_state=0).fit(X, y)
    fs = s.get_feature_scores()
    assert isinstance(fs, pd.Series)
    assert list(fs.index) == list(fs.sort_values(ascending=False).index)
    assert set(fs.index) == set(X.columns)
    assert len(fs) == X.shape[1]


def test_get_feature_scores_before_fit_raises():
    with pytest.raises(RuntimeError):
        FRAMESelector().get_feature_scores()
