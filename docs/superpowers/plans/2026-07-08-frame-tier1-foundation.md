# FRAME Tier 1 Foundation Slice — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the "Foundation + cheap wins" capabilities to `FRAMESelector` — `num_features='auto'`, `ranking_`/`scores_` fitted attributes, `get_feature_scores()`, `n_jobs`, and `verbose` — without changing default-call results.

**Architecture:** Extend the single param-driven `FRAMESelector` (strategy `rfe_forward` only). Forward-selection auto mode delegates to scikit-learn's native `SequentialFeatureSelector(n_features_to_select='auto', tol=...)`. Scores derive from an always-available RFE baseline, refined by the fitted model's importances for selected features. All additions are backward-compatible.

**Tech Stack:** Python, numpy, pandas, scikit-learn (≥1.1 for SFS auto mode), xgboost. Tests via pytest. Lint: flake8 (max-line-length=100, extend-ignore=E203,W503) + black (line-length 100).

## Global Constraints

- Run everything with the `autods` interpreter: `/c/Users/chinmay/anaconda3/envs/autods/python` (system `python` 3.14 has no scientific deps).
- Backward compatibility is mandatory: a call using only `model`/`num_features`/`top_k`/`random_state` must produce the same `selected_features_` as before this slice.
- Scores are exposed as fitted attributes (`ranking_`, `scores_`) — never returned from `transform` (which must keep returning a `pd.DataFrame` of `X`).
- `strategy` and `stability` are NOT added in this slice (YAGNI — added when implemented). Only `n_jobs`, `verbose`, `tol` are new constructor params.
- Keep code black-clean and flake8-clean (line length ≤ 100).
- Every commit message ends with a trailing:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`

---

## File Structure

- **Modify:** `frame/frame_selector.py` — add constructor params, `num_features='auto'` resolution, `n_jobs`/`verbose` threading, `ranking_`/`scores_` computation, `_get_importances` module helper, and `get_feature_scores()`.
- **Create:** `tests/test_foundation.py` — all tests for this slice.

No other files change. `time` is imported at the top of `frame/frame_selector.py` for verbose timing (Task 4).

---

## Task 1: Constructor params + `num_features='auto'` + `n_jobs`

**Files:**
- Modify: `frame/frame_selector.py` (`__init__`, `fit`)
- Test: `tests/test_foundation.py`

**Interfaces:**
- Consumes: existing `FRAMESelector(model, num_features, top_k, random_state)`, `self.model_`, `self.n_features_in_`, `self.feature_names_in_`.
- Produces: `FRAMESelector(..., n_jobs=None, verbose=0, tol=None)`; `num_features` now accepts `None | positive int | 'auto'`; `n_jobs` threaded into the default estimator and `SequentialFeatureSelector`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_foundation.py`:

```python
"""Tier 1 Foundation slice: auto num_features, scores, get_feature_scores, n_jobs, verbose."""
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `"/c/Users/chinmay/anaconda3/envs/autods/python" -m pytest tests/test_foundation.py -q -p no:warnings`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'n_jobs'`.

- [ ] **Step 3: Add the new constructor params**

In `frame/frame_selector.py`, replace the `__init__` with:

```python
    def __init__(
        self,
        model: Optional[Union[XGBClassifier, XGBRegressor]] = None,
        num_features: Optional[Union[int, str]] = None,
        top_k: int = 20,
        random_state: Optional[int] = None,
        n_jobs: Optional[int] = None,
        verbose: int = 0,
        tol: Optional[float] = None,
    ):
        self.model = model
        self.num_features = num_features
        self.top_k = top_k
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.tol = tol
```

- [ ] **Step 4: Replace the `num_features` resolution block in `fit`**

Find this block in `fit`:

```python
        num_features = self.num_features if self.num_features else max(1, X.shape[1] // 2)
        if num_features > self.top_k:
            raise ValueError(
                f"num_features={num_features} cannot be greater than top_k={self.top_k}."
            )
```

Replace it with:

```python
        nf = self.num_features
        if nf is None:
            sfs_n_features = max(1, X.shape[1] // 2)
        elif isinstance(nf, str):
            if nf != "auto":
                raise ValueError(
                    f"num_features must be None, a positive int, or 'auto'; got {nf!r}."
                )
            sfs_n_features = "auto"
        elif isinstance(nf, (int, np.integer)) and not isinstance(nf, bool) and nf > 0:
            resolved_n = int(nf)
            if resolved_n > self.top_k:
                raise ValueError(
                    f"num_features={resolved_n} cannot be greater than top_k={self.top_k}."
                )
            sfs_n_features = resolved_n
        else:
            raise ValueError(
                f"num_features must be None, a positive int, or 'auto'; got {nf!r}."
            )
```

- [ ] **Step 5: Thread `n_jobs` into the default estimator**

Find the default-model block:

```python
        if self.model is None:
            self.model_ = (
                XGBClassifier(eval_metric="logloss", random_state=self.random_state)
                if is_classification
                else XGBRegressor(random_state=self.random_state)
            )
        else:
            self.model_ = clone(self.model)
```

Replace it with:

```python
        if self.model is None:
            self.model_ = (
                XGBClassifier(
                    eval_metric="logloss",
                    random_state=self.random_state,
                    n_jobs=self.n_jobs,
                )
                if is_classification
                else XGBRegressor(random_state=self.random_state, n_jobs=self.n_jobs)
            )
        else:
            self.model_ = clone(self.model)
```

- [ ] **Step 6: Update the SequentialFeatureSelector call**

Find:

```python
        # Step 2: Forward Selection
        forward_selector = SequentialFeatureSelector(
            self.model_, n_features_to_select=num_features, direction="forward"
        )
        forward_selector.fit(X[rfe_selected_features], y)
        final_selected_features = rfe_selected_features[forward_selector.get_support()]
```

Replace it with:

```python
        # Step 2: Forward Selection
        sfs_kwargs = dict(
            n_features_to_select=sfs_n_features,
            direction="forward",
            n_jobs=self.n_jobs,
        )
        if sfs_n_features == "auto":
            sfs_kwargs["tol"] = self.tol
        forward_selector = SequentialFeatureSelector(self.model_, **sfs_kwargs)
        forward_selector.fit(X[rfe_selected_features], y)
        final_selected_features = rfe_selected_features[forward_selector.get_support()]
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `"/c/Users/chinmay/anaconda3/envs/autods/python" -m pytest tests/test_foundation.py -q -p no:warnings`
Expected: PASS (5 tests).

- [ ] **Step 8: Lint + commit**

Run:
```bash
"/c/Users/chinmay/anaconda3/envs/autods/python" -m black frame tests
"/c/Users/chinmay/anaconda3/envs/autods/python" -m flake8 frame tests --max-line-length=100 --extend-ignore=E203,W503
```
Then:
```bash
git add frame/frame_selector.py tests/test_foundation.py
git commit -m "feat(frame): support num_features='auto', n_jobs, and new params

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: `ranking_` and `scores_` fitted attributes

**Files:**
- Modify: `frame/frame_selector.py` (module-level helper `_get_importances`, end of `fit`)
- Test: `tests/test_foundation.py`

**Interfaces:**
- Consumes: `self.model_`, `self.selected_features_`, `self.n_features_in_`, `self.feature_names_in_`, and the local `rfe` object (its `ranking_`) from `fit`.
- Produces: `self.ranking_` (np.ndarray[int], shape `(n_features_in_,)`, permutation of `1..n`, selected occupy `1..k`); `self.scores_` (np.ndarray[float], shape `(n_features_in_,)`, finite, higher = better); module helper `_get_importances(estimator) -> np.ndarray | None`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_foundation.py`:

```python
import numpy as np


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `"/c/Users/chinmay/anaconda3/envs/autods/python" -m pytest tests/test_foundation.py -q -p no:warnings -k "ranking or scores or informative"`
Expected: FAIL — `AttributeError: 'FRAMESelector' object has no attribute 'ranking_'`.

- [ ] **Step 3: Add the `_get_importances` module helper**

At module level in `frame/frame_selector.py` (after the imports, before the class):

```python
def _get_importances(estimator):
    """Return per-feature importances from a fitted estimator, or None.

    Uses ``feature_importances_`` when present, else absolute ``coef_`` (summed
    across outputs for multiclass). Returns None when the estimator exposes
    neither.
    """
    if hasattr(estimator, "feature_importances_"):
        return np.asarray(estimator.feature_importances_, dtype=float)
    if hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_, dtype=float)
        if coef.ndim > 1:
            return np.abs(coef).sum(axis=0)
        return np.abs(coef)
    return None
```

- [ ] **Step 4: Compute `scores_` and `ranking_` at the end of `fit`**

Find, near the end of `fit`:

```python
        self.selected_features_ = final_selected_features.tolist()
        return self
```

Replace with:

```python
        self.selected_features_ = final_selected_features.tolist()

        # Fit the resolved estimator on the selected features for scoring.
        self.model_.fit(X[self.selected_features_], y)

        # scores_: RFE-ranking baseline (higher = better), refined by model
        # importances for selected features. Aligned to feature_names_in_.
        scores = 1.0 / rfe.ranking_.astype(float)
        importances = _get_importances(self.model_)
        if importances is not None and len(importances) == len(self.selected_features_):
            for i, feat in enumerate(self.selected_features_):
                scores[X.columns.get_loc(feat)] = float(importances[i])
        self.scores_ = scores

        # ranking_: selected features occupy ranks 1..k (by score desc); the rest
        # follow (by RFE elimination order, better RFE rank first).
        n = self.n_features_in_
        ranking = np.empty(n, dtype=int)
        selected_idx = [X.columns.get_loc(f) for f in self.selected_features_]
        selected_set = set(selected_idx)
        for rank, i in enumerate(
            sorted(selected_idx, key=lambda j: scores[j], reverse=True), start=1
        ):
            ranking[i] = rank
        rest = [i for i in range(n) if i not in selected_set]
        for rank, i in enumerate(
            sorted(rest, key=lambda j: rfe.ranking_[j]), start=len(selected_idx) + 1
        ):
            ranking[i] = rank
        self.ranking_ = ranking

        return self
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `"/c/Users/chinmay/anaconda3/envs/autods/python" -m pytest tests/test_foundation.py -q -p no:warnings`
Expected: PASS (all tests so far).

- [ ] **Step 6: Lint + commit**

Run:
```bash
"/c/Users/chinmay/anaconda3/envs/autods/python" -m black frame tests
"/c/Users/chinmay/anaconda3/envs/autods/python" -m flake8 frame tests --max-line-length=100 --extend-ignore=E203,W503
```
Then:
```bash
git add frame/frame_selector.py tests/test_foundation.py
git commit -m "feat(frame): expose ranking_ and scores_ fitted attributes

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: `get_feature_scores()` convenience method

**Files:**
- Modify: `frame/frame_selector.py` (new method on `FRAMESelector`)
- Test: `tests/test_foundation.py`

**Interfaces:**
- Consumes: `self.scores_`, `self.feature_names_in_`.
- Produces: `get_feature_scores() -> pd.Series` indexed by feature name, sorted descending, name `"score"`; raises `RuntimeError` if unfitted.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_foundation.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `"/c/Users/chinmay/anaconda3/envs/autods/python" -m pytest tests/test_foundation.py -q -p no:warnings -k "get_feature_scores"`
Expected: FAIL — `AttributeError: 'FRAMESelector' object has no attribute 'get_feature_scores'`.

- [ ] **Step 3: Add the method**

In `frame/frame_selector.py`, add this method to `FRAMESelector` (e.g., after `get_feature_names_out`):

```python
    def get_feature_scores(self) -> pd.Series:
        """Return per-feature relevance scores as a sorted pandas Series.

        Returns
        -------
        pd.Series
            Scores indexed by input feature name, sorted descending
            (highest relevance first).
        """
        if not hasattr(self, "scores_"):
            raise RuntimeError(
                "The FRAMESelector has not been fitted yet. "
                "Call fit() before get_feature_scores()."
            )
        return pd.Series(
            self.scores_, index=self.feature_names_in_, name="score"
        ).sort_values(ascending=False)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `"/c/Users/chinmay/anaconda3/envs/autods/python" -m pytest tests/test_foundation.py -q -p no:warnings`
Expected: PASS.

- [ ] **Step 5: Lint + commit**

Run:
```bash
"/c/Users/chinmay/anaconda3/envs/autods/python" -m black frame tests
"/c/Users/chinmay/anaconda3/envs/autods/python" -m flake8 frame tests --max-line-length=100 --extend-ignore=E203,W503
```
Then:
```bash
git add frame/frame_selector.py tests/test_foundation.py
git commit -m "feat(frame): add get_feature_scores() convenience method

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: `verbose` progress output

**Files:**
- Modify: `frame/frame_selector.py` (top-of-file import, RFE/SFS stages in `fit`)
- Test: `tests/test_foundation.py`

**Interfaces:**
- Consumes: `self.verbose`, the `rfe`/`forward_selector` stages in `fit`.
- Produces: stdout progress at `verbose>=1` (stage banners) and `verbose>=2` (adds per-stage timing). `verbose=0` prints nothing.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_foundation.py`:

```python
def test_verbose_zero_is_silent(clf_data, capsys):
    X, y = clf_data
    FRAMESelector(num_features=4, top_k=8, random_state=0, verbose=0).fit(X, y)
    assert capsys.readouterr().out == ""


def test_verbose_one_prints_stage_banners(clf_data, capsys):
    X, y = clf_data
    FRAMESelector(num_features=4, top_k=8, random_state=0, verbose=1).fit(X, y)
    out = capsys.readouterr().out
    assert "RFE" in out
    assert "Forward selection" in out


def test_verbose_two_includes_timing(clf_data, capsys):
    X, y = clf_data
    FRAMESelector(num_features=4, top_k=8, random_state=0, verbose=2).fit(X, y)
    out = capsys.readouterr().out
    assert "s" in out  # timing suffix like "in 0.12s"
    assert "RFE" in out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `"/c/Users/chinmay/anaconda3/envs/autods/python" -m pytest tests/test_foundation.py -q -p no:warnings -k "verbose"`
Expected: FAIL — `test_verbose_one_prints_stage_banners` fails (no output captured).

- [ ] **Step 3: Import `time` at the top of the file**

At the top of `frame/frame_selector.py`, add `import time` with the other stdlib import:

```python
import time
import numpy as np
import pandas as pd
```

- [ ] **Step 4: Add timing + banner around the RFE stage**

Find:

```python
        # Step 1: RFE
        rfe = RFE(estimator=self.model_, n_features_to_select=self.top_k)
        rfe.fit(X, y)
        rfe_selected_features = X.columns[rfe.support_]
```

Replace with:

```python
        # Step 1: RFE
        t_rfe = time.perf_counter()
        rfe = RFE(estimator=self.model_, n_features_to_select=self.top_k)
        rfe.fit(X, y)
        rfe_selected_features = X.columns[rfe.support_]
        if self.verbose:
            msg = (
                f"[FRAME] RFE kept {len(rfe_selected_features)} of "
                f"{X.shape[1]} features (top_k={self.top_k})"
            )
            if self.verbose >= 2:
                msg += f" in {time.perf_counter() - t_rfe:.2f}s"
            print(msg)
```

- [ ] **Step 5: Add timing + banner around the forward-selection stage**

Find:

```python
        forward_selector = SequentialFeatureSelector(self.model_, **sfs_kwargs)
        forward_selector.fit(X[rfe_selected_features], y)
        final_selected_features = rfe_selected_features[forward_selector.get_support()]
```

Replace with:

```python
        t_fwd = time.perf_counter()
        forward_selector = SequentialFeatureSelector(self.model_, **sfs_kwargs)
        forward_selector.fit(X[rfe_selected_features], y)
        final_selected_features = rfe_selected_features[forward_selector.get_support()]
        if self.verbose:
            msg = (
                f"[FRAME] Forward selection kept "
                f"{len(final_selected_features)} features"
            )
            if self.verbose >= 2:
                msg += f" in {time.perf_counter() - t_fwd:.2f}s"
            print(msg)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `"/c/Users/chinmay/anaconda3/envs/autods/python" -m pytest tests/test_foundation.py -q -p no:warnings`
Expected: PASS (all foundation tests).

- [ ] **Step 7: Full regression + lint**

Run:
```bash
"/c/Users/chinmay/anaconda3/envs/autods/python" -m black --check frame tests
"/c/Users/chinmay/anaconda3/envs/autods/python" -m flake8 frame tests --max-line-length=100 --extend-ignore=E203,W503
"/c/Users/chinmay/anaconda3/envs/autods/python" -m pytest tests/ -q -p no:warnings
```
Expected: black clean, flake8 clean, all tests pass (previous suite + new foundation tests; CSV tests skip).

- [ ] **Step 8: Commit**

```bash
git add frame/frame_selector.py tests/test_foundation.py
git commit -m "feat(frame): add verbose progress output to fit

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Docs — mark Foundation params as shipped

**Files:**
- Modify: `README.md` (Project Status + parameters table)
- Modify: `CLAUDE.md` (algorithm section note)

**Interfaces:**
- Consumes: nothing (documentation only).
- Produces: README reflects `num_features='auto'`, `n_jobs`, `verbose`, `ranking_`/`scores_`/`get_feature_scores` as available.

- [ ] **Step 1: Update the README Project Status "Available now" list**

In `README.md`, under **✅ Available now**, add these bullets:

```markdown
- `num_features='auto'` (score-plateau forward selection via `tol`).
- Feature importances exposed: `ranking_`, `scores_`, and `get_feature_scores()`.
- `n_jobs` (parallel forward-selection CV) and `verbose` progress output.
```

And remove `return_scores` from the 🚧 roadmap list, replacing it with a note:

```markdown
- (Feature scores now ship via `scores_` / `get_feature_scores()`.)
```

- [ ] **Step 2: Update the README parameters table**

In `README.md`, change the `return_scores` row and add rows so the table reads:

```markdown
| n_jobs         | int       | ✅ Available   | Parallelism for forward-selection CV and the default estimator.            |
| verbose        | int       | ✅ Available   | 0 silent, 1 stage banners, 2 adds timing.                                  |
| tol            | float     | ✅ Available   | Stop tolerance for `num_features='auto'`.                                  |
```

Change the `num_features` row's description to: `Final number of features; int, None (→ n_features // 2), or 'auto'.`

- [ ] **Step 3: Update CLAUDE.md algorithm note**

In `CLAUDE.md`, in the algorithm paragraph, append a sentence:

```markdown
`num_features` also accepts `'auto'` (score-plateau via `tol`). After fit, `ranking_`, `scores_`, and `get_feature_scores()` expose per-feature relevance; `n_jobs`/`verbose` are supported.
```

- [ ] **Step 4: Commit**

```bash
git add README.md CLAUDE.md
git commit -m "docs: mark Tier 1 Foundation params/attributes as shipped

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review Notes

- **Spec coverage:** `num_features='auto'` (T1), `ranking_`/`scores_` with RFE baseline + importance refinement and multiclass `coef_` handling (T2), `get_feature_scores()` (T3), `n_jobs` + estimator threading (T1), `verbose` 0/1/2 (T4), backward-compat test (T1), docs (T5). All spec sections covered.
- **Placeholder scan:** none — every step has concrete code/commands.
- **Type consistency:** `_get_importances` returns `np.ndarray | None`; `scores_`/`ranking_` are `np.ndarray`; `get_feature_scores` returns `pd.Series`. `sfs_n_features` is the single name used across T1 for the resolved count/`'auto'`. `num_features` accepts `Optional[Union[int, str]]` consistently.
- **Known consideration:** T2 mixes score scales (RFE baseline for non-selected vs model importances for selected). `ranking_` is computed to *guarantee* selected occupy ranks 1..k regardless of scale, and the signal/noise test asserts on `ranking_` (robust) rather than raw `scores_` magnitude.
