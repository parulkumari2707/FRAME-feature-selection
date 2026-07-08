# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`frame-feature-selector` is a small PyPI library implementing **FRAME** (Forward Recursive Adaptive Model Extraction), a two-stage hybrid feature selection method. The public surface is a single scikit-learn-style transformer: `FRAMESelector` in [frame/frame_selector.py](frame/frame_selector.py), re-exported from [frame/__init__.py](frame/__init__.py).

## Commands

```bash
pip install -e .                    # editable install of the package
pip install -r dev_requirements.txt # dev tools: pytest, pytest-cov, flake8, black

pytest                              # run tests that use synthetic/generated data
pytest tests/test_frame_selector.py # run a single test file
pytest tests/test_frame_selector.py::test_frame_fit  # run a single test

flake8 frame tests                  # lint
black frame tests                   # format
```

## The algorithm (why it's structured this way)

`FRAMESelector.fit` runs two sequential sklearn selectors on top of an XGBoost (or user-supplied) estimator:

1. **RFE** narrows the full feature set down to `top_k` features (default 20).
2. **SequentialFeatureSelector** (forward) then picks the final `num_features` from that `top_k` subset.

Constraints enforced in `fit`: `top_k <= n_features` and `num_features <= top_k`. If `num_features` is `None` it defaults to `n_features // 2`. If `model` is `None`, the task is auto-detected via `type_of_target(y)` — `binary`/`multiclass` → `XGBClassifier`, otherwise `XGBRegressor`, seeded with `random_state`. A user-supplied `model` is `clone`d into `self.model_` and never mutated (sklearn contract). Result is stored in `self.selected_features_` (a list of column-name strings). numpy inputs are coerced to a DataFrame with generated `feature_{i}` names, so selected features are always strings. `fit` also records `n_features_in_` / `feature_names_in_` and the selector exposes `get_support()` / `get_feature_names_out()`, so it works inside sklearn `Pipeline`/`ColumnTransformer`. `num_features` also accepts `'auto'` (score-plateau via `tol`). After fit, `ranking_`, `scores_`, and `get_feature_scores()` expose per-feature relevance; `n_jobs`/`verbose` are supported.

## Environment

There is no committed venv. Use the conda env `autods` — its interpreter is `/c/Users/chinmay/anaconda3/envs/autods/python` (numpy, pandas, sklearn 1.7, xgboost 3.2, pytest, flake8, black). The system default `python` (3.14) has no scientific deps. The full `tests/` suite takes ~4 min (SequentialFeatureSelector is slow).

## Critical gotchas

- **CSV-dependent tests skip when data is absent.** `*.csv` is gitignored and no `data/` directory is committed. `test_frame_cardiovascular.py`, `test_frame_student.py`, `test_frame_parkinsons.py`, and `synthetic_test/test_synthetic.py` are guarded with `pytest.mark.skipif`/`pytest.skip` on file existence, so `pytest` is green on a clean clone (those tests just skip). Drop the CSVs into `data/` to actually run them.
- **Packaging lives in [pyproject.toml](pyproject.toml)** (setuptools). There is no `setup.py`. The version is read statically from [frame/version.py](frame/version.py) via `[tool.setuptools.dynamic]` — keep the `__version__ = "x.y.z"` format intact. There is intentionally **no** console entry point (the old broken `frame-selector = frame.frame_selector:main` was removed).
- **Lint/format is enforced in CI** ([.github/workflows/ci.yml](.github/workflows/ci.yml)): `flake8 --max-line-length=100 --extend-ignore=E203,W503` and `black --check` (line-length 100, configured in pyproject). Run `black frame tests` and `flake8 frame tests` before pushing.
- **The dead duplicate class is gone.** `frame/frame_selector.py` now contains a single live implementation.
