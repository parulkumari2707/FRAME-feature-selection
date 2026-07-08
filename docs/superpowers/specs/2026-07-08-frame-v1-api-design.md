# FRAMESelector v1.0 API Design

**Date:** 2026-07-08
**Status:** Approved (design); first implementation slice pending plan
**Related:** README (spec/roadmap), CLAUDE.md, Tier 0 foundation work

## Context

`frame-feature-selector` implements FRAME (Forward Recursive Adaptive Model
Extraction), a two-stage hybrid feature selector (`RFE` → forward
`SequentialFeatureSelector`) exposed as a single scikit-learn transformer,
`FRAMESelector`. The initial version backs the author's research paper
(arXiv:2501.11972); the library is being extended with new findings, so
**correctness, reproducibility, and defensible measurement matter more than
usual**.

Tier 0 (foundation) is complete: `random_state`, no-mutation `model_`, the
sklearn selector surface (`get_support`, `get_feature_names_out`,
`n_features_in_`, `feature_names_in_`), packaging via `pyproject.toml`, and CI.

This document defines the **target v1.0 public API contract** for
`FRAMESelector` so that the planned algorithmic features land as **non-breaking
additions**, and specifies the **first implementation slice** ("Foundation +
cheap wins").

## Goals

- Define one stable, backward-compatible `FRAMESelector` contract for v1.0.
- Keep the API scikit-learn idiomatic (single param-driven estimator; scores as
  fitted attributes, not return values).
- Implement the Foundation slice now; leave clear, reserved room for later
  features (consensus ensemble, stability selection, Boruta mode).

## Non-Goals (deferred; documented as roadmap)

`strategy='consensus'` (multi-ranker aggregation), `stability=` (bootstrap
stability selection), Boruta all-relevant mode, SHAP-based scoring, automatic
categorical/NaN handling, plotting helpers, and the benchmark suite. The
contract reserves parameters for these so they arrive without breaking changes.

## Architecture Decision

**One class, param-driven** (`FRAMESelector`). Rationale:

- Most scikit-learn idiomatic — users expect a single estimator whose behavior
  varies by constructor params, and it composes with `GridSearchCV`/`Pipeline`.
- Cheapest to grow — each new capability is a new param + internal branch, no
  new class scaffolding or duplicated sklearn plumbing.
- Backward compatible — existing calls keep identical behavior; every new param
  defaults to today's behavior.

A `strategy=` parameter groups the larger behaviors so the param list stays
legible as the library grows. Only `strategy='rfe_forward'` (the current
pipeline) is implemented in this slice.

Alternatives considered and rejected: multiple classes + shared base (more
surface to maintain, splits a small library); strategy objects via composition
(most extensible but heavier and less beginner-friendly, uncommon in sklearn).

## Target v1.0 Constructor Contract

```python
FRAMESelector(
    model=None,                 # existing — base estimator; default XGBoost by task
    num_features=None,          # existing — int; now ALSO accepts 'auto'; None -> n_features // 2
    top_k=20,                   # existing — features kept by RFE before forward selection
    random_state=None,          # existing — seeds default estimator / stochastic steps
    n_jobs=None,                # NEW (this slice) — parallelism for forward-selection CV + estimator
    verbose=0,                  # NEW (this slice) — 0 silent, 1 stage banners, 2 adds timing
    tol=None,                   # NEW (this slice) — stop tolerance, used only when num_features='auto'
    # ---- reserved in the contract; NOT implemented this slice ----
    strategy='rfe_forward',     # later: 'consensus' | 'boruta'
    stability=None,             # later: int -> bootstrap stability selection
)
```

Backward compatibility: a call using only the existing four parameters produces
**byte-identical** results to the pre-slice implementation.

## First Implementation Slice — "Foundation + cheap wins"

### 1. `num_features='auto'`
Delegate to scikit-learn's native
`SequentialFeatureSelector(n_features_to_select='auto', tol=tol)` (available
since sklearn 1.1): forward selection keeps adding features until the
cross-validated score improvement drops below `tol`.

- `num_features=None` → unchanged: `max(1, n_features // 2)`.
- `num_features=<int>` → unchanged: exactly that many.
- `num_features='auto'` → sklearn auto mode with `tol` (opt-in).
- `tol` is only meaningful with `'auto'`; validate/ignore otherwise. If `'auto'`
  and `tol is None`, use sklearn's default behavior for `tol=None` (select until
  half the features, per sklearn) — documented explicitly.
- The `num_features <= top_k` constraint still applies for the int case; for
  `'auto'` the forward stage runs within the `top_k` RFE-selected subset.

### 2. `ranking_` (fitted attribute, always populated)
1-based integer rank over **all** input features, aligned to
`feature_names_in_`, mirroring `RFE.ranking_` semantics (1 = best).

- Selected features receive ranks `1..k`, ordered by their `scores_` (see below;
  highest score = rank 1).
- Non-selected features receive ranks `k+1 ..` derived from the RFE elimination
  order (features eliminated later rank better).

### 3. `scores_` (fitted attribute, always populated)
Float relevance score per input feature, aligned to `feature_names_in_`, higher
= better.

- Baseline (always available): every input feature gets a score derived from the
  RFE stage (`RFE.ranking_`, inverted so higher = better), guaranteeing a finite,
  comparable score for **all** features regardless of estimator type.
- Refinement (when available): if the final fitted model exposes
  `feature_importances_` or `coef_`, selected features' scores are replaced by
  those importances (abs value for `coef_`), giving finer ordering among the
  selected set. Estimators exposing neither fall back to the RFE baseline only —
  no error.
- Exact normalization documented in the docstring; values need not sum to 1 but
  must be finite and comparable across features.

### 4. `n_jobs`
Thread into `SequentialFeatureSelector(n_jobs=...)` (parallelizes its
candidate-feature CV evaluation — the dominant cost) and into the default
XGBoost estimator when `model is None`. Results must be identical to the serial
path (parallelism must not change selection).

### 5. `verbose`
- `0` — silent (default; unchanged).
- `1` — stage banners: RFE complete, forward-selection complete, N features
  selected.
- `2` — level 1 plus wall-clock timing per stage.

### New fitted attributes / methods after this slice
- `ranking_`, `scores_` (above).
- `get_feature_scores()` — convenience returning
  `pd.Series(scores_, index=feature_names_in_)` sorted descending.
- Unchanged: `selected_features_`, `model_`, `n_features_in_`,
  `feature_names_in_`, `get_support()`, `get_feature_names_out()`.

## Error Handling

- `num_features`: accept `None`, positive `int`, or the literal `'auto'`; any
  other value raises `ValueError` with a clear message.
- `tol`: if provided with a non-`'auto'` `num_features`, ignore it but document
  the behavior (no error, to stay forgiving in grid search).
- `n_jobs`, `verbose`: standard sklearn semantics; invalid types raise on use.
- All existing validation (`top_k <= n_features`, `num_features <= top_k` for the
  int case, input dtype/shape checks) is retained.

## Testing Strategy (TDD)

Written test-first, one behavior per test, run against the `autods` env.

- `num_features='auto'` selects a sensible count within `top_k`; a stricter
  `tol` selects fewer features than a looser one.
- `ranking_`: shape `(n_features_in_,)`, values are a permutation of `1..n`;
  `ranking_` at selected positions are exactly `1..k`.
- `scores_`: aligned to `feature_names_in_`, all finite; on a signal-vs-noise
  synthetic dataset, informative features score above noise features.
- `get_feature_scores()` returns a descending `pd.Series` indexed by feature
  names.
- `n_jobs`: `n_jobs=-1` yields identical `selected_features_`/`ranking_` to
  `n_jobs=None`.
- `verbose`: emits expected output at levels 1 and 2 (captured), silent at 0.
- **Backward compatibility:** a default-parameter call yields identical
  `selected_features_` to the pre-slice behavior (guard against regressions).

## Rollout / Sequencing

1. This slice (Foundation) — this spec → implementation plan → TDD build.
2. Later slices (separate spec/plan each): `strategy='consensus'`, then
   `stability=`, then Boruta mode, then SHAP scoring, then benchmark suite.

Because every later feature is an additive param or a new `strategy` value, none
require breaking the contract defined here.
