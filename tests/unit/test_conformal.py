"""Tests for exact split-conformal quantiles and battery-grouped conformal intervals."""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from battery_soh.uncertainty.conformal import (
    DEFAULT_CONFIDENCES,
    GroupConformalRegressor,
    confidence_key,
    confidence_quantiles,
    conformal_quantile,
    conformal_rank,
    min_calibration_size,
)

ALPHAS = (0.2, 0.1, 0.05)


class OLS:
    """Minimal least-squares regressor (fast, deterministic) for simulations."""

    def fit(self, X: Any, y: Any) -> OLS:
        A = np.column_stack([np.ones(len(y)), np.asarray(X, dtype=float)])
        self.beta_ = np.linalg.lstsq(A, np.asarray(y, dtype=float), rcond=None)[0]
        return self

    def predict(self, X: Any) -> np.ndarray:
        return self.beta_[0] + np.asarray(X, dtype=float) @ self.beta_[1:]


class Recorder:
    """Regressor that remembers its training rows and predicts a constant."""

    def fit(self, X: Any, y: Any) -> Recorder:
        self.X_ = np.asarray(X, dtype=float).copy()
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X: Any) -> np.ndarray:
        return np.full(len(X), self.mean_)


# --------------------------------------------------------------------- exact rank / quantile
@pytest.mark.parametrize(
    ("n", "alpha", "k"),
    [
        (100, 0.1, 91),  # v1 returned rank 92 here
        (19, 0.05, 19),
        (9, 1 - 0.9, 9),  # the float 0.09999999999999998 is read as exactly 1/10
        (9, 0.1, 9),
        (8, 0.1, 9),  # k > n -> +inf
        (4, 0.2, 4),
        (0, 0.1, 1),
        (99, 0.05, 95),
    ],
)
def test_conformal_rank_is_exact(n: int, alpha: float, k: int) -> None:
    assert conformal_rank(n, alpha) == k


@settings(max_examples=200, deadline=None, derandomize=True)
@given(
    n=st.integers(min_value=0, max_value=400),
    alpha_milli=st.integers(min_value=1, max_value=999),
)
def test_conformal_rank_matches_rational_arithmetic(n: int, alpha_milli: int) -> None:
    alpha = alpha_milli / 1000
    expected = math.ceil((n + 1) * (1 - Fraction(alpha_milli, 1000)))
    assert conformal_rank(n, alpha) == expected


@settings(max_examples=150, deadline=None, derandomize=True)
@given(
    scores=st.lists(
        st.floats(min_value=0.0, max_value=1e6, allow_nan=False), min_size=0, max_size=60
    ),
    alpha=st.sampled_from([0.01, 0.05, 0.1, 0.2, 0.25, 0.5, 0.9]),
)
def test_conformal_quantile_is_kth_order_statistic(scores: list[float], alpha: float) -> None:
    n = len(scores)
    k = math.ceil((n + 1) * (1 - Fraction(alpha).limit_denominator(1000)))
    q = conformal_quantile(scores, alpha)
    if k > n:
        assert q == math.inf
    else:
        assert q == sorted(scores)[k - 1]


def test_conformal_quantile_returns_inf_not_max_when_too_few_scores() -> None:
    scores = [0.5, 1.0, 2.0, 3.0, 4.0]  # n = 5 < 9 needed for alpha = 0.1
    assert conformal_quantile(scores, 0.1) == math.inf
    assert conformal_quantile([], 0.5) == math.inf
    # With enough scores the quantile is finite and an actual score.
    assert conformal_quantile(np.arange(1.0, 10.0), 0.1) == 9.0


@pytest.mark.parametrize(("alpha", "expected"), [(0.2, 4), (0.1, 9), (0.05, 19), (0.3, 3)])
def test_min_calibration_size(alpha: float, expected: int) -> None:
    assert min_calibration_size(alpha) == expected
    rng = np.random.default_rng(0)
    assert math.isinf(conformal_quantile(rng.random(expected - 1), alpha))
    assert math.isfinite(conformal_quantile(rng.random(expected), alpha))


@settings(max_examples=100, deadline=None, derandomize=True)
@given(
    scores=st.lists(st.floats(min_value=0, max_value=100, allow_nan=False), max_size=50),
    a1=st.floats(min_value=0.001, max_value=0.999),
    a2=st.floats(min_value=0.001, max_value=0.999),
)
def test_conformal_quantile_is_monotone_in_alpha(scores: list[float], a1: float, a2: float) -> None:
    lo, hi = sorted((a1, a2))
    # Smaller miscoverage -> higher (or equal) quantile.
    assert conformal_quantile(scores, lo) >= conformal_quantile(scores, hi)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5, float("nan")])
def test_invalid_alpha_raises(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        conformal_quantile([1.0, 2.0], alpha)


def test_invalid_scores_raise() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        conformal_quantile(np.ones((3, 2)), 0.1)
    with pytest.raises(ValueError, match="NaN"):
        conformal_quantile([1.0, float("nan")], 0.1)
    with pytest.raises(ValueError, match="n must be"):
        conformal_rank(-1, 0.1)


@pytest.mark.parametrize(("n", "alpha"), [(19, 0.1), (49, 0.1), (20, 0.2), (39, 0.05), (5, 0.1)])
def test_split_conformal_coverage_on_exchangeable_data(n: int, alpha: float) -> None:
    """Marginal coverage equals k / (n + 1) (or 1 when q = inf), within binomial noise."""
    rng = np.random.default_rng(1234 + n)
    trials = 4000
    draws = rng.exponential(size=(trials, n + 1))  # continuous: no ties
    covered = np.array([row[n] <= conformal_quantile(row[:n], alpha) for row in draws], dtype=float)
    k = conformal_rank(n, alpha)
    exact = 1.0 if k > n else k / (n + 1)
    tol = 4.0 * math.sqrt(max(exact * (1 - exact), 1e-12) / trials) + 1e-9
    assert abs(covered.mean() - exact) <= tol
    # The finite-sample guarantee: never below 1 - alpha (beyond noise).
    assert covered.mean() >= 1 - alpha - tol
    assert exact <= 1 - alpha + 1 / (n + 1) + 1e-12 or k > n


def test_confidence_helpers() -> None:
    assert confidence_key(0.9) == "0.9"
    assert confidence_key("0.950") == "0.95"
    assert confidence_key(1 - 0.2) == "0.8"
    with pytest.raises(ValueError, match="confidence"):
        confidence_key(1.0)
    scores = np.linspace(0.1, 10.0, 100)
    out = confidence_quantiles(scores)
    assert list(out) == ["0.8", "0.9", "0.95"]
    for c in DEFAULT_CONFIDENCES:
        assert out[confidence_key(c)] == conformal_quantile(scores, 1 - c)
    assert confidence_quantiles(scores[:5])["0.95"] == math.inf


# --------------------------------------------------------------------- grouped conformal
def _grouped_problem(
    rng: np.random.Generator, n_groups: int, rows: int, group_sd: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = rng.normal(size=(n_groups * rows, 2))
    groups = np.repeat([f"B{g:02d}" for g in range(n_groups)], rows)
    offsets = rng.normal(0.0, group_sd, n_groups).repeat(rows)
    y = 3.0 * X[:, 0] - X[:, 1] + offsets + rng.normal(0.0, 1.0, len(groups))
    return X, y, groups


def test_scores_are_out_of_group_residuals() -> None:
    rng = np.random.default_rng(3)
    X, y, groups = _grouped_problem(rng, 4, 6, 1.0)
    gcr = GroupConformalRegressor(Recorder, alphas=(0.1,)).fit(X, y, groups)
    assert gcr.groups_ == ("B00", "B01", "B02", "B03")
    for g, model in gcr.fold_estimators_.items():
        held_out_rows = X[groups == g]
        # The fold model never saw a row of its held-out battery ...
        assert not any((row == model.X_).all(axis=1).any() for row in held_out_rows)
        # ... and that battery's scores are residuals of exactly that model.
        mask = gcr.score_groups_ == g
        np.testing.assert_allclose(gcr.oof_predictions_[mask], np.mean(y[groups != g]))
    np.testing.assert_allclose(gcr.scores_, np.abs(y - gcr.oof_predictions_))
    assert gcr.n_scores_ == len(y)
    for a in (0.1,):
        assert gcr.quantiles_[a] == conformal_quantile(gcr.scores_, a)
    # The full model is trained on every training row.
    assert gcr.estimator_.X_.shape == X.shape


def test_cv_plus_bounds_match_brute_force() -> None:
    rng = np.random.default_rng(7)
    X, y, groups = _grouped_problem(rng, 3, 7, 0.5)
    X_new = rng.normal(size=(4, 2))
    alpha = 0.2
    gcr = GroupConformalRegressor(OLS, alphas=(alpha,), method="cv_plus").fit(X, y, groups)
    yhat, lo, hi = gcr.predict_interval(X_new, alpha)
    n = len(y)
    k_hi = math.ceil((n + 1) * (1 - Fraction(1, 5)))
    k_lo = math.floor(Fraction(1, 5) * (n + 1))
    for j in range(len(X_new)):
        mu = np.array(
            [gcr.fold_estimators_[g].predict(X_new[j : j + 1])[0] for g in gcr.score_groups_]
        )
        assert lo[j] == pytest.approx(sorted(mu - gcr.scores_)[k_lo - 1], abs=1e-12)
        assert hi[j] == pytest.approx(sorted(mu + gcr.scores_)[k_hi - 1], abs=1e-12)
    np.testing.assert_allclose(yhat, OLS().fit(X, y).predict(X_new))
    # predict_intervals (fold predictions computed once) agrees with predict_interval.
    lo2, hi2 = gcr.predict_intervals(X_new)[alpha]
    np.testing.assert_array_equal(lo, lo2)
    np.testing.assert_array_equal(hi, hi2)


def test_cv_method_is_symmetric_half_width() -> None:
    rng = np.random.default_rng(11)
    X, y, groups = _grouped_problem(rng, 5, 8, 1.0)
    gcr = GroupConformalRegressor(OLS, method="cv").fit(X, y, groups)
    X_new = rng.normal(size=(3, 2))
    for alpha in ALPHAS:
        yhat, lo, hi = gcr.predict_interval(X_new, alpha)
        q = conformal_quantile(gcr.scores_, alpha)
        np.testing.assert_allclose(hi - yhat, q)
        np.testing.assert_allclose(yhat - lo, q)
    assert gcr.quantiles_by_confidence() == {
        "0.8": gcr.quantiles_[0.2],
        "0.9": gcr.quantiles_[0.1],
        "0.95": gcr.quantiles_[0.05],
    }
    # Wider intervals for higher confidence.
    q = gcr.quantiles_by_confidence()
    assert q["0.8"] <= q["0.9"] <= q["0.95"]


@pytest.mark.parametrize("method", ["cv", "cv_plus"])
def test_too_few_scores_give_unbounded_intervals(method: str) -> None:
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 1.1, 1.9, 3.2])
    groups = np.array(["a", "a", "b", "b"])
    gcr = GroupConformalRegressor(OLS, alphas=(0.05,), method=method).fit(X, y, groups)  # type: ignore[arg-type]
    assert gcr.quantiles_[0.05] == math.inf
    _, lo, hi = gcr.predict_interval(np.array([[1.5]]), 0.05)
    assert lo[0] == -math.inf
    assert hi[0] == math.inf


@pytest.mark.parametrize("method", ["cv", "cv_plus"])
@pytest.mark.parametrize("alpha", [0.1, 0.2])
def test_group_conformal_coverage_on_new_batteries(method: str, alpha: float) -> None:
    """Exchangeable batteries with battery-level offsets: coverage of a NEW battery ~ 1 - alpha."""
    rng = np.random.default_rng(2024)
    trials, n_groups, rows = 300, 6, 5
    coverage = np.empty(trials)
    for t in range(trials):
        X, y, groups = _grouped_problem(rng, n_groups + 1, rows, group_sd=1.0)
        test = groups == f"B{n_groups:02d}"
        gcr = GroupConformalRegressor(OLS, alphas=(alpha,), method=method)  # type: ignore[arg-type]
        gcr.fit(X[~test], y[~test], groups[~test])
        _, lo, hi = gcr.predict_interval(X[test], alpha)
        coverage[t] = np.mean((y[test] >= lo) & (y[test] <= hi))
    se = coverage.std(ddof=1) / math.sqrt(trials)
    assert coverage.mean() >= 1 - alpha - 3 * se - 0.01
    assert coverage.mean() <= 1 - alpha + 0.08  # calibrated, not trivially wide


def test_group_conformal_accepts_dataframes_and_validates() -> None:
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(5)
    X, y, groups = _grouped_problem(rng, 3, 4, 0.0)
    frame = pd.DataFrame(X, columns=["a", "b"], index=np.arange(100, 100 + len(y)))
    gcr = GroupConformalRegressor(OLS).fit(frame, pd.Series(y), pd.Series(groups))
    assert gcr.predict(frame).shape == (len(y),)
    with pytest.raises(RuntimeError, match="not fitted"):
        GroupConformalRegressor(OLS).predict(X)
    with pytest.raises(ValueError, match="equal length"):
        GroupConformalRegressor(OLS).fit(X, y[:-1], groups)
    with pytest.raises(ValueError, match="at least 2"):
        GroupConformalRegressor(OLS).fit(X, y, np.full(len(y), "only"))
    with pytest.raises(ValueError, match="finite"):
        GroupConformalRegressor(OLS).fit(X, np.where(np.arange(len(y)) == 0, np.nan, y), groups)
    with pytest.raises(ValueError, match="method"):
        GroupConformalRegressor(OLS, method="jackknife")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="alphas"):
        GroupConformalRegressor(OLS, alphas=())
