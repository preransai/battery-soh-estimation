"""Tests for point/interval metrics and per-battery summaries."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    max_error,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from battery_soh.evaluation.metrics import (
    SUMMARY_ROWS,
    bootstrap_ci,
    interval_metrics,
    interval_score,
    regression_metrics,
    summarize_by_group,
)


def test_regression_metrics_match_sklearn() -> None:
    rng = np.random.default_rng(0)
    y = rng.uniform(70, 100, 200)
    yhat = y + rng.normal(0.5, 2.0, 200)
    m = regression_metrics(y, yhat)
    assert m["mae"] == pytest.approx(mean_absolute_error(y, yhat))
    assert m["rmse"] == pytest.approx(math.sqrt(mean_squared_error(y, yhat)))
    assert m["r2"] == pytest.approx(r2_score(y, yhat))
    assert m["max_abs_error"] == pytest.approx(max_error(y, yhat))
    assert m["bias"] == pytest.approx(np.mean(yhat - y))
    assert m["n"] == 200


def test_bias_sign_convention() -> None:
    # Positive bias = the model over-estimates health (the dangerous direction).
    assert regression_metrics([80.0, 90.0], [82.0, 92.0])["bias"] == pytest.approx(2.0)


def test_r2_is_nan_when_undefined() -> None:
    assert math.isnan(regression_metrics([90.0, 90.0, 90.0], [89.0, 90.0, 91.0])["r2"])
    assert math.isnan(regression_metrics([90.0], [91.0])["r2"])


def test_regression_metrics_validation() -> None:
    with pytest.raises(ValueError, match="equal length"):
        regression_metrics([1.0, 2.0], [1.0])
    with pytest.raises(ValueError, match="empty"):
        regression_metrics([], [])
    with pytest.raises(ValueError, match="finite"):
        regression_metrics([1.0, np.nan], [1.0, 2.0])
    # Column vectors (e.g. PLS output) are accepted.
    assert regression_metrics([1.0, 2.0], np.array([[1.0], [2.0]]))["mae"] == 0.0


def test_interval_score_hand_computed() -> None:
    # Gneiting & Raftery (2007) eq. 43 with alpha = 0.1 (2/alpha = 20).
    y = np.array([5.0, 1.0, -1.0])
    lo = np.array([0.0, 0.0, 0.0])
    hi = np.array([4.0, 4.0, 4.0])
    np.testing.assert_allclose(interval_score(y, lo, hi, 0.1), [4 + 20 * 1, 4.0, 4 + 20 * 1])
    assert interval_score([1.0], [-np.inf], [2.0], 0.1)[0] == np.inf


def test_interval_score_is_minimised_by_true_quantiles() -> None:
    """Propriety check: the true central interval beats narrower and wider ones."""
    rng = np.random.default_rng(1)
    y = rng.normal(size=200_000)
    alpha = 0.1
    z = 1.6448536269514722  # standard normal 0.95 quantile

    def mean_score(h: float) -> float:
        return float(np.mean(interval_score(y, np.full_like(y, -h), np.full_like(y, h), alpha)))

    best = mean_score(z)
    assert best < mean_score(0.8 * z)
    assert best < mean_score(1.25 * z)


def test_interval_metrics() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    lo = np.array([0.0, 2.5, 2.0, -np.inf])
    hi = np.array([2.0, 3.0, 2.5, np.inf])
    m = interval_metrics(y, lo, hi, alpha=0.2)
    assert m["coverage"] == pytest.approx(0.5)
    assert m["below"] == pytest.approx(0.25)
    assert m["above"] == pytest.approx(0.25)
    assert m["mean_width"] == np.inf
    assert m["median_width"] == pytest.approx(1.25)
    assert m["frac_unbounded"] == pytest.approx(0.25)
    assert m["nominal_coverage"] == pytest.approx(0.8)
    assert m["interval_score"] == np.inf
    finite = interval_metrics(y[:3], lo[:3], hi[:3], alpha=0.2)
    expected = np.mean([2.0, 0.5 + 10 * 0.5, 0.5 + 10 * 0.5])
    assert finite["interval_score"] == pytest.approx(expected)
    assert "interval_score" not in interval_metrics(y, lo, hi)


@pytest.mark.parametrize(
    ("lo", "hi", "match"),
    [
        ([0.0, 3.0], [1.0, 2.0], "lower bound"),
        ([0.0, np.nan], [1.0, 2.0], "NaN"),
        ([0.0, np.inf], [1.0, np.inf], "< \\+inf"),
        ([0.0], [1.0], "equal length"),
    ],
)
def test_interval_validation(lo: list[float], hi: list[float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        interval_metrics([0.5, 1.5], lo, hi)


def test_interval_score_rejects_bad_alpha() -> None:
    with pytest.raises(ValueError, match="alpha"):
        interval_score([1.0], [0.0], [2.0], 1.0)


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "battery_id": ["B2"] * 3 + ["B1"] * 3,
            "y": [90.0, 85.0, 80.0, 95.0, 90.0, 85.0],
            "yhat": [91.0, 86.0, 81.0, 93.0, 90.0, 87.0],
            "lo": [89.0, 84.0, 82.0, 92.0, 88.0, 86.0],
            "hi": [93.0, 88.0, 84.0, 94.0, 92.0, 89.0],
        }
    )


def test_summarize_by_group() -> None:
    df = _predictions()
    out = summarize_by_group(df, "battery_id", lo_col="lo", hi_col="hi", alpha=0.1).set_index(
        "battery_id"
    )
    assert list(out.index) == ["B1", "B2", *SUMMARY_ROWS]
    b1 = regression_metrics([95.0, 90.0, 85.0], [93.0, 90.0, 87.0])
    b2 = regression_metrics([90.0, 85.0, 80.0], [91.0, 86.0, 81.0])
    assert out.loc["B1", "mae"] == pytest.approx(b1["mae"])
    assert out.loc["B2", "bias"] == pytest.approx(1.0)
    assert out.loc["mean", "mae"] == pytest.approx((b1["mae"] + b2["mae"]) / 2)  # macro average
    assert out.loc["worst", "mae"] == pytest.approx(max(b1["mae"], b2["mae"]))
    assert out.loc["worst", "r2"] == pytest.approx(min(b1["r2"], b2["r2"]))
    assert out.loc["pooled", "mae"] == pytest.approx(regression_metrics(df["y"], df["yhat"])["mae"])
    assert out.loc["B2", "coverage"] == pytest.approx(2 / 3)  # 80 < 82
    assert out.loc["B1", "coverage"] == pytest.approx(1 / 3)  # 95 > 94, 85 < 86
    assert out.loc["worst", "coverage"] == pytest.approx(1 / 3)
    assert out.loc["mean", "coverage"] == pytest.approx(0.5)
    assert out.loc["mean", "n"] == 6
    assert np.isfinite(out.loc["B1", "interval_score"])


def test_summarize_by_group_validation() -> None:
    df = _predictions()
    with pytest.raises(ValueError, match="together"):
        summarize_by_group(df, "battery_id", lo_col="lo")
    with pytest.raises(ValueError, match="not found"):
        summarize_by_group(df, "cell")
    with pytest.raises(ValueError, match="empty"):
        summarize_by_group(df.iloc[:0], "battery_id")


def test_bootstrap_ci() -> None:
    values = [1.2, 1.8, 2.5, 1.6]
    low, high = bootstrap_ci(values, n_boot=500, random_state=3)
    assert low <= np.mean(values) <= high
    assert (low, high) == bootstrap_ci(values, n_boot=500, random_state=3)  # deterministic
    assert bootstrap_ci([2.0], n_boot=50) == (2.0, 2.0)
    with pytest.raises(ValueError, match="finite"):
        bootstrap_ci([1.0, np.nan])
    with pytest.raises(ValueError, match="confidence"):
        bootstrap_ci([1.0, 2.0], confidence=1.5)
