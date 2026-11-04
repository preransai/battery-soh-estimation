"""Tests for the reference baselines."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone

from battery_soh.evaluation.baselines import (
    BASELINE_NAMES,
    BASELINES,
    MeanBaseline,
    baseline_features,
    make_baseline,
)


def test_mean_baseline_predicts_training_mean_and_ignores_inputs() -> None:
    X = np.array([[1.0, np.nan], [2.0, 5.0], [np.nan, 1.0]])
    y = np.array([80.0, 90.0, 100.0])
    model = MeanBaseline().fit(X, y)
    np.testing.assert_array_equal(model.predict(np.full((4, 2), np.nan)), np.full(4, 90.0))
    assert clone(model).get_params() == {}
    with pytest.raises(ValueError, match="features"):
        model.predict(np.ones((2, 3)))  # column count is still checked


def test_registry() -> None:
    assert BASELINE_NAMES == ("mean", "cycle_only", "capacity_oracle")
    assert baseline_features("mean") is None
    assert baseline_features("cycle_only") == ("ctx_cycle_index",)
    assert baseline_features("capacity_oracle") == ("leg_capacity_ah",)
    oracle = BASELINES["capacity_oracle"]
    assert oracle.leaky
    assert "LEAKY" in oracle.label
    assert not BASELINES["cycle_only"].leaky
    with pytest.raises(ValueError, match="Unknown baseline"):
        make_baseline("oracle")
    with pytest.raises(ValueError, match="Unknown baseline"):
        baseline_features("oracle")


def test_cycle_only_learns_a_linear_fade() -> None:
    cycles = np.arange(50, dtype=float)[:, None]
    soh = 100.0 - 0.2 * cycles[:, 0]
    model = make_baseline("cycle_only").fit(cycles, soh)
    # Ridge(alpha=1) on 50 standardised rows shrinks the slope by 50/51 only.
    assert model.predict(np.array([[40.0]]))[0] == pytest.approx(92.0, abs=0.2)
    assert np.all(np.diff(model.predict(cycles)) < 0)


@pytest.mark.parametrize("name", BASELINE_NAMES)
def test_every_baseline_fits_and_predicts(name: str) -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 1))
    y = 90 + 5 * X[:, 0]
    pred = make_baseline(name).fit(X, y).predict(X)
    assert pred.shape == (30,)
    assert np.all(np.isfinite(pred))
