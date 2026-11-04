"""Tests for leave-one-battery-out permutation importance (battery_soh.explain)."""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd
import pytest

from battery_soh.explain import grouped_permutation_importance
from battery_soh.models.zoo import make_model


@pytest.fixture(scope="module")
def data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    n_per, n_groups = 40, 4
    X = pd.DataFrame(
        {
            "strong": rng.normal(size=n_per * n_groups),
            "weak": rng.normal(size=n_per * n_groups),
            "noise": rng.normal(size=n_per * n_groups),
        }
    )
    y = 90 + 5.0 * X["strong"] + 1.0 * X["weak"] + rng.normal(0, 0.2, len(X))
    groups = np.repeat([f"B{g}" for g in range(n_groups)], n_per)
    return X, y.to_numpy(), groups


RIDGE = functools.partial(make_model, "ridge")


def test_importance_ranks_features_by_their_effect(
    data: tuple[pd.DataFrame, np.ndarray, np.ndarray],
) -> None:
    X, y, groups = data
    result = grouped_permutation_importance(RIDGE, X, y, groups, n_repeats=5)
    summary = result.summary.set_index("feature")
    assert list(result.summary["feature"]) == ["strong", "weak", "noise"]
    assert list(result.summary["rank"]) == [1, 2, 3]
    assert summary.loc["strong", "importance_mean"] > 3 * summary.loc["weak", "importance_mean"]
    assert summary.loc["weak", "importance_mean"] > 0.3
    assert abs(summary.loc["noise", "importance_mean"]) < 0.05
    assert (summary["n_groups"] == 4).all()
    assert len(result.per_group) == 4 * 3
    assert result.metric == "mae"
    # Baseline = held-out MAE of the unpermuted model, identical for all features of a fold.
    assert result.per_group.groupby("group")["baseline"].nunique().eq(1).all()


def test_importance_is_deterministic_and_order_independent(
    data: tuple[pd.DataFrame, np.ndarray, np.ndarray],
) -> None:
    X, y, groups = data
    a = grouped_permutation_importance(RIDGE, X, y, groups, n_repeats=3, random_state=1)
    b = grouped_permutation_importance(
        RIDGE, X[["noise", "strong", "weak"]], y, groups, n_repeats=3, random_state=1
    )
    left = a.per_group.set_index(["group", "feature"]).sort_index()
    right = b.per_group.set_index(["group", "feature"]).sort_index()
    pd.testing.assert_frame_equal(left, right, rtol=1e-9)


def test_feature_groups_are_permuted_jointly(
    data: tuple[pd.DataFrame, np.ndarray, np.ndarray],
) -> None:
    X, y, groups = data
    result = grouped_permutation_importance(
        RIDGE,
        X,
        y,
        groups,
        feature_groups={"signal": ["strong", "weak"], "noise": ["noise"]},
        n_repeats=3,
        metric="rmse",
    )
    summary = result.summary.set_index("feature")
    assert list(summary.index) == ["signal", "noise"]
    assert summary.loc["signal", "importance_mean"] > 3.0


def test_validation(data: tuple[pd.DataFrame, np.ndarray, np.ndarray]) -> None:
    X, y, groups = data
    with pytest.raises(ValueError, match="DataFrame"):
        grouped_permutation_importance(RIDGE, X.to_numpy(), y, groups)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n_repeats"):
        grouped_permutation_importance(RIDGE, X, y, groups, n_repeats=0)
    with pytest.raises(ValueError, match="metric"):
        grouped_permutation_importance(RIDGE, X, y, groups, metric="r2")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="equal length"):
        grouped_permutation_importance(RIDGE, X, y[:-1], groups)
    with pytest.raises(ValueError, match="unknown columns"):
        grouped_permutation_importance(RIDGE, X, y, groups, feature_groups={"g": ["nope"]})
    with pytest.raises(ValueError, match="no columns"):
        grouped_permutation_importance(RIDGE, X, y, groups, feature_groups={"g": []})
