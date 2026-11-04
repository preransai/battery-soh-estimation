"""Tests for nested (inner leave-one-group-out) feature selection for Ridge."""

from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from battery_soh.evaluation.nested import (
    DEFAULT_ALPHAS,
    FALLBACK_ALPHA,
    GROUP_COLUMN,
    InnerLoboScorer,
    NestedSelectionRidge,
    backward_elimination,
)
from battery_soh.evaluation.splits import leave_one_group_out
from battery_soh.models.zoo import make_model
from battery_soh.uncertainty.conformal import GroupConformalRegressor


def _cells(seed: int, *, offset_sd: float, n_groups: int = 5, n: int = 80) -> pd.DataFrame:
    """Synthetic cells: ``signal`` is a noisy unbiased indicator of age; ``nuisance`` is a
    precise indicator plus a per-cell offset (like the CV-phase duration in the NASA cells)."""
    rng = np.random.default_rng(seed)
    offsets = rng.normal(0.0, offset_sd, n_groups)
    parts = []
    for g in range(n_groups):
        age = rng.uniform(0.0, 1.0, n)
        parts.append(
            pd.DataFrame(
                {
                    "signal": age + rng.normal(0.0, 0.08, n),
                    "nuisance": age + offsets[g] + rng.normal(0.0, 0.01, n),
                    "group": f"G{g}",
                    "y": 100.0 - 20.0 * age + rng.normal(0.0, 0.3, n),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def _with_groups(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    return df[features].assign(**{GROUP_COLUMN: df["group"].to_numpy()})


def _oracle_decision(df: pd.DataFrame) -> tuple[str, ...]:
    """What one elimination step must do, recomputed from the inner-LOBO scores directly."""
    X = df[["signal", "nuisance"]].to_numpy()
    scorer = InnerLoboScorer(X, df["y"].to_numpy(), df["group"].to_numpy())
    both = scorer.best_alpha([0, 1], DEFAULT_ALPHAS)[1]
    only_signal = scorer.best_alpha([0], DEFAULT_ALPHAS)[1]
    only_nuisance = scorer.best_alpha([1], DEFAULT_ALPHAS)[1]
    if min(only_signal, only_nuisance) >= both:
        return ("signal", "nuisance")
    return ("signal",) if only_signal <= only_nuisance else ("nuisance",)


@pytest.mark.parametrize("offset_sd", [0.0, 0.3])
@pytest.mark.parametrize("seed", range(6))
def test_drops_a_feature_exactly_when_inner_folds_show_it_hurts(
    seed: int, offset_sd: float
) -> None:
    df = _cells(seed, offset_sd=offset_sd)
    result = backward_elimination(df[["signal", "nuisance"]], df["y"], df["group"])
    assert result.selected == _oracle_decision(df)
    assert result.inner_mae == result.path[-1].inner_mae
    for before, after in zip(result.path, result.path[1:], strict=False):
        assert after.inner_mae < before.inner_mae  # every accepted removal improved inner MAE
    assert result.alpha in DEFAULT_ALPHAS
    assert result.n_groups == 5
    assert not result.fallback


def test_per_group_offset_nuisance_is_usually_dropped_and_clean_feature_kept() -> None:
    # The same precise indicator is kept when it is clean and dropped when it carries a
    # cell-specific offset (like the CV-phase duration in the NASA cells).
    seeds = range(20)
    dropped_with_offset = sum(
        "nuisance"
        not in backward_elimination(d[["signal", "nuisance"]], d["y"], d["group"]).selected
        for d in (_cells(s, offset_sd=0.5) for s in seeds)
    )
    kept_without_offset = sum(
        "nuisance" in backward_elimination(d[["signal", "nuisance"]], d["y"], d["group"]).selected
        for d in (_cells(s, offset_sd=0.0) for s in seeds)
    )
    assert dropped_with_offset >= 16
    assert kept_without_offset == 20


def test_inner_scorer_matches_sklearn_ridge_pipeline() -> None:
    df = _cells(5, offset_sd=0.3)
    rng = np.random.default_rng(1)
    X = df[["signal", "nuisance"]].to_numpy().copy()
    X[rng.random(X.shape) < 0.1] = np.nan  # median imputation must match too
    X = np.column_stack([X, np.full(len(X), 3.0)])  # constant column (scale 1 in sklearn)
    y = df["y"].to_numpy()
    groups = df["group"].to_numpy()
    scorer = InnerLoboScorer(X, y, groups)
    for subset in ([0], [1], [0, 1], [0, 1, 2]):
        for alpha in (0.1, 1.0, 10.0):
            maes = []
            for train, test, _ in leave_one_group_out(groups):
                model = make_model("ridge", alpha=alpha).fit(X[train][:, subset], y[train])
                maes.append(np.mean(np.abs(model.predict(X[test][:, subset]) - y[test])))
            assert scorer.score(subset, alpha) == pytest.approx(np.mean(maes), rel=1e-9)


def test_selection_is_deterministic_and_row_order_invariant() -> None:
    df = _cells(7, offset_sd=0.3)
    columns = ["signal", "nuisance"]
    first = backward_elimination(df[columns], df["y"], df["group"])
    shuffled = df.sample(frac=1.0, random_state=0).reset_index(drop=True)
    second = backward_elimination(shuffled[columns], shuffled["y"], shuffled["group"])
    assert first.selected == second.selected
    assert first.alpha == second.alpha
    assert first.inner_mae == pytest.approx(second.inner_mae, rel=1e-12)


def test_min_features_and_tolerance_stop_elimination() -> None:
    df = _cells(0, offset_sd=0.3)
    columns = ["signal", "nuisance"]
    kept = backward_elimination(df[columns], df["y"], df["group"], min_features=2)
    assert kept.selected == ("signal", "nuisance")
    assert len(kept.path) == 1
    huge_tol = backward_elimination(df[columns], df["y"], df["group"], tol=1e6)
    assert huge_tol.selected == ("signal", "nuisance")


def test_single_group_falls_back_to_all_features() -> None:
    df = _cells(0, offset_sd=0.3)
    one = df[df["group"] == "G0"]
    result = backward_elimination(one[["signal", "nuisance"]], one["y"], one["group"])
    assert result.fallback
    assert result.selected == ("signal", "nuisance")
    assert result.alpha == FALLBACK_ALPHA
    assert np.isnan(result.inner_mae)
    assert result.to_dict()["inner_mae"] is None


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"alphas": ()}, "alphas"),
        ({"alphas": (0.0,)}, "alphas"),
        ({"alphas": (float("nan"),)}, "alphas"),
        ({"min_features": 0}, "min_features"),
        ({"tol": -1.0}, "tol"),
    ],
)
def test_invalid_arguments(kwargs: dict[str, object], match: str) -> None:
    df = _cells(0, offset_sd=0.3)
    with pytest.raises(ValueError, match=match):
        backward_elimination(df[["signal"]], df["y"], df["group"], **kwargs)  # type: ignore[arg-type]


def test_backward_elimination_needs_named_columns() -> None:
    df = _cells(0, offset_sd=0.3)
    with pytest.raises(ValueError, match="DataFrame"):
        backward_elimination(df[["signal"]].to_numpy(), df["y"], df["group"])  # type: ignore[arg-type]


def test_scorer_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="finite"):
        InnerLoboScorer(np.ones((4, 1)), [1.0, np.nan, 2.0, 3.0], ["a", "a", "b", "b"])
    with pytest.raises(ValueError, match="2 distinct groups"):
        InnerLoboScorer(np.ones((2, 1)), [1.0, 2.0], ["a", "a"])
    scorer = InnerLoboScorer(np.arange(4.0)[:, None], [1.0, 2.0, 3.0, 4.0], ["a", "a", "b", "b"])
    with pytest.raises(ValueError, match="at least one feature"):
        scorer.score([], 1.0)
    with pytest.raises(ValueError, match="alpha"):
        scorer.score([0], -1.0)


def test_estimator_selects_then_refits_zoo_ridge() -> None:
    df = _cells(2, offset_sd=0.3)
    X = _with_groups(df, ["signal", "nuisance"])
    model = NestedSelectionRidge().fit(X, df["y"])
    assert model.selected_features_ == ["signal"]
    assert list(model.feature_names_in_) == ["signal", "nuisance"]
    reference = make_model("ridge", alpha=model.alpha_).fit(df[["signal"]], df["y"])
    expected = reference.predict(df[["signal"]])
    np.testing.assert_allclose(model.predict(X), expected, rtol=1e-12)
    # The group column and column order do not matter at prediction time.
    np.testing.assert_allclose(model.predict(df[["nuisance", "signal"]]), expected, rtol=1e-12)
    assert clone(model).get_params()["alphas"] == DEFAULT_ALPHAS


def test_estimator_input_validation() -> None:
    df = _cells(2, offset_sd=0.3)
    with pytest.raises(ValueError, match="group column"):
        NestedSelectionRidge().fit(df[["signal", "nuisance"]], df["y"])
    with pytest.raises(ValueError, match="DataFrame"):
        NestedSelectionRidge().fit(df[["signal"]].to_numpy(), df["y"])  # type: ignore[arg-type]
    model = NestedSelectionRidge().fit(_with_groups(df, ["signal", "nuisance"]), df["y"])
    with pytest.raises(ValueError, match="selected feature"):
        model.predict(df[["nuisance"]])


def test_nested_selection_beats_plain_ridge_out_of_group_when_offset_is_large() -> None:
    # Strong offsets: plain ridge leans on the precise-but-offset feature and pays for it on
    # unseen groups; the nested procedure (selection inside each outer training fold) does not.
    df = _cells(6, offset_sd=0.3)
    columns = ["signal", "nuisance"]
    X = _with_groups(df, columns)
    plain, nested = [], []
    for train, test, _ in leave_one_group_out(df["group"]):
        y_train, y_test = df["y"].iloc[train], df["y"].iloc[test].to_numpy()
        ridge = make_model("ridge").fit(df[columns].iloc[train], y_train)
        plain.append(np.mean(np.abs(ridge.predict(df[columns].iloc[test]) - y_test)))
        nfs = NestedSelectionRidge().fit(X.iloc[train], y_train)
        assert "nuisance" not in nfs.selected_features_
        nested.append(np.mean(np.abs(nfs.predict(X.iloc[test]) - y_test)))
    assert np.mean(nested) < np.mean(plain)


def test_works_as_conformal_fold_model() -> None:
    df = _cells(3, offset_sd=0.3)
    X = _with_groups(df, ["signal", "nuisance"])
    gcr = GroupConformalRegressor(partial(NestedSelectionRidge), alphas=(0.1,), method="cv_plus")
    gcr.fit(X, df["y"], df["group"])
    # Every fold model ran its own selection on its own training groups only.
    assert {m.selection_.n_groups for m in gcr.fold_estimators_.values()} == {4}
    assert gcr.estimator_.selection_.n_groups == 5
    yhat, lo, hi = gcr.predict_interval(X.iloc[:10], 0.1)
    assert np.all(np.isfinite(lo))
    assert np.all(np.isfinite(hi))
    assert np.all(lo < hi)
    assert yhat.shape == (10,)
