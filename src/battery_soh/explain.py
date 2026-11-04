"""Battery-grouped permutation importance for any regressor.

Linear bundles have exact attributions (:mod:`battery_soh.models.linear_bundle`);
non-linear models get *permutation importance*: how much does the error on a
held-out battery grow when one feature's values are shuffled within that
battery? Computing it under leave-one-battery-out (instead of on training
data) measures what a feature contributes to generalisation to a new cell,
and the spread across held-out batteries shows how stable that is.

Correlated health indicators share credit (shuffling one leaves its twin
intact), which understates both. Pass ``feature_groups`` to permute related
columns jointly - e.g. all charge-side indicators - and measure the group.
"""

from __future__ import annotations

import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from battery_soh._typing import FloatArray
from battery_soh.config import RANDOM_SEED
from battery_soh.evaluation.splits import leave_one_group_out
from battery_soh.uncertainty.conformal import Regressor

__all__ = ["PermutationImportance", "grouped_permutation_importance"]

Metric = Literal["mae", "rmse"]


@dataclass(frozen=True)
class PermutationImportance:
    """Result of :func:`grouped_permutation_importance`.

    Attributes:
        summary: One row per feature (or feature group), sorted by decreasing
            importance: ``feature, importance_mean, importance_std, importance_min,
            importance_max, n_groups, rank``. Importance is the increase of the
            error metric (label units) when the feature is permuted, averaged
            over held-out batteries (each battery weighs the same).
        per_group: One row per (held-out battery, feature): ``group, feature,
            baseline, importance_mean, importance_std`` (std over repeats).
        metric: Error metric used.
    """

    summary: pd.DataFrame
    per_group: pd.DataFrame
    metric: Metric


def _stream_id(text: str) -> int:
    """Stable (process-independent) integer id of a label, for seeding."""
    return zlib.crc32(text.encode("utf-8"))


def _score(metric: Metric, y: FloatArray, pred: FloatArray) -> float:
    err = pred - y
    if metric == "mae":
        return float(np.mean(np.abs(err)))
    return float(np.sqrt(np.mean(err**2)))


def grouped_permutation_importance(
    estimator_factory: Callable[[], Regressor],
    X: pd.DataFrame,
    y: ArrayLike,
    groups: pd.Series | ArrayLike,
    *,
    feature_groups: Mapping[str, Sequence[str]] | None = None,
    n_repeats: int = 10,
    metric: Metric = "mae",
    random_state: int = RANDOM_SEED,
) -> PermutationImportance:
    """Leave-one-battery-out permutation importance.

    For each held-out battery: fit a fresh model on the other batteries, score
    it on the held-out battery, then re-score after permuting each feature (or
    feature group, jointly) among the held-out rows, ``n_repeats`` times.

    Args:
        estimator_factory: Zero-argument callable returning an unfitted regressor.
        X: Feature table (columns are the model inputs, in order).
        y: Labels (finite), aligned with ``X``.
        groups: Battery id per row.
        feature_groups: ``{name: [columns]}`` to permute together; defaults to
            one group per column.
        n_repeats: Permutations per feature and battery.
        metric: ``"mae"`` or ``"rmse"``.
        random_state: Seed; each (held-out battery, feature) pair gets its own
            random stream derived from their names, so results do not depend
            on the order of features or batteries.

    Returns:
        A :class:`PermutationImportance`.

    Raises:
        ValueError: On misaligned inputs, unknown columns or bad settings.
    """
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a pandas DataFrame (column names identify features)")
    if n_repeats < 1:
        raise ValueError(f"n_repeats must be >= 1, got {n_repeats}")
    if metric not in ("mae", "rmse"):
        raise ValueError(f"metric must be 'mae' or 'rmse', got {metric!r}")
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    group_arr = groups.to_numpy() if isinstance(groups, pd.Series) else np.asarray(groups)
    if not (len(X) == y_arr.size == group_arr.size):
        raise ValueError(
            f"X, y and groups must have equal length, got {len(X)}, {y_arr.size}, {group_arr.size}"
        )
    if not np.all(np.isfinite(y_arr)):
        raise ValueError("y must be finite")
    if not X.columns.is_unique:
        raise ValueError("X must have unique column names")
    blocks = dict(feature_groups) if feature_groups else {str(c): [c] for c in X.columns}
    empty = sorted(name for name, cols in blocks.items() if not list(cols))
    if empty:
        raise ValueError(f"feature_groups {empty} list no columns")
    unknown = sorted({c for cols in blocks.values() for c in cols} - set(X.columns))
    if unknown:
        raise ValueError(f"feature_groups reference unknown columns {unknown}")
    positions = {name: [X.columns.get_loc(c) for c in cols] for name, cols in blocks.items()}

    records: list[dict[str, object]] = []
    for train_idx, test_idx, held_out in leave_one_group_out(group_arr):
        model = estimator_factory()
        model.fit(X.iloc[train_idx], y_arr[train_idx])
        X_test = X.iloc[test_idx].reset_index(drop=True)
        y_test = y_arr[test_idx]
        baseline = _score(metric, y_test, np.asarray(model.predict(X_test), float).reshape(-1))
        for name, cols in positions.items():
            rng = np.random.default_rng([random_state, _stream_id(held_out), _stream_id(name)])
            deltas = np.empty(n_repeats)
            for r in range(n_repeats):
                shuffled = X_test.copy()
                order = rng.permutation(len(X_test))
                shuffled.iloc[:, cols] = X_test.iloc[order, cols].to_numpy()
                pred = np.asarray(model.predict(shuffled), dtype=np.float64).reshape(-1)
                deltas[r] = _score(metric, y_test, pred) - baseline
            records.append(
                {
                    "group": held_out,
                    "feature": name,
                    "baseline": baseline,
                    "importance_mean": float(deltas.mean()),
                    "importance_std": float(deltas.std(ddof=1)) if n_repeats > 1 else 0.0,
                }
            )

    per_group = pd.DataFrame(records)
    summary = (
        per_group.groupby("feature", sort=False)["importance_mean"]
        .agg(
            importance_mean="mean",
            importance_std="std",
            importance_min="min",
            importance_max="max",
            n_groups="count",
        )
        .reset_index()
        .sort_values(["importance_mean", "feature"], ascending=[False, True], kind="stable")
        .reset_index(drop=True)
    )
    summary["rank"] = np.arange(1, len(summary) + 1)
    return PermutationImportance(summary=summary, per_group=per_group, metric=metric)
