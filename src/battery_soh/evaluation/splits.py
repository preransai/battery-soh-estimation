"""Battery-grouped data splits and train/test disjointness guards.

Cycles of the same cell are strongly autocorrelated: a model that has seen
cycle 120 of a battery can "predict" cycle 121 almost perfectly. Every split
in this package therefore holds out whole batteries (leave-one-group-out),
and :func:`assert_disjoint` turns any accidental overlap between training and
evaluation rows into a hard :class:`~battery_soh.evaluation.errors.LeakageError`.
"""

from __future__ import annotations

import logging
from collections.abc import Hashable, Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from battery_soh.evaluation.errors import LeakageError

__all__ = [
    "DEFAULT_KEY_COLUMNS",
    "LeakageError",
    "assert_disjoint",
    "assert_groups_disjoint",
    "leave_one_group_out",
    "row_keys",
    "temporal_holdout",
]

logger = logging.getLogger(__name__)

#: Columns that identify one row of the feature table (one discharge cycle).
DEFAULT_KEY_COLUMNS: tuple[str, str] = ("battery_id", "cycle_index")

_MAX_EXAMPLES = 5


def _as_group_array(groups: pd.Series | ArrayLike) -> NDArray[np.object_]:
    """Convert group labels to a 1-D object array and validate them."""
    values = groups.to_numpy() if isinstance(groups, pd.Series) else np.asarray(groups)
    if values.ndim != 1:
        raise ValueError(f"groups must be one-dimensional, got shape {values.shape}")
    if values.size == 0:
        raise ValueError("groups is empty")
    if bool(pd.isna(values).any()):
        raise ValueError("groups contains missing values; every row needs a battery id")
    return values.astype(object)


def leave_one_group_out(
    groups: pd.Series | ArrayLike,
) -> Iterator[tuple[NDArray[np.intp], NDArray[np.intp], str]]:
    """Yield leave-one-group-out splits (one fold per battery).

    Folds are produced in sorted order of the group labels, so results are
    deterministic regardless of row order. Indices are *positional* (suitable
    for ``numpy`` indexing and ``DataFrame.iloc``), not index labels.

    Args:
        groups: Group label (battery id) of every row.

    Yields:
        ``(train_idx, test_idx, held_out)`` where ``held_out`` is the string
        form of the held-out group label.

    Raises:
        ValueError: If ``groups`` is empty, not 1-D, has missing labels or has
            fewer than two distinct groups.
    """
    values = _as_group_array(groups)
    labels = sorted(pd.unique(values), key=str)
    if len(labels) < 2:
        raise ValueError(f"leave-one-group-out needs at least 2 distinct groups, got {len(labels)}")
    positions = np.arange(values.size, dtype=np.intp)
    for label in labels:
        test_mask = values == label
        yield positions[~test_mask], positions[test_mask], str(label)


def row_keys(
    df: pd.DataFrame, key_columns: Sequence[str] = DEFAULT_KEY_COLUMNS
) -> list[tuple[Hashable, ...]]:
    """Return one hashable key per row, e.g. ``("B0005", 17)``.

    Args:
        df: Table containing ``key_columns``.
        key_columns: Columns that jointly identify a row.

    Raises:
        ValueError: If a key column is missing.
    """
    missing = [c for c in key_columns if c not in df.columns]
    if missing:
        raise ValueError(f"key columns {missing} not found in table")
    return [tuple(row) for row in df.loc[:, list(key_columns)].itertuples(index=False)]


def assert_disjoint(train_keys: Iterable[Hashable], test_keys: Iterable[Hashable]) -> None:
    """Raise if any evaluation row also appears in the training rows.

    The v1 project scored its "test" set on rows the model had been trained on.
    Calling this before every evaluation makes that class of bug impossible.

    Args:
        train_keys: Keys of the training rows, typically ``(battery_id, cycle_index)``
            tuples from :func:`row_keys`.
        test_keys: Keys of the evaluation rows.

    Raises:
        LeakageError: If the two key sets intersect. The message reports how
            many keys overlap and a few examples.
    """
    train = set(train_keys)
    overlap = [key for key in dict.fromkeys(test_keys) if key in train]
    if overlap:
        examples = ", ".join(repr(k) for k in overlap[:_MAX_EXAMPLES])
        raise LeakageError(
            f"{len(overlap)} evaluation row(s) also appear in the training data (e.g. {examples})"
        )


def assert_groups_disjoint(
    train_groups: pd.Series | ArrayLike, test_groups: pd.Series | ArrayLike
) -> None:
    """Raise if any battery contributes rows to both training and evaluation.

    Row-level disjointness (:func:`assert_disjoint`) is necessary but not
    sufficient for cross-battery claims: two different cycles of one cell are
    distinct rows yet nearly identical. Use this guard whenever a result is
    reported as "on unseen batteries".

    Raises:
        LeakageError: If a group label occurs on both sides.
    """
    shared = sorted(set(_as_group_array(train_groups)) & set(_as_group_array(test_groups)), key=str)
    if shared:
        raise LeakageError(f"group(s) {shared} appear in both training and evaluation data")


def temporal_holdout(
    groups: pd.Series | ArrayLike,
    order: pd.Series | ArrayLike,
    *,
    test_fraction: float = 0.15,
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Within-group chronological split: the last ``test_fraction`` of each group is test.

    This is the honest version of a *within-battery* evaluation (it forecasts
    later cycles of cells whose early life was seen). It answers a different
    question than leave-one-battery-out and must never be reported as
    cross-battery performance. The leakage audit uses it to show how much of
    v1's apparent accuracy came from evaluating on seen batteries.

    Args:
        groups: Group label (battery id) of every row.
        order: Sort key within each group (e.g. ``cycle_index``).
        test_fraction: Fraction of each group's rows (rounded up, at least one)
            assigned to the test side.

    Returns:
        ``(train_idx, test_idx)`` positional indices, each sorted ascending.

    Raises:
        ValueError: If ``test_fraction`` is not in (0, 1) or lengths differ.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")
    values = _as_group_array(groups)
    keys = order.to_numpy() if isinstance(order, pd.Series) else np.asarray(order)
    if keys.shape != values.shape:
        raise ValueError(f"order has shape {keys.shape}, groups has shape {values.shape}")
    test_parts: list[NDArray[np.intp]] = []
    positions = np.arange(values.size, dtype=np.intp)
    for label in pd.unique(values):
        members = positions[values == label]
        members = members[np.argsort(keys[members], kind="stable")]
        n_test = max(1, int(np.ceil(test_fraction * members.size)))
        if n_test >= members.size:
            logger.debug("group %s has %d rows; all assigned to test", label, members.size)
        test_parts.append(members[members.size - n_test :])
    test_idx = np.sort(np.concatenate(test_parts)).astype(np.intp)
    train_idx = np.setdiff1d(positions, test_idx).astype(np.intp)
    return train_idx, test_idx
