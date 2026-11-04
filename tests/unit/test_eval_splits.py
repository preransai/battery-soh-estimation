"""Tests for battery-grouped splits and the train/test disjointness guards."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from battery_soh.evaluation.errors import LeakageError
from battery_soh.evaluation.splits import (
    assert_disjoint,
    assert_groups_disjoint,
    leave_one_group_out,
    row_keys,
    temporal_holdout,
)


@pytest.fixture
def table() -> pd.DataFrame:
    # Deliberately shuffled rows and a non-default index: indices must be positional.
    rows = [(b, c) for b in ("B0018", "B0005", "B0007") for c in range(6)]
    df = pd.DataFrame(rows, columns=["battery_id", "cycle_index"])
    df = df.sample(frac=1.0, random_state=0)
    df.index = np.arange(1000, 1000 + len(df))
    return df


def test_lobo_holds_out_each_battery_once(table: pd.DataFrame) -> None:
    folds = list(leave_one_group_out(table["battery_id"]))
    assert [held for _, _, held in folds] == ["B0005", "B0007", "B0018"]
    seen = []
    for train_idx, test_idx, held in folds:
        assert train_idx.dtype == np.intp
        assert test_idx.dtype == np.intp
        assert set(table["battery_id"].iloc[test_idx]) == {held}
        assert held not in set(table["battery_id"].iloc[train_idx])
        assert np.intersect1d(train_idx, test_idx).size == 0
        assert np.union1d(train_idx, test_idx).size == len(table)
        assert_disjoint(row_keys(table.iloc[train_idx]), row_keys(table.iloc[test_idx]))
        assert_groups_disjoint(
            table["battery_id"].iloc[train_idx], table["battery_id"].iloc[test_idx]
        )
        seen.extend(test_idx.tolist())
    assert sorted(seen) == list(range(len(table)))


def test_lobo_accepts_arrays_and_validates() -> None:
    folds = list(leave_one_group_out(np.array([2, 1, 2, 1])))
    assert [held for *_, held in folds] == ["1", "2"]
    with pytest.raises(ValueError, match="at least 2"):
        list(leave_one_group_out(["a", "a"]))
    with pytest.raises(ValueError, match="missing"):
        list(leave_one_group_out(pd.Series(["a", None, "b"])))
    with pytest.raises(ValueError, match="empty"):
        list(leave_one_group_out([]))
    with pytest.raises(ValueError, match="one-dimensional"):
        list(leave_one_group_out(np.array([["a", "b"]])))


def test_assert_disjoint_raises_on_overlap(table: pd.DataFrame) -> None:
    keys = row_keys(table)
    assert keys[0] == (table["battery_id"].iloc[0], table["cycle_index"].iloc[0])
    assert_disjoint(keys[:5], keys[5:])
    with pytest.raises(LeakageError, match=r"2 evaluation row\(s\)"):
        assert_disjoint(keys[:5], [*keys[3:5], *keys[10:]])
    # v1's failure mode: "test" rows that the model was trained on.
    with pytest.raises(LeakageError):
        assert_disjoint([("B0006", 1), ("B0006", 2)], [("B0006", 2), ("B0007", 0)])


def test_assert_groups_disjoint() -> None:
    assert_groups_disjoint(["B0005", "B0006"], ["B0007"])
    with pytest.raises(LeakageError, match="B0006"):
        assert_groups_disjoint(["B0005", "B0006"], ["B0006", "B0007"])


def test_row_keys_requires_columns(table: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="key columns"):
        row_keys(table, ("battery_id", "missing"))


def test_temporal_holdout_takes_the_last_cycles_of_each_battery(table: pd.DataFrame) -> None:
    train_idx, test_idx = temporal_holdout(
        table["battery_id"], table["cycle_index"], test_fraction=0.3
    )
    test = table.iloc[test_idx]
    # ceil(0.3 * 6) = 2 last cycles per battery.
    assert sorted(test.groupby("battery_id")["cycle_index"].apply(sorted).tolist()) == [
        [4, 5],
        [4, 5],
        [4, 5],
    ]
    assert np.intersect1d(train_idx, test_idx).size == 0
    assert len(train_idx) + len(test_idx) == len(table)
    with pytest.raises(ValueError, match="test_fraction"):
        temporal_holdout(table["battery_id"], table["cycle_index"], test_fraction=1.0)
    with pytest.raises(ValueError, match="shape"):
        temporal_holdout(table["battery_id"], table["cycle_index"].iloc[:-1])
