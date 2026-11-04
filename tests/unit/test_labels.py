"""Tests for SOH label construction (reference capacity, own-C0 and rated SOH)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from battery_soh.config import RATED_CAPACITY_AH
from battery_soh.labels import (
    EOL_SOH_PCT,
    LABEL_COLUMNS,
    label_battery,
    reference_capacity,
    soh_from_capacity,
)


def _series(values: list[float | None]) -> pd.Series:
    return pd.Series(values, dtype=object)


# ------------------------------------------------------------------------ reference_capacity


def test_reference_is_median_of_first_n_valid() -> None:
    caps = pd.Series([1.90, 1.80, 1.70, 1.84, 1.86, 1.82, 1.60, 1.50])
    valid = pd.Series([True, True, False, True, True, True, True, True])
    # First five valid: 1.90, 1.80, 1.84, 1.86, 1.82 -> median 1.84 (1.70 is skipped).
    assert reference_capacity(caps, valid) == pytest.approx(1.84)
    assert reference_capacity(caps, valid, n_ref=1) == pytest.approx(1.90)
    assert reference_capacity(caps, valid, n_ref=2) == pytest.approx(1.85)


def test_reference_skips_non_finite_even_if_flagged_valid() -> None:
    caps = _series([None, np.nan, 1.8, 1.7, 1.6])
    valid = pd.Series([True] * 5)
    assert reference_capacity(caps, valid, n_ref=2) == pytest.approx(1.75)


def test_reference_with_fewer_valid_than_n_ref() -> None:
    caps = pd.Series([1.8, 1.6])
    assert reference_capacity(caps, pd.Series([True, True]), n_ref=5) == pytest.approx(1.7)


def test_reference_is_nan_without_valid_cycles() -> None:
    caps = pd.Series([1.8, 1.7])
    assert np.isnan(reference_capacity(caps, pd.Series([False, False])))
    assert np.isnan(reference_capacity(pd.Series([], dtype=float), pd.Series([], dtype=bool)))


def test_reference_treats_missing_flags_as_invalid() -> None:
    caps = pd.Series([1.0, 1.8, 1.7])
    valid = pd.Series([None, True, True], dtype=object)
    assert reference_capacity(caps, valid) == pytest.approx(1.75)


@pytest.mark.parametrize("n_ref", [0, -1, 2.5, True])
def test_reference_rejects_bad_n_ref(n_ref: object) -> None:
    with pytest.raises(ValueError, match="n_ref"):
        reference_capacity(pd.Series([1.8]), pd.Series([True]), n_ref=n_ref)  # type: ignore[arg-type]


def test_reference_rejects_misaligned_inputs() -> None:
    with pytest.raises(ValueError, match="same length and index"):
        reference_capacity(pd.Series([1.8, 1.7]), pd.Series([True]))
    with pytest.raises(ValueError, match="same length and index"):
        reference_capacity(pd.Series([1.8], index=[3]), pd.Series([True], index=[4]))


# ------------------------------------------------------------------------- soh_from_capacity


def test_soh_from_capacity_percent_and_index() -> None:
    caps = pd.Series([2.0, 1.5, np.nan, 1.0], index=[7, 8, 9, 10])
    soh = soh_from_capacity(caps, 2.0)
    assert soh.name == "soh_pct"
    assert soh.index.tolist() == [7, 8, 9, 10]
    np.testing.assert_allclose(soh.to_numpy(), [100.0, 75.0, np.nan, 50.0])


def test_soh_from_capacity_accepts_sequences_and_none() -> None:
    soh = soh_from_capacity([1.8, None, float("inf")], 1.8)
    assert soh.iloc[0] == pytest.approx(100.0)
    assert soh.iloc[1:].isna().all()


def test_soh_with_undefined_reference_is_nan() -> None:
    soh = soh_from_capacity(pd.Series([1.8, 1.7]), float("nan"))
    assert soh.isna().all()
    assert soh.dtype == np.float64


@pytest.mark.parametrize("c0", [0.0, -1.0, float("inf")])
def test_soh_rejects_bad_reference(c0: float) -> None:
    with pytest.raises(ValueError, match="reference capacity"):
        soh_from_capacity(pd.Series([1.8]), c0)


# ------------------------------------------------------------------------------ label_battery


def test_label_battery_columns_and_values() -> None:
    caps = 1.85 - 0.005 * np.arange(30)
    labels = label_battery(caps)
    assert tuple(labels.columns) == LABEL_COLUMNS
    c0 = float(np.median(caps[:5]))
    assert (labels["c0_ah"] == c0).all()
    np.testing.assert_allclose(labels["soh_pct"], 100.0 * caps / c0)
    np.testing.assert_allclose(labels["soh_rated_pct"], 100.0 * caps / RATED_CAPACITY_AH)
    assert labels["qc_valid"].all()
    assert labels["soh_pct"].iloc[:5].median() == pytest.approx(100.0)


def test_label_battery_anomalous_first_cycle_does_not_set_c0() -> None:
    # Formation discharge (B0036: 1.00 Ah before ~1.80 Ah) must not define C0.
    caps = np.r_[1.00, 1.80 - 0.002 * np.arange(40)]
    labels = label_battery(caps)
    assert not labels["qc_valid"].iloc[0]
    assert labels["qc_reason"].iloc[0] == "hampel_outlier"
    assert labels["c0_ah"].iloc[0] == pytest.approx(np.median(caps[1:6]))
    # Labels are still reported for the rejected cycle; consumers filter on qc_valid.
    assert labels["soh_pct"].iloc[0] == pytest.approx(100.0 * 1.00 / labels["c0_ah"].iloc[0])


def test_label_battery_below_min_cycles() -> None:
    caps = 1.5 - 0.004 * np.arange(20)
    caps[3] = 0.0
    labels = label_battery(caps)
    assert labels["qc_reason"].iloc[3] == "below_min"
    assert labels["c0_ah"].iloc[0] == pytest.approx(np.median(caps[[0, 1, 2, 4, 5]]))


def test_label_battery_empty() -> None:
    labels = label_battery(pd.Series([], dtype=object))
    assert tuple(labels.columns) == LABEL_COLUMNS
    assert labels.empty


def test_label_battery_all_missing() -> None:
    labels = label_battery([None, np.nan, None])
    assert labels["c0_ah"].isna().all()
    assert labels["soh_pct"].isna().all()
    assert labels["soh_rated_pct"].isna().all()
    assert not labels["qc_valid"].any()
    assert (labels["qc_reason"] == "missing").all()


def test_label_battery_preserves_index() -> None:
    caps = pd.Series(1.8 - 0.01 * np.arange(10), index=range(100, 110))
    labels = label_battery(caps)
    assert labels.index.tolist() == list(range(100, 110))


def test_label_battery_parameters_are_forwarded() -> None:
    caps = 1.8 - 0.01 * np.arange(10)
    labels = label_battery(caps, n_ref=1, rated_capacity_ah=1.0)
    assert labels["c0_ah"].iloc[0] == pytest.approx(1.8)
    np.testing.assert_allclose(labels["soh_rated_pct"], 100.0 * caps)
    with pytest.raises(ValueError, match="window"):
        label_battery(caps, window=4)


def test_eol_threshold_constant() -> None:
    assert EOL_SOH_PCT == 80.0
