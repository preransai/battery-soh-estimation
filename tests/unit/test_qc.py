"""Tests for the per-cycle capacity quality control (Hampel-style filter)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from battery_soh.qc import (
    QC_MAX_DEV_AH,
    QC_MIN_AH,
    QC_REASONS,
    QC_WINDOW,
    flag_capacity_outliers,
    qc_summary,
)


def _fade(n: int, start: float = 1.85, step: float = 0.004) -> np.ndarray:
    return start - step * np.arange(n)


def test_defaults_match_the_spec() -> None:
    assert (QC_MIN_AH, QC_WINDOW, QC_MAX_DEV_AH) == (0.3, 9, 0.15)
    assert set(QC_REASONS) == {"ok", "missing", "below_min", "isolated", "hampel_outlier"}


def test_smooth_fade_is_all_valid() -> None:
    valid, reason = flag_capacity_outliers(_fade(60))
    assert valid.all()
    assert (reason == "ok").all()
    assert valid.name == "qc_valid"
    assert reason.name == "qc_reason"
    assert valid.dtype == bool


def test_empty_input() -> None:
    valid, reason = flag_capacity_outliers(pd.Series([], dtype=float))
    assert len(valid) == 0
    assert len(reason) == 0
    assert valid.dtype == bool
    assert qc_summary(reason) == dict.fromkeys(QC_REASONS, 0)


def test_all_nan_is_missing() -> None:
    valid, reason = flag_capacity_outliers([np.nan, None, float("inf"), "n/a"])
    assert not valid.any()
    assert reason.tolist() == ["missing"] * 4


def test_below_min_is_inclusive_at_threshold() -> None:
    caps = _fade(20)
    caps[5], caps[6], caps[7] = 0.0, 0.2999, 0.3
    valid, reason = flag_capacity_outliers(caps)
    assert reason[5] == "below_min"
    assert reason[6] == "below_min"
    # 0.3 Ah passes the plausibility floor, then fails the neighbourhood check.
    assert reason[7] == "hampel_outlier"
    assert valid.sum() == 17


def test_hampel_flags_isolated_spike_and_drop() -> None:
    caps = _fade(40)
    caps[10] += 0.6  # spike (B0036 has a 2.44 Ah reading among 1.68 Ah neighbours)
    caps[25] -= 0.4  # drop that is still above min_ah
    valid, reason = flag_capacity_outliers(caps)
    assert reason[10] == "hampel_outlier"
    assert reason[25] == "hampel_outlier"
    assert valid.sum() == 38


def test_hampel_keeps_capacity_recovery_after_rest() -> None:
    caps = _fade(40, step=0.005)
    caps[20:] += 0.09  # B0005 recovers ~0.09 Ah after a rest period
    valid, _ = flag_capacity_outliers(caps)
    assert valid.all()


def test_threshold_is_strictly_greater_than_max_dev() -> None:
    caps = np.full(21, 1.5)
    caps[10] = 1.5 + 0.149
    caps[15] = 1.5 + 0.151
    _, reason = flag_capacity_outliers(caps)
    assert reason[10] == "ok"
    assert reason[15] == "hampel_outlier"


def test_anomalous_first_cycles_are_flagged() -> None:
    # B0036/B0054-B0056 start with a formation discharge far below the real capacity.
    caps = _fade(30, start=1.80, step=0.002)
    caps[0] = 1.00
    valid, reason = flag_capacity_outliers(caps)
    assert reason[0] == "hampel_outlier"
    assert valid[1:].all()


def test_two_anomalous_first_cycles_are_flagged() -> None:
    caps = _fade(30, start=1.80, step=0.002)
    caps[0], caps[1] = 0.74, 0.95
    _, reason = flag_capacity_outliers(caps)
    assert reason[:2].tolist() == ["hampel_outlier", "hampel_outlier"]
    assert (reason[2:] == "ok").all()


def test_zero_capacity_runs_do_not_poison_neighbours() -> None:
    # B0046 has isolated 0.00 Ah records; they must not drag the local median down.
    caps = _fade(30, start=1.5, step=0.005)
    caps[[10, 11, 12]] = 0.0
    valid, reason = flag_capacity_outliers(caps)
    assert reason[10:13].tolist() == ["below_min"] * 3
    assert valid.sum() == 27


def test_isolated_value_cannot_be_checked() -> None:
    caps = [np.nan] * 6 + [1.4] + [np.nan] * 6  # one plausible value in a sea of gaps
    valid, reason = flag_capacity_outliers(caps)
    assert reason[6] == "isolated"
    assert not valid.any()


def test_short_series_uses_min_periods() -> None:
    valid, reason = flag_capacity_outliers([1.8, 1.79])
    assert reason.tolist() == ["isolated", "isolated"]  # 2 < min_periods=3
    valid, reason = flag_capacity_outliers([1.8, 1.79], min_periods=2)
    assert valid.all()


def test_index_is_preserved_and_mixed_types_coerced() -> None:
    caps = pd.Series(["1.80", 1.79, None, 1.78, 1.77], index=[10, 11, 12, 13, 14], dtype=object)
    valid, reason = flag_capacity_outliers(caps)
    assert valid.index.tolist() == [10, 11, 12, 13, 14]
    assert reason.index.equals(caps.index)
    assert reason.tolist() == ["ok", "ok", "missing", "ok", "ok"]


def test_accepts_list_and_ndarray_identically() -> None:
    caps = _fade(15)
    caps[7] = 2.4
    v1, r1 = flag_capacity_outliers(caps)
    v2, r2 = flag_capacity_outliers(caps.tolist())
    pd.testing.assert_series_equal(v1, v2)
    pd.testing.assert_series_equal(r1, r2)


def test_custom_parameters() -> None:
    caps = _fade(20)
    caps[10] -= 0.1
    assert flag_capacity_outliers(caps)[0].all()
    assert not flag_capacity_outliers(caps, max_dev_ah=0.05)[0][10]
    assert flag_capacity_outliers(caps, min_ah=1.9)[1].eq("below_min").all()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"min_ah": -0.1}, "min_ah"),
        ({"min_ah": float("nan")}, "min_ah"),
        ({"window": 8}, "window"),
        ({"window": 0}, "window"),
        ({"window": True}, "window"),
        ({"window": 9.0}, "window"),
        ({"max_dev_ah": 0.0}, "max_dev_ah"),
        ({"max_dev_ah": float("inf")}, "max_dev_ah"),
        ({"min_periods": 0}, "min_periods"),
        ({"min_periods": 10}, "min_periods"),
    ],
)
def test_invalid_parameters(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        flag_capacity_outliers([1.8, 1.7], **kwargs)


def test_qc_summary_counts_every_code() -> None:
    reason = pd.Series(["ok", "ok", "missing", "below_min", "hampel_outlier"])
    assert qc_summary(reason) == {
        "missing": 1,
        "below_min": 1,
        "isolated": 0,
        "hampel_outlier": 1,
        "ok": 2,
    }


@settings(max_examples=100, deadline=None)
@given(
    st.lists(
        st.one_of(
            st.floats(min_value=-1.0, max_value=3.0),
            st.just(float("nan")),
            st.just(float("inf")),
        ),
        max_size=60,
    )
)
def test_invariants(values: list[float]) -> None:
    valid, reason = flag_capacity_outliers(values)
    arr = np.asarray(values, dtype=float)
    assert len(valid) == len(reason) == len(values)
    assert set(reason) <= set(QC_REASONS)
    assert (valid == (reason == "ok")).all()
    flags = valid.to_numpy()
    finite = np.isfinite(arr)
    assert not flags[~finite].any()
    assert not flags[finite & (np.nan_to_num(arr) < QC_MIN_AH)].any()
    # Deterministic: a second call gives the same answer.
    pd.testing.assert_series_equal(reason, flag_capacity_outliers(values)[1])
