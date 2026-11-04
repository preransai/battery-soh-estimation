"""Low-level signal helpers shared by the extractors."""

from __future__ import annotations

import numpy as np
import pytest

from battery_soh.data.schema import Segment
from battery_soh.features._signal import (
    cumulative_charge_ah,
    directed_current,
    finite_or_nan,
    first_crossing,
    integrate_between,
    interp_at,
    reference_current,
    require_kind,
    sanitize,
)


def _segment(t: list[float], v: list[float], i: list[float], temp: list[float]) -> Segment:
    return Segment("discharge", np.array(t), np.array(v), np.array(i), np.array(temp))


def test_sanitize_sorts_dedupes_and_drops_non_finite_rows() -> None:
    seg = _segment(
        t=[3.0, 1.0, 2.0, 2.0, np.nan, 4.0],
        v=[3.3, 3.1, 3.2, 9.9, 3.0, np.inf],
        i=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        temp=[25.0, 23.0, np.nan, 99.0, 20.0, 26.0],
    )
    s = sanitize(seg)
    np.testing.assert_array_equal(s.t, [1.0, 2.0, 3.0])
    # The first recorded sample wins for a repeated timestamp (stable sort).
    np.testing.assert_array_equal(s.v, [3.1, 3.2, 3.3])
    # A NaN temperature does not drop the row (only temperature features suffer).
    assert np.isnan(s.temp[1])
    assert s.n == 3


def test_first_crossing_interpolates_between_samples() -> None:
    t = np.array([0.0, 10.0, 20.0, 30.0])
    x = np.array([4.0, 3.9, 3.7, 3.6])
    c = first_crossing(t, x, 3.8, rising=False)
    assert c is not None
    assert c.index == 2
    assert c.time == pytest.approx(15.0)


def test_first_crossing_requires_an_observed_crossing() -> None:
    t = np.arange(5.0)
    x = np.array([3.5, 3.6, 3.7, 3.8, 3.9])
    assert first_crossing(t, x, 3.4, rising=False) is None  # starts beyond the level
    assert first_crossing(t, x, 4.5, rising=True) is None  # never reached
    assert first_crossing(t, x, 3.75, rising=True, stop=3) is None  # outside the range
    c = first_crossing(t, x, 3.75, rising=True, start=1)
    assert c is not None
    assert c.time == pytest.approx(2.5)


def test_integrate_between_is_exact_for_piecewise_linear_signals() -> None:
    t = np.array([0.0, 10.0, 20.0])
    y = np.array([0.0, 10.0, 0.0])
    assert integrate_between(t, y, 5.0, 15.0) == pytest.approx(75.0)
    assert integrate_between(t, y, 0.0, 20.0) == pytest.approx(100.0)
    assert np.isnan(integrate_between(t, y, -1.0, 5.0))


def test_cumulative_charge_in_ampere_hours() -> None:
    t = np.linspace(0.0, 1000.0, 11)
    q = cumulative_charge_ah(t, np.full(11, 3.6))
    assert q[0] == 0.0
    assert q[-1] == pytest.approx(1.0)
    assert cumulative_charge_ah(np.empty(0), np.empty(0)).size == 0


def test_directed_current_orients_the_dominant_direction_positive() -> None:
    """A short opposite-sign glitch (NASA charge records start with -3.4 A) stays negative."""
    seg = _segment([0, 1, 2, 3, 4, 5], [4] * 6, [0.0, 3.4, -2.0, -2.0, -2.0, -2.0], [25] * 6)
    s = sanitize(seg)
    np.testing.assert_array_equal(directed_current(s), [0.0, -3.4, 2.0, 2.0, 2.0, 2.0])


def test_reference_current_ignores_isolated_spikes() -> None:
    i = np.array([0.0, 9.0, 1.5, 1.5, 1.5, 1.5, 0.1])
    assert reference_current(i) == pytest.approx(1.5)
    assert np.isnan(reference_current(np.zeros(10)))
    assert np.isnan(reference_current(np.array([1.0, 1.0])))


def test_interp_at_and_finite_or_nan() -> None:
    t = np.array([0.0, 10.0])
    assert interp_at(t, np.array([1.0, 3.0]), 5.0) == pytest.approx(2.0)
    assert np.isnan(interp_at(t, np.array([1.0, 3.0]), 11.0))
    assert np.isnan(finite_or_nan(None))
    assert np.isnan(finite_or_nan(float("inf")))
    assert finite_or_nan(2) == 2.0


def test_require_kind() -> None:
    with pytest.raises(ValueError, match="charge"):
        require_kind(Segment("discharge"), "charge")
    with pytest.raises(ValueError, match="Segment"):
        require_kind("not a segment", "charge")  # type: ignore[arg-type]
