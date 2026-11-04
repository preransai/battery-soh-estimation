"""Incremental capacity analysis on a uniform voltage grid."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import savgol_filter
from sklearn.isotonic import IsotonicRegression

from battery_soh.features.ica import (
    charge_on_grid,
    ica_peak,
    incremental_capacity,
    isotonic_increasing,
    monotone_envelope,
    validate_ica_params,
    voltage_grid,
)
from battery_soh.synthetic import LogisticStep, OCVCurve

GRID = voltage_grid(3.90, 4.18, 0.005)

#: One broad, isolated transition at 4.02 V: Q(V) = C * SOC(V) is known exactly.
WIDE_PEAK = OCVCurve(v_min=3.5, v_max=4.4, background=0.2, steps=(LogisticStep(4.02, 0.05, 1.0),))
#: A NASA-like narrow transition (FWHM ~ 90 mV).
NARROW_PEAK = OCVCurve(
    v_min=3.5, v_max=4.4, background=0.2, steps=(LogisticStep(4.02, 0.025, 1.0),)
)

Samples = tuple[np.ndarray, np.ndarray]


def _cc_samples(curve: OCVCurve, capacity_ah: float, n: int = 3000) -> Samples:
    """Voltage and charge sampled uniformly in charge (as under constant current)."""
    v_fine = np.linspace(curve.v_min, curve.v_max, 400_001)  # inverse by dense interpolation
    soc = np.linspace(float(curve.soc(3.80)), float(curve.soc(4.25)), n)
    return np.interp(soc, curve.soc(v_fine), v_fine), capacity_ah * (soc - soc[0])


@pytest.fixture(scope="module")
def wide_peak_samples() -> Samples:
    return _cc_samples(WIDE_PEAK, 2.0)


# ----------------------------------------------------------------------------- grid & monotone


def test_voltage_grid_is_inclusive_and_uniform() -> None:
    assert GRID.size == 57
    assert GRID[0] == pytest.approx(3.90)
    assert GRID[-1] == pytest.approx(4.18)
    np.testing.assert_allclose(np.diff(GRID), 0.005)


@pytest.mark.parametrize(
    ("args", "match"),
    [
        ((4.0, 3.9, 0.005), "invalid voltage grid"),
        ((3.9, 4.18, 0.0), "invalid voltage grid"),
        ((3.9, 4.18, 0.006), "multiple of step"),
        ((3.9, float("inf"), 0.005), "finite"),
    ],
)
def test_invalid_voltage_grid_raises(args: tuple[float, float, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        voltage_grid(*args)


def test_isotonic_fit_matches_sklearn() -> None:
    rng = np.random.default_rng(0)
    y = np.linspace(3.8, 4.2, 400) + rng.normal(0, 0.004, 400)
    expected = IsotonicRegression().fit_transform(np.arange(y.size), y)
    np.testing.assert_allclose(isotonic_increasing(y), expected, atol=1e-12)
    assert np.all(np.diff(isotonic_increasing(y)) >= 0)


def test_isotonic_fit_edge_cases() -> None:
    np.testing.assert_array_equal(isotonic_increasing(np.array([1.0, 2.0, 3.0])), [1, 2, 3])
    np.testing.assert_allclose(isotonic_increasing(np.array([3.0, 2.0, 1.0])), [2, 2, 2])
    assert isotonic_increasing(np.empty(0)).size == 0


def test_monotone_envelope_methods() -> None:
    v = np.array([1.0, 3.0, 2.0, 4.0])
    np.testing.assert_array_equal(monotone_envelope(v, "cummax"), [1, 3, 3, 4])
    np.testing.assert_allclose(monotone_envelope(v, "isotonic"), [1, 2.5, 2.5, 4])
    with pytest.raises(ValueError, match="monotone"):
        monotone_envelope(v, "median")  # type: ignore[arg-type]


# ----------------------------------------------------------------------------- dQ/dV


def test_linear_charge_curve_gives_constant_dqdv() -> None:
    """Savitzky-Golay derivatives are exact for polynomials: Q = 4 * (V - 3.8)."""
    v = np.linspace(3.8, 4.3, 200)
    dqdv = incremental_capacity(v, 4.0 * (v - 3.8), GRID)
    np.testing.assert_allclose(dqdv, 4.0, rtol=1e-9)


def test_peak_of_a_known_curve_is_recovered(wide_peak_samples: Samples) -> None:
    v, q = wide_peak_samples
    height, v_peak = ica_peak(incremental_capacity(v, q, GRID), GRID)
    grid_fine = np.linspace(3.9, 4.18, 280_001)
    true_curve = 2.0 * WIDE_PEAK.dsoc_dv(grid_fine)
    assert v_peak == pytest.approx(grid_fine[np.argmax(true_curve)], abs=5e-4)
    assert height == pytest.approx(true_curve.max(), rel=0.01)


def test_cached_filter_matrix_equals_savitzky_golay(wide_peak_samples: Samples) -> None:
    v, q = wide_peak_samples
    q_grid = charge_on_grid(v, q, GRID)
    direct = savgol_filter(q_grid, 9, 2, deriv=1, delta=0.005, mode="interp")
    np.testing.assert_allclose(incremental_capacity(v, q, GRID), direct, rtol=1e-10)


def test_dqdv_is_scaled_by_capacity(wide_peak_samples: Samples) -> None:
    v, q = wide_peak_samples
    full = incremental_capacity(v, q, GRID)
    np.testing.assert_allclose(incremental_capacity(v, 0.7 * q, GRID), 0.7 * full, rtol=1e-12)


def test_peak_is_robust_to_voltage_noise() -> None:
    v, q = _cc_samples(NARROW_PEAK, 2.0)
    clean_h, clean_v = ica_peak(incremental_capacity(v, q, GRID), GRID)
    for seed in range(5):
        noisy = v + np.random.default_rng(seed).normal(0.0, 1e-3, v.size)
        height, v_peak = ica_peak(incremental_capacity(noisy, q, GRID), GRID)
        assert v_peak == pytest.approx(clean_v, abs=3e-3)
        assert height == pytest.approx(clean_h, rel=0.03)


def test_isotonic_fit_removes_the_upward_bias_of_a_running_maximum() -> None:
    """Why the default is isotonic: cummax rides the noise envelope and shifts the peak up."""
    v, q = _cc_samples(NARROW_PEAK, 2.0)
    _, clean_v = ica_peak(incremental_capacity(v, q, GRID), GRID)
    shifts: dict[str, list[float]] = {"isotonic": [], "cummax": []}
    for seed in range(20):
        noisy = v + np.random.default_rng(seed).normal(0.0, 1e-3, v.size)
        for method, out in shifts.items():
            dqdv = incremental_capacity(noisy, q, GRID, monotone=method)  # type: ignore[arg-type]
            out.append(ica_peak(dqdv, GRID)[1] - clean_v)
    assert abs(np.mean(shifts["isotonic"])) < 0.5e-3
    assert np.mean(shifts["cummax"]) > 1.0e-3


def test_segment_not_covering_the_grid_gives_nan() -> None:
    v = np.linspace(3.95, 4.3, 100)  # starts above the grid
    dqdv = incremental_capacity(v, np.linspace(0, 1, 100), GRID)
    assert np.isnan(dqdv).all()
    assert all(np.isnan(x) for x in ica_peak(dqdv, GRID))
    assert np.isnan(charge_on_grid(v[:1], np.zeros(1), GRID)).all()
    v_nan = np.linspace(3.8, 4.3, 100)
    v_nan[50] = np.nan
    assert np.isnan(charge_on_grid(v_nan, np.linspace(0, 1, 100), GRID)).all()


def test_maximum_on_the_window_edge_is_censored_not_dropped() -> None:
    """A peak below the window (hot cell) leaves a decreasing curve: report the edge, not NaN/0."""
    hot = OCVCurve(v_min=3.5, v_max=4.4, background=0.2, steps=(LogisticStep(3.80, 0.05, 1.0),))
    v, q = _cc_samples(hot, 2.0)
    dqdv = incremental_capacity(v, q, GRID)
    height, v_peak = ica_peak(dqdv, GRID)
    assert v_peak == pytest.approx(GRID[0])
    assert height == pytest.approx(float(dqdv[0]))
    assert height > 0


def test_peak_label_is_the_global_maximum_not_a_prominence_rank() -> None:
    """v1 ordered peaks by prominence and swapped labels; the max is unambiguous."""
    two = OCVCurve(
        v_min=3.5,
        v_max=4.4,
        background=0.1,
        steps=(LogisticStep(3.98, 0.02, 0.30), LogisticStep(4.10, 0.02, 0.45)),
    )
    v, q = _cc_samples(two, 2.0, n=6000)
    _, v_peak = ica_peak(incremental_capacity(v, q, GRID), GRID)
    assert v_peak == pytest.approx(4.10, abs=2e-3)


def test_interior_peak_wins_a_near_tie_with_the_window_edge() -> None:
    """B0006 cycles 86-88: interior peak 3.05 Ah/V at 4.05 V, rising edge 3.10 at 4.18 V."""
    v = GRID
    bump = 3.05 * np.exp(-(((v - 4.05) / 0.04) ** 2))
    rising_edge = 3.10 * np.exp(-(((v - 4.18) / 0.03) ** 2))
    curve = bump + rising_edge
    assert int(np.argmax(curve)) == v.size - 1  # the plain maximum is the edge
    _, v_peak = ica_peak(curve, v)
    assert v_peak == pytest.approx(4.05, abs=5e-3)
    _, v_edge = ica_peak(curve, v, tie_tolerance=0.0)  # plain maximum: the edge
    assert v_edge == pytest.approx(4.18)


def test_small_ripple_does_not_beat_a_clearly_higher_edge() -> None:
    v = GRID
    curve = 5.0 - 10.0 * (v - 3.90) + 0.05 * np.sin(2 * np.pi * (v - 3.90) / 0.04)
    height, v_peak = ica_peak(curve, v)
    assert v_peak == pytest.approx(3.90)
    assert height == pytest.approx(curve[0])


@pytest.mark.parametrize("tolerance", [-0.1, 1.0])
def test_invalid_tie_tolerance_raises(tolerance: float) -> None:
    with pytest.raises(ValueError, match="tie_tolerance"):
        ica_peak(np.ones(GRID.size), GRID, tie_tolerance=tolerance)


def test_parabolic_refinement_and_short_inputs() -> None:
    grid = np.array([0.0, 1.0, 2.0])
    height, x = ica_peak(np.array([1.0, 2.0, 1.0]), grid)
    assert (height, x) == (2.0, 1.0)
    height, x = ica_peak(np.array([0.0, 3.0, 2.0]), grid)
    assert 1.0 < x < 1.5  # the vertex lies towards the larger neighbour
    assert height > 3.0
    assert all(np.isnan(ica_peak(np.array([1.0, 2.0]), np.array([0.0, 1.0]))))


@pytest.mark.parametrize(
    ("window", "polyorder", "monotone", "match"),
    [
        (8, 2, "isotonic", "odd"),
        (1, 0, "isotonic", "odd"),
        (9, 0, "isotonic", "polyorder"),
        (9, 9, "isotonic", "polyorder"),
        (99, 2, "isotonic", "exceeds the grid"),
        (9, 2, "median", "monotone"),
    ],
)
def test_invalid_filter_settings_raise(
    window: int, polyorder: int, monotone: str, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        validate_ica_params(GRID, window, polyorder, monotone)  # type: ignore[arg-type]


def test_non_uniform_grid_raises() -> None:
    with pytest.raises(ValueError, match="uniform"):
        incremental_capacity(np.linspace(3.8, 4.3, 50), np.linspace(0, 1, 50), GRID**2)
