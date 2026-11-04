"""The synthetic generator obeys the physics it claims, so it can serve as ground truth."""

from __future__ import annotations

import numpy as np
import pytest

from battery_soh.synthetic import (
    SENSOR_NOISE,
    CellModel,
    LogisticStep,
    Noise,
    OCVCurve,
    SyntheticSegment,
    simulate_charge,
    simulate_discharge,
)

# ----------------------------------------------------------------------------- OCV curve


def test_ocv_curve_is_normalised_and_strictly_increasing() -> None:
    ocv = OCVCurve()
    v = np.linspace(ocv.v_min, ocv.v_max, 5001)
    soc = ocv.soc(v)
    assert soc[0] == pytest.approx(0.0, abs=1e-12)
    assert soc[-1] == pytest.approx(1.0, abs=1e-12)
    assert np.all(np.diff(soc) > 0)


def test_dsoc_dv_is_the_derivative_of_soc() -> None:
    ocv = OCVCurve()
    v = np.linspace(3.0, 4.15, 400)
    h = 1e-6
    numeric = (ocv.soc(v + h) - ocv.soc(v - h)) / (2 * h)
    np.testing.assert_allclose(ocv.dsoc_dv(v), numeric, rtol=1e-6)
    scalar = np.array([ocv.dsoc_dv_scalar(float(x)) for x in v])
    np.testing.assert_allclose(scalar, ocv.dsoc_dv(v), rtol=1e-12)


@pytest.mark.parametrize("soc", [0.0, 0.03, 0.5, 0.97, 1.0])
def test_ocv_is_the_inverse_of_soc(soc: float) -> None:
    ocv = OCVCurve()
    assert float(ocv.soc(ocv.ocv(soc))) == pytest.approx(soc, abs=1e-10)


def test_default_curve_is_nasa_like() -> None:
    """Charge-side ICA peak ~4.0 V at 1.5 A and ~0.8 Ah in the 3.8 -> 3.5 V window at 2 A."""
    cell = CellModel()
    height, v_peak = cell.cc_ica_peak(1.5)
    assert v_peak == pytest.approx(3.90 + 1.5 * cell.r0_ohm, abs=2e-3)
    assert 4.0 < height < 6.5
    window_ah, _ = cell.cc_window(3.80, 3.50, -2.0)
    assert 0.35 < window_ah / cell.capacity_ah < 0.5


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"v_min": 4.2, "v_max": 2.5}, "v_min < v_max"),
        ({"background": 0.0}, "background"),
        ({"steps": (LogisticStep(3.9, 0.0, 0.2),)}, "width_v > 0"),
        ({"steps": (LogisticStep(3.9, 0.02, -0.1),)}, "weight >= 0"),
        ({"v_min": float("nan")}, "finite"),
    ],
)
def test_invalid_ocv_curve_raises(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        OCVCurve(**kwargs)  # type: ignore[arg-type]


def test_ocv_outside_unit_interval_raises() -> None:
    with pytest.raises(ValueError, match="soc"):
        OCVCurve().ocv(1.2)


# ----------------------------------------------------------------------------- cell model


def test_ageing_scales_capacity_and_grows_resistance() -> None:
    cell = CellModel(r1_ohm=0.02)
    old = cell.aged(0.8, resistance_growth=1.5)
    assert old.capacity_ah == pytest.approx(0.8 * cell.capacity_ah)
    assert old.r0_ohm == pytest.approx(cell.r0_ohm * 1.3)
    assert old.r1_ohm == pytest.approx(cell.r1_ohm * 1.3)
    assert cell.aged(1.0) == cell


@pytest.mark.parametrize("soh", [0.0, 1.1, -0.5])
def test_invalid_ageing_raises(soh: float) -> None:
    with pytest.raises(ValueError, match="soh"):
        CellModel().aged(soh)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"capacity_ah": 0.0}, "capacity_ah"),
        ({"r0_ohm": 0.0}, "r0_ohm"),
        ({"r1_ohm": -0.01}, "r1_ohm"),
        ({"heat_capacity_j_per_k": 0.0}, "thermal"),
        ({"tau1_s": float("inf")}, "finite"),
    ],
)
def test_invalid_cell_raises(kwargs: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        CellModel(**kwargs)  # type: ignore[arg-type]


def test_cc_window_closed_form_is_symmetric_in_direction() -> None:
    cell = CellModel()
    q_dis, t_dis = cell.cc_window(3.80, 3.50, -2.0)
    q_rev, _ = cell.cc_window(3.50, 3.80, -2.0)
    assert q_dis == pytest.approx(q_rev)
    assert t_dis == pytest.approx(q_dis * 3600.0 / 2.0)
    with pytest.raises(ValueError, match="non-zero"):
        cell.cc_window(3.8, 3.5, 0.0)


# ----------------------------------------------------------------------------- discharge


def test_discharge_follows_nasa_sign_convention(discharge_run: SyntheticSegment) -> None:
    seg, truth = discharge_run.segment, discharge_run.truth
    loaded = (seg.time_s >= truth["t_load_on_s"]) & (seg.time_s < truth["t_load_off_s"])
    np.testing.assert_allclose(seg.current_a[loaded], -2.0)
    np.testing.assert_allclose(seg.current_a[~loaded], 0.0)
    assert seg.kind == "discharge"
    assert seg.capacity_ah == pytest.approx(truth["delivered_ah"])


def test_delivered_charge_matches_closed_form(
    discharge_run: SyntheticSegment, fresh_cell: CellModel
) -> None:
    """With R1 = 0 the cutoff is reached when OCV = cutoff + I * R0."""
    soc_cut = float(fresh_cell.ocv.soc(2.7 + 2.0 * fresh_cell.r0_ohm))
    expected = fresh_cell.capacity_ah * (1.0 - soc_cut)
    truth = discharge_run.truth
    assert truth["delivered_ah"] == pytest.approx(expected, rel=1e-6)
    assert truth["delivered_ah"] == pytest.approx(
        2.0 * (truth["t_load_off_s"] - truth["t_load_on_s"]) / 3600.0
    )


def test_load_on_step_is_exactly_i_times_r0(
    discharge_run: SyntheticSegment, fresh_cell: CellModel
) -> None:
    seg = discharge_run.segment
    k0 = int(np.flatnonzero(seg.time_s >= discharge_run.truth["t_load_on_s"])[0])
    v_ocv = fresh_cell.ocv.ocv(1.0)
    assert seg.voltage_v[k0 - 1] == pytest.approx(v_ocv, abs=1e-9)
    assert seg.voltage_v[k0 - 1] - seg.voltage_v[k0] == pytest.approx(2.0 * fresh_cell.r0_ohm)


def test_crossing_time_matches_closed_form(
    discharge_run: SyntheticSegment, fresh_cell: CellModel
) -> None:
    t_on = discharge_run.truth["t_load_on_s"]
    _, t_from_full = fresh_cell.cc_window(
        fresh_cell.ocv.ocv(1.0) - 2.0 * fresh_cell.r0_ohm, 3.8, -2.0
    )
    t_380 = discharge_run.crossing_time("voltage_v", 3.80, rising=False)
    assert t_380 == pytest.approx(t_on + t_from_full, abs=1e-3)
    # The load-on voltage jump is a crossing at the switching instant.
    v_rest = fresh_cell.ocv.ocv(1.0)
    assert discharge_run.crossing_time("voltage_v", v_rest - 0.01, rising=False) == t_on
    assert np.isnan(discharge_run.crossing_time("voltage_v", 1.0, rising=False))


def test_rest_tail_relaxes_upwards(discharge_run: SyntheticSegment) -> None:
    seg, t_off = discharge_run.segment, discharge_run.truth["t_load_off_s"]
    tail = seg.voltage_v[seg.time_s > t_off]
    assert tail.size > 10
    assert tail.min() > 2.7  # the voltage jumps back up when the load is removed


def test_discharge_heats_the_cell_within_physical_bounds(
    discharge_run: SyntheticSegment, fresh_cell: CellModel
) -> None:
    temp = discharge_run.segment.temperature_c
    assert temp[0] == pytest.approx(24.0)
    steady_rise = 2.0**2 * fresh_cell.r0_ohm / fresh_cell.heat_transfer_w_per_k
    assert 1.0 < temp.max() - 24.0 < steady_rise
    loaded = discharge_run.segment.time_s <= discharge_run.truth["t_load_off_s"]
    assert np.all(np.diff(temp[loaded]) >= -1e-12)  # monotone heating under constant load


def test_rc_branch_keeps_the_instantaneous_step_ohmic(rc_cell: CellModel) -> None:
    run = simulate_discharge(rc_cell)
    seg = run.segment
    k0 = int(np.flatnonzero(seg.time_s >= run.truth["t_load_on_s"])[0])
    assert seg.voltage_v[k0 - 1] - seg.voltage_v[k0] == pytest.approx(2.0 * rc_cell.r0_ohm)
    # 10 s later the RC branch has added I * R1 * (1 - exp(-10 / tau)) of polarisation.
    model = run.noiseless(np.array([seg.time_s[k0] + 200.0]))
    v_ss = model["ocv_v"][0] - 2.0 * rc_cell.steady_resistance_ohm
    assert model["voltage_v"][0] == pytest.approx(v_ss, abs=1e-5)


def test_partial_discharge_is_capped(fresh_cell: CellModel) -> None:
    run = simulate_discharge(fresh_cell, max_load_s=600.0, rest_after_s=0.0)
    assert run.truth["t_load_off_s"] - run.truth["t_load_on_s"] == pytest.approx(600.0)
    assert run.segment.voltage_v.min() > 3.5


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"current_a": 0.0}, "current_a"),
        ({"current_a": -2.0}, "current_a"),
        ({"soc_start": 1.5}, "soc_start"),
        ({"cutoff_v": 2.0, "current_a": 1.0}, "over-discharge"),
        ({"soc_start": 0.01}, "starts at or below the cutoff"),
        ({"sample_s": 0.0}, "sample_s"),
        ({"rest_before_s": -1.0}, "rest"),
        ({"max_load_s": 0.0}, "max_load_s"),
    ],
)
def test_invalid_discharge_arguments_raise(kwargs: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        simulate_discharge(CellModel(), **kwargs)  # type: ignore[arg-type]


# ----------------------------------------------------------------------------- charge


def test_charge_is_cc_then_cv(charge_run: SyntheticSegment, fresh_cell: CellModel) -> None:
    seg, truth = charge_run.segment, charge_run.truth
    t, v, i = seg.time_s, seg.voltage_v, seg.current_a
    cc = (t >= truth["t_cc_start_s"]) & (t < truth["t_cv_start_s"])
    cv = (t >= truth["t_cv_start_s"]) & (t <= truth["t_cv_end_s"])
    np.testing.assert_allclose(i[cc], 1.5)  # positive: NASA charge convention
    np.testing.assert_allclose(v[cv], 4.2, atol=1e-7)
    assert np.all(np.diff(i[cv]) < 0)  # CV taper
    assert i[cv][-1] > 0.02 - 1e-9
    assert v.max() <= 4.2 + 1e-7
    # CC -> CV transition time in closed form: OCV reaches v_max - I * R0.
    soc_cv = float(fresh_cell.ocv.soc(4.2 - 1.5 * fresh_cell.r0_ohm))
    t_cc = (soc_cv - 0.05) * fresh_cell.capacity_ah * 3600.0 / 1.5
    assert truth["t_cv_start_s"] - truth["t_cc_start_s"] == pytest.approx(t_cc, rel=1e-6)


def test_charged_ah_matches_integrated_current(charge_run: SyntheticSegment) -> None:
    t_end = charge_run.truth["t_cv_end_s"]
    t = np.linspace(0.0, t_end, 40_001)
    i = charge_run.noiseless(t)["current_a"]
    q = float(np.sum(0.5 * (i[1:] + i[:-1]) * np.diff(t))) / 3600.0
    assert charge_run.truth["charged_ah"] == pytest.approx(q, rel=1e-5)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"cutoff_a": 0.0}, "cutoff_a"),
        ({"cutoff_a": 2.0}, "cutoff_a"),
        ({"v_max": 4.3}, "OCV range"),
        ({"soc_start": 0.99}, "above the CC start voltage"),
        ({"soc_start": -0.1}, "soc_start"),
    ],
)
def test_invalid_charge_arguments_raise(kwargs: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        simulate_charge(CellModel(), **kwargs)  # type: ignore[arg-type]


# ----------------------------------------------------------------------------- noise


def test_noise_is_seeded_and_has_the_requested_level(fresh_cell: CellModel) -> None:
    a = simulate_discharge(fresh_cell, noise=SENSOR_NOISE, seed=3)
    b = simulate_discharge(fresh_cell, noise=SENSOR_NOISE, seed=3)
    c = simulate_discharge(fresh_cell, noise=SENSOR_NOISE, seed=4)
    np.testing.assert_array_equal(a.segment.voltage_v, b.segment.voltage_v)
    assert not np.array_equal(a.segment.voltage_v, c.segment.voltage_v)
    residual = a.segment.voltage_v - a.noiseless(a.segment.time_s)["voltage_v"]
    assert np.std(residual) == pytest.approx(SENSOR_NOISE.voltage_v, rel=0.2)


def test_invalid_noise_raises() -> None:
    with pytest.raises(ValueError, match="noise"):
        Noise(voltage_v=-1.0)
