"""Discharge-side health indicators against the synthetic ground truth."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from battery_soh.data.schema import Segment
from battery_soh.features.discharge import (
    DISCHARGE_FEATURES,
    WINDOW_FEATURES,
    discharge_features,
    discharge_window_features,
    load_current,
    ohmic_resistance,
)
from battery_soh.synthetic import SENSOR_NOISE, CellModel, SyntheticSegment, simulate_discharge


def _subset(seg: Segment, keep: np.ndarray) -> Segment:
    return dataclasses.replace(
        seg,
        time_s=seg.time_s[keep],
        voltage_v=seg.voltage_v[keep],
        current_a=seg.current_a[keep],
        temperature_c=seg.temperature_c[keep],
    )


def _assert_same(a: dict[str, float], b: dict[str, float]) -> None:
    assert a.keys() == b.keys()
    for key in a:
        np.testing.assert_allclose(a[key], b[key], rtol=1e-12, atol=0, err_msg=key)


@pytest.fixture(scope="module")
def clean(discharge_run: SyntheticSegment) -> dict[str, float]:
    return discharge_features(discharge_run.segment)


# ----------------------------------------------------------------------------- ground truth


def test_returns_registry_keys_as_positive_floats(clean: dict[str, float]) -> None:
    assert tuple(clean) == DISCHARGE_FEATURES
    assert all(isinstance(v, float) and v > 0 for v in clean.values())


def test_window_charge_matches_closed_form(clean: dict[str, float], fresh_cell: CellModel) -> None:
    expected_ah, expected_s = fresh_cell.cc_window(3.80, 3.50, -2.0)
    assert clean["dis_q_380_350_ah"] == pytest.approx(expected_ah, rel=1e-4)
    assert clean["dis_time_380_350_s"] == pytest.approx(expected_s, rel=1e-4)


def test_window_charge_is_current_times_time(clean: dict[str, float]) -> None:
    """Under constant current, Ah in the window = I * dt exactly."""
    assert clean["dis_q_380_350_ah"] == pytest.approx(
        2.0 * clean["dis_time_380_350_s"] / 3600.0, rel=1e-12
    )


def test_load_on_step_recovers_r0_exactly(clean: dict[str, float], fresh_cell: CellModel) -> None:
    assert clean["dis_r0_ohm"] == pytest.approx(fresh_cell.r0_ohm, rel=1e-9)


def test_r0_with_rc_branch_is_still_the_ohmic_part(rc_cell: CellModel) -> None:
    assert ohmic_resistance(simulate_discharge(rc_cell).segment) == pytest.approx(
        rc_cell.r0_ohm, rel=1e-9
    )


def test_r0_between_samples_includes_the_ocv_drop(fresh_cell: CellModel) -> None:
    """Load switched on 5 s before the next sample: the OCV has already moved by then."""
    r0 = ohmic_resistance(simulate_discharge(fresh_cell, rest_before_s=25.0).segment)
    soc_at_sample = 1.0 - 2.0 * 5.0 / 3600.0 / fresh_cell.capacity_ah
    ocv_drop = fresh_cell.ocv.ocv(1.0) - fresh_cell.ocv.ocv(soc_at_sample)
    assert r0 == pytest.approx(fresh_cell.r0_ohm + ocv_drop / 2.0, rel=1e-5)


def test_temperature_rise_matches_model(
    clean: dict[str, float], discharge_run: SyntheticSegment
) -> None:
    t_hi = discharge_run.crossing_time("voltage_v", 3.80, rising=False)
    t_lo = discharge_run.crossing_time("voltage_v", 3.50, rising=False)
    temp = discharge_run.noiseless(np.array([t_hi, t_lo]))["temperature_c"]
    assert clean["dis_temp_rise_win_c"] == pytest.approx(temp[1] - temp[0], abs=1e-3)


def test_load_current_is_a_positive_magnitude(
    clean: dict[str, float], discharge_run: SyntheticSegment
) -> None:
    assert discharge_run.segment.current_a.min() == pytest.approx(-2.0)  # NASA: negative
    assert clean["ctx_discharge_current_a"] == pytest.approx(2.0)
    assert load_current(discharge_run.segment) == pytest.approx(2.0)


def test_combined_function_equals_the_individual_ones(
    clean: dict[str, float], discharge_run: SyntheticSegment
) -> None:
    seg = discharge_run.segment
    window = discharge_window_features(seg)
    assert tuple(window) == WINDOW_FEATURES
    for key, value in window.items():
        assert clean[key] == value
    assert clean["dis_r0_ohm"] == ohmic_resistance(seg)
    assert clean["ctx_discharge_current_a"] == load_current(seg)


def test_window_is_a_parameter(discharge_run: SyntheticSegment, fresh_cell: CellModel) -> None:
    feats = discharge_window_features(discharge_run.segment, v_high=3.90, v_low=3.60)
    expected_ah, _ = fresh_cell.cc_window(3.90, 3.60, -2.0)
    assert tuple(feats) == WINDOW_FEATURES  # keys keep their registry names
    assert feats["dis_q_380_350_ah"] == pytest.approx(expected_ah, rel=1e-4)


def test_noisy_discharge_stays_close_to_truth(
    noisy_discharge_run: SyntheticSegment, fresh_cell: CellModel
) -> None:
    feats = discharge_features(noisy_discharge_run.segment)
    expected_ah, _ = fresh_cell.cc_window(3.80, 3.50, -2.0)
    assert feats["dis_q_380_350_ah"] == pytest.approx(expected_ah, rel=0.01)
    assert feats["dis_r0_ohm"] == pytest.approx(fresh_cell.r0_ohm, rel=0.05)
    assert feats["ctx_discharge_current_a"] == pytest.approx(2.0, abs=0.005)


# ----------------------------------------------------------------------------- not measurable


def test_load_starting_below_the_window_gives_nan(fresh_cell: CellModel) -> None:
    run = simulate_discharge(fresh_cell, soc_start=0.45)
    assert run.segment.voltage_v[run.segment.current_a < -1].max() < 3.80
    feats = discharge_features(run.segment)
    assert all(np.isnan(feats[k]) for k in WINDOW_FEATURES)
    assert feats["dis_r0_ohm"] == pytest.approx(fresh_cell.r0_ohm, rel=1e-9)


def test_window_starting_inside_the_switch_on_transient_gives_nan(fresh_cell: CellModel) -> None:
    """Loaded voltage starts just above 3.80 V and crosses it within settle_s."""
    soc = float(fresh_cell.ocv.soc(3.80 + 2.0 * fresh_cell.r0_ohm + 0.002))
    run = simulate_discharge(fresh_cell, soc_start=soc)
    assert all(np.isnan(v) for v in discharge_window_features(run.segment).values())
    assert np.isfinite(discharge_window_features(run.segment, settle_s=0.0)["dis_q_380_350_ah"])


def test_cutoff_above_the_window_end_gives_nan_despite_rest_relaxation(
    fresh_cell: CellModel,
) -> None:
    """The load stops at 3.55 V; the rest tail relaxes upwards and must not be used."""
    run = simulate_discharge(fresh_cell, cutoff_v=3.55, rest_after_s=1800.0)
    feats = discharge_features(run.segment)
    assert all(np.isnan(feats[k]) for k in WINDOW_FEATURES)
    assert np.isfinite(feats["dis_r0_ohm"])


def test_partial_discharge_before_the_window_end_gives_nan(fresh_cell: CellModel) -> None:
    run = simulate_discharge(fresh_cell, max_load_s=900.0)
    assert all(np.isnan(v) for v in discharge_window_features(run.segment).values())


def test_record_starting_under_load_has_no_r0_but_keeps_the_window(
    discharge_run: SyntheticSegment, clean: dict[str, float]
) -> None:
    seg = discharge_run.segment
    feats = discharge_features(_subset(seg, seg.time_s >= discharge_run.truth["t_load_on_s"]))
    assert np.isnan(feats["dis_r0_ohm"])
    assert feats["dis_q_380_350_ah"] == pytest.approx(clean["dis_q_380_350_ah"], rel=1e-12)


@pytest.mark.parametrize("n", [0, 1, 9])
def test_too_short_segments_give_all_nan(discharge_run: SyntheticSegment, n: int) -> None:
    seg = discharge_run.segment
    keep = np.zeros(seg.n_samples, dtype=bool)
    keep[:n] = True
    assert all(np.isnan(v) for v in discharge_features(_subset(seg, keep)).values())


def test_rest_only_record_gives_all_nan_never_zero() -> None:
    """Sensor offset on a rest record must not be mistaken for a load (v1 returned 0.0 R0)."""
    rng = np.random.default_rng(0)
    t = np.arange(200.0) * 10
    seg = Segment(
        "discharge", t, 3.9 + rng.normal(0, 1e-3, 200), rng.normal(0, 3e-3, 200), np.full(200, 24.0)
    )
    assert all(np.isnan(v) for v in discharge_features(seg).values())


# ----------------------------------------------------------------------------- v1 regressions


def test_load_removal_is_never_picked_as_the_ohmic_step(
    discharge_run: SyntheticSegment, fresh_cell: CellModel
) -> None:
    """v1 used argmax |dI|: make the removal step the larger one and check the onset is used."""
    seg = discharge_run.segment
    i = seg.current_a.copy()
    last_on = int(np.flatnonzero(i < -1.0)[-1])
    i[last_on] = -2.6
    modified = dataclasses.replace(seg, current_a=i)
    d_i = np.abs(np.diff(i))
    assert int(np.argmax(d_i)) == last_on  # the v1 rule would pick the removal
    assert ohmic_resistance(modified) == pytest.approx(fresh_cell.r0_ohm, rel=1e-9)


@pytest.mark.parametrize("current_a", [1.0, 4.0])
def test_thresholds_are_relative_to_the_load_current(
    fresh_cell: CellModel, current_a: float
) -> None:
    """v1 used a fixed 1.0 A threshold: B0048 (1 A load) got DCIR = 0 in 36 of 69 cycles."""
    feats = discharge_features(simulate_discharge(fresh_cell, current_a=current_a).segment)
    assert feats["dis_r0_ohm"] == pytest.approx(fresh_cell.r0_ohm, rel=1e-9)
    assert feats["ctx_discharge_current_a"] == pytest.approx(current_a)
    expected_ah, _ = fresh_cell.cc_window(3.80, 3.50, -current_a)
    assert feats["dis_q_380_350_ah"] == pytest.approx(expected_ah, rel=1e-3)


def test_ramped_onset_gives_nan_not_zero(
    discharge_run: SyntheticSegment, fresh_cell: CellModel
) -> None:
    """A load that ramps up over three samples has no single-step dV/dI."""
    seg = discharge_run.segment
    k0 = int(np.flatnonzero(seg.current_a < -1.0)[0])
    i, v = seg.current_a.copy(), seg.voltage_v.copy()
    v_rest = v[k0 - 1]
    for k, amps in ((k0, 0.7), (k0 + 1, 1.4)):
        i[k] = -amps
        v[k] = v_rest - amps * fresh_cell.r0_ohm
    r0 = ohmic_resistance(dataclasses.replace(seg, current_a=i, voltage_v=v))
    assert np.isnan(r0)


def test_rest_tail_does_not_change_any_window_feature(fresh_cell: CellModel) -> None:
    short = discharge_features(simulate_discharge(fresh_cell, rest_after_s=0.0).segment)
    long = discharge_features(simulate_discharge(fresh_cell, rest_after_s=3600.0).segment)
    _assert_same(short, long)


# ----------------------------------------------------------------------------- robustness


def test_unsorted_duplicated_and_non_finite_samples(
    discharge_run: SyntheticSegment, clean: dict[str, float]
) -> None:
    seg = discharge_run.segment
    order = np.random.default_rng(5).permutation(seg.n_samples)
    t = np.concatenate([seg.time_s[order], seg.time_s[10:40], [np.nan, 5.0]])
    v = np.concatenate([seg.voltage_v[order], np.full(30, 1.0), [3.9, np.nan]])
    i = np.concatenate([seg.current_a[order], np.full(30, 7.0), [-2.0, -2.0]])
    temp = np.concatenate([seg.temperature_c[order], np.full(30, 80.0), [24.0, 24.0]])
    _assert_same(discharge_features(Segment("discharge", t, v, i, temp)), clean)


def test_positive_discharge_current_convention_gives_the_same_features(
    discharge_run: SyntheticSegment, clean: dict[str, float]
) -> None:
    seg = discharge_run.segment
    _assert_same(discharge_features(dataclasses.replace(seg, current_a=-seg.current_a)), clean)


def test_nan_temperature_only_affects_the_temperature_feature(
    discharge_run: SyntheticSegment, clean: dict[str, float]
) -> None:
    seg = discharge_run.segment
    feats = discharge_features(
        dataclasses.replace(seg, temperature_c=np.full(seg.n_samples, np.nan))
    )
    assert np.isnan(feats["dis_temp_rise_win_c"])
    assert feats["dis_q_380_350_ah"] == clean["dis_q_380_350_ah"]


def test_noise_seed_changes_values_but_not_conclusions(fresh_cell: CellModel) -> None:
    values = [
        discharge_features(simulate_discharge(fresh_cell, noise=SENSOR_NOISE, seed=s).segment)
        for s in (1, 2)
    ]
    assert values[0]["dis_q_380_350_ah"] != values[1]["dis_q_380_350_ah"]
    assert values[0]["dis_q_380_350_ah"] == pytest.approx(values[1]["dis_q_380_350_ah"], rel=0.01)


# ----------------------------------------------------------------------------- arguments


def test_wrong_segment_kind_raises(charge_run: SyntheticSegment) -> None:
    for func in (discharge_features, discharge_window_features, ohmic_resistance, load_current):
        with pytest.raises(ValueError, match="discharge"):
            func(charge_run.segment)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"v_high": 3.5, "v_low": 3.8}, "v_low < v_high"),
        ({"v_low": float("nan")}, "v_low < v_high"),
        ({"settle_s": -1.0}, "settle_s"),
    ],
)
def test_invalid_window_raises(kwargs: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        discharge_window_features(Segment("discharge"), **kwargs)
    with pytest.raises(ValueError, match=match):
        discharge_features(Segment("discharge"), **kwargs)
