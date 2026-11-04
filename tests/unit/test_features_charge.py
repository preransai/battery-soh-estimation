"""Charge-side health indicators against the synthetic ground truth."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from battery_soh.data.schema import Segment
from battery_soh.features.charge import CHARGE_FEATURES, charge_features
from battery_soh.features.ica import ica_peak, incremental_capacity, voltage_grid
from battery_soh.synthetic import CellModel, SyntheticSegment, simulate_charge

NAN_FREE = ("chg_cc_time_390_419_s", "chg_cc_time_400_419_s", "chg_cv_time_s")


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


def _cv_time_truth(run: SyntheticSegment) -> float:
    t419 = run.crossing_time("voltage_v", 4.19, rising=True)
    return run.crossing_time("current_a", 0.1, rising=False, t_min=t419) - t419


@pytest.fixture(scope="module")
def clean(charge_run: SyntheticSegment) -> dict[str, float]:
    return charge_features(charge_run.segment)


# ----------------------------------------------------------------------------- ground truth


def test_returns_registry_keys_as_floats(clean: dict[str, float]) -> None:
    assert tuple(clean) == CHARGE_FEATURES
    assert all(isinstance(v, float) and np.isfinite(v) for v in clean.values())


@pytest.mark.parametrize(
    ("key", "v_from"), [("chg_cc_time_390_419_s", 3.90), ("chg_cc_time_400_419_s", 4.00)]
)
def test_cc_window_time_matches_closed_form(
    clean: dict[str, float], fresh_cell: CellModel, key: str, v_from: float
) -> None:
    _, expected_s = fresh_cell.cc_window(v_from, 4.19, 1.5)
    assert clean[key] == pytest.approx(expected_s, rel=1e-4)


def test_cv_time_matches_simulated_taper(
    clean: dict[str, float], charge_run: SyntheticSegment
) -> None:
    assert clean["chg_cv_time_s"] == pytest.approx(_cv_time_truth(charge_run), abs=1.0)


def test_ica_peak_matches_the_known_curve(clean: dict[str, float], fresh_cell: CellModel) -> None:
    height, v_peak = fresh_cell.cc_ica_peak(1.5)
    assert clean["chg_ica_peak_voltage_v"] == pytest.approx(v_peak, abs=1e-3)
    # The 45 mV Savitzky-Golay window lowers the narrow peak by a few percent.
    assert clean["chg_ica_peak_height_ah_per_v"] == pytest.approx(height, rel=0.05)
    assert clean["chg_ica_peak_height_ah_per_v"] < height


def test_temperature_rise_matches_model(
    clean: dict[str, float], charge_run: SyntheticSegment
) -> None:
    t390 = charge_run.crossing_time("voltage_v", 3.90, rising=True)
    t419 = charge_run.crossing_time("voltage_v", 4.19, rising=True)
    temp = charge_run.noiseless(np.array([t390, t419]))["temperature_c"]
    assert clean["chg_temp_rise_cc_c"] == pytest.approx(temp[1] - temp[0], abs=0.01)
    assert clean["chg_temp_rise_cc_c"] > 0


def test_rc_branch_settled_window_matches_closed_form(rc_cell: CellModel) -> None:
    feats = charge_features(simulate_charge(rc_cell).segment)
    _, expected_s = rc_cell.cc_window(3.90, 4.19, 1.5)
    assert feats["chg_cc_time_390_419_s"] == pytest.approx(expected_s, rel=1e-3)
    _, v_peak = rc_cell.cc_ica_peak(1.5)
    assert feats["chg_ica_peak_voltage_v"] == pytest.approx(v_peak, abs=2e-3)


def test_noisy_charge_stays_close_to_truth(
    noisy_charge_run: SyntheticSegment, fresh_cell: CellModel
) -> None:
    feats = charge_features(noisy_charge_run.segment)
    _, expected_s = fresh_cell.cc_window(3.90, 4.19, 1.5)
    assert feats["chg_cc_time_390_419_s"] == pytest.approx(expected_s, rel=0.01)
    assert feats["chg_cv_time_s"] == pytest.approx(_cv_time_truth(noisy_charge_run), rel=0.02)
    height, v_peak = fresh_cell.cc_ica_peak(1.5)
    assert feats["chg_ica_peak_voltage_v"] == pytest.approx(v_peak, abs=4e-3)
    assert feats["chg_ica_peak_height_ah_per_v"] == pytest.approx(height, rel=0.08)


def test_window_parameters_are_arguments(
    charge_run: SyntheticSegment, fresh_cell: CellModel
) -> None:
    feats = charge_features(charge_run.segment, v_start=3.80, start_margin_v=0.03)
    _, expected_s = fresh_cell.cc_window(3.80, 4.19, 1.5)
    assert feats["chg_cc_time_390_419_s"] == pytest.approx(expected_s, rel=1e-4)


# ----------------------------------------------------------------------------- not measurable


def test_high_cc_start_voltage_gives_nan_for_the_390_window_only(fresh_cell: CellModel) -> None:
    """Start at ~3.92 V under load: the 3.90 V window and the ICA grid start are not observed."""
    run = simulate_charge(fresh_cell, soc_start=0.62)
    start_v = run.segment.voltage_v[run.segment.current_a > 1.0][0]
    assert 3.88 < start_v < 3.97
    feats = charge_features(run.segment)
    for key in ("chg_cc_time_390_419_s", "chg_temp_rise_cc_c", "chg_ica_peak_height_ah_per_v"):
        assert np.isnan(feats[key]), key
    _, expected_s = fresh_cell.cc_window(4.00, 4.19, 1.5)
    assert feats["chg_cc_time_400_419_s"] == pytest.approx(expected_s, rel=1e-4)
    assert np.isfinite(feats["chg_cv_time_s"])


def test_partial_charge_before_cv_gives_all_nan(charge_run: SyntheticSegment) -> None:
    seg = charge_run.segment
    feats = charge_features(_subset(seg, seg.time_s < charge_run.truth["t_cv_start_s"] - 100))
    assert all(np.isnan(v) for v in feats.values())


def test_charge_cut_during_cv_keeps_cc_features(
    charge_run: SyntheticSegment, clean: dict[str, float]
) -> None:
    seg = charge_run.segment
    feats = charge_features(_subset(seg, seg.time_s < charge_run.truth["t_cv_start_s"] + 120))
    assert np.isnan(feats["chg_cv_time_s"])
    for key in ("chg_cc_time_390_419_s", "chg_ica_peak_height_ah_per_v", "chg_ica_peak_voltage_v"):
        assert feats[key] == pytest.approx(clean[key], rel=1e-12)


def test_charger_pause_inside_the_window_invalidates_the_timing(
    charge_run: SyntheticSegment,
) -> None:
    seg = charge_run.segment
    t_395 = charge_run.crossing_time("voltage_v", 3.95, rising=True)
    i = seg.current_a.copy()
    i[(seg.time_s > t_395) & (seg.time_s < t_395 + 35)] = 0.0
    feats = charge_features(dataclasses.replace(seg, current_a=i))
    assert np.isnan(feats["chg_cc_time_390_419_s"])
    assert np.isfinite(feats["chg_cc_time_400_419_s"])  # the pause ends before 4.00 V
    assert np.isfinite(feats["chg_cv_time_s"])


@pytest.mark.parametrize("n", [0, 1, 5, 9])
def test_too_short_segments_give_all_nan(charge_run: SyntheticSegment, n: int) -> None:
    seg = charge_run.segment
    keep = np.zeros(seg.n_samples, dtype=bool)
    keep[:n] = True
    assert all(np.isnan(v) for v in charge_features(_subset(seg, keep)).values())


def test_rest_only_record_gives_all_nan() -> None:
    t = np.arange(100.0) * 10
    rng = np.random.default_rng(0)
    seg = Segment("charge", t, np.full(100, 3.6), rng.normal(0, 0.002, 100), np.full(100, 24.0))
    assert all(np.isnan(v) for v in charge_features(seg).values())


# ----------------------------------------------------------------------------- robustness


def test_unsorted_duplicated_and_non_finite_samples(
    charge_run: SyntheticSegment, clean: dict[str, float]
) -> None:
    seg = charge_run.segment
    n = seg.n_samples
    order = np.random.default_rng(3).permutation(n)
    # Shuffle, then append duplicated timestamps with garbage values and a NaN row.
    t = np.concatenate([seg.time_s[order], seg.time_s[:50], [np.nan]])
    v = np.concatenate([seg.voltage_v[order], np.full(50, 9.0), [4.0]])
    i = np.concatenate([seg.current_a[order], np.full(50, -5.0), [1.0]])
    temp = np.concatenate([seg.temperature_c[order], np.full(50, 99.0), [24.0]])
    messy = Segment("charge", t, v, i, temp)
    _assert_same(charge_features(messy), clean)


def test_opposite_sign_convention_gives_the_same_features(
    charge_run: SyntheticSegment, clean: dict[str, float]
) -> None:
    seg = charge_run.segment
    _assert_same(charge_features(dataclasses.replace(seg, current_a=-seg.current_a)), clean)


def test_nasa_start_glitch_is_ignored(
    charge_run: SyntheticSegment, clean: dict[str, float]
) -> None:
    """NASA charge records begin with one -3.4 A sample at ~3.0 V before the CC phase."""
    seg = charge_run.segment
    i, v = seg.current_a.copy(), seg.voltage_v.copy()
    i[1], v[1] = -3.4, 3.0
    feats = charge_features(dataclasses.replace(seg, current_a=i, voltage_v=v))
    _assert_same(feats, clean)


def test_rest_and_cv_samples_never_enter_the_ica(
    charge_run: SyntheticSegment, clean: dict[str, float]
) -> None:
    """Regression for v1: relaxation samples inside the voltage window corrupted Q(V)."""
    seg = charge_run.segment
    # Append a 30-minute rest whose voltage relaxes down through the whole ICA window.
    t_tail = seg.time_s[-1] + 10.0 * np.arange(1, 181)
    v_tail = np.linspace(4.19, 3.85, 180)
    rested = Segment(
        "charge",
        np.concatenate([seg.time_s, t_tail]),
        np.concatenate([seg.voltage_v, v_tail]),
        np.concatenate([seg.current_a, np.zeros(180)]),
        np.concatenate([seg.temperature_c, np.full(180, 24.0)]),
    )
    _assert_same(charge_features(rested), clean)
    # And the ICA equals the one computed from the CC samples alone.
    cc = (seg.current_a > 0.75) & (seg.time_s <= charge_run.truth["t_cv_start_s"] + 10.0)
    t, v, i = seg.time_s[cc], seg.voltage_v[cc], seg.current_a[cc]
    q = np.concatenate(([0.0], np.cumsum(0.5 * (i[1:] + i[:-1]) * np.diff(t)))) / 3600.0
    grid = voltage_grid(3.90, 4.18, 0.005)
    height, v_peak = ica_peak(incremental_capacity(v, q, grid), grid)
    assert clean["chg_ica_peak_height_ah_per_v"] == pytest.approx(height, rel=1e-12)
    assert clean["chg_ica_peak_voltage_v"] == pytest.approx(v_peak, rel=1e-12)


# ----------------------------------------------------------------------------- arguments


def test_wrong_segment_kind_raises(discharge_run: SyntheticSegment) -> None:
    with pytest.raises(ValueError, match="charge"):
        charge_features(discharge_run.segment)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"v_start": 4.2}, "v_start"),
        ({"alt_v_start": 4.19}, "alt_v_start"),
        ({"cv_cutoff_a": 0.0}, "cv_cutoff_a"),
        ({"start_margin_v": -0.01}, "margins"),
        ({"ica_start_margin_v": -0.01}, "margins"),
        ({"ica_grid": (3.9, 4.18, 0.006)}, "multiple of step"),
        ({"ica_window": 8}, "odd"),
        ({"ica_polyorder": 0}, "polyorder"),
        ({"ica_monotone": "median"}, "monotone"),
        ({"v_end": float("nan")}, "finite"),
    ],
)
def test_invalid_arguments_raise_even_for_unusable_data(
    kwargs: dict[str, object], match: str
) -> None:
    empty = Segment("charge")
    with pytest.raises(ValueError, match=match):
        charge_features(empty, **kwargs)  # type: ignore[arg-type]
