"""features_from_segments: the train/serve parity point, plus EIS and legacy columns."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable

import numpy as np
import pytest

from battery_soh.data.schema import Segment
from battery_soh.features.charge import charge_features
from battery_soh.features.discharge import discharge_features
from battery_soh.features.extract import (
    METADATA_FEATURES,
    SEGMENT_FEATURES,
    features_from_segments,
)
from battery_soh.features.impedance import EIS_FEATURES, eis_features, impedance_values
from battery_soh.features.legacy import LEGACY_FEATURES, legacy_features
from battery_soh.features.registry import FEATURES
from battery_soh.synthetic import CellModel, SyntheticSegment

AgeingRuns = list[tuple[CellModel, SyntheticSegment, SyntheticSegment]]


@pytest.fixture(scope="module")
def full_row(charge_run: SyntheticSegment, discharge_run: SyntheticSegment) -> dict[str, float]:
    return features_from_segments(charge_run.segment, discharge_run.segment, eis=(0.05, 0.08))


def _degenerate_segments(kind: str) -> list[Segment]:
    rng = np.random.default_rng(0)
    t = np.arange(50.0)
    return [
        Segment(kind),  # type: ignore[arg-type]
        Segment(kind, t[:3], np.full(3, 3.7), np.ones(3), np.full(3, 24.0)),  # type: ignore[arg-type]
        Segment(kind, t, np.full(50, 3.7), rng.normal(0, 1e-3, 50), np.full(50, 24.0)),  # type: ignore[arg-type]
        Segment(kind, t, np.full(50, np.nan), np.full(50, np.nan), np.full(50, np.nan)),  # type: ignore[arg-type]
    ]


# ----------------------------------------------------------------------------- contract


def test_keys_are_every_array_derived_registry_feature_in_registry_order(
    full_row: dict[str, float],
) -> None:
    assert tuple(full_row) == SEGMENT_FEATURES
    assert tuple(n for n in FEATURES if n not in METADATA_FEATURES) == SEGMENT_FEATURES
    assert set(METADATA_FEATURES) == {"ctx_cycle_index", "ctx_ambient_c"}
    assert "ctx_discharge_current_a" in SEGMENT_FEATURES
    assert all(isinstance(v, float) for v in full_row.values())


def test_registry_availability_metadata_means_not_measured_from_segments() -> None:
    # The registry's "metadata" availability and the extractor's METADATA_FEATURES must agree:
    # the measured load current is an operating condition but needs a (partial) discharge.
    metadata = {n for n, spec in FEATURES.items() if spec.availability == "metadata"}
    assert metadata == set(METADATA_FEATURES)
    assert FEATURES["ctx_discharge_current_a"].availability == "partial_discharge"


def test_values_equal_the_individual_extractors(
    full_row: dict[str, float], charge_run: SyntheticSegment, discharge_run: SyntheticSegment
) -> None:
    expected = {
        **charge_features(charge_run.segment),
        **discharge_features(discharge_run.segment),
        **legacy_features(discharge_run.segment),
        **eis_features((0.05, 0.08)),
    }
    assert full_row == expected


def test_every_value_is_physically_plausible(full_row: dict[str, float]) -> None:
    for name, value in full_row.items():
        lo, hi = FEATURES[name].plausible_range
        assert lo <= value <= hi, name


def test_missing_segments_give_nan_never_zero() -> None:
    row = features_from_segments(None, None)
    assert tuple(row) == SEGMENT_FEATURES
    assert all(math.isnan(v) for v in row.values())


def test_charge_only_and_discharge_only(
    charge_run: SyntheticSegment, discharge_run: SyntheticSegment, full_row: dict[str, float]
) -> None:
    chg = features_from_segments(charge_run.segment, None)
    dis = features_from_segments(None, discharge_run.segment)
    for name in SEGMENT_FEATURES:
        from_charge = name.startswith("chg_")
        from_discharge = name.startswith(("dis_", "leg_")) or name == "ctx_discharge_current_a"
        assert (chg[name] == full_row[name]) if from_charge else math.isnan(chg[name]), name
        assert (dis[name] == full_row[name]) if from_discharge else math.isnan(dis[name]), name


@pytest.mark.parametrize("kind", ["charge", "discharge"])
def test_degenerate_segments_never_produce_zero_placeholders(kind: str) -> None:
    """Regression for v1, which returned 0.0 (ICA, DCIR) when a feature failed."""
    for seg in _degenerate_segments(kind):
        row = (
            features_from_segments(seg, None)
            if kind == "charge"
            else features_from_segments(None, dataclasses.replace(seg, capacity_ah=None))
        )
        computed = {k: v for k, v in row.items() if not math.isnan(v)}
        # Only the whole-record legacy statistics may exist for a non-empty record.
        assert set(computed) <= {
            "leg_record_time_s",
            "leg_current_mean_a",
            "leg_voltage_mean_v",
            "leg_temp_max_c",
        }, computed
        assert 0.0 not in {v for k, v in computed.items() if k != "leg_current_mean_a"}


def test_is_deterministic(charge_run: SyntheticSegment, discharge_run: SyntheticSegment) -> None:
    a = features_from_segments(charge_run.segment, discharge_run.segment)
    b = features_from_segments(charge_run.segment, discharge_run.segment)
    np.testing.assert_array_equal(list(a.values()), list(b.values()))


def test_wrong_segment_kinds_raise(
    charge_run: SyntheticSegment, discharge_run: SyntheticSegment
) -> None:
    with pytest.raises(ValueError, match="expected a 'charge' segment"):
        features_from_segments(discharge_run.segment, None)
    with pytest.raises(ValueError, match="expected a 'discharge' segment"):
        features_from_segments(None, charge_run.segment)
    with pytest.raises(ValueError, match="eis must be"):
        features_from_segments(None, None, eis=(0.1,))  # type: ignore[arg-type]


# ----------------------------------------------------------------------------- EIS & legacy


def test_eis_values_map_to_features() -> None:
    assert eis_features((0.05, 0.08)) == {"eis_re_ohm": 0.05, "eis_rct_ohm": 0.08}
    assert all(math.isnan(v) for v in eis_features(None).values())
    # Failed equivalent-circuit fits (non-positive / non-finite) are NaN.
    bad = eis_features((0.0, float("inf")))
    assert tuple(bad) == EIS_FEATURES
    assert all(math.isnan(v) for v in bad.values())
    with pytest.raises(ValueError, match="eis must be"):
        eis_features(("a", "b"))  # type: ignore[arg-type]


def test_impedance_segment_values() -> None:
    assert impedance_values(Segment("impedance", re_ohm=0.06, rct_ohm=0.09)) == (0.06, 0.09)
    re_ohm, rct_ohm = impedance_values(Segment("impedance"))
    assert math.isnan(re_ohm)
    assert math.isnan(rct_ohm)
    with pytest.raises(ValueError, match="impedance"):
        impedance_values(Segment("charge"))


def test_legacy_columns_reproduce_v1_semantics(discharge_run: SyntheticSegment) -> None:
    seg = discharge_run.segment
    leg = legacy_features(seg)
    assert tuple(leg) == LEGACY_FEATURES
    assert leg["leg_capacity_ah"] == pytest.approx(discharge_run.truth["delivered_ah"])
    assert leg["leg_record_time_s"] == pytest.approx(seg.time_s[-1] - seg.time_s[0])
    assert leg["leg_current_mean_a"] == pytest.approx(np.mean(np.abs(seg.current_a)))
    assert leg["leg_voltage_mean_v"] == pytest.approx(np.mean(seg.voltage_v))
    assert leg["leg_temp_max_c"] == pytest.approx(np.max(seg.temperature_c))
    # The record length includes the post-cutoff rest: it exceeds the load duration.
    load_s = discharge_run.truth["t_load_off_s"] - discharge_run.truth["t_load_on_s"]
    assert leg["leg_record_time_s"] > load_s


def test_legacy_on_empty_record_keeps_only_the_dataset_capacity() -> None:
    leg = legacy_features(Segment("discharge", capacity_ah=1.8))
    assert leg["leg_capacity_ah"] == 1.8
    assert all(math.isnan(leg[k]) for k in LEGACY_FEATURES[1:])
    with pytest.raises(ValueError, match="discharge"):
        legacy_features(Segment("charge"))


# ----------------------------------------------------------------------------- ageing trends


@pytest.fixture(scope="module")
def ageing_rows(ageing_runs: AgeingRuns) -> list[dict[str, float]]:
    return [features_from_segments(c.segment, d.segment) for _, c, d in ageing_runs]


@pytest.fixture(scope="module")
def noisy_ageing_rows(
    ageing_runs: AgeingRuns, noisy: Callable[[Segment, int], Segment]
) -> list[dict[str, float]]:
    return [
        features_from_segments(noisy(c.segment, k), noisy(d.segment, 100 + k))
        for k, (_, c, d) in enumerate(ageing_runs)
    ]


TRENDS = [
    ("dis_q_380_350_ah", -1),
    ("dis_time_380_350_s", -1),
    ("chg_cc_time_390_419_s", -1),
    ("chg_cc_time_400_419_s", -1),
    ("chg_ica_peak_height_ah_per_v", -1),
    ("chg_ica_peak_voltage_v", 1),
    ("chg_cv_time_s", 1),
    ("dis_r0_ohm", 1),
]


@pytest.mark.parametrize(("name", "direction"), TRENDS)
def test_indicators_move_monotonically_as_the_cell_fades(
    ageing_rows: list[dict[str, float]], name: str, direction: int
) -> None:
    """SOH 100 -> 70 % in 2 % steps (capacity fades, resistance grows): strict trends."""
    values = np.array([row[name] for row in ageing_rows])
    assert np.isfinite(values).all()
    assert (np.diff(values) * direction > 0).all(), values


@pytest.mark.parametrize(("name", "direction"), TRENDS)
def test_trends_survive_sensor_noise(
    ageing_runs: AgeingRuns, noisy_ageing_rows: list[dict[str, float]], name: str, direction: int
) -> None:
    capacity = np.array([cell.capacity_ah for cell, _, _ in ageing_runs])
    values = np.array([row[name] for row in noisy_ageing_rows])
    ranks = np.argsort(np.argsort(values)), np.argsort(np.argsort(capacity))
    spearman = np.corrcoef(*ranks)[0, 1]
    assert -direction * spearman > 0.9


@pytest.mark.parametrize(
    "name", ["dis_q_380_350_ah", "chg_cc_time_390_419_s", "chg_ica_peak_height_ah_per_v"]
)
def test_capacity_indicators_track_capacity_linearly(
    ageing_runs: AgeingRuns, noisy_ageing_rows: list[dict[str, float]], name: str
) -> None:
    capacity = [cell.capacity_ah for cell, _, _ in ageing_runs]
    values = [row[name] for row in noisy_ageing_rows]
    assert np.corrcoef(capacity, values)[0, 1] > 0.99


def test_ica_peak_voltage_has_no_label_jumps(noisy_ageing_rows: list[dict[str, float]]) -> None:
    """Regression for v1 (peak labels swapped: 45 of 167 cycles jumped by > 0.1 V)."""
    v_peak = np.array([row["chg_ica_peak_voltage_v"] for row in noisy_ageing_rows])
    assert np.abs(np.diff(v_peak)).max() < 0.01
