"""Real NASA cells: the health indicators reproduce the audit's per-battery correlations.

Skipped automatically when the raw ``.mat`` files are absent (see ``tests/conftest.py``).
Reference values (audit prototype, per-battery Pearson r with capacity, QC-valid cycles):
dis_q 0.995-0.998, chg_cc_time_390 0.984-0.993, ica height 0.972-0.994, cv time < 0.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from battery_soh.data.nasa import find_mat_files, load_battery
from battery_soh.data.schema import BatteryRecord, Segment
from battery_soh.features.extract import features_from_segments
from battery_soh.features.registry import FEATURES
from battery_soh.qc import flag_capacity_outliers

pytestmark = pytest.mark.raw_data

CORE = ("B0005", "B0006", "B0007", "B0018")


def _cycle_table(record: BatteryRecord) -> pd.DataFrame:
    """One row per discharge, paired with the immediately preceding charge and the last EIS."""
    rows = []
    last_charge: Segment | None = None
    previous_kind: str | None = None
    eis: tuple[float, float] | None = None
    for seg in record.segments:
        if seg.kind == "impedance":
            if seg.re_ohm is not None and seg.rct_ohm is not None:
                eis = (seg.re_ohm, seg.rct_ohm)
            continue
        if seg.kind == "charge":
            last_charge = seg
        else:
            charge = last_charge if previous_kind == "charge" else None
            row = features_from_segments(charge, seg, eis=eis)
            row["capacity_ah"] = np.nan if seg.capacity_ah is None else seg.capacity_ah
            rows.append(row)
            last_charge = None
        previous_kind = seg.kind
    table = pd.DataFrame(rows)
    table["qc_valid"], _ = flag_capacity_outliers(table["capacity_ah"])
    return table


@pytest.fixture(scope="module", params=CORE)
def core_table(request: pytest.FixtureRequest, nasa_raw_dir: Path) -> pd.DataFrame:
    battery_id = request.param
    record = load_battery(find_mat_files(nasa_raw_dir)[battery_id], battery_id)
    table = _cycle_table(record)
    return table[table["qc_valid"]]


def _r(table: pd.DataFrame, name: str) -> float:
    return float(table[name].corr(table["capacity_ah"]))


def test_discharge_window_charge_tracks_capacity(core_table: pd.DataFrame) -> None:
    assert core_table["dis_q_380_350_ah"].isna().mean() <= 0.01
    assert _r(core_table, "dis_q_380_350_ah") > 0.99


def test_charge_indicators_track_capacity(core_table: pd.DataFrame) -> None:
    assert _r(core_table, "chg_cc_time_390_419_s") > 0.975
    assert _r(core_table, "chg_ica_peak_height_ah_per_v") > 0.96
    assert _r(core_table, "chg_ica_peak_voltage_v") < -0.85
    assert _r(core_table, "chg_cv_time_s") < -0.8


def test_resistance_indicators_grow_as_capacity_fades(core_table: pd.DataFrame) -> None:
    assert core_table["dis_r0_ohm"].notna().all()  # v1 returned 0.0 in 19-64 % of cycles
    assert _r(core_table, "dis_r0_ohm") < -0.6
    assert _r(core_table, "eis_rct_ohm") < 0


def test_real_values_are_inside_the_registry_ranges(core_table: pd.DataFrame) -> None:
    for name, spec in FEATURES.items():
        if name not in core_table:
            continue
        values = core_table[name].dropna()
        lo, hi = spec.plausible_range
        assert values.between(lo, hi).all(), name
    assert (core_table["ctx_discharge_current_a"].dropna().between(1.9, 2.1)).all()
