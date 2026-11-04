"""Feature-table builder: cycle pairing, EIS as-of join, labels, column contract, CSV I/O.

Every test runs on synthetic data (``battery_soh.synthetic`` segments, hand-made impedance
segments and NASA-format ``.mat`` files written with ``scipy.io.savemat``), so no raw data is
needed. Distinct charges are made by stretching the time axis of one simulated charge: a
factor ``s`` multiplies every CC window time by exactly ``s``, so a row's charge features
identify the charge it was paired with.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import math
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import scipy.io

from battery_soh import __version__
from battery_soh.data.nasa import load_battery
from battery_soh.data.schema import BatteryRecord, Segment
from battery_soh.features import build
from battery_soh.features.build import (
    FEATURE_TABLE_COLUMNS,
    UNREGISTERED_GROUP,
    battery_table,
    build_feature_table,
    cycle_rows,
    load_feature_table,
    metadata_path,
    write_feature_table,
)
from battery_soh.features.charge import CHARGE_FEATURES, charge_features
from battery_soh.features.extract import METADATA_FEATURES, features_from_segments
from battery_soh.features.registry import FEATURES
from battery_soh.labels import LABEL_COLUMNS as LABELS_MODULE_COLUMNS
from battery_soh.labels import label_battery
from battery_soh.synthetic import SyntheticSegment

#: The feature-table contract, spelled out as in the build specification.
SPEC_COLUMNS = (
    "battery_id",
    "group",
    "cycle_index",
    "start_time",
    "ambient_c",
    "discharge_current_a",
    "has_charge",
    "capacity_ah",
    "c0_ah",
    "soh_pct",
    "soh_rated_pct",
    "qc_valid",
    "qc_reason",
    *FEATURES,
)
T0 = datetime(2008, 4, 2, 15, 25, 41, 593000)


# --------------------------------------------------------------------------- builders


@pytest.fixture(scope="module")
def base_charge(charge_run: SyntheticSegment) -> Segment:
    return charge_run.segment


@pytest.fixture(scope="module")
def base_discharge(discharge_run: SyntheticSegment) -> Segment:
    return discharge_run.segment


def stretched(charge: Segment, factor: float) -> Segment:
    """The same charge with every time interval multiplied by ``factor``."""
    return dataclasses.replace(charge, time_s=charge.time_s * factor)


def discharge(
    base: Segment,
    capacity: float | None = 1.9,
    *,
    ambient: float | None = 24.0,
    start: datetime | None = None,
) -> Segment:
    return dataclasses.replace(base, capacity_ah=capacity, ambient_c=ambient, start_time=start)


def eis(re_ohm: float | None, rct_ohm: float | None) -> Segment:
    return Segment("impedance", re_ohm=re_ohm, rct_ohm=rct_ohm)


def record(segments: Sequence[Segment], battery_id: str = "B0005") -> BatteryRecord:
    return BatteryRecord(battery_id=battery_id, segments=tuple(segments))


def charge_values(row: dict[str, Any]) -> dict[str, float]:
    return {name: float(row[name]) for name in CHARGE_FEATURES}


def assert_same_features(actual: dict[str, float], expected: dict[str, float]) -> None:
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        if math.isnan(value):
            assert math.isnan(actual[name]), name
        else:
            assert actual[name] == value, name


# --------------------------------------------------------------------------- contract


def test_column_contract_matches_the_specification() -> None:
    assert FEATURE_TABLE_COLUMNS == SPEC_COLUMNS
    assert len(set(FEATURE_TABLE_COLUMNS)) == len(FEATURE_TABLE_COLUMNS)
    assert build.LABEL_COLUMNS == LABELS_MODULE_COLUMNS  # guards drift from battery_soh.labels


def test_battery_table_has_the_contract_columns_and_dtypes(
    base_charge: Segment, base_discharge: Segment
) -> None:
    table = battery_table(
        record([base_charge, discharge(base_discharge, start=T0), discharge(base_discharge)])
    )
    assert tuple(table.columns) == FEATURE_TABLE_COLUMNS
    assert table["cycle_index"].dtype == np.int64
    assert table["has_charge"].dtype == bool
    assert table["qc_valid"].dtype == bool
    assert str(table["start_time"].dtype) == "datetime64[ms]"
    assert pd.api.types.is_string_dtype(table["battery_id"])
    assert pd.api.types.is_string_dtype(table["qc_reason"])
    numeric = [c for c in FEATURE_TABLE_COLUMNS if c not in build.STRING_COLUMNS]
    numeric = [c for c in numeric if c not in (*build.BOOL_COLUMNS, "start_time", "cycle_index")]
    assert all(table[c].dtype == np.float64 for c in numeric)
    assert table["start_time"].iloc[0] == pd.Timestamp(T0)
    assert pd.isna(table["start_time"].iloc[1])


def test_cycle_rows_keys_and_types(base_charge: Segment, base_discharge: Segment) -> None:
    (row,) = cycle_rows(record([base_charge, discharge(base_discharge, start=T0)]))
    assert tuple(row) == build.CYCLE_ROW_COLUMNS
    assert row["battery_id"] == "B0005"
    assert row["cycle_index"] == 0
    assert isinstance(row["cycle_index"], int)
    assert row["has_charge"] is True
    assert row["start_time"] == T0
    assert all(isinstance(row[name], float) for name in FEATURES)


# --------------------------------------------------------------------------- pairing


def test_one_row_per_discharge_with_zero_based_cycle_index(
    base_charge: Segment, base_discharge: Segment
) -> None:
    segments = [base_charge, discharge(base_discharge)] * 3 + [base_charge]  # trailing charge
    rows = cycle_rows(record(segments))
    assert [r["cycle_index"] for r in rows] == [0, 1, 2]
    assert [r["ctx_cycle_index"] for r in rows] == [0.0, 1.0, 2.0]


def test_discharge_pairs_with_the_immediately_preceding_charge(
    base_charge: Segment, base_discharge: Segment
) -> None:
    c = {s: stretched(base_charge, s) for s in (1.0, 1.1, 1.2, 1.3, 1.4)}
    d = discharge(base_discharge)
    # fmt: off
    segments = [
        c[1.0], d,  # 0: plain pair
        d,  # 1: discharge after discharge -> no charge (never borrowed from c[1.0])
        c[1.1], c[1.2], d,  # 2: restarted charge -> the second one conditioned the cell
        c[1.3], eis(0.05, 0.08), d,  # 3: impedance in between does not break the pair
        eis(0.05, 0.08), c[1.4], eis(0.05, 0.08), d,  # 4: likewise on both sides
    ]
    # fmt: on
    rows = cycle_rows(record(segments))
    assert [r["has_charge"] for r in rows] == [True, False, True, True, True]
    for row, scale in zip(rows, (1.0, None, 1.2, 1.3, 1.4), strict=True):
        expected = (
            dict.fromkeys(CHARGE_FEATURES, math.nan) if scale is None else charge_features(c[scale])
        )
        assert_same_features(charge_values(row), expected)
    # The stretch factor really is recoverable from the paired row (the test is discriminating).
    base_time = charge_features(c[1.0])["chg_cc_time_390_419_s"]
    assert rows[2]["chg_cc_time_390_419_s"] == pytest.approx(1.2 * base_time, rel=1e-9)


def test_leading_discharge_and_empty_charge_have_no_charge(
    base_charge: Segment, base_discharge: Segment
) -> None:
    d = discharge(base_discharge)
    rows = cycle_rows(record([d, Segment("charge"), d, base_charge, d]))
    assert [r["has_charge"] for r in rows] == [False, False, True]
    for row in rows[:2]:
        assert all(math.isnan(v) for v in charge_values(row).values())
    assert not math.isnan(float(rows[2]["chg_cc_time_390_419_s"]))


def test_rows_equal_features_from_segments_train_serve_parity(
    base_charge: Segment, base_discharge: Segment
) -> None:
    c = stretched(base_charge, 1.05)
    d = discharge(base_discharge, 1.87, ambient=24.0)
    (row,) = cycle_rows(record([eis(0.051, 0.083), c, d]))
    expected = features_from_segments(c, d, eis=(0.051, 0.083))
    assert_same_features({k: row[k] for k in expected}, expected)
    assert row["discharge_current_a"] == row["ctx_discharge_current_a"]
    assert row["capacity_ah"] == row["leg_capacity_ah"] == 1.87
    assert set(FEATURES) - set(expected) == set(METADATA_FEATURES)


# --------------------------------------------------------------------------- EIS as-of


def test_eis_is_the_latest_prior_measurement_without_backfill(
    base_charge: Segment, base_discharge: Segment
) -> None:
    c, d = base_charge, discharge(base_discharge)
    # fmt: off
    segments = [
        c, d,  # 0: no measurement yet -> NaN even though later ones exist
        eis(0.050, 0.070),
        c, d,  # 1
        eis(0.060, 0.080), eis(0.061, 0.081),
        c, d,  # 2: the most recent of two
        c, eis(0.070, 0.090), d,  # 3: measured after the charge, before the discharge
        c, d,  # 4: carried forward (as-of), not reset
        eis(0.099, 0.099),  # after the last discharge: affects nothing
    ]
    # fmt: on
    rows = cycle_rows(record(segments))
    got = [(r["eis_re_ohm"], r["eis_rct_ohm"]) for r in rows]
    assert all(math.isnan(v) for v in got[0])
    assert got[1:] == [(0.050, 0.070), (0.061, 0.081), (0.070, 0.090), (0.070, 0.090)]


@pytest.mark.parametrize(
    "failed",
    [
        eis(1e15, 2e15),  # diverged fit (B0050/B0052 hold such values)
        eis(0.05, 5_000.0),  # one value outside the registry's plausible range
        eis(None, 0.08),  # missing value
        eis(-0.05, 0.08),  # non-positive value
        eis(float("nan"), 0.08),
        eis(float("inf"), 0.08),
    ],
    ids=["diverged", "out_of_range", "missing", "negative", "nan", "inf"],
)
def test_failed_eis_fits_are_skipped(
    failed: Segment, base_charge: Segment, base_discharge: Segment
) -> None:
    c, d = base_charge, discharge(base_discharge)
    rows = cycle_rows(record([eis(0.05, 0.07), c, d, failed, c, d, failed]))
    assert [(r["eis_re_ohm"], r["eis_rct_ohm"]) for r in rows] == [(0.05, 0.07)] * 2
    (only_failed,) = cycle_rows(record([failed, c, d]))
    assert math.isnan(only_failed["eis_re_ohm"])
    assert math.isnan(only_failed["eis_rct_ohm"])


# --------------------------------------------------------------------------- context


def test_ambient_comes_from_the_segment_with_protocol_fallback(
    base_charge: Segment, base_discharge: Segment
) -> None:
    c = base_charge
    segments = [
        c,
        discharge(base_discharge, ambient=30.0),
        c,
        discharge(base_discharge, ambient=None),
    ]
    rows = cycle_rows(record(segments, "B0029"))
    assert [r["ambient_c"] for r in rows] == [30.0, 43.0]  # B0029: 43 deg C campaign
    assert [r["ctx_ambient_c"] for r in rows] == [30.0, 43.0]
    (mixed,) = cycle_rows(record([c, discharge(base_discharge, ambient=None)], "B0038"))
    assert math.isnan(mixed["ambient_c"])  # mixed-temperature cell: no nominal value


def test_group_comes_from_the_protocol_registry(
    base_charge: Segment, base_discharge: Segment
) -> None:
    table = battery_table(record([base_charge, discharge(base_discharge)], "B0036"))
    assert table["group"].tolist() == ["ARC33-36_2A_24C"]


def test_unregistered_battery_is_built_with_a_warning(
    base_charge: Segment, base_discharge: Segment, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="battery_soh.features.build"):
        table = battery_table(
            record([base_charge, discharge(base_discharge, ambient=None)], "B0099")
        )
    assert table["group"].tolist() == [UNREGISTERED_GROUP]
    assert math.isnan(table["ambient_c"].iloc[0])
    assert "not in the protocol registry" in caplog.text


# --------------------------------------------------------------------------- labels


def test_labels_and_qc_are_computed_per_battery(
    base_charge: Segment, base_discharge: Segment
) -> None:
    capacities = [1.90, 1.89, 1.88, 0.10, 1.87, 1.86, 1.85, 1.84, None]
    segments: list[Segment] = []
    for cap in capacities:
        segments += [base_charge, discharge(base_discharge, cap)]
    table = battery_table(record(segments))

    assert table["qc_reason"].tolist() == ["ok"] * 3 + ["below_min"] + ["ok"] * 4 + ["missing"]
    assert table["qc_valid"].tolist() == [r == "ok" for r in table["qc_reason"]]
    c0 = float(np.median([1.90, 1.89, 1.88, 1.87, 1.86]))  # first five QC-valid capacities
    assert (table["c0_ah"] == c0).all()
    cap = table["capacity_ah"]
    np.testing.assert_allclose(table["soh_pct"], 100.0 * cap / c0, rtol=1e-12)
    np.testing.assert_allclose(table["soh_rated_pct"], 100.0 * cap / 2.0, rtol=1e-12)
    assert table["soh_pct"].iloc[2] == pytest.approx(100.0)
    assert math.isnan(table["soh_pct"].iloc[-1])
    expected = label_battery(pd.Series([np.nan if c is None else c for c in capacities]))
    pd.testing.assert_frame_equal(
        table[list(LABELS_MODULE_COLUMNS)].reset_index(drop=True),
        expected[list(LABELS_MODULE_COLUMNS)].astype(
            table[list(LABELS_MODULE_COLUMNS)].dtypes.to_dict()
        ),
    )


def test_record_without_discharges_gives_an_empty_contract_table(base_charge: Segment) -> None:
    table = battery_table(record([base_charge, eis(0.05, 0.07)]))
    assert table.empty
    assert tuple(table.columns) == FEATURE_TABLE_COLUMNS


# --------------------------------------------------------------------------- CSV I/O


@pytest.fixture(scope="module")
def small_table(base_charge: Segment, base_discharge: Segment) -> pd.DataFrame:
    """Two batteries, NaN features (no EIS / no charge), a missing start time, QC failures."""
    frames = []
    for battery_id, offset in (("B0018", 0.0), ("B0005", 0.01)):
        segments: list[Segment] = []
        for k, cap in enumerate((1.86, 1.85, 0.05, 1.84, 1.83, 1.82)):
            start = None if k == 4 else T0 + timedelta(hours=4 * k, milliseconds=125 * k)
            if k != 1:
                segments.append(stretched(base_charge, 1.0 + 0.01 * k))
            if k == 3:
                segments.append(eis(0.05 + offset, 0.08))
            segments.append(discharge(base_discharge, cap + offset, start=start))
        frames.append(battery_table(record(segments, battery_id)))
    return pd.concat(frames, ignore_index=True)


def test_write_is_deterministic_lf_six_digits_and_sorted(
    small_table: pd.DataFrame, tmp_path: Path
) -> None:
    first = write_feature_table(small_table, tmp_path / "a.csv")
    shuffled = small_table.sample(frac=1.0, random_state=0)
    second = write_feature_table(shuffled, tmp_path / "b.csv")
    data = first.read_bytes()
    assert data == second.read_bytes()
    assert b"\r" not in data
    lines = data.decode("utf-8").splitlines()
    assert lines[0] == ",".join(FEATURE_TABLE_COLUMNS)
    keys = [tuple(line.split(",")[:3:2]) for line in lines[1:]]
    assert keys == sorted(keys, key=lambda k: (k[0], int(k[1])))
    assert keys[0] == ("B0005", "0")

    header = lines[0].split(",")
    row = dict(zip(header, lines[1].split(","), strict=True))
    source = small_table[small_table["battery_id"] == "B0005"].iloc[0]
    assert row["chg_cc_time_390_419_s"] == f"{source['chg_cc_time_390_419_s']:.6g}"
    assert row["start_time"] == "2008-04-02T15:25:41.593"
    assert row["eis_re_ohm"] == ""  # NaN is an empty field
    assert row["has_charge"] == "True"
    assert len(row["soh_pct"].replace(".", "").replace("-", "").lstrip("0")) <= 6


def test_sidecar_records_hash_counts_and_nan_rates(
    small_table: pd.DataFrame, tmp_path: Path
) -> None:
    path = write_feature_table(small_table, tmp_path / "features.csv")
    meta_file = metadata_path(path)
    assert meta_file == tmp_path / "features.meta.json"
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    assert meta["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert meta["bytes"] == path.stat().st_size
    assert meta["package_version"] == __version__
    assert meta["n_rows"] == len(small_table)
    assert meta["columns"] == list(FEATURE_TABLE_COLUMNS)
    assert meta["row_counts"] == {"B0005": 6, "B0018": 6}
    assert meta["qc_valid_counts"] == {"B0005": 5, "B0018": 5}
    assert set(meta["nan_rate"]) == set(FEATURES)
    assert meta["nan_rate"]["eis_re_ohm"] == pytest.approx(
        small_table["eis_re_ohm"].isna().mean(), abs=1e-4
    )
    assert meta["nan_rate"]["dis_q_380_350_ah"] == 0.0
    created = datetime.fromisoformat(meta["created_utc"])
    assert created.utcoffset() == timedelta(0)


def test_round_trip_restores_dtypes_and_values(small_table: pd.DataFrame, tmp_path: Path) -> None:
    path = write_feature_table(small_table, tmp_path / "features.csv")
    loaded = load_feature_table(path)
    expected = small_table.sort_values(["battery_id", "cycle_index"], ignore_index=True)
    assert tuple(loaded.columns) == FEATURE_TABLE_COLUMNS
    assert loaded.dtypes.to_dict() == expected.dtypes.to_dict()
    exact = [*build.STRING_COLUMNS, *build.BOOL_COLUMNS, "cycle_index", "start_time"]
    pd.testing.assert_frame_equal(loaded[exact], expected[exact])
    floats = [c for c in FEATURE_TABLE_COLUMNS if c not in exact]
    np.testing.assert_array_equal(loaded[floats].isna(), expected[floats].isna())
    np.testing.assert_allclose(loaded[floats], expected[floats], rtol=5e-6, equal_nan=True)
    assert pd.isna(loaded.loc[4, "start_time"])


def test_write_rejects_tables_that_break_the_contract(
    small_table: pd.DataFrame, tmp_path: Path
) -> None:
    target = tmp_path / "x.csv"
    with pytest.raises(ValueError, match="missing"):
        write_feature_table(small_table.drop(columns=["soh_pct"]), target)
    with pytest.raises(ValueError, match="extra"):
        write_feature_table(small_table.assign(is_reference_cycle=False), target)
    with pytest.raises(ValueError, match="duplicate"):
        write_feature_table(pd.concat([small_table, small_table.iloc[:1]]), target)
    assert not target.exists()


def test_load_rejects_missing_files_and_foreign_headers(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="soh data build"):
        load_feature_table(tmp_path / "absent.csv")
    stale = tmp_path / "stale.csv"
    stale.write_text("battery_id,cycle_index,soh_pct\nB0005,0,100\n", encoding="utf-8")
    with pytest.raises(ValueError, match="contract"):
        load_feature_table(stale)


def test_load_rejects_non_boolean_flags(small_table: pd.DataFrame, tmp_path: Path) -> None:
    path = write_feature_table(small_table, tmp_path / "features.csv")
    text = path.read_text(encoding="utf-8").replace(",True,", ",yes,", 1)
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="non-boolean"):
        load_feature_table(path)


def test_load_defaults_to_the_project_feature_table(
    small_table: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write_feature_table(small_table, tmp_path / "data" / "processed" / "nasa_features.csv")

    class _Paths:
        features_csv = path

    monkeypatch.setattr(build, "get_paths", lambda: _Paths())
    assert len(load_feature_table()) == len(small_table)


# --------------------------------------------------------------------------- end to end


def _write_nasa_mat(path: Path, battery_id: str, segments: Sequence[Segment]) -> None:
    """Write segments in the layout of the official archive (see test_data_nasa.py)."""
    cycle = np.empty(
        (1, len(segments)),
        dtype=[
            ("type", object),
            ("ambient_temperature", object),
            ("time", object),
            ("data", object),
        ],
    )
    for i, seg in enumerate(segments):
        start = seg.start_time or T0
        seconds = start.second + start.microsecond / 1e6
        cycle["type"][0, i] = np.array([seg.kind])
        cycle["ambient_temperature"][0, i] = np.array([[24]], dtype=np.uint8)
        cycle["time"][0, i] = np.array(
            [[start.year, start.month, start.day, start.hour, start.minute, seconds]], dtype=float
        )
        if seg.kind == "impedance":
            fields = {"Re": np.array([[seg.re_ohm]]), "Rct": np.array([[seg.rct_ohm]])}
        else:
            fields = {
                "Voltage_measured": seg.voltage_v[None, :],
                "Current_measured": seg.current_a[None, :],
                "Temperature_measured": seg.temperature_c[None, :],
                "Time": seg.time_s[None, :],
            }
            if seg.kind == "discharge":
                fields["Capacity"] = np.array([[seg.capacity_ah]])
        data = np.empty((1, 1), dtype=[(name, object) for name in fields])
        for name, value in fields.items():
            data[name][0, 0] = value
        cycle["data"][0, i] = data
    top = np.empty((1, 1), dtype=[("cycle", object)])
    top["cycle"][0, 0] = cycle
    path.parent.mkdir(parents=True, exist_ok=True)
    scipy.io.savemat(str(path), {battery_id: top})


@pytest.fixture(scope="module")
def raw_tree(
    tmp_path_factory: pytest.TempPathFactory, base_charge: Segment, base_discharge: Segment
) -> Path:
    root = tmp_path_factory.mktemp("nasa")
    for battery_id, folder in (("B0018", "1. FY08Q4"), ("B0005", "1. FY08Q4"), ("B0029", "3. ARC")):
        segments: list[Segment] = []
        for k in range(4):
            start = T0 + timedelta(hours=3 * k)
            segments += [
                dataclasses.replace(stretched(base_charge, 1.0 + 0.02 * k), start_time=start),
                eis(0.05 + 0.001 * k, 0.08),
                discharge(base_discharge, 1.9 - 0.01 * k, start=start + timedelta(hours=1)),
            ]
        _write_nasa_mat(root / folder / f"{battery_id}.mat", battery_id, segments)
    return root


def test_build_from_mat_files_is_sorted_and_matches_battery_table(raw_tree: Path) -> None:
    table = build_feature_table(raw_tree)
    assert tuple(table.columns) == FEATURE_TABLE_COLUMNS
    assert table["battery_id"].unique().tolist() == ["B0005", "B0018", "B0029"]
    assert table["cycle_index"].tolist() == [0, 1, 2, 3] * 3
    assert table.index.equals(pd.RangeIndex(12))
    assert table.loc[table["battery_id"] == "B0029", "group"].unique().tolist() == [
        "ARC29-32_4A_43C"
    ]
    expected = battery_table(load_battery(raw_tree / "1. FY08Q4" / "B0018.mat"))
    actual = table[table["battery_id"] == "B0018"].reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected)
    assert table["has_charge"].all()
    assert (table["eis_re_ohm"].notna()).all()  # every discharge follows an EIS here


def test_build_subset_and_errors(raw_tree: Path, tmp_path: Path) -> None:
    subset = build_feature_table(raw_tree, ["b18", "B0005"])
    assert subset["battery_id"].unique().tolist() == ["B0005", "B0018"]
    with pytest.raises(FileNotFoundError):
        build_feature_table(raw_tree, ["B0006"])
    with pytest.raises(ValueError, match="must not be empty"):
        build_feature_table(raw_tree, [])
    with pytest.raises(FileNotFoundError):
        build_feature_table(tmp_path / "missing")
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="no NASA battery files"):
        build_feature_table(tmp_path / "empty")


def test_build_write_load_end_to_end(raw_tree: Path, tmp_path: Path) -> None:
    table = build_feature_table(raw_tree)
    path = write_feature_table(table, tmp_path / "nasa_features.csv")
    loaded = load_feature_table(path)
    assert loaded[["battery_id", "cycle_index"]].equals(table[["battery_id", "cycle_index"]])
    np.testing.assert_allclose(loaded["soh_pct"], table["soh_pct"], rtol=5e-6)
    assert loaded["start_time"].iloc[0] == pd.Timestamp(T0 + timedelta(hours=1))
