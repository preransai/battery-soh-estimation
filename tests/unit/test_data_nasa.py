"""Tests for the NASA ``.mat`` reader.

The loader tests run on synthetic files written with ``scipy.io.savemat`` in the exact layout
of the official archive (``mat[bid][0, 0]["cycle"][0, i]["data"][0, 0][field]``, verified
against a real file by ``test_synthetic_layout_matches_real_file``), so they need no raw data.
Tests marked ``raw_data`` check the real files and are skipped when they are absent.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import scipy.io

from battery_soh.config import get_paths
from battery_soh.data.nasa import (
    NASA_BATTERY_IDS,
    MatFormatError,
    datevec_to_datetime,
    find_mat_files,
    iter_batteries,
    load_battery,
    normalize_battery_id,
)
from battery_soh.data.schema import BatteryRecord

# ------------------------------------------------------------------ synthetic NASA .mat files

_CHARGE_EXTRA = ("Current_charge", "Voltage_charge")
_DISCHARGE_EXTRA = ("Current_load", "Voltage_load")
_EIS_SPECTRA = (
    "Sense_current",
    "Battery_current",
    "Current_ratio",
    "Battery_impedance",
    "Rectified_Impedance",
)


@dataclass
class Op:
    """One operation to write into a synthetic NASA file."""

    kind: str
    datevec: Sequence[float] = (2008.0, 4.0, 2.0, 13.0, 8.0, 17.921)
    ambient: int = 24
    time: Sequence[float] = ()
    voltage: Sequence[float] = ()
    current: Sequence[float] = ()
    temperature: Sequence[float] = ()
    capacity: float | None = None
    re: complex | float = 0.05
    rct: complex | float = 0.08
    extra_fields: Mapping[str, Any] = field(default_factory=dict)
    drop_fields: Sequence[str] = ()


def _row(values: Sequence[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(1, -1)


def _struct(fields: Mapping[str, Any]) -> np.ndarray:
    out = np.empty((1, 1), dtype=[(name, object) for name in fields])
    for name, value in fields.items():
        out[name][0, 0] = value
    return out


def _data_struct(op: Op) -> np.ndarray:
    if op.kind == "impedance":
        spectrum = np.array([[0.07 - 0.001j, 0.068 - 0.002j]])
        fields: dict[str, Any] = dict.fromkeys(_EIS_SPECTRA, spectrum)
        fields["Re"] = np.array([[op.re]])
        fields["Rct"] = np.array([[op.rct]])
    else:
        n = len(op.time)
        fields = {
            "Voltage_measured": _row(op.voltage),
            "Current_measured": _row(op.current),
            "Temperature_measured": _row(op.temperature),
        }
        extra = _DISCHARGE_EXTRA if op.kind == "discharge" else _CHARGE_EXTRA
        for name in extra:
            fields[name] = np.zeros((1, n))
        fields["Time"] = _row(op.time)
        if op.kind == "discharge":
            # Real files store a missing capacity as an empty uint8 matrix.
            fields["Capacity"] = (
                np.zeros((1, 0), dtype=np.uint8)
                if op.capacity is None
                else np.array([[op.capacity]])
            )
    fields.update(op.extra_fields)
    return _struct({k: v for k, v in fields.items() if k not in op.drop_fields})


def write_nasa_mat(path: Path, variable: str, ops: Sequence[Op]) -> Path:
    """Write ``ops`` as a NASA-format ``.mat`` file holding one struct named ``variable``."""
    cycle = np.empty(
        (1, len(ops)),
        dtype=[
            ("type", object),
            ("ambient_temperature", object),
            ("time", object),
            ("data", object),
        ],
    )
    for i, op in enumerate(ops):
        cycle["type"][0, i] = np.array([op.kind])
        cycle["ambient_temperature"][0, i] = np.array([[op.ambient]], dtype=np.uint8)
        cycle["time"][0, i] = _row(op.datevec)
        cycle["data"][0, i] = _data_struct(op)
    top = np.empty((1, 1), dtype=[("cycle", object)])
    top["cycle"][0, 0] = cycle
    path.parent.mkdir(parents=True, exist_ok=True)
    scipy.io.savemat(str(path), {variable: top})
    return path


def charge_op(n: int = 50, *, start_v: float = 3.6, minute: float = 8.0) -> Op:
    t = np.linspace(0.0, 3600.0, n)
    return Op(
        kind="charge",
        datevec=(2008.0, 4.0, 2.0, 13.0, minute, 17.921),
        time=t,
        voltage=np.linspace(start_v, 4.2, n),
        current=np.full(n, 1.5),
        temperature=np.linspace(24.0, 26.0, n),
    )


def discharge_op(capacity: float | None = 1.85, n: int = 40, *, minute: float = 25.0) -> Op:
    t = np.linspace(0.0, 3300.0, n)
    return Op(
        kind="discharge",
        datevec=(2008.0, 4.0, 2.0, 15.0, minute, 41.593),
        time=t,
        voltage=np.linspace(4.19, 2.7, n),
        current=np.full(n, -2.0),
        temperature=np.linspace(24.0, 38.0, n),
        capacity=capacity,
    )


def impedance_op(re: complex | float = 0.05, rct: complex | float = 0.08) -> Op:
    return Op(kind="impedance", datevec=(2008.0, 4.0, 18.0, 20.0, 55.0, 29.859), re=re, rct=rct)


def standard_ops() -> list[Op]:
    """charge, discharge, impedance, charge, discharge (with a complex/negative EIS fit)."""
    return [
        charge_op(),
        discharge_op(1.856),
        impedance_op(re=0.0446 + 0.001j, rct=-0.0695),
        charge_op(start_v=3.5, minute=40.0),
        discharge_op(1.846, minute=50.0),
    ]


@pytest.fixture
def mat_file(tmp_path: Path) -> Path:
    return write_nasa_mat(tmp_path / "B0005.mat", "B0005", standard_ops())


# ----------------------------------------------------------------------- raw-data plumbing

_RAW_DIR = get_paths().nasa_raw
_HAS_RAW = _RAW_DIR.is_dir() and any(_RAW_DIR.rglob("*.mat"))
requires_raw = pytest.mark.skipif(not _HAS_RAW, reason=f"no NASA .mat files under {_RAW_DIR}")


@pytest.fixture(scope="module")
def raw_files() -> dict[str, Path]:
    return find_mat_files(_RAW_DIR)


@pytest.fixture(scope="module")
def core_records(raw_files: dict[str, Path]) -> dict[str, BatteryRecord]:
    return {b: load_battery(raw_files[b]) for b in ("B0005", "B0006", "B0007", "B0018")}


# ------------------------------------------------------------------------- identifiers


def test_nasa_battery_ids_are_the_34_cells_in_order() -> None:
    assert len(NASA_BATTERY_IDS) == 34
    assert len(set(NASA_BATTERY_IDS)) == 34
    assert list(NASA_BATTERY_IDS) == sorted(NASA_BATTERY_IDS)
    assert NASA_BATTERY_IDS[:4] == ("B0005", "B0006", "B0007", "B0018")
    assert "B0035" not in NASA_BATTERY_IDS
    assert "B0037" not in NASA_BATTERY_IDS
    assert NASA_BATTERY_IDS[-1] == "B0056"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("B0005", "B0005"),
        ("b0005", "B0005"),
        ("B5", "B0005"),
        ("5", "B0005"),
        (" B56 ", "B0056"),
        ("B00005", "B0005"),
    ],
)
def test_normalize_battery_id(raw: str, expected: str) -> None:
    assert normalize_battery_id(raw) == expected


@pytest.mark.parametrize("raw", ["", "B", "X0005", "B12345", "B-5", "B0005.mat"])
def test_normalize_battery_id_rejects_garbage(raw: str) -> None:
    with pytest.raises(ValueError, match="not a NASA battery id"):
        normalize_battery_id(raw)


# ------------------------------------------------------------------------------ datevec


def test_datevec_basic_with_fractional_seconds() -> None:
    assert datevec_to_datetime([2008, 4, 2, 13, 8, 17.921]) == datetime(
        2008, 4, 2, 13, 8, 17, 921000
    )


def test_datevec_accepts_matlab_row_matrix() -> None:
    vec = np.array([[2010.0, 9.0, 3.0, 12.0, 10.0, 27.156]])
    assert datevec_to_datetime(vec) == datetime(2010, 9, 3, 12, 10, 27, 156000)


def test_datevec_rolls_over_sixty_seconds() -> None:
    assert datevec_to_datetime([2008, 12, 31, 23, 59, 60.0]) == datetime(2009, 1, 1)


@pytest.mark.parametrize(
    "vec",
    [
        [2008, 4, 2, 13, 8],  # too short
        [2008, 4, 2, 13, 8, 1, 2],  # too long
        [2008, 4, 2, 13, float("nan"), 1],  # non-finite
        [2008, 13, 2, 0, 0, 0],  # month out of range
        [2008, 2, 30, 0, 0, 0],  # day out of range
        "not a date",
        None,
        [],
    ],
)
def test_datevec_invalid_returns_none(vec: object) -> None:
    assert datevec_to_datetime(vec) is None


# --------------------------------------------------------------------------- find_mat_files


def _touch(path: Path, payload: bytes = b"MATLAB 5.0 MAT-file stub") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def test_find_mat_files_recursive_and_filtered(tmp_path: Path) -> None:
    _touch(tmp_path / "1. FY08Q4" / "B0005.mat")
    _touch(tmp_path / "1. FY08Q4" / "b0006.MAT")
    _touch(tmp_path / "1. FY08Q4" / "._B0007.mat")  # macOS resource fork
    _touch(tmp_path / "1. FY08Q4" / "notes.mat")
    _touch(tmp_path / "1. FY08Q4" / "README.txt")
    _touch(tmp_path / "deep" / "er" / "B0018.mat")
    _touch(tmp_path / ".staging" / "B0029.mat")  # hidden folder (interrupted download)
    found = find_mat_files(tmp_path)
    assert list(found) == ["B0005", "B0006", "B0018"]
    assert found["B0006"].name == "b0006.MAT"
    assert found["B0018"] == tmp_path / "deep" / "er" / "B0018.mat"


def test_find_mat_files_dedups_identical_copies_keeping_sorted_first(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    first = _touch(tmp_path / "2. ARC_25_28" / "B0025.mat", b"same bytes")
    _touch(tmp_path / "3. ARC_25-44" / "B0025.mat", b"same bytes")
    with caplog.at_level(logging.DEBUG, logger="battery_soh.data.nasa"):
        found = find_mat_files(tmp_path)
    assert found == {"B0025": first}
    records = [r for r in caplog.records if "Duplicate B0025" in r.getMessage()]
    assert len(records) == 1
    assert records[0].levelno == logging.DEBUG
    assert "byte-identical" in records[0].getMessage()


def test_find_mat_files_warns_on_conflicting_copies(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    first = _touch(tmp_path / "a" / "B0026.mat", b"version one")
    _touch(tmp_path / "b" / "B0026.mat", b"version two")
    with caplog.at_level(logging.WARNING, logger="battery_soh.data.nasa"):
        found = find_mat_files(tmp_path)
    assert found == {"B0026": first}
    assert any("Conflicting copies of B0026" in r.getMessage() for r in caplog.records)


def test_find_mat_files_is_deterministic(tmp_path: Path) -> None:
    for folder in ("z", "a", "m"):
        _touch(tmp_path / folder / "B0027.mat", b"x")
    assert find_mat_files(tmp_path)["B0027"] == tmp_path / "a" / "B0027.mat"


def test_find_mat_files_missing_root(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="soh data download"):
        find_mat_files(tmp_path / "nope")


def test_find_mat_files_empty_dir(tmp_path: Path) -> None:
    assert find_mat_files(tmp_path) == {}


# ----------------------------------------------------------------------------- load_battery


def test_load_battery_parses_every_operation_in_order(mat_file: Path) -> None:
    record = load_battery(mat_file)
    assert record.battery_id == "B0005"
    assert record.source_path == mat_file
    assert [s.kind for s in record.segments] == [
        "charge",
        "discharge",
        "impedance",
        "charge",
        "discharge",
    ]
    assert [s.capacity_ah for s in record.of_kind("discharge")] == pytest.approx([1.856, 1.846])
    assert all(s.capacity_ah is None for s in record.of_kind("charge"))


def test_load_battery_arrays_metadata_and_types(mat_file: Path) -> None:
    record = load_battery(mat_file)
    expected = discharge_op(1.856)
    seg = record.segments[1]
    np.testing.assert_array_equal(seg.time_s, np.asarray(expected.time))
    np.testing.assert_array_equal(seg.voltage_v, np.asarray(expected.voltage))
    np.testing.assert_array_equal(seg.current_a, np.asarray(expected.current))
    np.testing.assert_array_equal(seg.temperature_c, np.asarray(expected.temperature))
    for arr in (seg.time_s, seg.voltage_v, seg.current_a, seg.temperature_c):
        assert arr.dtype == np.float64
        assert arr.ndim == 1
    assert seg.start_time == datetime(2008, 4, 2, 15, 25, 41, 593000)
    assert seg.ambient_c == 24.0
    assert isinstance(seg.ambient_c, float)  # stored as uint8 in the file


def test_load_battery_sign_convention_is_preserved(mat_file: Path) -> None:
    record = load_battery(mat_file)
    assert all(np.all(s.current_a < 0) for s in record.of_kind("discharge"))
    assert all(np.all(s.current_a > 0) for s in record.of_kind("charge"))


def test_load_battery_impedance_is_abs_real(mat_file: Path) -> None:
    eis = load_battery(mat_file).of_kind("impedance")
    assert len(eis) == 1
    assert eis[0].re_ohm == pytest.approx(0.0446)  # complex fit -> real part
    assert eis[0].rct_ohm == pytest.approx(0.0695)  # negative fit -> absolute value
    assert eis[0].n_samples == 0
    assert eis[0].capacity_ah is None


def test_load_battery_empty_capacity_becomes_none(tmp_path: Path) -> None:
    path = write_nasa_mat(tmp_path / "B0052.mat", "B0052", [charge_op(), discharge_op(None)])
    assert load_battery(path).of_kind("discharge")[0].capacity_ah is None


def test_load_battery_drops_non_finite_samples_jointly(tmp_path: Path) -> None:
    op = charge_op(n=10)
    voltage = np.asarray(op.voltage, dtype=float).copy()
    temperature = np.asarray(op.temperature, dtype=float).copy()
    voltage[3] = np.nan
    temperature[7] = np.inf
    op.voltage, op.temperature = voltage, temperature
    path = write_nasa_mat(tmp_path / "B0018.mat", "B0018", [op])
    seg = load_battery(path).segments[0]
    keep = np.array([i not in (3, 7) for i in range(10)])
    assert seg.n_samples == 8
    np.testing.assert_array_equal(seg.time_s, np.asarray(op.time)[keep])
    np.testing.assert_array_equal(seg.voltage_v, voltage[keep])
    assert np.all(np.isfinite(seg.temperature_c))


def test_load_battery_skips_unknown_operation_types(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    ops = [charge_op(), Op(kind="rest", time=[0.0], voltage=[4.1], current=[0.0], temperature=[24])]
    path = write_nasa_mat(tmp_path / "B0006.mat", "B0006", ops)
    with caplog.at_level(logging.WARNING, logger="battery_soh.data.nasa"):
        record = load_battery(path)
    assert [s.kind for s in record.segments] == ["charge"]
    assert any("unknown type 'rest'" in r.getMessage() for r in caplog.records)


def test_load_battery_uses_explicit_id_and_single_variable(tmp_path: Path) -> None:
    path = write_nasa_mat(tmp_path / "renamed.mat", "B0007", [charge_op()])
    assert load_battery(path).battery_id == "B0007"  # taken from the variable name
    assert load_battery(path, "b7").battery_id == "B0007"  # normalised explicit id


def test_load_battery_variable_lookup_is_case_insensitive(tmp_path: Path) -> None:
    path = write_nasa_mat(tmp_path / "B0029.mat", "b0029", [charge_op()])
    assert load_battery(path).battery_id == "B0029"


def test_load_battery_empty_cycle_array(tmp_path: Path) -> None:
    record = load_battery(write_nasa_mat(tmp_path / "B0031.mat", "B0031", []))
    assert record.segments == ()


def test_load_battery_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="battery file not found"):
        load_battery(tmp_path / "B0005.mat")


@pytest.mark.parametrize(
    "payload", [b"", b"this is not a MATLAB file " * 8], ids=["empty", "garbage"]
)
def test_load_battery_unreadable_file(tmp_path: Path, payload: bytes) -> None:
    path = _touch(tmp_path / "B0005.mat", payload)
    with pytest.raises(MatFormatError, match="cannot read MATLAB file"):
        load_battery(path)


def test_load_battery_truncated_file(tmp_path: Path, mat_file: Path) -> None:
    truncated = _touch(tmp_path / "trunc" / "B0005.mat", mat_file.read_bytes()[:300])
    with pytest.raises(MatFormatError):
        load_battery(truncated)


def test_load_battery_rejects_ambiguous_variables(tmp_path: Path) -> None:
    path = tmp_path / "B0005.mat"
    scipy.io.savemat(str(path), {"foo": np.ones(2), "bar": np.ones(2)})
    with pytest.raises(MatFormatError, match="no variable named 'B0005'"):
        load_battery(path)


def test_load_battery_rejects_non_struct(tmp_path: Path) -> None:
    path = tmp_path / "B0005.mat"
    scipy.io.savemat(str(path), {"B0005": np.ones((3, 3))})
    with pytest.raises(MatFormatError, match="not a MATLAB struct"):
        load_battery(path)


def test_load_battery_rejects_missing_cycle_field(tmp_path: Path) -> None:
    path = tmp_path / "B0005.mat"
    scipy.io.savemat(str(path), {"B0005": _struct({"other": np.ones((1, 1))})})
    with pytest.raises(MatFormatError, match="missing field 'cycle'"):
        load_battery(path)


def test_load_battery_rejects_missing_measurement(tmp_path: Path) -> None:
    op = discharge_op()
    op.drop_fields = ("Voltage_measured",)
    path = write_nasa_mat(tmp_path / "B0005.mat", "B0005", [op])
    with pytest.raises(MatFormatError, match="B0005 operation 0: missing field 'Voltage_measured'"):
        load_battery(path)


def test_load_battery_rejects_unequal_array_lengths(tmp_path: Path) -> None:
    op = discharge_op(n=10)
    op.voltage = np.linspace(4.2, 3.0, 9)
    path = write_nasa_mat(tmp_path / "B0005.mat", "B0005", [op])
    with pytest.raises(MatFormatError, match="unequal lengths"):
        load_battery(path)


# ---------------------------------------------------------------------------- iter_batteries


@pytest.fixture
def small_tree(tmp_path: Path) -> Path:
    for folder, battery_id in (("1", "B0005"), ("1", "B0006"), ("2", "B0018")):
        write_nasa_mat(tmp_path / folder / f"{battery_id}.mat", battery_id, standard_ops())
    return tmp_path


def test_iter_batteries_all_sorted(small_tree: Path) -> None:
    assert [r.battery_id for r in iter_batteries(small_tree)] == ["B0005", "B0006", "B0018"]


def test_iter_batteries_requested_order_normalised_and_deduplicated(small_tree: Path) -> None:
    records = list(iter_batteries(small_tree, ["b18", "B0005", "18"]))
    assert [r.battery_id for r in records] == ["B0018", "B0005"]


def test_iter_batteries_missing_id_fails_eagerly(small_tree: Path) -> None:
    with pytest.raises(FileNotFoundError, match="B0007"):
        iter_batteries(small_tree, ["B0005", "B0007"])  # raised before iteration


def test_iter_batteries_is_lazy(small_tree: Path) -> None:
    iterator: Iterator[BatteryRecord] = iter_batteries(small_tree)
    (small_tree / "2" / "B0018.mat").unlink()  # not loaded yet, so this is not an error yet
    assert next(iterator).battery_id == "B0005"
    assert next(iterator).battery_id == "B0006"
    with pytest.raises(FileNotFoundError):
        next(iterator)


# --------------------------------------------------------------------------- real NASA files


@pytest.mark.raw_data
@requires_raw
def test_real_archive_has_the_34_batteries(raw_files: dict[str, Path]) -> None:
    assert tuple(raw_files) == NASA_BATTERY_IDS
    # B0025-B0028 ship twice; the copy in the dedicated campaign folder wins.
    assert raw_files["B0025"].parent.name.startswith("2.")


@pytest.mark.raw_data
@requires_raw
def test_synthetic_layout_matches_real_file(raw_files: dict[str, Path], tmp_path: Path) -> None:
    """The synthetic writer must reproduce the real nesting and field names."""
    real = scipy.io.loadmat(str(raw_files["B0005"]))["B0005"]
    fake_path = write_nasa_mat(tmp_path / "B0005.mat", "B0005", standard_ops())
    fake = scipy.io.loadmat(str(fake_path))["B0005"]
    assert real.dtype.names == fake.dtype.names == ("cycle",)
    assert real.shape == fake.shape == (1, 1)
    real_cycle, fake_cycle = real[0, 0]["cycle"], fake[0, 0]["cycle"]
    assert real_cycle.dtype.names == fake_cycle.dtype.names
    assert real_cycle.shape[0] == fake_cycle.shape[0] == 1

    def fields_by_kind(cycle: np.ndarray) -> dict[str, tuple[str, ...]]:
        out: dict[str, tuple[str, ...]] = {}
        for op in cycle[0]:
            out.setdefault(str(op["type"][0]), op["data"].dtype.names)
        return out

    assert fields_by_kind(real_cycle) == fields_by_kind(fake_cycle)
    real_op, fake_op = real_cycle[0, 1], fake_cycle[0, 1]
    for name in ("type", "ambient_temperature", "time"):
        assert real_op[name].dtype.kind == fake_op[name].dtype.kind, name
        assert real_op[name].shape == fake_op[name].shape, name


@pytest.mark.raw_data
@requires_raw
@pytest.mark.parametrize(
    ("battery_id", "n_discharge"), [("B0005", 168), ("B0006", 168), ("B0007", 168), ("B0018", 132)]
)
def test_core_cycle_counts(
    core_records: dict[str, BatteryRecord], battery_id: str, n_discharge: int
) -> None:
    record = core_records[battery_id]
    assert len(record.of_kind("discharge")) == n_discharge
    assert all(s.capacity_ah is not None for s in record.of_kind("discharge"))


@pytest.mark.raw_data
@requires_raw
def test_core_sign_convention_and_timing(core_records: dict[str, BatteryRecord]) -> None:
    for record in core_records.values():
        for seg in record.of_kind("discharge"):
            loaded = seg.current_a[np.abs(seg.current_a) > 0.5]
            assert loaded.size > 0
            assert np.all(loaded < 0), record.battery_id  # discharge current is negative
            assert np.median(loaded) == pytest.approx(-2.0, abs=0.05)
            assert np.all(np.diff(seg.time_s) >= 0)
        cc_currents = [
            np.median(seg.current_a[seg.current_a > 1.0])
            for seg in record.of_kind("charge")
            if np.any(seg.current_a > 1.0)
        ]
        assert len(cc_currents) > 100
        assert np.median(cc_currents) == pytest.approx(1.5, abs=0.05)  # CC charge is positive
        starts = [s.start_time for s in record.segments]
        assert all(t is not None for t in starts)
        assert starts == sorted(starts)
        assert {s.ambient_c for s in record.segments} == {24.0}


@pytest.mark.raw_data
@requires_raw
def test_real_impedance_values_are_finite_positive(core_records: dict[str, BatteryRecord]) -> None:
    eis = core_records["B0005"].of_kind("impedance")
    assert len(eis) == 278
    re = np.array([s.re_ohm for s in eis], dtype=float)
    rct = np.array([s.rct_ohm for s in eis], dtype=float)
    assert np.all(np.isfinite(re))
    assert np.all(re > 0)
    assert np.all(rct > 0)
    assert 0.02 < np.median(re) < 0.2  # tens of milliohms for an 18650 cell


@pytest.mark.raw_data
@requires_raw
def test_real_duplicates_are_identical_and_deduplicated(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG, logger="battery_soh.data.nasa"):
        found = find_mat_files(_RAW_DIR)
    duplicates = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Duplicate")]
    assert sorted(m.split(":")[0] for m in duplicates) == [
        "Duplicate B0025",
        "Duplicate B0026",
        "Duplicate B0027",
        "Duplicate B0028",
    ]
    assert all("byte-identical" in m for m in duplicates)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(found) == 34
