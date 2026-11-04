"""Tests for the battery protocol registry and the usability rule."""

from __future__ import annotations

import dataclasses
import re

import numpy as np
import pandas as pd
import pytest

from battery_soh.config import get_paths
from battery_soh.data.nasa import NASA_BATTERY_IDS
from battery_soh.data.protocols import (
    CORE_BATTERIES,
    EXTENDED_BATTERIES,
    MAX_LABEL_NOISE_PCT,
    MIN_VALID_CYCLES,
    PROTOCOLS,
    USABILITY_CRITERIA,
    BatteryEvidence,
    BatteryProtocol,
    assess_usability,
    battery_evidence,
    evidence_table,
    get_protocol,
)
from battery_soh.data.schema import BatteryRecord, Segment

_RAW_DIR = get_paths().nasa_raw
_HAS_RAW = _RAW_DIR.is_dir() and any(_RAW_DIR.rglob("*.mat"))
requires_raw = pytest.mark.skipif(not _HAS_RAW, reason=f"no NASA .mat files under {_RAW_DIR}")

# ------------------------------------------------------------------------------- registry


def test_registry_lists_the_34_batteries_in_order() -> None:
    assert tuple(PROTOCOLS) == NASA_BATTERY_IDS
    assert all(key == p.battery_id for key, p in PROTOCOLS.items())


def test_registry_is_read_only() -> None:
    with pytest.raises(TypeError):
        PROTOCOLS["B0005"] = PROTOCOLS["B0006"]  # type: ignore[index]


@pytest.mark.parametrize("alias", ["B0005", "b0005", "B5", "5"])
def test_get_protocol_accepts_aliases(alias: str) -> None:
    assert get_protocol(alias) is PROTOCOLS["B0005"]


def test_get_protocol_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown NASA battery 'B0035'"):
        get_protocol("B0035")
    with pytest.raises(ValueError, match="not a NASA battery id"):
        get_protocol("battery five")


def test_core_batteries() -> None:
    assert CORE_BATTERIES == ("B0005", "B0006", "B0007", "B0018")
    for battery_id in CORE_BATTERIES:
        p = PROTOCOLS[battery_id]
        assert p.usable
        assert (p.discharge_mode, p.discharge_current_a, p.ambient_c) == ("cc", 2.0, 24.0)
        assert p.condition == "2A_24C"


def test_extended_batteries_are_the_documented_clean_set() -> None:
    assert EXTENDED_BATTERIES == (
        "B0005",
        "B0006",
        "B0007",
        "B0018",
        "B0029",
        "B0030",
        "B0031",
        "B0032",
        "B0036",
        "B0046",
        "B0047",
        "B0048",
    )
    assert tuple(b for b, p in PROTOCOLS.items() if p.usable) == EXTENDED_BATTERIES
    assert set(CORE_BATTERIES) <= set(EXTENDED_BATTERIES)
    conditions = {PROTOCOLS[b].condition for b in EXTENDED_BATTERIES}
    assert conditions == {"2A_24C", "4A_43C", "1A_4C"}
    assert all(PROTOCOLS[b].discharge_mode == "cc" for b in EXTENDED_BATTERIES)


@pytest.mark.parametrize(
    ("battery_ids", "criterion"),
    [
        (("B0025", "B0026", "B0027", "B0028"), "protocol"),  # square-wave load
        (("B0038", "B0039", "B0040", "B0041", "B0042", "B0043", "B0044"), "protocol"),  # mixed
        (("B0033",), "reference"),  # formation anomaly
        (("B0045",), "reference"),  # anomalous initial capacity
        (("B0049", "B0050", "B0051", "B0052"), "history"),  # software crash
        (("B0034", "B0053", "B0054", "B0055", "B0056"), "labels"),  # noisy labels
    ],
)
def test_documented_exclusions(battery_ids: tuple[str, ...], criterion: str) -> None:
    for battery_id in battery_ids:
        p = PROTOCOLS[battery_id]
        assert not p.usable, battery_id
        assert criterion in p.failed_criteria, battery_id


def test_every_exclusion_has_a_specific_reason() -> None:
    for battery_id, p in PROTOCOLS.items():
        if p.usable:
            assert p.exclusion_reason is None
            assert p.failed_criteria == ()
            continue
        assert p.exclusion_reason is not None
        assert len(p.exclusion_reason) > 40, battery_id
        assert "pending" not in p.exclusion_reason.lower(), battery_id
        assert re.search(r"\d", p.exclusion_reason), f"{battery_id}: reason quotes no evidence"
        assert p.exclusion_reason.isascii(), battery_id
        assert set(p.failed_criteria) <= set(USABILITY_CRITERIA)
        ordered = [c for c in USABILITY_CRITERIA if c in p.failed_criteria]
        assert list(p.failed_criteria) == ordered, battery_id


def test_registry_text_is_ascii() -> None:
    for p in PROTOCOLS.values():
        assert p.notes.isascii()
        assert p.eol_criterion.isascii()
        assert p.group.isascii()


def test_conditions_and_groups() -> None:
    assert PROTOCOLS["B0029"].condition == "4A_43C"
    assert PROTOCOLS["B0046"].condition == "1A_4C"
    assert PROTOCOLS["B0025"].condition == "SQW4A_24C"
    assert PROTOCOLS["B0042"].condition == "mixed"
    assert np.isnan(PROTOCOLS["B0042"].discharge_current_a)
    assert PROTOCOLS["B0036"].group != PROTOCOLS["B0033"].group  # different loads, same README
    assert PROTOCOLS["B0028"].discharge_current_levels_a == (2.0,)  # README says 4 A


@pytest.mark.parametrize(
    ("usable", "reason", "failed", "match"),
    [
        (True, "some reason", (), "exclusion_reason"),
        (False, None, ("labels",), "exclusion_reason"),
        (False, "some reason", (), "failed_criteria"),
        (True, None, ("labels",), "failed_criteria"),
        (False, "some reason", ("bogus",), "unknown criteria"),
    ],
)
def test_battery_protocol_validation(
    usable: bool, reason: str | None, failed: tuple[str, ...], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        BatteryProtocol(
            battery_id="B0005",
            group="G",
            ambient_c=24.0,
            discharge_current_a=2.0,
            discharge_mode="cc",
            cutoff_v=2.7,
            eol_criterion="",
            usable=usable,
            exclusion_reason=reason,
            failed_criteria=failed,  # type: ignore[arg-type]
        )


# ------------------------------------------------------------------------- assess_usability

_GOOD = BatteryEvidence(
    battery_id="B0005",
    n_discharge=168,
    n_valid=168,
    qc_counts={"ok": 168},
    c0_ah=1.84,
    final_soh_pct=72.0,
    min_soh_pct=70.0,
    frac_above_reference=0.0,
    label_noise_pct=0.28,
    ambient_levels_c=(24.0,),
    discharge_current_levels_a=(2.0,),
    frac_zero_current_under_load=0.0,
    min_voltage_median_v=2.66,
    frac_discharge_window=1.0,
    frac_charge_window=0.98,
    frac_charge_window_390=0.98,
)
_CC = PROTOCOLS["B0005"]


def test_good_evidence_passes() -> None:
    assert assess_usability(_CC, _GOOD) == []
    assert assess_usability(_CC, _GOOD, group_c0_ah=1.86) == []


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"discharge_current_levels_a": (1.0, 2.0, 4.0)}, ["protocol"]),
        ({"ambient_levels_c": (24.0, 44.0)}, ["protocol"]),
        ({"frac_zero_current_under_load": 0.5}, ["protocol"]),
        ({"n_valid": MIN_VALID_CYCLES - 1}, ["history"]),
        ({"frac_above_reference": 0.84}, ["reference"]),
        ({"c0_ah": float("nan")}, ["reference"]),
        ({"frac_discharge_window": 0.0, "frac_charge_window": 0.4}, ["windows"]),
        ({"frac_discharge_window": float("nan"), "frac_charge_window": float("nan")}, ["windows"]),
        ({"label_noise_pct": MAX_LABEL_NOISE_PCT + 0.01}, ["labels"]),
        ({"label_noise_pct": float("nan")}, ["labels"]),
        ({"final_soh_pct": 97.0}, ["labels"]),  # 3 points of fade: no trajectory to learn
        (
            {"n_valid": 25, "label_noise_pct": 1.8, "discharge_current_levels_a": ()},
            ["protocol", "history", "labels"],
        ),
    ],
)
def test_each_criterion(changes: dict[str, object], expected: list[str]) -> None:
    evidence = dataclasses.replace(_GOOD, **changes)  # type: ignore[arg-type]
    assert assess_usability(_CC, evidence) == expected


def test_windows_pass_with_charge_window_only() -> None:
    evidence = dataclasses.replace(_GOOD, frac_discharge_window=0.0, frac_charge_window=1.0)
    assert assess_usability(_CC, evidence) == []  # B0032: charge HIs only


def test_non_cc_protocol_fails_even_with_clean_data() -> None:
    assert assess_usability(PROTOCOLS["B0025"], _GOOD) == ["protocol"]


def test_c0_far_from_group_median_fails_reference() -> None:
    evidence = dataclasses.replace(_GOOD, c0_ah=0.86)  # B0045 vs siblings at 1.48 Ah
    assert assess_usability(_CC, evidence, group_c0_ah=1.48) == ["reference"]
    assert assess_usability(_CC, evidence, group_c0_ah=None) == []  # no peers: not checked
    assert assess_usability(_CC, evidence, group_c0_ah=1.10) == []  # within 25 %


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_group_c0(bad: float) -> None:
    with pytest.raises(ValueError, match="group_c0_ah"):
        assess_usability(_CC, _GOOD, group_c0_ah=bad)


# ------------------------------------------------------------------------- battery_evidence


def _charge(v_start: float = 3.6, n: int = 120) -> Segment:
    t = np.arange(n) * 30.0
    v = np.minimum(np.linspace(v_start, 4.3, n), 4.2)
    i = np.where(v < 4.2, 1.5, np.linspace(1.5, 0.02, n))
    return Segment("charge", t, v, i, np.full(n, 25.0), ambient_c=24.0)


def _cc_discharge(capacity: float, current: float = 2.0, ambient: float = 24.0) -> Segment:
    n = 200
    t = np.arange(n) * 15.0
    i = np.full(n, -current)
    i[:3] = -0.002  # rest before the load is switched on
    v = np.r_[np.full(3, 4.19), np.linspace(4.05, 2.7, n - 3)]
    return Segment("discharge", t, v, i, np.full(n, 30.0), ambient_c=ambient, capacity_ah=capacity)


def _square_wave_discharge(capacity: float) -> Segment:
    n = 400
    t = np.arange(n) * 10.0
    on = (np.arange(n) // 2) % 2 == 0  # 0.05 Hz square wave sampled at 0.1 Hz
    i = np.where(on, -4.0, -0.001)
    v = np.linspace(4.0, 2.0, n) - np.where(on, 0.2, 0.0)
    return Segment("discharge", t, v, i, np.full(n, 30.0), ambient_c=24.0, capacity_ah=capacity)


def _eis() -> Segment:
    return Segment("impedance", re_ohm=0.05, rct_ohm=0.08)


def _record(discharges: list[Segment], *, with_charges: bool = True) -> BatteryRecord:
    segments: list[Segment] = []
    for seg in discharges:
        if with_charges:
            segments.extend([_charge(), _eis()])
        segments.append(seg)
    return BatteryRecord("B0005", tuple(segments))


def test_evidence_of_a_clean_cc_cell() -> None:
    caps = 1.85 - 0.006 * np.arange(60)
    evidence = battery_evidence(_record([_cc_discharge(c) for c in caps]))
    assert evidence.n_discharge == evidence.n_valid == 60
    assert dict(evidence.qc_counts) == {"ok": 60}
    assert evidence.c0_ah == pytest.approx(np.median(caps[:5]))
    assert evidence.final_soh_pct == pytest.approx(100 * np.median(caps[-5:]) / evidence.c0_ah)
    assert evidence.min_soh_pct == pytest.approx(100 * caps[-1] / evidence.c0_ah)
    assert evidence.frac_above_reference == 0.0
    assert evidence.label_noise_pct == pytest.approx(0.0, abs=1e-9)  # perfectly linear fade
    assert evidence.ambient_levels_c == (24.0,)
    assert evidence.discharge_current_levels_a == (2.0,)
    assert evidence.frac_zero_current_under_load == 0.0
    assert evidence.min_voltage_median_v == pytest.approx(2.7)
    assert evidence.frac_discharge_window == 1.0
    assert evidence.frac_charge_window == 1.0
    assert evidence.frac_charge_window_390 == 1.0
    assert assess_usability(_CC, evidence) == []


def test_evidence_noise_is_robust_and_in_soh_points() -> None:
    rng = np.random.default_rng(0)
    caps = 1.8 - 0.004 * np.arange(400) + rng.normal(0.0, 0.018, 400)  # 1 SOH point of noise
    evidence = battery_evidence(_record([_cc_discharge(c) for c in caps], with_charges=False))
    assert evidence.label_noise_pct == pytest.approx(1.0, rel=0.15)


def test_evidence_without_preceding_charge_has_no_charge_window() -> None:
    caps = 1.85 - 0.006 * np.arange(40)
    evidence = battery_evidence(_record([_cc_discharge(c) for c in caps], with_charges=False))
    assert evidence.frac_charge_window == 0.0
    assert evidence.frac_discharge_window == 1.0


def test_evidence_charge_starting_high_has_no_charge_window() -> None:
    segments: list[Segment] = []
    for cap in 1.0 - 0.003 * np.arange(40):
        segments.extend([_charge(v_start=4.06), _cc_discharge(cap, current=1.0, ambient=4.0)])
    evidence = battery_evidence(BatteryRecord("B0045", tuple(segments)))
    assert evidence.frac_charge_window == 0.0  # B0045: CC phase starts at 4.06 V
    assert evidence.discharge_current_levels_a == (1.0,)
    assert evidence.ambient_levels_c == (4.0,)


def test_evidence_detects_square_wave_and_mixed_loads() -> None:
    caps = 1.85 - 0.002 * np.arange(30)
    square = battery_evidence(_record([_square_wave_discharge(c) for c in caps]))
    assert square.frac_zero_current_under_load == pytest.approx(0.5, abs=0.02)
    assert "protocol" in assess_usability(_CC, square)

    mixed_segments = [
        _cc_discharge(c, current=2.0 if k < 15 else 4.0, ambient=22.0 if k < 15 else 4.0)
        for k, c in enumerate(caps)
    ]
    mixed = battery_evidence(_record(mixed_segments))
    assert mixed.discharge_current_levels_a == (2.0, 4.0)
    assert mixed.ambient_levels_c == (4.0, 22.0)
    assert "protocol" in assess_usability(_CC, mixed)


def test_evidence_formation_anomaly_fails_reference() -> None:
    caps = np.r_[np.linspace(0.9, 1.3, 8), np.linspace(1.35, 1.88, 50)]  # B0033-like climb
    evidence = battery_evidence(_record([_cc_discharge(c) for c in caps]))
    assert evidence.frac_above_reference > 0.5
    assert "reference" in assess_usability(_CC, evidence)


def test_evidence_of_an_empty_record() -> None:
    evidence = battery_evidence(BatteryRecord("B0005", ()))
    assert evidence.n_discharge == evidence.n_valid == 0
    assert np.isnan(evidence.c0_ah)
    assert np.isnan(evidence.frac_discharge_window)
    assert assess_usability(_CC, evidence) == [
        "protocol",
        "history",
        "reference",
        "windows",
        "labels",
    ]


def test_evidence_with_missing_capacities() -> None:
    caps: list[float | None] = [None] * 10 + list(1.8 - 0.005 * np.arange(40))
    segs = [
        _cc_discharge(c)
        if c is not None
        else dataclasses.replace(_cc_discharge(1.0), capacity_ah=None)
        for c in caps
    ]
    evidence = battery_evidence(_record(segs))
    assert evidence.qc_counts["missing"] == 10
    assert evidence.n_valid == 40
    assert evidence.c0_ah == pytest.approx(np.median([c for c in caps[10:15] if c is not None]))


# --------------------------------------------------------------------------- real NASA files


@pytest.mark.raw_data
@requires_raw
def test_readmes_exist_and_state_the_cutoff_voltages() -> None:
    for battery_id, p in PROTOCOLS.items():
        readme = _RAW_DIR / p.readme
        assert readme.is_file(), f"{battery_id}: {readme}"
        text = readme.read_text(encoding="latin-1")
        cutoffs = {f"{p.cutoff_v:g}V", f"{p.cutoff_v:.1f}V"}
        assert any(c in text for c in cutoffs), f"{battery_id}: cut-off {p.cutoff_v} V"
        number = str(int(battery_id[1:]))
        assert re.search(rf"\b{number}\b", text), f"{battery_id} not mentioned in {p.readme}"


@pytest.mark.raw_data
@requires_raw
def test_core_cells_pass_the_rule_on_real_data() -> None:
    table = evidence_table(_RAW_DIR, CORE_BATTERIES)
    assert table["battery_id"].tolist() == list(CORE_BATTERIES)
    assert table["rule_usable"].all()
    assert (table["n_valid"] == [168, 168, 168, 132]).all()
    assert (table["label_noise_pct"] < 0.3).all()
    assert (table["frac_discharge_window"] == 1.0).all()
    assert table["c0_ah"].between(1.83, 2.02).all()


def _quoted(pattern: str, text: str) -> list[float]:
    return [float(x) for x in re.findall(pattern, text)]


@pytest.mark.raw_data
@pytest.mark.slow
@requires_raw
def test_registry_matches_rule_on_all_34_batteries() -> None:
    """Re-derive every usability decision (and the numbers quoted in reasons) from raw data."""
    table = evidence_table(_RAW_DIR).set_index("battery_id")
    assert tuple(table.index) == NASA_BATTERY_IDS
    for battery_id, row in table.iterrows():
        p = PROTOCOLS[str(battery_id)]
        rule = tuple(x for x in str(row["rule_failures"]).split(",") if x)
        assert rule == p.failed_criteria, battery_id
        assert bool(row["rule_usable"]) == p.usable, battery_id
        assert tuple(row["ambient_levels_c"]) == p.ambient_levels_c, battery_id
        assert tuple(row["discharge_current_levels_a"]) == p.discharge_current_levels_a, battery_id
        reason = p.exclusion_reason or ""
        for noise in _quoted(r"label noise (\d+\.\d\d)", reason):
            assert noise == pytest.approx(row["label_noise_pct"], abs=0.006), battery_id
        for valid in _quoted(r"(\d+) QC-valid", reason):
            assert valid == row["n_valid"], battery_id
        for c0 in _quoted(r"C0 (\d\.\d\d) Ah", reason):
            assert c0 == pytest.approx(row["c0_ah"], abs=0.006), battery_id
    b45 = table.loc["B0045"]
    assert 1.0 - b45["c0_ah"] / b45["group_c0_ah"] == pytest.approx(0.42, abs=0.01)
    assert pd.isna(table.loc["B0036", "group_c0_ah"])  # single-cell group: no peer check
