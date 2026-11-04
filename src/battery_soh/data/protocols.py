"""Test protocols of the 34 NASA PCoE cells and which of them are usable for SOH modelling.

The registry transcribes the README files shipped with the dataset
(``data/raw/nasa/**/README*.txt``) and corrects them where the ``.mat`` files disagree (see
each entry's ``notes``). Every cell was charged the same way (CC 1.5 A to 4.2 V, then CV until
20 mA); the campaigns differ in ambient temperature, discharge load and cut-off voltage, and
several have known data problems.

Inclusion rule
--------------
Usability is decided by an explicit rule, not by inspection. :func:`battery_evidence` measures
each cell from its raw file and :func:`assess_usability` applies five criteria; a cell is usable
when it passes all of them:

1. ``protocol`` - one constant discharge current at one ambient temperature: the per-cycle load
   has a single level and the current is never switched off while the load is on (at most
   :data:`MAX_ZERO_CURRENT_FRACTION` near-zero samples). Square-wave and mixed-load histories
   make capacity and partial-window HIs incomparable from one cycle to the next.
2. ``history`` - at least :data:`MIN_VALID_CYCLES` QC-valid discharges (see :mod:`battery_soh.qc`).
3. ``reference`` - the reference capacity ``C0`` (label denominator) is trustworthy: at most
   :data:`MAX_FRAC_ABOVE_REFERENCE` of the valid cycles exceed ``C0`` by more than 5 % (a
   formation/break-in period makes ``C0`` meaningless), and ``C0`` lies within
   :data:`MAX_C0_DEVIATION` of the median ``C0`` of the cell's campaign group (computed when the
   group has at least three cells).
4. ``windows`` - a partial measurement window is observable in at least
   :data:`MIN_WINDOW_COVERAGE` of the valid cycles: the 3.80 -> 3.50 V discharge window or the
   4.00 -> 4.19 V CC-charge window.
5. ``labels`` - the SOH trajectory carries signal: the robust cycle-to-cycle label noise is at
   most :data:`MAX_LABEL_NOISE_PCT` SOH points and the total fade is at least
   :data:`MIN_FADE_PCT` points.

The thresholds sit in clear gaps of the measured distributions (for example, label noise is
0.23-0.54 SOH points for every cell that passes the other criteria and 0.97 or more for every
cell that fails on noise). ``usable``, ``exclusion_reason`` and ``failed_criteria`` in
:data:`PROTOCOLS` are the cached outcome of the rule; the numbers quoted in the reasons come
from :func:`evidence_table`, and ``tests/unit/test_data_protocols.py`` re-derives every decision
from the raw files whenever they are present.

Resulting sets
--------------
* :data:`CORE_BATTERIES` (headline benchmark): B0005, B0006, B0007, B0018 - 2 A CC discharge at
  24 deg C, 132-168 cycles each, label noise 0.26-0.28 SOH points.
* :data:`EXTENDED_BATTERIES` (multi-condition benchmark, 12 cells, 3 conditions): the core cells
  plus B0036 (2 A, 24 deg C), B0029-B0032 (4 A, 43 deg C) and B0046-B0048 (1 A, 4 deg C).

Two borderline groups were examined in detail and are excluded:

* **B0045** (1 A, 4 deg C): ``C0`` = 0.86 Ah is 42 % below the 1.48 Ah of its three identically
  tested siblings, its label noise is 1.02 SOH points (siblings 0.37-0.54), and its CC charge
  starts at 4.06 V (median), so the 4.00 -> 4.19 V charge window exists in 1 % of its cycles.
  It is an outlier cell, not a representative of its condition (in the audit's 19-cell
  leave-one-battery-out run with relative charge + EIS HIs, gradient boosting scored MAE 6.9
  on B0045 against 1.3-1.7 on B0046-B0048).
* **B0053-B0056** (2 A, 4 deg C): the per-cycle capacity jitters by 1.15-2.31 SOH points (4-9x
  the core cells) against a total fade of 7-22 points, so the label itself is too noisy to
  score an estimator; under the cold 2 A load the voltage starts below 3.80 V in B0053, B0054
  and B0056, so the discharge window never exists there. In the same audit run (relative HIs,
  all partial windows) their group MAE was 7.9-8.2 SOH points against 1.3-3.6 for B0029-B0032
  and B0036.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal

import numpy as np
import pandas as pd

from battery_soh._typing import FloatArray
from battery_soh.config import RATED_CAPACITY_AH
from battery_soh.data.nasa import NASA_BATTERY_IDS, iter_batteries, normalize_battery_id
from battery_soh.data.schema import BatteryRecord, Segment
from battery_soh.labels import label_battery

logger = logging.getLogger(__name__)

__all__ = [
    "CHARGE_PROTOCOL",
    "CORE_BATTERIES",
    "EXTENDED_BATTERIES",
    "MAX_C0_DEVIATION",
    "MAX_FRAC_ABOVE_REFERENCE",
    "MAX_LABEL_NOISE_PCT",
    "MAX_ZERO_CURRENT_FRACTION",
    "MIN_FADE_PCT",
    "MIN_VALID_CYCLES",
    "MIN_WINDOW_COVERAGE",
    "PROTOCOLS",
    "USABILITY_CRITERIA",
    "BatteryEvidence",
    "BatteryProtocol",
    "Criterion",
    "DischargeMode",
    "assess_usability",
    "battery_evidence",
    "evidence_table",
    "get_protocol",
]

DischargeMode = Literal["cc", "square_wave", "mixed"]
Criterion = Literal["protocol", "history", "reference", "windows", "labels"]

#: The inclusion criteria, in the order :func:`assess_usability` reports them.
USABILITY_CRITERIA: Final[tuple[Criterion, ...]] = (
    "protocol",
    "history",
    "reference",
    "windows",
    "labels",
)

#: Charge protocol shared by all 34 cells (README).
CHARGE_PROTOCOL: Final = "CC 1.5 A to 4.2 V, then CV at 4.2 V until the current falls to 20 mA"

# ------------------------------------------------------------------------ inclusion criteria
#: Minimum number of QC-valid discharge cycles.
MIN_VALID_CYCLES: Final = 30
#: Largest tolerated fraction of valid cycles with ``soh_pct > 105`` (unstable reference).
MAX_FRAC_ABOVE_REFERENCE: Final = 0.10
#: Largest tolerated relative deviation of ``C0`` from the campaign-group median.
MAX_C0_DEVIATION: Final = 0.25
#: Minimum fraction of valid cycles with an observable partial window.
MIN_WINDOW_COVERAGE: Final = 0.5
#: Largest tolerated cycle-to-cycle label noise (SOH points, robust estimate).
MAX_LABEL_NOISE_PCT: Final = 0.75
#: Smallest total fade (``100 - final SOH``, SOH points) that makes a trajectory informative.
MIN_FADE_PCT: Final = 5.0
#: Largest tolerated fraction of near-zero current while the load is on (pulsed loads).
MAX_ZERO_CURRENT_FRACTION: Final = 0.1

_SOH_ABOVE_REFERENCE_PCT: Final = 105.0
_MIN_GROUP_SIZE: Final = 3
_LOAD_ON_A: Final = 0.5
_CC_CHARGE_A: Final = 1.0
_DIS_WINDOW_V: Final = (3.80, 3.50)
_CHG_WINDOW_END_V: Final = 4.19
_CHG_WINDOW_390_START_V: Final = 3.85
_CHG_WINDOW_400_START_V: Final = 3.97

_NAN: Final = float("nan")


@dataclass(frozen=True)
class BatteryProtocol:
    """Test conditions of one cell, transcribed from its README and checked against data.

    Attributes:
        battery_id: Cell identifier, e.g. ``"B0005"``.
        group: Campaign and condition, e.g. ``"FY08Q4_2A_24C"``.
        ambient_c: Nominal ambient temperature (deg C); NaN when several were used.
        discharge_current_a: Nominal discharge current (A; square-wave amplitude for
            ``square_wave``, as stated in the README); NaN when several loads were used.
        discharge_mode: ``"cc"``, ``"square_wave"`` or ``"mixed"``.
        cutoff_v: Discharge cut-off voltage (V).
        eol_criterion: Stopping criterion, quoted from the README.
        usable: Whether the cell enters SOH modelling (see the module docstring).
        exclusion_reason: Evidence-backed reason when ``usable`` is False, else None.
        failed_criteria: The :data:`USABILITY_CRITERIA` the cell fails (empty iff usable).
        readme: README path relative to the NASA data root.
        ambient_levels_c: Every ambient temperature recorded for discharges in the ``.mat``.
        discharge_current_levels_a: Every discharge load level found in the ``.mat`` file.
        notes: Discrepancies between README and data, and other remarks.
    """

    battery_id: str
    group: str
    ambient_c: float
    discharge_current_a: float
    discharge_mode: DischargeMode
    cutoff_v: float
    eol_criterion: str
    usable: bool
    exclusion_reason: str | None
    failed_criteria: tuple[Criterion, ...] = ()
    readme: str = ""
    ambient_levels_c: tuple[float, ...] = ()
    discharge_current_levels_a: tuple[float, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        if self.usable == (self.exclusion_reason is not None):
            raise ValueError(f"{self.battery_id}: exclusion_reason must be set iff not usable")
        if self.usable == bool(self.failed_criteria):
            raise ValueError(f"{self.battery_id}: failed_criteria must be empty iff usable")
        unknown = set(self.failed_criteria) - set(USABILITY_CRITERIA)
        if unknown:
            raise ValueError(f"{self.battery_id}: unknown criteria {sorted(unknown)}")

    @property
    def condition(self) -> str:
        """Operating condition label, e.g. ``"2A_24C"``, ``"SQW4A_24C"`` or ``"mixed"``."""
        if self.discharge_mode == "mixed":
            return "mixed"
        prefix = "SQW" if self.discharge_mode == "square_wave" else ""
        return f"{prefix}{self.discharge_current_a:g}A_{self.ambient_c:g}C"


# --------------------------------------------------------------------------- README extracts
_R_FY08Q4 = "1. BatteryAgingARC-FY08Q4/README.txt"
_R_25_28 = "2. BatteryAgingARC_25_26_27_28_P1/README.txt"
_R_29_32 = "3. BatteryAgingARC_25-44/README_29_30_31_32.txt"
_R_33_36 = "3. BatteryAgingARC_25-44/README_33_34_36.txt"
_R_38_40 = "3. BatteryAgingARC_25-44/README_38_39_40.txt"
_R_41_44 = "3. BatteryAgingARC_25-44/README_41_42_43_44.txt"
_R_45_48 = "4. BatteryAgingARC_45_46_47_48/README_45_46_47_48.txt"
_R_49_52 = "5. BatteryAgingARC_49_50_51_52/README_49_50_51_52.txt"
_R_53_56 = "6. BatteryAgingARC_53_54_55_56/README_53_54_55_56.txt"

_EOL_30PCT_FY08 = (
    "The experiments were stopped when the batteries reached end-of-life (EOL) criteria, "
    "which was a 30% fade in rated capacity (from 2Ahr to 1.4Ahr)."
)
_EOL_20PCT = "The experiments were carried out until the capacity had reduced to 1.6Ahr (20% fade)."
_EOL_30PCT = "The experiments were carried out until the capacity had reduced to 1.4Ahr (30% fade)."
_EOL_CRASH = "The experiments were carried out until the experiment control software crashed."
_EOL_NONE = "Not stated in the README."


def _p(
    battery_id: str,
    group: str,
    ambient_c: float,
    current_a: float,
    mode: DischargeMode,
    cutoff_v: float,
    eol: str,
    readme: str,
    *,
    excluded: tuple[str, tuple[Criterion, ...]] | None = None,
    ambient_levels: tuple[float, ...] | None = None,
    current_levels: tuple[float, ...] | None = None,
    notes: str = "",
) -> BatteryProtocol:
    reason, failed = excluded if excluded is not None else (None, ())
    return BatteryProtocol(
        battery_id=battery_id,
        group=group,
        ambient_c=ambient_c,
        discharge_current_a=current_a,
        discharge_mode=mode,
        cutoff_v=cutoff_v,
        eol_criterion=eol,
        usable=excluded is None,
        exclusion_reason=reason,
        failed_criteria=failed,
        readme=readme,
        ambient_levels_c=ambient_levels if ambient_levels is not None else (ambient_c,),
        discharge_current_levels_a=current_levels if current_levels is not None else (current_a,),
        notes=notes,
    )


def _square_wave(final_soh: float, n_valid: int) -> tuple[str, tuple[Criterion, ...]]:
    return (
        "square-wave load (README: 0.05 Hz, 50 % duty): the current is off for half of the "
        "loaded time, so constant-current partial-window HIs are undefined; only "
        f"{n_valid} valid discharges and {100 - final_soh:.1f} SOH points of fade",
        ("protocol", "history", "labels"),
    )


def _mixed_38_40(final_soh: float, c0_ah: float) -> tuple[str, tuple[Criterion, ...]]:
    return (
        "mixed protocol: loads of 1, 2 and 4 A at 24 and 44 deg C within one history, so "
        f"capacity tracks the test condition rather than ageing (C0 {c0_ah:.2f} Ah from the "
        f"first 4 A runs; final SOH {final_soh:.0f} %)",
        ("protocol", "reference", "labels"),
    )


_MIXED_42_44: Final = (
    "mixed protocol: 41 discharges at 22 deg C / 2 A, then 46 at 4 deg C / 4 A that all "
    "deliver < 0.3 Ah (QC below_min), then 25 at 4 deg C / 1 A",
    ("protocol",),
)
_NOTE_42_44: Final = (
    "README states 4 deg C with 4 A and 1 A loads; the .mat file also records 22 deg C "
    "ambient and 2 A loads for the first 41 discharges."
)


def _cold_2a(detail: str) -> tuple[str, tuple[Criterion, ...]]:
    return (f"2 A at 4 deg C: {detail}", ("labels",))


_PROTOCOL_LIST: tuple[BatteryProtocol, ...] = (
    # FY08Q4: the four reference cells, 2 A CC at room temperature (24 deg C in the .mat).
    *(
        _p(
            b,
            "FY08Q4_2A_24C",
            24.0,
            2.0,
            "cc",
            v,
            _EOL_30PCT_FY08,
            _R_FY08Q4,
            notes="README says 'room temperature'; the .mat records 24 deg C.",
        )
        for b, v in (("B0005", 2.7), ("B0006", 2.5), ("B0007", 2.2), ("B0018", 2.5))
    ),
    # 25-28: 0.05 Hz square wave, 4 A amplitude (README), 50 % duty cycle, 24 deg C.
    _p(
        "B0025",
        "ARC25-28_SQW4A_24C",
        24.0,
        4.0,
        "square_wave",
        2.0,
        _EOL_NONE,
        _R_25_28,
        excluded=_square_wave(96.9, 28),
    ),
    _p(
        "B0026",
        "ARC25-28_SQW4A_24C",
        24.0,
        4.0,
        "square_wave",
        2.2,
        _EOL_NONE,
        _R_25_28,
        excluded=_square_wave(97.7, 27),
    ),
    _p(
        "B0027",
        "ARC25-28_SQW4A_24C",
        24.0,
        4.0,
        "square_wave",
        2.5,
        _EOL_NONE,
        _R_25_28,
        excluded=_square_wave(98.5, 28),
    ),
    _p(
        "B0028",
        "ARC25-28_SQW4A_24C",
        24.0,
        4.0,
        "square_wave",
        2.7,
        _EOL_NONE,
        _R_25_28,
        excluded=_square_wave(96.6, 28),
        current_levels=(2.0,),
        notes="README states a 4 A amplitude; Current_measured shows a 2 A square wave.",
    ),
    # 29-32: 4 A CC at 43 deg C.
    *(
        _p(b, "ARC29-32_4A_43C", 43.0, 4.0, "cc", v, _EOL_NONE, _R_29_32)
        for b, v in (("B0029", 2.0), ("B0030", 2.2), ("B0031", 2.5))
    ),
    _p(
        "B0032",
        "ARC29-32_4A_43C",
        43.0,
        4.0,
        "cc",
        2.7,
        _EOL_NONE,
        _R_29_32,
        notes=(
            "The loaded voltage starts at 3.77 V (median), below the 3.80 V start of the "
            "discharge window, so only charge-window HIs exist for this cell."
        ),
    ),
    # 33, 34: 4 A CC; 36: 2 A CC; 24 deg C.
    _p(
        "B0033",
        "ARC33-36_4A_24C",
        24.0,
        4.0,
        "cc",
        2.0,
        _EOL_20PCT,
        _R_33_36,
        excluded=(
            "formation anomaly: capacity climbs from 0.90 to 1.89 Ah, so 84 % of valid cycles "
            "exceed C0 (1.29 Ah) by more than 5 %; label noise 0.97 SOH points; the 3.80 V "
            "discharge window never exists (the 4 A load starts at 3.44 V)",
            ("reference", "labels"),
        ),
    ),
    _p(
        "B0034",
        "ARC33-36_4A_24C",
        24.0,
        4.0,
        "cc",
        2.2,
        _EOL_20PCT,
        _R_33_36,
        excluded=(
            "erratic capacity: label noise 1.03 SOH points per cycle (core cells 0.26-0.28) "
            "with recovery jumps of up to 0.23 Ah; the 3.80 V discharge window never exists "
            "(the 4 A load starts at 3.49 V)",
            ("labels",),
        ),
    ),
    _p(
        "B0036",
        "ARC33-36_2A_24C",
        24.0,
        2.0,
        "cc",
        2.7,
        _EOL_20PCT,
        _R_33_36,
        notes=(
            "Same condition as the core cells (2 A, 24 deg C) but a different campaign; "
            "QC removes a 1.00 Ah first discharge and 1.99 Ah / 2.44 Ah spikes."
        ),
    ),
    # 38-40: loads of 1, 2 and 4 A at 24 and 44 deg C.
    *(
        _p(
            b,
            "ARC38-40_MIXED",
            _NAN,
            _NAN,
            "mixed",
            v,
            _EOL_20PCT,
            _R_38_40,
            excluded=_mixed_38_40(final, c0),
            ambient_levels=(24.0, 44.0),
            current_levels=(1.0, 2.0, 4.0),
        )
        for b, v, final, c0 in (
            ("B0038", 2.2, 163.0, 1.06),
            ("B0039", 2.5, 345.2, 0.47),
            ("B0040", 2.7, 211.7, 0.78),
        )
    ),
    # 41-44: README says 4 deg C with loads of 4 A and 1 A.
    _p(
        "B0041",
        "ARC41-44_MIXED",
        4.0,
        _NAN,
        "mixed",
        2.0,
        _EOL_30PCT,
        _R_41_44,
        excluded=(
            "mixed protocol: 42 discharges at 4 A that all deliver < 0.3 Ah (QC below_min), "
            "then 25 at 1 A; only 25 valid cycles",
            ("protocol", "history", "reference", "labels"),
        ),
        ambient_levels=(4.0,),
        current_levels=(1.0, 4.0),
    ),
    *(
        _p(
            b,
            "ARC41-44_MIXED",
            _NAN,
            _NAN,
            "mixed",
            v,
            _EOL_30PCT,
            _R_41_44,
            excluded=_MIXED_42_44,
            ambient_levels=(4.0, 22.0),
            current_levels=(1.0, 2.0, 4.0),
            notes=_NOTE_42_44,
        )
        for b, v in (("B0042", 2.2), ("B0043", 2.5), ("B0044", 2.7))
    ),
    # 45-48: 1 A CC at 4 deg C.
    _p(
        "B0045",
        "ARC45-48_1A_4C",
        4.0,
        1.0,
        "cc",
        2.0,
        _EOL_30PCT,
        _R_45_48,
        excluded=(
            "anomalous cell: C0 0.86 Ah is 42 % below the 1.48 Ah median of its identically "
            "tested siblings; label noise 1.02 SOH points (siblings 0.37-0.54); the CC charge "
            "starts at 4.06 V, so the 4.00 V charge window exists in 1 % of cycles",
            ("reference", "labels"),
        ),
    ),
    *(
        _p(
            b,
            "ARC45-48_1A_4C",
            4.0,
            1.0,
            "cc",
            v,
            _EOL_30PCT,
            _R_45_48,
            notes="QC removes 0.00 Ah records (below_min) and a high first discharge.",
        )
        for b, v in (("B0046", 2.2), ("B0047", 2.5), ("B0048", 2.7))
    ),
    # 49-52: 2 A CC at 4 deg C until the control software crashed.
    _p(
        "B0049",
        "ARC49-52_2A_4C",
        4.0,
        2.0,
        "cc",
        2.0,
        _EOL_CRASH,
        _R_49_52,
        excluded=(
            "software crash after 25 discharges: 21 QC-valid; label noise 1.76 SOH points",
            ("history", "labels"),
        ),
    ),
    _p(
        "B0050",
        "ARC49-52_2A_4C",
        4.0,
        2.0,
        "cc",
        2.2,
        _EOL_CRASH,
        _R_49_52,
        excluded=(
            "software crash after 25 discharges: 9 QC-valid (4 without a capacity value); "
            "EIS fits diverge (Re up to 1e15 Ohm)",
            ("history", "reference", "labels"),
        ),
    ),
    _p(
        "B0051",
        "ARC49-52_2A_4C",
        4.0,
        2.0,
        "cc",
        2.5,
        _EOL_CRASH,
        _R_49_52,
        excluded=(
            "software crash after 25 discharges: 22 QC-valid; partial windows observable in "
            "under half of them",
            ("history", "windows", "labels"),
        ),
    ),
    _p(
        "B0052",
        "ARC49-52_2A_4C",
        4.0,
        2.0,
        "cc",
        2.7,
        _EOL_CRASH,
        _R_49_52,
        excluded=(
            "software crash after 25 discharges: 21 have no capacity value, 3 QC-valid; EIS "
            "fits diverge (Re up to 1e5 Ohm)",
            ("history", "labels"),
        ),
    ),
    # 53-56: 2 A CC at 4 deg C.
    _p(
        "B0053",
        "ARC53-56_2A_4C",
        4.0,
        2.0,
        "cc",
        2.0,
        _EOL_30PCT,
        _R_53_56,
        excluded=_cold_2a(
            "label noise 2.31 SOH points per cycle against 7.4 points of total fade; the "
            "3.80 V discharge window never exists (load starts at 3.74 V)"
        ),
    ),
    _p(
        "B0054",
        "ARC53-56_2A_4C",
        4.0,
        2.0,
        "cc",
        2.2,
        _EOL_30PCT,
        _R_53_56,
        excluded=_cold_2a(
            "label noise 1.40 SOH points per cycle (5x the core cells); the 3.80 V discharge "
            "window never exists (load starts at 3.77 V)"
        ),
    ),
    _p(
        "B0055",
        "ARC53-56_2A_4C",
        4.0,
        2.0,
        "cc",
        2.5,
        _EOL_30PCT,
        _R_53_56,
        excluded=_cold_2a(
            "label noise 1.15 SOH points per cycle (4x the core cells); C0 1.26 Ah is 63 % "
            "of the 2 Ah rating"
        ),
    ),
    _p(
        "B0056",
        "ARC53-56_2A_4C",
        4.0,
        2.0,
        "cc",
        2.7,
        _EOL_30PCT,
        _R_53_56,
        excluded=_cold_2a(
            "label noise 1.85 SOH points per cycle (7x the core cells); the 3.80 V discharge "
            "window never exists (load starts at 3.77 V)"
        ),
    ),
)

#: Protocol of every cell, keyed by battery id (read-only).
PROTOCOLS: Final[Mapping[str, BatteryProtocol]] = MappingProxyType(
    {p.battery_id: p for p in _PROTOCOL_LIST}
)
if tuple(PROTOCOLS) != NASA_BATTERY_IDS:  # pragma: no cover - guards the table above
    raise RuntimeError("PROTOCOLS must list exactly the 34 NASA batteries in order")

#: Headline benchmark cells: 2 A constant-current discharge at 24 deg C.
CORE_BATTERIES: Final[tuple[str, ...]] = ("B0005", "B0006", "B0007", "B0018")

#: Multi-condition benchmark: every usable cell (core + B0029-B0032, B0036, B0046-B0048).
EXTENDED_BATTERIES: Final[tuple[str, ...]] = tuple(b for b, p in PROTOCOLS.items() if p.usable)


def get_protocol(battery_id: str) -> BatteryProtocol:
    """Return the protocol of one cell.

    Args:
        battery_id: Identifier such as ``"B0005"`` (``"b5"`` and ``"5"`` are accepted).

    Returns:
        The registry entry.

    Raises:
        ValueError: If the identifier is malformed or not one of the 34 NASA cells.
    """
    key = normalize_battery_id(battery_id)
    try:
        return PROTOCOLS[key]
    except KeyError:
        raise ValueError(f"unknown NASA battery {battery_id!r}; known: {list(PROTOCOLS)}") from None


# ---------------------------------------------------------------------------------- evidence


@dataclass(frozen=True)
class BatteryEvidence:
    """Data-derived facts about one cell, used to verify the registry and decide usability.

    Attributes:
        battery_id: Cell identifier.
        n_discharge: Discharge operations in the file.
        n_valid: QC-valid discharges (see :mod:`battery_soh.qc`).
        qc_counts: Number of discharges per QC reason.
        c0_ah: Reference capacity (median of the first 5 valid capacities).
        final_soh_pct: Median ``soh_pct`` of the last 5 valid cycles.
        min_soh_pct: Lowest valid ``soh_pct``.
        frac_above_reference: Fraction of valid cycles with ``soh_pct > 105``.
        label_noise_pct: Robust cycle-to-cycle noise of the valid ``soh_pct`` sequence
            (``1.4826 * MAD`` of first differences divided by ``sqrt(2)``), in SOH points.
        ambient_levels_c: Distinct ambient temperatures recorded for discharges.
        discharge_current_levels_a: Distinct load levels (per-cycle median ``|I|`` under load,
            rounded to 0.5 A, seen in at least two cycles).
        frac_zero_current_under_load: Median fraction of near-zero-current samples between
            load-on and load-off (about 0.5 for a 50 % duty square wave).
        min_voltage_median_v: Median of the per-cycle minimum discharge voltage.
        frac_discharge_window: Fraction of valid cycles whose 3.80 -> 3.50 V discharge window
            is observable (loaded voltage starts above 3.80 V and reaches 3.50 V).
        frac_charge_window: Fraction of valid cycles preceded by a charge whose CC phase
            starts below 3.97 V and reaches 4.19 V (4.00 -> 4.19 V window observable).
        frac_charge_window_390: Same with the stricter start below 3.85 V (3.90 V window).
    """

    battery_id: str
    n_discharge: int
    n_valid: int
    qc_counts: Mapping[str, int] = field(default_factory=dict)
    c0_ah: float = _NAN
    final_soh_pct: float = _NAN
    min_soh_pct: float = _NAN
    frac_above_reference: float = _NAN
    label_noise_pct: float = _NAN
    ambient_levels_c: tuple[float, ...] = ()
    discharge_current_levels_a: tuple[float, ...] = ()
    frac_zero_current_under_load: float = _NAN
    min_voltage_median_v: float = _NAN
    frac_discharge_window: float = _NAN
    frac_charge_window: float = _NAN
    frac_charge_window_390: float = _NAN


def _load_current(seg: Segment) -> float:
    on = np.abs(seg.current_a) > _LOAD_ON_A
    return float(np.median(np.abs(seg.current_a[on]))) if on.any() else _NAN


def _zero_current_fraction(seg: Segment) -> float:
    on = np.flatnonzero(np.abs(seg.current_a) > _LOAD_ON_A)
    if on.size < 10:
        return _NAN
    inside = np.abs(seg.current_a[on[0] : on[-1] + 1])
    return float(np.mean(inside < _LOAD_ON_A))


def _discharge_window_observable(seg: Segment) -> bool:
    on = np.flatnonzero(np.abs(seg.current_a) > _LOAD_ON_A)
    if on.size < 10:
        return False
    v_high, v_low = _DIS_WINDOW_V
    k0 = int(on[0])
    v_after_load_on = seg.voltage_v[min(k0 + 2, seg.n_samples - 1)]
    loaded = seg.voltage_v[k0 : int(on[-1]) + 1]
    return bool(v_after_load_on > v_high and np.any(loaded <= v_low))


def _charge_window_observable(seg: Segment | None, start_below_v: float) -> bool:
    if seg is None or seg.n_samples < 20:
        return False
    cc = np.flatnonzero(seg.current_a > _CC_CHARGE_A)
    if cc.size == 0:
        return False
    k0 = int(cc[0])
    v_start = float(np.median(seg.voltage_v[k0 : k0 + 3]))
    return bool(v_start < start_below_v and np.any(seg.voltage_v[k0:] >= _CHG_WINDOW_END_V))


def _robust_noise(values: FloatArray) -> float:
    if values.size < 3:
        return _NAN
    diffs = np.diff(values)
    mad = float(np.median(np.abs(diffs - np.median(diffs))))
    return float(1.4826 * mad / np.sqrt(2.0))


def _current_levels(currents: FloatArray) -> tuple[float, ...]:
    finite = currents[np.isfinite(currents)]
    if finite.size == 0:
        return ()
    levels, counts = np.unique(np.round(finite * 2.0) / 2.0, return_counts=True)
    return tuple(float(lv) for lv, c in zip(levels, counts, strict=True) if c >= 2)


def _fraction(flags: Iterable[bool]) -> float:
    arr = np.fromiter(flags, dtype=bool)
    return float(arr.mean()) if arr.size else _NAN


def battery_evidence(record: BatteryRecord) -> BatteryEvidence:
    """Compute :class:`BatteryEvidence` for one cell from its raw record.

    A discharge is paired with a charge only when the charge is the last charge/discharge
    operation before it (impedance measurements in between are allowed).

    Args:
        record: A parsed battery.

    Returns:
        The evidence; fields that cannot be computed are NaN (or empty).
    """
    discharges: list[Segment] = []
    preceding_charge: list[Segment | None] = []
    last_charge: Segment | None = None
    for seg in record.segments:
        if seg.kind == "charge":
            last_charge = seg
        elif seg.kind == "discharge":
            discharges.append(seg)
            preceding_charge.append(last_charge)
            last_charge = None

    capacity = pd.Series([s.capacity_ah for s in discharges], dtype=object)
    labels = label_battery(capacity)
    valid = labels["qc_valid"].to_numpy(dtype=bool)
    soh_valid = labels["soh_pct"].to_numpy(dtype=np.float64)[valid]
    reasons = labels["qc_reason"].value_counts()
    valid_idx = np.flatnonzero(valid)

    ambient = sorted({s.ambient_c for s in discharges if s.ambient_c is not None})
    currents = np.array([_load_current(s) for s in discharges], dtype=np.float64)
    zero_frac = np.array([_zero_current_fraction(s) for s in discharges], dtype=np.float64)
    v_min = np.array(
        [float(s.voltage_v.min()) if s.n_samples else _NAN for s in discharges], dtype=np.float64
    )
    return BatteryEvidence(
        battery_id=record.battery_id,
        n_discharge=len(discharges),
        n_valid=int(valid.sum()),
        qc_counts={str(k): int(v) for k, v in reasons.items()},
        c0_ah=float(labels["c0_ah"].iloc[0]) if len(labels) else _NAN,
        final_soh_pct=float(np.median(soh_valid[-5:])) if soh_valid.size else _NAN,
        min_soh_pct=float(soh_valid.min()) if soh_valid.size else _NAN,
        frac_above_reference=_fraction(soh_valid > _SOH_ABOVE_REFERENCE_PCT),
        label_noise_pct=_robust_noise(soh_valid),
        ambient_levels_c=tuple(float(a) for a in ambient),
        discharge_current_levels_a=_current_levels(currents),
        frac_zero_current_under_load=(
            float(np.nanmedian(zero_frac)) if np.isfinite(zero_frac).any() else _NAN
        ),
        min_voltage_median_v=float(np.nanmedian(v_min)) if np.isfinite(v_min).any() else _NAN,
        frac_discharge_window=_fraction(
            _discharge_window_observable(discharges[i]) for i in valid_idx
        ),
        frac_charge_window=_fraction(
            _charge_window_observable(preceding_charge[i], _CHG_WINDOW_400_START_V)
            for i in valid_idx
        ),
        frac_charge_window_390=_fraction(
            _charge_window_observable(preceding_charge[i], _CHG_WINDOW_390_START_V)
            for i in valid_idx
        ),
    )


def assess_usability(
    protocol: BatteryProtocol,
    evidence: BatteryEvidence,
    *,
    group_c0_ah: float | None = None,
) -> list[Criterion]:
    """Apply the inclusion criteria listed in the module docstring.

    Args:
        protocol: Registry entry of the cell (only its ``discharge_mode`` is used).
        evidence: Facts computed by :func:`battery_evidence`.
        group_c0_ah: Median ``C0`` of the cell's campaign group (the cell included), or
            ``None`` to skip the peer comparison (groups with fewer than three cells).

    Returns:
        Names of the failed criteria, in :data:`USABILITY_CRITERIA` order; an empty list
        means usable.

    Raises:
        ValueError: If ``group_c0_ah`` is given but not a positive finite number.
    """
    if group_c0_ah is not None and not (np.isfinite(group_c0_ah) and group_c0_ah > 0):
        raise ValueError(f"group_c0_ah must be positive and finite, got {group_c0_ah!r}")
    failed: list[Criterion] = []
    single_condition = (
        len(evidence.ambient_levels_c) == 1
        and len(evidence.discharge_current_levels_a) == 1
        and not evidence.frac_zero_current_under_load > MAX_ZERO_CURRENT_FRACTION
    )
    if protocol.discharge_mode != "cc" or not single_condition:
        failed.append("protocol")
    if evidence.n_valid < MIN_VALID_CYCLES:
        failed.append("history")
    c0_off = group_c0_ah is not None and not (
        abs(evidence.c0_ah / group_c0_ah - 1.0) <= MAX_C0_DEVIATION
    )
    if (
        not np.isfinite(evidence.c0_ah)
        or not evidence.frac_above_reference <= MAX_FRAC_ABOVE_REFERENCE
        or c0_off
    ):
        failed.append("reference")
    coverage = max(
        np.nan_to_num(evidence.frac_discharge_window), np.nan_to_num(evidence.frac_charge_window)
    )
    if coverage < MIN_WINDOW_COVERAGE:
        failed.append("windows")
    if not (
        evidence.label_noise_pct <= MAX_LABEL_NOISE_PCT
        and 100.0 - evidence.final_soh_pct >= MIN_FADE_PCT
    ):
        failed.append("labels")
    return failed


def evidence_table(root: Path, battery_ids: Iterable[str] | None = None) -> pd.DataFrame:
    """Recompute the usability evidence from the raw files.

    Loads every requested battery (about 0.5 s per file), so this is a maintenance tool, not
    part of the feature pipeline.

    Args:
        root: NASA data directory.
        battery_ids: Restrict to these cells (default: all 34). Group medians of ``C0`` are
            computed over the loaded cells only.

    Returns:
        One row per battery: the :class:`BatteryEvidence` fields, ``group``, ``condition``,
        ``c0_rated_pct``, ``group_c0_ah`` (NaN for groups under three cells), the registry
        decision ``usable`` and the rule outcome ``rule_failures`` (comma-separated) and
        ``rule_usable``.
    """
    ids = list(battery_ids) if battery_ids is not None else list(NASA_BATTERY_IDS)
    evidence = {rec.battery_id: battery_evidence(rec) for rec in iter_batteries(root, ids)}
    group_c0: dict[str, list[float]] = {}
    for battery_id, ev in evidence.items():
        if np.isfinite(ev.c0_ah):
            group_c0.setdefault(get_protocol(battery_id).group, []).append(ev.c0_ah)
    rows = []
    for battery_id, ev in evidence.items():
        protocol = get_protocol(battery_id)
        peers = group_c0.get(protocol.group, [])
        group_median = float(np.median(peers)) if len(peers) >= _MIN_GROUP_SIZE else None
        failures = assess_usability(protocol, ev, group_c0_ah=group_median)
        row = asdict(ev)
        row.update(
            group=protocol.group,
            condition=protocol.condition,
            c0_rated_pct=100.0 * ev.c0_ah / RATED_CAPACITY_AH,
            group_c0_ah=_NAN if group_median is None else group_median,
            usable=protocol.usable,
            rule_failures=",".join(failures),
            rule_usable=not failures,
        )
        rows.append(row)
    logger.info("Computed usability evidence for %d batteries", len(rows))
    return pd.DataFrame(rows)
