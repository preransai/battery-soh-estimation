"""Typed containers for raw cycling data.

A :class:`Segment` is one recorded operation (a charge, a discharge or an
impedance measurement) with its raw arrays. Feature extraction only ever sees
``Segment`` objects, which is what lets the training pipeline and the API share
exactly the same code path (train/serve parity).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np

from battery_soh._typing import FloatArray

SegmentKind = Literal["charge", "discharge", "impedance"]


def _as_float_array(values: object) -> FloatArray:
    return np.asarray(values, dtype=np.float64).ravel()


@dataclass(frozen=True)
class Segment:
    """One charge, discharge or impedance operation.

    Attributes:
        kind: Operation type.
        time_s: Seconds since the start of the operation (non-decreasing).
        voltage_v: Terminal voltage in volts.
        current_a: Current in amperes, NASA sign convention (discharge < 0, charge > 0).
        temperature_c: Cell surface temperature in degrees Celsius.
        start_time: Wall-clock start of the operation, when known.
        ambient_c: Ambient (chamber) temperature, when known.
        capacity_ah: Discharge capacity from the dataset. Label-side information:
            it defines SOH and must never be used as a model input.
        re_ohm: Electrolyte resistance from EIS (impedance segments only).
        rct_ohm: Charge-transfer resistance from EIS (impedance segments only).
    """

    kind: SegmentKind
    time_s: FloatArray = field(default_factory=lambda: np.empty(0))
    voltage_v: FloatArray = field(default_factory=lambda: np.empty(0))
    current_a: FloatArray = field(default_factory=lambda: np.empty(0))
    temperature_c: FloatArray = field(default_factory=lambda: np.empty(0))
    start_time: datetime | None = None
    ambient_c: float | None = None
    capacity_ah: float | None = None
    re_ohm: float | None = None
    rct_ohm: float | None = None

    def __post_init__(self) -> None:
        arrays = {
            "time_s": _as_float_array(self.time_s),
            "voltage_v": _as_float_array(self.voltage_v),
            "current_a": _as_float_array(self.current_a),
            "temperature_c": _as_float_array(self.temperature_c),
        }
        for name, arr in arrays.items():
            object.__setattr__(self, name, arr)
        lengths = {name: arr.size for name, arr in arrays.items()}
        if len(set(lengths.values())) > 1:
            raise ValueError(f"Segment arrays must have equal length, got {lengths}")

    @property
    def n_samples(self) -> int:
        """Number of samples in the time series."""
        return int(self.time_s.size)


@dataclass(frozen=True)
class BatteryRecord:
    """All recorded operations of one cell, in acquisition order."""

    battery_id: str
    segments: tuple[Segment, ...]
    source_path: Path | None = None

    def of_kind(self, kind: SegmentKind) -> tuple[Segment, ...]:
        """Return the segments of one kind, preserving order."""
        return tuple(s for s in self.segments if s.kind == kind)
