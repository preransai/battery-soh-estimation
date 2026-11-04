"""Electrochemical impedance spectroscopy (EIS) indicators.

NASA fitted an equivalent-circuit model to every EIS sweep and stored the
electrolyte resistance ``Re`` and the charge-transfer resistance ``Rct``.
Both grow as the cell ages (audit: |r| = 0.85-0.95 with the next capacity).
The v1 loader skipped these 278 measurements per battery entirely.

Joining an EIS measurement to a discharge cycle must be *as-of*: a cycle may
only use the most recent sweep recorded before it (never a later one, and no
back-filling of the first cycles); that join lives in the table builder.
"""

from __future__ import annotations

import math

from battery_soh.data.schema import Segment
from battery_soh.features._signal import NAN, require_kind

#: Keys produced from an impedance measurement (registry names).
EIS_FEATURES: tuple[str, ...] = ("eis_re_ohm", "eis_rct_ohm")


def _positive(value: float | None) -> float:
    if value is None:
        return NAN
    number = float(value)
    return number if math.isfinite(number) and number > 0 else NAN


def impedance_values(seg: Segment) -> tuple[float, float]:
    """Return ``(re_ohm, rct_ohm)`` of an impedance segment.

    Missing, non-finite or non-positive values (failed equivalent-circuit
    fits) become NaN.

    Raises:
        ValueError: If ``seg`` is not an impedance segment.
    """
    require_kind(seg, "impedance")
    return _positive(seg.re_ohm), _positive(seg.rct_ohm)


def eis_features(eis: tuple[float, float] | None) -> dict[str, float]:
    """Map an ``(re_ohm, rct_ohm)`` pair (or ``None``) to the EIS feature names.

    Raises:
        ValueError: If ``eis`` is not ``None`` or a pair of numbers.
    """
    if eis is None:
        return dict.fromkeys(EIS_FEATURES, NAN)
    try:
        re_ohm, rct_ohm = eis
        values = (float(re_ohm), float(rct_ohm))
    except (TypeError, ValueError):
        raise ValueError(f"eis must be a (re_ohm, rct_ohm) pair or None, got {eis!r}") from None
    return dict(zip(EIS_FEATURES, (_positive(v) for v in values), strict=True))
