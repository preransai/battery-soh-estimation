"""Health indicators from a partial constant-current / constant-voltage (CC-CV) charge.

All NASA PCoE cells were charged with the same protocol (1.5 A CC to 4.2 V,
then CV until 20 mA), so charge-side indicators are comparable across every
test group. As a cell ages its capacity fades and its resistance grows, so

* the CC voltage climbs faster: the time from 3.90 V to 4.19 V shrinks;
* the CV phase (current taper at 4.2 V) lasts longer;
* the incremental-capacity peak gets lower and moves to a higher voltage.

Only quantities observable inside a voltage window are used; nothing here
needs the full charge, and nothing reads the discharge capacity. The ICA is
computed on the CC samples only (from CC onset to the CC -> CV transition),
so neither the pre-charge rest, the CV taper nor the post-charge relaxation
can enter ``Q(V)``.
"""

from __future__ import annotations

import logging

import numpy as np

from battery_soh._typing import FloatArray
from battery_soh.data.schema import Segment
from battery_soh.features._signal import (
    MIN_SAMPLES,
    NAN,
    Series,
    cumulative_charge_ah,
    directed_current,
    first_crossing,
    interp_at,
    reference_current,
    require_kind,
    sanitize,
)
from battery_soh.features.ica import (
    SG_POLYORDER,
    SG_WINDOW,
    MonotoneMethod,
    ica_peak,
    incremental_capacity,
    validate_ica_params,
    voltage_grid,
)

logger = logging.getLogger(__name__)

#: Keys returned by :func:`charge_features` (registry names, registry order).
CHARGE_FEATURES: tuple[str, ...] = (
    "chg_cc_time_390_419_s",
    "chg_cc_time_400_419_s",
    "chg_cv_time_s",
    "chg_ica_peak_height_ah_per_v",
    "chg_ica_peak_voltage_v",
    "chg_temp_rise_cc_c",
)

#: Samples with directed current above this fraction of the CC level are "CC on".
CC_ON_FRACTION = 0.5
#: Inside a CC timing window the current may not dip below this fraction of
#: the CC level (a paused or derated charger would inflate the window time).
CC_MIN_FRACTION = 0.8
#: Number of samples after CC onset whose median defines the CC start voltage.
_START_SAMPLES = 3


def charge_features(
    seg: Segment,
    *,
    v_start: float = 3.90,
    v_end: float = 4.19,
    alt_v_start: float = 4.00,
    cv_cutoff_a: float = 0.1,
    ica_grid: tuple[float, float, float] = (3.90, 4.18, 0.005),
    start_margin_v: float = 0.05,
    alt_start_margin_v: float = 0.03,
    ica_start_margin_v: float = 0.02,
    ica_window: int = SG_WINDOW,
    ica_polyorder: int = SG_POLYORDER,
    ica_monotone: MonotoneMethod = "isotonic",
) -> dict[str, float]:
    """Compute the charge-side health indicators of one charge segment.

    Window definitions (the defaults reproduce the registry features):

    * ``chg_cc_time_390_419_s``: time for the CC voltage to rise ``v_start`` ->
      ``v_end``. Requires the CC start voltage (median of the first three CC
      samples) to be below ``v_start - start_margin_v`` (3.85 V), so the
      window never begins inside the switch-on transient.
    * ``chg_cc_time_400_419_s``: same for ``alt_v_start`` -> ``v_end`` with
      ``alt_start_margin_v`` (start below 3.97 V); available for more cells.
    * ``chg_cv_time_s``: from the first ``v_end`` crossing (CC -> CV) until the
      current first falls below ``cv_cutoff_a``.
    * ``chg_ica_peak_*``: maximum of dQ/dV on the CC samples over the uniform
      grid ``ica_grid = (lo, hi, step)`` (see :func:`ica.ica_peak`); needs the
      CC start below ``lo - ica_start_margin_v`` (3.88 V).
    * ``chg_temp_rise_cc_c``: temperature change across the main CC window.

    Crossing times are interpolated linearly between samples. Every quantity
    that cannot be measured (window not reached, start too high, charger
    paused inside the window, record too short) is NaN; this function never
    raises for bad data, only for invalid arguments. When the window
    parameters differ from the defaults, the keys keep their registry names
    (window-sweep experiment).

    Args:
        seg: A ``"charge"`` segment. NASA convention (charge current positive)
            is expected, but the charging direction is inferred from the net
            transferred charge, so the opposite convention gives the same result.
        v_start: Lower voltage of the main CC timing window (V).
        v_end: Upper voltage of the CC windows; also the CC -> CV transition (V).
        alt_v_start: Lower voltage of the alternative CC timing window (V).
        cv_cutoff_a: Current that ends the CV-time measurement (A).
        ica_grid: ``(lo, hi, step)`` of the ICA voltage grid (V).
        start_margin_v: Required margin of the CC start below ``v_start``.
        alt_start_margin_v: Required margin of the CC start below ``alt_v_start``.
        ica_start_margin_v: Required margin of the CC start below the grid start.
        ica_window: Savitzky-Golay window (grid points) of the ICA derivative.
        ica_polyorder: Savitzky-Golay polynomial order.
        ica_monotone: Monotone fit applied to the voltage before inverting ``Q(V)``.

    Returns:
        Mapping from the keys of :data:`CHARGE_FEATURES` to floats (NaN when
        not computable).

    Raises:
        ValueError: If ``seg`` is not a charge segment or a parameter is invalid.
    """
    require_kind(seg, "charge")
    grid = voltage_grid(*ica_grid)
    validate_ica_params(grid, ica_window, ica_polyorder, ica_monotone)
    _validate(
        v_start,
        v_end,
        alt_v_start,
        cv_cutoff_a,
        start_margin_v,
        alt_start_margin_v,
        ica_start_margin_v,
    )

    out = dict.fromkeys(CHARGE_FEATURES, NAN)
    series = sanitize(seg)
    if series.n < MIN_SAMPLES:
        return out
    i_chg = directed_current(series)
    i_cc = reference_current(i_chg)
    if not np.isfinite(i_cc) or i_cc <= cv_cutoff_a:
        logger.debug("charge segment has no CC plateau above the CV cutoff")
        return out

    k0 = int(np.flatnonzero(i_chg > CC_ON_FRACTION * i_cc)[0])  # CC onset
    v_cc_start = float(np.median(series.v[k0 : k0 + _START_SAMPLES]))
    # CC -> CV transition: first observed rise through v_end after CC onset.
    cv_start = first_crossing(series.t, series.v, v_end, rising=True, start=k0)
    if cv_start is None:
        return out
    j_end = cv_start.index

    for key, lo, margin in (
        ("chg_cc_time_390_419_s", v_start, start_margin_v),
        ("chg_cc_time_400_419_s", alt_v_start, alt_start_margin_v),
    ):
        if v_cc_start >= lo - margin:
            continue
        t_lo = _cc_window_start(series, i_chg, i_cc, lo, k0, j_end)
        if t_lo is None:
            continue
        out[key] = cv_start.time - t_lo
        if key == "chg_cc_time_390_419_s":
            out["chg_temp_rise_cc_c"] = interp_at(series.t, series.temp, cv_start.time) - interp_at(
                series.t, series.temp, t_lo
            )

    cv_end = first_crossing(series.t, i_chg, cv_cutoff_a, rising=False, start=j_end)
    if cv_end is not None:
        out["chg_cv_time_s"] = cv_end.time - cv_start.time

    if v_cc_start < grid[0] - ica_start_margin_v:
        cc = slice(k0, j_end + 1)  # CC samples only: onset .. first sample at/above v_end
        dqdv = incremental_capacity(
            series.v[cc],
            cumulative_charge_ah(series.t[cc], i_chg[cc]),
            grid,
            window=ica_window,
            polyorder=ica_polyorder,
            monotone=ica_monotone,
        )
        height, v_peak = ica_peak(dqdv, grid)
        out["chg_ica_peak_height_ah_per_v"] = height
        out["chg_ica_peak_voltage_v"] = v_peak
    return out


def _cc_window_start(
    series: Series,
    i_chg: FloatArray,
    i_cc: float,
    v_lo: float,
    k0: int,
    j_end: int,
) -> float | None:
    """Crossing time of ``v_lo`` inside the CC phase, or None if the window is invalid."""
    start = first_crossing(series.t, series.v, v_lo, rising=True, start=k0, stop=j_end + 1)
    if start is None:
        return None
    inside = i_chg[start.index : j_end]  # samples strictly between the two crossings
    if inside.size and float(inside.min()) < CC_MIN_FRACTION * i_cc:
        logger.debug("CC current dipped inside the %.2f V window; window rejected", v_lo)
        return None
    return start.time


def _validate(
    v_start: float,
    v_end: float,
    alt_v_start: float,
    cv_cutoff_a: float,
    *margins: float,
) -> None:
    values = (v_start, v_end, alt_v_start, cv_cutoff_a, *margins)
    if not all(np.isfinite(x) for x in values):
        raise ValueError(f"charge window parameters must be finite, got {values}")
    if not v_start < v_end:
        raise ValueError(f"v_start ({v_start}) must be below v_end ({v_end})")
    if not alt_v_start < v_end:
        raise ValueError(f"alt_v_start ({alt_v_start}) must be below v_end ({v_end})")
    if cv_cutoff_a <= 0:
        raise ValueError(f"cv_cutoff_a must be positive, got {cv_cutoff_a}")
    if any(m < 0 for m in margins):
        raise ValueError(f"start margins must be >= 0, got {margins}")
