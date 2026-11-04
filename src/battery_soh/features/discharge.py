"""Health indicators from a partial discharge.

Two families, both observable long before the discharge cutoff:

* **Voltage-window indicators.** The charge delivered while the loaded
  terminal voltage falls from ``v_high`` to ``v_low`` (default 3.80 -> 3.50 V,
  about 40-50 % of a NASA 2 A discharge) is proportional to the usable
  capacity. Integrating the measured current between interpolated crossing
  times makes the indicator exact under constant current and still correct
  under a varying load.
* **Ohmic step at load-on.** ``R = dV / dI`` across the first sample pair that
  brackets the load switch-on. The v1 code took the arg-max of ``|dI|``,
  which is the load *removal* in 19-64 % of cycles, and then returned a 0.0
  placeholder; here the onset is located explicitly and failures are NaN.

Only samples under load are used for the window, so the post-cutoff rest
(where the voltage relaxes back up) can never leak into a feature.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from battery_soh._typing import FloatArray
from battery_soh.data.schema import Segment
from battery_soh.features._signal import (
    MIN_SAMPLES,
    NAN,
    Series,
    directed_current,
    first_crossing,
    integrate_between,
    interp_at,
    reference_current,
    require_kind,
    sanitize,
)

logger = logging.getLogger(__name__)

#: Keys returned by :func:`discharge_window_features`.
WINDOW_FEATURES: tuple[str, ...] = (
    "dis_q_380_350_ah",
    "dis_time_380_350_s",
    "dis_temp_rise_win_c",
)

#: Keys returned by :func:`discharge_features` (registry order).
DISCHARGE_FEATURES: tuple[str, ...] = (
    "dis_q_380_350_ah",
    "dis_time_380_350_s",
    "dis_r0_ohm",
    "dis_temp_rise_win_c",
    "ctx_discharge_current_a",
)

#: Samples whose directed current exceeds this fraction of the load plateau are "under load".
LOAD_ON_FRACTION = 0.5
#: A load plateau below this current (A) is treated as no load (sensor offset or a rest record).
MIN_LOAD_CURRENT_A = 0.1
#: Minimum share of the load current that must be switched between the two
#: samples used for the ohmic step (rejects slow current ramps).
MIN_STEP_FRACTION = 0.5


@dataclass(frozen=True)
class LoadPeriod:
    """The load-on part of a discharge record.

    Attributes:
        first: Index of the first sample under load.
        last: Index of the last sample under load.
        current_a: Median load current (A, positive).
    """

    first: int
    last: int
    current_a: float


@dataclass(frozen=True)
class _Prepared:
    series: Series
    i_load: FloatArray
    load: LoadPeriod


def find_load(i_load: FloatArray) -> LoadPeriod | None:
    """Locate the load-on period from the directed current (``None`` if there is no load)."""
    i_ref = reference_current(i_load)
    if not (np.isfinite(i_ref) and i_ref >= MIN_LOAD_CURRENT_A):
        return None
    on = np.flatnonzero(i_load > LOAD_ON_FRACTION * i_ref)
    if on.size < 2:
        return None
    return LoadPeriod(first=int(on[0]), last=int(on[-1]), current_a=float(np.median(i_load[on])))


def _prepare(seg: Segment) -> _Prepared | None:
    require_kind(seg, "discharge")
    series = sanitize(seg)
    if series.n < MIN_SAMPLES:
        return None
    i_load = directed_current(series)
    load = find_load(i_load)
    if load is None:
        return None
    return _Prepared(series=series, i_load=i_load, load=load)


def _validate_window(v_high: float, v_low: float, settle_s: float) -> None:
    if not (np.isfinite(v_high) and np.isfinite(v_low) and v_low < v_high):
        raise ValueError(f"need finite v_low < v_high, got v_high={v_high}, v_low={v_low}")
    if not (np.isfinite(settle_s) and settle_s >= 0):
        raise ValueError(f"settle_s must be finite and >= 0, got {settle_s}")


def _window(
    prep: _Prepared | None, v_high: float, v_low: float, settle_s: float
) -> dict[str, float]:
    out = dict.fromkeys(WINDOW_FEATURES, NAN)
    if prep is None:
        return out
    s, load = prep.series, prep.load
    stop = load.last + 1  # never look at the post-cutoff rest
    start = first_crossing(s.t, s.v, v_high, rising=False, start=load.first, stop=stop)
    if start is None or start.time < s.t[load.first] + settle_s:
        return out
    end = first_crossing(s.t, s.v, v_low, rising=False, start=start.index, stop=stop)
    if end is None:
        return out
    out["dis_q_380_350_ah"] = integrate_between(s.t, prep.i_load, start.time, end.time) / 3600.0
    out["dis_time_380_350_s"] = end.time - start.time
    out["dis_temp_rise_win_c"] = interp_at(s.t, s.temp, end.time) - interp_at(
        s.t, s.temp, start.time
    )
    return out


def _ohmic(prep: _Prepared | None) -> float:
    if prep is None:
        return NAN
    i_load, load = prep.i_load, prep.load
    k0 = load.first  # first sample above LOAD_ON_FRACTION of the load: the onset, never the removal
    if k0 == 0:
        return NAN  # record starts under load: no pre-load sample
    d_i = float(i_load[k0] - i_load[k0 - 1])
    if d_i < MIN_STEP_FRACTION * load.current_a:
        logger.debug("load onset is a ramp (dI=%.3f A), no ohmic step", d_i)
        return NAN
    resistance = float(prep.series.v[k0 - 1] - prep.series.v[k0]) / d_i
    return resistance if np.isfinite(resistance) and resistance > 0 else NAN


def discharge_window_features(
    seg: Segment,
    *,
    v_high: float = 3.80,
    v_low: float = 3.50,
    settle_s: float = 30.0,
) -> dict[str, float]:
    """Charge, duration and temperature rise of the ``v_high -> v_low`` load window.

    The window is measured on the load-on samples only. It is valid when the
    loaded voltage crosses ``v_high`` at least ``settle_s`` seconds after the
    first loaded sample (so the cell started above the window and the window
    does not begin inside the switch-on transient) and then crosses ``v_low``
    while still under load. Otherwise every value is NaN.

    Args:
        seg: A ``"discharge"`` segment. NASA convention (discharge current
            negative) is expected; the load direction is inferred from the net
            transferred charge, so positive discharge currents work too.
        v_high: Upper (start) voltage of the window (V).
        v_low: Lower (end) voltage of the window (V).
        settle_s: Minimum time between load-on and the ``v_high`` crossing (s).

    Returns:
        ``dis_q_380_350_ah`` (Ah, = integral of |I| dt), ``dis_time_380_350_s``
        (s) and ``dis_temp_rise_win_c`` (degC). The keys keep their registry
        names when the window is changed (window-sweep experiment).

    Raises:
        ValueError: If ``seg`` is not a discharge segment or the window is invalid.
    """
    _validate_window(v_high, v_low, settle_s)
    return _window(_prepare(seg), v_high, v_low, settle_s)


def ohmic_resistance(seg: Segment) -> float:
    """Apparent ohmic resistance from the voltage step at load-on (ohm).

    ``R = (V[k0-1] - V[k0]) / (I[k0] - I[k0-1])`` with currents in the load
    direction and ``k0`` the first sample whose load current exceeds
    :data:`LOAD_ON_FRACTION` of the load plateau
    (:func:`~battery_soh.features._signal.reference_current`, see
    :func:`find_load`). With the usual ~0 A pre-load sample this equals the
    registry form ``(V[k0-1] - V[k0]) / (|I[k0]| - |I[k0-1]|)``. The two samples
    are one logging interval apart (~19 s on NASA discharges), so the value
    includes fast polarisation as well as the pure ohmic drop; it is
    nevertheless a strong ageing indicator.

    Returns NaN when there is no load, the record starts under load (no
    pre-load sample), the current ramps instead of stepping, or the result is
    not positive.

    Raises:
        ValueError: If ``seg`` is not a discharge segment.
    """
    return _ohmic(_prepare(seg))


def load_current(seg: Segment) -> float:
    """Median absolute load current of a discharge (A), NaN if there is no load.

    Raises:
        ValueError: If ``seg`` is not a discharge segment.
    """
    prep = _prepare(seg)
    return NAN if prep is None else prep.load.current_a


def discharge_features(
    seg: Segment,
    *,
    v_high: float = 3.80,
    v_low: float = 3.50,
    settle_s: float = 30.0,
) -> dict[str, float]:
    """All discharge-side indicators plus the load-current context feature.

    Equivalent to calling :func:`discharge_window_features`,
    :func:`ohmic_resistance` and :func:`load_current`, but sanitises the
    segment once.

    Returns:
        Mapping with the keys of :data:`DISCHARGE_FEATURES`.

    Raises:
        ValueError: If ``seg`` is not a discharge segment or the window is invalid.
    """
    _validate_window(v_high, v_low, settle_s)
    prep = _prepare(seg)
    window = _window(prep, v_high, v_low, settle_s)
    return {
        "dis_q_380_350_ah": window["dis_q_380_350_ah"],
        "dis_time_380_350_s": window["dis_time_380_350_s"],
        "dis_r0_ohm": _ohmic(prep),
        "dis_temp_rise_win_c": window["dis_temp_rise_win_c"],
        "ctx_discharge_current_a": NAN if prep is None else prep.load.current_a,
    }
