"""Internal signal-processing helpers shared by the feature extractors.

Everything here is pure and NumPy-only. The helpers encode the conventions
that make the health indicators robust to the quirks of real cycler data:

* Non-finite samples are dropped and samples are sorted by time; repeated
  timestamps keep their first occurrence (:func:`sanitize`).
* Current direction is inferred from the net transferred charge, so the
  extractors work with either sign convention and ignore short opposite-sign
  glitches such as the -3.4 A pulse at the start of every NASA charge record
  (:func:`directed_current`).
* Threshold crossings are located with linear interpolation between the two
  bracketing samples instead of snapping to a sample, which removes the
  quantisation error of coarse logging (NASA discharges are sampled every
  ~19 s, i.e. ~10 mAh per sample at 2 A).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from battery_soh._typing import FloatArray
from battery_soh.data.schema import Segment, SegmentKind

NAN = float("nan")

#: Segments with fewer usable samples than this yield all-NaN features.
MIN_SAMPLES = 10


@dataclass(frozen=True)
class Series:
    """Sanitised arrays of one segment: finite, time-ordered, unique timestamps."""

    t: FloatArray
    v: FloatArray
    i: FloatArray
    temp: FloatArray

    @property
    def n(self) -> int:
        """Number of samples."""
        return int(self.t.size)


@dataclass(frozen=True)
class Crossing:
    """First crossing of a level by a sampled signal.

    Attributes:
        index: Index of the first sample at or beyond the level.
        time: Linearly interpolated crossing time between ``index - 1`` and ``index``.
    """

    index: int
    time: float


def require_kind(seg: Segment, kind: SegmentKind) -> None:
    """Raise ``ValueError`` unless ``seg`` is of the expected kind."""
    if not isinstance(seg, Segment):
        raise ValueError(f"expected a Segment, got {type(seg).__name__}")
    if seg.kind != kind:
        raise ValueError(f"expected a {kind!r} segment, got {seg.kind!r}")


def sanitize(seg: Segment) -> Series:
    """Return finite, time-sorted samples with duplicate timestamps removed.

    Rows whose time, voltage or current is not finite are dropped. Temperature
    may stay NaN (only temperature features are affected). Sorting is stable,
    so for repeated timestamps the first recorded sample wins.
    """
    t, v, i, temp = seg.time_s, seg.voltage_v, seg.current_a, seg.temperature_c
    keep = np.isfinite(t) & np.isfinite(v) & np.isfinite(i)
    if not bool(keep.all()):
        t, v, i, temp = t[keep], v[keep], i[keep], temp[keep]
    if t.size > 1:
        dt = np.diff(t)
        if bool((dt < 0).any()):
            order = np.argsort(t, kind="stable")
            t, v, i, temp = t[order], v[order], i[order], temp[order]
            dt = np.diff(t)
        if bool((dt == 0).any()):
            first = np.concatenate(([True], dt > 0))
            t, v, i, temp = t[first], v[first], i[first], temp[first]
    return Series(t=t, v=v, i=i, temp=temp)


def directed_current(series: Series) -> FloatArray:
    """Return the current oriented so that the segment's dominant direction is positive.

    The orientation is the sign of the net transferred charge (trapezoidal
    integral of current over time). A charge therefore has positive current
    under either sign convention, and a discharge has a positive load current,
    while brief opposite-sign glitches stay negative and are ignored by the
    threshold logic of the extractors.
    """
    if series.n < 2:
        return series.i.copy()
    net = float(np.trapezoid(series.i, series.t))
    return series.i if net >= 0.0 else -series.i


def reference_current(i_dir: FloatArray) -> float:
    """Robust magnitude of the main current plateau (CC charge or discharge load).

    Uses the maximum of a 3-sample running median, which ignores isolated
    spikes but still works when the load occupies only a small part of the
    record. Returns NaN when no positive current is present.
    """
    if i_dir.size < 3:
        return NAN
    a, b, c = i_dir[:-2], i_dir[1:-1], i_dir[2:]
    running_median = np.maximum(np.minimum(a, b), np.minimum(np.maximum(a, b), c))
    level = float(running_median.max())
    return level if level > 0.0 else NAN


def first_crossing(
    t: FloatArray,
    x: FloatArray,
    level: float,
    *,
    rising: bool,
    start: int = 0,
    stop: int | None = None,
) -> Crossing | None:
    """Locate the first crossing of ``level`` by ``x`` within ``[start, stop)``.

    A crossing counts only if it is *observed*: the first sample at or beyond
    the level must be preceded (inside the search range) by a sample strictly
    on the other side. A signal that already starts beyond the level therefore
    returns ``None``, which callers turn into NaN features.

    Args:
        t: Sample times (strictly increasing).
        x: Signal values.
        level: Threshold to cross.
        rising: ``True`` to search for ``x >= level``, ``False`` for ``x <= level``.
        start: First index of the search range.
        stop: One past the last index of the search range (default: end).

    Returns:
        The crossing, or ``None`` if the level is not crossed inside the range.
    """
    stop = x.size if stop is None else min(stop, x.size)
    if stop - start < 2:
        return None
    window = x[start:stop]
    hits = np.flatnonzero(window >= level if rising else window <= level)
    if hits.size == 0 or hits[0] == 0:
        return None
    j = start + int(hits[0])
    x0, x1 = float(x[j - 1]), float(x[j])
    t0, t1 = float(t[j - 1]), float(t[j])
    frac = (level - x0) / (x1 - x0) if x1 != x0 else 1.0
    return Crossing(index=j, time=t0 + frac * (t1 - t0))


def interp_at(t: FloatArray, y: FloatArray, tq: float) -> float:
    """Linearly interpolate ``y(t)`` at a single time ``tq`` (NaN outside the range)."""
    if t.size == 0 or not (t[0] <= tq <= t[-1]):
        return NAN
    return float(np.interp(tq, t, y))


def integrate_between(t: FloatArray, y: FloatArray, t0: float, t1: float) -> float:
    """Trapezoidal integral of the piecewise-linear ``y(t)`` over ``[t0, t1]``.

    The integrand is linearly interpolated at the boundaries, so the result is
    exact for piecewise-linear signals and does not depend on where samples fall.
    """
    if not (t.size >= 2 and t[0] <= t0 <= t1 <= t[-1]):
        return NAN
    inner = (t > t0) & (t < t1)
    tt = np.concatenate(([t0], t[inner], [t1]))
    yy = np.concatenate(([np.interp(t0, t, y)], y[inner], [np.interp(t1, t, y)]))
    return float(np.trapezoid(yy, tt))


def cumulative_charge_ah(t: FloatArray, i: FloatArray) -> FloatArray:
    """Cumulative trapezoidal charge in Ah, starting at 0 at the first sample."""
    if t.size == 0:
        return np.empty(0)
    increments = 0.5 * (i[1:] + i[:-1]) * np.diff(t)
    return np.concatenate(([0.0], np.cumsum(increments))) / 3600.0


def finite_or_nan(value: float | None) -> float:
    """Return ``value`` as a float, or NaN when it is ``None`` or not finite."""
    if value is None:
        return NAN
    number = float(value)
    return number if np.isfinite(number) else NAN
