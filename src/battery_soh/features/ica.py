"""Incremental capacity analysis (ICA): dQ/dV on a uniform voltage grid.

The v1 implementation differentiated raw, voltage-sorted samples that still
contained the post-cutoff relaxation, ran ``find_peaks`` with a prominence
three orders of magnitude below the signal, ordered peaks by prominence (so
"peak 1" and "peak 2" swapped between cycles) and padded missing peaks with
0 V. The audit measured the consequence: the peak-voltage feature flipped its
correlation sign between batteries.

This module does it the textbook way:

1. Take only the constant-current samples (the caller passes them).
2. Enforce a monotone voltage trajectory with an isotonic (least-squares
   monotone) fit, which removes measurement noise without the upward bias of
   a running-maximum envelope.
3. Interpolate the cumulative charge Q onto a uniform voltage grid.
4. Differentiate with a Savitzky-Golay filter (local polynomial derivative)
   on that grid, which is correctly scaled because the grid is uniform.
5. Report the peak: the highest interior local maximum when it is within
   ``tie_tolerance`` (3 %) of the window maximum, else the window maximum,
   with sub-grid parabolic refinement for an interior peak. The tie rule
   stops the label jumping to a rising window edge that is marginally higher
   than the real peak (B0006 cycles 86-88: interior 3.05 Ah/V at 4.05 V vs
   3.10 Ah/V at the 4.18 V edge), the same failure v1 had between two peaks.
   When the window holds no interior peak (at 43 degC the charge peak moves
   below 3.90 V, at 4 degC above 4.18 V) the edge value and edge voltage are
   reported, censored at the window boundary instead of replaced by NaN or 0:
   the edge height still falls with capacity.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

import numpy as np
from scipy.optimize import isotonic_regression
from scipy.signal import savgol_filter

from battery_soh._typing import FloatArray
from battery_soh.features._signal import NAN

MonotoneMethod = Literal["isotonic", "cummax"]

#: Default Savitzky-Golay window (grid points) and polynomial order.
SG_WINDOW = 9
SG_POLYORDER = 2
#: An interior local maximum within this fraction of the window maximum is the peak.
TIE_TOLERANCE = 0.03


def voltage_grid(lo: float, hi: float, step: float) -> FloatArray:
    """Build an inclusive, uniform voltage grid ``lo, lo + step, ..., hi``.

    Raises:
        ValueError: If the bounds or step are not finite, ``hi <= lo``,
            ``step <= 0`` or ``hi - lo`` is not (close to) a multiple of ``step``.
    """
    if not all(np.isfinite(x) for x in (lo, hi, step)):
        raise ValueError(f"voltage grid bounds must be finite, got {(lo, hi, step)}")
    if step <= 0 or hi <= lo:
        raise ValueError(f"invalid voltage grid (lo={lo}, hi={hi}, step={step})")
    n_steps = (hi - lo) / step
    if abs(n_steps - round(n_steps)) > 1e-6:
        raise ValueError(f"grid span {hi - lo:.6g} V is not a multiple of step {step:.6g} V")
    return np.linspace(lo, hi, round(n_steps) + 1)


def isotonic_increasing(y: FloatArray) -> FloatArray:
    """Least-squares non-decreasing fit (pool-adjacent-violators, unit weights).

    Args:
        y: 1-D samples.

    Returns:
        The non-decreasing sequence closest to ``y`` in the L2 sense. Blocks of
        pooled samples share one value (their mean).
    """
    y = np.asarray(y, dtype=np.float64)
    if y.size == 0:
        return y.copy()
    return np.asarray(isotonic_regression(y, increasing=True).x, dtype=np.float64)


def monotone_envelope(voltage: FloatArray, method: MonotoneMethod = "isotonic") -> FloatArray:
    """Return a non-decreasing version of ``voltage`` (see :func:`isotonic_increasing`).

    Args:
        voltage: Voltage samples in time order.
        method: ``"isotonic"`` (least-squares fit, unbiased under noise) or
            ``"cummax"`` (running maximum, as in the audit prototype).
    """
    if method == "isotonic":
        return isotonic_increasing(voltage)
    if method == "cummax":
        return np.maximum.accumulate(voltage)
    raise ValueError(f"unknown monotone method {method!r}")


def charge_on_grid(
    voltage: FloatArray,
    charge_ah: FloatArray,
    grid: FloatArray,
    *,
    monotone: MonotoneMethod = "isotonic",
) -> FloatArray:
    """Interpolate cumulative charge ``Q`` onto a voltage grid.

    The voltage is first made monotone; samples that share one fitted voltage
    (a pooled block) are represented by their mean charge, so ``Q(V)`` is a
    proper function. Returns an all-NaN array when the (monotone) voltage does
    not span the whole grid.
    """
    out = np.full(grid.size, np.nan)
    if voltage.size < 2 or voltage.size != charge_ah.size:
        return out
    if not (np.isfinite(voltage).all() and np.isfinite(charge_ah).all()):
        return out
    v_mono = monotone_envelope(voltage, monotone)
    if not (v_mono[0] <= grid[0] and v_mono[-1] >= grid[-1]):
        return out
    v_unique, inverse = np.unique(v_mono, return_inverse=True)
    q_mean = np.bincount(inverse, weights=charge_ah) / np.bincount(inverse)
    return np.interp(grid, v_unique, q_mean)


def incremental_capacity(
    voltage: FloatArray,
    charge_ah: FloatArray,
    grid: FloatArray,
    *,
    window: int = SG_WINDOW,
    polyorder: int = SG_POLYORDER,
    monotone: MonotoneMethod = "isotonic",
) -> FloatArray:
    """Incremental capacity dQ/dV (Ah/V) on a uniform voltage grid.

    Args:
        voltage: Terminal voltage of a constant-current segment, in time order.
        charge_ah: Cumulative charge at the same samples (Ah, increasing).
        grid: Uniform, increasing voltage grid (see :func:`voltage_grid`).
        window: Savitzky-Golay window length in grid points (odd).
        polyorder: Savitzky-Golay polynomial order (< ``window``).
        monotone: Monotone-envelope method, see :func:`monotone_envelope`.

    Returns:
        dQ/dV at every grid voltage, or an all-NaN array when the segment does
        not cover the grid.

    Raises:
        ValueError: If the grid is not uniform and increasing or the filter
            parameters are invalid.
    """
    grid = np.asarray(grid, dtype=np.float64)
    validate_ica_params(grid, window, polyorder, monotone)
    q_grid = charge_on_grid(
        np.asarray(voltage, dtype=np.float64),
        np.asarray(charge_ah, dtype=np.float64),
        grid,
        monotone=monotone,
    )
    if not bool(np.isfinite(q_grid).all()):
        return q_grid
    step = float(grid[1] - grid[0])
    return np.asarray(_savgol_derivative(grid.size, window, polyorder) @ q_grid / step)


@lru_cache(maxsize=32)
def _savgol_derivative(n: int, window: int, polyorder: int) -> FloatArray:
    """Matrix ``D`` with ``D @ q == savgol_filter(q, window, polyorder, deriv=1, mode="interp")``.

    The Savitzky-Golay filter (including its polynomial edge fits) is linear,
    so applying it to the identity yields its matrix once; every cycle then
    costs one small matrix-vector product instead of fresh edge polyfits.
    """
    matrix = savgol_filter(np.eye(n), window, polyorder, deriv=1, delta=1.0, mode="interp", axis=0)
    out = np.asarray(matrix, dtype=np.float64)
    out.setflags(write=False)
    return out


def ica_peak(
    dqdv: FloatArray, grid: FloatArray, *, tie_tolerance: float = TIE_TOLERANCE
) -> tuple[float, float]:
    """Height and voltage of the incremental-capacity peak inside the grid window.

    The peak is the highest interior local maximum if its height is at least
    ``(1 - tie_tolerance)`` times the window maximum; otherwise the window
    maximum itself. An interior peak is refined with a parabola through it and
    its two neighbours (removes the grid quantisation). A maximum on the first
    or last grid point means the peak lies outside the window: the edge value
    and edge voltage are returned (censored at the boundary), which keeps the
    height a monotone health indicator for cells whose peak drifts out of the
    window with temperature or ageing.

    Args:
        dqdv: Output of :func:`incremental_capacity`.
        grid: The voltage grid it was computed on.
        tie_tolerance: Relative height margin within which an interior peak is
            preferred over a (higher) window edge; 0 selects the plain maximum.

    Returns:
        ``(height_ah_per_v, voltage_v)``, or ``(nan, nan)`` when ``dqdv`` is not
        finite (the segment did not cover the window).

    Raises:
        ValueError: If ``tie_tolerance`` is not in ``[0, 1)``.
    """
    if not 0.0 <= tie_tolerance < 1.0:
        raise ValueError(f"tie_tolerance must be in [0, 1), got {tie_tolerance}")
    if dqdv.size < 3 or dqdv.size != grid.size or not bool(np.isfinite(dqdv).all()):
        return NAN, NAN
    top = float(dqdv.max())
    inner = np.arange(1, dqdv.size - 1)
    local = inner[(dqdv[1:-1] >= dqdv[:-2]) & (dqdv[1:-1] > dqdv[2:])]
    local = local[dqdv[local] >= top - tie_tolerance * abs(top)]
    if local.size == 0:  # no interior peak: censored at the window edge
        j = int(np.argmax(dqdv))
        return float(dqdv[j]), float(grid[j])
    j = int(local[np.argmax(dqdv[local])])
    y0, y1, y2 = float(dqdv[j - 1]), float(dqdv[j]), float(dqdv[j + 1])
    curvature = y0 - 2.0 * y1 + y2
    offset = 0.5 * (y0 - y2) / curvature if curvature < 0.0 else 0.0
    step = float(grid[j + 1] - grid[j])
    height = y1 - 0.25 * (y0 - y2) * offset
    return height, float(grid[j]) + offset * step


def validate_ica_params(
    grid: FloatArray, window: int, polyorder: int, monotone: MonotoneMethod = "isotonic"
) -> None:
    """Check the ICA grid and filter settings.

    Raises:
        ValueError: If the grid is not uniform/increasing, the Savitzky-Golay
            settings are invalid, or ``monotone`` is unknown.
    """
    _validate_filter(np.asarray(grid, dtype=np.float64), window, polyorder)
    if monotone not in ("isotonic", "cummax"):
        raise ValueError(f"unknown monotone method {monotone!r}")


def _validate_filter(grid: FloatArray, window: int, polyorder: int) -> None:
    if grid.ndim != 1 or grid.size < 3:
        raise ValueError("ICA grid must be 1-D with at least 3 points")
    steps = np.diff(grid)
    if not bool((steps > 0).all()) or not np.allclose(steps, steps[0], rtol=1e-6, atol=1e-12):
        raise ValueError("ICA grid must be uniform and strictly increasing")
    if window % 2 != 1 or window < 3:
        raise ValueError(f"Savitzky-Golay window must be odd and >= 3, got {window}")
    if not 1 <= polyorder < window:
        # polyorder 0 would make the first derivative identically zero.
        raise ValueError(f"polyorder must satisfy 1 <= polyorder < window, got {polyorder}")
    if window > grid.size:
        raise ValueError(f"Savitzky-Golay window {window} exceeds the grid size {grid.size}")
