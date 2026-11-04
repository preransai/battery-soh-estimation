"""Point and interval metrics, and per-battery summaries.

All error metrics are in the units of the label (SOH percentage points).
Conventions:

* ``bias`` is ``mean(y_pred - y_true)``: positive means the model
  *over*-estimates health, which is the dangerous direction for a battery.
* ``r2`` is NaN when it is undefined (fewer than two rows or a constant
  target) instead of the arbitrary 0.0/1.0 that scikit-learn substitutes.
* The interval score is the negatively oriented score of Gneiting & Raftery
  (2007, eq. 43): lower is better; it rewards narrow intervals and charges
  ``2 / alpha`` per unit of miss distance.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from battery_soh._typing import FloatArray
from battery_soh.config import RANDOM_SEED

__all__ = [
    "SUMMARY_ROWS",
    "bootstrap_ci",
    "interval_metrics",
    "interval_score",
    "regression_metrics",
    "round_significant",
    "summarize_by_group",
]


#: Labels of the aggregate rows appended by :func:`summarize_by_group`.
SUMMARY_ROWS: tuple[str, str, str] = ("mean", "worst", "pooled")

# How each metric is aggregated in the "worst" row of summarize_by_group.
_WORST_MAX = ("mae", "rmse", "max_abs_error", "mean_width", "interval_score")
_WORST_MIN = ("r2", "coverage")


def round_significant(value: Any, digits: int = 6) -> float | None:
    """Round to ``digits`` significant digits for strict-JSON reports.

    This is the precision of every number the audit, the benchmark and the model
    bundles record. ``None``, NaN and infinities become ``None`` (JSON ``null``).
    """
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return float(f"{number:.{digits}g}")


def _as_1d(values: ArrayLike, name: str) -> FloatArray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 2 and 1 in arr.shape:  # column/row vectors, e.g. PLS predictions
        arr = arr.reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {arr.shape}")
    return arr


def _validate_same_length(**arrays: FloatArray) -> int:
    lengths = {name: arr.size for name, arr in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"inputs must have equal length, got {lengths}")
    n = next(iter(lengths.values()))
    if n == 0:
        raise ValueError("cannot compute metrics on empty inputs")
    return n


def regression_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    """Point-prediction metrics.

    Args:
        y_true: Observed labels.
        y_pred: Predictions, same length.

    Returns:
        ``{"mae", "rmse", "r2", "max_abs_error", "bias", "n"}``.

    Raises:
        ValueError: On empty, misaligned or non-finite inputs.
    """
    y = _as_1d(y_true, "y_true")
    p = _as_1d(y_pred, "y_pred")
    n = _validate_same_length(y_true=y, y_pred=p)
    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(p))):
        raise ValueError("y_true and y_pred must be finite (drop or impute NaN rows first)")
    err = p - y
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if n >= 2 and ss_tot > 0.0 else float("nan")
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(ss_res / n)),
        "r2": float(r2),
        "max_abs_error": float(np.max(np.abs(err))),
        "bias": float(np.mean(err)),
        "n": n,
    }


def interval_score(
    y_true: ArrayLike, lower: ArrayLike, upper: ArrayLike, alpha: float
) -> FloatArray:
    """Per-observation interval (Winkler) score of a central ``(1 - alpha)`` interval.

    ``IS = (u - l) + (2 / alpha) * (l - y) * 1{y < l} + (2 / alpha) * (y - u) * 1{y > u}``
    (Gneiting & Raftery, 2007, "Strictly proper scoring rules, prediction, and
    estimation", JASA 102(477), eq. 43). Lower is better. Unbounded intervals
    score ``+inf``.

    Raises:
        ValueError: If ``alpha`` is not in (0, 1) or the inputs are invalid.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    y = _as_1d(y_true, "y_true")
    lo = _as_1d(lower, "lower")
    hi = _as_1d(upper, "upper")
    _validate_interval(y, lo, hi)
    below = np.where(y < lo, lo - y, 0.0)
    above = np.where(y > hi, y - hi, 0.0)
    with np.errstate(invalid="ignore"):
        width = hi - lo
    return np.asarray(width + (2.0 / alpha) * (below + above), dtype=np.float64)


def _validate_interval(y: FloatArray, lo: FloatArray, hi: FloatArray) -> None:
    _validate_same_length(y_true=y, lower=lo, upper=hi)
    if not np.all(np.isfinite(y)):
        raise ValueError("y_true must be finite")
    if np.any(np.isnan(lo)) or np.any(np.isnan(hi)):
        raise ValueError("interval bounds must not be NaN (+/-inf is allowed)")
    if np.any(lo == np.inf) or np.any(hi == -np.inf):
        raise ValueError("lower bounds must be < +inf and upper bounds > -inf")
    if np.any(lo > hi):
        raise ValueError("every lower bound must be <= its upper bound")


def interval_metrics(
    y_true: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
    *,
    alpha: float | None = None,
) -> dict[str, float]:
    """Coverage, sharpness and (optionally) the interval score of prediction intervals.

    Args:
        y_true: Observed labels.
        lower: Lower bounds (``-inf`` allowed: an unbounded side).
        upper: Upper bounds (``+inf`` allowed).
        alpha: Nominal miscoverage of the intervals (``0.1`` for 90 %). When
            given, the mean interval score is included.

    Returns:
        ``coverage`` (fraction of ``lower <= y <= upper``), ``below`` and
        ``above`` (miss rates on each side), ``mean_width`` and
        ``median_width`` (``inf`` if any interval is unbounded),
        ``frac_unbounded``, ``n`` and, when ``alpha`` is given,
        ``interval_score`` and ``nominal_coverage``.

    Raises:
        ValueError: On empty/misaligned inputs, NaN bounds or ``lower > upper``.
    """
    y = _as_1d(y_true, "y_true")
    lo = _as_1d(lower, "lower")
    hi = _as_1d(upper, "upper")
    _validate_interval(y, lo, hi)
    width = hi - lo
    out: dict[str, float] = {
        "coverage": float(np.mean((y >= lo) & (y <= hi))),
        "below": float(np.mean(y < lo)),
        "above": float(np.mean(y > hi)),
        "mean_width": float(np.mean(width)),
        "median_width": float(np.median(width)),
        "frac_unbounded": float(np.mean(~np.isfinite(width))),
        "n": int(y.size),
    }
    if alpha is not None:
        out["interval_score"] = float(np.mean(interval_score(y, lo, hi, alpha)))
        out["nominal_coverage"] = 1.0 - alpha
    return out


def _worst(column: str, values: pd.Series) -> float:
    finite = values.dropna()
    if finite.empty:
        return float("nan")
    if column in _WORST_MAX:
        return float(finite.max())
    if column in _WORST_MIN:
        return float(finite.min())
    if column == "bias":
        return float(finite.iloc[int(np.argmax(np.abs(finite.to_numpy())))])
    return float("nan")


def summarize_by_group(
    df: pd.DataFrame,
    group_col: str = "battery_id",
    *,
    y_col: str = "y",
    yhat_col: str = "yhat",
    lo_col: str | None = None,
    hi_col: str | None = None,
    alpha: float | None = None,
) -> pd.DataFrame:
    """Per-group metrics plus ``mean``, ``worst`` and ``pooled`` summary rows.

    The ``mean`` row is the macro average over groups (each battery counts
    once, which is the headline LOBO number). The ``worst`` row takes the
    worst value per column (max error, min R² and coverage, largest
    ``|bias|``). The ``pooled`` row treats all rows as one sample. ``n`` in a
    summary row is the total number of rows.

    Args:
        df: One row per prediction (e.g. out-of-fold predictions of a LOBO run).
        group_col: Column holding the group label (battery id).
        y_col: Observed label column.
        yhat_col: Prediction column.
        lo_col: Optional lower-bound column (requires ``hi_col``).
        hi_col: Optional upper-bound column.
        alpha: Nominal miscoverage of the interval columns, for the interval score.

    Returns:
        A DataFrame with ``group_col`` as the first column, one row per group in
        sorted order followed by the summary rows listed in :data:`SUMMARY_ROWS`.

    Raises:
        ValueError: If columns are missing, only one interval column is given,
            or the table is empty.
    """
    if (lo_col is None) != (hi_col is None):
        raise ValueError("lo_col and hi_col must be given together")
    needed = [group_col, y_col, yhat_col] + ([lo_col, hi_col] if lo_col and hi_col else [])
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"columns {missing} not found in table")
    if df.empty:
        raise ValueError("cannot summarise an empty table")

    def _metrics(part: pd.DataFrame) -> dict[str, float]:
        row = regression_metrics(part[y_col], part[yhat_col])
        if lo_col and hi_col:
            iv = interval_metrics(part[y_col], part[lo_col], part[hi_col], alpha=alpha)
            row.update({k: iv[k] for k in ("coverage", "mean_width")})
            if alpha is not None:
                row["interval_score"] = iv["interval_score"]
        return row

    groups = sorted(df[group_col].unique(), key=str)
    per_group = pd.DataFrame([{group_col: g, **_metrics(df[df[group_col] == g])} for g in groups])
    metric_cols = [c for c in per_group.columns if c not in (group_col, "n")]
    total_n = int(per_group["n"].sum())
    mean_row: dict[str, object] = {group_col: "mean", "n": total_n}
    worst_row: dict[str, object] = {group_col: "worst", "n": total_n}
    for col in metric_cols:
        mean_row[col] = float(per_group[col].mean())
        worst_row[col] = _worst(col, per_group[col])
    pooled_row: dict[str, object] = {group_col: "pooled", **_metrics(df)}
    summary = pd.DataFrame([mean_row, worst_row, pooled_row])
    out = pd.concat([per_group, summary], ignore_index=True)
    return out.loc[:, [group_col, "n", *metric_cols]]


def bootstrap_ci(
    values: ArrayLike,
    *,
    statistic: Callable[[FloatArray], float] = np.mean,
    n_boot: int = 2000,
    confidence: float = 0.95,
    random_state: int = RANDOM_SEED,
) -> tuple[float, float]:
    """Percentile bootstrap interval of a statistic over independent units.

    Pass one value per *battery* (e.g. per-battery MAE), never per cycle:
    cycles of one cell are not independent. With few batteries the interval is
    coarse; it communicates between-battery spread, not a precise CI.

    Args:
        values: One number per independent unit.
        statistic: Reduction applied to each resample.
        n_boot: Number of bootstrap resamples.
        confidence: Central coverage of the returned interval.
        random_state: Seed of the resampling generator.

    Returns:
        ``(low, high)`` percentile bounds.

    Raises:
        ValueError: On empty/non-finite input or invalid settings.
    """
    arr = _as_1d(values, "values")
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError("values must be a non-empty array of finite numbers")
    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1, got {n_boot}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    rng = np.random.default_rng(random_state)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    stats = np.array([statistic(arr[row]) for row in idx], dtype=np.float64)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(stats, [tail, 1.0 - tail])
    return float(low), float(high)
