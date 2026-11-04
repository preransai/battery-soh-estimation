"""Commissioning-referenced ("relative") health indicators.

Cells differ at birth: the four NASA reference cells start between 1.84 and
2.02 Ah, and their health indicators differ by a similar spread. Dividing
each indicator by the same cell's value at commissioning removes that
cell-to-cell offset and leaves the *change* caused by ageing, which is what
the SOH label (capacity relative to the cell's own initial capacity) measures.

The reference of a feature, for one cell, is the mean of its first ``n_ref``
values that are (a) on QC-valid cycles and (b) finite (not NaN), in cycle
order. It needs only those early cycles of the cell, so it is available in
deployment: record a few cycles when a cell enters service, store the
reference, and pass it to :func:`apply_reference` with every later
measurement.

Which early cycles are eligible is decided by the table's ``qc_valid`` flag,
and that flag is computed from each cycle's *measured discharge capacity*
(:mod:`battery_soh.qc`: missing, implausibly low or Hampel-outlier capacities
are invalid). Capacity therefore never enters a reference value, but it does
choose the reference rows. For the four core cells every cycle passes QC, so
their references are simply the first ``n_ref`` finite values. In the
extended set the flag excludes the first discharge of B0032, B0036 and
B0046-B0048, whose capacity is a Hampel outlier (B0036's is a low formation
discharge), so their references start one cycle later. A deployment without
capacity tests has to pick its reference cycles by other means (for example
the first routine cycles in service, discarding records that are obviously
faulty); :func:`apply_reference` accepts whatever reference it is given.

Rules shared by :func:`relativize` (training tables) and
:func:`apply_reference` (serving), so both produce identical values:

* Context features (registry family ``"context"``, or a ``ctx_`` prefix for
  names outside the registry) are passed through unchanged.
* Every other feature becomes ``value / reference``; the result is NaN when
  the value or the reference is NaN/infinite or the reference is 0.

The rows whose values entered at least one reference are flagged in
``is_reference_cycle``: their relative values are partly computed from
themselves, so evaluation must not score them.

The ratio is meaningful for ratio-scale indicators (Ah, s, Ah/V, ohm). For
temperature rises, whose reference can be near zero, the ratio is unstable;
prefer the absolute value for those.
"""

from __future__ import annotations

import logging
import math
import operator
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from battery_soh._typing import FloatArray
from battery_soh.config import DEFAULT_N_REFERENCE_CYCLES
from battery_soh.features.registry import FEATURES

logger = logging.getLogger(__name__)

#: Boolean column added by :func:`relativize`.
REFERENCE_FLAG_COLUMN = "is_reference_cycle"


def is_context_feature(name: str) -> bool:
    """Return True for context features, which are never relativized."""
    spec = FEATURES.get(name)
    if spec is not None:
        return spec.family == "context"
    return name.startswith("ctx_")


def _ratio(value: float, reference: float | None) -> float:
    if reference is None:
        return math.nan
    num, den = float(value), float(reference)
    if not (math.isfinite(num) and math.isfinite(den)) or den == 0.0:
        return math.nan
    return num / den


def _as_float(values: pd.Series) -> FloatArray:
    """Float view of a numeric column; missing values (NaN, None, pd.NA) become NaN."""
    return np.asarray(values.to_numpy(dtype=np.float64, na_value=np.nan), dtype=np.float64)


def _check_n_ref(n_ref: int) -> int:
    if isinstance(n_ref, bool):
        raise ValueError("n_ref must be a positive integer, got a bool")
    try:
        value = operator.index(n_ref)
    except TypeError:
        raise ValueError(f"n_ref must be a positive integer, got {n_ref!r}") from None
    if value < 1:
        raise ValueError(f"n_ref must be >= 1, got {value}")
    return value


def _reference(
    df: pd.DataFrame,
    features: Sequence[str],
    *,
    n_ref: int,
    group_col: str,
    valid_col: str | None,
    order_col: str | None,
) -> tuple[pd.DataFrame, NDArray[np.bool_]]:
    """Reference table (groups x relativized features) and the per-row "used" mask."""
    n_ref = _check_n_ref(n_ref)
    if isinstance(features, str):
        raise ValueError("features must be a sequence of column names, not a string")
    names = list(features)
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate feature names in {names}")
    required = [group_col, *names]
    required += [c for c in (valid_col, order_col) if c is not None]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"columns missing from the frame: {missing}")
    rel = [n for n in names if not is_context_feature(n)]
    non_numeric = [n for n in rel if not pd.api.types.is_numeric_dtype(df[n])]
    if non_numeric:
        raise ValueError(f"features must be numeric, got non-numeric columns {non_numeric}")

    work = df[list(dict.fromkeys(required))].reset_index(drop=True)
    valid = (
        np.ones(len(work), dtype=bool)
        if valid_col is None
        else work[valid_col].eq(True).to_numpy(dtype=bool)
    )
    order = (
        np.arange(len(work))
        if order_col is None
        else np.argsort(work[order_col].to_numpy(), kind="stable")
    )
    ordered = work.iloc[order]
    used = np.zeros(len(work), dtype=bool)
    table: dict[str, pd.Series] = {}
    for name in rel:
        finite = np.isfinite(_as_float(ordered[name]))
        first = ordered[valid[order] & finite].groupby(group_col, sort=False).head(n_ref)
        used[first.index.to_numpy()] = True
        table[name] = first.groupby(group_col, sort=True)[name].mean()
    groups = pd.Index(work[group_col].dropna().unique(), name=group_col).sort_values()
    reference = pd.DataFrame(table, index=groups, columns=rel)
    for name in rel:
        short = reference[name].isna()
        if short.any():
            logger.debug("no reference for %s in groups %s", name, list(reference.index[short]))
    return reference, used


def reference_table(
    df: pd.DataFrame,
    features: Sequence[str],
    *,
    n_ref: int = DEFAULT_N_REFERENCE_CYCLES,
    group_col: str = "battery_id",
    valid_col: str | None = "qc_valid",
    order_col: str | None = "cycle_index",
) -> pd.DataFrame:
    """Per-group reference values: the mean of the first ``n_ref`` valid non-NaN values.

    Args:
        df: Feature table with one row per cycle.
        features: Feature columns; context features are skipped.
        n_ref: Number of early values averaged into each reference.
        group_col: Column identifying the cell.
        valid_col: Boolean QC column; only rows where it is ``True`` count
            (``None``: all rows count). In the feature table this is the
            capacity-based ``qc_valid`` flag (see the module docstring).
        order_col: Column giving the cycle order (``None``: row order).

    Returns:
        DataFrame indexed by group, one column per non-context feature; NaN
        where a group has no valid non-NaN value of that feature.

    Raises:
        ValueError: If a column is missing, a feature is not numeric, or
            ``n_ref`` is not a positive integer.
    """
    table, _ = _reference(
        df, features, n_ref=n_ref, group_col=group_col, valid_col=valid_col, order_col=order_col
    )
    return table


def relativize(
    df: pd.DataFrame,
    features: Sequence[str],
    *,
    n_ref: int = DEFAULT_N_REFERENCE_CYCLES,
    group_col: str = "battery_id",
    valid_col: str | None = "qc_valid",
    order_col: str | None = "cycle_index",
) -> pd.DataFrame:
    """Divide each health indicator by its cell's commissioning reference.

    For every non-context feature, each value is divided by the mean of that
    cell's first ``n_ref`` QC-valid non-NaN values (see :func:`reference_table`).
    The selection is per feature, so a feature that is NaN on an early cycle
    takes its reference from the next valid cycles. Context features are
    returned unchanged. The input frame is not modified.

    Args:
        df: Feature table with one row per cycle.
        features: Feature columns to transform (context features pass through).
        n_ref: Number of early values averaged into each reference.
        group_col: Column identifying the cell.
        valid_col: Boolean QC column; only rows where it is ``True`` can be
            reference rows (``None``: all rows).
        order_col: Column giving the cycle order (``None``: row order).

    Returns:
        A copy of ``df`` (same index and row order) with the features replaced
        by their relative values and a boolean column ``is_reference_cycle``
        that is True on every row used in at least one feature's reference.

    Raises:
        ValueError: If a column is missing, a feature is not numeric, or
            ``n_ref`` is not a positive integer.
    """
    reference, used = _reference(
        df, features, n_ref=n_ref, group_col=group_col, valid_col=valid_col, order_col=order_col
    )
    out = df.copy()
    groups = df[group_col]
    for name in reference.columns:
        ref = _as_float(groups.map(reference[name]))
        values = _as_float(df[name])
        ok = np.isfinite(ref) & (ref != 0.0) & np.isfinite(values)
        ratio = np.full(values.shape, np.nan)
        np.divide(values, ref, out=ratio, where=ok)
        out[name] = ratio
    out[REFERENCE_FLAG_COLUMN] = used
    return out


def apply_reference(
    features: Mapping[str, float], reference: Mapping[str, float]
) -> dict[str, float]:
    """Serving-side counterpart of :func:`relativize` for one feature row.

    Args:
        features: Absolute feature values of one cycle.
        reference: The cell's commissioning reference per feature (e.g. one
            row of :func:`reference_table`).

    Returns:
        A new dict with the same keys: context features unchanged, every other
        feature divided by its reference, NaN when the reference is missing,
        NaN, infinite or zero (or the value itself is not finite).

    Raises:
        ValueError: If a value or reference entry is not a real number.
    """
    out: dict[str, float] = {}
    try:
        for name, value in features.items():
            if is_context_feature(name):
                out[name] = float(value)
            else:
                ref = reference.get(name)
                out[name] = _ratio(value, None if ref is None else float(ref))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"feature and reference values must be real numbers ({exc})") from None
    return out
