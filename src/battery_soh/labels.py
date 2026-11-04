"""State-of-health labels.

``SOH = 100 * capacity / C0``. Two references are provided:

* ``soh_pct`` (headline label) uses the cell's own reference capacity ``C0``: the median of
  its first ``n_ref`` QC-valid discharge capacities. Cells leave the factory with different
  capacities (1.84-2.02 Ah for the four reference cells), so the own-reference label measures
  each cell's *fade* rather than manufacturing spread.
* ``soh_rated_pct`` uses the 2.0 Ah nameplate capacity, the convention of the NASA README
  ("30 % fade in rated capacity, from 2 Ahr to 1.4 Ahr").

The discharge capacity is label-side information: it defines the target and is never a model
input (see ``docs/LEAKAGE_AUDIT.md``).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Final, TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from battery_soh.config import DEFAULT_N_REFERENCE_CYCLES, RATED_CAPACITY_AH
from battery_soh.qc import (
    QC_MAX_DEV_AH,
    QC_MIN_AH,
    QC_MIN_PERIODS,
    QC_WINDOW,
    flag_capacity_outliers,
)

logger = logging.getLogger(__name__)

__all__ = [
    "EOL_SOH_PCT",
    "LABEL_COLUMNS",
    "label_battery",
    "reference_capacity",
    "soh_from_capacity",
]

#: Conventional end-of-life threshold (documentation and plots only, never used for training).
EOL_SOH_PCT: Final = 80.0

#: Columns produced by :func:`label_battery`, in order.
LABEL_COLUMNS: Final[tuple[str, ...]] = (
    "capacity_ah",
    "c0_ah",
    "soh_pct",
    "soh_rated_pct",
    "qc_valid",
    "qc_reason",
)

ArrayLike1D: TypeAlias = pd.Series | Sequence[float] | NDArray[np.floating]


def _as_float_series(values: ArrayLike1D, name: str) -> pd.Series:
    series = values if isinstance(values, pd.Series) else pd.Series(values, dtype=object)
    return pd.to_numeric(series, errors="coerce").astype(np.float64).rename(name)


def reference_capacity(
    capacity: pd.Series, valid: pd.Series, n_ref: int = DEFAULT_N_REFERENCE_CYCLES
) -> float:
    """Return a cell's reference capacity ``C0``.

    Args:
        capacity: Discharge capacities (Ah) of one battery, in cycle order.
        valid: QC flags aligned with ``capacity`` (same index); missing flags count as False.
        n_ref: Number of leading valid cycles whose median defines ``C0``.

    Returns:
        The median of the first ``n_ref`` valid, finite capacities (fewer if the battery has
        fewer), or NaN when there is none.

    Raises:
        ValueError: If ``n_ref < 1`` or the two Series are not aligned.
    """
    if isinstance(n_ref, bool) or not isinstance(n_ref, int) or n_ref < 1:
        raise ValueError(f"n_ref must be a positive integer, got {n_ref!r}")
    if len(capacity) != len(valid) or not capacity.index.equals(valid.index):
        raise ValueError("capacity and valid must have the same length and index")
    values = _as_float_series(capacity, "capacity_ah").to_numpy()
    flags = valid.astype("boolean").fillna(False).to_numpy(dtype=bool)
    usable = values[flags & np.isfinite(values)]
    if usable.size == 0:
        logger.debug("No valid capacity available; C0 is undefined")
        return float("nan")
    return float(np.median(usable[:n_ref]))


def soh_from_capacity(capacity: ArrayLike1D, c0: float) -> pd.Series:
    """Convert capacities to state of health in percent of ``c0``.

    Args:
        capacity: Capacities in Ah. Non-finite entries give NaN.
        c0: Reference capacity in Ah. NaN (no valid reference) gives an all-NaN result.

    Returns:
        ``100 * capacity / c0`` as a float Series named ``soh_pct`` (index preserved).

    Raises:
        ValueError: If ``c0`` is finite but not positive, or infinite.
    """
    values = _as_float_series(capacity, "soh_pct")
    c0 = float(c0)
    if np.isnan(c0):
        return pd.Series(np.nan, index=values.index, name="soh_pct", dtype=np.float64)
    if not np.isfinite(c0) or c0 <= 0:
        raise ValueError(f"reference capacity must be positive and finite, got {c0!r}")
    with np.errstate(invalid="ignore"):
        soh = 100.0 * values / c0
    return soh.where(np.isfinite(soh)).rename("soh_pct")


def label_battery(
    capacity: ArrayLike1D,
    *,
    n_ref: int = DEFAULT_N_REFERENCE_CYCLES,
    rated_capacity_ah: float = RATED_CAPACITY_AH,
    min_ah: float = QC_MIN_AH,
    window: int = QC_WINDOW,
    max_dev_ah: float = QC_MAX_DEV_AH,
    min_periods: int = QC_MIN_PERIODS,
) -> pd.DataFrame:
    """Quality-control one battery's capacities and derive both SOH labels.

    Labels are computed for every cycle, including QC-invalid ones, so that the table stays
    complete; training and scoring code filters on ``qc_valid``.

    Args:
        capacity: Discharge capacities (Ah) of one battery, in cycle order.
        n_ref: Leading valid cycles that define ``C0``.
        rated_capacity_ah: Nameplate capacity for ``soh_rated_pct``.
        min_ah: See :func:`battery_soh.qc.flag_capacity_outliers`.
        window: See :func:`battery_soh.qc.flag_capacity_outliers`.
        max_dev_ah: See :func:`battery_soh.qc.flag_capacity_outliers`.
        min_periods: See :func:`battery_soh.qc.flag_capacity_outliers`.

    Returns:
        A DataFrame with :data:`LABEL_COLUMNS`, aligned to the input index.

    Raises:
        ValueError: If a parameter is invalid.
    """
    values = _as_float_series(capacity, "capacity_ah")
    valid, reason = flag_capacity_outliers(
        values, min_ah=min_ah, window=window, max_dev_ah=max_dev_ah, min_periods=min_periods
    )
    c0 = reference_capacity(values, valid, n_ref)
    return pd.DataFrame(
        {
            "capacity_ah": values,
            "c0_ah": pd.Series(c0, index=values.index, dtype=np.float64),
            "soh_pct": soh_from_capacity(values, c0),
            "soh_rated_pct": soh_from_capacity(values, rated_capacity_ah),
            "qc_valid": valid,
            "qc_reason": reason,
        },
        index=values.index,
    )
