"""Quality control of per-cycle discharge capacities (the label numerator).

NASA's README files warn that "there are several discharge runs where the capacity was very
low"; the files also contain truncated records with ``Capacity = 0`` and formation cycles far
below the cell's real capacity. Such cycles must not define a cell's reference capacity
``C0`` or be scored as ground truth, so every discharge gets a validity flag and a reason.

The rule, applied in order (first match wins):

1. ``missing`` - no finite capacity was recorded.
2. ``below_min`` - capacity below ``min_ah`` (default 0.3 Ah, 15 % of the 2 Ah nameplate):
   physically implausible for these cells under the tested protocols.
3. ``isolated`` - fewer than ``min_periods`` plausible capacities in the centred window, so
   the value cannot be checked against its neighbours; rejected conservatively.
4. ``hampel_outlier`` - deviates by more than ``max_dev_ah`` from the median of the plausible
   capacities in a centred rolling window of ``window`` cycles.
5. ``ok`` - everything else.

Step 4 is a Hampel-style filter with a fixed absolute tolerance instead of the classic
``k * MAD`` band. Capacity fades smoothly, so the within-window MAD is tiny (a few mAh) and a
MAD band would reject the genuine capacity-recovery jumps that follow rest periods (up to
0.09 Ah in B0005). A 0.15 Ah tolerance keeps every cycle of the four reference cells and
removes glitches such as B0036's ``2.44 Ah`` spike or the low formation discharge that starts
B0036 and B0054-B0056. Values rejected by step 2 (for example B0046's ``0.00 Ah`` records) are
excluded from the rolling median, so they cannot drag it towards zero.

``isolated`` exists for sparse histories in which a value has too few plausible neighbours to
be judged; with the default parameters no NASA cell triggers it, but the code keeps such
values out of the labels instead of silently accepting them.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray

__all__ = [
    "QC_MAX_DEV_AH",
    "QC_MIN_AH",
    "QC_MIN_PERIODS",
    "QC_REASONS",
    "QC_WINDOW",
    "flag_capacity_outliers",
    "qc_summary",
]

#: Minimum plausible capacity (Ah).
QC_MIN_AH: Final = 0.3
#: Width of the centred rolling window (cycles, odd).
QC_WINDOW: Final = 9
#: Maximum allowed absolute deviation from the local median (Ah).
QC_MAX_DEV_AH: Final = 0.15
#: Minimum number of plausible capacities needed in the window to compute the median.
QC_MIN_PERIODS: Final = 3
#: Every reason code, in the order the rule evaluates them.
QC_REASONS: Final[tuple[str, ...]] = ("missing", "below_min", "isolated", "hampel_outlier", "ok")


def _validate(min_ah: float, window: int, max_dev_ah: float, min_periods: int) -> None:
    if not (np.isfinite(min_ah) and min_ah >= 0):
        raise ValueError(f"min_ah must be a finite number >= 0, got {min_ah!r}")
    if isinstance(window, bool) or not isinstance(window, int) or window < 1 or window % 2 == 0:
        raise ValueError(f"window must be a positive odd integer, got {window!r}")
    if not (np.isfinite(max_dev_ah) and max_dev_ah > 0):
        raise ValueError(f"max_dev_ah must be a finite number > 0, got {max_dev_ah!r}")
    if (
        isinstance(min_periods, bool)
        or not isinstance(min_periods, int)
        or not 1 <= min_periods <= window
    ):
        raise ValueError(f"min_periods must be an integer in [1, window], got {min_periods!r}")


def flag_capacity_outliers(
    capacity: pd.Series | Sequence[float] | NDArray[np.floating],
    *,
    min_ah: float = QC_MIN_AH,
    window: int = QC_WINDOW,
    max_dev_ah: float = QC_MAX_DEV_AH,
    min_periods: int = QC_MIN_PERIODS,
) -> tuple[pd.Series, pd.Series]:
    """Flag implausible per-cycle capacities of one battery.

    Args:
        capacity: Discharge capacities (Ah) of a single battery in cycle order. ``None``,
            NaN, infinities and non-numeric entries count as missing.
        min_ah: Lowest plausible capacity (inclusive).
        window: Centred rolling-window width in cycles (positive, odd).
        max_dev_ah: Largest accepted absolute deviation from the local median.
        min_periods: Plausible values required in the window to form the median.

    Returns:
        ``(valid, reason)``: a boolean Series named ``qc_valid`` and a string Series named
        ``qc_reason`` (values from :data:`QC_REASONS`), both aligned to the input index.

    Raises:
        ValueError: If a parameter is out of range.
    """
    _validate(min_ah, window, max_dev_ah, min_periods)
    series = capacity if isinstance(capacity, pd.Series) else pd.Series(capacity, dtype=object)
    values = pd.to_numeric(series, errors="coerce").astype(np.float64)
    raw = values.to_numpy()

    missing = ~np.isfinite(raw)
    below_min = ~missing & (raw < min_ah)
    plausible = values.where(~missing & ~below_min)
    local_median = (
        plausible.rolling(window, center=True, min_periods=min_periods).median().to_numpy()
    )
    candidate = ~missing & ~below_min
    isolated = candidate & np.isnan(local_median)
    with np.errstate(invalid="ignore"):
        outlier = candidate & ~isolated & (np.abs(raw - local_median) > max_dev_ah)

    reason = np.select(
        [missing, below_min, isolated, outlier],
        ["missing", "below_min", "isolated", "hampel_outlier"],
        default="ok",
    )
    valid = pd.Series(reason == "ok", index=series.index, name="qc_valid", dtype=bool)
    reasons = pd.Series(reason.astype(object), index=series.index, name="qc_reason")
    return valid, reasons


def qc_summary(reason: pd.Series) -> dict[str, int]:
    """Count cycles per QC reason.

    Args:
        reason: The ``qc_reason`` Series returned by :func:`flag_capacity_outliers`.

    Returns:
        ``{reason: count}`` for every code in :data:`QC_REASONS` (zeros included).
    """
    counts = reason.value_counts()
    return {code: int(counts.get(code, 0)) for code in QC_REASONS}
