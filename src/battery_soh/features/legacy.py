"""The v1 model inputs, reproduced for the leakage audit only.

v1 fed these whole-record discharge statistics to its models. Each needs the
*complete* discharge, and each is a near-deterministic function of capacity,
which is the numerator of the SOH label:

* ``leg_capacity_ah`` *is* the label numerator (``SOH = 100 * capacity / C0``).
* ``leg_record_time_s`` is the record length ``t[-1] - t[0]``. Under constant
  current the load-on time is capacity / current; the record adds a
  post-cutoff rest tail (125-700 s), yet still has r = 0.98-0.99 with capacity.
* ``leg_current_mean_a`` is the whole-record mean ``|I|``, i.e. the load
  current times the load share of the record (encodes the rest-tail fraction).
* ``leg_voltage_mean_v`` and ``leg_temp_max_c`` summarise the full discharge.

The registry marks them ``leakage_risk="label"``/``"proxy"``, so
:func:`battery_soh.features.registry.get_feature_set` refuses them unless the
caller explicitly opts in (only the audit does).
"""

from __future__ import annotations

import numpy as np

from battery_soh.data.schema import Segment
from battery_soh.features._signal import NAN, finite_or_nan, require_kind, sanitize

#: Keys returned by :func:`legacy_features` (registry order).
LEGACY_FEATURES: tuple[str, ...] = (
    "leg_capacity_ah",
    "leg_record_time_s",
    "leg_current_mean_a",
    "leg_voltage_mean_v",
    "leg_temp_max_c",
)


def legacy_features(seg: Segment) -> dict[str, float]:
    """Compute the v1 whole-record discharge statistics (leaky; audit only).

    Semantics follow v1 (``legacy/src/feature_extraction.py``) on the
    sanitised record (finite, time-ordered samples). An empty record yields
    NaN for every statistic except ``leg_capacity_ah``, which is copied from
    the segment's dataset capacity.

    Raises:
        ValueError: If ``seg`` is not a discharge segment.
    """
    require_kind(seg, "discharge")
    out = dict.fromkeys(LEGACY_FEATURES, NAN)
    out["leg_capacity_ah"] = finite_or_nan(seg.capacity_ah)
    series = sanitize(seg)
    if series.n == 0:
        return out
    out["leg_record_time_s"] = float(series.t[-1] - series.t[0])
    out["leg_current_mean_a"] = float(np.mean(np.abs(series.i)))
    out["leg_voltage_mean_v"] = float(np.mean(series.v))
    temp = series.temp[np.isfinite(series.temp)]
    out["leg_temp_max_c"] = float(temp.max()) if temp.size else NAN
    return out
