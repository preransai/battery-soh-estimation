"""One function from raw segments to a feature row: the train/serve parity point.

The offline table builder (:mod:`battery_soh.features.build`) and the API
endpoint ``POST /v1/soh/estimate-from-cycle`` both call
:func:`features_from_segments`. Because the exact same code turns raw arrays
into features in training and in serving, a model can never see differently
computed inputs in production than it was evaluated on.

Metadata-only context features (``ctx_cycle_index``, ``ctx_ambient_c``) are
not measurable from the arrays; the caller adds them (the table builder from
the cycle count and the protocol registry).
"""

from __future__ import annotations

from battery_soh.data.schema import Segment
from battery_soh.features._signal import NAN
from battery_soh.features.charge import CHARGE_FEATURES, charge_features
from battery_soh.features.discharge import DISCHARGE_FEATURES, discharge_features
from battery_soh.features.impedance import eis_features
from battery_soh.features.legacy import LEGACY_FEATURES, legacy_features
from battery_soh.features.registry import FEATURES

#: Version of the extraction semantics. Bump it whenever a definition changes,
#: and store it with every model artifact so stale models are detectable.
FEATURE_EXTRACTOR_VERSION = "2.0.0"

#: Context features that come from metadata, not from the measured arrays.
METADATA_FEATURES: tuple[str, ...] = ("ctx_cycle_index", "ctx_ambient_c")

#: Keys returned by :func:`features_from_segments`, in registry order: every
#: registry feature except :data:`METADATA_FEATURES`.
SEGMENT_FEATURES: tuple[str, ...] = tuple(n for n in FEATURES if n not in METADATA_FEATURES)


def features_from_segments(
    charge: Segment | None,
    discharge: Segment | None,
    *,
    eis: tuple[float, float] | None = None,
) -> dict[str, float]:
    """Compute every array-derived registry feature of one cycle.

    Uses the default (registry) window parameters of
    :func:`~battery_soh.features.charge.charge_features` and
    :func:`~battery_soh.features.discharge.discharge_features`; experiments that
    vary the windows call those functions directly.

    Args:
        charge: The charge preceding the discharge (possibly partial), or ``None``.
        discharge: The discharge (possibly partial), or ``None``.
        eis: ``(re_ohm, rct_ohm)`` of the latest EIS measurement *before* the
            cycle, or ``None``.

    Returns:
        A dict with exactly the keys of :data:`SEGMENT_FEATURES`, in registry
        order: the ``chg_*`` features from ``charge``; the ``dis_*`` features and
        ``ctx_discharge_current_a`` from ``discharge``; ``eis_*`` from ``eis``;
        and the leaky ``leg_*`` audit columns from ``discharge``
        (``leg_capacity_ah`` is the segment's dataset capacity, NaN when absent;
        the other ``leg_*`` are whole-record statistics, meaningful only for a
        full discharge). Anything not computable is NaN, never 0.

    Raises:
        ValueError: If a segment has the wrong kind or ``eis`` is malformed.
    """
    values: dict[str, float] = {}
    if charge is not None:
        values.update(charge_features(charge))
    else:
        values.update(dict.fromkeys(CHARGE_FEATURES, NAN))
    if discharge is not None:
        values.update(discharge_features(discharge))
        values.update(legacy_features(discharge))
    else:
        values.update(dict.fromkeys((*DISCHARGE_FEATURES, *LEGACY_FEATURES), NAN))
    values.update(eis_features(eis))

    if set(values) != set(SEGMENT_FEATURES):  # pragma: no cover - guards registry drift
        raise RuntimeError(
            "extractor/registry mismatch: "
            f"missing={sorted(set(SEGMENT_FEATURES) - set(values))}, "
            f"extra={sorted(set(values) - set(SEGMENT_FEATURES))}"
        )
    return {name: float(values[name]) for name in SEGMENT_FEATURES}
