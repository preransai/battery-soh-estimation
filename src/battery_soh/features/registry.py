"""Feature registry: the single source of truth for every model input.

Each feature declares *when it becomes available* (``availability``) and
whether it leaks the label (``leakage_risk``). Training code asks for a named
feature set through :func:`get_feature_set`, which refuses label-side or proxy
features unless the caller explicitly opts in (only the leakage audit does).
This turns the v1 failure mode, capacity used to predict capacity/C0, into an
error instead of a silent bug.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

Family = Literal["charge", "discharge", "impedance", "context", "legacy"]
Availability = Literal["partial_charge", "partial_discharge", "eis", "metadata", "full_discharge"]
LeakageRisk = Literal["none", "proxy", "label"]


@dataclass(frozen=True)
class FeatureSpec:
    """Metadata describing one model input."""

    name: str
    unit: str
    family: Family
    availability: Availability
    leakage_risk: LeakageRisk
    description: str
    plausible_range: tuple[float, float]


_SPECS: tuple[FeatureSpec, ...] = (
    # ---------------------------------------------------------------- partial CC charge
    FeatureSpec(
        "chg_cc_time_390_419_s",
        "s",
        "charge",
        "partial_charge",
        "none",
        "Time for the voltage to rise from 3.90 V to 4.19 V during constant-current charge.",
        (0.0, 20_000.0),
    ),
    FeatureSpec(
        "chg_cc_time_400_419_s",
        "s",
        "charge",
        "partial_charge",
        "none",
        "Time for the voltage to rise from 4.00 V to 4.19 V during constant-current charge.",
        (0.0, 20_000.0),
    ),
    FeatureSpec(
        "chg_cv_time_s",
        "s",
        "charge",
        "partial_charge",
        "none",
        "Duration of the constant-voltage phase (V >= 4.19 V until current < 0.1 A).",
        (0.0, 50_000.0),
    ),
    FeatureSpec(
        "chg_ica_peak_height_ah_per_v",
        "Ah/V",
        "charge",
        "partial_charge",
        "none",
        "Height of the incremental-capacity (dQ/dV) peak on the CC charge segment, 3.90-4.18 V.",
        (0.0, 100.0),
    ),
    FeatureSpec(
        "chg_ica_peak_voltage_v",
        "V",
        "charge",
        "partial_charge",
        "none",
        "Voltage at which the incremental-capacity peak occurs.",
        (3.0, 4.5),
    ),
    FeatureSpec(
        "chg_temp_rise_cc_c",
        "degC",
        "charge",
        "partial_charge",
        "none",
        "Cell temperature rise between 3.90 V and 4.19 V in the CC charge phase.",
        (-20.0, 40.0),
    ),
    # ---------------------------------------------------------------- partial discharge window
    FeatureSpec(
        "dis_q_380_350_ah",
        "Ah",
        "discharge",
        "partial_discharge",
        "none",
        "Charge delivered while the loaded voltage falls 3.80 V -> 3.50 V (~40% of a discharge).",
        (0.0, 5.0),
    ),
    FeatureSpec(
        "dis_time_380_350_s",
        "s",
        "discharge",
        "partial_discharge",
        "none",
        "Duration of the 3.80 V to 3.50 V discharge window.",
        (0.0, 20_000.0),
    ),
    FeatureSpec(
        "dis_r0_ohm",
        "ohm",
        "discharge",
        "partial_discharge",
        "none",
        "Ohmic resistance from the voltage step at load-on (dV/dI).",
        (0.0, 2.0),
    ),
    FeatureSpec(
        "dis_temp_rise_win_c",
        "degC",
        "discharge",
        "partial_discharge",
        "none",
        "Cell temperature rise inside the 3.80 V to 3.50 V discharge window.",
        (-20.0, 40.0),
    ),
    # ---------------------------------------------------------------- impedance (EIS)
    FeatureSpec(
        "eis_re_ohm",
        "ohm",
        "impedance",
        "eis",
        "none",
        "Electrolyte resistance from the most recent EIS measurement before the cycle.",
        (0.0, 2.0),
    ),
    FeatureSpec(
        "eis_rct_ohm",
        "ohm",
        "impedance",
        "eis",
        "none",
        "Charge-transfer resistance from the most recent EIS measurement before the cycle.",
        (0.0, 2.0),
    ),
    # ---------------------------------------------------------------- context / metadata
    FeatureSpec(
        "ctx_cycle_index",
        "cycles",
        "context",
        "metadata",
        "none",
        "Zero-based count of discharge cycles completed by the cell.",
        (0.0, 10_000.0),
    ),
    FeatureSpec(
        "ctx_ambient_c",
        "degC",
        "context",
        "metadata",
        "none",
        "Ambient (chamber) temperature of the test protocol.",
        (-40.0, 80.0),
    ),
    FeatureSpec(
        # An operating condition (family "context", never relativized), but measured from the
        # discharge arrays rather than taken from the protocol, so it needs a partial discharge.
        "ctx_discharge_current_a",
        "A",
        "context",
        "partial_discharge",
        "none",
        "Median absolute current under load during discharge.",
        (0.0, 20.0),
    ),
    # ---------------------------------------------------------------- v1 features (audit only)
    FeatureSpec(
        "leg_capacity_ah",
        "Ah",
        "legacy",
        "full_discharge",
        "label",
        "v1 input: full-discharge capacity. It is the numerator of the SOH label (target leakage).",
        (0.0, 5.0),
    ),
    FeatureSpec(
        "leg_record_time_s",
        "s",
        "legacy",
        "full_discharge",
        "proxy",
        "v1 'discharge_time': record length including the post-cutoff rest (a capacity proxy).",
        (0.0, 20_000.0),
    ),
    FeatureSpec(
        "leg_current_mean_a",
        "A",
        "legacy",
        "full_discharge",
        "proxy",
        "v1 whole-record mean |I|; encodes the load fraction of the record (a capacity proxy).",
        (0.0, 20.0),
    ),
    FeatureSpec(
        "leg_voltage_mean_v",
        "V",
        "legacy",
        "full_discharge",
        "proxy",
        "v1 whole-record mean voltage (needs the full discharge).",
        (0.0, 5.0),
    ),
    FeatureSpec(
        "leg_temp_max_c",
        "degC",
        "legacy",
        "full_discharge",
        "proxy",
        "v1 whole-record maximum temperature (needs the full discharge).",
        (-40.0, 100.0),
    ),
)

FEATURES: Mapping[str, FeatureSpec] = MappingProxyType({spec.name: spec for spec in _SPECS})

_CHARGE = (
    "chg_cc_time_390_419_s",
    "chg_cv_time_s",
    "chg_ica_peak_height_ah_per_v",
    "chg_ica_peak_voltage_v",
    "chg_temp_rise_cc_c",
)
_CHARGE_EXT = ("chg_cc_time_400_419_s", *_CHARGE[1:])
_DISCHARGE = ("dis_q_380_350_ah", "dis_r0_ohm", "dis_temp_rise_win_c")
_EIS = ("eis_re_ohm", "eis_rct_ohm")
_CONDITIONS = ("ctx_ambient_c", "ctx_discharge_current_a")

FEATURE_SETS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "charge": _CHARGE,
        "discharge": _DISCHARGE,
        "combined": (*_CHARGE, *_DISCHARGE),
        "combined_eis": (*_CHARGE, *_DISCHARGE, *_EIS),
        "charge_ext": (*_CHARGE_EXT, *_CONDITIONS),
        "combined_ext": (*_CHARGE_EXT, *_DISCHARGE, *_CONDITIONS),
        # Post-hoc diagnostic ablations (defined AFTER leave-one-battery-out showed that the
        # CV-phase duration has a large cell-specific offset). Report them only as labelled
        # ablations; unbiased estimates of feature selection come from nested LOBO.
        "charge_no_cv": tuple(f for f in _CHARGE if f != "chg_cv_time_s"),
        "combined_no_cv": tuple(f for f in (*_CHARGE, *_DISCHARGE) if f != "chg_cv_time_s"),
        "cycle_only": ("ctx_cycle_index",),
        "legacy_v1": (
            "leg_capacity_ah",
            "leg_record_time_s",
            "leg_current_mean_a",
            "leg_voltage_mean_v",
            "leg_temp_max_c",
            "ctx_cycle_index",
        ),
        "legacy_v1_no_capacity": (
            "leg_record_time_s",
            "leg_current_mean_a",
            "leg_voltage_mean_v",
            "leg_temp_max_c",
            "ctx_cycle_index",
        ),
        "capacity_oracle": ("leg_capacity_ah",),
    }
)


def get_feature_set(name: str, *, allow_leaky: bool = False) -> tuple[str, ...]:
    """Return the feature names of a named set.

    Args:
        name: Key of :data:`FEATURE_SETS`.
        allow_leaky: Permit label-side or proxy features. Only the leakage audit
            and explicitly labelled sanity baselines should pass ``True``.

    Raises:
        ValueError: If the set is unknown, or contains leaky features while
            ``allow_leaky`` is false.
    """
    try:
        names = FEATURE_SETS[name]
    except KeyError:
        known = ", ".join(sorted(FEATURE_SETS))
        raise ValueError(f"Unknown feature set {name!r}; known sets: {known}") from None
    if not allow_leaky:
        leaky = [n for n in names if is_leaky(n)]
        if leaky:
            raise ValueError(
                f"Feature set {name!r} contains label-side/proxy features {leaky}; "
                "pass allow_leaky=True only for audits or labelled sanity baselines."
            )
    return names


def is_leaky(feature: str) -> bool:
    """Return True if ``feature`` is label-side or a label proxy."""
    return FEATURES[feature].leakage_risk != "none"
