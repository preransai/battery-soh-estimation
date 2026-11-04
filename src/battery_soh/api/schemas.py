"""Request and response models of the ``/v1`` API.

Input models are strict: unknown fields are rejected (``extra="forbid"``),
numbers must be real JSON numbers (no strings, no booleans), and floats must be
finite (``NaN``/``Infinity`` tokens, which Python's JSON parser accepts, are
refused). Health-indicator values are bounded by the physical
``plausible_range`` of each feature in :mod:`battery_soh.features.registry`;
the feature model is generated from the registry, so the API, its OpenAPI
schema and the training code share one source of truth. Label-side and proxy
features (the v1 ``leg_*`` inputs) are not accepted at all: a leakage-free
service has no use for them.

Response models document the JSON the service returns and also forbid extra
fields, so a drift between the service functions and the documented contract
fails loudly in tests instead of silently dropping data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

from battery_soh.api.settings import HARD_MAX_BATCH_ITEMS, HARD_MAX_SAMPLES
from battery_soh.data.schema import Segment
from battery_soh.features.registry import FEATURES

__all__ = [
    "BATTERY_ID_PATTERN",
    "INPUT_FEATURES",
    "MAX_BATCH_ITEMS",
    "MODEL_ID_PATTERN",
    "BatchRequest",
    "BatchResponse",
    "Confidence",
    "CycleEstimateRequest",
    "CycleEstimateResponse",
    "DemoBattery",
    "DemoCycle",
    "EisIn",
    "EstimateRequest",
    "EstimateResponse",
    "FeatureValues",
    "LiveStatus",
    "ModelSummary",
    "ProblemDetails",
    "ReadyStatus",
    "SegmentIn",
    "problem_responses",
]

#: Schema ceiling on items in one ``/v1/soh/batch`` request; the served limit is the
#: (lower by default) ``Settings.max_batch_items``, enforced in the handler.
MAX_BATCH_ITEMS = HARD_MAX_BATCH_ITEMS

#: Identifiers that appear in URLs and file names (model ids, battery ids).
MODEL_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$"
BATTERY_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$"

#: Registry features a client may send: everything except label-side/proxy inputs.
INPUT_FEATURES: tuple[str, ...] = tuple(
    name for name, spec in FEATURES.items() if spec.leakage_risk == "none"
)
_LEAKY_FEATURES = frozenset(name for name in FEATURES if name not in INPUT_FEATURES)

#: Supported conformal confidence levels (a float enum; mypy cannot express float literals).
if TYPE_CHECKING:
    Confidence = float
else:
    Confidence = Literal[0.8, 0.9, 0.95]
ModelId = Annotated[str, Field(pattern=MODEL_ID_PATTERN, examples=["combined-absolute-ridge"])]

# Physical envelopes of raw time series (generous; they exist to keep arithmetic finite).
TimeValue = Annotated[float, Field(ge=0.0, le=1e7)]
VoltageValue = Annotated[float, Field(ge=0.0, le=10.0)]
CurrentValue = Annotated[float, Field(ge=-100.0, le=100.0)]
TemperatureValue = Annotated[float, Field(ge=-100.0, le=200.0)]


class _StrictInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False, frozen=True)


class _Output(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------- feature values
class FeatureValuesBase(_StrictInput):
    """Health indicators of one cycle, keyed by registry name; omitted or null = missing."""

    @model_validator(mode="before")
    @classmethod
    def _reject_leaky(cls, data: Any) -> Any:
        if isinstance(data, dict):
            leaky = sorted(k for k in data if k in _LEAKY_FEATURES)
            if leaky:
                raise ValueError(
                    f"label-side/proxy feature(s) {', '.join(leaky)} are not accepted "
                    "(they need the full discharge and leak the SOH label)"
                )
        return data

    def provided(self) -> dict[str, float]:
        """The features that have a value, in registry order."""
        values = self.model_dump()
        return {name: float(v) for name, v in values.items() if v is not None}


def _feature_field(name: str) -> tuple[Any, Any]:
    spec = FEATURES[name]
    low, high = spec.plausible_range
    return (
        float | None,
        Field(
            default=None,
            ge=low,
            le=high,
            description=(
                f"{spec.description} Unit: {spec.unit}. Plausible range [{low:g}, {high:g}]."
            ),
        ),
    )


if TYPE_CHECKING:
    FeatureValues = FeatureValuesBase
else:
    FeatureValues = create_model(
        "FeatureValues",
        __base__=FeatureValuesBase,
        __doc__=FeatureValuesBase.__doc__,
        **{name: _feature_field(name) for name in INPUT_FEATURES},
    )


# ---------------------------------------------------------------------- raw segments
class SegmentIn(_StrictInput):
    """One (possibly partial) charge or discharge record, NASA sign convention.

    Current is positive while charging and negative while discharging; ``time_s``
    counts seconds from the start of the operation and must not decrease.
    """

    time_s: list[TimeValue] = Field(min_length=2, max_length=HARD_MAX_SAMPLES)
    voltage_v: list[VoltageValue] = Field(min_length=2, max_length=HARD_MAX_SAMPLES)
    current_a: list[CurrentValue] = Field(min_length=2, max_length=HARD_MAX_SAMPLES)
    temperature_c: list[TemperatureValue] = Field(min_length=2, max_length=HARD_MAX_SAMPLES)

    @model_validator(mode="after")
    def _consistent(self) -> SegmentIn:
        lengths = {
            "time_s": len(self.time_s),
            "voltage_v": len(self.voltage_v),
            "current_a": len(self.current_a),
            "temperature_c": len(self.temperature_c),
        }
        if len(set(lengths.values())) != 1:
            raise ValueError(f"arrays must have equal length, got {lengths}")
        decreasing = np.flatnonzero(np.diff(np.asarray(self.time_s)) < 0)
        if decreasing.size:
            raise ValueError(
                f"time_s must not decrease (it does at index {int(decreasing[0]) + 1})"
            )
        return self

    @property
    def n_samples(self) -> int:
        """Number of samples."""
        return len(self.time_s)

    def to_segment(self, kind: Literal["charge", "discharge"]) -> Segment:
        """Convert to the :class:`~battery_soh.data.schema.Segment` the extractors use."""
        return Segment(
            kind=kind,
            time_s=np.asarray(self.time_s, dtype=np.float64),
            voltage_v=np.asarray(self.voltage_v, dtype=np.float64),
            current_a=np.asarray(self.current_a, dtype=np.float64),
            temperature_c=np.asarray(self.temperature_c, dtype=np.float64),
        )


_RE_LOW, _RE_HIGH = FEATURES["eis_re_ohm"].plausible_range
_RCT_LOW, _RCT_HIGH = FEATURES["eis_rct_ohm"].plausible_range


class EisIn(_StrictInput):
    """Latest impedance (EIS) fit before the cycle."""

    re_ohm: float = Field(ge=_RE_LOW, le=_RE_HIGH, description="Electrolyte resistance (ohm).")
    rct_ohm: float = Field(
        ge=_RCT_LOW, le=_RCT_HIGH, description="Charge-transfer resistance (ohm)."
    )


# ---------------------------------------------------------------------- requests
class EstimateRequest(_StrictInput):
    """Estimate SOH from health indicators."""

    features: FeatureValues = Field(
        description="Health indicators of the cycle. Missing ones are median-imputed "
        "(reported in `warnings`)."
    )
    reference: FeatureValues | None = Field(
        default=None,
        description="The cell's commissioning reference (mean of each indicator over its "
        "first valid cycles). Required by relative-mode models.",
    )
    model_id: ModelId | None = Field(
        default=None, description="Model to use; default: best model for the given features."
    )
    confidence: Confidence = Field(default=0.9, description="Interval confidence level.")

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "features": {
                        "chg_cc_time_390_419_s": 2075.4,
                        "chg_cv_time_s": 4429.8,
                        "chg_ica_peak_height_ah_per_v": 3.942,
                        "chg_ica_peak_voltage_v": 4.008,
                        "chg_temp_rise_cc_c": 2.95,
                        "dis_q_380_350_ah": 0.6619,
                        "dis_r0_ohm": 0.1011,
                        "dis_temp_rise_win_c": 5.19,
                    },
                    "confidence": 0.9,
                }
            ]
        }
    )


class CycleEstimateRequest(_StrictInput):
    """Extract health indicators from raw arrays, then estimate SOH."""

    charge: SegmentIn | None = None
    discharge: SegmentIn | None = None
    eis: EisIn | None = None
    reference: FeatureValues | None = None
    model_id: ModelId | None = None
    confidence: Confidence = 0.9

    @model_validator(mode="after")
    def _something_to_measure(self) -> CycleEstimateRequest:
        if self.charge is None and self.discharge is None and self.eis is None:
            raise ValueError("provide at least one of 'charge', 'discharge' or 'eis'")
        return self


class BatchRequest(_StrictInput):
    """Several independent estimate requests (processed atomically)."""

    items: list[EstimateRequest] = Field(min_length=1, max_length=MAX_BATCH_ITEMS)


# ---------------------------------------------------------------------- responses
class ValidationIssue(_Output):
    type: str
    loc: list[str | int]
    msg: str


class ProblemDetails(_Output):
    """RFC 7807 problem details (``application/problem+json``)."""

    type: str = Field(examples=["/problems/model-not-found"])
    title: str
    status: int
    detail: str
    request_id: str | None
    errors: list[ValidationIssue] | None = Field(
        default=None, description="Field-level validation errors (422 only)."
    )


def problem_responses(*codes: int) -> dict[int | str, dict[str, Any]]:
    """OpenAPI ``responses`` entries documenting problem-details errors."""
    descriptions = {
        400: "Malformed request",
        404: "Not found",
        413: "Request body too large",
        422: "Invalid input",
        503: "Service not ready",
    }
    return {
        code: {"model": ProblemDetails, "description": descriptions.get(code, "Error")}
        for code in codes
    }


class LiveStatus(_Output):
    status: Literal["ok"]


class ReadyStatus(_Output):
    status: Literal["ready"]
    models: list[str]


class Interval(_Output):
    lower: float
    upper: float
    confidence: float
    method: str


class Contribution(_Output):
    feature: str
    value: float = Field(description="Input value used (the training median if missing).")
    contribution: float = Field(description="coef_j * z_j in SOH percentage points.")


class Domain(_Output):
    in_domain: bool
    out_of_range: list[str]
    mahalanobis: float
    threshold: float
    missing: list[str]


class EstimateResponse(_Output):
    model_id: str
    soh_pct: float
    interval: Interval
    base_value: float = Field(description="Prediction for an average training cycle.")
    contributions: list[Contribution]
    domain: Domain
    warnings: list[str]


class CycleEstimateResponse(_Output):
    features: dict[str, float | None] = Field(description="Extracted indicators (null = n/a).")
    estimate: EstimateResponse


class BatchResponse(_Output):
    results: list[EstimateResponse]


class ModelMetrics(_Output):
    lobo_mae: float | None
    lobo_rmse: float | None
    coverage_90: float | None


class ModelSummary(_Output):
    model_id: str
    feature_set: str
    mode: Literal["absolute", "relative"]
    features: list[str]
    metrics: ModelMetrics


class DemoCycleRef(_Output):
    cycle_index: int
    soh_pct: float | None
    is_reference: bool


class DemoBattery(_Output):
    battery_id: str
    n_cycles: int = Field(description="Discharge cycles of the cell in the full dataset.")
    reference: dict[str, float | None] = Field(
        description="Commissioning reference per indicator (mean over the first valid cycles)."
    )
    cycles: list[DemoCycleRef]


class SegmentOut(_Output):
    time_s: list[float]
    voltage_v: list[float]
    current_a: list[float]
    temperature_c: list[float]


class EisOut(_Output):
    re_ohm: float | None
    rct_ohm: float | None


class DemoCycle(_Output):
    battery_id: str
    cycle_index: int
    soh_pct: float | None
    qc_valid: bool
    is_reference: bool
    features: dict[str, float | None]
    charge: SegmentOut | None
    discharge: SegmentOut | None
    eis: EisOut | None
