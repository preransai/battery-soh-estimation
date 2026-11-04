"""SOH estimation service logic (framework-free, shared by the routers).

:func:`estimate` turns validated health indicators into the
``POST /v1/soh/estimate`` response; :func:`estimate_from_cycle` first extracts
the indicators from raw arrays with
:func:`battery_soh.features.extract.features_from_segments`, the same function
that built the training table (train/serve parity).

Model choice when the caller does not name a model:

* absolute-mode bundles are always eligible; relative-mode bundles only when a
  commissioning ``reference`` is supplied, and only for the indicators that
  have a usable (finite, non-zero) reference value;
* among the bundles whose inputs are all available, the lowest recorded
  leave-one-battery-out MAE wins;
* if no bundle is complete, the absolute bundle covering the most provided
  indicators is used and the missing ones are median-imputed (reported in
  ``warnings``). A relative bundle is never used with imputed inputs unless the
  caller names it explicitly.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from battery_soh.api.errors import ProblemError
from battery_soh.api.registry import ModelEntry, ModelRegistry
from battery_soh.api.schemas import INPUT_FEATURES
from battery_soh.data.schema import Segment
from battery_soh.features.extract import METADATA_FEATURES, features_from_segments
from battery_soh.features.registry import FEATURES
from battery_soh.features.relative import apply_reference, is_context_feature

__all__ = ["choose_model", "estimate", "estimate_from_cycle", "extract_features"]

logger = logging.getLogger(__name__)

_HI_PREFIXES = ("chg_", "dis_", "eis_")


def _usable_reference(value: float | None) -> bool:
    return value is not None and math.isfinite(value) and value != 0.0


def _referenced(features: Mapping[str, float], reference: Mapping[str, float]) -> set[str]:
    """Provided features usable in relative mode (context needs no reference)."""
    return {f for f in features if is_context_feature(f) or _usable_reference(reference.get(f))}


def choose_model(
    registry: ModelRegistry,
    features: Mapping[str, float],
    *,
    reference: Mapping[str, float] | None = None,
    model_id: str | None = None,
) -> ModelEntry:
    """Resolve the bundle for one request (see the module docstring).

    Raises:
        ProblemError: 503 if no model is loaded, 404 for an unknown ``model_id``,
            422 if no model can use any of the provided features.
    """
    if not registry:
        raise ProblemError(
            503,
            "No model available",
            "The service has no verified model bundle loaded; check /readyz.",
            type_="/problems/no-models",
        )
    if model_id is not None:
        entry = registry.get(model_id)
        if entry is None:
            raise ProblemError(
                404,
                "Model not found",
                f"No model bundle with id '{model_id}'. See GET /v1/models.",
                type_="/problems/model-not-found",
            )
        return entry

    present = set(features)
    referenced = _referenced(features, reference) if reference else set()
    complete = [registry.select(present, "absolute")]
    if reference:
        complete.append(registry.select(referenced, "relative"))
    found = [e for e in complete if e is not None]
    if found:
        return min(found, key=lambda e: e.rank_key)

    # Partial inputs: impute the rest, absolute models only (a relative model is used
    # automatically only when every input has a value and a reference).
    entry = registry.select(present, "absolute", allow_partial=True)
    if entry is not None:
        return entry

    hint = ""
    if not registry.has_mode("absolute"):
        hint = (
            " The loaded models are relative: provide every input together with its "
            "commissioning 'reference'."
        )
    needed = sorted({f for e in registry.entries for f in e.features})
    raise ProblemError(
        422,
        "No usable features",
        f"No loaded model uses any of the provided features; models need some of: "
        f"{', '.join(needed)}.{hint}",
        type_="/problems/no-usable-features",
    )


def _model_inputs(
    entry: ModelEntry, features: Mapping[str, float], reference: Mapping[str, float] | None
) -> dict[str, float]:
    """The bundle's inputs in the bundle's mode (relative ones divided by the reference)."""
    provided = {f: features[f] for f in entry.features if f in features}
    if entry.mode != "relative":
        return provided
    bad = [
        f
        for f in provided
        if not is_context_feature(f) and not _usable_reference((reference or {}).get(f))
    ]
    if bad:
        raise ProblemError(
            422,
            "Reference required",
            f"Model '{entry.model_id}' is relative: provide non-zero 'reference' values for "
            f"{', '.join(bad)}.",
            type_="/problems/reference-required",
        )
    relative = apply_reference(provided, reference or {})
    non_finite = sorted(f for f, v in relative.items() if not math.isfinite(v))
    if non_finite:
        raise ProblemError(
            422,
            "Reference required",
            f"The reference values for {', '.join(non_finite)} give non-finite ratios.",
            type_="/problems/reference-required",
        )
    return relative


def _all_finite(*values: float) -> bool:
    return all(math.isfinite(v) for v in values)


def _short(exc: BaseException, limit: int = 200) -> str:
    """``Type: message`` on one printable line, truncated (safe for a log line)."""
    text = f"{type(exc).__name__}: {exc}"
    text = "".join(ch if ch.isprintable() else " " for ch in text)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def estimate(
    registry: ModelRegistry,
    features: Mapping[str, float],
    *,
    reference: Mapping[str, float] | None = None,
    model_id: str | None = None,
    confidence: float = 0.9,
) -> dict[str, Any]:
    """Estimate SOH (percent) with a conformal interval, attribution and domain check.

    Args:
        registry: Loaded models.
        features: Validated, finite health indicators (absolute values).
        reference: The cell's commissioning reference per indicator (relative models).
        model_id: Explicit model, or ``None`` for automatic selection.
        confidence: Interval confidence level (one of the bundle's levels).

    Returns:
        The ``POST /v1/soh/estimate`` response document.

    Raises:
        ProblemError: For client errors (unknown model, no usable features,
            missing reference, unsupported confidence) or no loaded model.
    """
    entry = choose_model(registry, features, reference=reference, model_id=model_id)
    bundle = entry.bundle
    inputs = _model_inputs(entry, features, reference)
    if not inputs:
        raise ProblemError(
            422,
            "No usable features",
            f"Model '{entry.model_id}' needs at least one of: {', '.join(bundle.features)}.",
            type_="/problems/no-usable-features",
        )
    try:
        half_width = bundle.half_width(confidence)
    except ValueError:
        levels = ", ".join(f"{c:g}" for c in bundle.confidences)
        raise ProblemError(
            422,
            "Unsupported confidence",
            f"Model '{entry.model_id}' has no conformal interval at {confidence:g}; "
            f"available: {levels}.",
            type_="/problems/unsupported-confidence",
        ) from None

    explanation = bundle.explain(inputs)
    domain = bundle.domain_check(inputs)
    parts = [c for _, _, c in explanation.contributions]
    if not _all_finite(explanation.raw_prediction, domain.mahalanobis, *parts):
        raise ProblemError(
            422,
            "Input out of numeric range",
            "The inputs produce a non-finite model output; check their units and scale.",
            type_="/problems/non-finite-result",
        )
    prediction = explanation.prediction
    lower, upper = bundle.interval(prediction, confidence)

    warnings: list[str] = []
    if domain.missing:
        warnings.append(
            f"Imputed the training median for {len(domain.missing)} missing feature(s): "
            f"{', '.join(domain.missing)}."
        )
    if explanation.clipped:
        low, high = bundle.output_clip
        warnings.append(f"Prediction clipped to the physical range [{low:g}, {high:g}].")
    if math.isinf(half_width):
        warnings.append(
            f"No finite conformal interval at confidence {confidence:g} (too few calibration "
            "scores); the interval spans the whole output range."
        )
    if not domain.in_domain:
        warnings.append("Input is outside the training domain; the interval may not hold.")

    return {
        "model_id": entry.model_id,
        "soh_pct": prediction,
        "interval": {
            "lower": lower,
            "upper": upper,
            "confidence": float(confidence),
            "method": bundle.conformal_method,
        },
        "base_value": bundle.base_value,
        "contributions": [
            {"feature": name, "value": value, "contribution": contribution}
            for name, value, contribution in explanation.contributions
        ],
        "domain": {
            "in_domain": domain.in_domain,
            "out_of_range": list(domain.out_of_range),
            "mahalanobis": domain.mahalanobis,
            "threshold": domain.threshold,
            "missing": list(domain.missing),
        },
        "warnings": warnings,
    }


#: Indicators :func:`features_from_segments` can measure (non-leaky ones only; the
#: metadata context features are not measurable from the arrays).
EXTRACTED_FEATURES: tuple[str, ...] = tuple(
    name for name in INPUT_FEATURES if name not in METADATA_FEATURES
)


def extract_features(
    charge: Segment | None,
    discharge: Segment | None,
    eis: tuple[float, float] | None,
) -> tuple[dict[str, float | None], list[str]]:
    """Measure the non-leaky indicators of one cycle.

    Values that are not computable are ``None``; values outside the feature's
    plausible range (e.g. a corrupted record) are dropped to ``None`` with a
    warning, exactly as ``/v1/soh/estimate`` would reject them.

    Returns:
        ``(features, warnings)``; ``features`` has every key of
        :data:`EXTRACTED_FEATURES`.
    """
    raw = features_from_segments(charge, discharge, eis=eis)
    features: dict[str, float | None] = {}
    warnings: list[str] = []
    for name in EXTRACTED_FEATURES:
        value = raw.get(name, math.nan)
        if not math.isfinite(value):
            features[name] = None
            continue
        low, high = FEATURES[name].plausible_range
        if not low <= value <= high:
            warnings.append(
                f"Extracted {name}={value:.6g} is outside its plausible range "
                f"[{low:g}, {high:g}] and was ignored."
            )
            features[name] = None
            continue
        features[name] = value
    for seg, expected in ((charge, 1.0), (discharge, -1.0)):
        if seg is not None and seg.n_samples:
            loaded = seg.current_a[np.abs(seg.current_a) > 0.1]
            if loaded.size and float(expected * loaded.mean()) < 0:
                warnings.append(
                    f"The {seg.kind} current has the wrong sign for the NASA convention "
                    "(charge positive, discharge negative); indicators may be missing."
                )
    return features, warnings


def estimate_from_cycle(
    registry: ModelRegistry,
    charge: Segment | None,
    discharge: Segment | None,
    *,
    eis: tuple[float, float] | None = None,
    reference: Mapping[str, float] | None = None,
    model_id: str | None = None,
    confidence: float = 0.9,
) -> dict[str, Any]:
    """Extract indicators from raw segments and estimate SOH.

    Raises:
        ProblemError: 422 if the extractor rejects the record or no health
            indicator can be measured from it, plus everything :func:`estimate` raises.
    """
    try:
        features, warnings = extract_features(charge, discharge, eis)
    except ValueError as exc:
        # A client can trigger this at will: one short line at WARNING, the traceback
        # only at DEBUG (no log flooding, no stack traces for bad input).
        logger.warning("the feature extractor rejected a record: %s", _short(exc))
        logger.debug("feature extractor traceback", exc_info=True)
        raise ProblemError(
            422,
            "Invalid record",
            "The feature extractor rejected the record; check the arrays and units.",
            type_="/problems/invalid-record",
        ) from None
    present = {k: v for k, v in features.items() if v is not None}
    if not any(k.startswith(_HI_PREFIXES) for k in present):
        raise ProblemError(
            422,
            "No health indicators could be extracted",
            "The record never spans a measurement window (3.90-4.19 V of a CC charge, "
            "3.80-3.50 V of a discharge under load) and no EIS values were given.",
            type_="/problems/no-health-indicators",
        )
    result = estimate(
        registry, present, reference=reference, model_id=model_id, confidence=confidence
    )
    result["warnings"] = [*warnings, *result["warnings"]]
    return {"features": features, "estimate": result}
