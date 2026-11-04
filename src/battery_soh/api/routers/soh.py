"""SOH estimation endpoints.

All three handlers are plain ``def``: the work is CPU-bound (feature extraction,
numpy arithmetic), so FastAPI runs them in its threadpool instead of blocking
the event loop.
"""

from __future__ import annotations

import time
from typing import Annotated, Any

from fastapi import APIRouter, Depends
from fastapi.exceptions import RequestValidationError

from battery_soh.api import inference
from battery_soh.api.dependencies import get_metrics, get_registry, get_settings
from battery_soh.api.errors import ProblemError
from battery_soh.api.metrics import ApiMetrics
from battery_soh.api.registry import ModelRegistry
from battery_soh.api.schemas import (
    BatchRequest,
    BatchResponse,
    CycleEstimateRequest,
    CycleEstimateResponse,
    EstimateRequest,
    EstimateResponse,
    problem_responses,
)
from battery_soh.api.settings import Settings

router = APIRouter(prefix="/v1/soh", tags=["soh"])

_ERRORS = problem_responses(404, 413, 422, 503)


def _run_estimate(
    registry: ModelRegistry, metrics: ApiMetrics | None, body: EstimateRequest
) -> dict[str, Any]:
    start = time.perf_counter()
    result = inference.estimate(
        registry,
        body.features.provided(),
        reference=body.reference.provided() if body.reference is not None else None,
        model_id=body.model_id,
        confidence=body.confidence,
    )
    if metrics is not None:
        metrics.observe_prediction(
            result["model_id"], result["domain"]["in_domain"], time.perf_counter() - start
        )
    return result


@router.post(
    "/estimate",
    response_model=EstimateResponse,
    summary="Estimate SOH from health indicators",
    responses=_ERRORS,
)
def estimate(
    body: EstimateRequest,
    registry: Annotated[ModelRegistry, Depends(get_registry)],
    metrics: Annotated[ApiMetrics | None, Depends(get_metrics)],
) -> dict[str, Any]:
    """SOH (percent of the cell's own initial capacity) with a battery-level conformal
    interval, the exact per-feature attribution, and a training-domain check."""
    return _run_estimate(registry, metrics, body)


@router.post(
    "/estimate-from-cycle",
    response_model=CycleEstimateResponse,
    summary="Extract health indicators from raw arrays, then estimate SOH",
    responses=_ERRORS,
)
def estimate_from_cycle(
    body: CycleEstimateRequest,
    registry: Annotated[ModelRegistry, Depends(get_registry)],
    settings: Annotated[Settings, Depends(get_settings)],
    metrics: Annotated[ApiMetrics | None, Depends(get_metrics)],
) -> dict[str, Any]:
    """Runs the same feature extractor that built the training data (train/serve parity)
    on a (partial) CC charge and/or discharge record, then estimates SOH."""
    too_long = [
        {
            "type": "too_long",
            "loc": ("body", name, "time_s"),
            "msg": f"List should have at most {settings.max_samples} items",
            "input": None,
        }
        for name, segment in (("charge", body.charge), ("discharge", body.discharge))
        if segment is not None and segment.n_samples > settings.max_samples
    ]
    if too_long:
        raise RequestValidationError(too_long)
    start = time.perf_counter()
    result = inference.estimate_from_cycle(
        registry,
        body.charge.to_segment("charge") if body.charge is not None else None,
        body.discharge.to_segment("discharge") if body.discharge is not None else None,
        eis=(body.eis.re_ohm, body.eis.rct_ohm) if body.eis is not None else None,
        reference=body.reference.provided() if body.reference is not None else None,
        model_id=body.model_id,
        confidence=body.confidence,
    )
    if metrics is not None:
        estimate_doc = result["estimate"]
        metrics.observe_prediction(
            estimate_doc["model_id"],
            estimate_doc["domain"]["in_domain"],
            time.perf_counter() - start,
        )
    return result


@router.post(
    "/batch",
    response_model=BatchResponse,
    summary="Estimate SOH for a batch of inputs (100 per request by default)",
    responses=_ERRORS,
)
def batch(
    body: BatchRequest,
    registry: Annotated[ModelRegistry, Depends(get_registry)],
    settings: Annotated[Settings, Depends(get_settings)],
    metrics: Annotated[ApiMetrics | None, Depends(get_metrics)],
) -> dict[str, Any]:
    """Independent estimates, in request order. The batch is atomic: if any item fails,
    the whole request fails with the item's index in the problem detail. The server's
    limit on items per request is `SOH_MAX_BATCH_ITEMS` (default 100, at most 1000);
    larger batches get 422 before any estimate is computed."""
    if len(body.items) > settings.max_batch_items:
        raise RequestValidationError(
            [
                {
                    "type": "too_long",
                    "loc": ("body", "items"),
                    "msg": (
                        f"List should have at most {settings.max_batch_items} items "
                        f"(this server's batch limit), not {len(body.items)}; "
                        "split the request"
                    ),
                    "input": None,
                }
            ]
        )
    results = []
    for index, item in enumerate(body.items):
        try:
            results.append(_run_estimate(registry, metrics, item))
        except ProblemError as exc:
            raise ProblemError(
                exc.status, exc.title, f"items[{index}]: {exc.detail}", type_=exc.type
            ) from None
    return {"results": results}
