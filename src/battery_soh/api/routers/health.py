"""Liveness and readiness probes.

``/livez`` answers as long as the process serves HTTP. ``/readyz`` answers 200
only once the model registry holds at least one verified bundle; until then (or
when verification failed) it returns 503 problem details, so orchestrators keep
traffic away from an instance that cannot predict.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends

from battery_soh.api.dependencies import get_registry
from battery_soh.api.errors import ProblemError
from battery_soh.api.registry import ModelRegistry
from battery_soh.api.schemas import LiveStatus, ReadyStatus, problem_responses

router = APIRouter(tags=["health"])


@router.get("/livez", response_model=LiveStatus, summary="Liveness probe")
async def livez() -> dict[str, Any]:
    """The process is up."""
    return {"status": "ok"}


@router.get(
    "/readyz",
    response_model=ReadyStatus,
    summary="Readiness probe",
    responses=problem_responses(503),
)
async def readyz(registry: Annotated[ModelRegistry, Depends(get_registry)]) -> dict[str, Any]:
    """Ready when at least one verified model bundle is loaded; lists the model ids."""
    if not registry:
        raise ProblemError(
            503,
            "Not ready",
            "No verified model bundle is loaded.",
            type_="/problems/not-ready",
        )
    return {"status": "ready", "models": registry.model_ids}
