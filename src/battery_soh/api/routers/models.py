"""Model listing and model cards."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Path

from battery_soh.api.dependencies import get_registry
from battery_soh.api.errors import ProblemError
from battery_soh.api.registry import ModelRegistry
from battery_soh.api.schemas import MODEL_ID_PATTERN, ModelSummary, problem_responses

router = APIRouter(prefix="/v1/models", tags=["models"])


def list_models_payload(registry: ModelRegistry) -> list[dict[str, Any]]:
    """``GET /v1/models``: one summary per bundle, best recorded LOBO MAE first."""
    return [entry.summary() for entry in registry.entries]


def model_card_payload(registry: ModelRegistry, model_id: str) -> dict[str, Any]:
    """``GET /v1/models/{model_id}``: the bundle without its precision matrix.

    Raises:
        ProblemError: 404 if no such model is loaded.
    """
    entry = registry.get(model_id)
    if entry is None:
        raise ProblemError(
            404,
            "Model not found",
            f"No model bundle with id '{model_id}'. See GET /v1/models.",
            type_="/problems/model-not-found",
        )
    return entry.bundle.to_card()


@router.get("", response_model=list[ModelSummary], summary="List the served models")
def list_models(
    registry: Annotated[ModelRegistry, Depends(get_registry)],
) -> list[dict[str, Any]]:
    """Models with their feature set, mode, inputs and leave-one-battery-out metrics."""
    return list_models_payload(registry)


@router.get(
    "/{model_id}",
    response_model=dict[str, Any],
    summary="Model card",
    responses=problem_responses(404, 422),
)
def get_model(
    model_id: Annotated[str, Path(pattern=MODEL_ID_PATTERN)],
    registry: Annotated[ModelRegistry, Depends(get_registry)],
) -> dict[str, Any]:
    """The full model bundle (coefficients, preprocessing, conformal quantiles, domain
    envelope, training record and metrics) minus the large precision matrix."""
    return model_card_payload(registry, model_id)
