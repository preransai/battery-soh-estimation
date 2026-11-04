"""Demo endpoints: real NASA cycles bundled with the service (:mod:`battery_soh.api.demo`).

Responses are pre-serialized per demo-file version and carry an ``ETag`` and
``Cache-Control: public, max-age=300`` (:mod:`battery_soh.api.http_cache`).
"""

from __future__ import annotations

import logging
from pathlib import Path as FsPath
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Path, Request, Response

from battery_soh.api.demo import DemoDataError, DemoStore, get_demo_store
from battery_soh.api.dependencies import get_settings
from battery_soh.api.errors import ProblemError
from battery_soh.api.http_cache import NOT_MODIFIED_RESPONSE, CachedJson, cached_json_response
from battery_soh.api.schemas import (
    BATTERY_ID_PATTERN,
    DemoBattery,
    DemoCycle,
    problem_responses,
)
from battery_soh.api.settings import Settings

router = APIRouter(prefix="/v1/demo", tags=["demo"])
logger = logging.getLogger(__name__)


def demo_store(demo_dir: FsPath) -> DemoStore:
    """The installed demo data.

    Raises:
        ProblemError: 404 if no demo asset is installed, 503 if it is corrupt.
    """
    try:
        store = get_demo_store(demo_dir)
    except DemoDataError as exc:
        logger.error("demo asset is invalid: %s", exc)
        raise ProblemError(
            503,
            "Demo data unavailable",
            "The installed demo dataset failed validation.",
            type_="/problems/demo-unavailable",
        ) from None
    if store is None:
        raise ProblemError(
            404,
            "Demo data not installed",
            "No demo dataset is installed; build it with `soh demo-assets`.",
            type_="/problems/demo-unavailable",
        )
    return store


def batteries_payload(store: DemoStore) -> list[dict[str, Any]]:
    """``GET /v1/demo/batteries``."""
    return store.battery_list()


def _cycle_not_found(battery_id: str, cycle_index: int) -> ProblemError:
    return ProblemError(
        404,
        "Demo cycle not found",
        f"No demo data for battery '{battery_id}' cycle {cycle_index}; "
        "see GET /v1/demo/batteries for the available cycles.",
        type_="/problems/demo-cycle-not-found",
    )


def cycle_payload(store: DemoStore, battery_id: str, cycle_index: int) -> dict[str, Any]:
    """``GET /v1/demo/batteries/{battery_id}/cycles/{cycle_index}``.

    Raises:
        ProblemError: 404 if the cycle is not part of the demo data.
    """
    cycle = store.cycle(battery_id, cycle_index)
    if cycle is None:
        raise _cycle_not_found(battery_id, cycle_index)
    return cycle


def cycle_document(store: DemoStore, battery_id: str, cycle_index: int) -> CachedJson:
    """:func:`cycle_payload`, serialized once per demo-file version.

    Raises:
        ProblemError: 404 if the cycle is not part of the demo data.
    """
    cached = store.cycle_json(battery_id, cycle_index)
    if cached is None:
        raise _cycle_not_found(battery_id, cycle_index)
    return cached


@router.get(
    "/batteries",
    response_model=list[DemoBattery],
    summary="Demo cells and their available cycles",
    responses={**NOT_MODIFIED_RESPONSE, **problem_responses(404, 503)},
)
def list_batteries(
    request: Request, settings: Annotated[Settings, Depends(get_settings)]
) -> Response:
    """The demo cells, their commissioning reference and the cycles with raw data."""
    return cached_json_response(request, demo_store(settings.demo_dir).battery_list_json())


@router.get(
    "/batteries/{battery_id}/cycles/{cycle_index}",
    response_model=DemoCycle,
    summary="One demo cycle: downsampled raw arrays, indicators and true SOH",
    responses={**NOT_MODIFIED_RESPONSE, **problem_responses(404, 422, 503)},
)
def get_cycle(
    request: Request,
    battery_id: Annotated[str, Path(pattern=BATTERY_ID_PATTERN)],
    cycle_index: Annotated[int, Path(ge=0, le=100_000)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> Response:
    """Downsampled charge/discharge records (the request body format of
    ``/v1/soh/estimate-from-cycle``), the indicators measured on the full-resolution
    record, and the true SOH label."""
    document = cycle_document(demo_store(settings.demo_dir), battery_id, cycle_index)
    return cached_json_response(request, document)
