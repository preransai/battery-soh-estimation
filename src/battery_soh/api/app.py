"""Application factory: ``uvicorn --factory battery_soh.api.app:create_app``.

The app serves the versioned JSON API under ``/v1``, health probes, Prometheus
metrics, the OpenAPI docs and (when the folder exists) the static web UI at
``/``. Model bundles are loaded and verified once, in the lifespan handler;
until that succeeds ``/readyz`` answers 503 while ``/livez`` stays 200.

Middleware, outermost first:

1. :class:`~battery_soh.api.middleware.RequestContextMiddleware` - request id,
   security headers, access log, RED metrics, 500 safety net;
2. CORS (only when ``SOH_CORS_ORIGINS`` is set; same-origin otherwise);
3. gzip for responses over 1 kB;
4. :class:`~battery_soh.api.middleware.BodySizeLimitMiddleware` - 413 before parsing.

The service makes no outbound network calls and writes nothing to disk.
"""

from __future__ import annotations

import logging
import mimetypes
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import PurePosixPath

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException
from starlette.responses import Response as StarletteResponse
from starlette.routing import Match, Mount
from starlette.types import Scope

from battery_soh import __version__
from battery_soh.api.demo import DemoDataError, get_demo_store
from battery_soh.api.docs import install_docs
from battery_soh.api.errors import register_exception_handlers
from battery_soh.api.metrics import ApiMetrics
from battery_soh.api.middleware import (
    REQUEST_ID_HEADER,
    BodySizeLimitMiddleware,
    RequestContextMiddleware,
    install_log_record_factory,
)
from battery_soh.api.registry import ModelRegistry
from battery_soh.api.routers import benchmark, demo, health, models, soh
from battery_soh.api.settings import Settings
from battery_soh.logs import configure_logging

__all__ = ["create_app"]

logger = logging.getLogger(__name__)

_DESCRIPTION = """
Leakage-free Li-ion battery **state-of-health** estimation from partial charge /
discharge windows, with battery-level conformal prediction intervals.

* SOH is `100 * capacity / C0` of the cell's own initial capacity; capacity and
  other full-discharge statistics are never model inputs.
* Intervals are conformal half-widths of leave-one-battery-out residuals. Their
  coverage is measured on held-out cells, not guaranteed: four training cells are too
  few for a finite-sample guarantee at the battery level.
* Every error is RFC 7807 `application/problem+json` with a `request_id`.
"""


class _PublicFiles(StaticFiles):
    """Static files without hidden files, answering 404 (not 405) to non-GET methods."""

    async def get_response(self, path: str, scope: Scope) -> StarletteResponse:
        parts = PurePosixPath(path.replace("\\", "/")).parts
        if scope["method"] not in ("GET", "HEAD") or any(p.startswith(".") for p in parts):
            raise HTTPException(status_code=404)
        return await super().get_response(path, scope)


class _FrontendMount(Mount):
    """Mount of the web UI at ``/`` that never claims ``/v1`` paths.

    Without this, a request such as ``GET /v1/soh/estimate`` would be a full
    match for the catch-all mount and answer 404 instead of 405.
    """

    def matches(self, scope: Scope) -> tuple[Match, Scope]:
        path = scope.get("path", "")
        if path == "/v1" or path.startswith("/v1/"):
            return Match.NONE, {}
        return super().matches(scope)


def _register_media_types() -> None:
    # Windows registries sometimes map .js to text/plain; with `nosniff` browsers
    # would then refuse the ES modules of the web UI.
    for media_type, extension in (
        ("text/javascript", ".js"),
        ("text/javascript", ".mjs"),
        ("text/css", ".css"),
        ("image/svg+xml", ".svg"),
        ("application/json", ".json"),
    ):
        mimetypes.add_type(media_type, extension)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI application.

    Args:
        settings: Configuration; read from ``SOH_*`` environment variables when omitted.

    Returns:
        The application. Model bundles load when its lifespan starts (``uvicorn``,
        or ``with TestClient(app)`` in tests).
    """
    settings = settings or Settings()
    configure_logging(settings.log_level, json_format=settings.log_json)
    install_log_record_factory()
    _register_media_types()
    metrics = ApiMetrics() if settings.enable_metrics else None

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        registry = await run_in_threadpool(ModelRegistry.load, settings.models_dir)
        app.state.registry = registry
        if metrics is not None:
            metrics.models_loaded.set(len(registry))
        if registry:
            logger.info("ready: serving %s", ", ".join(registry.model_ids))
        else:
            logger.error("not ready: %s", registry.error)
        try:  # warm the demo cache so the first demo request is fast
            store = await run_in_threadpool(get_demo_store, settings.demo_dir)
        except DemoDataError as exc:
            logger.error("the demo asset is invalid; demo endpoints will fail: %s", exc)
        else:
            if store is None:
                logger.info("no demo asset in %s; demo endpoints answer 404", settings.demo_dir)
        yield
        logger.info("shutting down")

    app = FastAPI(
        title="battery-soh",
        version=__version__,
        description=_DESCRIPTION,
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url="/openapi.json" if settings.enable_docs else None,
        swagger_ui_oauth2_redirect_url=None,
    )
    app.state.settings = settings
    app.state.registry = ModelRegistry.empty("starting")
    app.state.metrics = metrics
    register_exception_handlers(app)

    for router in (health.router, models.router, soh.router, demo.router, benchmark.router):
        app.include_router(router)

    if metrics is not None:

        @app.get("/metrics", include_in_schema=False)
        def prometheus_metrics() -> Response:
            return Response(metrics.render(), media_type=metrics.content_type)

    csp_overrides: dict[str, str] = {}
    if settings.enable_docs:
        csp_overrides = install_docs(
            app, title="battery-soh API", favicon_url="/assets/favicon.svg"
        )

    frontend = settings.frontend_dir
    if (frontend / "index.html").is_file():
        ui = _PublicFiles(directory=frontend, html=True)
        app.router.routes.append(_FrontendMount("/", app=ui, name="frontend"))
    else:
        logger.info("no web UI at %s; serving the API only", frontend)

    # add_middleware prepends: the last one added is the outermost.
    app.add_middleware(BodySizeLimitMiddleware, max_body_bytes=settings.max_body_bytes)
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type", REQUEST_ID_HEADER],
            expose_headers=[REQUEST_ID_HEADER],
            allow_credentials=False,
            max_age=600,
        )
    app.add_middleware(RequestContextMiddleware, metrics=metrics, csp_overrides=csp_overrides)
    return app
