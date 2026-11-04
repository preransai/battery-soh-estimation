"""Accessors for per-application state (used as FastAPI dependencies)."""

from __future__ import annotations

from typing import cast

from fastapi import Request

from battery_soh.api.metrics import ApiMetrics
from battery_soh.api.registry import ModelRegistry
from battery_soh.api.settings import Settings

__all__ = ["get_metrics", "get_registry", "get_settings"]


def get_settings(request: Request) -> Settings:
    """Settings the application was created with."""
    return cast(Settings, request.app.state.settings)


def get_registry(request: Request) -> ModelRegistry:
    """The model registry loaded at startup (empty until the lifespan has run)."""
    return cast(ModelRegistry, request.app.state.registry)


def get_metrics(request: Request) -> ApiMetrics | None:
    """Metric families, or ``None`` when metrics are disabled."""
    return cast("ApiMetrics | None", request.app.state.metrics)
