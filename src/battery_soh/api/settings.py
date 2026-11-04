"""Runtime configuration of the API service (environment variables only).

Every setting has a safe default and can be overridden with an ``SOH_``-prefixed
environment variable (``SOH_MODELS_DIR``, ``SOH_MAX_BODY_BYTES``, ...). Nothing is
read from a ``.env`` file: the service must not pick up unrelated keys from a
developer's working directory, and a container gets its configuration from its
environment. Unknown ``SOH_*`` variables (for example ``SOH_PROJECT_ROOT``, which
:func:`battery_soh.config.get_paths` reads) are ignored.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from battery_soh.config import get_paths

__all__ = ["HARD_MAX_BATCH_ITEMS", "HARD_MAX_SAMPLES", "Settings"]

#: Upper bound on samples per array that no configuration can raise (schema limit).
HARD_MAX_SAMPLES = 20_000

#: Upper bound on items per ``/v1/soh/batch`` request that no configuration can raise
#: (schema limit); the served limit is :attr:`Settings.max_batch_items`.
HARD_MAX_BATCH_ITEMS = 1000

_LOG_LEVELS = ("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG")
_ORIGIN_RE = re.compile(r"^https?://[A-Za-z0-9.-]+(:\d{1,5})?$")


class Settings(BaseSettings):
    """Service settings.

    Attributes:
        models_dir: Folder with the model bundles and their ``MANIFEST.json``.
        demo_dir: Folder with ``demo_cycles.json.gz`` (optional; demo endpoints 404 without it).
        benchmark_dir: Folder with the benchmark ``summary.json`` and ``predictions.csv``.
        audit_dir: Folder with the leakage-audit ``audit.json`` (optional).
        frontend_dir: Static web UI served at ``/`` when the folder exists.
        cors_origins: Origins allowed to call the API from a browser. Empty (default):
            same-origin only, no CORS headers are sent. Accepts a JSON list or a
            comma-separated string in the environment.
        max_body_bytes: Largest accepted request body; larger requests get 413.
        max_samples: Largest number of samples per raw array in
            ``/v1/soh/estimate-from-cycle`` (at most :data:`HARD_MAX_SAMPLES`).
        max_batch_items: Largest number of items in one ``/v1/soh/batch`` request
            (at most :data:`HARD_MAX_BATCH_ITEMS`). Each item is a full estimate, so this
            bounds the CPU one request can buy; larger batches get 422.
        log_level: Python log level name.
        log_json: Emit one JSON object per log line.
        enable_metrics: Expose Prometheus metrics at ``/metrics``.
        enable_docs: Serve the interactive OpenAPI docs at ``/docs`` and ``/redoc``.
    """

    model_config = SettingsConfigDict(
        env_prefix="SOH_",
        env_file=None,
        extra="ignore",
        case_sensitive=False,
        validate_default=True,
    )

    models_dir: Path = Field(default_factory=lambda: get_paths().models_dir)
    demo_dir: Path = Field(default_factory=lambda: get_paths().demo_dir)
    benchmark_dir: Path = Field(default_factory=lambda: get_paths().benchmark_dir)
    audit_dir: Path = Field(default_factory=lambda: get_paths().reports / "audit")
    frontend_dir: Path = Field(default_factory=lambda: get_paths().frontend)
    cors_origins: Annotated[list[str], NoDecode] = Field(default_factory=list)
    max_body_bytes: int = Field(default=1_000_000, ge=1_024, le=50_000_000)
    max_samples: int = Field(default=HARD_MAX_SAMPLES, ge=2, le=HARD_MAX_SAMPLES)
    max_batch_items: int = Field(default=100, ge=1, le=HARD_MAX_BATCH_ITEMS)
    log_level: str = "INFO"
    log_json: bool = False
    enable_metrics: bool = True
    enable_docs: bool = True

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _parse_origins(cls, value: Any) -> Any:
        """Accept a JSON list or a comma-separated string; validate each origin."""
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            if text.startswith("["):
                try:
                    value = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"cors_origins is not a valid JSON list: {exc}") from None
            else:
                value = [part.strip() for part in text.split(",") if part.strip()]
        if not isinstance(value, list):
            raise ValueError("cors_origins must be a list of origins")
        origins: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError("cors_origins must contain strings")
            origin = item.strip().rstrip("/")
            if origin != "*" and not _ORIGIN_RE.match(origin):
                raise ValueError(
                    f"invalid CORS origin {item!r}: expected scheme://host[:port] or '*'"
                )
            origins.append(origin)
        return origins

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalise_level(cls, value: Any) -> str:
        level = str(value).strip().upper()
        if level not in _LOG_LEVELS:
            raise ValueError(f"log_level must be one of {', '.join(_LOG_LEVELS)}, got {value!r}")
        return level

    @property
    def log_level_number(self) -> int:
        """Numeric log level (``logging.INFO`` etc.)."""
        return int(logging.getLevelName(self.log_level))
