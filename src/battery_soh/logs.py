"""Logging configuration shared by the CLI and the API.

Library modules only ever call ``logging.getLogger(__name__)``; handlers are
attached once, here, by the entry point. Calling :func:`configure_logging`
repeatedly is safe (it replaces the handler instead of stacking duplicates).
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

_HANDLER_NAME = "battery_soh"


class JsonFormatter(logging.Formatter):
    """Render log records as single-line JSON (one event per line)."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key in ("request_id", "path", "method", "status", "duration_ms"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str = "INFO", *, json_format: bool = False) -> None:
    """Attach a single stderr handler to the root logger (idempotent)."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        if handler.get_name() == _HANDLER_NAME:
            root.removeHandler(handler)
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.set_name(_HANDLER_NAME)
    if json_format:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S")
        )
    root.addHandler(handler)
    root.setLevel(level.upper())
