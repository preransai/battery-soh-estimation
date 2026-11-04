"""Logging set-up shared by the CLI and the API: JSON lines and an idempotent root handler."""

from __future__ import annotations

import json
import logging
import re
import sys
from collections.abc import Iterator
from datetime import datetime, timedelta
from typing import Any

import pytest

from battery_soh.logs import JsonFormatter, configure_logging

HANDLER_NAME = "battery_soh"


def _record(
    msg: str = "hello %s", args: tuple[Any, ...] = ("world",), **attrs: Any
) -> logging.LogRecord:
    record = logging.LogRecord("battery_soh.test", logging.WARNING, __file__, 1, msg, args, None)
    for key, value in attrs.items():
        setattr(record, key, value)
    return record


# --------------------------------------------------------------------------- JsonFormatter


def test_json_line_has_the_core_fields() -> None:
    line = JsonFormatter().format(_record("caf\u00e9 %s \u03a9", ("ok",)))
    assert "\n" not in line
    assert "\u03a9" in line  # not ASCII-escaped
    payload = json.loads(line)
    assert set(payload) == {"ts", "level", "logger", "msg"}
    assert payload["level"] == "WARNING"
    assert payload["logger"] == "battery_soh.test"
    assert payload["msg"] == "caf\u00e9 ok \u03a9"  # arguments are interpolated
    assert datetime.fromisoformat(payload["ts"]).utcoffset() == timedelta(0)


def test_json_line_copies_the_request_context_and_nothing_else() -> None:
    context = {
        "request_id": "req-1",
        "path": "/v1/soh/estimate",
        "method": "POST",
        "status": 200,
        "duration_ms": 1.25,
    }
    payload = json.loads(JsonFormatter().format(_record(**context, user="not logged")))
    assert {k: payload[k] for k in context} == context
    assert "user" not in payload


def test_json_line_carries_the_traceback() -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        record = _record(exc_info=sys.exc_info())
    line = JsonFormatter().format(record)
    assert "\n" not in line  # the traceback is escaped inside one JSON string
    trace = json.loads(line)["exc_info"]
    assert trace.startswith("Traceback (most recent call last):")
    assert trace.rstrip().endswith("ValueError: boom")


# --------------------------------------------------------------------------- configure_logging


@pytest.fixture
def root_logger() -> Iterator[logging.Logger]:
    """The root logger, with its level and any earlier handler of ours restored afterwards."""
    root = logging.getLogger()
    level = root.level
    previous = [h for h in root.handlers if h.get_name() == HANDLER_NAME]
    yield root
    for handler in _ours(root):
        root.removeHandler(handler)
    for handler in previous:
        root.addHandler(handler)
    root.setLevel(level)


def _ours(root: logging.Logger) -> list[logging.Handler]:
    return [h for h in root.handlers if h.get_name() == HANDLER_NAME]


def test_repeated_calls_replace_the_handler(
    root_logger: logging.Logger, capsys: pytest.CaptureFixture[str]
) -> None:
    configure_logging("debug")
    configure_logging("warning", json_format=True)
    (handler,) = _ours(root_logger)
    assert isinstance(handler.formatter, JsonFormatter)
    assert root_logger.level == logging.WARNING
    log = logging.getLogger("battery_soh.test")
    log.info("below the level")
    log.warning("ready %d", 1)
    lines = capsys.readouterr().err.splitlines()
    assert [json.loads(line)["msg"] for line in lines] == ["ready 1"]  # once, not twice


def test_plain_format_and_foreign_handlers_are_kept(
    root_logger: logging.Logger, capsys: pytest.CaptureFixture[str]
) -> None:
    foreign = logging.NullHandler()
    root_logger.addHandler(foreign)
    try:
        configure_logging("INFO")
        configure_logging("INFO")
        assert foreign in root_logger.handlers
        (handler,) = _ours(root_logger)
        assert not isinstance(handler.formatter, JsonFormatter)
        logging.getLogger("battery_soh.test").info("plain")
        assert re.fullmatch(
            r"\d\d:\d\d:\d\d INFO {4}battery_soh\.test: plain", capsys.readouterr().err.strip()
        )
    finally:
        root_logger.removeHandler(foreign)
