"""Benchmark results (``reports/benchmark``) and the leakage audit (``reports/audit``).

``GET /v1/benchmark`` returns ``summary.json`` as written by
:func:`battery_soh.evaluation.benchmark.run_benchmark`, enriched with:

* ``predictions`` - the out-of-fold per-cycle predictions of the *headline*
  specification (rows of ``predictions.csv`` whose ``experiment``/``feature_set``/
  ``mode``/``model`` columns equal ``summary["headline"]``), unless the summary
  already contains them. Rows: ``{battery_id, cycle_index, y, yhat, lo_80, hi_80,
  lo_90, hi_90, lo_95, hi_95, ...}`` (whichever bounds and flags the CSV has).
* ``audit`` - the content of ``audit.json``, unless the summary already has one.

Files are re-read only when they change, and the serialized response is cached
per file version (keys: path, modification time and size of the summary, the
predictions and the audit), so a request costs no JSON encoding; responses carry
an ``ETag`` and ``Cache-Control: public, max-age=300`` and a matching
``If-None-Match`` gets ``304`` (:mod:`battery_soh.api.http_cache`). Non-finite
numbers are returned as ``null`` (strict JSON).
"""

from __future__ import annotations

import csv
import json
import logging
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request, Response

from battery_soh.api.dependencies import get_settings
from battery_soh.api.errors import ProblemError
from battery_soh.api.http_cache import (
    NOT_MODIFIED_RESPONSE,
    CachedJson,
    cached_json_response,
    to_cached_json,
)
from battery_soh.api.schemas import problem_responses
from battery_soh.api.settings import Settings

router = APIRouter(prefix="/v1", tags=["benchmark"])
logger = logging.getLogger(__name__)

SUMMARY_FILE = "summary.json"
PREDICTIONS_FILE = "predictions.csv"
AUDIT_FILE = "audit.json"

#: Upper bounds that keep a malformed or unexpected report from blowing up responses.
MAX_REPORT_BYTES = 20_000_000
MAX_PREDICTION_ROWS = 20_000

_SPEC_COLUMNS = ("experiment", "feature_set", "mode", "model")
_BOUND_COLUMN = re.compile(r"^(lo|hi)_(\d{2})$")
_FLAG_COLUMNS = ("is_reference_cycle", "qc_valid")
_COLUMN_ALIASES = {
    "battery_id": ("battery_id", "battery"),
    "cycle_index": ("cycle_index", "cycle"),
    "y": ("y", "y_true", "soh_pct"),
    "yhat": ("yhat", "y_pred", "prediction"),
}


def _sanitize(value: Any) -> Any:
    """Strict-JSON copy: NaN/inf become None."""
    if isinstance(value, dict):
        return {str(k): _sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _file_key(path: Path) -> tuple[str, int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    if stat.st_size > MAX_REPORT_BYTES:
        logger.error("%s is larger than %d bytes; ignoring it", path.name, MAX_REPORT_BYTES)
        return None
    return str(path.resolve()), stat.st_mtime_ns, stat.st_size


@lru_cache(maxsize=8)
def _read_json(path: str, mtime_ns: int, size: int) -> Any:
    del mtime_ns, size
    try:
        return _sanitize(json.loads(Path(path).read_text(encoding="utf-8")))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.error("cannot read %s: %s", Path(path).name, exc)
        return None


def _number(text: str | None) -> float | None:
    if text is None or not text.strip():
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _flag(text: str | None) -> bool | None:
    if text is None:
        return None
    lowered = text.strip().lower()
    if lowered in ("true", "1", "yes"):
        return True
    if lowered in ("false", "0", "no"):
        return False
    return None


@lru_cache(maxsize=8)
def _read_headline_predictions(
    path: str, mtime_ns: int, size: int, spec: tuple[tuple[str, str], ...]
) -> list[dict[str, Any]] | None:
    """Rows of ``predictions.csv`` matching ``spec``; ``None`` if they cannot be identified."""
    del mtime_ns, size
    wanted = dict(spec)
    try:
        with Path(path).open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            columns = list(reader.fieldnames or [])
            resolved = {
                key: next((c for c in aliases if c in columns), None)
                for key, aliases in _COLUMN_ALIASES.items()
            }
            if any(c is None for c in resolved.values()):
                logger.warning("predictions.csv lacks battery/cycle/y/yhat columns")
                return None
            spec_columns = [c for c in _SPEC_COLUMNS if c in columns]
            if not spec_columns or any(c not in wanted for c in spec_columns):
                logger.warning("cannot match predictions.csv rows to the headline specification")
                return None
            bounds = [c for c in columns if _BOUND_COLUMN.match(c)]
            flags = [c for c in _FLAG_COLUMNS if c in columns]
            rows: list[dict[str, Any]] = []
            seen: set[tuple[str, float | None]] = set()
            for record in reader:
                if any(record.get(c) != wanted[c] for c in spec_columns):
                    continue
                if len(rows) >= MAX_PREDICTION_ROWS:
                    logger.warning("more than %d headline predictions; omitting them", len(rows))
                    return None
                battery = str(record[resolved["battery_id"] or ""])
                cycle = _number(record.get(resolved["cycle_index"] or ""))
                if (battery, cycle) in seen:
                    logger.warning("duplicate headline predictions for %s/%s", battery, cycle)
                    return None
                seen.add((battery, cycle))
                row: dict[str, Any] = {
                    "battery_id": battery,
                    "cycle_index": int(cycle) if cycle is not None else None,
                    "y": _number(record.get(resolved["y"] or "")),
                    "yhat": _number(record.get(resolved["yhat"] or "")),
                }
                row.update({c: _number(record.get(c)) for c in bounds})
                row.update({c: _flag(record.get(c)) for c in flags})
                rows.append(row)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        logger.error("cannot read predictions.csv: %s", exc)
        return None
    rows.sort(key=lambda r: (r["battery_id"], r["cycle_index"] is None, r["cycle_index"] or 0))
    return rows


_FileKey = tuple[str, int, int]


def _build_payload(
    summary_key: _FileKey | None, predictions_key: _FileKey | None, audit_key: _FileKey | None
) -> dict[str, Any]:
    """The response document; nested values are shared with the file caches (read-only)."""
    summary = _read_json(*summary_key) if summary_key is not None else None
    if not isinstance(summary, dict):
        raise ProblemError(
            404,
            "Benchmark results not available",
            "No benchmark summary has been generated yet; run `soh benchmark`.",
            type_="/problems/benchmark-unavailable",
        )
    payload: dict[str, Any] = dict(summary)
    headline = payload.get("headline")
    if "predictions" not in payload and isinstance(headline, dict):
        spec = tuple(
            (column, headline[column])
            for column in _SPEC_COLUMNS
            if isinstance(headline.get(column), str)
        )
        if predictions_key is not None and spec:
            rows = _read_headline_predictions(*predictions_key, spec)
            if rows:
                payload["predictions"] = rows
    if "audit" not in payload and audit_key is not None:
        audit = _read_json(*audit_key)
        if isinstance(audit, dict):
            payload["audit"] = audit
    return payload


@lru_cache(maxsize=4)
def _cached_document(
    summary_key: _FileKey | None, predictions_key: _FileKey | None, audit_key: _FileKey | None
) -> CachedJson:
    return to_cached_json(_build_payload(summary_key, predictions_key, audit_key))


def benchmark_document(benchmark_dir: Path, audit_dir: Path | None = None) -> CachedJson:
    """``GET /v1/benchmark`` serialized once per version of the three source files.

    Raises:
        ProblemError: 404 if no benchmark summary exists (or it is unreadable).
    """
    benchmark_dir = Path(benchmark_dir)
    return _cached_document(
        _file_key(benchmark_dir / SUMMARY_FILE),
        _file_key(benchmark_dir / PREDICTIONS_FILE),
        _file_key(Path(audit_dir) / AUDIT_FILE) if audit_dir is not None else None,
    )


def benchmark_payload(benchmark_dir: Path, audit_dir: Path | None = None) -> dict[str, Any]:
    """``GET /v1/benchmark``: the benchmark summary plus headline predictions and audit.

    Returns a fresh copy (decoded from the cached response body), so callers may
    modify it.

    Raises:
        ProblemError: 404 if no benchmark summary exists (or it is unreadable).
    """
    document: dict[str, Any] = json.loads(benchmark_document(benchmark_dir, audit_dir).body)
    return document


@router.get(
    "/benchmark",
    response_model=dict[str, Any],
    summary="Leave-one-battery-out benchmark results and the leakage audit",
    responses={**NOT_MODIFIED_RESPONSE, **problem_responses(404)},
)
def get_benchmark(
    request: Request, settings: Annotated[Settings, Depends(get_settings)]
) -> Response:
    """Headline numbers, leaderboard, per-battery and coverage tables, per-cycle
    predictions of the headline model, and the v1 leakage audit (when available).
    Cacheable: `ETag` plus `Cache-Control: public, max-age=300`; send `If-None-Match`
    to get `304` when nothing changed."""
    return cached_json_response(
        request, benchmark_document(settings.benchmark_dir, settings.audit_dir)
    )
