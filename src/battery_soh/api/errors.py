"""RFC 7807 problem details for every error the service returns.

Every 4xx/5xx response is ``application/problem+json``::

    {"type": "/problems/model-not-found", "title": "Model not found", "status": 404,
     "detail": "No model bundle with id 'x'.", "request_id": "..."}

Validation failures (422) additionally carry ``errors``: FastAPI/Pydantic's
error list reduced to ``{type, loc, msg}`` (the offending input is never echoed
back, long strings are truncated and the list is capped). Unhandled exceptions
become a generic 500 whose body never contains a traceback, an exception
message or a file path; the details go to the server log under the same
request id.

Service code raises :class:`ProblemError`; it is framework-agnostic so the
static-site exporter can reuse the same service functions.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from battery_soh.errors import BatterySohError

__all__ = [
    "PROBLEM_MEDIA_TYPE",
    "ProblemError",
    "problem_body",
    "problem_response",
    "register_exception_handlers",
    "request_id_of",
    "sanitize_errors",
    "summarize_errors",
]

logger = logging.getLogger(__name__)

PROBLEM_MEDIA_TYPE = "application/problem+json"

#: Most validation errors reported back in one response.
MAX_REPORTED_ERRORS = 50
_MAX_TEXT = 300
_MAX_LOC_PART = 80


class ProblemError(BatterySohError):
    """An expected, client-visible failure rendered as problem details.

    Args:
        status: HTTP status code.
        title: Short, stable summary of the problem type.
        detail: Human-readable explanation of this occurrence (no secrets, no paths).
        type_: Problem type URI (``/problems/<slug>``; ``about:blank`` for plain HTTP errors).
    """

    def __init__(self, status: int, title: str, detail: str, *, type_: str = "about:blank"):
        super().__init__(detail)
        self.status = int(status)
        self.title = title
        self.detail = detail
        self.type = type_


def _status_title(status: int) -> str:
    try:
        return HTTPStatus(status).phrase
    except ValueError:
        return "Error"


def problem_body(
    status: int,
    title: str | None = None,
    detail: str = "",
    *,
    type_: str = "about:blank",
    request_id: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a problem-details document."""
    body: dict[str, Any] = {
        "type": type_,
        "title": title or _status_title(status),
        "status": int(status),
        "detail": detail,
        "request_id": request_id,
    }
    if extra:
        body.update(extra)
    return body


def problem_response(
    status: int,
    title: str | None = None,
    detail: str = "",
    *,
    type_: str = "about:blank",
    request_id: str | None = None,
    extra: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    """A :class:`JSONResponse` carrying problem details."""
    return JSONResponse(
        problem_body(status, title, detail, type_=type_, request_id=request_id, extra=extra),
        status_code=status,
        media_type=PROBLEM_MEDIA_TYPE,
        headers=dict(headers) if headers else None,
    )


def request_id_of(scope_or_request: Request | Mapping[str, Any]) -> str | None:
    """Request id stored by :class:`~battery_soh.api.middleware.RequestContextMiddleware`."""
    scope = scope_or_request.scope if isinstance(scope_or_request, Request) else scope_or_request
    state = scope.get("state")
    if isinstance(state, Mapping):
        value = state.get("request_id")
        return value if isinstance(value, str) else None
    return None


def _clip(text: str, limit: int = _MAX_TEXT) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def sanitize_errors(errors: Sequence[Any]) -> list[dict[str, Any]]:
    """Reduce Pydantic errors to ``{type, loc, msg}`` without echoing inputs."""
    cleaned: list[dict[str, Any]] = []
    for err in list(errors)[:MAX_REPORTED_ERRORS]:
        if not isinstance(err, Mapping):
            continue
        loc = [
            part if isinstance(part, int) else _clip(str(part), _MAX_LOC_PART)
            for part in err.get("loc", ())
        ]
        cleaned.append(
            {
                "type": _clip(str(err.get("type", "value_error")), 60),
                "loc": loc,
                "msg": _clip(str(err.get("msg", "Invalid value"))),
            }
        )
    return cleaned


def summarize_errors(errors: Sequence[Mapping[str, Any]], total: int) -> str:
    """One-line ``detail`` of a 422: the first three sanitized errors, then ``(+N more)``.

    The static web demo (``frontend/assets/js/static-backend.js``) reproduces this text.
    """
    parts = []
    for err in errors[:3]:
        loc = [str(p) for p in err["loc"]]
        if loc and loc[0] in ("body", "query", "path", "header"):
            loc = loc[1:]
        where = ".".join(loc)
        parts.append(f"{where}: {err['msg']}" if where else str(err["msg"]))
    text = "; ".join(parts)
    if total > 3:
        text += f" (+{total - 3} more)"
    return text or "The request is invalid."


async def _problem_error_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, ProblemError)
    return problem_response(
        exc.status, exc.title, exc.detail, type_=exc.type, request_id=request_id_of(request)
    )


async def _validation_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    raw = exc.errors()
    errors = sanitize_errors(raw)
    return problem_response(
        422,
        "Invalid input",
        summarize_errors(errors, len(raw)),
        type_="/problems/validation-error",
        request_id=request_id_of(request),
        extra={"errors": errors},
    )


async def _http_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)
    status = exc.status_code
    # Framework-generated details ("Not Found", "Method Not Allowed") are static
    # strings; anything else is replaced by the generic phrase to avoid leaks.
    detail = exc.detail if isinstance(exc.detail, str) and len(exc.detail) <= 200 else ""
    if status >= 500:
        detail = ""
    headers = {k: v for k, v in (exc.headers or {}).items() if k.lower() == "allow"}
    return problem_response(
        status,
        _status_title(status),
        detail,
        request_id=request_id_of(request),
        headers=headers or None,
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Install the problem-details handlers on ``app``."""
    app.add_exception_handler(ProblemError, _problem_error_handler)
    app.add_exception_handler(RequestValidationError, _validation_handler)
    app.add_exception_handler(StarletteHTTPException, _http_exception_handler)
