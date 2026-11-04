"""ASGI middleware: request id, access log, metrics, security headers, body limit.

Both classes are plain ASGI middleware (no ``BaseHTTPMiddleware``), so they add
no task/stream indirection and work for every response, including errors
produced by inner layers and the framework itself.

:class:`RequestContextMiddleware` (outermost) gives every request an id (a sane
incoming ``X-Request-ID`` is kept, otherwise a UUID4 is generated), echoes it in
the response, stamps the security headers on *every* response, records RED
metrics per route template, writes one structured access-log line, and turns
an unhandled exception into a generic RFC 7807 500 (the traceback goes to the
log only).

:class:`BodySizeLimitMiddleware` rejects bodies above the configured size with
413 before any parsing: early from ``Content-Length`` and, for chunked or
mis-declared bodies, by counting the bytes actually received (the body is
buffered up to the limit and replayed to the application).
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from collections.abc import Mapping
from contextvars import ContextVar
from typing import Any

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from battery_soh.api.errors import problem_response, request_id_of
from battery_soh.api.metrics import ApiMetrics

__all__ = [
    "CONTENT_SECURITY_POLICY",
    "REQUEST_ID_HEADER",
    "SECURITY_HEADERS",
    "BodySizeLimitMiddleware",
    "RequestContextMiddleware",
    "install_log_record_factory",
]

access_logger = logging.getLogger("battery_soh.api.access")
logger = logging.getLogger(__name__)

#: Strict policy for the web UI and API: same-origin only, no inline code, no framing.
CONTENT_SECURITY_POLICY = (
    "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; "
    "connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
)

#: Headers set on every response (the CSP may be overridden per path, see ``csp_overrides``).
SECURITY_HEADERS: Mapping[str, str] = {
    "Content-Security-Policy": CONTENT_SECURITY_POLICY,
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
    "Permissions-Policy": (
        "accelerometer=(), camera=(), geolocation=(), gyroscope=(), magnetometer=(), "
        "microphone=(), payment=(), usb=(), interest-cohort=()"
    ),
    "Cross-Origin-Opener-Policy": "same-origin",
    # Other sites cannot embed responses as no-cors subresources (<img>, <script>, ...);
    # CORS requests, when SOH_CORS_ORIGINS allows them, are not affected.
    "Cross-Origin-Resource-Policy": "same-origin",
}

REQUEST_ID_HEADER = "X-Request-ID"
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_PROBE_PATHS = frozenset({"/livez", "/readyz", "/metrics"})
_KNOWN_METHODS = frozenset({"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"})
_BODY_METHODS = frozenset({"POST", "PUT", "PATCH"})

_request_id_var: ContextVar[str | None] = ContextVar("soh_request_id", default=None)
_factory_installed = False


def install_log_record_factory() -> None:
    """Attach the current request id to every log record created during a request.

    Idempotent. The JSON log formatter (:mod:`battery_soh.logs`) emits the
    ``request_id`` attribute, so application logs correlate with access logs.
    """
    global _factory_installed
    if _factory_installed:
        return
    previous = logging.getLogRecordFactory()

    def factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
        record = previous(*args, **kwargs)
        request_id = _request_id_var.get()
        if request_id is not None:
            record.request_id = request_id
        return record

    logging.setLogRecordFactory(factory)
    _factory_installed = True


def _safe_for_log(text: str, limit: int = 200) -> str:
    """Printable, length-limited text (no log injection through the URL)."""
    cleaned = "".join(ch if ch.isprintable() else "?" for ch in text)
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 3] + "..."


def _route_label(scope: Scope, status: int) -> str:
    route = scope.get("route")
    template = getattr(route, "path", None)
    if isinstance(template, str) and template:
        return template
    return "<static>" if status < 400 else "<unmatched>"


class RequestContextMiddleware:
    """Request id, security headers, access log, RED metrics and the 500 safety net.

    Args:
        app: Inner ASGI application.
        metrics: Metric families to update (``None`` disables metrics).
        csp_overrides: Exact paths that need a different Content-Security-Policy
            (the interactive docs, which load their assets from a CDN).
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        metrics: ApiMetrics | None = None,
        csp_overrides: Mapping[str, str] | None = None,
    ) -> None:
        self.app = app
        self.metrics = metrics
        self.csp_overrides = dict(csp_overrides or {})

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming = Headers(scope=scope).get(REQUEST_ID_HEADER)
        request_id = incoming if incoming and _REQUEST_ID_RE.match(incoming) else str(uuid.uuid4())
        scope.setdefault("state", {})["request_id"] = request_id
        token = _request_id_var.set(request_id)
        path: str = scope.get("path", "")
        raw_method = str(scope.get("method", "GET"))
        # Clients choose the method token: bound it before it becomes a metric label.
        method = raw_method if raw_method in _KNOWN_METHODS else "OTHER"
        status = 500
        started = False
        start = time.perf_counter()
        if self.metrics is not None:
            self.metrics.in_flight.inc()

        async def send_wrapper(message: Message) -> None:
            nonlocal status, started
            if message["type"] == "http.response.start":
                started = True
                status = int(message["status"])
                headers = MutableHeaders(scope=message)
                headers[REQUEST_ID_HEADER] = request_id
                for name, value in SECURITY_HEADERS.items():
                    headers[name] = value
                if path in self.csp_overrides:
                    headers["Content-Security-Policy"] = self.csp_overrides[path]
                if "cache-control" not in headers and (
                    scope.get("route") is not None or status >= 400
                ):
                    headers["Cache-Control"] = "no-store"
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            logger.exception("unhandled error while serving %s %s", method, _safe_for_log(path))
            if started:
                raise
            response = problem_response(
                500,
                "Internal Server Error",
                "An unexpected error occurred. Quote the request id when reporting it.",
                type_="/problems/internal-error",
                request_id=request_id,
            )
            await response(scope, receive, send_wrapper)
        finally:
            elapsed = time.perf_counter() - start
            route = _route_label(scope, status)
            if self.metrics is not None:
                self.metrics.in_flight.dec()
                self.metrics.observe_request(method, route, status, elapsed)
            duration_ms = round(elapsed * 1000.0, 2)
            access_logger.log(
                logging.DEBUG if path in _PROBE_PATHS and status < 500 else logging.INFO,
                "%s %s %d %.1fms rid=%s",
                method,
                _safe_for_log(path),
                status,
                duration_ms,
                request_id,
                extra={
                    "method": method,
                    "path": _safe_for_log(path),
                    "status": status,
                    "duration_ms": duration_ms,
                },
            )
            _request_id_var.reset(token)


class BodySizeLimitMiddleware:
    """Reject request bodies larger than ``max_body_bytes`` with 413 problem details.

    Args:
        app: Inner ASGI application.
        max_body_bytes: Largest accepted body in bytes.
    """

    def __init__(self, app: ASGIApp, *, max_body_bytes: int) -> None:
        if max_body_bytes < 1:
            raise ValueError("max_body_bytes must be >= 1")
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def _reject(self, scope: Scope, receive: Receive, send: Send, status: int) -> None:
        if status == 413:
            response = problem_response(
                413,
                "Request body too large",
                f"Request bodies are limited to {self.max_body_bytes} bytes.",
                type_="/problems/payload-too-large",
                request_id=request_id_of(scope),
                headers={"Connection": "close"},
            )
        else:
            response = problem_response(
                400,
                "Bad Request",
                "Invalid Content-Length header.",
                request_id=request_id_of(scope),
                headers={"Connection": "close"},
            )
        await response(scope, receive, send)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        declared = headers.get("content-length")
        if declared is not None:
            try:
                size = int(declared)
                if size < 0:
                    raise ValueError(declared)
            except ValueError:
                await self._reject(scope, receive, send, 400)
                return
            if size > self.max_body_bytes:
                await self._reject(scope, receive, send, 413)
                return
        has_body = (
            scope.get("method") in _BODY_METHODS
            or (declared is not None and declared != "0")
            or "transfer-encoding" in headers
        )
        if not has_body:
            await self.app(scope, receive, send)
            return

        chunks: list[bytes] = []
        received = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return  # the client went away; nothing to answer
            chunk: bytes = message.get("body", b"")
            received += len(chunk)
            if received > self.max_body_bytes:
                await self._reject(scope, receive, send, 413)
                return
            chunks.append(chunk)
            if not message.get("more_body", False):
                break
        body = b"".join(chunks)
        replayed = False

        async def replay() -> Message:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, replay, send)
