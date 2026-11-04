"""Pre-serialized, cacheable JSON responses for the read-only data endpoints.

``GET /v1/benchmark`` and the ``/v1/demo`` endpoints return documents that only
change when a file on disk changes. Their callers cache the serialized bytes
(:class:`CachedJson`, keyed by the source file's path, ``st_mtime_ns`` and size),
so a request costs a dictionary lookup instead of a deep copy plus JSON
encoding of up to half a megabyte.

Each document carries a weak ``ETag`` (a digest of the exact bytes; weak because
the gzip middleware may change the content coding) and
``Cache-Control: public, max-age=300``. A request whose ``If-None-Match`` lists
that tag gets ``304 Not Modified`` without a body.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from fastapi import Request, Response

__all__ = [
    "CACHE_CONTROL",
    "NOT_MODIFIED_RESPONSE",
    "CachedJson",
    "cached_json_response",
    "etag_matches",
    "to_cached_json",
]

#: Data that changes only with a redeployment: cacheable for five minutes, then revalidated.
CACHE_CONTROL = "public, max-age=300"

#: OpenAPI documentation of the conditional-request answer.
NOT_MODIFIED_RESPONSE: dict[int | str, dict[str, Any]] = {
    304: {"description": "Not modified: `If-None-Match` matched the current `ETag`."}
}


@dataclass(frozen=True)
class CachedJson:
    """A JSON document serialized once: the response body and its entity tag."""

    body: bytes
    etag: str


def to_cached_json(document: Any) -> CachedJson:
    """Serialize ``document`` exactly like FastAPI's ``JSONResponse`` (strict, compact UTF-8).

    Raises:
        ValueError: If the document contains NaN or infinity (not valid JSON).
    """
    body = json.dumps(
        document, ensure_ascii=False, allow_nan=False, indent=None, separators=(",", ":")
    ).encode("utf-8")
    return CachedJson(body=body, etag=f'W/"{hashlib.sha256(body).hexdigest()[:32]}"')


def etag_matches(if_none_match: str | None, etag: str) -> bool:
    """Weak comparison of ``etag`` with an ``If-None-Match`` header (RFC 9110 13.1.2)."""
    if not if_none_match:
        return False
    if if_none_match.strip() == "*":
        return True
    opaque = etag.removeprefix("W/")
    return any(
        candidate.strip().removeprefix("W/") == opaque for candidate in if_none_match.split(",")
    )


def cached_json_response(request: Request, cached: CachedJson) -> Response:
    """``200`` with the cached body, or ``304`` if the client already has this version."""
    headers = {"ETag": cached.etag, "Cache-Control": CACHE_CONTROL}
    if etag_matches(request.headers.get("if-none-match"), cached.etag):
        return Response(status_code=304, headers=headers)
    return Response(content=cached.body, media_type="application/json", headers=headers)
