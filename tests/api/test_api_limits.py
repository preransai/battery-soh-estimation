"""Request-body limits (413), malformed Content-Length (400) and request-id plumbing."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from battery_soh.api.middleware import BodySizeLimitMiddleware, RequestContextMiddleware

LIMIT = 1_000_000


def test_declared_oversized_body_is_413(client: TestClient) -> None:
    response = client.post(
        "/v1/soh/estimate",
        content=b" " * (LIMIT + 1),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 413
    body = response.json()
    assert body["type"] == "/problems/payload-too-large"
    assert body["request_id"] == response.headers["x-request-id"]
    assert response.headers["content-security-policy"].startswith("default-src 'self'")


def test_chunked_oversized_body_is_413(client: TestClient) -> None:
    def chunks() -> Iterator[bytes]:
        for _ in range(11):
            yield b" " * 100_000

    response = client.post(
        "/v1/soh/estimate", content=chunks(), headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 413


def test_chunked_body_within_limit_is_processed(
    client: TestClient, sample_features: dict[str, float]
) -> None:
    payload = json.dumps({"features": sample_features}).encode()

    def chunks() -> Iterator[bytes]:
        for start in range(0, len(payload), 7):
            yield payload[start : start + 7]

    response = client.post(
        "/v1/soh/estimate", content=chunks(), headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 200, response.text


# ---------------------------------------------------------------------- ASGI-level checks
class Recorder:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []

    async def __call__(self, message: dict[str, Any]) -> None:
        self.messages.append(message)

    @property
    def status(self) -> int:
        return int(self.messages[0]["status"])

    @property
    def body(self) -> dict[str, Any]:
        raw = b"".join(m.get("body", b"") for m in self.messages[1:])
        return dict(json.loads(raw))


def _scope(headers: list[tuple[bytes, bytes]], method: str = "POST") -> dict[str, Any]:
    return {
        "type": "http",
        "method": method,
        "path": "/v1/soh/estimate",
        "headers": headers,
        "state": {"request_id": "rid-1"},
    }


async def _echo_app(scope: Any, receive: Any, send: Any) -> None:
    message = await receive()
    body = message.get("body", b"")
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": json.dumps({"n": len(body)}).encode()})


def _receiver(*chunks: bytes) -> Any:
    queue = [
        {"type": "http.request", "body": c, "more_body": i < len(chunks) - 1}
        for i, c in enumerate(chunks)
    ]

    async def receive() -> dict[str, Any]:
        return queue.pop(0) if queue else {"type": "http.disconnect"}

    return receive


@pytest.mark.parametrize("value", [b"abc", b"-5", b"1e3"])
def test_malformed_content_length_is_400(value: bytes) -> None:
    middleware = BodySizeLimitMiddleware(_echo_app, max_body_bytes=100)
    send = Recorder()
    asyncio.run(middleware(_scope([(b"content-length", value)]), _receiver(b"{}"), send))
    assert send.status == 400
    assert send.body["request_id"] == "rid-1"


def test_counted_bytes_win_over_a_lying_content_length() -> None:
    middleware = BodySizeLimitMiddleware(_echo_app, max_body_bytes=100)
    send = Recorder()
    receive = _receiver(b"x" * 60, b"x" * 60)
    asyncio.run(middleware(_scope([(b"content-length", b"10")]), receive, send))
    assert send.status == 413


def test_body_is_replayed_intact_to_the_app() -> None:
    middleware = BodySizeLimitMiddleware(_echo_app, max_body_bytes=100)
    send = Recorder()
    receive = _receiver(b"x" * 40, b"y" * 40, b"z" * 20)
    asyncio.run(middleware(_scope([(b"transfer-encoding", b"chunked")]), receive, send))
    assert send.status == 200
    assert send.body == {"n": 100}


def test_client_disconnect_while_reading_sends_nothing() -> None:
    middleware = BodySizeLimitMiddleware(_echo_app, max_body_bytes=100)
    send = Recorder()

    async def receive() -> dict[str, Any]:
        return {"type": "http.disconnect"}

    asyncio.run(middleware(_scope([]), receive, send))
    assert send.messages == []


def test_get_without_body_passes_through() -> None:
    middleware = BodySizeLimitMiddleware(_echo_app, max_body_bytes=100)
    send = Recorder()
    asyncio.run(middleware(_scope([], method="GET"), _receiver(b""), send))
    assert send.status == 200


def test_limit_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_body_bytes"):
        BodySizeLimitMiddleware(_echo_app, max_body_bytes=0)


def test_error_after_response_start_is_reraised() -> None:
    async def half_app(scope: Any, receive: Any, send: Any) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        raise RuntimeError("mid-stream failure")

    middleware = RequestContextMiddleware(half_app)
    send = Recorder()
    with pytest.raises(RuntimeError):
        asyncio.run(middleware(_scope([], method="GET"), _receiver(b""), send))
    headers = dict(send.messages[0]["headers"])
    assert headers[b"x-request-id"]


def test_lifespan_and_non_http_scopes_pass_through() -> None:
    seen: list[str] = []

    async def app(scope: Any, receive: Any, send: Any) -> None:
        seen.append(scope["type"])

    for middleware in (
        RequestContextMiddleware(app),
        BodySizeLimitMiddleware(app, max_body_bytes=10),
    ):
        asyncio.run(middleware({"type": "lifespan"}, _receiver(), Recorder()))
    assert seen == ["lifespan", "lifespan"]


def test_deeply_nested_json_is_a_client_error(client: TestClient) -> None:
    response = client.post(
        "/v1/soh/estimate",
        content=b'{"features": ' + b"[" * 200_000 + b"]" * 200_000 + b"}",
        headers={"Content-Type": "application/json"},
    )
    assert 400 <= response.status_code < 500
    assert response.headers["content-type"] == "application/problem+json"
