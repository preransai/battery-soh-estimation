"""HTTP contract tests: health probes, headers, models, errors, metrics, CORS, static UI."""

from __future__ import annotations

import base64
import hashlib
import logging
import re
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from battery_soh.api import inference
from battery_soh.api.app import create_app
from battery_soh.api.middleware import CONTENT_SECURITY_POLICY
from battery_soh.api.settings import Settings

PROBLEM_KEYS = {"type", "title", "status", "detail", "request_id"}
STRICT_CSP = (
    "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; "
    "connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
)


def assert_security_headers(response: object, csp: str = STRICT_CSP) -> None:
    headers = response.headers  # type: ignore[attr-defined]
    assert headers["content-security-policy"] == csp
    assert headers["x-content-type-options"] == "nosniff"
    assert headers["referrer-policy"] == "no-referrer"
    assert headers["x-frame-options"] == "DENY"
    assert "camera=()" in headers["permissions-policy"]
    assert headers["cross-origin-opener-policy"] == "same-origin"
    assert headers["cross-origin-resource-policy"] == "same-origin"
    assert headers["x-request-id"]


def assert_problem(response: object, status: int, type_: str | None = None) -> dict:
    assert response.status_code == status  # type: ignore[attr-defined]
    assert response.headers["content-type"] == "application/problem+json"  # type: ignore[attr-defined]
    body = response.json()  # type: ignore[attr-defined]
    assert set(body) >= PROBLEM_KEYS
    assert body["status"] == status
    assert body["request_id"] == response.headers["x-request-id"]  # type: ignore[attr-defined]
    if type_ is not None:
        assert body["type"] == type_
    return body


# ---------------------------------------------------------------------- health
def test_csp_constant_is_the_documented_policy() -> None:
    assert CONTENT_SECURITY_POLICY == STRICT_CSP


def test_livez(client: TestClient) -> None:
    response = client.get("/livez")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert_security_headers(response)


def test_readyz_lists_models_best_first(client: TestClient) -> None:
    response = client.get("/readyz")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "models": [
            "combined-relative-ridge",
            "combined-absolute-ridge",
            "discharge-absolute-ridge",
            "charge-absolute-ridge",
        ],
    }
    assert_security_headers(response)


def test_readyz_is_503_without_models_but_livez_is_200(tmp_path: Path, settings: Settings) -> None:
    empty = settings.model_copy(update={"models_dir": tmp_path})
    with TestClient(create_app(empty)) as client:
        ready = client.get("/readyz")
        body = assert_problem(ready, 503, "/problems/not-ready")
        assert str(tmp_path) not in ready.text  # no server paths in responses
        assert body["detail"]
        assert_security_headers(ready)
        assert client.get("/livez").status_code == 200
        estimate = client.post("/v1/soh/estimate", json={"features": {"dis_r0_ohm": 0.1}})
        assert_problem(estimate, 503, "/problems/no-models")


def test_readyz_is_503_before_the_lifespan_runs(settings: Settings) -> None:
    client = TestClient(create_app(settings))  # no `with`: startup never happens
    assert client.get("/readyz").status_code == 503


# ---------------------------------------------------------------------- headers
@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/livez"),
        ("GET", "/readyz"),
        ("GET", "/v1/models"),
        ("GET", "/v1/benchmark"),
        ("GET", "/openapi.json"),
        ("GET", "/no/such/route"),
        ("GET", "/"),
        ("GET", "/assets/config.js"),
        ("POST", "/v1/soh/estimate"),
    ],
)
def test_security_headers_on_every_response(client: TestClient, method: str, path: str) -> None:
    response = client.request(method, path, json={} if method == "POST" else None)
    assert_security_headers(response)


def test_request_id_is_echoed_when_sane(client: TestClient) -> None:
    response = client.get("/v1/models", headers={"X-Request-ID": "trace-abc.123:4"})
    assert response.headers["x-request-id"] == "trace-abc.123:4"


@pytest.mark.parametrize("bad", ["has space", "x" * 129, "semi;colon", "\u00e9t\u00e9"])
def test_insane_request_id_is_replaced_by_a_uuid(client: TestClient, bad: str) -> None:
    response = client.get("/livez", headers={"X-Request-ID": bad.encode("utf-8")})  # type: ignore[dict-item]
    generated = response.headers["x-request-id"]
    assert generated != bad
    assert uuid.UUID(generated).version == 4


def test_request_id_in_problem_body_matches_header(client: TestClient) -> None:
    response = client.get("/v1/models/unknown-model", headers={"X-Request-ID": "abc-123"})
    body = assert_problem(response, 404, "/problems/model-not-found")
    assert body["request_id"] == "abc-123"


def test_api_responses_are_not_cacheable(client: TestClient) -> None:
    assert client.get("/v1/models").headers["cache-control"] == "no-store"
    assert client.get("/readyz").headers["cache-control"] == "no-store"


def test_docs_get_a_path_scoped_csp_with_the_inline_script_hash(client: TestClient) -> None:
    response = client.get("/docs")
    assert response.status_code == 200
    csp = response.headers["content-security-policy"]
    assert csp != STRICT_CSP
    inline = re.findall(r"<script>(.*?)</script>", response.text, re.DOTALL)
    assert len(inline) == 1
    digest = base64.b64encode(hashlib.sha256(inline[0].encode()).digest()).decode()
    assert f"'sha256-{digest}'" in csp
    assert "'unsafe-inline'" not in csp.split("script-src")[1].split(";")[0]
    assert "frame-ancestors 'none'" in csp
    redoc = client.get("/redoc")
    assert redoc.status_code == 200
    assert "worker-src 'self' blob:" in redoc.headers["content-security-policy"]
    assert "fonts.googleapis" not in redoc.text
    # Only the docs pages are relaxed.
    assert client.get("/openapi.json").headers["content-security-policy"] == STRICT_CSP


@pytest.mark.parametrize(
    ("path", "package"),
    [
        ("/docs", "https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.17.14/"),
        ("/redoc", "https://cdn.jsdelivr.net/npm/redoc@2.1.5/"),
    ],
)
def test_docs_csp_is_scoped_to_the_pinned_package_and_tags_carry_sri(
    client: TestClient, path: str, package: str
) -> None:
    response = client.get(path)
    csp = response.headers["content-security-policy"]
    directives = dict(d.strip().split(" ", 1) for d in csp.split(";") if d.strip())
    for name in ("script-src", "style-src"):
        sources = directives[name].split()
        assert package in sources
        # Never the whole CDN, and never another package.
        assert not [s for s in sources if s.startswith("https://") and s != package]
    tags = re.findall(r"<(?:script|link)\b[^>]*cdn\.jsdelivr\.net[^>]*>", response.text)
    assert tags
    for tag in tags:
        url = re.search(r'(?:src|href)="([^"]+)"', tag)
        assert url is not None
        assert url.group(1).startswith(package)
        assert re.search(r'integrity="sha384-[A-Za-z0-9+/]{64}"', tag), tag
        assert 'crossorigin="anonymous"' in tag


def test_docs_can_be_disabled(settings: Settings) -> None:
    with TestClient(create_app(settings.model_copy(update={"enable_docs": False}))) as client:
        assert client.get("/docs").status_code == 404
        assert client.get("/openapi.json").status_code == 404


def test_openapi_documents_the_contract(client: TestClient) -> None:
    spec = client.get("/openapi.json").json()
    paths = set(spec["paths"])
    assert {
        "/livez",
        "/readyz",
        "/v1/models",
        "/v1/models/{model_id}",
        "/v1/soh/estimate",
        "/v1/soh/estimate-from-cycle",
        "/v1/soh/batch",
        "/v1/demo/batteries",
        "/v1/demo/batteries/{battery_id}/cycles/{cycle_index}",
        "/v1/benchmark",
    } <= paths
    features = spec["components"]["schemas"]["FeatureValues"]
    assert features["additionalProperties"] is False
    r0 = features["properties"]["dis_r0_ohm"]["anyOf"][0]
    assert (r0["minimum"], r0["maximum"]) == (0.0, 2.0)
    assert "leg_capacity_ah" not in features["properties"]


# ---------------------------------------------------------------------- static UI
def test_web_ui_is_served_with_correct_media_types(client: TestClient) -> None:
    index = client.get("/")
    assert index.status_code == 200
    assert index.headers["content-type"].startswith("text/html")
    script = client.get("/assets/js/app.js")
    assert script.status_code == 200
    assert script.headers["content-type"].startswith("text/javascript")


def test_hidden_files_are_never_served(client: TestClient) -> None:
    assert_problem(client.get("/.secret"), 404)


def test_unknown_post_is_404_not_405(client: TestClient) -> None:
    assert_problem(client.post("/v1/nothing", json={}), 404)


def test_method_not_allowed_is_a_problem_with_allow_header(client: TestClient) -> None:
    response = client.get("/v1/soh/estimate")
    assert_problem(response, 405)
    assert "POST" in response.headers["allow"]


# ---------------------------------------------------------------------- models
def test_models_list_shape_and_order(client: TestClient) -> None:
    response = client.get("/v1/models")
    assert response.status_code == 200
    models = response.json()
    assert [m["model_id"] for m in models] == [
        "combined-relative-ridge",
        "combined-absolute-ridge",
        "discharge-absolute-ridge",
        "charge-absolute-ridge",
    ]
    for model in models:
        assert set(model) == {"model_id", "feature_set", "mode", "features", "metrics"}
        assert set(model["metrics"]) == {"lobo_mae", "lobo_rmse", "coverage_90"}
        assert model["metrics"]["coverage_90"] == 0.9
    combined = models[1]
    assert combined["mode"] == "absolute"
    assert combined["feature_set"] == "combined"
    assert len(combined["features"]) == 8
    assert combined["metrics"]["lobo_mae"] == 1.4


def test_model_card_omits_the_precision_matrix(client: TestClient) -> None:
    card = client.get("/v1/models/combined-absolute-ridge").json()
    assert card["model_id"] == "combined-absolute-ridge"
    assert "precision_matrix" not in card["domain"]
    assert len(card["estimator"]["coef"]) == len(card["features"]) == 8
    assert card["conformal"]["quantiles"] == {"0.8": 1.5, "0.9": 2.0, "0.95": 2.6}


def test_unknown_model_is_404(client: TestClient) -> None:
    body = assert_problem(client.get("/v1/models/nope"), 404, "/problems/model-not-found")
    assert "nope" in body["detail"]


def test_malformed_model_id_is_422(client: TestClient) -> None:
    body = assert_problem(client.get("/v1/models/bad%20id%21"), 422, "/problems/validation-error")
    assert body["errors"][0]["loc"] == ["path", "model_id"]


# ---------------------------------------------------------------------- 500 safety net
def test_unhandled_errors_become_generic_problems(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, sample_features: dict[str, float]
) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError(r"secret detail C:\internal\path.py")

    monkeypatch.setattr(inference, "estimate", boom)
    response = client.post("/v1/soh/estimate", json={"features": sample_features})
    body = assert_problem(response, 500, "/problems/internal-error")
    assert "secret" not in response.text
    assert "Traceback" not in response.text
    assert body["detail"] == "An unexpected error occurred. Quote the request id when reporting it."
    assert_security_headers(response)


# ---------------------------------------------------------------------- metrics
def test_metrics_use_route_templates(
    fresh_client: TestClient, sample_features: dict[str, float]
) -> None:
    client = fresh_client
    client.get("/v1/models/combined-absolute-ridge")
    client.get("/v1/models/nope")
    client.get("/totally/unknown/path")
    client.post("/v1/soh/estimate", json={"features": sample_features})
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    text = response.text
    assert (
        'soh_http_requests_total{method="GET",route="/v1/models/{model_id}",status="200"} 1.0'
        in text
    )
    assert 'route="/v1/models/{model_id}",status="404"' in text
    assert 'route="<unmatched>",status="404"' in text
    assert "/totally/unknown/path" not in text
    assert 'soh_predictions_total{in_domain="true",model_id="combined-absolute-ridge"} 1.0' in text
    assert 'soh_inference_duration_seconds_count{model_id="combined-absolute-ridge"} 1.0' in text
    assert "soh_models_loaded 4.0" in text
    assert "soh_http_request_duration_seconds_bucket" in text


def test_metrics_can_be_disabled(settings: Settings) -> None:
    with TestClient(create_app(settings.model_copy(update={"enable_metrics": False}))) as client:
        assert client.get("/metrics").status_code == 404
        assert client.get("/v1/models").status_code == 200


# ---------------------------------------------------------------------- CORS
def test_no_cors_headers_by_default(client: TestClient) -> None:
    response = client.get("/v1/models", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in response.headers


def test_cors_allows_only_configured_origins(settings: Settings) -> None:
    configured = settings.model_copy(update={"cors_origins": ["https://ui.example"]})
    with TestClient(create_app(configured)) as client:
        preflight = client.options(
            "/v1/soh/estimate",
            headers={
                "Origin": "https://ui.example",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        assert preflight.status_code == 200
        assert preflight.headers["access-control-allow-origin"] == "https://ui.example"
        assert_security_headers(preflight)
        simple = client.get("/v1/models", headers={"Origin": "https://ui.example"})
        assert simple.headers["access-control-allow-origin"] == "https://ui.example"
        assert "x-request-id" in simple.headers["access-control-expose-headers"].lower()
        other = client.get("/v1/models", headers={"Origin": "https://evil.example"})
        assert "access-control-allow-origin" not in other.headers


def test_large_responses_are_gzipped(client: TestClient) -> None:
    response = client.get("/v1/demo/batteries/B0005/cycles/0", headers={"Accept-Encoding": "gzip"})
    assert response.status_code == 200
    assert response.headers["content-encoding"] == "gzip"
    assert_security_headers(response)


def test_unknown_methods_share_one_metric_label(fresh_client: TestClient) -> None:
    for verb in ("FOO", "BAR", "BAZ"):
        fresh_client.request(verb, "/v1/models")
    text = fresh_client.get("/metrics").text
    assert 'method="OTHER"' in text
    assert 'method="FOO"' not in text


def test_access_log_carries_request_id_and_fields(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="battery_soh.api.access"):
        client.get("/v1/models", headers={"X-Request-ID": "log-test-1"})
    records = [r for r in caplog.records if r.name == "battery_soh.api.access"]
    assert records, "no access log line"
    record = records[-1]
    assert record.request_id == "log-test-1"  # type: ignore[attr-defined]
    assert (record.method, record.path, record.status) == ("GET", "/v1/models", 200)  # type: ignore[attr-defined]
    assert record.duration_ms >= 0  # type: ignore[attr-defined]


def test_log_injection_through_the_path_is_neutralised(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="battery_soh.api.access"):
        client.get("/v1/models%0Aforged-line")
    record = [r for r in caplog.records if r.name == "battery_soh.api.access"][-1]
    assert "\n" not in record.getMessage()
