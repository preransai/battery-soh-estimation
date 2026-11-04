"""POST /v1/soh/estimate and /v1/soh/batch: model choice, exactness, and input validation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from battery_soh.api import inference
from battery_soh.api.app import create_app
from battery_soh.api.errors import ProblemError
from battery_soh.api.registry import ModelRegistry
from battery_soh.api.settings import Settings
from battery_soh.features.registry import FEATURE_SETS, FEATURES
from battery_soh.models.linear_bundle import LinearBundle

ESTIMATE_KEYS = {
    "model_id",
    "soh_pct",
    "interval",
    "base_value",
    "contributions",
    "domain",
    "warnings",
}


def post(client: TestClient, body: Any) -> Any:
    return client.post("/v1/soh/estimate", json=body)


def assert_estimate_shape(body: dict[str, Any]) -> None:
    assert set(body) == ESTIMATE_KEYS
    assert set(body["interval"]) == {"lower", "upper", "confidence", "method"}
    assert set(body["domain"]) == {
        "in_domain",
        "out_of_range",
        "mahalanobis",
        "threshold",
        "missing",
    }
    for item in body["contributions"]:
        assert set(item) == {"feature", "value", "contribution"}
    assert body["interval"]["lower"] <= body["soh_pct"] <= body["interval"]["upper"]


def test_estimate_happy_path_is_exact_and_accurate(
    client: TestClient, sample_features: dict[str, float], models_dir: Path
) -> None:
    response = post(client, {"features": sample_features})
    assert response.status_code == 200, response.text
    body = response.json()
    assert_estimate_shape(body)
    # Best complete *absolute* model: the relative one needs a reference.
    assert body["model_id"] == "combined-absolute-ridge"
    bundle = LinearBundle.from_json(models_dir / "combined-absolute-ridge.json")
    assert body["soh_pct"] == pytest.approx(bundle.predict(sample_features), abs=1e-12)
    assert abs(body["soh_pct"] - 85.0) < 3.0  # truth of the sample cycle
    total = body["base_value"] + sum(c["contribution"] for c in body["contributions"])
    assert total == pytest.approx(body["soh_pct"], abs=1e-9)
    assert body["interval"] == {
        "lower": pytest.approx(body["soh_pct"] - 2.0),
        "upper": pytest.approx(body["soh_pct"] + 2.0),
        "confidence": 0.9,
        "method": bundle.conformal_method,
    }
    assert body["domain"]["in_domain"] is True
    assert body["warnings"] == []


def test_confidence_levels_widen_the_interval(
    client: TestClient, sample_features: dict[str, float]
) -> None:
    widths = []
    for level in (0.8, 0.9, 0.95):
        body = post(client, {"features": sample_features, "confidence": level}).json()
        assert body["interval"]["confidence"] == level
        widths.append(body["interval"]["upper"] - body["interval"]["lower"])
    assert widths == pytest.approx([3.0, 4.0, 5.2])


@pytest.mark.parametrize(
    ("feature_set", "expected"),
    [("discharge", "discharge-absolute-ridge"), ("charge", "charge-absolute-ridge")],
)
def test_selection_uses_the_best_complete_model(
    client: TestClient, sample_features: dict[str, float], feature_set: str, expected: str
) -> None:
    features = {f: sample_features[f] for f in FEATURE_SETS[feature_set]}
    body = post(client, {"features": features}).json()
    assert body["model_id"] == expected
    assert body["domain"]["missing"] == []


def test_partial_inputs_fall_back_to_best_coverage_and_warn(
    client: TestClient, sample_features: dict[str, float]
) -> None:
    features = {f: sample_features[f] for f in ("dis_q_380_350_ah", "dis_r0_ohm")}
    features["chg_cv_time_s"] = sample_features["chg_cv_time_s"]
    body = post(client, {"features": features}).json()
    # discharge model covers 2/3, combined 3/8 -> combined has the most inputs
    assert body["model_id"] == "combined-absolute-ridge"
    assert len(body["domain"]["missing"]) == 5
    assert body["warnings"][0].startswith("Imputed the training median for 5 missing feature(s)")


def test_null_feature_counts_as_missing(
    client: TestClient, sample_features: dict[str, float]
) -> None:
    features: dict[str, float | None] = dict(sample_features)
    features["dis_r0_ohm"] = None
    body = post(client, {"features": features, "model_id": "combined-absolute-ridge"}).json()
    assert body["domain"]["missing"] == ["dis_r0_ohm"]


def test_explicit_model_id(client: TestClient, sample_features: dict[str, float]) -> None:
    body = post(
        client, {"features": sample_features, "model_id": "discharge-absolute-ridge"}
    ).json()
    assert body["model_id"] == "discharge-absolute-ridge"
    assert [c["feature"] for c in body["contributions"]] == list(FEATURE_SETS["discharge"])


def test_unknown_model_id_is_404(client: TestClient, sample_features: dict[str, float]) -> None:
    response = post(client, {"features": sample_features, "model_id": "no-such-model"})
    assert response.status_code == 404
    assert response.json()["type"] == "/problems/model-not-found"


def test_model_without_any_provided_input_is_422(client: TestClient) -> None:
    response = post(client, {"features": {"dis_r0_ohm": 0.08}, "model_id": "charge-absolute-ridge"})
    assert response.status_code == 422
    assert response.json()["type"] == "/problems/no-usable-features"


def test_features_no_model_uses_are_422(client: TestClient) -> None:
    response = post(client, {"features": {"eis_re_ohm": 0.05, "ctx_ambient_c": 24.0}})
    assert response.status_code == 422
    body = response.json()
    assert body["type"] == "/problems/no-usable-features"
    assert "dis_r0_ohm" in body["detail"]


# ---------------------------------------------------------------------- relative mode
def _reference(training_frame: Any, battery_id: str = "B0005") -> dict[str, float]:
    first = training_frame[training_frame.battery_id == battery_id].sort_values("cycle_index")
    return {f: float(first[f].head(2).mean()) for f in FEATURE_SETS["combined"]}


def test_reference_enables_the_better_relative_model(
    client: TestClient, sample_features: dict[str, float], training_frame: Any, models_dir: Path
) -> None:
    reference = _reference(training_frame)
    body = post(client, {"features": sample_features, "reference": reference}).json()
    assert body["model_id"] == "combined-relative-ridge"
    bundle = LinearBundle.from_json(models_dir / "combined-relative-ridge.json")
    ratios = {f: sample_features[f] / reference[f] for f in sample_features}
    assert body["soh_pct"] == pytest.approx(bundle.predict(ratios), abs=1e-12)
    assert abs(body["soh_pct"] - 85.0) < 3.0
    values = {c["feature"]: c["value"] for c in body["contributions"]}
    assert values["dis_q_380_350_ah"] == pytest.approx(ratios["dis_q_380_350_ah"])


def test_relative_model_without_reference_is_422(
    client: TestClient, sample_features: dict[str, float]
) -> None:
    response = post(client, {"features": sample_features, "model_id": "combined-relative-ridge"})
    assert response.status_code == 422
    body = response.json()
    assert body["type"] == "/problems/reference-required"
    assert "dis_r0_ohm" in body["detail"]


def test_zero_reference_is_422(
    client: TestClient, sample_features: dict[str, float], training_frame: Any
) -> None:
    reference = _reference(training_frame)
    reference["dis_r0_ohm"] = 0.0
    response = post(
        client,
        {
            "features": sample_features,
            "reference": reference,
            "model_id": "combined-relative-ridge",
        },
    )
    assert response.status_code == 422
    assert response.json()["type"] == "/problems/reference-required"


def test_tiny_reference_giving_infinite_ratio_is_422(
    client: TestClient, sample_features: dict[str, float], training_frame: Any
) -> None:
    reference = _reference(training_frame)
    reference["chg_cv_time_s"] = 5e-324  # 600 / 5e-324 overflows to inf
    response = post(
        client,
        {
            "features": sample_features,
            "reference": reference,
            "model_id": "combined-relative-ridge",
        },
    )
    assert response.status_code == 422


def test_incomplete_reference_keeps_the_absolute_model(
    client: TestClient, sample_features: dict[str, float], training_frame: Any
) -> None:
    reference = _reference(training_frame)
    del reference["dis_r0_ohm"]
    body = post(client, {"features": sample_features, "reference": reference}).json()
    assert body["model_id"] == "combined-absolute-ridge"


def test_unbounded_conformal_level_is_explicit(
    client: TestClient, sample_features: dict[str, float], training_frame: Any
) -> None:
    body = post(
        client,
        {
            "features": sample_features,
            "reference": _reference(training_frame),
            "confidence": 0.95,
        },
    ).json()
    assert body["model_id"] == "combined-relative-ridge"
    assert (body["interval"]["lower"], body["interval"]["upper"]) == (0.0, 120.0)
    assert any("No finite conformal interval" in w for w in body["warnings"])


def test_out_of_domain_input_is_flagged(
    client: TestClient, sample_features: dict[str, float]
) -> None:
    features = dict(sample_features)
    features["dis_r0_ohm"] = 1.5  # plausible physically, far outside the training cells
    body = post(client, {"features": features}).json()
    assert body["domain"]["in_domain"] is False
    assert "dis_r0_ohm" in body["domain"]["out_of_range"]
    assert any("outside the training domain" in w for w in body["warnings"])


def test_clipped_prediction_is_reported(
    client: TestClient, sample_features: dict[str, float]
) -> None:
    features = {f: sample_features[f] for f in FEATURE_SETS["discharge"]}
    features["dis_q_380_350_ah"] = 5.0  # ~8x a fresh synthetic cell: far above 120 % SOH
    body = post(client, {"features": features}).json()
    assert body["soh_pct"] == 120.0
    assert "Prediction clipped to the physical range [0, 120]." in body["warnings"]
    total = body["base_value"] + sum(c["contribution"] for c in body["contributions"])
    assert total > 120.0  # contributions explain the raw (unclipped) model output
    assert body["interval"]["upper"] == 120.0


# ---------------------------------------------------------------------- validation (422)
def _validation(response: Any) -> list[dict[str, Any]]:
    assert response.status_code == 422, response.text
    body = response.json()
    assert response.headers["content-type"] == "application/problem+json"
    assert body["type"] == "/problems/validation-error"
    assert body["detail"]
    for error in body["errors"]:
        assert set(error) == {"type", "loc", "msg"}  # never echoes the input back
    return list(body["errors"])


@pytest.mark.parametrize(
    "raw",
    [
        b'{"features": {"dis_r0_ohm": NaN}}',
        b'{"features": {"dis_r0_ohm": Infinity}}',
        b'{"features": {"dis_r0_ohm": -Infinity}}',
    ],
)
def test_non_finite_numbers_are_422(client: TestClient, raw: bytes) -> None:
    response = client.post(
        "/v1/soh/estimate", content=raw, headers={"Content-Type": "application/json"}
    )
    errors = _validation(response)
    assert errors[0]["loc"] == ["body", "features", "dis_r0_ohm"]


@pytest.mark.parametrize("name", [n for n, s in FEATURES.items() if s.leakage_risk == "none"])
def test_out_of_plausible_range_is_422_for_every_feature(client: TestClient, name: str) -> None:
    low, high = FEATURES[name].plausible_range
    for value in (high + abs(high) * 0.01 + 1e-6, low - abs(low) * 0.01 - 1e-6):
        errors = _validation(post(client, {"features": {name: value}}))
        assert errors[0]["loc"] == ["body", "features", name]


@pytest.mark.parametrize("name", [n for n, s in FEATURES.items() if s.leakage_risk != "none"])
def test_leaky_features_are_refused(client: TestClient, name: str) -> None:
    errors = _validation(post(client, {"features": {name: 1.0, "dis_r0_ohm": 0.08}}))
    assert "not accepted" in errors[0]["msg"]


@pytest.mark.parametrize(
    ("body", "loc"),
    [
        ({"features": {"bogus_feature": 1.0}}, ["body", "features", "bogus_feature"]),
        ({"features": {"dis_r0_ohm": 0.1}, "extra": 1}, ["body", "extra"]),
        ({"features": {"dis_r0_ohm": True}}, ["body", "features", "dis_r0_ohm"]),
        ({"features": {"dis_r0_ohm": "0.1"}}, ["body", "features", "dis_r0_ohm"]),
        ({"features": {"dis_r0_ohm": [0.1]}}, ["body", "features", "dis_r0_ohm"]),
        ({"features": {"dis_r0_ohm": 0.1}, "confidence": 0.85}, ["body", "confidence"]),
        ({"features": {"dis_r0_ohm": 0.1}, "confidence": "0.9"}, ["body", "confidence"]),
        ({"features": {"dis_r0_ohm": 0.1}, "confidence": 1}, ["body", "confidence"]),
        ({"features": {"dis_r0_ohm": 0.1}, "model_id": "../etc/passwd"}, ["body", "model_id"]),
        ({"features": {"dis_r0_ohm": 0.1}, "model_id": 7}, ["body", "model_id"]),
        ({"features": {"dis_r0_ohm": 0.1}, "reference": {"x": 1.0}}, ["body", "reference", "x"]),
        ({"features": [1, 2]}, ["body", "features"]),
        ({}, ["body", "features"]),
    ],
)
def test_invalid_bodies_are_422(client: TestClient, body: Any, loc: list[Any]) -> None:
    errors = _validation(post(client, body))
    assert loc in [e["loc"] for e in errors]


def test_non_object_and_non_json_bodies_are_422(client: TestClient) -> None:
    _validation(client.post("/v1/soh/estimate", json=[1, 2, 3]))
    _validation(
        client.post(
            "/v1/soh/estimate", content=b"{not json", headers={"Content-Type": "application/json"}
        )
    )
    _validation(
        client.post("/v1/soh/estimate", content=b"a=b", headers={"Content-Type": "text/plain"})
    )


def test_error_list_is_capped_and_inputs_are_not_echoed(client: TestClient) -> None:
    secret = "s3cr3t-" + "x" * 500
    body = {"features": {f"unknown_{i}": secret for i in range(200)}}
    response = post(client, body)
    errors = _validation(response)
    assert len(errors) == 50
    assert secret not in response.text
    assert "(+197 more)" in response.json()["detail"]


# ---------------------------------------------------------------------- batch
def test_batch_matches_individual_estimates(
    client: TestClient, sample_features: dict[str, float]
) -> None:
    items = [
        {"features": sample_features},
        {"features": {f: sample_features[f] for f in FEATURE_SETS["discharge"]}},
        {"features": sample_features, "confidence": 0.8, "model_id": "charge-absolute-ridge"},
    ]
    response = client.post("/v1/soh/batch", json={"items": items})
    assert response.status_code == 200
    results = response.json()["results"]
    assert [r["model_id"] for r in results] == [
        "combined-absolute-ridge",
        "discharge-absolute-ridge",
        "charge-absolute-ridge",
    ]
    for item, result in zip(items, results, strict=True):
        assert result == post(client, item).json()


def test_batch_is_atomic_and_names_the_failing_item(
    client: TestClient, sample_features: dict[str, float]
) -> None:
    items = [{"features": sample_features}, {"features": sample_features, "model_id": "nope"}]
    response = client.post("/v1/soh/batch", json={"items": items})
    assert response.status_code == 404
    assert response.json()["detail"].startswith("items[1]: ")


@pytest.mark.parametrize("n", [0, 1001])
def test_batch_size_limits(client: TestClient, n: int) -> None:
    items = [{"features": {"dis_r0_ohm": 0.08}}] * n
    errors = _validation(client.post("/v1/soh/batch", json={"items": items}))
    assert errors[0]["loc"] == ["body", "items"]


def test_default_batch_limit_is_100_items(
    client: TestClient, sample_features: dict[str, float]
) -> None:
    items = [{"features": sample_features}] * 100
    assert client.post("/v1/soh/batch", json={"items": items}).status_code == 200
    response = client.post("/v1/soh/batch", json={"items": [*items, items[0]]})
    errors = _validation(response)
    assert errors == [
        {
            "type": "too_long",
            "loc": ["body", "items"],
            "msg": "List should have at most 100 items (this server's batch limit), not 101; "
            "split the request",
        }
    ]
    assert response.json()["detail"].startswith("items: List should have at most 100 items")


def test_batch_of_1000_is_accepted_when_configured(
    settings: Settings, sample_features: dict[str, float]
) -> None:
    items = [{"features": sample_features}] * 1000
    assert len(json.dumps({"items": items})) < 1_000_000
    with TestClient(create_app(settings.model_copy(update={"max_batch_items": 1000}))) as client:
        response = client.post("/v1/soh/batch", json={"items": items})
    assert response.status_code == 200
    soh = {r["soh_pct"] for r in response.json()["results"]}
    assert len(soh) == 1
    assert math.isfinite(soh.pop())


def test_configured_batch_limit_is_enforced_before_estimating(
    settings: Settings, sample_features: dict[str, float], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []
    real = inference.estimate

    def counting(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(inference, "estimate", counting)
    with TestClient(create_app(settings.model_copy(update={"max_batch_items": 2}))) as client:
        items = [{"features": sample_features}] * 3
        assert client.post("/v1/soh/batch", json={"items": items}).status_code == 422
        assert calls == []
        assert client.post("/v1/soh/batch", json={"items": items[:2]}).status_code == 200
        assert len(calls) == 2


def test_relative_model_is_never_auto_selected_with_imputed_inputs(
    client: TestClient, sample_features: dict[str, float], training_frame: Any
) -> None:
    names = ("dis_q_380_350_ah", "dis_r0_ohm", "chg_cv_time_s")
    reference = _reference(training_frame)
    body = post(
        client,
        {
            "features": {n: sample_features[n] for n in names},
            "reference": {n: reference[n] for n in names},
        },
    ).json()
    assert body["model_id"] == "combined-absolute-ridge"
    assert len(body["domain"]["missing"]) == 5


def test_only_relative_models_loaded_explains_the_reference(
    models_dir: Path, sample_features: dict[str, float]
) -> None:
    full = ModelRegistry.load(models_dir)
    relative_only = ModelRegistry([e for e in full.entries if e.mode == "relative"])
    with pytest.raises(ProblemError) as info:
        inference.estimate(relative_only, sample_features)
    assert info.value.status == 422
    assert "reference" in info.value.detail
