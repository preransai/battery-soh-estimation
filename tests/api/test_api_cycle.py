"""POST /v1/soh/estimate-from-cycle: train/serve parity and raw-array validation."""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from battery_soh.api import inference
from battery_soh.api.app import create_app
from battery_soh.api.inference import EXTRACTED_FEATURES
from battery_soh.api.settings import Settings
from battery_soh.data.schema import Segment
from battery_soh.features.extract import features_from_segments

URL = "/v1/soh/estimate-from-cycle"


def arrays(seg: Segment) -> dict[str, list[float]]:
    return {
        "time_s": seg.time_s.tolist(),
        "voltage_v": seg.voltage_v.tolist(),
        "current_a": seg.current_a.tolist(),
        "temperature_c": seg.temperature_c.tolist(),
    }


@pytest.fixture(scope="module")
def cycle_85(cycle_simulator: Any) -> tuple[Segment, Segment]:
    """A fresh (unseen) synthetic cycle at SOH 85 % of a cell between the training cells."""
    return cycle_simulator(2.01, 0.85, 0.071, seed=987)  # type: ignore[no-any-return]


def test_features_equal_the_training_extractor(
    client: TestClient, cycle_85: tuple[Segment, Segment]
) -> None:
    charge, discharge = cycle_85
    response = client.post(URL, json={"charge": arrays(charge), "discharge": arrays(discharge)})
    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == {"features", "estimate"}
    assert set(body["features"]) == set(EXTRACTED_FEATURES)
    assert not any(name.startswith("leg_") for name in body["features"])
    expected = features_from_segments(charge, discharge)
    for name, value in body["features"].items():
        if math.isnan(expected[name]):
            assert value is None, name
        else:
            assert value == pytest.approx(expected[name], rel=1e-12), name
    estimate = body["estimate"]
    assert estimate["model_id"] == "combined-absolute-ridge"
    assert abs(estimate["soh_pct"] - 85.0) < 3.0
    assert estimate["interval"]["lower"] <= estimate["soh_pct"] <= estimate["interval"]["upper"]


def test_estimate_equals_estimate_endpoint_on_extracted_features(
    client: TestClient, cycle_85: tuple[Segment, Segment]
) -> None:
    charge, discharge = cycle_85
    body = client.post(URL, json={"charge": arrays(charge), "discharge": arrays(discharge)}).json()
    present = {k: v for k, v in body["features"].items() if v is not None}
    direct = client.post("/v1/soh/estimate", json={"features": present}).json()
    assert body["estimate"] == direct


def test_discharge_only_selects_the_discharge_model(
    client: TestClient, cycle_85: tuple[Segment, Segment]
) -> None:
    _, discharge = cycle_85
    body = client.post(URL, json={"discharge": arrays(discharge), "confidence": 0.8}).json()
    assert body["estimate"]["model_id"] == "discharge-absolute-ridge"
    assert body["features"]["chg_cv_time_s"] is None
    assert body["estimate"]["interval"]["confidence"] == 0.8


def test_eis_values_flow_into_the_features(
    client: TestClient, cycle_85: tuple[Segment, Segment]
) -> None:
    _, discharge = cycle_85
    body = client.post(
        URL, json={"discharge": arrays(discharge), "eis": {"re_ohm": 0.05, "rct_ohm": 0.08}}
    ).json()
    assert body["features"]["eis_re_ohm"] == 0.05
    assert body["features"]["eis_rct_ohm"] == 0.08


def test_partial_window_record_still_works(
    client: TestClient, cycle_85: tuple[Segment, Segment]
) -> None:
    """Only the part of the discharge down to 3.4 V: the 3.80-3.50 V window is complete."""
    _, discharge = cycle_85
    keep = int(np.flatnonzero(discharge.voltage_v < 3.4)[0])
    payload = {k: v[:keep] for k, v in arrays(discharge).items()}
    body = client.post(URL, json={"discharge": payload}).json()
    assert body["features"]["dis_q_380_350_ah"] is not None
    assert body["estimate"]["model_id"] == "discharge-absolute-ridge"


def test_record_without_any_window_is_422(client: TestClient) -> None:
    n = 50
    flat = {
        "time_s": [10.0 * i for i in range(n)],
        "voltage_v": [3.3] * n,
        "current_a": [0.0] * n,
        "temperature_c": [24.0] * n,
    }
    response = client.post(URL, json={"discharge": flat})
    assert response.status_code == 422
    assert response.json()["type"] == "/problems/no-health-indicators"


def test_wrong_current_sign_is_warned(
    client: TestClient, cycle_85: tuple[Segment, Segment]
) -> None:
    charge, discharge = cycle_85
    flipped = arrays(charge)
    flipped["current_a"] = [-c for c in flipped["current_a"]]
    body = client.post(URL, json={"charge": flipped, "discharge": arrays(discharge)}).json()
    assert any("wrong sign" in w for w in body["estimate"]["warnings"])


def _segment(n: int = 10) -> dict[str, list[float]]:
    return {
        "time_s": [float(i) for i in range(n)],
        "voltage_v": [3.9] * n,
        "current_a": [-2.0] * n,
        "temperature_c": [25.0] * n,
    }


def _set_first(name: str, value: float) -> Any:
    return lambda seg: seg.update({name: [value, *seg[name][1:]]})


@pytest.mark.parametrize(
    ("mutate", "loc"),
    [
        (lambda s: s.update(voltage_v=s["voltage_v"][:-1]), ["body", "discharge"]),
        (lambda s: s.update(time_s=[0.0, 2.0, 1.0, *s["time_s"][3:]]), ["body", "discharge"]),
        (_set_first("time_s", -1.0), ["body", "discharge", "time_s", 0]),
        (_set_first("voltage_v", 11.0), ["body", "discharge", "voltage_v", 0]),
        (_set_first("current_a", 500.0), ["body", "discharge", "current_a", 0]),
        (_set_first("temperature_c", 250.0), ["body", "discharge", "temperature_c", 0]),
        (lambda s: s.pop("temperature_c"), ["body", "discharge", "temperature_c"]),
        (lambda s: s.update(extra=[1.0]), ["body", "discharge", "extra"]),
        (lambda s: s.update(time_s=[0.0]), ["body", "discharge", "time_s"]),
    ],
)
def test_invalid_segments_are_422(client: TestClient, mutate: Any, loc: list[Any]) -> None:
    segment = _segment()
    mutate(segment)
    response = client.post(URL, json={"discharge": segment})
    assert response.status_code == 422, response.text
    assert loc in [e["loc"] for e in response.json()["errors"]]


def test_nan_in_arrays_is_422(client: TestClient) -> None:
    raw = (
        b'{"discharge": {"time_s": [0, 1], "voltage_v": [3.9, NaN], '
        b'"current_a": [-2, -2], "temperature_c": [25, 25]}}'
    )
    response = client.post(URL, content=raw, headers={"Content-Type": "application/json"})
    assert response.status_code == 422
    assert response.json()["errors"][0]["loc"] == ["body", "discharge", "voltage_v", 1]


def test_empty_body_needs_some_measurement(client: TestClient) -> None:
    response = client.post(URL, json={})
    assert response.status_code == 422
    assert "at least one of" in response.json()["errors"][0]["msg"]


def test_schema_caps_samples_at_20000(client: TestClient) -> None:
    n = 20_001
    segment = {
        "time_s": list(range(n)),
        "voltage_v": [4] * n,
        "current_a": [0] * n,
        "temperature_c": [25] * n,
    }
    response = client.post(URL, json={"discharge": segment})
    assert response.status_code == 422
    assert ["body", "discharge", "time_s"] in [e["loc"] for e in response.json()["errors"]]


def test_configured_sample_limit_is_enforced(settings: Settings) -> None:
    strict = settings.model_copy(update={"max_samples": 20})
    with TestClient(create_app(strict)) as client:
        response = client.post(URL, json={"discharge": _segment(21)})
        assert response.status_code == 422
        error = response.json()["errors"][0]
        assert error["type"] == "too_long"
        assert error["loc"] == ["body", "discharge", "time_s"]


def test_rejected_record_logs_one_line_and_keeps_the_traceback_at_debug(
    client: TestClient,
    cycle_85: tuple[Segment, Segment],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def reject(*_args: Any) -> Any:
        raise ValueError("time_s\nmust increase")

    monkeypatch.setattr(inference, "extract_features", reject)
    _, discharge = cycle_85
    with caplog.at_level(logging.DEBUG, logger="battery_soh.api.inference"):
        response = client.post(URL, json={"discharge": arrays(discharge)})
    assert response.status_code == 422
    assert response.json()["type"] == "/problems/invalid-record"
    records = [r for r in caplog.records if r.name == "battery_soh.api.inference"]
    warning = [r for r in records if r.levelno == logging.WARNING]
    assert len(warning) == 1
    assert warning[0].exc_info is None  # client-triggerable: no traceback at WARNING
    assert "\n" not in warning[0].getMessage()
    assert "ValueError: time_s must increase" in warning[0].getMessage()
    debug = [r for r in records if r.levelno == logging.DEBUG]
    assert len(debug) == 1
    assert debug[0].exc_info is not None
