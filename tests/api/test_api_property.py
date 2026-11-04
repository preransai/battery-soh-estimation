"""Property test: arbitrary JSON-ish request bodies never produce a server error."""

from __future__ import annotations

import json
from typing import Any

from fastapi.testclient import TestClient
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from battery_soh.features.registry import FEATURES

NAMES = st.sampled_from(sorted(FEATURES)) | st.text(max_size=12)
SCALARS = (
    st.floats(allow_nan=True, allow_infinity=True)
    | st.floats(min_value=-1e3, max_value=1e4)
    | st.integers(min_value=-(10**20), max_value=10**20)
    | st.booleans()
    | st.none()
    | st.text(max_size=8)
)
VALUES = st.recursive(
    SCALARS, lambda inner: st.lists(inner, max_size=3) | st.dictionaries(NAMES, inner, max_size=3),
    max_leaves=4,
)  # fmt: skip
FEATURE_DICTS = st.dictionaries(NAMES, VALUES, max_size=8)
BODIES = st.fixed_dictionaries(
    {"features": FEATURE_DICTS | VALUES},
    optional={
        "reference": FEATURE_DICTS | st.none(),
        "model_id": st.sampled_from(["combined-absolute-ridge", "combined-relative-ridge", "x"])
        | st.text(max_size=10)
        | st.none(),
        "confidence": st.sampled_from([0.8, 0.9, 0.95, 0.5, "0.9"]) | st.floats(),
    },
)


@settings(
    max_examples=80,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
@given(body=BODIES)
def test_estimate_never_returns_5xx(client: TestClient, body: dict[str, Any]) -> None:
    # json.dumps writes NaN/Infinity as bare tokens, which the server must also refuse.
    raw = json.dumps(body).encode()
    response = client.post(
        "/v1/soh/estimate", content=raw, headers={"Content-Type": "application/json"}
    )
    assert response.status_code in (200, 404, 422), response.text
    payload = response.json()
    if response.status_code == 200:
        assert payload["interval"]["lower"] <= payload["soh_pct"] <= payload["interval"]["upper"]
    else:
        assert payload["status"] == response.status_code


@settings(
    max_examples=40,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
@given(
    features=st.dictionaries(
        st.sampled_from(sorted(n for n, s in FEATURES.items() if s.leakage_risk == "none")),
        st.floats(allow_nan=False, allow_infinity=False),
        min_size=1,
    ),
    reference=st.none()
    | st.dictionaries(
        st.sampled_from(sorted(n for n, s in FEATURES.items() if s.leakage_risk == "none")),
        st.floats(min_value=0.0, max_value=2.0),
    ),
)
def test_in_range_features_always_give_a_finite_answer_or_a_clean_422(
    client: TestClient, features: dict[str, float], reference: dict[str, float] | None
) -> None:
    # Clamp into the plausible range so the body passes schema validation.
    clamped = {}
    for name, value in features.items():
        low, high = FEATURES[name].plausible_range
        clamped[name] = min(max(value, low), high)
    body: dict[str, Any] = {"features": clamped}
    if reference is not None:
        body["reference"] = {
            n: min(max(v, FEATURES[n].plausible_range[0]), FEATURES[n].plausible_range[1])
            for n, v in reference.items()
        }
    response = client.post("/v1/soh/estimate", json=body)
    assert response.status_code in (200, 422), response.text
    if response.status_code == 200:
        result = response.json()
        assert 0.0 <= result["soh_pct"] <= 120.0
    else:
        assert response.json()["type"] in (
            "/problems/no-usable-features",
            "/problems/reference-required",
            "/problems/non-finite-result",
        )
