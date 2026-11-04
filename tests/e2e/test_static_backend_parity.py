"""Train/serve parity of the static (GitHub Pages) demo.

The static site estimates SOH in the browser with ``frontend/assets/js/static-backend.js``, a
JavaScript port of the Python serving path. These tests run that module in Node.js on the
committed model bundles and demo cycles and require exactly the answers of
:func:`battery_soh.api.inference.estimate` (the code behind ``POST /v1/soh/estimate``): the same
model choice, estimates, intervals, attributions and domain flags within 1e-9, the same
warnings, and the same error status, title and message when a request is refused.

Requests go through the API's request schema first (``EstimateRequest`` validated the way
FastAPI validates a JSON body), so validation is covered too: leaky ``leg_*`` features,
unknown keys, wrong types, out-of-range values, bad model ids and confidences must give the
same 422 title, ``detail`` text and ``errors`` list in the browser as from the API. One test
anchors that reference to the HTTP API itself.

Skipped when Node.js is not installed or the committed artifacts are absent.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from battery_soh.api.app import create_app
from battery_soh.api.demo import DemoStore, get_demo_store
from battery_soh.api.errors import ProblemError, sanitize_errors, summarize_errors
from battery_soh.api.inference import estimate
from battery_soh.api.registry import ModelRegistry
from battery_soh.api.routers.models import list_models_payload
from battery_soh.api.schemas import INPUT_FEATURES, EstimateRequest
from battery_soh.api.settings import Settings
from battery_soh.config import get_paths
from battery_soh.features.registry import FEATURES

NODE = shutil.which("node")
TOL = 1e-9
BACKEND = get_paths().frontend / "assets" / "js" / "static-backend.js"

RUNNER = """
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
const [backendFile, jobsFile] = process.argv.slice(2);
const backend = await import(pathToFileURL(backendFile).href);
const { models, bundles, jobs } = JSON.parse(readFileSync(jobsFile, "utf8"));
const deps = { listModels: async () => models, loadBundle: async (id) => bundles[id] };
const out = [];
for (const body of jobs) {
  try {
    out.push({ ok: true, result: await backend.estimate(body, deps) });
  } catch (err) {
    if (!(err instanceof backend.StaticBackendError)) throw err;
    out.push({ ok: false, status: err.status, body: err.body });
  }
}
process.stdout.write(JSON.stringify(out));
"""

pytestmark = pytest.mark.skipif(NODE is None, reason="Node.js is not installed")


@pytest.fixture(scope="module")
def registry() -> ModelRegistry:
    loaded = ModelRegistry.load(get_paths().models_dir)
    if not loaded:
        pytest.skip(f"no verified model bundles in {get_paths().models_dir}")
    return loaded


@pytest.fixture(scope="module")
def store() -> DemoStore:
    demo = get_demo_store(get_paths().demo_dir)
    if demo is None:
        pytest.skip(f"no demo asset in {get_paths().demo_dir}")
    return demo


def _bodies(registry: ModelRegistry, store: DemoStore) -> list[dict[str, Any]]:
    """Requests over every demo cycle: automatic choice (with and without a reference),
    every model explicitly, partial inputs and refused requests."""
    bodies: list[dict[str, Any]] = []
    for battery in store.battery_list():
        reference = {k: v for k, v in battery["reference"].items() if v is not None}
        for entry in battery["cycles"]:
            cycle = store.cycle(battery["battery_id"], entry["cycle_index"])
            assert cycle is not None
            features = {k: v for k, v in cycle["features"].items() if v is not None}
            confidence = (0.8, 0.9, 0.95)[entry["cycle_index"] % 3]
            bodies.append({"features": features, "confidence": confidence})
            bodies.append({"features": features, "reference": reference, "confidence": confidence})
            for model in registry.entries:
                body: dict[str, Any] = {
                    "features": features,
                    "model_id": model.model_id,
                    "confidence": confidence,
                }
                if model.mode == "relative":
                    body["reference"] = reference
                bodies.append(body)
                if model.mode == "relative":  # refused: no reference
                    bodies.append({"features": features, "model_id": model.model_id})
            # one indicator only: imputation (or a refusal) depending on the model
            bodies.append({"features": {"dis_q_380_350_ah": features.get("dis_q_380_350_ah", 0.7)}})
    bodies += [
        {"features": {"ctx_cycle_index": 10.0}},  # no model uses it: refused
        {"features": {"dis_q_380_350_ah": 4.9, "dis_r0_ohm": 1.9}},  # far outside the domain
        {"features": {"dis_q_380_350_ah": 5.0, "dis_r0_ohm": 0.0}},  # clipped prediction
        {"features": {"dis_q_380_350_ah": 0.7}, "model_id": "no-such-model"},
    ]
    return bodies + _validation_bodies(registry, store)


def _validation_bodies(registry: ModelRegistry, store: DemoStore) -> list[Any]:
    """Requests the API's schema refuses (plus a few edge cases it accepts)."""
    q = {"dis_q_380_350_ah": 0.66}
    bodies: list[Any] = [
        # label-side/proxy features are refused as a whole, before anything else in the map
        {"features": {"leg_capacity_ah": 1.8, **q}},
        {"features": {**q, "leg_temp_max_c": 30.0, "leg_capacity_ah": 1.8, "made_up": 1.0}},
        {"features": q, "reference": {"leg_record_time_s": 3000.0}},
        {"features": {"leg_capacity_ah": None, **q}},
        # unknown keys, wrong types, missing or malformed objects
        {"features": {**q, "made_up": 1.0, "zzz": 2}},
        {"features": {"k" * 200: 1.0}},  # long key: truncated in the error location
        {"features": {"dis_r0_ohm": "0.1"}},
        {"features": {"dis_r0_ohm": True}},
        {"features": {"dis_r0_ohm": [0.1]}},
        {"features": {"dis_r0_ohm": {"value": 0.1}}},
        {"features": {"dis_r0_ohm": None, **q}},  # accepted: null means missing
        {"features": [0.66]},
        {"features": "x"},
        {"features": None},
        {},
        {"reference": {}},
        {"features": q, "extra": 1, "another": None},
        {"features": q, "reference": [1]},
        {"features": q, "reference": None},  # accepted
        {"features": q, "reference": {}},  # accepted: an empty reference is no reference
        # confidence and model id
        {"features": q, "confidence": 0.85},
        {"features": q, "confidence": "0.9"},
        {"features": q, "confidence": None},
        {"features": q, "confidence": True},
        {"features": q, "confidence": 1},
        {"features": q, "model_id": "../../etc"},
        {"features": q, "model_id": ""},
        {"features": q, "model_id": "x" * 65},
        {"features": q, "model_id": 7},
        {"features": q, "model_id": None},  # accepted
        # many errors at once: summary cut-off and "(+N more)"
        {
            "features": {name: FEATURES[name].plausible_range[1] + 1 for name in INPUT_FEATURES},
            "confidence": 0.5,
            "model_id": "!",
            "x": 1,
        },
        # not an object at all
        [1, 2],
        "body",
        7,
    ]
    for name in INPUT_FEATURES:  # both bounds of every input (message formatting of the bound)
        low, high = FEATURES[name].plausible_range
        bodies.append({"features": {name: low - 1}})
        bodies.append({"features": {name: high + 1}})
        bodies.append({"features": {name: high}})  # the bound itself is valid
    relative = [e for e in registry.entries if e.mode == "relative"]
    if relative:  # a subnormal reference gives an infinite ratio: "reference required"
        battery = store.battery_list()[0]
        cycle = store.cycle(battery["battery_id"], battery["cycles"][-1]["cycle_index"])
        assert cycle is not None
        entry = relative[0]
        features = {f: cycle["features"][f] for f in entry.features}
        reference = {f: battery["reference"][f] for f in entry.features}
        target = next(f for f in entry.features if not f.startswith("ctx_"))
        reference[target] = 5e-324
        bodies.append({"features": features, "reference": reference, "model_id": entry.model_id})
    return bodies


def _validate(body: Any) -> tuple[EstimateRequest | None, dict[str, Any] | None]:
    """``EstimateRequest`` exactly as FastAPI validates a JSON body (``from_attributes=True``,
    errors located under ``body``), and the API's 422 answer when that fails."""
    try:
        return EstimateRequest.model_validate(body, from_attributes=True), None
    except ValidationError as exc:
        raw = [{**error, "loc": ("body", *error["loc"])} for error in exc.errors()]
        errors = sanitize_errors(raw)
        return None, {
            "ok": False,
            "status": 422,
            "title": "Invalid input",
            "detail": summarize_errors(errors, len(raw)),
            "errors": errors,
        }


def _python(registry: ModelRegistry, body: Any) -> dict[str, Any]:
    """The API's answer for ``body``: schema validation, then the service layer exactly as
    the ``POST /v1/soh/estimate`` router calls it."""
    request, refused = _validate(body)
    if refused is not None:
        return refused
    assert request is not None
    try:
        result = estimate(
            registry,
            request.features.provided(),
            reference=request.reference.provided() if request.reference is not None else None,
            model_id=request.model_id,
            confidence=request.confidence,
        )
    except ProblemError as exc:
        return {"ok": False, "status": exc.status, "title": exc.title, "detail": exc.detail}
    return {"ok": True, "result": result}


def _assert_same_estimate(js: dict[str, Any], py: dict[str, Any], where: str) -> None:
    assert js["model_id"] == py["model_id"], where
    for key in ("soh_pct", "base_value"):
        assert js[key] == pytest.approx(py[key], abs=TOL), f"{where}: {key}"
    for key in ("lower", "upper", "confidence"):
        assert js["interval"][key] == pytest.approx(py["interval"][key], abs=TOL), where
    assert js["interval"]["method"] == py["interval"]["method"], where
    assert [c["feature"] for c in js["contributions"]] == [
        c["feature"] for c in py["contributions"]
    ], where
    for a, b in zip(js["contributions"], py["contributions"], strict=True):
        assert a["value"] == pytest.approx(b["value"], abs=TOL), where
        assert a["contribution"] == pytest.approx(b["contribution"], abs=TOL), where
    jd, pd = js["domain"], py["domain"]
    assert jd["in_domain"] == pd["in_domain"], where
    assert jd["out_of_range"] == pd["out_of_range"], where
    assert jd["missing"] == pd["missing"], where
    assert jd["mahalanobis"] == pytest.approx(pd["mahalanobis"], abs=TOL), where
    assert jd["threshold"] == pytest.approx(pd["threshold"], abs=TOL), where
    assert js["warnings"] == py["warnings"], where


def test_static_backend_matches_the_api(
    registry: ModelRegistry, store: DemoStore, tmp_path: Path
) -> None:
    bodies = _bodies(registry, store)
    jobs = tmp_path / "jobs.json"
    jobs.write_text(
        json.dumps(
            {
                "models": list_models_payload(registry),
                "bundles": {e.model_id: e.bundle.to_dict() for e in registry.entries},
                "jobs": bodies,
            }
        ),
        encoding="utf-8",
    )
    runner = tmp_path / "runner.mjs"
    runner.write_text(RUNNER, encoding="utf-8")
    assert NODE is not None
    done = subprocess.run(
        [NODE, str(runner), str(BACKEND), str(jobs)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    results = json.loads(done.stdout)
    assert len(results) == len(bodies)

    n_estimates = n_errors = n_validation = 0
    chosen: set[str] = set()
    for i, (body, js) in enumerate(zip(bodies, results, strict=True)):
        py = _python(registry, body)
        where = f"request {i}: {json.dumps(body)[:160]}"
        assert js["ok"] == py["ok"], f"{where}: browser {js} vs API {py}"
        if py["ok"]:
            _assert_same_estimate(js["result"], py["result"], where)
            chosen.add(py["result"]["model_id"])
            n_estimates += 1
        else:
            assert (js["status"], js["body"]["title"], js["body"]["detail"]) == (
                py["status"],
                py["title"],
                py["detail"],
            ), where
            if "errors" in py:
                assert js["body"]["errors"] == py["errors"], where
                assert js["body"]["type"] == "/problems/validation-error", where
                n_validation += 1
            n_errors += 1
    # The sweep must actually exercise every model, both outcomes and the validation layer.
    assert chosen == set(registry.model_ids)
    assert n_estimates > 100
    assert n_errors > 0
    assert n_validation > 50


def test_validation_reference_is_the_http_api(registry: ModelRegistry, store: DemoStore) -> None:
    """The schema-level reference used above answers exactly like the running service."""
    settings = Settings(
        models_dir=get_paths().models_dir, demo_dir=get_paths().demo_dir, log_level="ERROR"
    )
    with TestClient(create_app(settings)) as client:
        for body in _validation_bodies(registry, store):
            response = client.post(
                "/v1/soh/estimate",
                content=json.dumps(body),
                headers={"Content-Type": "application/json"},
            )
            py = _python(registry, body)
            where = json.dumps(body)[:160]
            if py["ok"]:
                assert response.status_code == 200, where
                continue
            got = response.json()
            assert (response.status_code, got["title"], got["detail"]) == (
                py["status"],
                py["title"],
                py["detail"],
            ), where
            if "errors" in py:
                assert got["errors"] == py["errors"], where
