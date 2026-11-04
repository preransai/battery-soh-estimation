"""Model registry: verified, fail-closed loading and deterministic selection."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from battery_soh.api.app import create_app
from battery_soh.api.registry import ModelRegistry
from battery_soh.api.settings import Settings
from battery_soh.features.registry import FEATURE_SETS
from battery_soh.models.linear_bundle import write_manifest

COMBINED = set(FEATURE_SETS["combined"])
DISCHARGE = set(FEATURE_SETS["discharge"])


@pytest.fixture
def model_copy(models_dir: Path, tmp_path: Path) -> Path:
    target = tmp_path / "models"
    shutil.copytree(models_dir, target)
    return target


def _edit_bundle(directory: Path, name: str, edit: Any, *, remanifest: bool = True) -> None:
    path = directory / name
    doc = json.loads(path.read_text(encoding="utf-8"))
    edit(doc)
    path.write_text(json.dumps(doc), encoding="utf-8")
    if remanifest:
        write_manifest(directory)


def test_loads_every_listed_bundle(models_dir: Path) -> None:
    registry = ModelRegistry.load(models_dir)
    assert len(registry) == 4
    assert registry.error is None
    assert registry.model_ids[0] == "combined-relative-ridge"  # lowest recorded MAE first
    entry = registry.get("combined-absolute-ridge")
    assert entry is not None
    assert entry.metrics == {"lobo_mae": 1.4, "lobo_rmse": 1.82, "coverage_90": 0.9}
    assert len(entry.sha256) == 64


def test_tampered_bundle_means_no_model_at_all(model_copy: Path, settings: Settings) -> None:
    path = model_copy / "discharge-absolute-ridge.json"
    path.write_bytes(path.read_bytes().replace(b'"alpha": 1.0', b'"alpha": 2.0'))
    registry = ModelRegistry.load(model_copy)
    assert not registry
    assert registry.error == "model manifest verification failed"
    with TestClient(create_app(settings.model_copy(update={"models_dir": model_copy}))) as client:
        assert client.get("/readyz").status_code == 503
        assert client.get("/v1/models").json() == []


@pytest.mark.parametrize("case", ["missing_manifest", "missing_dir", "listed_file_deleted"])
def test_missing_artifacts_fail_closed(model_copy: Path, case: str) -> None:
    if case == "missing_manifest":
        (model_copy / "MANIFEST.json").unlink()
        directory = model_copy
    elif case == "missing_dir":
        directory = model_copy / "nope"
    else:
        (model_copy / "charge-absolute-ridge.json").unlink()
        directory = model_copy
    registry = ModelRegistry.load(directory)
    assert not registry
    assert registry.error


def test_unlisted_files_are_never_loaded(model_copy: Path) -> None:
    manifest = json.loads((model_copy / "MANIFEST.json").read_text(encoding="utf-8"))
    del manifest["files"]["charge-absolute-ridge.json"]
    (model_copy / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    registry = ModelRegistry.load(model_copy)
    assert "charge-absolute-ridge" not in registry
    assert len(registry) == 3


def test_bundle_with_leaky_feature_is_refused(model_copy: Path) -> None:
    def leak(doc: dict[str, Any]) -> None:
        doc["features"][0] = "leg_capacity_ah"

    _edit_bundle(model_copy, "discharge-absolute-ridge.json", leak)
    registry = ModelRegistry.load(model_copy)
    assert "discharge-absolute-ridge" not in registry
    assert len(registry) == 3


@pytest.mark.parametrize("bad_id", ["../escape", "a/b", "", "x" * 65, "has space"])
def test_unsafe_model_id_is_refused(model_copy: Path, bad_id: str) -> None:
    def rename(doc: dict[str, Any]) -> None:
        doc["model_id"] = bad_id

    _edit_bundle(model_copy, "charge-absolute-ridge.json", rename)
    registry = ModelRegistry.load(model_copy)
    assert bad_id not in registry
    assert len(registry) == 3


MALFORMED_EDITS = {
    "margin_null": lambda doc: doc["domain"].__setitem__("margin", None),
    "margin_string": lambda doc: doc["domain"].__setitem__("margin", "abc"),
    "ragged_precision": lambda doc: doc["domain"].__setitem__(
        "precision_matrix", [[1.0], [1.0, 2.0]]
    ),
    "metrics_list": lambda doc: doc.__setitem__("metrics", [1, 2]),
}


@pytest.mark.parametrize("case", sorted(MALFORMED_EDITS))
def test_manifest_valid_but_malformed_bundle_is_skipped(model_copy: Path, case: str) -> None:
    _edit_bundle(model_copy, "charge-absolute-ridge.json", MALFORMED_EDITS[case])
    registry = ModelRegistry.load(model_copy)  # must not raise
    assert "charge-absolute-ridge" not in registry
    assert len(registry) == 3


def test_deeply_nested_bundle_is_skipped(model_copy: Path) -> None:
    path = model_copy / "charge-absolute-ridge.json"
    text = path.read_text(encoding="utf-8").rstrip()
    assert text.endswith("}")
    path.write_text(text[:-1] + ', "extra": ' + "[" * 100_000 + "]" * 100_000 + "}", "utf-8")
    write_manifest(model_copy)
    registry = ModelRegistry.load(model_copy)  # RecursionError inside json must not escape
    assert "charge-absolute-ridge" not in registry
    assert len(registry) == 3


@pytest.mark.parametrize("case", ["margin_null", "ragged_precision"])
def test_only_malformed_bundles_mean_not_ready_not_a_crash(
    model_copy: Path, settings: Settings, case: str
) -> None:
    for path in sorted(model_copy.glob("*.json")):
        if path.name != "MANIFEST.json":
            _edit_bundle(model_copy, path.name, MALFORMED_EDITS[case], remanifest=False)
    write_manifest(model_copy)  # the folder verifies; every bundle in it is malformed
    registry = ModelRegistry.load(model_copy)
    assert not registry
    assert registry.error == "no usable model bundle"
    with TestClient(create_app(settings.model_copy(update={"models_dir": model_copy}))) as client:
        ready = client.get("/readyz")
        assert ready.status_code == 503
        assert ready.json()["type"] == "/problems/not-ready"
        assert client.get("/livez").status_code == 200


def test_non_standard_json_constants_are_refused(model_copy: Path) -> None:
    path = model_copy / "charge-absolute-ridge.json"
    text = path.read_text(encoding="utf-8").replace('"r2": 0.95', '"r2": NaN', 1)
    path.write_text(text, encoding="utf-8")
    write_manifest(model_copy)
    registry = ModelRegistry.load(model_copy)
    assert "charge-absolute-ridge" not in registry


def test_duplicate_model_ids_keep_the_first_file(model_copy: Path) -> None:
    def clash(doc: dict[str, Any]) -> None:
        doc["model_id"] = "charge-absolute-ridge"

    _edit_bundle(model_copy, "discharge-absolute-ridge.json", clash)
    registry = ModelRegistry.load(model_copy)
    assert len(registry) == 3
    entry = registry.get("charge-absolute-ridge")
    assert entry is not None
    assert entry.file_name == "charge-absolute-ridge.json"


def test_bundle_without_metrics_ranks_last(model_copy: Path) -> None:
    def drop(doc: dict[str, Any]) -> None:
        doc["metrics"] = {}

    _edit_bundle(model_copy, "combined-relative-ridge.json", drop)
    registry = ModelRegistry.load(model_copy)
    assert registry.model_ids[-1] == "combined-relative-ridge"
    entry = registry.get("combined-relative-ridge")
    assert entry is not None
    assert entry.summary()["metrics"]["lobo_mae"] is None


# ---------------------------------------------------------------------- selection
@pytest.fixture(scope="module")
def registry(models_dir: Path) -> ModelRegistry:
    return ModelRegistry.load(models_dir)


def test_select_complete_lowest_mae(registry: ModelRegistry) -> None:
    entry = registry.select(COMBINED, "absolute")
    assert entry is not None
    assert entry.model_id == "combined-absolute-ridge"
    entry = registry.select(COMBINED)  # any mode: the relative bundle has the lowest MAE
    assert entry is not None
    assert entry.model_id == "combined-relative-ridge"
    entry = registry.select(DISCHARGE, "absolute")
    assert entry is not None
    assert entry.model_id == "discharge-absolute-ridge"


def test_select_relative_requires_its_inputs(registry: ModelRegistry) -> None:
    assert registry.select(DISCHARGE, "relative") is None


def test_select_partial_prefers_coverage_then_mae(registry: ModelRegistry) -> None:
    available = {"dis_q_380_350_ah", "chg_cv_time_s"}
    assert registry.select(available, "absolute") is None
    entry = registry.select(available, "absolute", allow_partial=True)
    # combined covers 2 (MAE 1.4); discharge and charge cover 1 each
    assert entry is not None
    assert entry.model_id == "combined-absolute-ridge"
    assert registry.select({"eis_re_ohm"}, "absolute", allow_partial=True) is None


def test_empty_registry() -> None:
    registry = ModelRegistry.empty("why")
    assert not registry
    assert registry.select(COMBINED) is None
    assert registry.model_ids == []
    assert registry.error == "why"


def test_undecodable_manifest_fails_closed(model_copy: Path) -> None:
    (model_copy / "MANIFEST.json").write_bytes(b"\xff\xfe\x00garbage")
    registry = ModelRegistry.load(model_copy)
    assert not registry
