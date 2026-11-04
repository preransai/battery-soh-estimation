"""GET /v1/benchmark: summary pass-through, headline predictions, audit, strict JSON."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from battery_soh.api.app import create_app
from battery_soh.api.errors import ProblemError
from battery_soh.api.routers.benchmark import benchmark_payload
from battery_soh.api.settings import Settings


def test_benchmark_endpoint(client: TestClient) -> None:
    response = client.get("/v1/benchmark")
    assert response.status_code == 200
    body = response.json()
    assert body["headline"]["mae"] == 1.4
    assert body["leaderboard"][1]["rmse"] is None  # NaN in the file -> null, strict JSON
    assert body["audit"] == {"rows": [{"id": "v1_reported", "mae": 0.9, "r2": 0.99}]}
    predictions = body["predictions"]
    assert len(predictions) == 9  # headline spec only: 3 cells x 3 cycles
    assert predictions[0] == {
        "battery_id": "B0005",
        "cycle_index": 0,
        "y": 100.0,
        "yhat": 99.5,
        "lo_80": 98.0,
        "hi_80": 101.0,
        "lo_90": 97.5,
        "hi_90": 101.5,
        "lo_95": 96.9,
        "hi_95": 102.1,
        "is_reference_cycle": True,
    }
    assert [(p["battery_id"], p["cycle_index"]) for p in predictions] == sorted(
        (p["battery_id"], p["cycle_index"]) for p in predictions
    )


def test_missing_summary_is_404(settings: Settings, tmp_path: Path) -> None:
    with TestClient(create_app(settings.model_copy(update={"benchmark_dir": tmp_path}))) as client:
        response = client.get("/v1/benchmark")
        assert response.status_code == 404
        assert response.json()["type"] == "/problems/benchmark-unavailable"


@pytest.fixture
def bench_copy(reports_dirs: tuple[Path, Path], tmp_path: Path) -> tuple[Path, Path]:
    bench, audit = reports_dirs
    shutil.copytree(bench, tmp_path / "benchmark")
    shutil.copytree(audit, tmp_path / "audit")
    return tmp_path / "benchmark", tmp_path / "audit"


def test_existing_predictions_and_audit_are_kept(bench_copy: tuple[Path, Path]) -> None:
    bench, audit = bench_copy
    summary = json.loads((bench / "summary.json").read_text(encoding="utf-8"))
    summary["predictions"] = [{"battery_id": "B0005", "cycle_index": 1}]
    summary["audit"] = {"own": True}
    (bench / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    payload = benchmark_payload(bench, audit)
    assert payload["predictions"] == [{"battery_id": "B0005", "cycle_index": 1}]
    assert payload["audit"] == {"own": True}


def test_unidentifiable_predictions_are_omitted(bench_copy: tuple[Path, Path]) -> None:
    bench, _audit = bench_copy
    summary = json.loads((bench / "summary.json").read_text(encoding="utf-8"))
    del summary["headline"]["experiment"]  # the CSV has the column, so rows are ambiguous
    (bench / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    payload = benchmark_payload(bench, None)
    assert "predictions" not in payload
    assert "audit" not in payload


def test_files_are_reloaded_when_they_change(bench_copy: tuple[Path, Path]) -> None:
    bench, audit = bench_copy
    assert benchmark_payload(bench, audit)["headline"]["mae"] == 1.4
    path = bench / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["headline"]["mae"] = 1.25
    path.write_text(json.dumps(summary), encoding="utf-8")
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    assert benchmark_payload(bench, audit)["headline"]["mae"] == 1.25


def test_payload_is_a_copy(bench_copy: tuple[Path, Path]) -> None:
    bench, audit = bench_copy
    first = benchmark_payload(bench, audit)
    first["headline"]["mae"] = -1
    first["predictions"].clear()
    second = benchmark_payload(bench, audit)
    assert second["headline"]["mae"] == 1.4
    assert len(second["predictions"]) == 9


def test_invalid_summary_is_404(tmp_path: Path) -> None:
    (tmp_path / "summary.json").write_text("[1, 2", encoding="utf-8")
    with pytest.raises(ProblemError) as info:
        benchmark_payload(tmp_path)
    assert info.value.status == 404


def test_benchmark_response_is_cacheable_and_conditional(client: TestClient) -> None:
    first = client.get("/v1/benchmark")
    assert first.headers["cache-control"] == "public, max-age=300"
    etag = first.headers["etag"]
    assert etag.startswith('W/"')
    not_modified = client.get("/v1/benchmark", headers={"If-None-Match": f'"x", {etag}'})
    assert not_modified.status_code == 304
    assert not_modified.content == b""
    assert not_modified.headers["etag"] == etag
    assert client.get("/v1/benchmark", headers={"If-None-Match": "*"}).status_code == 304
    assert client.get("/v1/benchmark", headers={"If-None-Match": '"stale"'}).status_code == 200


def test_etag_changes_when_a_source_file_changes(
    bench_copy: tuple[Path, Path], settings: Settings
) -> None:
    bench, audit = bench_copy
    app = create_app(settings.model_copy(update={"benchmark_dir": bench, "audit_dir": audit}))
    with TestClient(app) as client:
        before = client.get("/v1/benchmark")
        path = audit / "audit.json"
        path.write_text(json.dumps({"rows": [{"id": "v1_reported", "mae": 0.5}]}), "utf-8")
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        after = client.get("/v1/benchmark", headers={"If-None-Match": before.headers["etag"]})
        assert after.status_code == 200
        assert after.headers["etag"] != before.headers["etag"]
        assert after.json()["audit"]["rows"][0]["mae"] == 0.5
