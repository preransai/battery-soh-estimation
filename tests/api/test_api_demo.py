"""Demo assets: downsampling, deterministic building, safe loading, and the /v1/demo endpoints."""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from battery_soh.api import demo as demo_module
from battery_soh.api.app import create_app
from battery_soh.api.demo import (
    DEMO_FILE_NAME,
    DemoDataError,
    build_demo_assets,
    demo_document,
    downsample_indices,
    load_demo_store,
    write_demo_assets,
)
from battery_soh.api.schemas import INPUT_FEATURES
from battery_soh.api.settings import Settings
from battery_soh.config import get_paths
from battery_soh.data.protocols import CORE_BATTERIES
from battery_soh.data.schema import BatteryRecord, Segment
from battery_soh.features.extract import features_from_segments


# ---------------------------------------------------------------------- downsampling
@pytest.fixture(scope="module")
def raw_cycle(synthetic_records: dict[str, BatteryRecord]) -> tuple[Segment, Segment]:
    _, charge, discharge = synthetic_records["B0005"].segments[0:3]
    return charge, discharge


@pytest.mark.parametrize("max_points", [16, 60, 120])
def test_downsampling_keeps_landmarks_and_bounds(
    raw_cycle: tuple[Segment, Segment], max_points: int
) -> None:
    for seg in raw_cycle:
        keep = downsample_indices(seg, max_points)
        assert seg.n_samples > max_points
        assert keep.size <= max_points
        assert np.all(np.diff(keep) > 0)
        assert keep[0] == 0
        assert keep[-1] == seg.n_samples - 1
        on = np.flatnonzero(np.abs(seg.current_a) > 0.5)[0]
        assert {on - 1, on} <= set(keep.tolist())  # the load-on / charger-on step


def test_window_is_kept_at_full_resolution_when_it_fits(raw_cycle: tuple[Segment, Segment]) -> None:
    _, discharge = raw_cycle
    v, i = discharge.voltage_v, discharge.current_a
    loaded = np.abs(i) > 0.5
    window = np.flatnonzero(loaded & (v >= 3.45) & (v <= 3.85))
    # The window gets up to 3/4 of the budget; the rest of the record keeps the remainder.
    keep = downsample_indices(discharge, (4 * window.size) // 3 + 20)
    assert keep.size < discharge.n_samples
    assert set(window.tolist()) <= set(keep.tolist())


def test_short_records_are_untouched(raw_cycle: tuple[Segment, Segment]) -> None:
    charge, _ = raw_cycle
    assert downsample_indices(charge, 10_000).tolist() == list(range(charge.n_samples))
    with pytest.raises(ValueError, match="max_points"):
        downsample_indices(charge, 15)


# ---------------------------------------------------------------------- building
@pytest.fixture(scope="module")
def document(synthetic_records: dict[str, BatteryRecord]) -> dict[str, Any]:
    return demo_document([synthetic_records["B0005"]], every=4, max_points=120, n_ref=2)


def test_document_selection_and_content(
    document: dict[str, Any], synthetic_records: dict[str, BatteryRecord]
) -> None:
    (battery,) = document["batteries"]
    assert battery["battery_id"] == "B0005"
    assert battery["n_cycles"] == 9
    cycles = battery["cycles"]
    # first 2 valid (reference) + every 4th valid + the last one
    assert [c["cycle_index"] for c in cycles] == [0, 1, 4, 8]
    assert [c["is_reference"] for c in cycles] == [True, True, False, False]
    segments = synthetic_records["B0005"].segments
    for cycle in cycles:
        k = cycle["cycle_index"]
        impedance, charge, discharge = segments[3 * k : 3 * k + 3]
        expected = features_from_segments(
            charge, discharge, eis=(impedance.re_ohm or 0.0, impedance.rct_ohm or 0.0)
        )
        assert set(cycle["features"]) == set(INPUT_FEATURES)  # no leg_* values
        for name, value in cycle["features"].items():
            if name == "ctx_cycle_index":
                assert value == k
            elif name == "ctx_ambient_c":
                assert value == 24.0
            elif math.isnan(expected[name]):
                assert value is None
            else:
                assert value == pytest.approx(expected[name], rel=1e-12)
        assert len(cycle["charge"]["time_s"]) <= 120
        assert len(cycle["discharge"]["voltage_v"]) <= 120
        assert cycle["eis"] == {"re_ohm": impedance.re_ohm, "rct_ohm": impedance.rct_ohm}
    # Label: 100 * capacity / C0 with C0 = median of the first 5 valid capacities
    # (the label definition does not depend on the demo's n_ref).
    capacities = [s.capacity_ah for s in segments if s.kind == "discharge"]
    c0 = float(np.median(capacities[:5]))
    for cycle in cycles:
        expected_soh = 100.0 * capacities[cycle["cycle_index"]] / c0
        assert cycle["soh_pct"] == pytest.approx(expected_soh, rel=1e-12)
    first_two = [c for c in cycles if c["is_reference"]]
    for name, value in battery["reference"].items():
        values = [c["features"][name] for c in first_two]
        if all(v is not None for v in values):
            assert value == pytest.approx(sum(values) / 2, rel=1e-12)
    assert not any(k.startswith("ctx_") for k in battery["reference"])


def test_output_is_byte_for_byte_deterministic(document: dict[str, Any], tmp_path: Path) -> None:
    first = write_demo_assets(document, tmp_path / "a").read_bytes()
    second = write_demo_assets(document, tmp_path / "b").read_bytes()
    assert first == second
    assert gzip.decompress(first).startswith(b'{"schema_version":"1.0"')


def test_invalid_document_is_not_written(document: dict[str, Any], tmp_path: Path) -> None:
    broken = json.loads(json.dumps(document))
    broken["batteries"][0]["cycles"][0]["charge"]["time_s"] = "oops"
    with pytest.raises(DemoDataError):
        write_demo_assets(broken, tmp_path)
    assert not (tmp_path / DEMO_FILE_NAME).exists()


@pytest.mark.parametrize("kwargs", [{"every": 0}, {"n_ref": 0}, {"max_points": 8}])
def test_invalid_build_arguments(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError, match=next(iter(kwargs))):
        demo_document([], **kwargs)


def test_missing_raw_files_raise(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_demo_assets(tmp_path / "nowhere", tmp_path)
    with pytest.raises(FileNotFoundError, match="B0005"):
        build_demo_assets(tmp_path, tmp_path)


# ---------------------------------------------------------------------- loading
def test_corrupt_assets_raise_demo_data_error(tmp_path: Path) -> None:
    path = tmp_path / DEMO_FILE_NAME
    path.write_bytes(b"not gzip at all")
    with pytest.raises(DemoDataError):
        load_demo_store(path)
    path.write_bytes(gzip.compress(b"{not json"))
    with pytest.raises(DemoDataError):
        load_demo_store(path)
    path.write_bytes(gzip.compress(b'{"batteries": [{"battery_id": "../x", "cycles": []}]}'))
    with pytest.raises(DemoDataError, match="battery id"):
        load_demo_store(path)
    with pytest.raises(FileNotFoundError):
        load_demo_store(tmp_path / "missing.json.gz")


def test_decompression_is_bounded(
    tmp_path: Path, document: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write_demo_assets(document, tmp_path)
    monkeypatch.setattr(demo_module, "MAX_DECOMPRESSED_BYTES", 1000)
    with pytest.raises(DemoDataError, match="too large"):
        load_demo_store(path)


# ---------------------------------------------------------------------- endpoints
def test_batteries_endpoint(client: TestClient) -> None:
    response = client.get("/v1/demo/batteries")
    assert response.status_code == 200
    batteries = response.json()
    assert [b["battery_id"] for b in batteries] == ["B0005", "B0006"]
    for battery in batteries:
        assert set(battery) == {"battery_id", "n_cycles", "reference", "cycles"}
        assert battery["n_cycles"] == 9
        for cycle in battery["cycles"]:
            assert set(cycle) == {"cycle_index", "soh_pct", "is_reference"}
        indices = [c["cycle_index"] for c in battery["cycles"]]
        assert indices == sorted(indices)


def test_cycle_endpoint_and_round_trip_to_estimate_from_cycle(client: TestClient) -> None:
    response = client.get("/v1/demo/batteries/B0006/cycles/4")
    assert response.status_code == 200
    cycle = response.json()
    assert set(cycle) == {
        "battery_id", "cycle_index", "soh_pct", "qc_valid", "is_reference", "features",
        "charge", "discharge", "eis",
    }  # fmt: skip
    assert (cycle["battery_id"], cycle["cycle_index"], cycle["qc_valid"]) == ("B0006", 4, True)
    # The downsampled arrays are a valid estimate-from-cycle body.
    body = {"charge": cycle["charge"], "discharge": cycle["discharge"]}
    estimate = client.post("/v1/soh/estimate-from-cycle", json=body)
    assert estimate.status_code == 200, estimate.text
    extracted = estimate.json()["features"]
    assert extracted["dis_q_380_350_ah"] == pytest.approx(
        cycle["features"]["dis_q_380_350_ah"], rel=0.02
    )
    # ... and estimating from them is close to estimating from the full-resolution features.
    full = {k: v for k, v in cycle["features"].items() if v is not None}
    direct = client.post("/v1/soh/estimate", json={"features": full}).json()
    assert estimate.json()["estimate"]["model_id"] == direct["model_id"]
    assert estimate.json()["estimate"]["soh_pct"] == pytest.approx(direct["soh_pct"], abs=0.5)


@pytest.mark.parametrize(
    ("path", "status"),
    [
        ("/v1/demo/batteries/B0005/cycles/3", 404),
        ("/v1/demo/batteries/B9999/cycles/0", 404),
        ("/v1/demo/batteries/B0005/cycles/-1", 422),
        ("/v1/demo/batteries/B0005/cycles/abc", 422),
        ("/v1/demo/batteries/..%2F..%2Fetc/cycles/0", 404),
    ],
)
def test_unknown_demo_cycles(client: TestClient, path: str, status: int) -> None:
    response = client.get(path)
    assert response.status_code == status
    assert response.headers["content-type"] == "application/problem+json"


def test_demo_endpoints_without_asset_are_404(settings: Settings, tmp_path: Path) -> None:
    with TestClient(create_app(settings.model_copy(update={"demo_dir": tmp_path}))) as client:
        response = client.get("/v1/demo/batteries")
        assert response.status_code == 404
        assert response.json()["type"] == "/problems/demo-unavailable"


def test_corrupt_asset_is_503_without_details(settings: Settings, tmp_path: Path) -> None:
    (tmp_path / DEMO_FILE_NAME).write_bytes(gzip.compress(b'{"batteries": 3}'))
    with TestClient(create_app(settings.model_copy(update={"demo_dir": tmp_path}))) as client:
        response = client.get("/v1/demo/batteries")
        assert response.status_code == 503
        assert str(tmp_path) not in response.text


# ---------------------------------------------------------------------- real data
@pytest.mark.raw_data
def test_build_from_real_nasa_files(nasa_raw_dir: Path, tmp_path: Path) -> None:
    path = build_demo_assets(nasa_raw_dir, tmp_path, battery_ids=["B0018"], every=16)
    assert path.stat().st_size < 400_000
    store = load_demo_store(path)
    (battery,) = store.battery_list()
    assert battery["battery_id"] == "B0018"
    assert battery["n_cycles"] == 132
    refs = [c["cycle_index"] for c in battery["cycles"] if c["is_reference"]]
    assert refs == [0, 1, 2, 3, 4]
    cycle = store.cycle("B0018", battery["cycles"][-1]["cycle_index"])
    assert cycle is not None
    assert len(cycle["charge"]["time_s"]) <= 400
    assert 60.0 < cycle["soh_pct"] < 80.0  # B0018 ends its test near 72 % SOH


def test_null_and_invalid_reference_values(document: dict[str, Any], tmp_path: Path) -> None:
    doc = json.loads(json.dumps(document))
    doc["batteries"][0]["reference"]["dis_r0_ohm"] = None
    store = load_demo_store(write_demo_assets(doc, tmp_path))
    assert store.battery_list()[0]["reference"]["dis_r0_ohm"] is None
    doc["batteries"][0]["reference"]["dis_r0_ohm"] = "0.1"
    with pytest.raises(DemoDataError, match="reference"):
        write_demo_assets(doc, tmp_path / "bad")


def test_demo_features_equal_the_committed_feature_table() -> None:
    """Shipped demo values == data/processed/nasa_features.csv for the same (cell, cycle).

    Needs only the two committed artifacts (no raw files), so it also guards against
    a demo asset that is stale after a feature-extractor change.
    """
    paths = get_paths()
    asset = paths.demo_dir / DEMO_FILE_NAME
    if not asset.is_file() or not paths.features_csv.is_file():
        pytest.skip("demo asset or feature table not built")
    table = pd.read_csv(paths.features_csv).set_index(["battery_id", "cycle_index"])
    store = load_demo_store(asset)
    assert [b["battery_id"] for b in store.battery_list()] == list(CORE_BATTERIES)
    compared = 0
    for battery in store.battery_list():
        for entry in battery["cycles"]:
            cycle = store.cycle(battery["battery_id"], entry["cycle_index"])
            assert cycle is not None
            row = table.loc[(battery["battery_id"], entry["cycle_index"])]
            assert cycle["soh_pct"] == pytest.approx(row["soh_pct"], rel=1e-5)
            for name, value in cycle["features"].items():
                if value is None:
                    assert math.isnan(row[name]), name
                else:  # the CSV stores 6 significant digits
                    assert value == pytest.approx(row[name], rel=1e-5, abs=1e-9), name
                compared += 1
    assert compared > 1000


@pytest.mark.parametrize("path", ["/v1/demo/batteries", "/v1/demo/batteries/B0006/cycles/4"])
def test_demo_responses_are_cacheable_and_conditional(client: TestClient, path: str) -> None:
    first = client.get(path)
    assert first.status_code == 200
    assert first.headers["cache-control"] == "public, max-age=300"
    etag = first.headers["etag"]
    assert etag.startswith('W/"')
    assert client.get(path).content == first.content  # served from the serialized cache
    again = client.get(path, headers={"If-None-Match": etag})
    assert again.status_code == 304
    assert again.content == b""
    assert again.headers["etag"] == etag
    assert again.headers["x-request-id"]
    assert client.get(path, headers={"If-None-Match": 'W/"other"'}).status_code == 200


def test_demo_store_accessors_return_copies(demo_dir: Path) -> None:
    store = demo_module.get_demo_store(demo_dir)
    assert store is not None
    batteries = store.battery_list()
    batteries[0]["cycles"].clear()
    assert store.battery_list()[0]["cycles"]
    cycle = store.cycle("B0005", 0)
    assert cycle is not None
    cycle["features"].clear()
    assert store.cycle("B0005", 0)["features"]  # type: ignore[index]
    assert json.loads(store.battery_list_json().body) == store.battery_list()
