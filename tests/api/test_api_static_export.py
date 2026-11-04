"""Static-site export: same JSON as the live API, safe file handling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from battery_soh.api.static_export import MARKER_FILE, STATIC_CONFIG_JS, export_static_site
from battery_soh.config import ProjectPaths


class FakePaths(ProjectPaths):
    """ProjectPaths whose locations point at the test fixtures."""

    def __init__(self, root: Path, **locations: Path) -> None:
        super().__init__(root=root)
        object.__setattr__(self, "_locations", locations)

    def __getattribute__(self, name: str) -> Any:
        locations = object.__getattribute__(self, "_locations")
        if name in locations:
            return locations[name]
        return object.__getattribute__(self, name)


@pytest.fixture
def paths(
    tmp_path: Path,
    models_dir: Path,
    demo_dir: Path,
    reports_dirs: tuple[Path, Path],
    frontend_dir: Path,
) -> FakePaths:
    bench, audit = reports_dirs
    return FakePaths(
        tmp_path / "project",
        frontend=frontend_dir,
        models_dir=models_dir,
        demo_dir=demo_dir,
        benchmark_dir=bench,
        reports=audit.parent,
    )


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def test_export_mirrors_the_api(paths: FakePaths, tmp_path: Path, client: TestClient) -> None:
    out = export_static_site(tmp_path / "site", paths)
    assert (out / "index.html").is_file()
    assert (out / "assets" / "js" / "app.js").is_file()
    config = (out / "assets" / "config.js").read_text(encoding="utf-8")
    assert config == STATIC_CONFIG_JS
    assert config == 'window.SOH_CONFIG = Object.freeze({ mode: "static", dataBase: "data/" });\n'
    assert not (out / ".secret").exists()  # hidden files are never published
    assert (out / MARKER_FILE).is_file()
    data = out / "data"

    assert read(data / "models.json") == client.get("/v1/models").json()
    for model in read(data / "models.json"):
        bundle = read(data / "models" / f"{model['model_id']}.json")
        assert "precision_matrix" in bundle["domain"]  # the browser needs it for inference
        bundle["domain"].pop("precision_matrix")
        assert bundle == client.get(f"/v1/models/{model['model_id']}").json()

    batteries = read(data / "demo" / "batteries.json")
    assert batteries == client.get("/v1/demo/batteries").json()
    n_files = 0
    for battery in batteries:
        for cycle in battery["cycles"]:
            path = data / "demo" / battery["battery_id"] / f"{cycle['cycle_index']}.json"
            api = client.get(
                f"/v1/demo/batteries/{battery['battery_id']}/cycles/{cycle['cycle_index']}"
            )
            assert read(path) == api.json()
            n_files += 1
    assert n_files == len(list((data / "demo").rglob("*.json"))) - 1

    assert read(data / "benchmark.json") == client.get("/v1/benchmark").json()


def test_missing_optional_inputs_still_export(paths: FakePaths, tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    sparse = FakePaths(
        paths.root,
        frontend=paths.frontend,
        models_dir=empty,
        demo_dir=empty,
        benchmark_dir=empty,
        reports=empty,
    )
    out = export_static_site(tmp_path / "site", sparse)
    assert read(out / "data" / "models.json") == []
    assert not (out / "data" / "demo").exists()
    assert not (out / "data" / "benchmark.json").exists()


def test_reexport_replaces_a_previous_export(paths: FakePaths, tmp_path: Path) -> None:
    out = export_static_site(tmp_path / "site", paths)
    (out / "stale.txt").write_text("old", encoding="utf-8")
    export_static_site(tmp_path / "site", paths)
    assert not (out / "stale.txt").exists()


def test_refuses_to_overwrite_foreign_directories(paths: FakePaths, tmp_path: Path) -> None:
    foreign = tmp_path / "important"
    foreign.mkdir()
    (foreign / "keep.txt").write_text("precious", encoding="utf-8")
    with pytest.raises(ValueError, match="not created by"):
        export_static_site(foreign, paths)
    assert (foreign / "keep.txt").read_text(encoding="utf-8") == "precious"


def test_refuses_targets_overlapping_the_frontend(paths: FakePaths) -> None:
    with pytest.raises(ValueError, match="frontend"):
        export_static_site(paths.frontend / "site", paths)
    with pytest.raises(ValueError, match="frontend"):
        export_static_site(paths.frontend.parent, paths)
    with pytest.raises(ValueError, match="refusing"):
        export_static_site(paths.root, paths)


def test_missing_frontend_raises(paths: FakePaths, tmp_path: Path) -> None:
    broken = FakePaths(paths.root, frontend=tmp_path / "nope")
    with pytest.raises(FileNotFoundError):
        export_static_site(tmp_path / "site", broken)


def test_symlinks_are_not_followed(paths: FakePaths, tmp_path: Path) -> None:
    secret = tmp_path / "outside.txt"
    secret.write_text("outside the frontend", encoding="utf-8")
    link = paths.frontend / "assets" / "linked.txt"
    try:
        link.symlink_to(secret)
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links are not available")
    try:
        out = export_static_site(tmp_path / "site", paths)
        assert not (out / "assets" / "linked.txt").exists()
    finally:
        link.unlink()
