"""Export the web UI plus precomputed JSON as a backend-free static site.

``soh export-static --out-dir site`` produces a folder that GitHub Pages (or any
static host) can serve. The web UI switches to *static mode*
(``assets/config.js``) and reads the precomputed responses under ``data/``::

    data/models.json                    = GET /v1/models
    data/models/<model_id>.json         = the full bundle (the browser runs the
                                          numpy-free linear model from it, so it
                                          includes the domain precision matrix
                                          that GET /v1/models/<id> omits)
    data/demo/batteries.json            = GET /v1/demo/batteries
    data/demo/<battery>/<cycle>.json    = GET /v1/demo/batteries/<battery>/cycles/<cycle>
    data/benchmark.json                 = GET /v1/benchmark

Every document comes from the same payload function the API endpoint returns,
so the shapes cannot drift. Missing inputs (no models, no demo asset, no
benchmark yet) are skipped with a warning; the export still succeeds.

Safety: only regular files inside the frontend folder are copied (symbolic
links and hidden files are skipped), every output path is checked to stay
inside the output folder, and an existing output folder is only replaced when
it was produced by this exporter (marker file), never an arbitrary directory.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any

from battery_soh import __version__
from battery_soh.api.demo import DemoDataError, get_demo_store
from battery_soh.api.errors import ProblemError
from battery_soh.api.registry import ModelRegistry
from battery_soh.api.routers.benchmark import benchmark_payload
from battery_soh.api.routers.demo import batteries_payload, cycle_payload
from battery_soh.api.routers.models import list_models_payload
from battery_soh.config import ProjectPaths

__all__ = ["MARKER_FILE", "STATIC_CONFIG_JS", "export_static_site"]

logger = logging.getLogger(__name__)

#: Marks a folder as produced by :func:`export_static_site` (safe to replace).
MARKER_FILE = ".soh-static-export"

#: Exact content of ``assets/config.js`` in the exported site (the web UI's contract).
STATIC_CONFIG_JS = 'window.SOH_CONFIG = Object.freeze({ mode: "static", dataBase: "data/" });\n'


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _check_target(out_dir: Path, frontend: Path, project_root: Path) -> None:
    forbidden = {Path(out_dir.anchor), Path.home().resolve(), project_root.resolve()}
    if out_dir in forbidden:
        raise ValueError(f"refusing to export into {out_dir}")
    if _is_within(frontend, out_dir) or _is_within(out_dir, frontend):
        raise ValueError("the output folder must not contain or be inside the frontend folder")


def _prepare(out_dir: Path) -> None:
    if out_dir.exists():
        if not out_dir.is_dir():
            raise ValueError(f"{out_dir} exists and is not a directory")
        if any(out_dir.iterdir()):
            if not (out_dir / MARKER_FILE).is_file():
                raise ValueError(
                    f"{out_dir} is not empty and was not created by `soh export-static`; "
                    "choose another folder or remove it"
                )
            shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def _copy_frontend(frontend: Path, out_dir: Path) -> int:
    copied = 0
    for dirpath, dirnames, filenames in os.walk(frontend, followlinks=False):
        current = Path(dirpath)
        dirnames[:] = sorted(
            d
            for d in dirnames
            if not d.startswith(".") and d != "__pycache__" and not (current / d).is_symlink()
        )
        for name in sorted(filenames):
            source = current / name
            if name.startswith(".") or source.is_symlink() or not source.is_file():
                continue
            target = out_dir / source.relative_to(frontend)
            if not _is_within(target.parent, out_dir):  # pragma: no cover - defensive
                raise ValueError(f"unsafe path {source}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            copied += 1
    return copied


def _write_json(root: Path, relative: str, document: Any) -> Path:
    target = root / relative
    if not _is_within(target.parent, root) or target.name in ("", ".", ".."):
        raise ValueError(f"unsafe output path {relative!r}")
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(document, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    target.write_text(text, encoding="utf-8")
    return target


def export_static_site(out_dir: Path, paths: ProjectPaths) -> Path:
    """Write the static site.

    Args:
        out_dir: Output folder (created; replaced only if a previous export made it).
        paths: Project locations (frontend, models, demo data, reports).

    Returns:
        The resolved output folder.

    Raises:
        FileNotFoundError: If the frontend folder has no ``index.html``.
        ValueError: If ``out_dir`` is unsafe to write (see the module docstring).
    """
    frontend = paths.frontend.resolve()
    if not (frontend / "index.html").is_file():
        raise FileNotFoundError(f"no web UI (index.html) in {frontend}")
    out = Path(out_dir).resolve()
    _check_target(out, frontend, paths.root)
    _prepare(out)
    n_files = _copy_frontend(frontend, out)
    (out / "assets").mkdir(exist_ok=True)
    (out / "assets" / "config.js").write_text(STATIC_CONFIG_JS, encoding="utf-8")
    data = out / "data"

    registry = ModelRegistry.load(paths.models_dir)
    if not registry:
        logger.warning(
            "no verified model bundle (%s); the static site has no models", registry.error
        )
    _write_json(data, "models.json", list_models_payload(registry))
    for entry in registry.entries:
        _write_json(data, f"models/{entry.model_id}.json", entry.bundle.to_dict())

    n_cycles = 0
    try:
        store = get_demo_store(paths.demo_dir)
    except DemoDataError as exc:
        logger.error("demo asset is invalid, skipping the demo data: %s", exc)
        store = None
    if store is None:
        logger.warning("no demo data in %s; the static site has no demo cycles", paths.demo_dir)
    else:
        batteries = batteries_payload(store)
        _write_json(data, "demo/batteries.json", batteries)
        for battery in batteries:
            for cycle in battery["cycles"]:
                battery_id, index = battery["battery_id"], int(cycle["cycle_index"])
                document = cycle_payload(store, battery_id, index)
                _write_json(data, f"demo/{battery_id}/{index}.json", document)
                n_cycles += 1

    try:
        _write_json(
            data, "benchmark.json", benchmark_payload(paths.benchmark_dir, paths.reports / "audit")
        )
    except ProblemError:
        logger.warning("no benchmark summary in %s; skipping benchmark.json", paths.benchmark_dir)

    (out / ".nojekyll").write_text("", encoding="utf-8")
    (out / MARKER_FILE).write_text(f"battery-soh {__version__}\n", encoding="utf-8")
    logger.info(
        "static site: %d UI file(s), %d model(s), %d demo cycle(s) in %s",
        n_files,
        len(registry),
        n_cycles,
        out,
    )
    return out
