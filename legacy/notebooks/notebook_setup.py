
"""
Portable notebook bootstrap for Windows, macOS, and Linux.

Usage (first cell of each notebook):
    import importlib.util
    from pathlib import Path
    # ... load this module via spec_from_file_location, then:
    PROJECT_ROOT = notebook_setup.setup()

Requires: run Jupyter with current working directory either the repo root or the
notebooks/ folder (both are supported).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Any of these next to src/ identifies the project root after unzipping.
_ROOT_MARKERS = ("config.yaml", "requirements.txt", "phase2_config.yaml")


def find_project_root(start: Path | None = None) -> Path:
    """Walk upward from cwd (or start) until we find src/ and a known marker file."""
    start = (start or Path.cwd()).resolve()
    for p in [start, *start.parents][:14]:
        src = p / "src"
        if not src.is_dir():
            continue
        if any((p / name).exists() for name in _ROOT_MARKERS):
            return p
    raise FileNotFoundError(
        "Could not find project root (expected a folder containing 'src/' and "
        "config.yaml, requirements.txt, or phase2_config.yaml).\n"
        "Fix: open a terminal, cd into the unzipped project folder, then run:\n"
        "  jupyter notebook\n"
        "or start Jupyter from the repo root (not from a random directory)."
    )


def setup(*, with_api: bool = False) -> Path:
    """
    chdir to project root and extend sys.path for src (and optional api).

    Parameters
    ----------
    with_api
        If True, also prepend ``api/`` (for FastAPI client tests in notebooks).
    """
    root = find_project_root()
    os.chdir(root)

    for sub in ("src", "src/phase2", "src/phase3", "src/phase4"):
        p = root / sub
        if p.is_dir():
            s = str(p)
            if s not in sys.path:
                sys.path.insert(0, s)

    if with_api:
        ap = root / "api"
        if ap.is_dir():
            s = str(ap)
            if s not in sys.path:
                sys.path.insert(0, s)

    return root
