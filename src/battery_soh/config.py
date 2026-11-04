"""Project-wide paths and constants.

Every module resolves files through :func:`get_paths` instead of relative
paths or ``sys.path`` tricks, so the code behaves the same from the repo root,
from a notebook, from tests, and inside the Docker image (where
``SOH_PROJECT_ROOT`` is set explicitly).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

#: Global seed used by every stochastic component unless overridden.
RANDOM_SEED = 42

#: Nameplate capacity of the NASA PCoE 18650 cells (Ah).
RATED_CAPACITY_AH = 2.0

#: Number of early valid cycles used for the initial-capacity reference (label)
#: and, in "relative" mode, for the commissioning feature fingerprint.
DEFAULT_N_REFERENCE_CYCLES = 5


def _find_project_root(start: Path) -> Path:
    """Walk upwards from ``start`` until the repository root is found."""
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "src" / "battery_soh"
        ).is_dir():
            return candidate
    return Path.cwd()


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default


@dataclass(frozen=True)
class ProjectPaths:
    """Canonical locations of data, artifacts and reports."""

    root: Path

    @property
    def data_raw(self) -> Path:
        return _env_path("SOH_RAW_DIR", self.root / "data" / "raw")

    @property
    def nasa_raw(self) -> Path:
        return self.data_raw / "nasa"

    @property
    def data_processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def features_csv(self) -> Path:
        return self.data_processed / "nasa_features.csv"

    @property
    def configs(self) -> Path:
        return self.root / "configs"

    @property
    def artifacts(self) -> Path:
        return _env_path("SOH_ARTIFACTS_DIR", self.root / "artifacts")

    @property
    def models_dir(self) -> Path:
        return self.artifacts / "models"

    @property
    def demo_dir(self) -> Path:
        return self.artifacts / "demo"

    @property
    def reports(self) -> Path:
        return _env_path("SOH_REPORTS_DIR", self.root / "reports")

    @property
    def benchmark_dir(self) -> Path:
        return self.reports / "benchmark"

    @property
    def figures_dir(self) -> Path:
        return self.reports / "figures"

    @property
    def docs(self) -> Path:
        return self.root / "docs"

    @property
    def frontend(self) -> Path:
        return _env_path("SOH_FRONTEND_DIR", self.root / "frontend")


@lru_cache(maxsize=1)
def get_paths() -> ProjectPaths:
    """Return the project paths, honouring ``SOH_PROJECT_ROOT`` when set."""
    root_env = os.environ.get("SOH_PROJECT_ROOT")
    root = (
        Path(root_env).resolve()
        if root_env
        else _find_project_root(Path(__file__).resolve().parent)
    )
    return ProjectPaths(root=root)
