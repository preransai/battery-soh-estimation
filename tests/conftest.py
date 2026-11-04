"""Shared pytest fixtures and hooks.

* Synthetic cycling data with exact ground truth (:mod:`battery_soh.synthetic`),
  simulated once per session because the ODE integration is the slow part.
* ``raw_data`` marker handling: tests that need the NASA ``.mat`` files are
  skipped automatically when ``get_paths().nasa_raw`` holds none.
* One quick benchmark run and one leakage audit on the committed feature table,
  shared by the integration tests and the CLI tests (outputs are read, never modified).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from battery_soh.config import get_paths
from battery_soh.data.schema import Segment
from battery_soh.synthetic import (
    SENSOR_NOISE,
    CellModel,
    SyntheticSegment,
    simulate_charge,
    simulate_discharge,
)

#: Sampling intervals close to the NASA logs (charge ~10.7 s, discharge ~18.7 s).
CHARGE_SAMPLE_S = 10.0
DISCHARGE_SAMPLE_S = 10.0


def _has_raw_nasa_data() -> bool:
    root = get_paths().nasa_raw
    return root.is_dir() and any(root.rglob("*.mat"))


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip ``@pytest.mark.raw_data`` tests when the raw NASA files are absent."""
    del config
    raw_items = [item for item in items if item.get_closest_marker("raw_data")]
    if raw_items and not _has_raw_nasa_data():
        skip = pytest.mark.skip(reason=f"no NASA .mat files under {get_paths().nasa_raw}")
        for item in raw_items:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def nasa_raw_dir() -> Path:
    """Directory holding the raw NASA ``.mat`` files (tests using it need ``raw_data``)."""
    return get_paths().nasa_raw


@pytest.fixture(scope="session")
def quick_benchmark_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """``run_benchmark(<committed config>, out, quick=True)``; outputs in ``out / "quick"``."""
    from battery_soh.evaluation.benchmark import run_benchmark

    out = tmp_path_factory.mktemp("bench")
    run_benchmark(get_paths().configs / "benchmark.yaml", out, quick=True)
    return out


@pytest.fixture(scope="session")
def audit_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """``run_audit`` on the committed feature table (``audit.json`` and ``audit_table.csv``)."""
    from battery_soh.evaluation.audit import run_audit

    out = tmp_path_factory.mktemp("audit")
    run_audit(get_paths().features_csv, out)
    return out


@pytest.fixture(scope="session")
def fresh_cell() -> CellModel:
    """Default synthetic 2.0 Ah cell (R0 = 70 mOhm, no RC branch)."""
    return CellModel()


@pytest.fixture(scope="session")
def rc_cell() -> CellModel:
    """Cell with an RC polarisation branch (R1 = 30 mOhm, tau = 20 s)."""
    return CellModel(r1_ohm=0.03, tau1_s=20.0)


@pytest.fixture(scope="session")
def charge_run(fresh_cell: CellModel) -> SyntheticSegment:
    """Noise-free NASA-protocol CC-CV charge of the fresh cell."""
    return simulate_charge(fresh_cell, sample_s=CHARGE_SAMPLE_S)


@pytest.fixture(scope="session")
def discharge_run(fresh_cell: CellModel) -> SyntheticSegment:
    """Noise-free 2 A discharge to 2.7 V of the fresh cell, load on at a sample instant."""
    return simulate_discharge(fresh_cell, sample_s=DISCHARGE_SAMPLE_S)


@pytest.fixture(scope="session")
def noisy_charge_run(fresh_cell: CellModel) -> SyntheticSegment:
    """CC-CV charge with realistic sensor noise (seeded)."""
    return simulate_charge(fresh_cell, sample_s=CHARGE_SAMPLE_S, noise=SENSOR_NOISE, seed=7)


@pytest.fixture(scope="session")
def noisy_discharge_run(fresh_cell: CellModel) -> SyntheticSegment:
    """2 A discharge with realistic sensor noise (seeded)."""
    return simulate_discharge(fresh_cell, sample_s=DISCHARGE_SAMPLE_S, noise=SENSOR_NOISE, seed=7)


@pytest.fixture(scope="session")
def ageing_runs(
    fresh_cell: CellModel,
) -> list[tuple[CellModel, SyntheticSegment, SyntheticSegment]]:
    """Noise-free ``(cell, charge, discharge)`` for SOH 100 % down to 70 % in 2 % steps."""
    runs = []
    for k in range(16):
        cell = fresh_cell.aged(1.0 - 0.02 * k)
        runs.append(
            (
                cell,
                simulate_charge(cell, sample_s=CHARGE_SAMPLE_S),
                simulate_discharge(cell, sample_s=DISCHARGE_SAMPLE_S),
            )
        )
    return runs


def add_sensor_noise(seg: Segment, seed: int) -> Segment:
    """Return ``seg`` with :data:`SENSOR_NOISE` added (seeded), without re-simulating."""
    rng = np.random.default_rng(seed)
    n = seg.n_samples
    return dataclasses.replace(
        seg,
        voltage_v=seg.voltage_v + rng.normal(0.0, SENSOR_NOISE.voltage_v, n),
        current_a=seg.current_a + rng.normal(0.0, SENSOR_NOISE.current_a, n),
        temperature_c=seg.temperature_c + rng.normal(0.0, SENSOR_NOISE.temperature_c, n),
    )


@pytest.fixture(scope="session")
def noisy() -> Callable[[Segment, int], Segment]:
    """Fixture form of :func:`add_sensor_noise` (test modules cannot import conftest)."""
    return add_sensor_noise
