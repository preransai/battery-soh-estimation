"""Leave-one-battery-out benchmark with battery-grouped conformal intervals.

``run_benchmark`` reads ``configs/benchmark.yaml`` and the committed feature
table, then runs every configured experiment:

* ``core_lobo`` - the four 2 A / 24 degC reference cells, every feature set x
  mode x model, plus baselines. Holds the **pre-specified headline**
  (``combined`` / ``absolute`` / ``ridge_nfs``: Ridge with nested feature
  selection, :mod:`battery_soh.evaluation.nested`), fixed in the config
  before any ``ridge_nfs`` result was computed (plain-ridge results for the
  same feature set and mode were already known) and reported whatever it
  shows.
* ``extended_lobo`` - twelve usable cells in three operating conditions.
* ``leave_one_condition_out`` - the extended cells, holding out a whole
  operating condition at a time.
* ``window_sweep`` - sensitivity of the partial-window health indicators to
  the voltage window (needs the raw ``.mat`` files; the committed JSON is
  reused when they are absent).

Protocol rules enforced here (each is a leakage rule):

* Every split holds out whole groups (batteries or conditions);
  :func:`~battery_soh.evaluation.splits.assert_disjoint` and
  :func:`~battery_soh.evaluation.splits.assert_groups_disjoint` run on every
  fold.
* Every feature set entering a model passes
  :func:`~battery_soh.evaluation.leakage.check_no_leakage`; leaky sets exist
  only as the labelled ``capacity_oracle`` baseline.
* Feature selection and the ridge penalty of ``ridge_nfs`` are chosen inside
  each outer training fold; conformal calibration uses leave-one-group-out
  residuals of the *training* groups only
  (:class:`~battery_soh.uncertainty.conformal.GroupConformalRegressor`).
* Relative mode divides each health indicator by the cell's own mean over its
  first ``n_reference_cycles`` QC-valid measurements. Those reference cycles
  are never used for training, calibration or scoring (they are predicted
  for display only and flagged ``is_reference_cycle``).
* Rows: QC-valid cycles with a finite label on which every feature set of the
  experiment has at least one measured (non-context) indicator, so all
  feature sets and models of an experiment are scored on the same cycles.

Outputs (``out_dir``; quick mode writes to ``out_dir/quick``):
``results.csv`` (one row per experiment x feature set x mode x model x
held-out battery), ``predictions.csv`` (per-cycle predictions of the
configured specs), ``summary.json`` (headline, baselines, leaderboard,
per-battery and coverage tables, selection stability, extended, LOCO and
window-sweep results, provenance) and ``window_sweep.json``. Apart from
``generated_utc`` and the run time, reruns produce identical files.

Modules (the public names below are re-exported here and loaded on first use,
so importing the config-hash helpers does not import scikit-learn):

* :mod:`.hashing` - scoped config hashes (YAML and JSON only);
* :mod:`.config` - parsing and validation of the YAML;
* :mod:`.frames` - evaluation rows, specs and the leakage gate;
* :mod:`.fitting` - outer-fold tasks, conformal wrapping, per-spec metrics;
* :mod:`.report` - ``summary.json`` blocks and the output files;
* :mod:`.window_sweep` - the raw-data window sensitivity analysis;
* :mod:`.runner` - :func:`run_benchmark` and :func:`evaluate_specs`.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from battery_soh.evaluation.benchmark.config import (
        CONFORMAL_METHODS,
        NESTED_MODEL,
        BenchmarkConfig,
        ExperimentConfig,
        SpecKey,
        WindowSweepConfig,
        load_config,
    )
    from battery_soh.evaluation.benchmark.hashing import (
        BENCHMARK_HASH_SECTIONS,
        PRODUCTION_HASH_SECTIONS,
        UNHASHED_SECTIONS,
        config_hashes,
        config_sections_sha256,
    )
    from battery_soh.evaluation.benchmark.report import SCHEMA_VERSION
    from battery_soh.evaluation.benchmark.runner import evaluate_specs, run_benchmark
    from battery_soh.evaluation.benchmark.window_sweep import run_window_sweep

__all__ = [
    "BENCHMARK_HASH_SECTIONS",
    "CONFORMAL_METHODS",
    "NESTED_MODEL",
    "PRODUCTION_HASH_SECTIONS",
    "SCHEMA_VERSION",
    "UNHASHED_SECTIONS",
    "BenchmarkConfig",
    "ExperimentConfig",
    "SpecKey",
    "WindowSweepConfig",
    "config_hashes",
    "config_sections_sha256",
    "evaluate_specs",
    "load_config",
    "run_benchmark",
    "run_window_sweep",
]

#: Submodule defining each public name.
_SOURCES: dict[str, str] = {
    "BENCHMARK_HASH_SECTIONS": "hashing",
    "PRODUCTION_HASH_SECTIONS": "hashing",
    "UNHASHED_SECTIONS": "hashing",
    "config_hashes": "hashing",
    "config_sections_sha256": "hashing",
    "CONFORMAL_METHODS": "config",
    "NESTED_MODEL": "config",
    "BenchmarkConfig": "config",
    "ExperimentConfig": "config",
    "SpecKey": "config",
    "WindowSweepConfig": "config",
    "load_config": "config",
    "SCHEMA_VERSION": "report",
    "evaluate_specs": "runner",
    "run_benchmark": "runner",
    "run_window_sweep": "window_sweep",
}


def __getattr__(name: str) -> Any:
    """Import the submodule defining ``name`` on first access (PEP 562)."""
    source = _SOURCES.get(name)
    if source is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{__name__}.{source}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
