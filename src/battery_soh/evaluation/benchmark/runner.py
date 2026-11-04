"""Entry points: :func:`run_benchmark` (``soh benchmark``) and :func:`evaluate_specs` (audit)."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from battery_soh.evaluation.benchmark.config import (
    REPORTED_CONFIDENCE,
    ExperimentConfig,
    SpecKey,
    check_in_grid,
    in_grid,
    load_config,
    percent_label,
)
from battery_soh.evaluation.benchmark.fitting import run_experiments
from battery_soh.evaluation.benchmark.frames import leakage_checks, prepare_frames
from battery_soh.evaluation.benchmark.report import (
    build_summary,
    find_result,
    prediction_rows,
    public_row,
    results_frame,
    write_csv,
    write_json,
)
from battery_soh.evaluation.benchmark.window_sweep import window_sweep_section
from battery_soh.features.build import load_feature_table

__all__ = ["evaluate_specs", "run_benchmark"]

logger = logging.getLogger(__name__)


def evaluate_specs(
    config_path: Path,
    keys: Sequence[SpecKey],
    *,
    table: pd.DataFrame | None = None,
    intervals: bool = False,
) -> dict[str, dict[str, Any]]:
    """Evaluate selected specs of the benchmark grid in-process (used by the leakage audit).

    Rows, folds and procedures are those of :func:`run_benchmark`, so the
    returned metrics equal the corresponding ``summary.json`` / ``results.csv``
    numbers. Without ``intervals`` only point predictions are computed, which
    gives the same point predictions (the conformal wrapper's final model is
    fitted on the same rows).

    Args:
        config_path: Benchmark YAML.
        keys: Specs to evaluate; each must be part of the configured grid.
        table: Feature table (default: the config's ``features_csv``).
        intervals: Also compute conformal intervals.

    Returns:
        ``{spec_id: summary_row}`` with the leaderboard row layout.

    Raises:
        ValueError: If a spec is not in the configured grid.
    """
    cfg = load_config(Path(config_path))
    data = load_feature_table(cfg.features_csv) if table is None else table
    by_experiment: dict[str, list[SpecKey]] = {}
    for key in keys:
        check_in_grid(key, cfg.experiments, "spec")
        by_experiment.setdefault(key.experiment, []).append(key)
    out: dict[str, dict[str, Any]] = {}
    for name, wanted in by_experiment.items():
        base = cfg.experiments[name]
        sets = tuple(dict.fromkeys(k.feature_set for k in wanted if k.feature_set != "baseline"))
        models = tuple(dict.fromkeys(k.model for k in wanted if k.model not in base.baselines))
        baselines = tuple(dict.fromkeys(k.model for k in wanted if k.model in base.baselines))
        sub = replace(
            base,
            feature_sets=sets,
            post_hoc_feature_sets=base.post_hoc_feature_sets & frozenset(sets),
            modes=tuple(dict.fromkeys(k.mode for k in wanted)),
            models=models,
            baselines=baselines,
            interval_models=frozenset((*models, *baselines)) if intervals else frozenset(),
            row_filter_sets=base.row_sets,
        )
        frames = {name: prepare_frames(data, sub, cfg.label, cfg.n_reference_cycles)}
        leakage_checks(frames, {name: sub}, cfg.label)
        results = run_experiments(frames, {name: sub}, cfg, n_jobs=1)
        for res in results[name]:
            if res.spec.key in wanted:
                out[res.spec.key.id] = public_row(res.summary, cfg)
    return out


def run_benchmark(config_path: Path, out_dir: Path, *, quick: bool = False) -> dict[str, Any]:
    """Run the configured benchmark; write ``results.csv``, ``predictions.csv``, ``summary.json``.

    Args:
        config_path: ``configs/benchmark.yaml``.
        out_dir: Output directory (``reports/benchmark``). Quick mode writes to
            ``out_dir/<quick.output_subdir>`` (default ``quick``) so it never
            overwrites the committed full results.
        quick: Smoke-test subset: the headline experiment's cells, the
            headline spec, plain ridge and the baselines, single process.

    Returns:
        The summary dict (also written to ``summary.json``).

    Raises:
        FileNotFoundError: If the config or feature table is missing.
        ValueError: If the config is invalid.
        battery_soh.evaluation.leakage.LeakageError: If a feature set fails the
            leakage checks or a split overlaps.
    """
    start = time.perf_counter()
    cfg = load_config(Path(config_path))
    out_dir = Path(out_dir)
    target = out_dir / cfg.quick_subdir if quick else out_dir
    table = load_feature_table(cfg.features_csv)
    if quick:
        experiments: dict[str, ExperimentConfig] = {cfg.headline.experiment: cfg.quick_experiment}
        n_jobs = 1
        secondary = cfg.quick_secondary
    else:
        experiments = dict(cfg.experiments)
        n_jobs = cfg.n_jobs
        secondary = cfg.secondary
    logger.info("benchmark (%s): %s -> %s", "quick" if quick else "full", list(experiments), target)

    frames = {
        name: prepare_frames(table, exp, cfg.label, cfg.n_reference_cycles)
        for name, exp in experiments.items()
    }
    for name, fr in frames.items():
        logger.info(
            "%s: %d QC-valid cycles from %d batteries (%d relative-mode reference cycles)",
            name,
            len(fr.absolute),
            fr.absolute["battery_id"].nunique(),
            int(fr.union_reference.sum()),
        )
    # Refuse to fit anything on a feature set that fails the leakage checks.
    leakage = leakage_checks(frames, experiments, cfg.label)
    results = run_experiments(frames, experiments, cfg, n_jobs)
    window_sweep = (
        {"status": "skipped", "reason": "quick mode"}
        if quick
        else window_sweep_section(cfg, table, target)
    )

    prediction_frames = [
        prediction_rows(find_result(results[s.experiment], s), cfg)
        for s in cfg.prediction_specs
        if s.experiment in experiments and in_grid(s, experiments)
    ]
    write_csv(target / "results.csv", results_frame(results, cfg))
    write_csv(target / "predictions.csv", pd.concat(prediction_frames, ignore_index=True))
    runtime = time.perf_counter() - start
    summary = build_summary(
        cfg,
        experiments,
        frames,
        results,
        leakage,
        window_sweep,
        secondary,
        quick=quick,
        runtime_s=runtime,
        n_jobs=n_jobs,
    )
    write_json(target / "summary.json", summary)
    head = summary["headline"]
    level = percent_label(REPORTED_CONFIDENCE)
    logger.info(
        "headline %s/%s/%s: MAE %.3f, coverage@%s %.3f (%.1f s)",
        head["feature_set"],
        head["mode"],
        head["model"],
        head["mae"],
        level,
        head.get(f"coverage_{level}") or math.nan,
        runtime,
    )
    return summary
