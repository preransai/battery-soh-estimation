"""Outer-fold fitting: one task per (spec, held-out group), run serially or in worker processes.

Each task fits a spec on the training groups of one outer fold and predicts
the held-out group. Specs with intervals are wrapped in
:class:`~battery_soh.uncertainty.conformal.GroupConformalRegressor`, which
calibrates by leave-one-group-out residuals of the *training* groups only.
Train/test disjointness is asserted on every fold, and relative-mode
reference cycles are never fitted on. The per-fold results are then assembled
into out-of-fold predictions and per-battery and per-spec metrics.
"""

from __future__ import annotations

import logging
import math
import time
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.parallel import Parallel, delayed

from battery_soh._typing import FloatArray
from battery_soh.evaluation.baselines import make_baseline
from battery_soh.evaluation.benchmark.config import (
    CONFORMAL_METHODS,
    NESTED_MODEL,
    BenchmarkConfig,
    ExperimentConfig,
    percent_label,
)
from battery_soh.evaluation.benchmark.frames import Frames, Spec, build_specs, spec_frame
from battery_soh.evaluation.metrics import interval_metrics, regression_metrics
from battery_soh.evaluation.nested import GROUP_COLUMN, NestedSelectionRidge
from battery_soh.evaluation.splits import assert_disjoint, assert_groups_disjoint, row_keys
from battery_soh.features.relative import REFERENCE_FLAG_COLUMN
from battery_soh.models.zoo import make_model
from battery_soh.uncertainty.conformal import GroupConformalRegressor

__all__ = ["SpecResult", "run_experiments"]

logger = logging.getLogger(__name__)


# Rough relative cost of one fit, used only to schedule slow tasks first.
_MODEL_COST: Final[Mapping[str, float]] = {
    "gpr": 40.0,
    "mlp": 6.0,
    "random_forest": 5.0,
    "hgb": 1.5,
    NESTED_MODEL: 0.5,
}


@dataclass(frozen=True)
class _Task:
    spec_index: int
    fold: str
    model: str
    kind: str
    seed: int
    nested: tuple[tuple[float, ...], int, float]
    X_train: pd.DataFrame
    y_train: FloatArray
    cal_groups: NDArray[np.object_]
    X_test: pd.DataFrame
    test_positions: NDArray[np.intp]
    intervals: bool
    alphas: tuple[float, ...]
    cost: float


@dataclass(frozen=True)
class _TaskResult:
    spec_index: int
    fold: str
    test_positions: NDArray[np.intp]
    yhat: FloatArray
    bounds: dict[str, tuple[FloatArray, FloatArray]]
    selection: dict[str, Any] | None
    n_train: int
    n_calibration: int
    convergence_warnings: int
    seconds: float


def _factory(model: str, kind: str, seed: int, nested: tuple[tuple[float, ...], int, float]) -> Any:
    if kind == "baseline":
        return partial(make_baseline, model, random_state=seed)
    if model == NESTED_MODEL:
        alphas, min_features, tol = nested
        return partial(NestedSelectionRidge, alphas=alphas, min_features=min_features, tol=tol)
    return partial(make_model, model, random_state=seed)


def _level(alpha: float) -> str:
    """``0.1 -> "90"``: percent label of the central interval."""
    return percent_label(1.0 - alpha)


def _run_task(task: _Task) -> _TaskResult:
    """Fit one outer fold of one spec and predict its held-out rows (worker entry point)."""
    started = time.perf_counter()
    factory = _factory(task.model, task.kind, task.seed, task.nested)
    bounds: dict[str, tuple[FloatArray, FloatArray]] = {}
    with warnings.catch_warnings(record=True) as caught:
        # Record every ConvergenceWarning, not only the first per code location, so the
        # count in summary.json's diagnostics is the number of fits that did not converge.
        warnings.simplefilter("always", ConvergenceWarning)
        if task.intervals:
            gcr = GroupConformalRegressor(factory, alphas=task.alphas, method="cv_plus")
            gcr.fit(task.X_train, task.y_train, task.cal_groups)
            yhat = gcr.predict(task.X_test)
            plus = gcr.predict_intervals(task.X_test)
            for alpha in task.alphas:
                q = gcr.quantile(alpha)
                bounds[f"{_level(alpha)}_cv"] = (yhat - q, yhat + q)
                bounds[f"{_level(alpha)}_cv_plus"] = plus[alpha]
            final = gcr.estimator_
            n_calibration = gcr.n_scores_
        else:
            final = factory()
            final.fit(task.X_train, task.y_train)
            yhat = np.asarray(final.predict(task.X_test), dtype=np.float64).reshape(-1)
            n_calibration = 0
    convergence = sum(issubclass(w.category, ConvergenceWarning) for w in caught)
    for w in caught:
        if not issubclass(w.category, ConvergenceWarning):
            logger.warning("%s while fitting %s: %s", w.category.__name__, task.model, w.message)
    if not np.all(np.isfinite(yhat)):
        raise RuntimeError(f"{task.model} produced non-finite predictions (fold {task.fold})")
    selection = final.selection_.to_dict() if isinstance(final, NestedSelectionRidge) else None
    return _TaskResult(
        spec_index=task.spec_index,
        fold=task.fold,
        test_positions=task.test_positions,
        yhat=np.asarray(yhat, dtype=np.float64),
        bounds=bounds,
        selection=selection,
        n_train=len(task.y_train),
        n_calibration=n_calibration,
        convergence_warnings=convergence,
        seconds=time.perf_counter() - started,
    )


def _tasks_for(
    spec_index: int, spec: Spec, frame: pd.DataFrame, cfg: BenchmarkConfig
) -> list[_Task]:
    columns = list(spec.features)
    X_all = frame[columns].copy()
    if spec.key.model == NESTED_MODEL:
        X_all[GROUP_COLUMN] = frame["battery_id"].to_numpy(dtype=object)
    y_all = frame[cfg.label].to_numpy(dtype=np.float64)
    fold_labels = frame["fold"].to_numpy(dtype=object)
    # Relative mode: reference cycles are partly computed from themselves - never fit on them.
    fit_ok = (
        ~frame[REFERENCE_FLAG_COLUMN].to_numpy(dtype=bool)
        if spec.key.mode == "relative"
        else np.ones(len(frame), dtype=bool)
    )
    alphas = tuple(round(1.0 - c, 10) for c in cfg.confidences)
    nested = (cfg.nested_alphas, cfg.nested_min_features, cfg.nested_tol)
    tasks: list[_Task] = []
    positions = np.arange(len(frame), dtype=np.intp)
    for fold in sorted(pd.unique(fold_labels), key=str):
        test = fold_labels == fold
        train = ~test & fit_ok
        train_rows, test_rows = frame.loc[train], frame.loc[test]
        assert_disjoint(row_keys(train_rows), row_keys(test_rows))
        assert_groups_disjoint(train_rows["battery_id"], test_rows["battery_id"])
        n_groups = len(pd.unique(fold_labels[train]))
        cost = _MODEL_COST.get(spec.key.model, 0.05) * int(train.sum()) / 500.0
        if spec.key.model == "gpr":
            cost *= (int(train.sum()) / 500.0) ** 2
        tasks.append(
            _Task(
                spec_index=spec_index,
                fold=str(fold),
                model=spec.key.model,
                kind=spec.kind,
                seed=cfg.seed,
                nested=nested,
                X_train=X_all.loc[train].reset_index(drop=True),
                y_train=y_all[train],
                cal_groups=fold_labels[train],
                X_test=X_all.loc[test].reset_index(drop=True),
                test_positions=positions[test],
                intervals=spec.intervals,
                alphas=alphas,
                cost=cost * ((n_groups + 1) if spec.intervals else 1),
            )
        )
    return tasks


def _execute(tasks: list[_Task], n_jobs: int) -> list[_TaskResult]:
    order = sorted(range(len(tasks)), key=lambda i: (-tasks[i].cost, i))
    if n_jobs == 1 or len(tasks) <= 1:
        results = [_run_task(tasks[i]) for i in order]
    else:
        logger.info("running %d fits on %s worker processes", len(tasks), n_jobs)
        results = list(
            Parallel(n_jobs=n_jobs, backend="loky")(delayed(_run_task)(tasks[i]) for i in order)
        )
    by_position = dict(zip(order, results, strict=True))
    return [by_position[i] for i in range(len(tasks))]


# =========================================================================== metrics


@dataclass
class SpecResult:
    """Out-of-fold predictions and metrics of one spec.

    Attributes:
        spec: The evaluated spec.
        oof: One row per evaluation cycle: ids, fold, ``y``, ``yhat``, interval
            bounds (``lo_<level>_<method>``/``hi_...``), the reference flag and
            whether the row is scored.
        per_battery: Metrics per held-out battery (``results.csv`` rows).
        summary: Spec-level metrics (means over batteries, pooled R^2).
        selections: ``ridge_nfs`` selection record per outer fold.
        convergence_warnings: ConvergenceWarnings raised while fitting.
    """

    spec: Spec
    oof: pd.DataFrame
    per_battery: list[dict[str, Any]]
    summary: dict[str, Any]
    selections: dict[str, dict[str, Any]]
    convergence_warnings: int


def _interval_columns(confidences: Sequence[float]) -> list[str]:
    return [
        f"{bound}_{percent_label(c)}_{method}"
        for c in confidences
        for method in CONFORMAL_METHODS
        for bound in ("lo", "hi")
    ]


def _assemble(
    spec: Spec,
    frame: pd.DataFrame,
    results: list[_TaskResult],
    cfg: BenchmarkConfig,
    *,
    n_abstained: int,
) -> SpecResult:
    n = len(frame)
    yhat = np.full(n, np.nan)
    bounds: dict[str, FloatArray] = {}
    if spec.intervals:
        bounds = {col: np.full(n, np.nan) for col in _interval_columns(cfg.confidences)}
    selections: dict[str, dict[str, Any]] = {}
    for res in results:
        yhat[res.test_positions] = res.yhat
        for key, (lo, hi) in res.bounds.items():
            level, method = key.split("_", 1)
            bounds[f"lo_{level}_{method}"][res.test_positions] = lo
            bounds[f"hi_{level}_{method}"][res.test_positions] = hi
        if res.selection is not None:
            selections[res.fold] = res.selection
    if not np.all(np.isfinite(yhat)):
        raise RuntimeError(f"{spec.key.id}: some rows received no out-of-fold prediction")
    reference = frame[REFERENCE_FLAG_COLUMN].to_numpy(dtype=bool)
    oof = pd.DataFrame(
        {
            "battery_id": frame["battery_id"].to_numpy(),
            "cycle_index": frame["cycle_index"].to_numpy(),
            "condition": frame["condition"].to_numpy(),
            "fold": frame["fold"].to_numpy(),
            "y": frame[cfg.label].to_numpy(dtype=np.float64),
            "yhat": yhat,
            **bounds,
            REFERENCE_FLAG_COLUMN: reference,
            "scored": ~reference if spec.key.mode == "relative" else np.ones(n, dtype=bool),
        }
    )
    per_battery = _per_battery(spec, oof, selections, cfg)
    summary = _spec_summary(spec, oof, per_battery, cfg)
    summary["n_abstained"] = n_abstained
    return SpecResult(
        spec=spec,
        oof=oof,
        per_battery=per_battery,
        summary=summary,
        selections=selections,
        convergence_warnings=sum(r.convergence_warnings for r in results),
    )


def _interval_stats(part: pd.DataFrame, cfg: BenchmarkConfig) -> dict[str, float]:
    out: dict[str, float] = {}
    for c in cfg.confidences:
        level = percent_label(c)
        for method in CONFORMAL_METHODS:
            lo, hi = part[f"lo_{level}_{method}"], part[f"hi_{level}_{method}"]
            iv = interval_metrics(part["y"], lo, hi)
            out[f"coverage_{level}_{method}"] = iv["coverage"]
            out[f"width_{level}_{method}"] = iv["mean_width"]
    return out


def _per_battery(
    spec: Spec, oof: pd.DataFrame, selections: Mapping[str, dict[str, Any]], cfg: BenchmarkConfig
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for battery in sorted(oof["battery_id"].unique()):
        part = oof[(oof["battery_id"] == battery) & oof["scored"]]
        if part.empty:
            raise RuntimeError(f"{spec.key.id}: battery {battery} has no scored rows")
        fold = str(part["fold"].iloc[0])
        metrics = regression_metrics(part["y"], part["yhat"])
        non_ref = part[~part[REFERENCE_FLAG_COLUMN]]
        row: dict[str, Any] = {
            **spec.key.as_dict(),
            "kind": spec.kind,
            "post_hoc": spec.post_hoc,
            "leaky": spec.leaky,
            "fold": fold,
            "battery_id": str(battery),
            "condition": str(part["condition"].iloc[0]),
            "n": int(metrics["n"]),
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "r2": metrics["r2"],
            "bias": metrics["bias"],
            "max_abs_error": metrics["max_abs_error"],
            "mae_non_reference": (
                regression_metrics(non_ref["y"], non_ref["yhat"])["mae"]
                if len(non_ref)
                else math.nan
            ),
        }
        if spec.intervals:
            row.update(_interval_stats(part, cfg))
        selection = selections.get(fold)
        row["selected_features"] = ";".join(selection["selected"]) if selection else ""
        row["alpha"] = selection["alpha"] if selection else math.nan
        rows.append(row)
    return rows


def _spec_summary(
    spec: Spec, oof: pd.DataFrame, per_battery: list[dict[str, Any]], cfg: BenchmarkConfig
) -> dict[str, Any]:
    scored = oof[oof["scored"]]
    pooled = regression_metrics(scored["y"], scored["yhat"])
    pb = pd.DataFrame(per_battery)
    worst = pb.loc[pb["mae"].idxmax()]
    summary: dict[str, Any] = {
        **spec.key.as_dict(),
        "kind": spec.kind,
        "post_hoc": spec.post_hoc,
        "leaky": spec.leaky,
        "baseline": spec.kind == "baseline",
        "n": int(pooled["n"]),
        "n_batteries": len(pb),
        "mae": float(pb["mae"].mean()),
        "rmse": float(pb["rmse"].mean()),
        "r2": pooled["r2"],
        "bias": float(pb["bias"].mean()),
        "max_abs_error": float(pb["max_abs_error"].max()),
        "worst_mae": float(worst["mae"]),
        "worst_battery": str(worst["battery_id"]),
        "mae_pooled": pooled["mae"],
        "r2_mean_per_battery": float(pb["r2"].mean()),
        "mae_non_reference": float(pb["mae_non_reference"].mean()),
    }
    if spec.intervals:
        pooled_iv = _interval_stats(scored, cfg)
        for key in pooled_iv:
            summary[key] = float(pb[key].mean())
            summary[f"{key}_pooled"] = pooled_iv[key]
            summary[f"{key}_min_battery"] = float(pb[key].min())
    return summary


def run_experiments(
    frames: Mapping[str, Frames],
    experiments: Mapping[str, ExperimentConfig],
    cfg: BenchmarkConfig,
    n_jobs: int,
) -> dict[str, list[SpecResult]]:
    """Fit and score every spec of every experiment; results grouped by experiment, in order."""
    all_specs: list[tuple[str, Spec, pd.DataFrame, int]] = []
    tasks: list[_Task] = []
    for name, exp in experiments.items():
        for spec in build_specs(exp):
            frame, n_abstained = spec_frame(spec, frames[name])
            index = len(all_specs)
            all_specs.append((name, spec, frame, n_abstained))
            tasks.extend(_tasks_for(index, spec, frame, cfg))
    results = _execute(tasks, n_jobs)
    grouped: dict[int, list[_TaskResult]] = {}
    seconds: dict[str, float] = {}
    for task_result in results:
        grouped.setdefault(task_result.spec_index, []).append(task_result)
        key = all_specs[task_result.spec_index][1].key.id
        seconds[key] = seconds.get(key, 0.0) + task_result.seconds
    slowest = sorted(seconds.items(), key=lambda kv: -kv[1])[:5]
    logger.info(
        "fit time %.0f s in total; slowest: %s",
        sum(seconds.values()),
        ", ".join(f"{k} {v:.0f}s" for k, v in slowest),
    )
    out: dict[str, list[SpecResult]] = {name: [] for name in experiments}
    for index, (name, spec, frame, n_abstained) in enumerate(all_specs):
        spec_result = _assemble(spec, frame, grouped[index], cfg, n_abstained=n_abstained)
        out[name].append(spec_result)
        logger.debug("%s: MAE %.3f", spec.key.id, spec_result.summary["mae"])
    return out
