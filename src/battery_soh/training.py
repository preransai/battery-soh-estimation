"""Production model bundles: fit on every core cell, export auditable JSON plus model cards.

``soh train`` (:func:`train_bundles`) turns the evaluated procedures of the
benchmark into the artifacts the API serves:

1. **Rows.** QC-valid cycles of the core cells with a finite label on which at
   least one of the bundle's indicators was measured - the same rows the
   benchmark scores. In relative mode every indicator is divided by the cell's
   commissioning reference (mean of its first ``n_reference_cycles`` QC-valid
   values) and the reference cycles themselves are never fitted or used for
   calibration, exactly as in the benchmark.
2. **Procedure.** The configured model (``production.model`` in
   ``configs/benchmark.yaml``): ``ridge_nfs`` re-runs the pre-specified nested
   selection (:mod:`battery_soh.evaluation.nested`) on all core cells to fix
   the inputs and the ridge penalty; ``ridge`` keeps every input of the set.
   The exported estimator is always ``SimpleImputer -> StandardScaler -> Ridge``.
3. **Intervals.** :class:`~battery_soh.uncertainty.conformal.GroupConformalRegressor`
   with ``method="cv"``: one model per held-out core cell (each repeating the
   whole procedure, selection included, without that cell), and the bundle's
   half-width is the exact conformal quantile of the pooled out-of-cell
   residuals.
4. **Parity with the benchmark.** Those per-cell models are, by construction,
   the outer-fold models of the benchmark's leave-one-battery-out run for the
   same spec. Their per-cell MAE (and, for ``ridge_nfs``, their selected
   features and penalty) must reproduce ``reports/benchmark/results.csv``;
   otherwise training fails, because the metrics copied into the bundle would
   not describe the procedure that produced it.
5. **Metrics** in the bundle are copied from ``results.csv`` (per held-out cell,
   mean over cells, coverage of the shipped ``cv`` interval construction).
6. **Provenance.** The benchmark run consumed (``summary.json`` next to
   ``results.csv``) must have been produced from the current config's benchmark
   sections and the current feature table; each bundle records that run's
   ``config_sha256`` as ``benchmark_config_sha256`` plus its own
   ``production_config_sha256`` (hash of the ``production`` section only, see
   :func:`battery_soh.evaluation.benchmark.config_hashes`).
7. **Outputs.** ``<model_id>.json`` (written with LF line endings, so the
   SHA-256 matches on every OS after a git checkout), ``model_cards/<model_id>.md``
   and ``MANIFEST.json`` with the SHA-256 of every file. ``created_utc`` comes
   from the feature table's build time (or a fixed value in the config), so a
   re-run on the same inputs writes byte-identical files.

Run order: ``soh benchmark`` -> ``soh train``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import platform
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import pandas as pd
import sklearn

from battery_soh.config import DEFAULT_N_REFERENCE_CYCLES, get_paths
from battery_soh.errors import BatterySohError
from battery_soh.evaluation.benchmark.config import BenchmarkConfig, load_config
from battery_soh.evaluation.benchmark.hashing import (
    PRODUCTION_HASH_SECTIONS,
    check_keys,
    config_sections_sha256,
    read_raw_config,
    require_mapping,
)
from battery_soh.evaluation.metrics import regression_metrics, round_significant
from battery_soh.evaluation.nested import GROUP_COLUMN, NestedSelectionRidge
from battery_soh.features.build import KEY_COLUMNS, load_feature_table, metadata_path
from battery_soh.features.registry import get_feature_set
from battery_soh.features.relative import (
    REFERENCE_FLAG_COLUMN,
    is_context_feature,
    relativize,
)
from battery_soh.models.linear_bundle import (
    LABEL_DEFINITION,
    MANIFEST_NAME,
    LinearBundle,
    Mode,
    export_linear_bundle,
    save_bundle,
    verify_manifest,
    write_manifest,
    write_text_lf,
)
from battery_soh.models.zoo import make_model
from battery_soh.reporting.model_card import render_model_card
from battery_soh.uncertainty.conformal import GroupConformalRegressor, confidence_key

__all__ = [
    "PRODUCTION_MODELS",
    "ProductionConfig",
    "ProductionSpec",
    "TrainingError",
    "load_production_config",
    "train_bundles",
    "training_frame",
]

logger = logging.getLogger(__name__)

#: Procedures that export as a single ridge pipeline.
PRODUCTION_MODELS: Final[tuple[str, ...]] = ("ridge", "ridge_nfs")
#: Sub-directory of the output folder holding the Markdown model cards.
MODEL_CARD_DIR: Final = "model_cards"
#: ``production.created_utc`` value meaning "use the feature table's build time".
CREATED_FROM_FEATURES: Final = "features_meta"

_PRODUCTION_KEYS: Final = frozenset({"description", "experiment", "model", "created_utc", "specs"})
_SPEC_KEYS: Final = frozenset({"feature_set", "mode", "model"})
_MODEL_ID_RE: Final = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
# results.csv stores 6 significant digits; the recomputed metrics must agree to that precision.
_PARITY_RTOL: Final = 2e-5
_COVERAGE_METHOD: Final = "cv"
_CONFORMAL_METHOD_TEXT: Final = (
    "group conformal (cv): full-data prediction +/- the exact conformal quantile of the pooled "
    "leave-one-battery-out absolute residuals over the training cells; each residual comes from "
    "a model fitted, selection included, without that residual's cell"
)


class TrainingError(RuntimeError, BatterySohError):
    """Production training cannot produce trustworthy bundles (missing or inconsistent inputs)."""


# =========================================================================== configuration


@dataclass(frozen=True)
class ProductionSpec:
    """One production bundle.

    Attributes:
        feature_set: Registry feature set (candidate inputs).
        mode: ``"absolute"`` or ``"relative"``.
        model: Training procedure, one of :data:`PRODUCTION_MODELS`.
        experiment: Benchmark experiment whose rows and metrics the bundle uses.
    """

    feature_set: str
    mode: Mode
    model: str
    experiment: str

    @property
    def model_id(self) -> str:
        """Served identifier ``<feature_set>-<mode>-ridge`` (the API and web UI rely on it)."""
        return f"{self.feature_set}-{self.mode}-ridge"

    @property
    def benchmark_id(self) -> str:
        """``experiment/feature_set/mode/model`` as used by the benchmark."""
        return f"{self.experiment}/{self.feature_set}/{self.mode}/{self.model}"


@dataclass(frozen=True)
class ProductionConfig:
    """Parsed ``production`` section plus the benchmark settings it depends on.

    Attributes:
        benchmark: The validated benchmark configuration.
        specs: Bundles to train.
        batteries: Training cells (the experiment's cells).
        keep_absolute: Indicators the experiment leaves absolute in relative mode.
        created_utc: ``"features_meta"`` or a fixed ISO-8601 timestamp.
        sha256: Scoped hash of the ``production`` section
            (:data:`~battery_soh.evaluation.benchmark.PRODUCTION_HASH_SECTIONS`);
            the benchmark sections are covered by ``benchmark.sha256``.
    """

    benchmark: BenchmarkConfig
    specs: tuple[ProductionSpec, ...]
    batteries: tuple[str, ...]
    keep_absolute: tuple[str, ...]
    created_utc: str
    sha256: str


def load_production_config(config_path: Path) -> ProductionConfig:
    """Parse the ``production`` section of the benchmark YAML.

    Args:
        config_path: ``configs/benchmark.yaml``.

    Returns:
        The validated :class:`ProductionConfig`.

    Raises:
        FileNotFoundError: If the config does not exist.
        ValueError: If the section is missing or invalid (unknown keys, a leaky
            or unknown feature set, a model that cannot be exported, a spec
            outside the benchmark grid, duplicate model ids, or a relative spec
            whose inputs the API could not relativize).
    """
    config_path = Path(config_path)
    cfg = load_config(config_path)  # validates the whole file
    raw = read_raw_config(config_path)
    section = require_mapping(raw.get("production"), "production")
    check_keys(section, _PRODUCTION_KEYS, "production")
    experiment_name = str(section.get("experiment", cfg.headline.experiment))
    if experiment_name not in cfg.experiments:
        raise ValueError(f"production.experiment {experiment_name!r} is not a benchmark experiment")
    experiment = cfg.experiments[experiment_name]
    if experiment.group_by != "battery":
        raise ValueError("production.experiment must hold out batteries (group_by: battery)")
    default_model = str(section.get("model", "ridge"))
    created_raw = section.get("created_utc", CREATED_FROM_FEATURES)
    if not isinstance(created_raw, str):
        # An unquoted YAML timestamp parses to a datetime whose str() is not ISO 8601.
        raise ValueError(
            "production.created_utc must be a string: 'features_meta' or a quoted ISO-8601 "
            "timestamp"
        )
    created = created_raw

    raw_specs = section.get("specs")
    if not isinstance(raw_specs, list) or not raw_specs:
        raise ValueError("production.specs must be a non-empty list")
    specs: list[ProductionSpec] = []
    for i, entry in enumerate(raw_specs):
        where = f"production.specs[{i}]"
        item = require_mapping(entry, where)
        check_keys(item, _SPEC_KEYS, where)
        feature_set = str(item.get("feature_set", ""))
        get_feature_set(feature_set)  # unknown or leaky sets raise ValueError
        mode = str(item.get("mode", "absolute"))
        if mode not in ("absolute", "relative"):
            raise ValueError(f"{where}.mode must be 'absolute' or 'relative', got {mode!r}")
        model = str(item.get("model", default_model))
        if model not in PRODUCTION_MODELS:
            raise ValueError(f"{where}.model must be one of {list(PRODUCTION_MODELS)}")
        if (
            feature_set not in experiment.feature_sets
            or mode not in experiment.modes
            or model not in experiment.models
        ):
            raise ValueError(
                f"{where} ({feature_set}/{mode}/{model}) is not evaluated by the "
                f"{experiment_name!r} benchmark experiment, so it has no metrics"
            )
        spec = ProductionSpec(feature_set, cast(Mode, mode), model, experiment_name)
        if not _MODEL_ID_RE.match(spec.model_id):
            raise ValueError(f"{where} gives an unsafe model id {spec.model_id!r}")
        if mode == "relative":
            kept = sorted(
                set(experiment.relative_keep_absolute) & set(get_feature_set(feature_set))
            )
            if kept:
                raise ValueError(
                    f"{where}: {kept} stay absolute in relative mode, but the API relativizes "
                    "every indicator of a relative bundle"
                )
        specs.append(spec)
    ids = [s.model_id for s in specs]
    duplicates = sorted({m for m in ids if ids.count(m) > 1})
    if duplicates:
        raise ValueError(f"production.specs define duplicate model ids {duplicates}")
    return ProductionConfig(
        benchmark=cfg,
        specs=tuple(specs),
        batteries=tuple(experiment.batteries),
        keep_absolute=tuple(experiment.relative_keep_absolute),
        created_utc=created,
        sha256=config_sections_sha256(raw, PRODUCTION_HASH_SECTIONS),
    )


# =========================================================================== data


def training_frame(
    table: pd.DataFrame,
    *,
    batteries: Sequence[str],
    features: Sequence[str],
    mode: Mode,
    label: str = "soh_pct",
    n_ref: int = DEFAULT_N_REFERENCE_CYCLES,
    keep_absolute: Sequence[str] = (),
) -> pd.DataFrame:
    """Evaluation rows of one bundle, with its inputs in the bundle's mode.

    Mirrors the benchmark's row rules: QC-valid cycles of ``batteries`` with a
    finite label and at least one measured (non-context) input. In relative
    mode the inputs are divided by each cell's commissioning reference and the
    column ``is_reference_cycle`` marks the cycles that entered a reference;
    callers must not fit or calibrate on them. In absolute mode the column is
    all ``False``.

    Args:
        table: Feature table (:func:`~battery_soh.features.build.load_feature_table`).
        batteries: Cells to use.
        features: Candidate inputs.
        mode: ``"absolute"`` or ``"relative"``.
        label: Label column.
        n_ref: Reference cycles per cell (relative mode).
        keep_absolute: Inputs left absolute in relative mode.

    Returns:
        Rows sorted by ``(battery_id, cycle_index)`` with a fresh index.

    Raises:
        ValueError: If a battery or column is missing from ``table``.
    """
    missing_cells = sorted(set(batteries) - set(table["battery_id"]))
    if missing_cells:
        raise ValueError(f"batteries {missing_cells} are not in the feature table")
    missing_cols = [c for c in (*features, label, "qc_valid") if c not in table.columns]
    if missing_cols:
        raise ValueError(f"feature table lacks columns {missing_cols}")
    cells = table[table["battery_id"].isin(list(batteries))]
    cells = cells.sort_values(list(KEY_COLUMNS), kind="stable").reset_index(drop=True)
    measured_cols = [f for f in features if not is_context_feature(f)]
    if mode == "relative":
        relative_cols = [f for f in measured_cols if f not in set(keep_absolute)]
        frame = relativize(cells, relative_cols, n_ref=n_ref)
    else:
        frame = cells.copy()
        frame[REFERENCE_FLAG_COLUMN] = False
    rows = frame["qc_valid"].to_numpy(dtype=bool) & np.isfinite(
        frame[label].to_numpy(dtype=np.float64)
    )
    if measured_cols:
        rows &= np.isfinite(frame[measured_cols].to_numpy(dtype=np.float64)).any(axis=1)
    return frame.loc[rows].reset_index(drop=True)


@dataclass(frozen=True)
class _FeatureTable:
    table: pd.DataFrame
    sha256: str
    created_utc: str
    file_name: str


def _load_features(csv_path: Path) -> _FeatureTable:
    """Load the feature table and its sidecar; the recorded SHA-256 must match the CSV."""
    meta_file = metadata_path(csv_path)
    if not meta_file.is_file():
        raise TrainingError(
            f"{meta_file} is missing; rebuild the table with `soh data build` so the bundles "
            "can record which data they were trained on"
        )
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    recorded = str(meta.get("sha256", ""))
    actual = hashlib.sha256(Path(csv_path).read_bytes()).hexdigest()
    if recorded != actual:
        raise TrainingError(
            f"{csv_path.name} does not match the sha256 in {meta_file.name} "
            f"({actual[:12]} != {recorded[:12]}); rebuild it with `soh data build`"
        )
    return _FeatureTable(
        table=load_feature_table(csv_path),
        sha256=actual,
        created_utc=str(meta.get("created_utc", "")),
        file_name=Path(csv_path).name,
    )


@dataclass(frozen=True)
class _BenchmarkRun:
    """Identity of the benchmark run whose results a training run consumes."""

    config_sha256: str
    data_sha256: str
    summary_path: Path


def _load_benchmark_run(
    summary_path: Path, *, cfg: BenchmarkConfig, features: _FeatureTable
) -> _BenchmarkRun:
    """Read the run's ``summary.json``; it must match the current config and feature table.

    Raises:
        TrainingError: If the summary is missing or unreadable, or was produced
            from other benchmark settings (``config_sha256``) or another
            feature table (``data_sha256``).
    """
    if not summary_path.is_file():
        raise TrainingError(
            f"{summary_path} not found: training copies metrics from that benchmark run and "
            "must check which config and data produced it; run `soh benchmark` first"
        )
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TrainingError(f"{summary_path} is not valid JSON: {exc}") from exc
    if not isinstance(summary, dict):
        raise TrainingError(f"{summary_path} must contain a JSON object")
    recorded_cfg = str(summary.get("config_sha256", ""))
    recorded_data = str(summary.get("data_sha256", ""))
    if recorded_cfg != cfg.sha256:
        raise TrainingError(
            f"{summary_path.name} was produced with other benchmark settings (config_sha256 "
            f"{recorded_cfg[:12] or 'missing'}) than {cfg.path.name} now has "
            f"({cfg.sha256[:12]}); re-run `soh benchmark` (edits of the production section "
            "alone do not require this)"
        )
    if recorded_data != features.sha256:
        raise TrainingError(
            f"{summary_path.name} was produced from another feature table (data_sha256 "
            f"{recorded_data[:12] or 'missing'}, now {features.sha256[:12]}); re-run "
            "`soh benchmark`"
        )
    return _BenchmarkRun(
        config_sha256=recorded_cfg, data_sha256=recorded_data, summary_path=summary_path
    )


# =========================================================================== benchmark metrics


def _results_rows(results: pd.DataFrame, spec: ProductionSpec, results_path: Path) -> pd.DataFrame:
    """Per-battery benchmark rows of ``spec`` (fails loudly when absent)."""
    needed = {"experiment", "feature_set", "mode", "model", "kind", "battery_id", "n", "mae"}
    missing = sorted(needed - set(results.columns))
    if missing:
        raise TrainingError(f"{results_path} lacks columns {missing}; re-run `soh benchmark`")
    rows = results[
        (results["experiment"] == spec.experiment)
        & (results["feature_set"] == spec.feature_set)
        & (results["mode"] == spec.mode)
        & (results["model"] == spec.model)
        & (results["kind"] == "model")
    ]
    if rows.empty:
        raise TrainingError(
            f"{results_path} has no rows for {spec.benchmark_id}; run `soh benchmark` "
            "(with this config) before `soh train`"
        )
    if rows["battery_id"].duplicated().any():
        raise TrainingError(f"{results_path} has duplicate rows for {spec.benchmark_id}")
    return rows.sort_values("battery_id").reset_index(drop=True)


def _coverage_columns(confidences: Sequence[float]) -> list[tuple[str, str, str]]:
    """``(confidence key, coverage column, width column)`` of the shipped interval method."""
    return [
        (
            confidence_key(c),
            f"coverage_{round(c * 100):d}_{_COVERAGE_METHOD}",
            f"width_{round(c * 100):d}_{_COVERAGE_METHOD}",
        )
        for c in confidences
    ]


def _benchmark_metrics(
    rows: pd.DataFrame,
    *,
    confidences: Sequence[float],
    pooled_r2: float,
    spec: ProductionSpec,
    source: str,
) -> dict[str, Any]:
    """Bundle ``metrics`` section from the benchmark's per-battery rows."""
    cov_cols = _coverage_columns(confidences)
    per_battery: dict[str, Any] = {}
    for row in rows.itertuples(index=False):
        rec = row._asdict()
        per_battery[str(rec["battery_id"])] = {
            "n": int(rec["n"]),
            "mae": round_significant(rec.get("mae")),
            "rmse": round_significant(rec.get("rmse")),
            "r2": round_significant(rec.get("r2")),
            "bias": round_significant(rec.get("bias")),
            "max_abs_error": round_significant(rec.get("max_abs_error")),
            "coverage": {key: round_significant(rec.get(cov)) for key, cov, _ in cov_cols},
            "mean_width": {key: round_significant(rec.get(width)) for key, _, width in cov_cols},
        }
    worst = rows.loc[rows["mae"].idxmax()]

    def _mean(column: str) -> float | None:
        if column not in rows.columns or rows[column].isna().all():
            return None
        return round_significant(rows[column].mean())

    return {
        "protocol": "leave-one-battery-out",
        "source": source,
        "spec": {
            "experiment": spec.experiment,
            "feature_set": spec.feature_set,
            "mode": spec.mode,
            "model": spec.model,
        },
        "evaluates": (
            "the whole training procedure (feature selection, penalty and fit) on each "
            "held-out cell; the shipped model is that procedure refitted on every cell"
        ),
        "per_battery": per_battery,
        "mean": {
            "mae": _mean("mae"),
            "rmse": _mean("rmse"),
            "bias": _mean("bias"),
            "r2": round_significant(pooled_r2),
            "worst_mae": round_significant(worst["mae"]),
            "worst_battery": str(worst["battery_id"]),
            "n_batteries": len(rows),
            "n_cycles": int(rows["n"].sum()),
        },
        "coverage_method": _COVERAGE_METHOD,
        "coverage": {key: _mean(cov) for key, cov, _ in cov_cols},
        "mean_width": {key: _mean(width) for key, _, width in cov_cols},
        "coverage_min_battery": {
            key: round_significant(rows[cov].min()) if cov in rows.columns else None
            for key, cov, _ in cov_cols
        },
        "definitions": {
            "mae, rmse, bias": "mean over held-out cells of the per-cell value (signed bias = "
            "prediction - truth)",
            "r2": "pooled over every held-out prediction",
            "coverage, mean_width": "mean over held-out cells; interval = the held-out fold's "
            "model +/- conformal quantile of its own training cells' out-of-cell residuals "
            "(the construction this bundle ships)",
        },
    }


def _check_parity(
    spec: ProductionSpec,
    rows: pd.DataFrame,
    gcr: GroupConformalRegressor,
    y: np.ndarray[Any, np.dtype[np.float64]],
    results_path: Path,
) -> None:
    """The per-cell models must reproduce the benchmark's held-out MAE (and selections)."""
    groups = np.asarray(gcr.score_groups_, dtype=object)
    problems: list[str] = []
    for row in rows.itertuples(index=False):
        rec = row._asdict()
        battery = str(rec["battery_id"])
        mask = groups == battery
        n = int(mask.sum())
        if n != int(rec["n"]):
            problems.append(f"{battery}: {n} rows here vs n={rec['n']} in the benchmark")
            continue
        mae = float(np.mean(np.abs(y[mask] - gcr.oof_predictions_[mask])))
        if not math.isclose(mae, float(rec["mae"]), rel_tol=_PARITY_RTOL, abs_tol=1e-9):
            problems.append(f"{battery}: MAE {mae:.6g} here vs {float(rec['mae']):.6g}")
        model = gcr.fold_estimators_.get(battery)
        if isinstance(model, NestedSelectionRidge):
            listed = rec.get("selected_features")
            expected = listed.split(";") if isinstance(listed, str) and listed else []
            if list(model.selected_features_) != expected:
                problems.append(
                    f"{battery}: selected {model.selected_features_} here vs {expected}"
                )
            alpha = float(rec.get("alpha", math.nan))
            if not math.isclose(float(model.alpha_), alpha):
                problems.append(f"{battery}: alpha {model.alpha_} here vs {alpha}")
    trained = set(groups)
    benchmarked = {str(b) for b in rows["battery_id"]}
    if trained != benchmarked:
        problems.append(f"cells {sorted(trained)} here vs {sorted(benchmarked)} in the benchmark")
    if problems:
        raise TrainingError(
            f"the production procedure for {spec.model_id} does not reproduce the benchmark's "
            f"held-out results for {spec.benchmark_id} ({results_path}): "
            + "; ".join(problems)
            + ". Re-run `soh benchmark` on the current feature table and config."
        )


# =========================================================================== fitting


def _factory(model: str, cfg: BenchmarkConfig) -> Any:
    if model == "ridge_nfs":
        return partial(
            NestedSelectionRidge,
            alphas=cfg.nested_alphas,
            min_features=cfg.nested_min_features,
            tol=cfg.nested_tol,
        )
    return partial(make_model, "ridge", random_state=cfg.seed)


def _selection_record(
    candidates: Sequence[str],
    gcr: GroupConformalRegressor,
    cfg: BenchmarkConfig,
) -> dict[str, Any]:
    final = gcr.estimator_
    if not isinstance(final, NestedSelectionRidge):
        return {
            "procedure": "none: every input of the feature set is used",
            "candidates": list(candidates),
            "selected": list(candidates),
        }
    folds = []
    counts = dict.fromkeys(candidates, 0)
    for battery in sorted(gcr.fold_estimators_):
        model = gcr.fold_estimators_[battery]
        assert isinstance(model, NestedSelectionRidge)
        for name in model.selected_features_:
            counts[name] += 1
        folds.append(
            {
                "held_out": battery,
                "selected": list(model.selected_features_),
                "alpha": float(model.alpha_),
                "inner_mae": round_significant(model.inner_mae_),
            }
        )
    n_folds = max(len(folds), 1)
    final_dict = final.selection_.to_dict()
    for step in final_dict["path"]:
        step["inner_mae"] = round_significant(step["inner_mae"])
    final_dict["inner_mae"] = round_significant(final_dict["inner_mae"])
    return {
        "procedure": (
            "nested backward elimination (battery_soh.evaluation.nested): start from every "
            "candidate; repeatedly drop the input whose removal most lowers the inner "
            "leave-one-battery-out MAE (penalty re-chosen from the grid at every step); stop "
            "when no removal improves it"
        ),
        "scoring": "mean over inner leave-one-battery-out folds of the held-out cell's MAE",
        "alphas": list(cfg.nested_alphas),
        "min_features": cfg.nested_min_features,
        "tol": cfg.nested_tol,
        "candidates": list(candidates),
        "final": final_dict,
        "lobo_folds": folds,
        "lobo_frequency": {name: round_significant(c / n_folds) for name, c in counts.items()},
        "note": (
            "'final' is the selection on all training cells (the shipped inputs); 'lobo_folds' "
            "are the selections of the per-cell models behind the metrics and the conformal "
            "residuals"
        ),
    }


def _display_path(path: Path) -> str:
    root = get_paths().root
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def _fit_spec(
    spec: ProductionSpec,
    *,
    pcfg: ProductionConfig,
    features: _FeatureTable,
    results: pd.DataFrame,
    results_path: Path,
    provenance: Mapping[str, Any],
    created_utc: str,
) -> dict[str, Any]:
    """Fit, verify and export one bundle document."""
    cfg = pcfg.benchmark
    candidates = list(get_feature_set(spec.feature_set))
    frame = training_frame(
        features.table,
        batteries=pcfg.batteries,
        features=candidates,
        mode=spec.mode,
        label=cfg.label,
        n_ref=cfg.n_reference_cycles,
        keep_absolute=pcfg.keep_absolute,
    )
    reference = frame[REFERENCE_FLAG_COLUMN].to_numpy(dtype=bool)
    fit = frame.loc[~reference].reset_index(drop=True)
    groups = fit["battery_id"].to_numpy(dtype=object)
    if len(pd.unique(groups)) < 2:
        raise TrainingError(f"{spec.model_id}: need at least two training cells")
    X = fit[candidates].copy()
    if spec.model == "ridge_nfs":
        X[GROUP_COLUMN] = groups
    y = fit[cfg.label].to_numpy(dtype=np.float64)

    alphas = tuple(round(1.0 - c, 10) for c in cfg.confidences)
    gcr = GroupConformalRegressor(_factory(spec.model, cfg), alphas=alphas, method="cv")
    gcr.fit(X, y, groups)

    rows = _results_rows(results, spec, results_path)
    _check_parity(spec, rows, gcr, y, results_path)
    pooled_r2 = regression_metrics(y, gcr.oof_predictions_)["r2"]
    metrics = _benchmark_metrics(
        rows,
        confidences=cfg.confidences,
        pooled_r2=pooled_r2,
        spec=spec,
        source=_display_path(results_path),
    )

    final = gcr.estimator_
    if isinstance(final, NestedSelectionRidge):
        pipeline, inputs = final.pipeline_, list(final.selected_features_)
    else:
        pipeline, inputs = final, candidates
    quantiles: dict[float | str, float] = dict(gcr.quantiles_by_confidence().items())
    extra = {
        "selection": _selection_record(candidates, gcr, cfg),
        "training_protocol": _training_protocol(spec, frame, fit, cfg),
        "provenance": dict(provenance),
        "model_card": f"{MODEL_CARD_DIR}/{spec.model_id}.md",
    }
    bundle = export_linear_bundle(
        pipeline,
        model_id=spec.model_id,
        feature_set=spec.feature_set,
        mode=spec.mode,
        features=inputs,
        training_df=fit[["battery_id", "cycle_index", *inputs]],
        conformal_quantiles=quantiles,
        metrics=metrics,
        n_conformal_scores=gcr.n_scores_,
        conformal_method=_CONFORMAL_METHOD_TEXT,
        label_col=cfg.label,
        label_definition=LABEL_DEFINITION,
        n_reference_cycles=cfg.n_reference_cycles,
        features_sha256=features.sha256,
        created_utc=created_utc,
        extra=extra,
    )
    logger.info(
        "%s: %d inputs %s, alpha=%g, %d cycles, LOBO MAE %.3f, 90%% half-width %.2f",
        spec.model_id,
        len(inputs),
        inputs,
        float(bundle["estimator"].get("alpha", math.nan)),
        len(fit),
        float(metrics["mean"]["mae"] or math.nan),
        float(quantiles.get("0.9", math.nan)),
    )
    return bundle


def _training_protocol(
    spec: ProductionSpec, frame: pd.DataFrame, fit: pd.DataFrame, cfg: BenchmarkConfig
) -> dict[str, Any]:
    """How the fitted rows were chosen, plus per-cell counts and label ranges."""
    relative = spec.mode == "relative"
    rows = "QC-valid cycles with a finite label and at least one measured input"
    if relative:
        rows += "; reference cycles excluded from fitting and calibration"
    return {
        "procedure": spec.model,
        "experiment": spec.experiment,
        "rows": rows,
        "n_evaluation_rows": len(frame),
        "n_reference_cycles_excluded": int(frame[REFERENCE_FLAG_COLUMN].sum()),
        "cycles_per_battery": {
            str(b): int(n) for b, n in fit["battery_id"].value_counts().sort_index().items()
        },
        "label_range_per_battery": {
            str(b): [round_significant(g[cfg.label].min()), round_significant(g[cfg.label].max())]
            for b, g in fit.groupby("battery_id", sort=True)
        },
        "relative_reference": (
            {
                "rule": (
                    f"each indicator divided by the mean of the cell's first "
                    f"{cfg.n_reference_cycles} QC-valid, finite values of that indicator"
                ),
                "context_features": "passed through unchanged",
            }
            if relative
            else None
        ),
    }


def _provenance(
    pcfg: ProductionConfig, results_path: Path, run: _BenchmarkRun, features: _FeatureTable
) -> dict[str, Any]:
    """Inputs and environment a bundle was built from."""
    return {
        "config": _display_path(pcfg.benchmark.path),
        "benchmark_config_sha256": run.config_sha256,
        "production_config_sha256": pcfg.sha256,
        "config_hashes": (
            "benchmark_config_sha256 = config_sha256 of the benchmark run the metrics come from "
            "(its summary.json; covers the sections that determine benchmark results); "
            "production_config_sha256 = hash of the production section; both computed by "
            "battery_soh.evaluation.benchmark.config_hashes"
        ),
        "benchmark_summary": _display_path(run.summary_path),
        "benchmark_results": _display_path(results_path),
        "benchmark_results_sha256": hashlib.sha256(results_path.read_bytes()).hexdigest(),
        "features_csv": features.file_name,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
    }


# =========================================================================== output


def _resolve_created(pcfg: ProductionConfig, features: _FeatureTable) -> str:
    if pcfg.created_utc != CREATED_FROM_FEATURES:
        return pcfg.created_utc
    if not features.created_utc:
        raise TrainingError(
            "the feature table's .meta.json has no created_utc; set production.created_utc"
        )
    return features.created_utc


def train_bundles(
    config_path: Path,
    out_dir: Path,
    *,
    results_csv: Path | None = None,
    summary_json: Path | None = None,
) -> list[Path]:
    """Fit every production bundle, write the bundles, model cards and ``MANIFEST.json``.

    Args:
        config_path: ``configs/benchmark.yaml`` (its ``production`` section lists
            the bundles; the rest supplies rows, nested-selection settings and
            confidence levels).
        out_dir: Output folder (``artifacts/models``).
        results_csv: The benchmark's per-battery results (default
            ``reports/benchmark/results.csv``); bundle metrics are copied from it.
        summary_json: The same run's ``summary.json`` (default: next to
            ``results_csv``); identifies the config and data of that run.

    Returns:
        Paths written: one bundle and one model card per spec, then the manifest.

    Raises:
        TrainingError: If the benchmark results lack a spec, disagree with the
            re-fitted per-cell models, come from other benchmark settings or
            another feature table, or the feature table's sidecar is stale.
        ValueError: If the configuration is invalid.
        FileNotFoundError: If the config, feature table or results are missing.
    """
    pcfg = load_production_config(Path(config_path))
    cfg = pcfg.benchmark
    results_path = (
        Path(results_csv) if results_csv is not None else get_paths().benchmark_dir / "results.csv"
    )
    if not results_path.is_file():
        raise FileNotFoundError(
            f"benchmark results not found: {results_path} (run `soh benchmark` first)"
        )
    results = pd.read_csv(results_path, keep_default_na=True)
    features = _load_features(cfg.features_csv)
    run = _load_benchmark_run(
        Path(summary_json) if summary_json is not None else results_path.with_name("summary.json"),
        cfg=cfg,
        features=features,
    )
    provenance = _provenance(pcfg, results_path, run, features)
    created_utc = _resolve_created(pcfg, features)

    # Fit and verify every bundle before writing anything: a failure leaves the
    # previous artifact folder (and its manifest) untouched.
    bundles: list[dict[str, Any]] = []
    for spec in pcfg.specs:
        logger.info("training %s (%s)", spec.model_id, spec.benchmark_id)
        bundles.append(
            _fit_spec(
                spec,
                pcfg=pcfg,
                features=features,
                results=results,
                results_path=results_path,
                provenance=provenance,
                created_utc=created_utc,
            )
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    manifest_files: list[str] = []
    for bundle in bundles:
        model_id = str(bundle["model_id"])
        bundle_path = save_bundle(bundle, out_dir / f"{model_id}.json")
        card_rel = f"{MODEL_CARD_DIR}/{model_id}.md"
        card_path = out_dir / card_rel
        write_text_lf(card_path, render_model_card(bundle))
        written += [bundle_path, card_path]
        manifest_files += [bundle_path.name, card_rel]

    stale = sorted(
        p.name
        for p in out_dir.glob("*.json")
        if p.name != MANIFEST_NAME and p.name not in set(manifest_files)
    )
    if stale:
        logger.warning("%s holds bundles this config does not produce: %s", out_dir, stale)
    manifest = write_manifest(out_dir, manifest_files, created_utc=created_utc)
    verify_manifest(out_dir)
    for path in written:
        if path.suffix == ".json":
            LinearBundle.from_json(path)
    written.append(manifest)
    logger.info(
        "wrote %d bundle(s) and %s to %s (benchmark config %s, production config %s)",
        len(pcfg.specs),
        MANIFEST_NAME,
        out_dir,
        run.config_sha256[:12],
        pcfg.sha256[:12],
    )
    return written
