"""Parsing and validation of ``configs/benchmark.yaml`` into typed, frozen settings."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Final, Literal, cast

from battery_soh.config import DEFAULT_N_REFERENCE_CYCLES, RANDOM_SEED, get_paths
from battery_soh.data.protocols import CORE_BATTERIES, EXTENDED_BATTERIES, PROTOCOLS
from battery_soh.evaluation.baselines import BASELINE_NAMES
from battery_soh.evaluation.benchmark.hashing import (
    BENCHMARK_HASH_SECTIONS,
    TOP_KEYS,
    check_keys,
    config_sections_sha256,
    read_raw_config,
    require_mapping,
)
from battery_soh.evaluation.nested import DEFAULT_ALPHAS, DEFAULT_TOL
from battery_soh.features.registry import FEATURE_SETS
from battery_soh.models.zoo import MODEL_NAMES
from battery_soh.uncertainty.conformal import confidence_key

__all__ = [
    "CONFORMAL_METHODS",
    "MODES",
    "NESTED_MODEL",
    "REPORTED_CONFIDENCE",
    "BenchmarkConfig",
    "ExperimentConfig",
    "GroupBy",
    "SpecKey",
    "WindowSweepConfig",
    "check_in_grid",
    "in_grid",
    "load_config",
    "percent_label",
]

#: Model name of Ridge with nested feature selection and penalty tuning.
NESTED_MODEL: Final = "ridge_nfs"
#: Interval constructions evaluated; the first is the one shipped in model bundles.
CONFORMAL_METHODS: Final[tuple[str, ...]] = ("cv", "cv_plus")
#: Feature modes: raw indicator values, or each divided by the cell's commissioning reference.
MODES: Final[tuple[str, ...]] = ("absolute", "relative")
#: Confidence level quoted by the headline, the LOCO table, the reports and the web UI
#: (``coverage_90``, ``lo_90``/``hi_90``); ``confidence_levels`` must include it.
REPORTED_CONFIDENCE: Final = 0.9

GroupBy = Literal["battery", "condition"]

_BATTERY_ALIASES: Final[Mapping[str, tuple[str, ...]]] = {
    "CORE_BATTERIES": CORE_BATTERIES,
    "EXTENDED_BATTERIES": EXTENDED_BATTERIES,
}
_EXPERIMENT_KEYS: Final = frozenset(
    {
        "description",
        "batteries",
        "group_by",
        "feature_sets",
        "post_hoc_feature_sets",
        "modes",
        "models",
        "baselines",
        "interval_models",
        "relative_keep_absolute",
        "bootstrap",
        "wilcoxon",
    }
)


@dataclass(frozen=True)
class SpecKey:
    """Identifies one evaluated configuration."""

    experiment: str
    feature_set: str
    mode: str
    model: str

    @property
    def id(self) -> str:
        """``experiment/feature_set/mode/model``."""
        return f"{self.experiment}/{self.feature_set}/{self.mode}/{self.model}"

    def as_dict(self) -> dict[str, str]:
        """The four fields as a dict."""
        return {
            "experiment": self.experiment,
            "feature_set": self.feature_set,
            "mode": self.mode,
            "model": self.model,
        }


@dataclass(frozen=True)
class ExperimentConfig:
    """One grouped cross-validation experiment.

    Attributes:
        name: Experiment key in the config.
        batteries: Cells included.
        group_by: ``"battery"`` (leave-one-battery-out) or ``"condition"``
            (leave-one-operating-condition-out).
        feature_sets: Registry feature sets evaluated.
        post_hoc_feature_sets: Sets defined after seeing results (labelled).
        modes: ``"absolute"`` and/or ``"relative"``.
        models: Zoo models plus ``ridge_nfs``.
        baselines: Baselines (registry names).
        interval_models: Models/baselines that get conformal intervals.
        relative_keep_absolute: Features left absolute in relative mode.
        bootstrap_n: Bootstrap resamples for the CI of the mean MAE (0 = none).
        bootstrap_confidence: Coverage of that CI.
        wilcoxon_reference: Model compared against the best candidate (or None).
        wilcoxon_candidates: Candidate models for the Wilcoxon comparison.
        row_filter_sets: Feature sets that define the evaluation rows and the
            relative-mode reference cycles (default: ``feature_sets``). Subsets of
            an experiment (quick mode, the audit) keep the full experiment's rows
            so their numbers equal the full run's.
    """

    name: str
    batteries: tuple[str, ...]
    group_by: GroupBy
    feature_sets: tuple[str, ...]
    post_hoc_feature_sets: frozenset[str]
    modes: tuple[str, ...]
    models: tuple[str, ...]
    baselines: tuple[str, ...]
    interval_models: frozenset[str]
    relative_keep_absolute: tuple[str, ...] = ()
    bootstrap_n: int = 0
    bootstrap_confidence: float = 0.95
    wilcoxon_reference: str | None = None
    wilcoxon_candidates: tuple[str, ...] = ()
    row_filter_sets: tuple[str, ...] = ()

    @property
    def row_sets(self) -> tuple[str, ...]:
        """Feature sets defining the evaluation rows."""
        return self.row_filter_sets or self.feature_sets


@dataclass(frozen=True)
class WindowSweepConfig:
    """Partial-window sensitivity analysis (raw data only)."""

    batteries: tuple[str, ...]
    modes: tuple[str, ...]
    discharge_v_high: tuple[float, ...]
    discharge_v_low: tuple[float, ...]
    charge_v_start: tuple[float, ...]
    charge_v_end: float


@dataclass(frozen=True)
class BenchmarkConfig:
    """Parsed ``configs/benchmark.yaml``.

    ``sha256`` is the scoped config hash (:data:`BENCHMARK_HASH_SECTIONS`, see
    :func:`~battery_soh.evaluation.benchmark.hashing.config_sections_sha256`),
    recorded as ``config_sha256`` in ``summary.json``; it ignores the
    ``production`` section and other settings that cannot change the
    benchmark's results.
    """

    path: Path
    sha256: str
    seed: int
    label: str
    n_reference_cycles: int
    confidences: tuple[float, ...]
    n_jobs: int
    features_csv: Path
    raw_dir: Path
    primary_method: str
    nested_alphas: tuple[float, ...]
    nested_min_features: int
    nested_tol: float
    headline: SpecKey
    secondary: tuple[tuple[SpecKey, str], ...]
    prediction_specs: tuple[SpecKey, ...]
    experiments: Mapping[str, ExperimentConfig]
    window_sweep: WindowSweepConfig | None
    quick_subdir: str
    quick_experiment: ExperimentConfig
    quick_secondary: tuple[tuple[SpecKey, str], ...] = field(default=())


def _str_list(value: Any, where: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, list):
        raise ValueError(f"{where} must be a list of strings")
    items = tuple(str(v) for v in value)
    if len(set(items)) != len(items):
        raise ValueError(f"{where} has duplicates: {list(items)}")
    return items


def _float_list(value: Any, where: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{where} must be a non-empty list of numbers")
    try:
        return tuple(float(v) for v in value)
    except (TypeError, ValueError):
        raise ValueError(f"{where} must contain numbers only") from None


def _batteries(value: Any, where: str) -> tuple[str, ...]:
    if isinstance(value, str):
        if value not in _BATTERY_ALIASES:
            raise ValueError(f"{where}: unknown battery list {value!r}")
        return _BATTERY_ALIASES[value]
    ids = _str_list(value, where)
    unknown = [b for b in ids if b not in PROTOCOLS]
    if unknown or len(ids) < 2:
        raise ValueError(f"{where}: need >= 2 known battery ids (unknown: {unknown})")
    return ids


def _parse_spec(raw: Any, where: str, default_experiment: str | None = None) -> SpecKey:
    entry = require_mapping(raw, where)
    check_keys(entry, {"experiment", "feature_set", "mode", "model", "label"}, where)
    experiment = entry.get("experiment", default_experiment)
    try:
        return SpecKey(
            experiment=str(experiment),
            feature_set=str(entry["feature_set"]),
            mode=str(entry["mode"]),
            model=str(entry["model"]),
        )
    except KeyError as exc:
        raise ValueError(f"{where} lacks {exc.args[0]!r}") from None


def _parse_experiment(name: str, raw: Any) -> ExperimentConfig:
    where = f"experiments.{name}"
    section = require_mapping(raw, where)
    check_keys(section, _EXPERIMENT_KEYS, where)
    feature_sets = _str_list(section.get("feature_sets"), f"{where}.feature_sets")
    unknown_sets = [s for s in feature_sets if s not in FEATURE_SETS]
    if not feature_sets or unknown_sets:
        raise ValueError(f"{where}.feature_sets: empty or unknown sets {unknown_sets}")
    post_hoc = frozenset(_str_list(section.get("post_hoc_feature_sets"), f"{where}.post_hoc"))
    if not post_hoc <= set(feature_sets):
        raise ValueError(f"{where}.post_hoc_feature_sets must be a subset of feature_sets")
    modes = _str_list(section.get("modes", ["absolute"]), f"{where}.modes")
    if not modes or any(m not in MODES for m in modes):
        raise ValueError(f"{where}.modes must be a non-empty subset of {list(MODES)}")
    models = _str_list(section.get("models"), f"{where}.models")
    bad_models = [m for m in models if m not in (*MODEL_NAMES, NESTED_MODEL)]
    if not models or bad_models:
        raise ValueError(f"{where}.models: empty or unknown models {bad_models}")
    baselines = _str_list(section.get("baselines"), f"{where}.baselines")
    bad_baselines = [b for b in baselines if b not in BASELINE_NAMES]
    if bad_baselines:
        raise ValueError(f"{where}.baselines: unknown {bad_baselines}")
    raw_interval = section.get("interval_models", "all")
    if raw_interval == "all":
        interval_models = frozenset((*models, *baselines))
    else:
        interval_models = frozenset(_str_list(raw_interval, f"{where}.interval_models"))
        if not interval_models <= {*models, *baselines}:
            raise ValueError(f"{where}.interval_models must name evaluated models")
    group_by = str(section.get("group_by", "battery"))
    if group_by not in ("battery", "condition"):
        raise ValueError(f"{where}.group_by must be 'battery' or 'condition'")
    bootstrap = require_mapping(section.get("bootstrap", {}), f"{where}.bootstrap")
    check_keys(bootstrap, {"n_boot", "confidence"}, f"{where}.bootstrap")
    wil = require_mapping(section.get("wilcoxon", {}), f"{where}.wilcoxon")
    check_keys(wil, {"reference", "candidates"}, f"{where}.wilcoxon")
    reference = wil.get("reference")
    candidates = _str_list(wil.get("candidates"), f"{where}.wilcoxon.candidates")
    if reference is not None and (
        reference not in models or not candidates or not set(candidates) <= set(models)
    ):
        raise ValueError(f"{where}.wilcoxon must compare models of the experiment")
    return ExperimentConfig(
        name=name,
        batteries=_batteries(section.get("batteries"), f"{where}.batteries"),
        group_by=cast(GroupBy, group_by),
        feature_sets=feature_sets,
        post_hoc_feature_sets=post_hoc,
        modes=modes,
        models=models,
        baselines=baselines,
        interval_models=interval_models,
        relative_keep_absolute=_str_list(
            section.get("relative_keep_absolute"), f"{where}.relative_keep_absolute"
        ),
        bootstrap_n=int(bootstrap.get("n_boot", 0)),
        bootstrap_confidence=float(bootstrap.get("confidence", 0.95)),
        wilcoxon_reference=None if reference is None else str(reference),
        wilcoxon_candidates=candidates,
    )


def _parse_window_sweep(raw: Any) -> WindowSweepConfig | None:
    if raw is None:
        return None
    section = require_mapping(raw, "window_sweep")
    check_keys(
        section,
        {"description", "requires_raw_data", "batteries", "model", "modes", "discharge", "charge"},
        "window_sweep",
    )
    if section.get("model", "ridge") != "ridge":
        raise ValueError("window_sweep.model must be 'ridge'")
    dis = require_mapping(section.get("discharge"), "window_sweep.discharge")
    chg = require_mapping(section.get("charge"), "window_sweep.charge")
    check_keys(dis, {"v_high", "v_low"}, "window_sweep.discharge")
    check_keys(chg, {"v_start", "v_end"}, "window_sweep.charge")
    modes = _str_list(section.get("modes", ["absolute"]), "window_sweep.modes")
    if not modes or any(m not in MODES for m in modes):
        raise ValueError(f"window_sweep.modes must be a non-empty subset of {list(MODES)}")
    return WindowSweepConfig(
        batteries=_batteries(section.get("batteries", "CORE_BATTERIES"), "window_sweep.batteries"),
        modes=modes,
        discharge_v_high=_float_list(dis.get("v_high"), "window_sweep.discharge.v_high"),
        discharge_v_low=_float_list(dis.get("v_low"), "window_sweep.discharge.v_low"),
        charge_v_start=_float_list(chg.get("v_start"), "window_sweep.charge.v_start"),
        charge_v_end=float(chg.get("v_end", 4.19)),
    )


def _resolve(path_value: Any, default: Path) -> Path:
    if path_value is None:
        return default
    path = Path(str(path_value))
    return path if path.is_absolute() else get_paths().root / path


def load_config(config_path: Path) -> BenchmarkConfig:
    """Parse and validate the benchmark YAML.

    Args:
        config_path: Path of ``benchmark.yaml``.

    Returns:
        The validated configuration.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: On unknown keys, unknown feature sets/models/batteries, or a
            headline/prediction spec that is not part of the experiment grid.
    """
    config_path = Path(config_path)
    raw = read_raw_config(config_path)
    check_keys(raw, TOP_KEYS, "benchmark config")
    paths = get_paths()

    confidences = _float_list(raw.get("confidence_levels", [0.8, 0.9, 0.95]), "confidence_levels")
    if any(not 0.0 < c < 1.0 for c in confidences):
        raise ValueError("confidence_levels must lie in (0, 1)")
    if not any(math.isclose(c, REPORTED_CONFIDENCE) for c in confidences):
        raise ValueError(
            f"confidence_levels must include {REPORTED_CONFIDENCE}: summary.json, the reports "
            f"and the web UI quote the {percent_label(REPORTED_CONFIDENCE)} % interval"
        )
    conformal = require_mapping(raw.get("conformal", {}), "conformal")
    check_keys(conformal, {"methods", "primary"}, "conformal")
    methods = _str_list(conformal.get("methods", list(CONFORMAL_METHODS)), "conformal.methods")
    if methods != CONFORMAL_METHODS:
        raise ValueError(f"conformal.methods must be {list(CONFORMAL_METHODS)}")
    primary = str(conformal.get("primary", "cv"))
    if primary not in CONFORMAL_METHODS:
        raise ValueError(f"conformal.primary must be one of {list(CONFORMAL_METHODS)}")
    nested = require_mapping(raw.get("nested_selection", {}), "nested_selection")
    check_keys(nested, {"alphas", "min_features", "tol", "scoring"}, "nested_selection")
    if nested.get("scoring", "inner_lobo_mae") != "inner_lobo_mae":
        raise ValueError("nested_selection.scoring must be 'inner_lobo_mae'")

    experiments_raw = require_mapping(raw.get("experiments"), "experiments")
    experiments = {name: _parse_experiment(name, sec) for name, sec in experiments_raw.items()}

    headline = _parse_spec(raw.get("headline"), "headline")
    check_in_grid(headline, experiments, "headline")
    secondary: list[tuple[SpecKey, str]] = []
    for i, entry in enumerate(raw.get("secondary") or []):
        spec = _parse_spec(entry, f"secondary[{i}]", headline.experiment)
        check_in_grid(spec, experiments, f"secondary[{i}]")
        secondary.append((spec, str(require_mapping(entry, "secondary").get("label", ""))))
    prediction_specs = tuple(
        _parse_spec(entry, f"predictions[{i}]", headline.experiment)
        for i, entry in enumerate(raw.get("predictions") or [])
    )
    for i, spec in enumerate(prediction_specs):
        check_in_grid(spec, experiments, f"predictions[{i}]")
    if headline not in prediction_specs:
        prediction_specs = (headline, *prediction_specs)

    quick_raw = require_mapping(raw.get("quick", {}), "quick")
    check_keys(
        quick_raw,
        {"description", "output_subdir", "feature_sets", "modes", "models", "baselines"},
        "quick",
    )
    base = experiments[headline.experiment]
    quick_section = {
        "batteries": list(base.batteries),
        "group_by": base.group_by,
        "feature_sets": quick_raw.get("feature_sets", [headline.feature_set]),
        "modes": quick_raw.get("modes", [headline.mode]),
        "models": quick_raw.get("models", [headline.model]),
        "baselines": quick_raw.get("baselines", list(base.baselines)),
        "relative_keep_absolute": list(base.relative_keep_absolute),
    }
    quick_experiment = replace(
        _parse_experiment(headline.experiment, quick_section), row_filter_sets=base.row_sets
    )
    check_in_grid(headline, {headline.experiment: quick_experiment}, "quick (headline)")
    quick_secondary = tuple(
        (s, label) for s, label in secondary if in_grid(s, {headline.experiment: quick_experiment})
    )

    return BenchmarkConfig(
        path=config_path,
        sha256=config_sections_sha256(raw, BENCHMARK_HASH_SECTIONS),
        seed=int(raw.get("seed", RANDOM_SEED)),
        label=str(raw.get("label", "soh_pct")),
        n_reference_cycles=int(raw.get("n_reference_cycles", DEFAULT_N_REFERENCE_CYCLES)),
        confidences=tuple(float(confidence_key(c)) for c in confidences),
        n_jobs=int(raw.get("n_jobs", 1)),
        features_csv=_resolve(raw.get("features_csv"), paths.features_csv),
        raw_dir=_resolve(raw.get("raw_dir"), paths.nasa_raw),
        primary_method=primary,
        nested_alphas=tuple(float(a) for a in nested.get("alphas", list(DEFAULT_ALPHAS))),
        nested_min_features=int(nested.get("min_features", 1)),
        nested_tol=float(nested.get("tol", DEFAULT_TOL)),
        headline=headline,
        secondary=tuple(secondary),
        prediction_specs=prediction_specs,
        experiments=experiments,
        window_sweep=_parse_window_sweep(raw.get("window_sweep")),
        quick_subdir=str(quick_raw.get("output_subdir", "quick")),
        quick_experiment=quick_experiment,
        quick_secondary=quick_secondary,
    )


def percent_label(confidence: float) -> str:
    """``0.9 -> "90"``: the percent label used in column and key names."""
    return f"{round(confidence * 100):d}"


def in_grid(spec: SpecKey, experiments: Mapping[str, ExperimentConfig]) -> bool:
    """Whether ``spec`` is one of the configurations the experiments evaluate."""
    exp = experiments.get(spec.experiment)
    if exp is None or spec.mode not in exp.modes:
        return False
    if spec.model in exp.baselines:
        return spec.feature_set == "baseline"
    return spec.feature_set in exp.feature_sets and spec.model in exp.models


def check_in_grid(spec: SpecKey, experiments: Mapping[str, ExperimentConfig], where: str) -> None:
    """Raise ``ValueError`` (naming ``where``) unless :func:`in_grid` holds."""
    if not in_grid(spec, experiments):
        raise ValueError(f"{where} {spec.id} is not part of the configured experiment grid")
