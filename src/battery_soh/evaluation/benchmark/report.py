"""Benchmark outputs: ``summary.json`` blocks, ``results.csv`` / ``predictions.csv`` frames, files.

Everything here is a pure function of the fitted :class:`SpecResult` objects
and the configuration. Floats are rounded to 6 significant digits, NaN/inf
become ``null`` and text is written with LF line endings, so a rerun on the
same inputs gives identical files apart from ``generated_utc`` and the run time.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import math
import platform
import sys
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import wilcoxon

from battery_soh import __version__
from battery_soh.config import get_paths
from battery_soh.data.protocols import PROTOCOLS
from battery_soh.evaluation.baselines import BASELINES
from battery_soh.evaluation.benchmark.config import (
    CONFORMAL_METHODS,
    NESTED_MODEL,
    REPORTED_CONFIDENCE,
    BenchmarkConfig,
    ExperimentConfig,
    SpecKey,
    in_grid,
    percent_label,
)
from battery_soh.evaluation.benchmark.fitting import SpecResult
from battery_soh.evaluation.benchmark.frames import Frames
from battery_soh.evaluation.benchmark.hashing import BENCHMARK_HASH_SECTIONS
from battery_soh.evaluation.metrics import bootstrap_ci, regression_metrics, round_significant
from battery_soh.features.build import metadata_path
from battery_soh.features.relative import REFERENCE_FLAG_COLUMN

__all__ = [
    "SCHEMA_VERSION",
    "build_summary",
    "clean",
    "find_result",
    "prediction_rows",
    "public_row",
    "results_frame",
    "write_csv",
    "write_json",
]

logger = logging.getLogger(__name__)

#: Version of the summary.json layout.
SCHEMA_VERSION: Final = "1.0"
#: ``summary.json``'s ``pre_registration.statement`` (the key name is part of the API contract).
_STATEMENT: Final = (
    "The headline specification and the nested-selection procedure were fixed in "
    "configs/benchmark.yaml before any ridge_nfs result was computed (plain-ridge results for "
    "the same feature set and mode were already known); they are reported whatever they show. "
    "Other rows are secondary or labelled post hoc."
)
_FLOAT_FORMAT: Final = "%.6g"


def _levels(cfg: BenchmarkConfig) -> list[str]:
    """Percent labels of the configured confidence levels, e.g. ``["80", "90", "95"]``."""
    return [percent_label(c) for c in cfg.confidences]


def _other_method(cfg: BenchmarkConfig) -> str:
    return "cv_plus" if cfg.primary_method == "cv" else "cv"


def clean(value: Any) -> Any:
    """Recursively round floats to 6 significant digits; NaN/inf -> None; numpy -> Python."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [clean(v) for v in value]
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        return round_significant(value)
    return value


def public_row(summary: Mapping[str, Any], cfg: BenchmarkConfig) -> dict[str, Any]:
    """Leaderboard/summary row: the shipped interval method under the plain key names."""
    row: dict[str, Any] = {
        k: summary[k]
        for k in (
            "experiment",
            "feature_set",
            "mode",
            "model",
            "n",
            "n_abstained",
            "n_batteries",
            "mae",
            "rmse",
            "r2",
            "bias",
            "worst_mae",
            "worst_battery",
            "mae_pooled",
            "mae_non_reference",
            "post_hoc",
            "leaky",
            "baseline",
        )
    }
    row["is_baseline"] = row["baseline"]
    primary, other = cfg.primary_method, _other_method(cfg)
    for level in _levels(cfg):
        row[f"coverage_{level}"] = summary.get(f"coverage_{level}_{primary}")
        row[f"mean_width_{level}"] = summary.get(f"width_{level}_{primary}")
        row[f"coverage_{level}_{other}"] = summary.get(f"coverage_{level}_{other}")
        row[f"mean_width_{level}_{other}"] = summary.get(f"width_{level}_{other}")
    return cast(dict[str, Any], clean(row))


def find_result(results: Iterable[SpecResult], key: SpecKey) -> SpecResult:
    """The result of spec ``key`` (``KeyError`` if it was not evaluated)."""
    for res in results:
        if res.spec.key == key:
            return res
    raise KeyError(key.id)


def _coverage_table(res: SpecResult, cfg: BenchmarkConfig) -> list[dict[str, Any]]:
    """Nominal vs empirical coverage of a spec (shipped method first, the other suffixed)."""
    primary, other = cfg.primary_method, _other_method(cfg)
    s = res.summary
    rows: list[dict[str, Any]] = []
    for c, level in zip(cfg.confidences, _levels(cfg), strict=True):
        rows.append(
            {
                "nominal": c,
                "method": primary,
                "empirical": s[f"coverage_{level}_{primary}"],
                "mean_width": s[f"width_{level}_{primary}"],
                "empirical_pooled": s[f"coverage_{level}_{primary}_pooled"],
                "min_battery_coverage": s[f"coverage_{level}_{primary}_min_battery"],
                f"empirical_{other}": s[f"coverage_{level}_{other}"],
                f"mean_width_{other}": s[f"width_{level}_{other}"],
                f"min_battery_coverage_{other}": s[f"coverage_{level}_{other}_min_battery"],
                "per_battery": {
                    pb["battery_id"]: {
                        "coverage": pb[f"coverage_{level}_{primary}"],
                        "mean_width": pb[f"width_{level}_{primary}"],
                        f"coverage_{other}": pb[f"coverage_{level}_{other}"],
                        f"mean_width_{other}": pb[f"width_{level}_{other}"],
                    }
                    for pb in res.per_battery
                },
            }
        )
    return cast(list[dict[str, Any]], clean(rows))


_PER_BATTERY_KEYS: Final = (
    "battery_id",
    "condition",
    "n",
    "mae",
    "rmse",
    "r2",
    "bias",
    "max_abs_error",
    "mae_non_reference",
)


def _per_battery_public(res: SpecResult, cfg: BenchmarkConfig) -> list[dict[str, Any]]:
    primary, other = cfg.primary_method, _other_method(cfg)
    rows = []
    for pb in res.per_battery:
        row: dict[str, Any] = {k: pb[k] for k in _PER_BATTERY_KEYS}
        if res.spec.intervals:
            for level in _levels(cfg):
                row[f"coverage_{level}"] = pb[f"coverage_{level}_{primary}"]
                row[f"mean_width_{level}"] = pb[f"width_{level}_{primary}"]
                row[f"coverage_{level}_{other}"] = pb[f"coverage_{level}_{other}"]
                row[f"mean_width_{level}_{other}"] = pb[f"width_{level}_{other}"]
        if pb["selected_features"]:
            row["selected_features"] = pb["selected_features"].split(";")
            row["alpha"] = pb["alpha"]
        rows.append(row)
    return cast(list[dict[str, Any]], clean(rows))


def prediction_rows(res: SpecResult, cfg: BenchmarkConfig) -> pd.DataFrame:
    """predictions.csv rows: shipped interval method as ``lo_XX``/``hi_XX``, the other suffixed."""
    oof = res.oof
    out = pd.DataFrame(
        {
            **res.spec.key.as_dict(),
            "battery_id": oof["battery_id"],
            "cycle_index": oof["cycle_index"],
            "fold": oof["fold"],
            "y": oof["y"],
            "yhat": oof["yhat"],
        }
    )
    for suffix, method in (
        ("", cfg.primary_method),
        (f"_{_other_method(cfg)}", _other_method(cfg)),
    ):
        for level in _levels(cfg):
            for bound in ("lo", "hi"):
                source = f"{bound}_{level}_{method}"
                out[f"{bound}_{level}{suffix}"] = oof.get(source, np.nan)
    out[REFERENCE_FLAG_COLUMN] = oof[REFERENCE_FLAG_COLUMN]
    out["scored"] = oof["scored"]
    return out


def _summary_predictions(res: SpecResult, cfg: BenchmarkConfig) -> list[dict[str, Any]]:
    """Scored per-cycle predictions of a spec with the shipped 80/90/95 % bounds."""
    frame = prediction_rows(res, cfg)
    frame = frame[frame["scored"]]
    values = ["y", "yhat"] + [f"{b}_{level}" for level in _levels(cfg) for b in ("lo", "hi")]
    records = []
    for battery, cycle, *numbers in frame[["battery_id", "cycle_index", *values]].itertuples(
        index=False
    ):
        rec: dict[str, Any] = {"battery_id": str(battery), "cycle_index": int(cycle)}
        for name, value in zip(values, numbers, strict=True):
            number = float(value)
            rec[name] = round(number, 4) if math.isfinite(number) else None
        records.append(rec)
    return records


def _selection_stability(res: SpecResult, *, detailed: bool = True) -> dict[str, Any]:
    """Which features each outer fold of a ``ridge_nfs`` spec selected, and how stable that is.

    ``detailed`` keeps every fold's elimination path (used for the headline only).
    """
    folds = sorted(res.selections)
    keep = ("selected", "dropped", "alpha", "inner_mae", "n_groups", "fallback")
    candidates = list(res.spec.features)
    sets = [set(res.selections[f]["selected"]) for f in folds]
    jaccard = [len(a & b) / len(a | b) for a, b in combinations(sets, 2) if a | b]
    return cast(
        dict[str, Any],
        clean(
            {
                "spec": res.spec.key.as_dict(),
                "n_folds": len(folds),
                "candidates": candidates,
                "frequency": {f: sum(f in s for s in sets) / len(sets) for f in candidates},
                "mean_pairwise_jaccard": float(np.mean(jaccard)) if jaccard else None,
                "always_selected": [f for f in candidates if all(f in s for s in sets)],
                "never_selected": [f for f in candidates if not any(f in s for s in sets)],
                "folds": [
                    {
                        "held_out": fold,
                        **(
                            res.selections[fold]
                            if detailed
                            else {k: res.selections[fold][k] for k in keep}
                        ),
                    }
                    for fold in folds
                ],
            }
        ),
    )


def _spec_brief(res: SpecResult, cfg: BenchmarkConfig, label: str = "") -> dict[str, Any]:
    row = public_row(res.summary, cfg)
    if label:
        row["label"] = label
    return row


def _headline_block(res: SpecResult, cfg: BenchmarkConfig) -> dict[str, Any]:
    row = public_row(res.summary, cfg)
    row["n_cycles"] = row["n"]
    row["conformal_method"] = cfg.primary_method
    row["pre_registered"] = True
    row["description"] = (
        "Ridge with nested feature selection: inside every outer leave-one-battery-out fold, "
        "greedy backward elimination over the feature set and the ridge penalty, scored by "
        "inner leave-one-battery-out MAE on the training cells only."
        if res.spec.key.model == NESTED_MODEL
        else "Pre-specified headline specification."
    )
    return row


def _reference_values(frames: Frames) -> dict[str, dict[str, float | None]]:
    ref = frames.reference_values
    return {
        str(b): {str(c): round_significant(v) for c, v in ref.loc[b].items()} for b in ref.index
    }


def _extended_block(
    results: list[SpecResult], exp: ExperimentConfig, frames: Frames, cfg: BenchmarkConfig
) -> dict[str, Any]:
    """Multi-condition leave-one-battery-out: per battery, per condition, bootstrap CI, Wilcoxon."""
    rows: list[dict[str, Any]] = []
    for res in results:
        pb = pd.DataFrame(res.per_battery)
        row = _spec_brief(res, cfg)
        if exp.bootstrap_n > 0 and len(pb) >= 2:
            lo, hi = bootstrap_ci(
                pb["mae"].to_numpy(dtype=np.float64),
                n_boot=exp.bootstrap_n,
                confidence=exp.bootstrap_confidence,
                random_state=cfg.seed,
            )
            row["mae_ci"] = [lo, hi]
            row["mae_ci_confidence"] = exp.bootstrap_confidence
        row["per_battery"] = {
            r["battery_id"]: {"mae": r["mae"], "n": r["n"], "condition": r["condition"]}
            for r in res.per_battery
        }
        row["per_condition"] = {
            str(cond): {
                "mae": float(part["mae"].mean()),
                "n_batteries": len(part),
                "batteries": sorted(part["battery_id"]),
            }
            for cond, part in pb.groupby("condition", sort=True)
        }
        rows.append(row)
    tests = (
        [
            _wilcoxon(results, exp, feature_set, mode)
            for mode in exp.modes
            for feature_set in exp.feature_sets
        ]
        if exp.wilcoxon_reference is not None
        else []
    )
    return cast(
        dict[str, Any],
        clean(
            {
                "experiment": exp.name,
                "group_by": exp.group_by,
                "batteries": list(exp.batteries),
                "conditions": {b: PROTOCOLS[b].condition for b in exp.batteries},
                "n_qc_valid_cycles": len(frames.absolute),
                "relative_keep_absolute": list(exp.relative_keep_absolute),
                "reference_values": _reference_values(frames),
                "bootstrap_unit": "battery",
                "results": rows,
                "wilcoxon": tests,
            }
        ),
    )


def _wilcoxon(
    results: list[SpecResult], exp: ExperimentConfig, feature_set: str, mode: str
) -> dict[str, Any]:
    """Paired Wilcoxon signed-rank test of per-battery MAE: reference vs best candidate."""
    by_model = {
        r.spec.key.model: r
        for r in results
        if r.spec.key.feature_set == feature_set and r.spec.key.mode == mode
    }
    reference = by_model[cast(str, exp.wilcoxon_reference)]
    best = min(
        (by_model[m] for m in exp.wilcoxon_candidates),
        key=lambda r: (r.summary["mae"], r.spec.key.model),
    )
    a = pd.DataFrame(reference.per_battery).set_index("battery_id")["mae"]
    b = pd.DataFrame(best.per_battery).set_index("battery_id")["mae"].reindex(a.index)
    diff = (a - b).to_numpy(dtype=np.float64)
    out: dict[str, Any] = {
        "feature_set": feature_set,
        "mode": mode,
        "reference_model": reference.spec.key.model,
        "best_candidate": best.spec.key.model,
        "candidates": list(exp.wilcoxon_candidates),
        "selection_note": (
            "best candidate = lowest mean MAE on the same folds, which favours the candidate"
        ),
        "n_pairs": int(diff.size),
        "mae_reference": reference.summary["mae"],
        "mae_best_candidate": best.summary["mae"],
        "median_difference_reference_minus_candidate": float(np.median(diff)),
        "reference_better_count": int(np.sum(diff < 0)),
        "alternative": "two-sided",
    }
    if np.all(diff == 0):
        out.update(statistic=None, p_value=None, note="identical per-battery MAE")
    else:
        test = wilcoxon(a.to_numpy(), b.to_numpy(), alternative="two-sided")
        out.update(statistic=float(test.statistic), p_value=float(test.pvalue))
    return out


def _loco_block(
    results: list[SpecResult], exp: ExperimentConfig, frames: Frames, cfg: BenchmarkConfig
) -> dict[str, Any]:
    """Leave-one-condition-out: metrics per held-out condition and per battery.

    Interval columns are reported at :data:`REPORTED_CONFIDENCE` (90 %).
    """
    primary = cfg.primary_method
    level = percent_label(REPORTED_CONFIDENCE)
    rows = []
    for res in results:
        row = _spec_brief(res, cfg)
        scored = res.oof[res.oof["scored"]]
        pb = pd.DataFrame(res.per_battery)
        per_condition: dict[str, Any] = {}
        for cond, part in pb.groupby("fold", sort=True):
            cond_rows = scored[scored["fold"] == cond]
            entry: dict[str, Any] = {
                "mae": float(part["mae"].mean()),
                "mae_pooled": regression_metrics(cond_rows["y"], cond_rows["yhat"])["mae"],
                "bias": float(part["bias"].mean()),
                "n": len(cond_rows),
                "batteries": sorted(part["battery_id"]),
            }
            if res.spec.intervals:
                entry[f"coverage_{level}"] = float(part[f"coverage_{level}_{primary}"].mean())
                entry[f"mean_width_{level}"] = float(part[f"width_{level}_{primary}"].mean())
            per_condition[str(cond)] = entry
        row["per_condition"] = per_condition
        row["per_battery"] = {
            r["battery_id"]: {"mae": r["mae"], "n": r["n"], "bias": r["bias"]}
            for r in res.per_battery
        }
        rows.append(row)
    conditions: dict[str, list[str]] = {}
    for b in exp.batteries:
        conditions.setdefault(PROTOCOLS[b].condition, []).append(b)
    return cast(
        dict[str, Any],
        clean(
            {
                "experiment": exp.name,
                "group_by": exp.group_by,
                "conditions": conditions,
                "n_qc_valid_cycles": len(frames.absolute),
                "relative_keep_absolute": list(exp.relative_keep_absolute),
                "note": (
                    "Conformal calibration groups are conditions (inner leave-one-condition-out "
                    "over the two training conditions); ridge_nfs still selects by inner "
                    "leave-one-battery-out."
                ),
                "results": rows,
            }
        ),
    )


def _leaderboard(results: list[SpecResult], cfg: BenchmarkConfig) -> list[dict[str, Any]]:
    """All specs of the headline experiment; non-leaky rows first, then by MAE."""
    rows = []
    for res in results:
        row = public_row(res.summary, cfg)
        row["headline"] = res.spec.key == cfg.headline
        rows.append(row)
    return sorted(
        rows,
        key=lambda r: (
            bool(r["leaky"]),
            math.inf if r["mae"] is None else r["mae"],
            r["feature_set"],
            r["mode"],
            r["model"],
        ),
    )


# =========================================================================== files


def write_json(path: Path, payload: Any) -> None:
    """Strict JSON (no NaN), indented, ASCII-escaped, LF line endings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, allow_nan=False, ensure_ascii=True) + "\n"
    path.write_bytes(text.encode("utf-8"))


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    """CSV with 6 significant digits and LF line endings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = frame.to_csv(index=False, float_format=_FLOAT_FORMAT, lineterminator="\n")
    path.write_bytes(text.encode("utf-8"))


def results_frame(results: Mapping[str, list[SpecResult]], cfg: BenchmarkConfig) -> pd.DataFrame:
    """``results.csv``: one row per experiment x feature set x mode x model x held-out battery."""
    rows = [pb for name in results for res in results[name] for pb in res.per_battery]
    frame = pd.DataFrame(rows)
    interval_cols = [
        f"{metric}_{percent_label(c)}_{method}"
        for method in CONFORMAL_METHODS
        for c in cfg.confidences
        for metric in ("coverage", "width")
    ]
    for col in interval_cols:
        if col not in frame:
            frame[col] = np.nan
    columns = [
        "experiment",
        "feature_set",
        "mode",
        "model",
        "kind",
        "post_hoc",
        "leaky",
        "fold",
        "battery_id",
        "condition",
        "n",
        "mae",
        "rmse",
        "r2",
        "bias",
        "max_abs_error",
        "mae_non_reference",
        *interval_cols,
        "alpha",
        "selected_features",
    ]
    return frame.loc[:, columns]


def _installed_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _provenance(cfg: BenchmarkConfig, runtime_s: float, n_jobs: int) -> dict[str, Any]:
    csv_path = cfg.features_csv
    actual = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    meta_file = metadata_path(csv_path)
    meta_sha: str | None = None
    extractor: str | None = None
    if meta_file.is_file():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        meta_sha = meta.get("sha256")
        extractor = meta.get("feature_extractor_version")
    if meta_sha is not None and meta_sha != actual:
        logger.warning("feature CSV sha256 differs from its .meta.json (stale sidecar?)")
    root = get_paths().root
    try:
        config_display = cfg.path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        config_display = cfg.path.name
    return {
        "config_path": config_display,
        "config_sha256": cfg.sha256,
        "config_sha256_sections": list(BENCHMARK_HASH_SECTIONS),
        "features_csv": csv_path.name,
        "features_sha256": actual,
        "features_sha256_meta": meta_sha,
        "feature_extractor_version": extractor,
        "package_version": __version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        # Not used to compute these results; `soh report` renders their figures with it.
        "matplotlib": _installed_version("matplotlib"),
        "platform": sys.platform,
        "seed": cfg.seed,
        "n_jobs": n_jobs,
        "runtime_s": round(runtime_s, 1),
    }


_NOTES: Final[tuple[str, ...]] = (
    "Protocol: whole batteries (or whole operating conditions for LOCO) are held out; "
    "train/test disjointness is asserted on every fold.",
    "mae, rmse, bias: mean over held-out batteries of the per-battery value (each battery counts "
    "once); r2: pooled over all scored out-of-fold predictions; worst_mae: worst battery.",
    "Coverage and mean width: mean over held-out batteries. 'cv' = full-model prediction +/- the "
    "conformal quantile of pooled leave-one-battery-out residuals of the training cells (what "
    "the production bundle ships); 'cv_plus' = group CV+ (Barber et al. 2021).",
    "Conformal calibration uses only the training batteries of each outer fold; the held-out "
    "battery never enters fitting, feature selection, penalty tuning or calibration.",
    "ridge_nfs (the pre-specified headline model) selects features and the ridge penalty by "
    "inner leave-one-battery-out MAE inside each outer training fold.",
    "Relative mode: indicators divided by the cell's own mean over its first n_reference_cycles "
    "QC-valid measurements; those reference cycles are excluded from training, calibration and "
    "scoring. mae_non_reference gives the absolute-mode MAE on the same non-reference cycles.",
    "post_hoc = feature set defined after seeing leave-one-battery-out results (diagnostic "
    "ablation, not a valid model selection). leaky = uses label-side information (sanity check).",
)


def build_summary(
    cfg: BenchmarkConfig,
    experiments: Mapping[str, ExperimentConfig],
    frames: Mapping[str, Frames],
    results: Mapping[str, list[SpecResult]],
    leakage: dict[str, Any],
    window_sweep: dict[str, Any],
    secondary: Sequence[tuple[SpecKey, str]],
    *,
    quick: bool,
    runtime_s: float,
    n_jobs: int,
) -> dict[str, Any]:
    """The ``summary.json`` document (cleaned: rounded floats, no NaN)."""
    head_exp = cfg.headline.experiment
    head_results = results[head_exp]
    headline = find_result(head_results, cfg.headline)
    baselines = {
        r.spec.key.model: clean(
            {
                "mae": r.summary["mae"],
                "rmse": r.summary["rmse"],
                "r2": r.summary["r2"],
                "leaky": r.spec.leaky,
                "label": BASELINES[r.spec.key.model].label,
            }
        )
        for r in head_results
        if r.spec.kind == "baseline" and r.spec.key.mode == cfg.headline.mode
    }
    stability: dict[str, Any] = {
        "headline": _selection_stability(headline) if headline.selections else None,
        "specs": {
            r.spec.key.id: _selection_stability(r, detailed=False)
            for name in results
            for r in results[name]
            if r.selections
        },
    }
    relative_key = next(
        (s for s in cfg.prediction_specs if s.mode == "relative" and s.experiment == head_exp),
        None,
    )
    relative_headline: dict[str, Any] | None = None
    if relative_key is not None and in_grid(relative_key, experiments):
        rel = find_result(head_results, relative_key)
        relative_headline = {
            **_spec_brief(rel, cfg),
            "per_battery": _per_battery_public(rel, cfg),
            "coverage": _coverage_table(rel, cfg) if rel.spec.intervals else [],
            "n_reference_cycles": cfg.n_reference_cycles,
            "reference_cycles_excluded": int(rel.oof[REFERENCE_FLAG_COLUMN].sum()),
        }
    skipped = {"status": "skipped", "reason": "quick mode"}
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "package_version": __version__,
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "quick": quick,
        "protocol": "leave-one-battery-out",
        "label": cfg.label,
        "config_sha256": cfg.sha256,
        "data_sha256": hashlib.sha256(cfg.features_csv.read_bytes()).hexdigest(),
        "runtime_s": round(runtime_s, 1),
        "headline": _headline_block(headline, cfg),
        "pre_registration": {
            "headline": cfg.headline.as_dict(),
            "statement": _STATEMENT,
            "nested_selection": {
                "scoring": "mean inner leave-one-battery-out MAE over the training cells",
                "search": "greedy backward elimination; stop when no removal improves",
                "alphas": list(cfg.nested_alphas),
                "min_features": cfg.nested_min_features,
                "tol": cfg.nested_tol,
            },
            "post_hoc_feature_sets": sorted(
                {s for exp in experiments.values() for s in exp.post_hoc_feature_sets}
            ),
        },
        "baselines": baselines,
        "leaderboard": _leaderboard(head_results, cfg),
        "per_battery": _per_battery_public(headline, cfg),
        "coverage": _coverage_table(headline, cfg) if headline.spec.intervals else [],
        "predictions": _summary_predictions(headline, cfg),
        "secondary": [
            _spec_brief(find_result(head_results, key), cfg, label)
            for key, label in secondary
            if in_grid(key, experiments)
        ],
        "relative_headline": relative_headline,
        "selection_stability": stability,
        "leakage_checks": leakage,
        "evaluation_rows": {
            name: {
                "n_qc_valid_cycles": len(fr.absolute),
                "n_reference_cycles_any_set": int(fr.union_reference.sum()),
                "batteries": sorted(fr.absolute["battery_id"].unique()),
            }
            for name, fr in frames.items()
        },
        "window_sweep": window_sweep,
        "extended": skipped,
        "loco": skipped,
        "diagnostics": {
            "convergence_warnings": {
                r.spec.key.id: r.convergence_warnings
                for name in results
                for r in results[name]
                if r.convergence_warnings
            }
        },
        "notes": list(_NOTES),
        "provenance": _provenance(cfg, runtime_s, n_jobs),
    }
    for name, exp in experiments.items():
        if name == head_exp:
            continue
        if exp.group_by == "condition":
            summary["loco"] = _loco_block(results[name], exp, frames[name], cfg)
        else:
            summary["extended"] = _extended_block(results[name], exp, frames[name], cfg)
    return cast(dict[str, Any], clean(summary))
