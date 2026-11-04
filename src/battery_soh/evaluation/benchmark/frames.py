"""Evaluation rows of an experiment, the specs evaluated on them, and the leakage gate.

Rows: QC-valid cycles with a finite label on which every row-defining
feature set of the experiment has at least one measured (non-context)
indicator, in both feature modes. Relative mode divides each indicator by the
cell's own mean over its first ``n_reference_cycles`` QC-valid measurements
and flags the cycles that entered a reference; those are never used for
training, calibration or scoring.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from battery_soh.data.protocols import PROTOCOLS
from battery_soh.evaluation.baselines import BASELINES
from battery_soh.evaluation.benchmark.config import ExperimentConfig, SpecKey
from battery_soh.evaluation.leakage import SET_ROW, check_no_leakage
from battery_soh.evaluation.metrics import round_significant
from battery_soh.features.build import KEY_COLUMNS
from battery_soh.features.registry import get_feature_set
from battery_soh.features.relative import (
    REFERENCE_FLAG_COLUMN,
    is_context_feature,
    reference_table,
    relativize,
)

__all__ = [
    "KEYS",
    "Frames",
    "Spec",
    "build_specs",
    "leakage_checks",
    "prepare_frames",
    "set_features",
    "spec_frame",
]

#: Row key columns (``battery_id``, ``cycle_index``).
KEYS: Final[list[str]] = list(KEY_COLUMNS)


@dataclass(frozen=True)
class Frames:
    """QC-valid labelled cycles of one experiment, in both feature modes (identical rows).

    Attributes:
        absolute: Absolute feature values plus ``condition`` and ``fold`` columns.
        relative: Same rows, relativized health indicators.
        reference_masks: Per relativized indicator, the rows that entered its reference.
        union_reference: Rows that are reference cycles of any indicator of the
            experiment's feature sets (used for the baselines in relative mode).
        reference_values: Per-cell reference of every relativized indicator.
    """

    absolute: pd.DataFrame
    relative: pd.DataFrame
    reference_masks: Mapping[str, NDArray[np.bool_]]
    union_reference: NDArray[np.bool_]
    reference_values: pd.DataFrame

    def frame(self, mode: str) -> pd.DataFrame:
        """The rows in ``mode`` (``"absolute"`` or ``"relative"``)."""
        return self.absolute if mode == "absolute" else self.relative


@dataclass(frozen=True)
class Spec:
    """One evaluated configuration with its inputs and labels.

    Attributes:
        key: Experiment / feature set / mode / model.
        kind: ``"model"`` or ``"baseline"``.
        features: Input columns.
        leaky: Uses label-side information (labelled sanity check).
        post_hoc: Feature set defined after seeing results (labelled ablation).
        intervals: Conformal intervals are computed.
    """

    key: SpecKey
    kind: Literal["model", "baseline"]
    features: tuple[str, ...]
    leaky: bool
    post_hoc: bool
    intervals: bool


def set_features(name: str) -> tuple[str, ...]:
    """Features of a non-leaky set (raises for leaky sets)."""
    return get_feature_set(name)


def _model_features(exp: ExperimentConfig, model: str, feature_set: str) -> tuple[str, ...]:
    if model in exp.baselines:
        spec = BASELINES[model]
        if spec.feature_set is None:
            return ("ctx_cycle_index",)  # the mean baseline ignores its input
        return get_feature_set(spec.feature_set, allow_leaky=spec.leaky)
    return set_features(feature_set)


def _relativizable(features: Iterable[str], keep_absolute: Sequence[str]) -> list[str]:
    return list(
        dict.fromkeys(f for f in features if not is_context_feature(f) and f not in keep_absolute)
    )


def prepare_frames(table: pd.DataFrame, exp: ExperimentConfig, label: str, n_ref: int) -> Frames:
    """Evaluation rows of ``exp`` in both modes, with the relative-mode reference flags.

    Raises:
        ValueError: If a battery of the experiment is not in ``table``.
    """
    missing = [b for b in exp.batteries if b not in set(table["battery_id"])]
    if missing:
        raise ValueError(f"{exp.name}: batteries {missing} are not in the feature table")
    cells = table[table["battery_id"].isin(exp.batteries)]
    cells = cells.sort_values(KEYS, kind="stable").reset_index(drop=True)
    row_set_his = _relativizable(
        (f for s in exp.row_sets for f in set_features(s)), exp.relative_keep_absolute
    )
    every_hi = _relativizable(
        [
            *row_set_his,
            *(f for s in exp.feature_sets for f in set_features(s)),
            *(f for b in exp.baselines for f in _model_features(exp, b, "baseline")),
        ],
        exp.relative_keep_absolute,
    )
    # References come from each cell's own early QC-valid measurements (no labels, no other
    # cells), so computing them on the whole cell before splitting leaks nothing. A reference
    # is per indicator, so the values do not depend on which other features are relativized.
    relative_all = relativize(cells, every_hi, n_ref=n_ref)
    masks_all = {
        f: relativize(cells, [f], n_ref=n_ref)[REFERENCE_FLAG_COLUMN].to_numpy(dtype=bool)
        for f in every_hi
    }
    reference_values = reference_table(cells, row_set_his, n_ref=n_ref)

    rows = cells["qc_valid"].to_numpy(dtype=bool) & np.isfinite(cells[label].to_numpy(float))
    union = np.zeros(len(cells), dtype=bool)
    for f in row_set_his:
        union |= masks_all[f]

    def _finish(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.loc[rows].drop(columns=[REFERENCE_FLAG_COLUMN], errors="ignore")
        out["condition"] = [PROTOCOLS[b].condition for b in out["battery_id"]]
        out["fold"] = out["battery_id"] if exp.group_by == "battery" else out["condition"]
        return out.reset_index(drop=True)

    return Frames(
        absolute=_finish(cells),
        relative=_finish(relative_all),
        reference_masks={f: m[rows] for f, m in masks_all.items()},
        union_reference=union[rows],
        reference_values=reference_values,
    )


def build_specs(exp: ExperimentConfig) -> list[Spec]:
    """Every (mode x feature set x model) and (mode x baseline) spec of ``exp``, in run order."""
    specs: list[Spec] = []
    for mode in exp.modes:
        for feature_set in exp.feature_sets:
            for model in exp.models:
                specs.append(
                    Spec(
                        key=SpecKey(exp.name, feature_set, mode, model),
                        kind="model",
                        features=_model_features(exp, model, feature_set),
                        leaky=False,
                        post_hoc=feature_set in exp.post_hoc_feature_sets,
                        intervals=model in exp.interval_models,
                    )
                )
        for baseline in exp.baselines:
            specs.append(
                Spec(
                    key=SpecKey(exp.name, "baseline", mode, baseline),
                    kind="baseline",
                    features=_model_features(exp, baseline, "baseline"),
                    leaky=BASELINES[baseline].leaky,
                    post_hoc=False,
                    intervals=baseline in exp.interval_models,
                )
            )
    return specs


def spec_frame(spec: Spec, frames: Frames) -> tuple[pd.DataFrame, int]:
    """Rows a spec is evaluated on, with its relative-mode reference flag.

    A model is fitted and scored on the cycles where at least one of its own
    (non-context) inputs was measured; the other cycles are abstentions
    (counted, not imputed). The reference flag is the union over the spec's
    relativized inputs; baselines use the experiment-wide union so they are
    scored on the same cycles in both modes.

    Returns:
        ``(frame, n_abstained)``.
    """
    base = frames.frame(spec.key.mode)
    measured_cols = [f for f in spec.features if not is_context_feature(f)]
    if spec.kind == "baseline" or not measured_cols:
        measured = np.ones(len(base), dtype=bool)
    else:
        measured = np.isfinite(base[measured_cols].to_numpy(dtype=np.float64)).any(axis=1)
    if spec.kind == "baseline":
        reference = frames.union_reference
    else:
        reference = np.zeros(len(base), dtype=bool)
        for f in measured_cols:
            if f in frames.reference_masks:
                reference = reference | frames.reference_masks[f]
    out = base.loc[measured].copy()
    out[REFERENCE_FLAG_COLUMN] = reference[measured]
    return out.reset_index(drop=True), int((~measured).sum())


def leakage_checks(
    frames: Mapping[str, Frames], experiments: Mapping[str, ExperimentConfig], label: str
) -> dict[str, Any]:
    """Run :func:`check_no_leakage` on every non-leaky set (raises LeakageError on failure)."""
    out: dict[str, Any] = {}
    for name, exp in experiments.items():
        frame = frames[name].absolute
        for feature_set in exp.feature_sets:
            report = check_no_leakage(frame, list(set_features(feature_set)), label)
            row = report[report["feature"] == SET_ROW].iloc[0]
            out[f"{name}/{feature_set}"] = {
                "passed": True,
                "max_within_battery_abs_r_with_label": round_significant(
                    row["max_within_group_abs_r"]
                ),
                "capacity_reconstruction_lobo_r2": round_significant(
                    row["capacity_reconstruction_r2"]
                ),
            }
    return out
