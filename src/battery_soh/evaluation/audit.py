"""Leakage audit: reproduce the v1 failure modes on the committed feature table.

v1 reported near-perfect SOH accuracy. This module rebuilds, from
``data/processed/nasa_features.csv`` alone, every number needed to explain
why, next to the honest alternatives:

1. **The label is capacity.** Within one battery ``SOH = 100 * capacity / C0``
   with ``C0`` constant, so ``R^2(soh_pct, capacity) = 1``.
2. **Contaminated split (as v1 reported it).** Ridge on the v1 inputs
   (``legacy_v1``: capacity, record time, whole-record means, cycle count)
   trained on the first 85 % of each of B0005/B0006/B0007, "tested" on *all*
   cycles of B0006 and B0007: 284 of the 336 test rows were training rows.
3. **Honest within-battery holdout.** Same model, scored on the last 15 % of
   each training battery (disjoint rows, but still batteries it has seen).
4. **Leave-one-battery-out with the v1 inputs.**
5. **Leave-one-battery-out with capacity alone** (the label numerator): not
   perfect, because the held-out cell's ``C0`` is unknown.
6. **Proxy reconstruction.** Without capacity, the remaining v1 inputs still
   rebuild capacity under leave-one-battery-out (R^2 ~ 0.99): dropping the
   capacity column does not remove the leak.
7. **v2 for comparison** - the pre-specified headline and its secondaries,
   computed with :func:`battery_soh.evaluation.benchmark.evaluate_specs`, so
   the numbers equal the benchmark's.

It also demonstrates that the v2 guards reject each v1 mistake (overlapping
splits, label-side features, leaky feature sets). Outputs: ``audit.json``
(shape consumed by the web UI: ``rows``, ``soh_vs_capacity_within_battery_r2``,
``capacity_reconstruction_r2``, ``contaminated_test_overlap``, plus details)
and ``audit_table.csv``.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from battery_soh import __version__
from battery_soh.config import RANDOM_SEED, get_paths
from battery_soh.data.protocols import CORE_BATTERIES
from battery_soh.evaluation.benchmark.config import SpecKey, load_config
from battery_soh.evaluation.benchmark.runner import evaluate_specs
from battery_soh.evaluation.errors import LeakageError
from battery_soh.evaluation.leakage import (
    check_no_leakage,
    leakage_report,
    lobo_r2,
    reconstruction_r2,
)
from battery_soh.evaluation.metrics import regression_metrics, round_significant
from battery_soh.evaluation.splits import (
    assert_disjoint,
    leave_one_group_out,
    row_keys,
    temporal_holdout,
)
from battery_soh.features.build import load_feature_table
from battery_soh.features.registry import get_feature_set
from battery_soh.models.zoo import make_model

__all__ = ["AUDIT_SCHEMA_VERSION", "V1_BATTERIES", "V1_TEST_BATTERIES", "run_audit"]

logger = logging.getLogger(__name__)

#: Version of the audit.json layout.
AUDIT_SCHEMA_VERSION: Final = "1.0"
#: Cells the v1 production model was trained on.
V1_BATTERIES: Final[tuple[str, ...]] = ("B0005", "B0006", "B0007")
#: Cells whose *full* histories formed v1's "test" file.
V1_TEST_BATTERIES: Final[tuple[str, ...]] = ("B0006", "B0007")
#: v1 trained on the first 85 % of each battery; the last 15 % is the honest holdout.
V1_HOLDOUT_FRACTION: Final = 0.15
_LABEL: Final = "soh_pct"
#: How the v2 rows name their model (the relative row is plain Ridge, not the headline's).
_MODEL_TEXT: Final[dict[str, str]] = {
    "ridge": "plain Ridge",
    "ridge_nfs": "Ridge with nested selection",
}


def _ridge() -> Any:
    return make_model("ridge", random_state=RANDOM_SEED)


def _row(
    row_id: str,
    label: str,
    category: str,
    metrics: dict[str, float],
    *,
    leaky: bool,
    protocol: str,
    feature_set: str,
    model: str,
    n_train: int | None,
    overlap: int = 0,
    note: str = "",
) -> dict[str, Any]:
    return {
        "id": row_id,
        "label": label,
        "category": category,
        "mae": round_significant(metrics["mae"]),
        "rmse": round_significant(metrics.get("rmse")),
        "r2": round_significant(metrics["r2"]),
        "leaky": leaky,
        "protocol": protocol,
        "feature_set": feature_set,
        "model": model,
        "n_train": n_train,
        "n_test": int(metrics["n"]),
        "overlapping_rows": overlap,
        "note": note,
    }


def _lobo(df: pd.DataFrame, features: Sequence[str]) -> dict[str, Any]:
    """Ridge leave-one-battery-out: mean per-battery MAE/RMSE, pooled R^2, per battery MAE."""
    groups = df["battery_id"].to_numpy(dtype=object)
    preds = np.full(len(df), np.nan)
    per: dict[str, float] = {}
    rmse: list[float] = []
    for train_idx, test_idx, held_out in leave_one_group_out(groups):
        assert_disjoint(row_keys(df.iloc[train_idx]), row_keys(df.iloc[test_idx]))
        model = _ridge().fit(df.iloc[train_idx][list(features)], df.iloc[train_idx][_LABEL])
        preds[test_idx] = model.predict(df.iloc[test_idx][list(features)])
        m = regression_metrics(df.iloc[test_idx][_LABEL], preds[test_idx])
        per[held_out] = m["mae"]
        rmse.append(m["rmse"])
    pooled = regression_metrics(df[_LABEL], preds)
    return {
        "mae": float(np.mean(list(per.values()))),
        "rmse": float(np.mean(rmse)),
        "r2": pooled["r2"],
        "n": pooled["n"],
        "per_battery_mae": per,
    }


def _guard(call: Callable[[], object]) -> dict[str, Any]:
    """Run a guard that is expected to raise; record what it raised."""
    try:
        call()
    except (LeakageError, ValueError) as exc:
        return {"raised": True, "error": type(exc).__name__, "message": str(exc)}
    return {"raised": False, "error": None, "message": ""}


def _within_battery_r2(df: pd.DataFrame) -> dict[str, Any]:
    per: dict[str, float | None] = {}
    identity_error = 0.0
    for battery in CORE_BATTERIES:
        part = df[(df["battery_id"] == battery) & df["capacity_ah"].notna() & df[_LABEL].notna()]
        r = float(np.corrcoef(part["capacity_ah"], part[_LABEL])[0, 1])
        per[battery] = round_significant(r * r)
        rebuilt = 100.0 * part["capacity_ah"] / part["c0_ah"]
        identity_error = max(identity_error, float(np.max(np.abs(rebuilt - part[_LABEL]))))
    finite = [v for v in per.values() if v is not None]
    return {
        "per_battery": per,
        "min": min(finite) if finite else None,
        "max_abs_error_soh_minus_100_capacity_over_c0": round_significant(identity_error),
        "note": (
            "Computed on the committed CSV (6 significant digits), so R^2 is 1 up to rounding "
            "and the identity error is at the rounding level."
        ),
    }


def _v2_rows(table: pd.DataFrame, config_path: Path) -> tuple[list[dict[str, Any]], str | None]:
    if not config_path.is_file():
        logger.warning("benchmark config %s not found; v2 rows skipped", config_path)
        return [], None
    cfg = load_config(config_path)
    head = cfg.headline
    candidates = [
        (head, "v2_headline", "v2 pre-specified headline: partial-window HIs, LOBO"),
        (
            SpecKey(head.experiment, head.feature_set, head.mode, "ridge"),
            "v2_same_set_plain_ridge",
            "v2 same feature set, plain Ridge, LOBO",
        ),
        (
            SpecKey(head.experiment, "discharge", head.mode, "ridge"),
            "v2_discharge_ridge",
            "v2 discharge-window HIs only, plain Ridge, LOBO",
        ),
    ]
    relative = next(
        (
            s
            for s in cfg.prediction_specs
            if s.mode == "relative" and s.experiment == head.experiment
        ),
        None,
    )
    if relative is not None:
        candidates.append(
            (
                relative,
                "v2_relative_mode",
                f"v2 relative mode, {_MODEL_TEXT.get(relative.model, relative.model)} "
                "(indicators / commissioning reference), LOBO",
            )
        )
    wanted: list[tuple[SpecKey, str, str]] = []
    for key, row_id, text in candidates:
        if key not in [k for k, _, _ in wanted]:
            wanted.append((key, row_id, text))
    keys = [k for k, _, _ in wanted]
    metrics = evaluate_specs(config_path, keys, table=table)
    rows = []
    for key, row_id, text in wanted:
        m = metrics[key.id]
        rows.append(
            {
                "id": row_id,
                "label": text,
                "category": "v2",
                "mae": m["mae"],
                "rmse": m["rmse"],
                "r2": m["r2"],
                "leaky": False,
                "protocol": "leave-one-battery-out (4 core cells)",
                "feature_set": key.feature_set,
                "model": key.model,
                "n_train": None,
                "n_test": m["n"],
                "overlapping_rows": 0,
                "note": (
                    f"{key.feature_set}/{key.mode}/{key.model}; mean over held-out batteries; "
                    "R2 pooled"
                    + ("; reference cycles not scored" if key.mode == "relative" else "")
                ),
            }
        )
    return rows, head.id


@dataclass(frozen=True)
class _V1Split:
    """v1's own protocol on B0005-7: a contaminated "test" set and the honest holdout."""

    train_keys: set[tuple[Hashable, ...]]
    contaminated_keys: list[tuple[Hashable, ...]]
    n_train: int
    n_honest: int
    overlap: int
    contaminated: dict[str, float]
    honest: dict[str, float]
    split_mae: dict[str, float | None]

    @property
    def n_contaminated(self) -> int:
        """Rows of v1's "test" file (all cycles of B0006 and B0007)."""
        return len(self.contaminated_keys)


def _v1_split(core: pd.DataFrame, legacy: list[str]) -> _V1Split:
    """(2) + (3): train on the first 85 % of B0005-7, score the contaminated and honest sets."""
    v1 = core[core["battery_id"].isin(V1_BATTERIES)].reset_index(drop=True)
    train_idx, holdout_idx = temporal_holdout(
        v1["battery_id"], v1["cycle_index"], test_fraction=V1_HOLDOUT_FRACTION
    )
    train = v1.iloc[train_idx]
    model = _ridge().fit(train[legacy], train[_LABEL])
    contaminated = v1[v1["battery_id"].isin(V1_TEST_BATTERIES)]
    train_keys = set(row_keys(train))
    contaminated_pred = model.predict(contaminated[legacy])
    m_contaminated = regression_metrics(contaminated[_LABEL], contaminated_pred)
    contaminated_keys = row_keys(contaminated)
    seen = np.array([k in train_keys for k in contaminated_keys], dtype=bool)
    split_mae = {
        "mae_on_overlapping_rows": round_significant(
            regression_metrics(contaminated[_LABEL][seen], contaminated_pred[seen])["mae"]
        ),
        "mae_on_unseen_rows": round_significant(
            regression_metrics(contaminated[_LABEL][~seen], contaminated_pred[~seen])["mae"]
        ),
    }
    honest = v1.iloc[holdout_idx]
    assert_disjoint(train_keys, row_keys(honest))
    m_honest = regression_metrics(honest[_LABEL], model.predict(honest[legacy]))
    return _V1Split(
        train_keys=train_keys,
        contaminated_keys=contaminated_keys,
        n_train=len(train),
        n_honest=len(honest),
        overlap=int(seen.sum()),
        contaminated=m_contaminated,
        honest=m_honest,
        split_mae=split_mae,
    )


def _capacity_reconstruction(core: pd.DataFrame, legacy_no_cap: list[str]) -> dict[str, Any]:
    """(6): LOBO R^2 of rebuilding capacity from the v1 inputs without capacity."""
    groups = core["battery_id"].to_numpy(dtype=object)
    X_proxy = core[legacy_no_cap].to_numpy(dtype=np.float64)
    capacity = core["capacity_ah"].to_numpy(dtype=np.float64)
    return {
        "feature_set": "legacy_v1_no_capacity",
        "features": legacy_no_cap,
        "target": "capacity_ah",
        "protocol": "leave-one-battery-out, near-OLS ridge (evaluation.leakage)",
        "r2": round_significant(reconstruction_r2(X_proxy, capacity, groups)),
        "r2_linear": round_significant(lobo_r2(X_proxy, capacity, groups)),
        "r2_log_linear": round_significant(lobo_r2(X_proxy, capacity, groups, log=True)),
    }


def _v1_rows(
    split: _V1Split,
    lobo_legacy: dict[str, Any],
    lobo_no_cap: dict[str, Any],
    lobo_capacity: dict[str, Any],
) -> list[dict[str, Any]]:
    """The five v1 rows of the audit table, in the order the web UI expects."""
    return [
        _row(
            "v1_contaminated_split",
            "v1 evaluation protocol: v1 inputs incl. capacity, 'test' = all of B0006+B0007",
            "v1_reported",
            split.contaminated,
            leaky=True,
            protocol="train first 85% of B0005-7; test all of B0006+B0007",
            feature_set="legacy_v1",
            model="ridge",
            n_train=split.n_train,
            overlap=split.overlap,
            note=f"{split.overlap} of {split.n_contaminated} test rows were training rows",
        ),
        _row(
            "v1_within_battery_holdout",
            "Same model, honest within-battery holdout (last 15% of B0005-7)",
            "honest",
            split.honest,
            leaky=True,
            protocol="train first 85% of B0005-7; test last 15% of each",
            feature_set="legacy_v1",
            model="ridge",
            n_train=split.n_train,
            note="disjoint rows, but the batteries were seen in training",
        ),
        _row(
            "v1_inputs_lobo",
            "v1 inputs incl. capacity, leave-one-battery-out",
            "honest",
            lobo_legacy,
            leaky=True,
            protocol="leave-one-battery-out (4 core cells)",
            feature_set="legacy_v1",
            model="ridge",
            n_train=None,
        ),
        _row(
            "v1_inputs_without_capacity_lobo",
            "v1 inputs without capacity (full-discharge proxies), leave-one-battery-out",
            "honest",
            lobo_no_cap,
            leaky=True,
            protocol="leave-one-battery-out (4 core cells)",
            feature_set="legacy_v1_no_capacity",
            model="ridge",
            n_train=None,
            note="the proxies still need a complete discharge",
        ),
        _row(
            "capacity_only_lobo",
            "Capacity alone (label numerator), leave-one-battery-out",
            "honest",
            lobo_capacity,
            leaky=True,
            protocol="leave-one-battery-out (4 core cells)",
            feature_set="capacity_oracle",
            model="ridge",
            n_train=None,
            note="not perfect only because each held-out cell's C0 is unknown",
        ),
    ]


def _guards(
    core: pd.DataFrame, legacy: list[str], legacy_no_cap: list[str], split: _V1Split
) -> dict[str, Any]:
    """Each v2 guard applied to the v1 mistake it exists for (every one must raise)."""
    return {
        "assert_disjoint_on_v1_split": _guard(
            lambda: assert_disjoint(split.train_keys, split.contaminated_keys)
        ),
        "check_no_leakage_legacy_v1": _guard(lambda: check_no_leakage(core, legacy)),
        "check_no_leakage_legacy_v1_no_capacity": _guard(
            lambda: check_no_leakage(core, legacy_no_cap)
        ),
        "get_feature_set_legacy_v1_without_opt_in": _guard(lambda: get_feature_set("legacy_v1")),
    }


def _leakage_table(core: pd.DataFrame, legacy: list[str]) -> list[dict[str, Any]]:
    """:func:`~battery_soh.evaluation.leakage.leakage_report` of the v1 inputs, JSON-ready."""
    report = leakage_report(core, legacy)
    return [
        {
            "feature": str(r.feature),
            "registry_risk": str(r.registry_risk),
            "max_within_battery_abs_r": round_significant(r.max_within_group_abs_r),
            "label_lobo_r2": round_significant(r.lobo_r2),
            "capacity_reconstruction_r2": round_significant(r.capacity_reconstruction_r2),
            "flagged": bool(r.flagged),
            "reasons": str(r.reasons),
        }
        for r in report.itertuples()
    ]


def _write_outputs(out_dir: Path, summary: dict[str, Any]) -> None:
    """``audit.json`` (strict JSON, LF) and ``audit_table.csv`` (the rows, 6 significant digits)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    text = json.dumps(summary, indent=2, allow_nan=False, ensure_ascii=True) + "\n"
    (out_dir / "audit.json").write_bytes(text.encode("utf-8"))
    csv_text = pd.DataFrame(summary["rows"]).to_csv(
        index=False, float_format="%.6g", lineterminator="\n"
    )
    (out_dir / "audit_table.csv").write_bytes(csv_text.encode("utf-8"))


def run_audit(features_csv: Path, out_dir: Path) -> dict[str, Any]:
    """Reproduce the v1 leakage findings and write ``audit.json`` + ``audit_table.csv``.

    Args:
        features_csv: The feature table (``data/processed/nasa_features.csv``).
        out_dir: Output directory (``reports/audit``).

    Returns:
        The audit summary (also written to ``audit.json``).

    Raises:
        FileNotFoundError: If the feature table is missing.
    """
    features_csv = Path(features_csv)
    table = load_feature_table(features_csv)
    core = table[
        table["battery_id"].isin(CORE_BATTERIES) & table["qc_valid"] & table[_LABEL].notna()
    ].reset_index(drop=True)
    legacy = list(get_feature_set("legacy_v1", allow_leaky=True))
    legacy_no_cap = list(get_feature_set("legacy_v1_no_capacity", allow_leaky=True))
    capacity_only = list(get_feature_set("capacity_oracle", allow_leaky=True))

    # (1) the label is capacity
    within = _within_battery_r2(table[table["battery_id"].isin(CORE_BATTERIES)])
    # (2) + (3) v1 split: first 85 % of B0005-7 for training
    split = _v1_split(core, legacy)
    # (4) (5) leave-one-battery-out on the four core cells
    lobo_legacy = _lobo(core, legacy)
    lobo_no_cap = _lobo(core, legacy_no_cap)
    lobo_capacity = _lobo(core, capacity_only)
    # (6) capacity reconstruction from the v1 inputs without capacity
    reconstruction = _capacity_reconstruction(core, legacy_no_cap)

    rows = _v1_rows(split, lobo_legacy, lobo_no_cap, lobo_capacity)
    # (7) v2 for comparison, computed by the benchmark code
    v2_rows, headline_id = _v2_rows(table, get_paths().configs / "benchmark.yaml")
    rows.extend(v2_rows)

    summary: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "package_version": __version__,
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "features_csv": features_csv.name,
        "features_sha256": hashlib.sha256(features_csv.read_bytes()).hexdigest(),
        "label": _LABEL,
        "rows": rows,
        "soh_vs_capacity_within_battery_r2": within["min"],
        "capacity_reconstruction_r2": reconstruction["r2"],
        "contaminated_test_overlap": {
            "overlapping_rows": split.overlap,
            "test_rows": split.n_contaminated,
        },
        "v2_headline_spec": headline_id,
        "details": {
            "soh_vs_capacity_within_battery": within,
            "capacity_reconstruction": reconstruction,
            "v1_split": {
                "train_batteries": list(V1_BATTERIES),
                "holdout_fraction": V1_HOLDOUT_FRACTION,
                "train_rows": split.n_train,
                "contaminated_test_batteries": list(V1_TEST_BATTERIES),
                "contaminated_test_rows": split.n_contaminated,
                "overlapping_rows": split.overlap,
                "honest_holdout_rows": split.n_honest,
                **split.split_mae,
            },
            "lobo_per_battery_mae": {
                "legacy_v1": {
                    k: round_significant(v) for k, v in lobo_legacy["per_battery_mae"].items()
                },
                "legacy_v1_no_capacity": {
                    k: round_significant(v) for k, v in lobo_no_cap["per_battery_mae"].items()
                },
                "capacity_oracle": {
                    k: round_significant(v) for k, v in lobo_capacity["per_battery_mae"].items()
                },
            },
            "leakage_report_legacy_v1": _leakage_table(core, legacy),
            "guards": _guards(core, legacy, legacy_no_cap, split),
            "v1_documented_claim": {
                "r2": 0.96,
                "source": "legacy/README.md ('reported R2 ~ 0.96')",
                "note": "withdrawn; measured on a test file whose rows were mostly training rows",
            },
        },
        "notes": [
            "Rows use QC-valid cycles of the four core cells (B0005, B0006, B0007, B0018).",
            "v1 rows use the legacy_v1 inputs (capacity, record time, whole-record means, cycle "
            "count) with the same Ridge as v2 (alpha=1 on standardised inputs).",
            "mae/rmse for leave-one-battery-out rows: mean over held-out batteries; r2 pooled.",
            "leaky = the inputs contain the label numerator or full-discharge proxies of it.",
        ],
    }
    _write_outputs(Path(out_dir), summary)
    logger.info(
        "audit: contaminated MAE %.3f (overlap %d/%d) vs LOBO legacy MAE %.3f",
        split.contaminated["mae"],
        split.overlap,
        split.n_contaminated,
        lobo_legacy["mae"],
    )
    return summary
