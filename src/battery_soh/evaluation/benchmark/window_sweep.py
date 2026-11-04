"""Exploratory sensitivity of the partial-window indicators to the voltage window (raw data).

Re-extracts single-window indicators from the raw ``.mat`` files for a grid
of discharge and charge windows and scores each with plain Ridge under
leave-one-battery-out. The registry windows were fixed before this sweep and
are never chosen from it; without the raw files the committed
``window_sweep.json`` is reused.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from battery_soh._typing import FloatArray
from battery_soh.config import get_paths
from battery_soh.data.nasa import find_mat_files, load_battery
from battery_soh.data.schema import BatteryRecord, Segment
from battery_soh.evaluation.benchmark.config import BenchmarkConfig, WindowSweepConfig
from battery_soh.evaluation.benchmark.frames import KEYS
from battery_soh.evaluation.benchmark.report import clean, write_json
from battery_soh.evaluation.metrics import regression_metrics, round_significant
from battery_soh.evaluation.splits import assert_groups_disjoint
from battery_soh.features.charge import charge_features
from battery_soh.features.discharge import discharge_window_features
from battery_soh.features.relative import REFERENCE_FLAG_COLUMN, relativize
from battery_soh.models.zoo import make_model

__all__ = ["run_window_sweep", "window_sweep_section"]

logger = logging.getLogger(__name__)


def _window_id(prefix: str, high: float, low: float) -> str:
    return f"{prefix}_{round(high * 100):d}_{round(low * 100):d}"


def _cc_current(seg: Segment) -> float:
    """Median charging current on the CC plateau (A); NaN if not identifiable."""
    current = np.asarray(seg.current_a, dtype=np.float64)
    current = current[np.isfinite(current)]
    if current.size == 0:
        return math.nan
    high = float(np.percentile(current, 95))
    if high <= 0:
        return math.nan
    plateau = current[current > 0.5 * high]
    return float(np.median(plateau)) if plateau.size else math.nan


def _sweep_rows(record: BatteryRecord, cfg: WindowSweepConfig) -> list[dict[str, Any]]:
    """Window features per discharge, pairing charges exactly like ``features.build.cycle_rows``."""
    rows: list[dict[str, Any]] = []
    pending: Segment | None = None
    for seg in record.segments:
        if seg.kind == "impedance":
            continue
        if seg.kind == "charge":
            pending = seg
            continue
        charge = pending if pending is not None and pending.n_samples else None
        pending = None
        row: dict[str, Any] = {"battery_id": record.battery_id, "cycle_index": len(rows)}
        for high in cfg.discharge_v_high:
            for low in cfg.discharge_v_low:
                wid = _window_id("dis", high, low)
                values = discharge_window_features(seg, v_high=high, v_low=low)
                row[f"{wid}__q_ah"] = values["dis_q_380_350_ah"]
                row[f"{wid}__temp_rise_c"] = values["dis_temp_rise_win_c"]
                row[f"{wid}__window_ah"] = values["dis_q_380_350_ah"]
        i_cc = _cc_current(charge) if charge is not None else math.nan
        for start in cfg.charge_v_start:
            wid = _window_id("chg", start, cfg.charge_v_end)
            if charge is None:
                values = {"chg_cc_time_390_419_s": math.nan, "chg_temp_rise_cc_c": math.nan}
            else:
                values = charge_features(charge, v_start=start, v_end=cfg.charge_v_end)
            seconds = values["chg_cc_time_390_419_s"]
            row[f"{wid}__time_s"] = seconds
            row[f"{wid}__temp_rise_c"] = values["chg_temp_rise_cc_c"]
            row[f"{wid}__window_ah"] = i_cc * seconds / 3600.0
        rows.append(row)
    return rows


def _check_reproduces(swept: FloatArray, table: FloatArray, registry: str) -> float | None:
    """Largest relative difference between a re-extracted window and the table's column.

    The sweep re-extracts the registry windows from the raw files; they must be
    measurable on exactly the same cycles as in the feature table and agree to
    within 1e-5 relative (identical values, zeros included, count as 0). A
    comparison that cannot be made (no cycle where both are finite, or a
    non-finite relative difference) fails instead of passing.

    Raises:
        RuntimeError: If the values do not reproduce the table.
    """
    finite = np.isfinite(swept)
    if not np.array_equal(finite, np.isfinite(table)):
        raise RuntimeError(
            f"window sweep does not reproduce {registry} of the table (measurable on other cycles)"
        )
    if not finite.any():
        raise RuntimeError(f"window sweep cannot check {registry}: no cycle where it is measured")
    a, b = swept[finite], table[finite]
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = np.where(a == b, 0.0, np.abs(a - b) / np.abs(b))
    largest = float(np.max(relative))
    if not math.isfinite(largest) or largest > 1e-5:
        raise RuntimeError(
            f"window sweep does not reproduce {registry} of the table "
            f"(max relative difference {largest:.3g})"
        )
    return round_significant(largest)


def _lobo_ridge(frame: pd.DataFrame, features: list[str], label: str, seed: int) -> dict[str, Any]:
    per: dict[str, float] = {}
    preds = np.full(len(frame), np.nan)
    batteries = frame["battery_id"].to_numpy(dtype=object)
    fit_ok = ~frame[REFERENCE_FLAG_COLUMN].to_numpy(dtype=bool)
    scored = fit_ok
    for battery in sorted(pd.unique(batteries)):
        test = batteries == battery
        train = ~test & fit_ok
        assert_groups_disjoint(batteries[train], batteries[test])
        model = make_model("ridge", random_state=seed)
        model.fit(frame.loc[train, features], frame.loc[train, label])
        preds[test] = model.predict(frame.loc[test, features])
        mask = test & scored
        per[str(battery)] = regression_metrics(frame.loc[mask, label], preds[mask])["mae"]
    pooled = regression_metrics(frame.loc[scored, label], preds[scored])
    return {
        "mae": float(np.mean(list(per.values()))),
        "rmse_pooled": pooled["rmse"],
        "r2": pooled["r2"],
        "n": int(pooled["n"]),
        "per_battery_mae": per,
    }


def run_window_sweep(
    cfg: WindowSweepConfig, table: pd.DataFrame, raw_dir: Path, *, label: str, n_ref: int, seed: int
) -> dict[str, Any]:
    """Ridge LOBO accuracy and capacity coverage of alternative partial windows.

    For each window, the model sees only that window's indicators (discharge:
    Ah delivered and temperature rise; charge: CC time and temperature rise).
    Rows are the QC-valid core cycles on which the window is measurable; the
    availability (share of QC-valid cycles where it is) is reported with the
    MAE because a wider window is more informative but observable less often.
    The capacity fraction uses the measured capacity only as a descriptive
    denominator, never as a model input. Exploratory: the headline windows
    were fixed before this sweep and are not chosen from it.

    Raises:
        FileNotFoundError: If the raw files of a battery are missing.
    """
    files = find_mat_files(raw_dir)
    missing = [b for b in cfg.batteries if b not in files]
    if missing:
        raise FileNotFoundError(f"no raw .mat files for {missing} under {raw_dir}")
    rows: list[dict[str, Any]] = []
    for battery in cfg.batteries:
        rows.extend(_sweep_rows(load_battery(files[battery], battery), cfg))
    sweep = pd.DataFrame(rows)
    base_cols = [
        *KEYS,
        "qc_valid",
        label,
        "capacity_ah",
        "dis_q_380_350_ah",
        "chg_cc_time_390_419_s",
    ]
    merged = (
        table.loc[table["battery_id"].isin(cfg.batteries), base_cols]
        .merge(sweep, on=KEYS, how="inner", validate="one_to_one")
        .sort_values(KEYS, kind="stable")
        .reset_index(drop=True)
    )
    if len(merged) != int(table["battery_id"].isin(cfg.batteries).sum()):
        raise RuntimeError("window sweep cycles do not align with the feature table")

    consistency: dict[str, float | None] = {}
    for column, registry in (
        (f"{_window_id('dis', 3.8, 3.5)}__q_ah", "dis_q_380_350_ah"),
        (f"{_window_id('chg', 3.9, cfg.charge_v_end)}__time_s", "chg_cc_time_390_419_s"),
    ):
        if column in merged:
            consistency[registry] = _check_reproduces(
                merged[column].to_numpy(dtype=np.float64),
                merged[registry].to_numpy(dtype=np.float64),
                registry,
            )

    windows: list[tuple[str, str, dict[str, float], list[str], str]] = []
    for high in cfg.discharge_v_high:
        for low in cfg.discharge_v_low:
            wid = _window_id("dis", high, low)
            windows.append(
                (
                    wid,
                    "discharge",
                    {"v_high": high, "v_low": low},
                    [f"{wid}__q_ah", f"{wid}__temp_rise_c"],
                    f"{wid}__q_ah",
                )
            )
    for start in cfg.charge_v_start:
        wid = _window_id("chg", start, cfg.charge_v_end)
        windows.append(
            (
                wid,
                "charge",
                {"v_start": start, "v_end": cfg.charge_v_end},
                [f"{wid}__time_s", f"{wid}__temp_rise_c"],
                f"{wid}__time_s",
            )
        )

    valid = merged["qc_valid"].to_numpy(dtype=bool) & np.isfinite(merged[label].to_numpy(float))
    cell = merged["battery_id"].to_numpy(dtype=object)
    out_windows: list[dict[str, Any]] = []
    for wid, kind, bounds, features, primary in windows:
        measured = valid & np.isfinite(merged[primary].to_numpy(dtype=np.float64))
        fraction = merged.loc[measured, f"{wid}__window_ah"] / merged.loc[measured, "capacity_ah"]
        entry: dict[str, Any] = {
            "id": wid,
            "kind": kind,
            **bounds,
            "features": [f.split("__", 1)[1] for f in features],
            "availability": float(measured.sum() / valid.sum()),
            "availability_per_battery": {
                str(b): float(measured[cell == b].sum() / max(int(valid[cell == b].sum()), 1))
                for b in cfg.batteries
            },
            "capacity_fraction_mean": float(fraction.mean()) if len(fraction) else None,
        }
        relative_all = relativize(merged, features, n_ref=n_ref)
        for mode in cfg.modes:
            source = merged if mode == "absolute" else relative_all
            frame = source.loc[measured, [*KEYS, label, *features]].copy()
            frame[REFERENCE_FLAG_COLUMN] = (
                relative_all.loc[measured, REFERENCE_FLAG_COLUMN].to_numpy(bool)
                if mode == "relative"
                else False
            )
            frame = frame.reset_index(drop=True)
            if frame["battery_id"].nunique() < 2:
                entry[mode] = None
                continue
            entry[mode] = _lobo_ridge(frame, features, label, seed)
        out_windows.append(entry)

    best = {
        mode: min(
            (w for w in out_windows if w.get(mode)),
            key=lambda w: (w[mode]["mae"], w["id"]),
        )["id"]
        for mode in cfg.modes
    }
    return cast(
        dict[str, Any],
        clean(
            {
                "status": "ok",
                "requires_raw_data": True,
                "batteries": list(cfg.batteries),
                "model": "ridge",
                "label": label,
                "n_reference_cycles": n_ref,
                "note": (
                    "Exploratory sensitivity analysis. The registry windows (discharge "
                    "3.80->3.50 V, charge 3.90->4.19 V) were fixed before this sweep and are not "
                    "selected from it. Rows: QC-valid core cycles where the window is measurable; "
                    "relative mode excludes each cell's reference cycles from fitting and scoring."
                ),
                "consistency_max_rel_diff": consistency,
                "windows": out_windows,
                "lowest_mae_window": best,
            }
        ),
    )


def _display(path: Path) -> str:
    """``path`` relative to the project root (POSIX), else its name: never a machine path."""
    try:
        return Path(path).resolve().relative_to(get_paths().root.resolve()).as_posix()
    except ValueError:
        return Path(path).name


def _window_ids(cfg: WindowSweepConfig) -> list[str]:
    """Window ids in the order :func:`run_window_sweep` reports them."""
    return [
        *(_window_id("dis", h, lo) for h in cfg.discharge_v_high for lo in cfg.discharge_v_low),
        *(_window_id("chg", s, cfg.charge_v_end) for s in cfg.charge_v_start),
    ]


def _cached_sweep_matches(data: Any, cfg: WindowSweepConfig, *, label: str, n_ref: int) -> bool:
    """Whether a stored sweep was computed for this window grid, cells, label and reference."""
    if not isinstance(data, dict) or data.get("status") != "ok":
        return False
    windows = data.get("windows") or []
    return (
        data.get("batteries") == list(cfg.batteries)
        and data.get("label") == label
        and data.get("n_reference_cycles") == n_ref
        and [w.get("id") for w in windows] == _window_ids(cfg)
        and all(mode in w for w in windows for mode in cfg.modes)
    )


def window_sweep_section(
    cfg: BenchmarkConfig, table: pd.DataFrame, out_dir: Path
) -> dict[str, Any]:
    """The ``window_sweep`` block of ``summary.json``.

    Computed from the raw data when every configured cell's ``.mat`` file is
    present, and written to ``out_dir/window_sweep.json``. Without the raw
    files, a stored sweep for the same window grid is reused: first
    ``out_dir/window_sweep.json``, then the committed
    ``reports/benchmark/window_sweep.json`` (copied to ``out_dir``). Otherwise
    the block says it was skipped. Paths in the block are repo-relative.
    """
    sweep_cfg = cfg.window_sweep
    if sweep_cfg is None:
        return {"status": "skipped", "reason": "not configured"}
    target = Path(out_dir) / "window_sweep.json"
    try:
        files = find_mat_files(cfg.raw_dir)
    except FileNotFoundError:
        files = {}
    if all(b in files for b in sweep_cfg.batteries):
        logger.info("window sweep: extracting %s from raw data", list(sweep_cfg.batteries))
        result = run_window_sweep(
            sweep_cfg,
            table,
            cfg.raw_dir,
            label=cfg.label,
            n_ref=cfg.n_reference_cycles,
            seed=cfg.seed,
        )
        write_json(target, result)
        return result
    missing = f"raw .mat files not found under {_display(cfg.raw_dir)}"
    committed = get_paths().benchmark_dir / "window_sweep.json"
    candidates = [target] if committed.resolve() == target.resolve() else [target, committed]
    for cached in candidates:
        if not cached.is_file():
            continue
        stored = cached.read_bytes()
        data = json.loads(stored.decode("utf-8"))
        if not _cached_sweep_matches(
            data, sweep_cfg, label=cfg.label, n_ref=cfg.n_reference_cycles
        ):
            logger.warning("%s holds a sweep of another window grid; not reused", _display(cached))
            continue
        logger.info("%s; reusing %s", missing, _display(cached))
        if cached is not target:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(stored)
        data["source"] = f"cached (raw data unavailable): {_display(cached)}"
        return cast(dict[str, Any], data)
    return {"status": "skipped", "reason": missing}
