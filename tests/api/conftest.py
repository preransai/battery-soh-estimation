"""Fixtures for the API tests: real bundles fitted on synthetic cells, demo data, reports.

Everything is built from :mod:`battery_soh.synthetic` (a physical cell simulator
with known ground truth), so the bundles, the demo asset and the extracted
features are realistic and in range without the 200 MB NASA download.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from battery_soh.api.app import create_app
from battery_soh.api.demo import demo_document, write_demo_assets
from battery_soh.api.settings import Settings
from battery_soh.data.schema import BatteryRecord, Segment
from battery_soh.features.extract import features_from_segments
from battery_soh.features.registry import FEATURE_SETS
from battery_soh.features.relative import relativize
from battery_soh.models.linear_bundle import export_linear_bundle, save_bundle, write_manifest
from battery_soh.models.zoo import make_model
from battery_soh.synthetic import SENSOR_NOISE, CellModel, simulate_charge, simulate_discharge

CREATED = "2026-01-01T00:00:00+00:00"
QUANTILES = {"0.8": 1.5, "0.9": 2.0, "0.95": 2.6}
#: (model_id, feature set, mode, recorded LOBO MAE)
BUNDLE_SPECS = (
    ("combined-absolute-ridge", "combined", "absolute", 1.4),
    ("discharge-absolute-ridge", "discharge", "absolute", 1.6),
    ("charge-absolute-ridge", "charge", "absolute", 2.7),
    ("combined-relative-ridge", "combined", "relative", 0.9),
)
N_REF = 2


def simulate_cycle(
    capacity_ah: float, soh: float, r0_ohm: float, seed: int
) -> tuple[Segment, Segment]:
    """One noisy NASA-protocol charge + 2 A discharge of an aged synthetic cell."""
    cell = CellModel(capacity_ah=capacity_ah, r0_ohm=r0_ohm).aged(soh)
    charge = simulate_charge(cell, sample_s=10.0, noise=SENSOR_NOISE, seed=seed)
    discharge = simulate_discharge(cell, sample_s=10.0, noise=SENSOR_NOISE, seed=seed + 1)
    return charge.segment, discharge.segment


CELLS = {  # battery id -> (initial capacity Ah, initial R0 ohm); ids must be NASA ids
    "B0005": (2.00, 0.070),
    "B0006": (1.95, 0.074),
    "B0007": (2.05, 0.067),
}
SOH_PATH = np.linspace(1.0, 0.76, 9)


def _eis(soh: float) -> tuple[float, float]:
    return 0.045 + 0.04 * (1 - soh), 0.07 + 0.08 * (1 - soh)


@pytest.fixture(scope="session")
def synthetic_records() -> dict[str, BatteryRecord]:
    """Three synthetic cells, nine cycles each (impedance, charge, discharge per cycle)."""
    records = {}
    for b, (battery_id, (cap, r0)) in enumerate(CELLS.items()):
        segments: list[Segment] = []
        for k, soh in enumerate(SOH_PATH):
            charge, discharge = simulate_cycle(cap, float(soh), r0, seed=100 * b + 2 * k)
            re_ohm, rct_ohm = _eis(float(soh))
            segments += [Segment(kind="impedance", re_ohm=re_ohm, rct_ohm=rct_ohm)]
            segments += [charge, discharge]
        records[battery_id] = BatteryRecord(battery_id=battery_id, segments=tuple(segments))
    return records


@pytest.fixture(scope="session")
def training_frame(synthetic_records: dict[str, BatteryRecord]) -> pd.DataFrame:
    """Feature table of the synthetic cells (label: 100 * capacity / C0)."""
    rows = []
    for battery_id, record in synthetic_records.items():
        segs = record.segments
        for k in range(len(SOH_PATH)):
            imp, charge, discharge = segs[3 * k : 3 * k + 3]
            row: dict[str, Any] = features_from_segments(
                charge, discharge, eis=(imp.re_ohm or 0.0, imp.rct_ohm or 0.0)
            )
            row.update(battery_id=battery_id, cycle_index=k, qc_valid=True)
            row["soh_pct"] = 100.0 * float(SOH_PATH[k])
            rows.append(row)
    return pd.DataFrame(rows)


def _metrics(mae: float) -> dict[str, Any]:
    return {
        "protocol": "leave-one-battery-out",
        "per_battery": {b: {"mae": mae, "rmse": mae * 1.3, "r2": 0.95} for b in CELLS},
        "mean": {"mae": mae, "rmse": round(mae * 1.3, 4), "r2": 0.95},
        "coverage": {"0.8": 0.81, "0.9": 0.9, "0.95": 0.94},
    }


def write_bundles(directory: Path, frame: pd.DataFrame) -> Path:
    """Fit and export the :data:`BUNDLE_SPECS` bundles plus MANIFEST.json into ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    for model_id, feature_set, mode, mae in BUNDLE_SPECS:
        features = list(FEATURE_SETS[feature_set])
        data = frame
        quantiles: dict[str, float] = dict(QUANTILES)
        if mode == "relative":
            data = relativize(frame, features, n_ref=N_REF)
            data = data[~data["is_reference_cycle"]].reset_index(drop=True)
            quantiles["0.95"] = float("inf")  # too few scores: unbounded interval
        pipeline = make_model("ridge", alpha=1.0).fit(data[features], data["soh_pct"])
        bundle = export_linear_bundle(
            pipeline,
            model_id=model_id,
            feature_set=feature_set,
            mode=mode,  # type: ignore[arg-type]
            features=features,
            training_df=data,
            conformal_quantiles=quantiles,
            metrics=_metrics(mae),
            n_conformal_scores=len(data),
            n_reference_cycles=N_REF,
            created_utc=CREATED,
        )
        save_bundle(bundle, directory / f"{model_id}.json")
    write_manifest(directory, created_utc=CREATED)
    return directory


@pytest.fixture(scope="session")
def models_dir(tmp_path_factory: pytest.TempPathFactory, training_frame: pd.DataFrame) -> Path:
    return write_bundles(tmp_path_factory.mktemp("models"), training_frame)


@pytest.fixture(scope="session")
def demo_dir(
    tmp_path_factory: pytest.TempPathFactory, synthetic_records: dict[str, BatteryRecord]
) -> Path:
    out = tmp_path_factory.mktemp("demo")
    records = [synthetic_records["B0005"], synthetic_records["B0006"]]
    write_demo_assets(demo_document(records, every=4, max_points=120, n_ref=N_REF), out)
    return out


HEADLINE = {
    "experiment": "core_lobo",
    "feature_set": "combined",
    "mode": "absolute",
    "model": "ridge",
    "mae": 1.4,
    "rmse": 1.8,
    "r2": 0.95,
    "coverage_90": 0.9,
    "n_batteries": 3,
}


@pytest.fixture(scope="session")
def reports_dirs(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """``(benchmark_dir, audit_dir)`` with a summary, predictions for two specs and an audit."""
    root = tmp_path_factory.mktemp("reports")
    bench, audit = root / "benchmark", root / "audit"
    bench.mkdir()
    audit.mkdir()
    summary = {
        "protocol": "leave-one-battery-out",
        "headline": HEADLINE,
        "leaderboard": [
            {**HEADLINE, "n": 27},
            {"experiment": "core_lobo", "feature_set": "discharge", "mode": "absolute",
             "model": "ridge", "mae": 1.6, "rmse": float("nan")},
        ],
        "per_battery": [{"battery_id": b, "mae": 1.4} for b in CELLS],
        "coverage": [{"nominal": 0.9, "coverage": 0.9, "mean_width": 4.0}],
    }  # fmt: skip
    # json.dumps writes NaN as a bare token, as a careless writer would.
    (bench / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    columns = [
        "experiment", "feature_set", "mode", "model", "battery_id", "cycle_index", "y", "yhat",
        "lo_80", "hi_80", "lo_90", "hi_90", "lo_95", "hi_95", "is_reference_cycle",
    ]  # fmt: skip
    with (bench / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for feature_set in ("combined", "discharge"):
            for battery_id in CELLS:
                for k in range(3):
                    y, yhat = 100.0 - 3 * k, 99.5 - 3 * k
                    writer.writerow(
                        ["core_lobo", feature_set, "absolute", "ridge", battery_id, k, y, yhat,
                         yhat - 1.5, yhat + 1.5, yhat - 2, yhat + 2, yhat - 2.6, yhat + 2.6,
                         k == 0]
                    )  # fmt: skip
    (audit / "audit.json").write_text(
        json.dumps({"rows": [{"id": "v1_reported", "mae": 0.9, "r2": 0.99}]}), encoding="utf-8"
    )
    return bench, audit


@pytest.fixture(scope="session")
def frontend_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("frontend")
    (root / "assets" / "js").mkdir(parents=True)
    (root / "index.html").write_text(
        '<!doctype html><title>t</title><script type="module" src="assets/js/app.js"></script>',
        encoding="utf-8",
    )
    (root / "assets" / "config.js").write_text(
        'window.SOH_CONFIG = Object.freeze({ mode: "api" });\n', encoding="utf-8"
    )
    (root / "assets" / "js" / "app.js").write_text("export const x = 1;\n", encoding="utf-8")
    (root / ".secret").write_text("do not publish", encoding="utf-8")
    return root


@pytest.fixture(scope="session")
def settings(
    models_dir: Path, demo_dir: Path, reports_dirs: tuple[Path, Path], frontend_dir: Path
) -> Settings:
    """Settings pointing at the fixtures (immutable: tests derive variants with model_copy)."""
    bench, audit = reports_dirs
    return Settings(
        models_dir=models_dir,
        demo_dir=demo_dir,
        benchmark_dir=bench,
        audit_dir=audit,
        frontend_dir=frontend_dir,
        log_level="WARNING",
    )


@pytest.fixture(scope="session")
def client(settings: Settings) -> Iterator[TestClient]:
    """One started application shared by the read-only contract tests."""
    with TestClient(create_app(settings)) as test_client:
        yield test_client


@pytest.fixture
def fresh_client(settings: Settings) -> Iterator[TestClient]:
    """A new application (own metrics registry) for tests that count requests."""
    with TestClient(create_app(settings)) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def sample_features(training_frame: pd.DataFrame) -> dict[str, float]:
    """The eight `combined` indicators of a mid-life training cycle (SOH 85 %)."""
    row = training_frame[(training_frame.battery_id == "B0005") & (training_frame.cycle_index == 5)]
    return {f: float(row[f].iloc[0]) for f in FEATURE_SETS["combined"]}


@pytest.fixture(scope="session")
def bundle_writer() -> Any:
    """:func:`write_bundles`, for tests that need their own (e.g. tampered) model folder."""
    return write_bundles


@pytest.fixture(scope="session")
def cycle_simulator() -> Any:
    """:func:`simulate_cycle`, for tests that post raw arrays."""
    return simulate_cycle
