"""Production training end to end: benchmark -> ``train_bundles`` -> API, on committed data.

A slice of the committed feature table (every third cycle of the four core
cells) goes through the real benchmark and the real training, so the tests
check ground truth rather than mocks:

* the bundles reproduce a fresh scikit-learn fit and an independent re-run of
  the nested selection, and their conformal half-widths are the exact
  quantiles of independently recomputed leave-one-battery-out residuals;
* their metrics are the benchmark's numbers for the same spec, and training
  refuses benchmark results it cannot reproduce;
* re-running training writes byte-identical files with LF line endings;
* the API serves the trained folder (``/readyz``, ``/v1/models`` and the CI
  smoke request).

The committed ``artifacts/models`` folder is checked against the committed
benchmark results as well, so a stale artifact fails CI.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient

from battery_soh.api.app import create_app
from battery_soh.api.registry import ModelRegistry
from battery_soh.api.settings import Settings
from battery_soh.config import get_paths
from battery_soh.data.protocols import CORE_BATTERIES
from battery_soh.evaluation.benchmark import config_hashes, run_benchmark
from battery_soh.evaluation.nested import backward_elimination
from battery_soh.features.build import load_feature_table, metadata_path, write_feature_table
from battery_soh.features.registry import FEATURE_SETS
from battery_soh.models.linear_bundle import LinearBundle, verify_manifest
from battery_soh.models.zoo import make_model
from battery_soh.training import TrainingError, train_bundles, training_frame
from battery_soh.uncertainty.conformal import conformal_quantile

PATHS = get_paths()
FIXTURE_REQUEST = PATHS.root / ".github" / "fixtures" / "sample_estimate_request.json"
SPECS = (("combined", "absolute"), ("combined", "relative"))
IDS = sorted(f"{fs}-{mode}-ridge" for fs, mode in SPECS)
NESTED = {"alphas": [0.1, 1.0, 10.0], "min_features": 1, "tol": 1.0e-9}

pytestmark = pytest.mark.skipif(
    not PATHS.features_csv.is_file(), reason="committed feature table missing"
)


def _config(features_csv: Path) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "seed": 42,
        "label": "soh_pct",
        "n_reference_cycles": 5,
        "confidence_levels": [0.8, 0.9, 0.95],
        "n_jobs": 1,
        "features_csv": str(features_csv),
        "conformal": {"methods": ["cv", "cv_plus"], "primary": "cv"},
        "nested_selection": {"scoring": "inner_lobo_mae", **NESTED},
        "headline": {
            "experiment": "core_lobo",
            "feature_set": "combined",
            "mode": "absolute",
            "model": "ridge_nfs",
        },
        "experiments": {
            "core_lobo": {
                "batteries": "CORE_BATTERIES",
                "group_by": "battery",
                "feature_sets": ["combined"],
                "modes": ["absolute", "relative"],
                "models": ["ridge_nfs"],
                "baselines": ["mean"],
                "interval_models": ["ridge_nfs"],
            }
        },
        "production": {
            "model": "ridge_nfs",
            "created_utc": "features_meta",
            "specs": [{"feature_set": fs, "mode": mode} for fs, mode in SPECS],
        },
    }


@pytest.fixture(scope="module")
def workspace(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Sliced table -> benchmark -> two independent training runs."""
    root = tmp_path_factory.mktemp("training")
    table = load_feature_table(PATHS.features_csv)
    sliced = table[table["battery_id"].isin(CORE_BATTERIES) & (table["cycle_index"] % 3 == 0)]
    csv = write_feature_table(sliced.reset_index(drop=True), root / "features.csv")
    config = root / "benchmark.yaml"
    config.write_text(yaml.safe_dump(_config(csv), sort_keys=False), encoding="utf-8")
    run_benchmark(config, root / "bench")
    results = root / "bench" / "results.csv"
    train_bundles(config, root / "models_a", results_csv=results)
    train_bundles(config, root / "models_b", results_csv=results)
    return {
        "root": root,
        "csv": csv,
        "config": config,
        "results": results,
        "a": root / "models_a",
        "b": root / "models_b",
    }


def _bundles(folder: Path) -> dict[str, dict[str, Any]]:
    return {mid: json.loads((folder / f"{mid}.json").read_text(encoding="utf-8")) for mid in IDS}


def _fit_rows(csv: Path, feature_set: str, mode: str) -> pd.DataFrame:
    frame = training_frame(
        load_feature_table(csv),
        batteries=CORE_BATTERIES,
        features=FEATURE_SETS[feature_set],
        mode=mode,  # type: ignore[arg-type]
    )
    return frame[~frame["is_reference_cycle"]].reset_index(drop=True)


# --------------------------------------------------------------------------- artifacts


def test_writes_bundles_cards_and_a_verified_manifest(workspace: dict[str, Path]) -> None:
    folder = workspace["a"]
    files = verify_manifest(folder, allow_unlisted=False)
    assert sorted(files) == sorted(
        [f"{mid}.json" for mid in IDS] + [f"model_cards/{mid}.md" for mid in IDS]
    )
    meta = json.loads(metadata_path(workspace["csv"]).read_text(encoding="utf-8"))
    for mid, bundle in _bundles(folder).items():
        assert bundle["model_id"] == mid
        assert bundle["training"]["features_sha256"] == meta["sha256"]
        assert bundle["created_utc"] == meta["created_utc"]  # deterministic timestamp
        assert bundle["training"]["batteries"] == list(CORE_BATTERIES)
        assert bundle["model_card"] == f"model_cards/{mid}.md"
        LinearBundle.from_json(folder / f"{mid}.json")


def test_bundles_record_both_scoped_config_hashes(workspace: dict[str, Path]) -> None:
    summary = json.loads((workspace["root"] / "bench" / "summary.json").read_text("utf-8"))
    hashes = config_hashes(workspace["config"])
    assert summary["config_sha256"] == hashes["benchmark"]
    for bundle in _bundles(workspace["a"]).values():
        provenance = bundle["provenance"]
        assert provenance["benchmark_config_sha256"] == summary["config_sha256"]
        assert provenance["production_config_sha256"] == hashes["production"]
        assert "config_sha256" not in provenance  # the old, unscoped whole-file hash


def _edited_config(workspace: dict[str, Path], name: str, edit: Any) -> Path:
    raw = yaml.safe_load(workspace["config"].read_text(encoding="utf-8"))
    edit(raw)
    path = workspace["root"] / name
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def test_production_edits_reuse_the_benchmark_and_other_edits_do_not(
    workspace: dict[str, Path],
) -> None:
    results = workspace["results"]
    # Only the production section changes: the benchmark run stays valid.
    production_only = _edited_config(
        workspace,
        "production_only.yaml",
        lambda c: c["production"].update(created_utc="2030-01-01T00:00:00+00:00"),
    )
    out = workspace["root"] / "models_production_edit"
    train_bundles(production_only, out, results_csv=results)
    for mid, bundle in _bundles(out).items():
        reference = _bundles(workspace["a"])[mid]
        assert bundle["created_utc"] == "2030-01-01T00:00:00+00:00"
        assert bundle["estimator"] == reference["estimator"]
        assert (
            bundle["provenance"]["benchmark_config_sha256"]
            == reference["provenance"]["benchmark_config_sha256"]
        )
        assert (
            bundle["provenance"]["production_config_sha256"]
            != reference["provenance"]["production_config_sha256"]
        )
    # A benchmark setting changes: the recorded results no longer describe the procedure.
    changed = _edited_config(
        workspace, "nested_changed.yaml", lambda c: c["nested_selection"].update(tol=1e-3)
    )
    with pytest.raises(TrainingError, match="other benchmark settings"):
        train_bundles(changed, workspace["root"] / "models_refused", results_csv=results)
    assert not (workspace["root"] / "models_refused").exists()


def test_rerun_is_byte_identical_with_lf_line_endings(workspace: dict[str, Path]) -> None:
    a, b = workspace["a"], workspace["b"]
    names = sorted(p.relative_to(a).as_posix() for p in a.rglob("*") if p.is_file())
    assert names == sorted(p.relative_to(b).as_posix() for p in b.rglob("*") if p.is_file())
    for name in names:
        assert (a / name).read_bytes() == (b / name).read_bytes(), name
        if name != "MANIFEST.json":  # hashed files must survive a git checkout unchanged
            assert b"\r\n" not in (a / name).read_bytes(), name


def test_bundle_reproduces_an_independent_selection_and_sklearn_fit(
    workspace: dict[str, Path],
) -> None:
    for mid, bundle in _bundles(workspace["a"]).items():
        fs, mode, _ = mid.split("-")
        rows = _fit_rows(workspace["csv"], fs, mode)
        y = rows["soh_pct"].to_numpy(float)
        selection = backward_elimination(
            rows[list(FEATURE_SETS[fs])],
            y,
            rows["battery_id"].to_numpy(object),
            alphas=NESTED["alphas"],
            min_features=1,
            tol=NESTED["tol"],
        )
        assert bundle["features"] == list(selection.selected)
        assert bundle["estimator"]["alpha"] == selection.alpha
        fresh = make_model("ridge", alpha=selection.alpha).fit(rows[bundle["features"]], y)
        runtime = LinearBundle.from_dict(bundle)
        X = rows[bundle["features"]].to_numpy(float)
        np.testing.assert_allclose(
            runtime.predict_raw_array(X), fresh.predict(rows[bundle["features"]]), rtol=1e-9
        )
        assert bundle["training"]["n_cycles"] == len(rows)


def test_conformal_half_widths_are_pooled_lobo_quantiles(workspace: dict[str, Path]) -> None:
    bundle = _bundles(workspace["a"])["combined-absolute-ridge"]
    rows = _fit_rows(workspace["csv"], "combined", "absolute")
    features = list(FEATURE_SETS["combined"])
    scores = []
    for battery in CORE_BATTERIES:
        train, test = rows[rows["battery_id"] != battery], rows[rows["battery_id"] == battery]
        sel = backward_elimination(
            train[features],
            train["soh_pct"].to_numpy(float),
            train["battery_id"].to_numpy(object),
            alphas=NESTED["alphas"],
            tol=NESTED["tol"],
        )
        model = make_model("ridge", alpha=sel.alpha).fit(
            train[list(sel.selected)], train["soh_pct"]
        )
        scores.append(
            np.abs(test["soh_pct"].to_numpy(float) - model.predict(test[list(sel.selected)]))
        )
    pooled = np.concatenate(scores)
    assert bundle["conformal"]["n_scores"] == pooled.size == len(rows)
    for key, value in bundle["conformal"]["quantiles"].items():
        assert value == pytest.approx(conformal_quantile(pooled, 1 - float(key)), rel=1e-9)


def test_metrics_are_the_benchmark_rows_of_the_same_spec(workspace: dict[str, Path]) -> None:
    results = pd.read_csv(workspace["results"])
    for mid, bundle in _bundles(workspace["a"]).items():
        fs, mode, _ = mid.split("-")
        rows = results[
            (results["feature_set"] == fs)
            & (results["mode"] == mode)
            & (results["model"] == "ridge_nfs")
        ].set_index("battery_id")
        metrics = bundle["metrics"]
        assert metrics["spec"] == {
            "experiment": "core_lobo",
            "feature_set": fs,
            "mode": mode,
            "model": "ridge_nfs",
        }
        assert sorted(metrics["per_battery"]) == sorted(rows.index)
        for battery, m in metrics["per_battery"].items():
            assert m["mae"] == pytest.approx(rows.loc[battery, "mae"], rel=1e-6)
            assert m["n"] == rows.loc[battery, "n"]
            assert m["coverage"]["0.9"] == pytest.approx(rows.loc[battery, "coverage_90_cv"])
        assert metrics["mean"]["mae"] == pytest.approx(rows["mae"].mean(), rel=1e-5)
        assert metrics["coverage"]["0.9"] == pytest.approx(rows["coverage_90_cv"].mean(), rel=1e-5)
        assert metrics["mean"]["worst_battery"] == rows["mae"].idxmax()


def test_relative_bundle_never_fits_on_reference_cycles(workspace: dict[str, Path]) -> None:
    bundle = _bundles(workspace["a"])["combined-relative-ridge"]
    assert bundle["mode"] == "relative"
    assert bundle["n_reference_cycles"] == 5
    protocol = bundle["training_protocol"]
    excluded = protocol["n_reference_cycles_excluded"]
    assert excluded >= 5 * len(CORE_BATTERIES)
    assert bundle["training"]["n_cycles"] == protocol["n_evaluation_rows"] - excluded
    assert bundle["conformal"]["n_scores"] == bundle["training"]["n_cycles"]
    # The domain envelope is that of the relativized (ratio) inputs, not the raw ones.
    rows = _fit_rows(workspace["csv"], "combined", "relative")
    np.testing.assert_allclose(
        bundle["domain"]["feature_min"], rows[bundle["features"]].min().to_numpy(float)
    )
    np.testing.assert_allclose(
        bundle["domain"]["feature_max"], rows[bundle["features"]].max().to_numpy(float)
    )


# --------------------------------------------------------------------------- failure modes


def _train_with(results: pd.DataFrame, workspace: dict[str, Path], name: str) -> Path:
    """Train with edited benchmark results into a copy of a valid artifact folder."""
    path = workspace["root"] / name
    results.to_csv(path, index=False)
    out = workspace["root"] / f"out_{name}"
    shutil.copytree(workspace["a"], out)
    try:
        train_bundles(
            workspace["config"],
            out,
            results_csv=path,
            summary_json=workspace["root"] / "bench" / "summary.json",
        )
    finally:
        # A refused run must leave the previous, verified artifacts untouched.
        verify_manifest(out, allow_unlisted=False)
        for item in workspace["a"].rglob("*"):
            if item.is_file():
                assert (out / item.relative_to(workspace["a"])).read_bytes() == item.read_bytes()
    return out


def test_results_that_do_not_match_the_procedure_are_rejected(workspace: dict[str, Path]) -> None:
    results = pd.read_csv(workspace["results"])
    hit = (
        (results["feature_set"] == "combined")
        & (results["mode"] == "absolute")
        & (results["model"] == "ridge_nfs")
        & (results["battery_id"] == "B0006")
    )
    results.loc[hit, "mae"] = results.loc[hit, "mae"] * 0.5  # a "better" number
    with pytest.raises(TrainingError, match="does not reproduce"):
        _train_with(results, workspace, "tampered.csv")


def test_missing_benchmark_spec_is_fatal(workspace: dict[str, Path]) -> None:
    results = pd.read_csv(workspace["results"])
    keep = ~((results["mode"] == "relative") & (results["model"] == "ridge_nfs"))
    with pytest.raises(TrainingError, match="no rows for core_lobo/combined/relative/ridge_nfs"):
        _train_with(results[keep], workspace, "partial.csv")


# --------------------------------------------------------------------------- serving


def _client(models_dir: Path, tmp: Path) -> TestClient:
    settings = Settings(
        models_dir=models_dir,
        demo_dir=tmp / "no-demo",
        benchmark_dir=tmp / "no-bench",
        audit_dir=tmp / "no-audit",
        frontend_dir=tmp / "no-frontend",
        log_level="WARNING",
    )
    return TestClient(create_app(settings))


def _smoke(client: TestClient, ids: list[str]) -> None:
    ready = client.get("/readyz")
    assert ready.status_code == 200
    assert sorted(ready.json()["models"]) == sorted(ids)
    listed = client.get("/v1/models").json()
    assert sorted(m["model_id"] for m in listed) == sorted(ids)
    for model in listed:
        assert model["metrics"]["lobo_mae"] > 0
        assert 0.0 <= model["metrics"]["coverage_90"] <= 1.0
    body = json.loads(FIXTURE_REQUEST.read_text(encoding="utf-8"))
    response = client.post("/v1/soh/estimate", json=body)
    assert response.status_code == 200, response.text
    estimate = response.json()
    assert estimate["model_id"] in ids
    assert estimate["interval"]["lower"] <= estimate["soh_pct"] <= estimate["interval"]["upper"]
    assert 0.0 <= estimate["soh_pct"] <= 120.0


def test_api_serves_the_trained_bundles(workspace: dict[str, Path], tmp_path: Path) -> None:
    with _client(workspace["a"], tmp_path) as client:
        _smoke(client, IDS)


# --------------------------------------------------------------------------- committed artifacts

COMMITTED = PATHS.models_dir


@pytest.mark.skipif(
    not (COMMITTED / "MANIFEST.json").is_file(), reason="no committed model bundles"
)
def test_committed_bundles_match_committed_benchmark_and_serve(tmp_path: Path) -> None:
    verify_manifest(COMMITTED)
    registry = ModelRegistry.load(COMMITTED)
    ids = registry.model_ids
    assert len(ids) == 4
    results = pd.read_csv(PATHS.benchmark_dir / "results.csv")
    summary = json.loads((PATHS.benchmark_dir / "summary.json").read_text(encoding="utf-8"))
    hashes = config_hashes(PATHS.configs / "benchmark.yaml")
    meta = json.loads(metadata_path(PATHS.features_csv).read_text(encoding="utf-8"))
    for entry in registry.entries:
        doc = entry.bundle.to_dict()
        assert doc["training"]["features_sha256"] == meta["sha256"]  # trained on this table
        # Trained on the committed benchmark run, under the committed production section.
        assert doc["provenance"]["benchmark_config_sha256"] == summary["config_sha256"]
        assert doc["provenance"]["production_config_sha256"] == hashes["production"]
        spec = doc["metrics"]["spec"]
        rows = results[
            (results["experiment"] == spec["experiment"])
            & (results["feature_set"] == spec["feature_set"])
            & (results["mode"] == spec["mode"])
            & (results["model"] == spec["model"])
        ].set_index("battery_id")
        for battery, m in doc["metrics"]["per_battery"].items():
            assert m["mae"] == pytest.approx(rows.loc[battery, "mae"], rel=1e-6)
    copy = tmp_path / "models"
    shutil.copytree(COMMITTED, copy)
    with _client(copy, tmp_path) as client:
        _smoke(client, ids)


@pytest.mark.slow
@pytest.mark.skipif(
    not (COMMITTED / "MANIFEST.json").is_file(), reason="no committed model bundles"
)
def test_committed_bundles_are_reproducible(tmp_path: Path) -> None:
    train_bundles(PATHS.configs / "benchmark.yaml", tmp_path)
    for name in sorted(p.name for p in COMMITTED.glob("*.json") if p.name != "MANIFEST.json"):
        fresh = json.loads((tmp_path / name).read_text(encoding="utf-8"))
        committed = json.loads((COMMITTED / name).read_text(encoding="utf-8"))
        assert fresh["features"] == committed["features"]
        assert fresh["metrics"] == committed["metrics"]
        np.testing.assert_allclose(
            fresh["estimator"]["coef"], committed["estimator"]["coef"], rtol=1e-9
        )
        assert (tmp_path / "model_cards" / name.replace(".json", ".md")).is_file()
