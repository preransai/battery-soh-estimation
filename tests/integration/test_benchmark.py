"""Benchmark runner on the committed feature table (no raw data needed).

Runs ``run_benchmark(..., quick=True)`` into a temporary directory and checks the
output contract the API and web UI consume, the internal consistency of the
numbers, the protocol guarantees (pre-specified headline, leaky/post-hoc labels,
relative-mode reference cycles never scored) and, when present, that the committed
full results in ``reports/benchmark`` reproduce. A small multi-experiment run
(core cells, the twelve extended cells leave-one-battery-out with bootstrap and
Wilcoxon, and leave-one-condition-out) covers the extended and LOCO blocks.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy.stats import wilcoxon

from battery_soh.config import get_paths
from battery_soh.data.protocols import CORE_BATTERIES, EXTENDED_BATTERIES, PROTOCOLS
from battery_soh.evaluation.benchmark import (
    BENCHMARK_HASH_SECTIONS,
    PRODUCTION_HASH_SECTIONS,
    UNHASHED_SECTIONS,
    SpecKey,
    config_hashes,
    config_sections_sha256,
    evaluate_specs,
    load_config,
    run_benchmark,
    run_window_sweep,
)
from battery_soh.evaluation.benchmark.hashing import TOP_KEYS
from battery_soh.features.build import load_feature_table
from battery_soh.features.registry import FEATURE_SETS
from battery_soh.features.relative import relativize

CONFIG = get_paths().configs / "benchmark.yaml"
COMMITTED = get_paths().benchmark_dir
PRE_SPECIFIED = {
    "experiment": "core_lobo",
    "feature_set": "combined",
    "mode": "absolute",
    "model": "ridge_nfs",
}


def _strict_json(path: Path) -> Any:
    def _reject(token: str) -> None:
        raise ValueError(f"non-finite number {token} in {path.name}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject)


@pytest.fixture(scope="module")
def summary(quick_benchmark_dir: Path) -> dict[str, Any]:
    data: dict[str, Any] = _strict_json(quick_benchmark_dir / "quick" / "summary.json")
    return data


def test_quick_mode_writes_only_to_its_subdirectory(quick_benchmark_dir: Path) -> None:
    assert sorted(p.name for p in quick_benchmark_dir.iterdir()) == ["quick"]
    files = sorted(p.name for p in (quick_benchmark_dir / "quick").iterdir())
    assert files == ["predictions.csv", "results.csv", "summary.json"]


def test_headline_is_the_pre_specified_spec(summary: dict[str, Any]) -> None:
    head = summary["headline"]
    assert {k: head[k] for k in PRE_SPECIFIED} == PRE_SPECIFIED
    # The JSON keys keep their names (API contract); the text states what was known.
    assert head["pre_registered"] is True
    assert summary["pre_registration"]["headline"] == PRE_SPECIFIED
    statement = summary["pre_registration"]["statement"]
    assert "before any ridge_nfs result was computed" in statement
    assert "plain-ridge results for the same feature set and mode were already known" in statement
    assert "pre-registered" not in json.dumps(summary).lower()
    cfg = load_config(CONFIG)
    assert cfg.headline == SpecKey(**PRE_SPECIFIED)


def test_frontend_contract(summary: dict[str, Any]) -> None:
    head = summary["headline"]
    for key in ("n_batteries", "n_cycles", "mae", "rmse", "r2", "coverage_90", "mean_width_90"):
        assert isinstance(head[key], int | float), key
    assert head["n_batteries"] == len(CORE_BATTERIES)
    assert set(summary["baselines"]) == {"mean", "cycle_only", "capacity_oracle"}
    assert summary["baselines"]["capacity_oracle"]["leaky"] is True
    assert summary["baselines"]["mean"]["leaky"] is False
    for row in summary["leaderboard"]:
        for key in ("feature_set", "mode", "model", "mae", "rmse", "r2", "leaky", "baseline"):
            assert key in row
        assert "post_hoc" in row
    assert [r["battery_id"] for r in summary["per_battery"]] == list(CORE_BATTERIES)
    for row in summary["per_battery"]:
        assert {"n", "mae", "rmse", "r2", "coverage_90"} <= set(row)
    assert [c["nominal"] for c in summary["coverage"]] == [0.8, 0.9, 0.95]
    for row in summary["coverage"]:
        assert 0.0 <= row["empirical"] <= 1.0
        assert row["mean_width"] > 0
    pred = summary["predictions"][0]
    assert {"battery_id", "cycle_index", "y", "yhat", "lo_90", "hi_90"} <= set(pred)
    assert summary["extended"]["status"] == "skipped"
    assert summary["window_sweep"]["status"] == "skipped"
    assert summary["provenance"]["features_sha256"] == summary["data_sha256"]


def test_numbers_are_internally_consistent(summary: dict[str, Any]) -> None:
    head = summary["headline"]
    per = summary["per_battery"]
    assert head["mae"] == pytest.approx(np.mean([r["mae"] for r in per]), rel=1e-5)
    assert head["worst_mae"] == pytest.approx(max(r["mae"] for r in per), rel=1e-6)
    preds = pd.DataFrame(summary["predictions"])
    assert len(preds) == head["n"] == sum(r["n"] for r in per)
    for row in per:
        part = preds[preds["battery_id"] == row["battery_id"]]
        mae = float(np.mean(np.abs(part["yhat"] - part["y"])))
        assert mae == pytest.approx(row["mae"], abs=1e-3)  # values rounded to 6 digits
        inside = (part["y"] >= part["lo_90"]) & (part["y"] <= part["hi_90"])
        assert float(inside.mean()) == pytest.approx(row["coverage_90"], abs=1e-6)
    cov90 = next(c for c in summary["coverage"] if c["nominal"] == 0.9)
    assert cov90["empirical"] == pytest.approx(head["coverage_90"], rel=1e-9)
    # "cv" intervals have one half-width per held-out battery (what the bundle ships).
    width = (preds["hi_90"] - preds["lo_90"]).groupby(preds["battery_id"])
    assert ((width.max() - width.min()) <= 2e-3).all()  # constant up to 6-digit rounding


def test_leaderboard_flags_and_order(summary: dict[str, Any]) -> None:
    board = summary["leaderboard"]
    leaky = [r for r in board if r["leaky"]]
    assert [r["model"] for r in leaky] == ["capacity_oracle"]
    assert board[-len(leaky) :] == leaky  # leaky rows are listed last
    clean = [r["mae"] for r in board if not r["leaky"]]
    assert clean == sorted(clean)
    assert sum(r["headline"] for r in board) == 1
    baselines = {r["model"] for r in board if r["baseline"]}
    assert baselines == {"mean", "cycle_only", "capacity_oracle"}


def test_selection_stability_reports_every_outer_fold(summary: dict[str, Any]) -> None:
    stab = summary["selection_stability"]["headline"]
    candidates = list(FEATURE_SETS["combined"])
    assert stab["candidates"] == candidates
    assert [f["held_out"] for f in stab["folds"]] == list(CORE_BATTERIES)
    for fold in stab["folds"]:
        assert fold["selected"]
        assert set(fold["selected"]) | set(fold["dropped"]) == set(candidates)
        assert fold["alpha"] in (0.1, 1.0, 10.0)
        assert fold["n_groups"] == 3  # selection saw the three training cells only
    for feature, share in stab["frequency"].items():
        n_selected = sum(feature in f["selected"] for f in stab["folds"])
        assert share == pytest.approx(n_selected / 4)


def test_results_and_predictions_csv(quick_benchmark_dir: Path, summary: dict[str, Any]) -> None:
    results = pd.read_csv(quick_benchmark_dir / "quick" / "results.csv")
    specs = results.groupby(["experiment", "feature_set", "mode", "model"]).size()
    assert (specs == len(CORE_BATTERIES)).all()
    assert len(specs) == len(summary["leaderboard"])
    head = results[(results["model"] == "ridge_nfs") & (results["feature_set"] == "combined")]
    assert head["mae"].mean() == pytest.approx(summary["headline"]["mae"], rel=1e-5)
    assert head["selected_features"].notna().all()
    preds = pd.read_csv(quick_benchmark_dir / "quick" / "predictions.csv")
    assert set(preds["model"]) == {"ridge_nfs"}
    assert len(preds) == summary["headline"]["n"]
    for col in ("lo_80", "hi_80", "lo_90", "hi_90", "lo_95", "hi_95", "lo_90_cv_plus"):
        assert preds[col].notna().all()
    assert (preds["lo_95"] <= preds["lo_90"]).all()
    assert (preds["lo_90"] <= preds["lo_80"]).all()


def _small_config(tmp_path: Path, **overrides: Any) -> Path:
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    raw["n_jobs"] = 1
    raw["window_sweep"] = None
    raw["experiments"] = {
        "core_lobo": {
            "batteries": "CORE_BATTERIES",
            "feature_sets": ["discharge"],
            "modes": ["absolute", "relative"],
            "models": ["ridge"],
            "baselines": ["mean"],
        }
    }
    raw["headline"] = {**PRE_SPECIFIED, "feature_set": "discharge", "model": "ridge"}
    raw["secondary"] = []
    raw["predictions"] = [{"feature_set": "discharge", "mode": "relative", "model": "ridge"}]
    raw["quick"] = {"feature_sets": ["discharge"], "models": ["ridge"], "baselines": ["mean"]}
    raw.update(overrides)
    path = tmp_path / "bench.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


#: Extended-set indicators whose reference is near zero stay absolute (as in the real config).
_KEEP_ABSOLUTE = ["chg_temp_rise_cc_c", "dis_temp_rise_win_c"]
#: A repo-relative raw-data directory that does not exist.
_ABSENT_RAW_DIR = "data/raw/__absent_for_tests__"


@pytest.fixture(scope="module")
def multi_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Full (non-quick) run of a small grid: core cells, extended LOBO and LOCO (ridge/PLS only).

    The committed window-sweep grid is configured but the raw files are not found, so the
    run must fall back to the committed ``reports/benchmark/window_sweep.json``.
    """
    root = tmp_path_factory.mktemp("multi")
    path = _small_config(root)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["window_sweep"] = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["window_sweep"]
    raw["raw_dir"] = _ABSENT_RAW_DIR
    raw["experiments"]["extended_lobo"] = {
        "batteries": "EXTENDED_BATTERIES",
        "group_by": "battery",
        "feature_sets": ["charge_ext"],
        "modes": ["relative", "absolute"],
        "models": ["ridge", "pls"],
        "baselines": ["mean"],
        "interval_models": ["ridge"],
        "relative_keep_absolute": _KEEP_ABSOLUTE,
        "bootstrap": {"n_boot": 50, "confidence": 0.95},
        "wilcoxon": {"reference": "ridge", "candidates": ["pls"]},
    }
    raw["experiments"]["leave_one_condition_out"] = {
        "batteries": "EXTENDED_BATTERIES",
        "group_by": "condition",
        "feature_sets": ["charge_ext"],
        "modes": ["relative"],
        "models": ["ridge", "pls"],
        "baselines": ["mean"],
        "interval_models": ["ridge"],
        "relative_keep_absolute": _KEEP_ABSOLUTE,
    }
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    run_benchmark(path, root / "out")
    return root / "out"


@pytest.fixture(scope="module")
def multi_summary(multi_run: Path) -> dict[str, Any]:
    data: dict[str, Any] = _strict_json(multi_run / "summary.json")
    return data


def test_relative_mode_never_fits_or_scores_reference_cycles(
    multi_run: Path, multi_summary: dict[str, Any]
) -> None:
    summary = multi_summary
    preds = pd.read_csv(multi_run / "predictions.csv")
    rel = preds[preds["mode"] == "relative"]
    table = load_feature_table()
    core = table[table["battery_id"].isin(CORE_BATTERIES)]
    features = list(FEATURE_SETS["discharge"])
    expected = relativize(core, features, n_ref=5)
    expected = expected[expected["qc_valid"]]
    flagged = set(
        map(
            tuple,
            expected.loc[expected["is_reference_cycle"], ["battery_id", "cycle_index"]].values,
        )
    )
    got = set(map(tuple, rel.loc[rel["is_reference_cycle"], ["battery_id", "cycle_index"]].values))
    assert got == flagged
    assert len(flagged) == 5 * len(CORE_BATTERIES)
    assert not rel.loc[rel["is_reference_cycle"], "scored"].any()
    assert rel.loc[~rel["is_reference_cycle"], "scored"].all()
    rel_head = summary["relative_headline"]
    assert rel_head["n"] == int(rel["scored"].sum())
    assert rel_head["reference_cycles_excluded"] == len(flagged)
    # The reported MAE is computed on the scored cycles only.
    scored = rel[rel["scored"]]
    per = (scored["yhat"] - scored["y"]).abs().groupby(scored["battery_id"]).mean()
    assert rel_head["mae"] == pytest.approx(per.mean(), rel=1e-4)


def _cells_by_condition(batteries: tuple[str, ...]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for b in batteries:
        out.setdefault(PROTOCOLS[b].condition, []).append(b)
    return {c: sorted(bs) for c, bs in out.items()}


def test_extended_block_per_cell_per_condition_bootstrap_and_wilcoxon(
    multi_run: Path, multi_summary: dict[str, Any]
) -> None:
    ext = multi_summary["extended"]
    assert ext["experiment"] == "extended_lobo"
    assert ext["group_by"] == "battery"
    assert ext["batteries"] == list(EXTENDED_BATTERIES)
    assert ext["conditions"] == {b: PROTOCOLS[b].condition for b in EXTENDED_BATTERIES}
    assert ext["relative_keep_absolute"] == _KEEP_ABSOLUTE
    assert set(ext["reference_values"]) == set(EXTENDED_BATTERIES)
    by_condition = _cells_by_condition(EXTENDED_BATTERIES)
    results = pd.read_csv(multi_run / "results.csv")
    rows = results[results["experiment"] == "extended_lobo"]
    assert (rows["fold"] == rows["battery_id"]).all()  # one cell held out at a time
    specs = {(r["feature_set"], r["mode"], r["model"]) for r in ext["results"]}
    assert specs == {
        (fs, mode, model)
        for mode in ("relative", "absolute")
        for fs, model in (("charge_ext", "ridge"), ("charge_ext", "pls"), ("baseline", "mean"))
    }
    for r in ext["results"]:
        per = r["per_battery"]
        assert set(per) == set(EXTENDED_BATTERIES)
        maes = [per[b]["mae"] for b in EXTENDED_BATTERIES]
        assert r["mae"] == pytest.approx(np.mean(maes), rel=1e-5)
        lo, hi = r["mae_ci"]  # bootstrap over cells
        assert lo <= r["mae"] <= hi
        assert r["mae_ci_confidence"] == 0.95
        assert set(r["per_condition"]) == set(by_condition)
        for cond, entry in r["per_condition"].items():
            assert entry["batteries"] == by_condition[cond]
            assert entry["n_batteries"] == len(by_condition[cond])
            expected = np.mean([per[b]["mae"] for b in by_condition[cond]])
            assert entry["mae"] == pytest.approx(expected, rel=1e-5)
        assert (r["coverage_90"] is not None) == (r["model"] == "ridge")
    assert [(w["feature_set"], w["mode"]) for w in ext["wilcoxon"]] == [
        ("charge_ext", "relative"),
        ("charge_ext", "absolute"),
    ]
    for w in ext["wilcoxon"]:
        assert w["reference_model"] == "ridge"
        assert w["best_candidate"] == "pls"
        assert w["n_pairs"] == len(EXTENDED_BATTERIES)
        part = rows[(rows["feature_set"] == "charge_ext") & (rows["mode"] == w["mode"])]
        ref = part[part["model"] == "ridge"].set_index("battery_id")["mae"]
        cand = part[part["model"] == "pls"].set_index("battery_id")["mae"].reindex(ref.index)
        assert w["reference_better_count"] == int(((ref - cand) < 0).sum())
        test = wilcoxon(ref.to_numpy(), cand.to_numpy(), alternative="two-sided")
        assert w["p_value"] == pytest.approx(float(test.pvalue), rel=1e-4)
        assert 0.0 < w["p_value"] <= 1.0


def test_leave_one_condition_out_holds_out_whole_conditions(
    multi_run: Path, multi_summary: dict[str, Any]
) -> None:
    loco = multi_summary["loco"]
    by_condition = _cells_by_condition(EXTENDED_BATTERIES)
    assert loco["group_by"] == "condition"
    assert {c: sorted(bs) for c, bs in loco["conditions"].items()} == by_condition
    results = pd.read_csv(multi_run / "results.csv")
    rows = results[results["experiment"] == "leave_one_condition_out"]
    assert (rows["fold"] == rows["condition"]).all()  # a whole condition held out at a time
    for r in loco["results"]:
        assert set(r["per_condition"]) == set(by_condition)
        per = r["per_battery"]
        assert set(per) == set(EXTENDED_BATTERIES)
        assert r["mae"] == pytest.approx(np.mean([v["mae"] for v in per.values()]), rel=1e-5)
        for cond, entry in r["per_condition"].items():
            assert entry["batteries"] == by_condition[cond]
            expected = np.mean([per[b]["mae"] for b in by_condition[cond]])
            assert entry["mae"] == pytest.approx(expected, rel=1e-5)
            assert entry["n"] == sum(per[b]["n"] for b in by_condition[cond])
            # Intervals only where configured; calibrated on the two training conditions.
            assert ("coverage_90" in entry) == (r["model"] == "ridge")
            if r["model"] == "ridge":
                assert 0.0 <= entry["coverage_90"] <= 1.0
                assert entry["mean_width_90"] > 0


def _strings(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [s for k, v in value.items() for s in (str(k), *_strings(v))]
    if isinstance(value, list):
        return [s for v in value for s in _strings(v)]
    return [value] if isinstance(value, str) else []


_ABSOLUTE_PATH = re.compile(r"(^|[\s(\"'])([A-Za-z]:[\\/]|/(home|Users|tmp|root|var|mnt)/)")


@pytest.mark.parametrize("which", ["quick", "multi"])
def test_summary_contains_no_machine_paths(
    which: str, summary: dict[str, Any], multi_summary: dict[str, Any], tmp_path: Path
) -> None:
    document = summary if which == "quick" else multi_summary
    root = str(get_paths().root.resolve())
    for text in _strings(document):
        assert root not in text, text
        assert str(tmp_path.parent.parent) not in text, text
        assert not _ABSOLUTE_PATH.search(text), text


def test_window_sweep_falls_back_to_the_committed_sweep(
    multi_run: Path, multi_summary: dict[str, Any]
) -> None:
    sweep = multi_summary["window_sweep"]
    committed = COMMITTED / "window_sweep.json"
    if not committed.is_file():
        assert sweep == {
            "status": "skipped",
            "reason": f"raw .mat files not found under {_ABSENT_RAW_DIR}",
        }
        return
    assert sweep["status"] == "ok"
    assert sweep["source"] == "cached (raw data unavailable): reports/benchmark/window_sweep.json"
    # The output folder is complete: the reused sweep is copied next to summary.json.
    assert (multi_run / "window_sweep.json").read_bytes() == committed.read_bytes()


def test_benchmark_refuses_leaky_feature_sets(tmp_path: Path) -> None:
    path = _small_config(tmp_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["experiments"]["core_lobo"]["feature_sets"] = ["discharge", "legacy_v1"]
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="label-side/proxy"):
        run_benchmark(path, tmp_path / "out")


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"unknown_key": 1}, "unknown key"),
        ({"headline": {**PRE_SPECIFIED, "model": "gpr"}}, "not part of the configured"),
        ({"confidence_levels": [0.9, 1.5]}, "confidence_levels"),
        ({"confidence_levels": [0.8, 0.95]}, "must include 0.9"),
    ],
)
def test_config_validation(tmp_path: Path, override: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        load_config(_small_config(tmp_path, **override))


# --------------------------------------------------------------------------- config hash scope


def _rewrite(tmp_path: Path, edit: Any, name: str = "edited.yaml") -> Path:
    """The committed config, re-serialised (other formatting and key order) after ``edit``."""
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    edit(raw)
    path = tmp_path / name
    reordered = dict(reversed(list(raw.items())))
    path.write_text(yaml.safe_dump(reordered, sort_keys=False, width=60), encoding="utf-8")
    return path


def test_every_top_level_key_has_a_hash_scope() -> None:
    hashed = set(BENCHMARK_HASH_SECTIONS)
    production = set(PRODUCTION_HASH_SECTIONS)
    unhashed = set(UNHASHED_SECTIONS)
    # Every key is in exactly one scope.
    assert len(hashed) + len(production) + len(unhashed) == len(set(TOP_KEYS))
    assert hashed | production | unhashed == set(TOP_KEYS)
    assert set(yaml.safe_load(CONFIG.read_text(encoding="utf-8"))) <= set(TOP_KEYS)


@pytest.mark.parametrize(
    "edit",
    [
        lambda c: c["production"].update(created_utc="2030-01-01T00:00:00+00:00"),
        lambda c: c["production"]["specs"].reverse(),
        lambda c: c.update(n_jobs=3, description="edited", raw_dir="elsewhere/nasa"),
        lambda c: c["experiments"]["core_lobo"].update(description="edited"),
        lambda c: None,  # formatting and key order only
    ],
    ids=["production-created", "production-order", "unhashed-keys", "nested-text", "format"],
)
def test_benchmark_hash_ignores_what_cannot_change_benchmark_results(
    tmp_path: Path, edit: Any
) -> None:
    edited = _rewrite(tmp_path, edit)
    assert load_config(edited).sha256 == load_config(CONFIG).sha256
    assert config_hashes(edited)["benchmark"] == config_hashes(CONFIG)["benchmark"]


@pytest.mark.parametrize(
    "edit",
    [
        lambda c: c.update(seed=7),
        lambda c: c["nested_selection"].update(tol=1e-6),
        lambda c: c["secondary"][0].update(label="renamed"),
        lambda c: c["experiments"]["core_lobo"]["models"].reverse(),
        lambda c: c["quick"].update(models=["ridge"]),
    ],
    ids=["seed", "nested", "secondary-label", "list-order", "quick"],
)
def test_benchmark_hash_tracks_every_result_setting_and_not_production(
    tmp_path: Path, edit: Any
) -> None:
    edited = _rewrite(tmp_path, edit)
    before, after = config_hashes(CONFIG), config_hashes(edited)
    assert after["benchmark"] != before["benchmark"]
    assert after["production"] == before["production"]


def test_production_hash_tracks_the_production_section(tmp_path: Path) -> None:
    edited = _rewrite(tmp_path, lambda c: c["production"]["specs"].pop())
    assert config_hashes(edited)["production"] != config_hashes(CONFIG)["production"]


def test_only_free_text_descriptions_are_left_out_of_the_hash() -> None:
    def digest(experiments: dict[str, Any]) -> str:
        return config_sections_sha256({"experiments": experiments}, ["experiments"])

    plain = digest({"a": {"models": ["ridge"]}})
    assert digest({"a": {"description": "any text", "models": ["ridge"]}}) == plain
    # A key that happens to be called "description" but holds settings still counts.
    assert digest({"a": {"models": ["ridge"]}, "description": {"models": ["gpr"]}}) != plain
    assert config_sections_sha256({}, ["experiments"]) != plain  # absent != present


def test_summary_records_the_scoped_config_hash(summary: dict[str, Any]) -> None:
    assert summary["config_sha256"] == config_hashes(CONFIG)["benchmark"]
    assert summary["provenance"]["config_sha256"] == summary["config_sha256"]
    assert summary["provenance"]["config_sha256_sections"] == list(BENCHMARK_HASH_SECTIONS)


def test_evaluate_specs_matches_quick_run(summary: dict[str, Any]) -> None:
    rows = evaluate_specs(CONFIG, [SpecKey(**PRE_SPECIFIED)])
    head = rows["core_lobo/combined/absolute/ridge_nfs"]
    assert head["mae"] == pytest.approx(summary["headline"]["mae"], rel=1e-9)
    assert head["r2"] == pytest.approx(summary["headline"]["r2"], rel=1e-9)
    assert head["coverage_90"] is None  # point predictions only


def _assert_close(a: Any, b: Any) -> None:
    """Equal structure; numbers equal to 1e-5 relative (parallel vs serial floating point)."""
    if isinstance(a, dict):
        assert set(a) == set(b)
        for key in a:
            _assert_close(a[key], b[key])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            _assert_close(x, y)
    elif isinstance(a, float) and isinstance(b, float):
        assert a == pytest.approx(b, rel=1e-5, abs=1e-9)
    else:
        assert a == b


@pytest.mark.skipif(
    not (COMMITTED / "summary.json").is_file(), reason="no committed benchmark results"
)
def test_committed_results_reproduce(summary: dict[str, Any]) -> None:
    committed = _strict_json(COMMITTED / "summary.json")
    csv_sha = hashlib.sha256(get_paths().features_csv.read_bytes()).hexdigest()
    # A stale result set must fail, not skip: re-run `soh benchmark` after rebuilding the table.
    assert committed["data_sha256"] == csv_sha, "committed results use another feature table"
    assert committed["quick"] is False
    # Recorded under the current benchmark settings (production-only edits do not count).
    assert committed["config_sha256"] == config_hashes(CONFIG)["benchmark"], "re-run soh benchmark"
    assert {k: committed["headline"][k] for k in PRE_SPECIFIED} == PRE_SPECIFIED
    for key in ("n", "mae", "rmse", "r2", "coverage_90", "mean_width_90", "coverage_90_cv_plus"):
        assert committed["headline"][key] == pytest.approx(summary["headline"][key], rel=1e-5)
    _assert_close(committed["per_battery"], summary["per_battery"])
    for section in ("extended", "loco", "window_sweep"):
        assert committed[section].get("status") != "skipped", section
    assert (COMMITTED / "window_sweep.json").is_file()


@pytest.mark.skipif(
    not (COMMITTED / "window_sweep.json").is_file(), reason="no committed window sweep"
)
def test_committed_window_sweep_shape() -> None:
    sweep = _strict_json(COMMITTED / "window_sweep.json")
    assert sweep["requires_raw_data"] is True
    ids = [w["id"] for w in sweep["windows"]]
    assert "dis_380_350" in ids
    assert "chg_390_419" in ids
    for window in sweep["windows"]:
        assert 0.0 < window["availability"] <= 1.0
        assert 0.0 < window["capacity_fraction_mean"] < 1.0
        for mode in ("absolute", "relative"):
            assert set(window[mode]["per_battery_mae"]) == set(CORE_BATTERIES)
    # The sweep re-extracts the registry windows from raw data and must match the table.
    assert all(v is not None and v < 1e-5 for v in sweep["consistency_max_rel_diff"].values())


@pytest.mark.raw_data
def test_window_sweep_reproduces_registry_windows_from_raw_data() -> None:
    cfg = load_config(CONFIG)
    assert cfg.window_sweep is not None
    small = type(cfg.window_sweep)(
        batteries=("B0005", "B0018"),
        modes=("absolute",),
        discharge_v_high=(3.8,),
        discharge_v_low=(3.5,),
        charge_v_start=(3.9,),
        charge_v_end=4.19,
    )
    result = run_window_sweep(
        small, load_feature_table(), get_paths().nasa_raw, label="soh_pct", n_ref=5, seed=42
    )
    assert set(result["consistency_max_rel_diff"]) == {"dis_q_380_350_ah", "chg_cc_time_390_419_s"}
    window = next(w for w in result["windows"] if w["id"] == "dis_380_350")
    assert window["availability"] == 1.0
    assert 0.3 < window["capacity_fraction_mean"] < 0.6
    assert math.isfinite(window["absolute"]["mae"])
