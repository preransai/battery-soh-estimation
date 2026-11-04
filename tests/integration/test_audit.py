"""Leakage audit on the committed feature table: the v1 failure modes reproduce exactly."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from battery_soh.config import get_paths
from battery_soh.reporting.style import AUDIT_LABELS

COMMITTED = get_paths().reports / "audit"


@pytest.fixture(scope="module")
def audit(audit_dir: Path) -> dict[str, Any]:
    def _reject(token: str) -> None:
        raise ValueError(f"non-finite number {token}")

    data: dict[str, Any] = json.loads(
        (audit_dir / "audit.json").read_text(encoding="utf-8"), parse_constant=_reject
    )
    return data


def _row(audit: dict[str, Any], row_id: str) -> dict[str, Any]:
    return next(r for r in audit["rows"] if r["id"] == row_id)


def test_contaminated_split_overlap_is_reproduced(audit: dict[str, Any]) -> None:
    # v1 trained on the first 85 % of B0005-7 (142 cycles each) and "tested" on all 336 cycles of
    # B0006 + B0007, so 2 x 142 = 284 test rows had been training rows.
    assert audit["contaminated_test_overlap"] == {"overlapping_rows": 284, "test_rows": 336}
    split = audit["details"]["v1_split"]
    assert split["train_rows"] == 3 * 142
    assert split["honest_holdout_rows"] == 3 * 26
    row = _row(audit, "v1_contaminated_split")
    assert row["category"] == "v1_reported"
    assert row["overlapping_rows"] == 284
    assert row["leaky"] is True


def test_contaminated_score_is_near_perfect_and_honest_ones_are_not(audit: dict[str, Any]) -> None:
    contaminated = _row(audit, "v1_contaminated_split")
    honest = _row(audit, "v1_within_battery_holdout")
    lobo = _row(audit, "v1_inputs_lobo")
    assert contaminated["r2"] > 0.97
    assert lobo["r2"] < contaminated["r2"] - 0.05
    assert lobo["mae"] > 2.0 * contaminated["mae"]
    assert honest["mae"] > 1.5 * contaminated["mae"]
    # The flattering number comes from the rows the model had already seen.
    split = audit["details"]["v1_split"]
    assert split["mae_on_overlapping_rows"] < split["mae_on_unseen_rows"]


def test_label_is_capacity_and_proxies_rebuild_it(audit: dict[str, Any]) -> None:
    assert audit["soh_vs_capacity_within_battery_r2"] >= 0.99999
    within = audit["details"]["soh_vs_capacity_within_battery"]
    assert set(within["per_battery"]) == {"B0005", "B0006", "B0007", "B0018"}
    assert within["max_abs_error_soh_minus_100_capacity_over_c0"] < 1e-2  # CSV rounding only
    assert audit["capacity_reconstruction_r2"] > 0.99
    capacity_only = _row(audit, "capacity_only_lobo")
    assert capacity_only["leaky"] is True
    assert capacity_only["mae"] > 1.0  # imperfect only because each held-out C0 is unknown


def test_v2_guards_reject_every_v1_mistake(audit: dict[str, Any]) -> None:
    guards = audit["details"]["guards"]
    assert set(guards) == {
        "assert_disjoint_on_v1_split",
        "check_no_leakage_legacy_v1",
        "check_no_leakage_legacy_v1_no_capacity",
        "get_feature_set_legacy_v1_without_opt_in",
    }
    for name, outcome in guards.items():
        assert outcome["raised"], name
    assert guards["assert_disjoint_on_v1_split"]["error"] == "LeakageError"
    assert "284 evaluation row(s)" in guards["assert_disjoint_on_v1_split"]["message"]
    flagged = {r["feature"] for r in audit["details"]["leakage_report_legacy_v1"] if r["flagged"]}
    assert "leg_capacity_ah" in flagged
    assert "ctx_cycle_index" not in flagged


def test_rows_match_frontend_contract_and_csv(audit: dict[str, Any], audit_dir: Path) -> None:
    ids = [r["id"] for r in audit["rows"]]
    assert ids[:5] == [
        "v1_contaminated_split",
        "v1_within_battery_holdout",
        "v1_inputs_lobo",
        "v1_inputs_without_capacity_lobo",
        "capacity_only_lobo",
    ]
    assert "v2_headline" in ids
    for row in audit["rows"]:
        assert row["category"] in {"v1_reported", "honest", "v2"}
        assert isinstance(row["mae"], float)
        assert isinstance(row["r2"], float)
        assert isinstance(row["leaky"], bool)
        assert row["leaky"] == (row["category"] != "v2")
    table = pd.read_csv(audit_dir / "audit_table.csv")
    assert list(table["id"]) == ids
    assert table["mae"].tolist() == pytest.approx([r["mae"] for r in audit["rows"]], rel=1e-5)


def test_v2_rows_are_labelled_by_their_model(audit: dict[str, Any]) -> None:
    head = _row(audit, "v2_headline")
    assert "pre-specified" in head["label"]
    assert "pre-registered" not in json.dumps(audit).lower()
    # The relative row is plain Ridge (the config's relative prediction spec), not the
    # headline's nested selection; its labels must say so.
    relative = _row(audit, "v2_relative_mode")
    assert relative["model"] == "ridge"
    assert relative["label"].startswith("v2 relative mode, plain Ridge")
    assert "plain Ridge" in AUDIT_LABELS["v2_relative_mode"]


def test_v2_rows_equal_the_benchmark_headline(audit: dict[str, Any]) -> None:
    head = _row(audit, "v2_headline")
    assert audit["v2_headline_spec"] == "core_lobo/combined/absolute/ridge_nfs"
    assert head["model"] == "ridge_nfs"
    summary_path = get_paths().benchmark_dir / "summary.json"
    if summary_path.is_file():
        committed = json.loads(summary_path.read_text(encoding="utf-8"))
        assert committed.get("data_sha256") == audit["features_sha256"], "re-run soh benchmark"
        assert head["mae"] == pytest.approx(committed["headline"]["mae"], rel=1e-5)
        assert head["r2"] == pytest.approx(committed["headline"]["r2"], rel=1e-5)


@pytest.mark.skipif(not (COMMITTED / "audit.json").is_file(), reason="no committed audit")
def test_committed_audit_reproduces(audit: dict[str, Any]) -> None:
    committed = json.loads((COMMITTED / "audit.json").read_text(encoding="utf-8"))
    # A stale audit must fail, not skip: re-run `soh audit` after rebuilding the table.
    assert committed["features_sha256"] == audit["features_sha256"], "re-run soh audit"
    assert committed["contaminated_test_overlap"] == audit["contaminated_test_overlap"]
    assert [r["id"] for r in committed["rows"]] == [r["id"] for r in audit["rows"]]
    for old, new in zip(committed["rows"], audit["rows"], strict=True):
        assert old["mae"] == pytest.approx(new["mae"], rel=1e-5)
        assert old["r2"] == pytest.approx(new["r2"], rel=1e-5)
