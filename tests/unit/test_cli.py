"""Smoke tests of the ``soh`` command line (in-process, via Typer's CliRunner)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from battery_soh import __version__
from battery_soh.cli import app
from battery_soh.config import get_paths

runner = CliRunner()

needs_feature_table = pytest.mark.skipif(
    not get_paths().features_csv.is_file(),
    reason="the committed feature table data/processed/nasa_features.csv is absent",
)


def test_help_lists_every_command() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    for command in ("info", "data", "audit", "benchmark", "train", "report", "serve"):
        assert command in result.output


def test_info_prints_the_version_and_paths() -> None:
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0, result.output
    assert __version__ in result.output
    assert "features_csv" in result.output
    assert "models_dir" in result.output


def test_audit_prints_n_a_for_a_missing_mae(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_run_audit(_features: Path, _out_dir: Path) -> dict[str, Any]:
        return {
            "rows": [
                {"mae": 1.2345, "category": "v1_reported", "label": "as reported"},
                {"mae": None, "category": "honest", "label": "no prediction"},
                {"category": None, "label": None},
            ]
        }

    monkeypatch.setattr("battery_soh.evaluation.audit.run_audit", fake_run_audit)
    result = runner.invoke(app, ["audit", "--out-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert lines[0].startswith("  1.234  v1_reported")
    assert lines[1].startswith("    n/a  honest")
    assert lines[2].startswith("    n/a")
    assert f"Audit written to {tmp_path}" in result.output


# The real audit and quick benchmark run once per session (the ``audit_dir`` and
# ``quick_benchmark_dir`` fixtures); the CLI tests replay those outputs into their own folder.


@needs_feature_table
def test_audit_writes_its_outputs_into_out_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, audit_dir: Path
) -> None:
    calls: list[tuple[Path, Path]] = []

    def replay_run_audit(features: Path, out_dir: Path) -> dict[str, Any]:
        calls.append((features, out_dir))
        shutil.copytree(audit_dir, out_dir)
        summary: dict[str, Any] = json.loads((out_dir / "audit.json").read_text(encoding="utf-8"))
        return summary

    monkeypatch.setattr("battery_soh.evaluation.audit.run_audit", replay_run_audit)
    out = tmp_path / "audit"
    result = runner.invoke(app, ["audit", "--out-dir", str(out)])
    assert result.exit_code == 0, result.output
    assert calls == [(get_paths().features_csv, out)]  # the committed table by default
    audit = json.loads((out / "audit.json").read_text(encoding="utf-8"))
    assert audit["rows"]
    assert all(row["label"] in result.output for row in audit["rows"])
    assert f"Audit written to {out}" in result.output


@needs_feature_table
def test_quick_benchmark_writes_into_out_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, quick_benchmark_dir: Path
) -> None:
    calls: list[tuple[Path, Path, bool]] = []

    def replay_run_benchmark(config: Path, out_dir: Path, *, quick: bool = False) -> dict[str, Any]:
        calls.append((config, out_dir, quick))
        shutil.copytree(quick_benchmark_dir, out_dir)
        summary: dict[str, Any] = json.loads(
            (out_dir / "quick" / "summary.json").read_text(encoding="utf-8")
        )
        return summary

    monkeypatch.setattr("battery_soh.evaluation.benchmark.run_benchmark", replay_run_benchmark)
    out = tmp_path / "benchmark"
    result = runner.invoke(app, ["benchmark", "--quick", "--out-dir", str(out)])
    assert result.exit_code == 0, result.output
    assert calls == [(get_paths().configs / "benchmark.yaml", out, True)]  # committed config
    summaries = list(out.rglob("summary.json"))
    assert len(summaries) == 1
    # The message names the folder that was actually written (quick mode: a subfolder).
    assert f"Quick results written to {summaries[0].parent}" in result.output
    assert "Full results" not in result.output
    headline = json.loads(summaries[0].read_text(encoding="utf-8"))["headline"]
    printed = json.loads(result.output[result.output.index("{") : result.output.rindex("}") + 1])
    assert printed["mae"] == pytest.approx(headline["mae"])
