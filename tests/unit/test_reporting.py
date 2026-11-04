"""Report rendering from fixture JSON: figures, generated docs, README blocks and model cards.

Everything is built in ``tmp_path`` from small, hand-written result files with
the same layout as the real ``summary.json`` / ``audit.json`` / bundles, so
these tests need no data, no network and no display (Agg canvas only).
"""

from __future__ import annotations

import json
import struct
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from battery_soh.config import ProjectPaths, get_paths
from battery_soh.evaluation.benchmark import config_hashes
from battery_soh.models.linear_bundle import write_manifest
from battery_soh.reporting import figures, style
from battery_soh.reporting.model_card import render_model_card
from battery_soh.reporting.render import (
    FIGURES,
    load_inputs,
    readme_audit_block,
    readme_results_block,
    render_all,
    render_results_markdown,
    replace_marked_block,
    update_readme,
)

CELLS = ("B0005", "B0006", "B0007", "B0018")
HEAD = {
    "experiment": "core_lobo",
    "feature_set": "combined",
    "mode": "absolute",
    "model": "ridge_nfs",
}
REL = {"experiment": "core_lobo", "feature_set": "combined", "mode": "relative", "model": "ridge"}
MAE = {"B0005": 1.0, "B0006": 6.5, "B0007": 1.4, "B0018": 1.2}
README = (
    "# Project\r\n\r\nIntro text stays.\r\n\r\n<!-- BEGIN:RESULTS -->\r\nold results\r\n"
    "<!-- END:RESULTS -->\r\n\r\nMiddle.\r\n<!-- BEGIN:AUDIT -->\r\n<!-- END:AUDIT -->\r\nTail.\r\n"
)


def _per_battery(scale: float = 1.0) -> list[dict[str, Any]]:
    return [
        {
            "battery_id": b,
            "n": 40,
            "mae": MAE[b] * scale,
            "rmse": MAE[b] * scale * 1.2,
            "r2": 0.9,
            "bias": (6.4 if b == "B0006" else -0.1) * scale,
            "max_abs_error": 3 * MAE[b],
            "coverage_90": 0.0 if b == "B0006" else 0.97,
            "mean_width_90": 6.0,
            "coverage_90_cv_plus": 0.1 if b == "B0006" else 0.98,
        }
        for b in CELLS
    ]


def _coverage() -> list[dict[str, Any]]:
    return [
        {
            "nominal": nominal,
            "method": "cv",
            "empirical": nominal - 0.15,
            "mean_width": 10 * nominal,
            "min_battery_coverage": 0.0,
            "empirical_cv_plus": nominal - 0.1,
            "mean_width_cv_plus": 11 * nominal,
            "min_battery_coverage_cv_plus": 0.05,
            "per_battery": {
                b: {
                    "coverage": 0.0 if b == "B0006" else 1.0,
                    "coverage_cv_plus": 0.1 if b == "B0006" else 1.0,
                }
                for b in CELLS
            },
        }
        for nominal in (0.8, 0.9, 0.95)
    ]


def _board() -> list[dict[str, Any]]:
    rows = []
    for mode, shift in (("absolute", 0.0), ("relative", -1.2)):
        for fs, base in (("combined", 2.5), ("discharge", 1.7), ("combined_no_cv", 1.3)):
            for model, delta in (("ridge_nfs", 0.0), ("ridge", 0.1), ("hgb", -0.2)):
                rows.append(
                    {
                        **HEAD,
                        "feature_set": fs,
                        "mode": mode,
                        "model": model,
                        "mae": base + delta + shift,
                        "rmse": base + 0.3,
                        "r2": 0.9,
                        "coverage_90": 0.8,
                        "worst_battery": "B0006",
                        "worst_mae": 6.0,
                        "post_hoc": fs == "combined_no_cv",
                        "leaky": False,
                        "baseline": False,
                        "headline": (fs, mode, model) == ("combined", "absolute", "ridge_nfs"),
                    }
                )
        for model, mae, leaky in (
            ("mean", 9.6, False),
            ("cycle_only", 5.3, False),
            ("capacity_oracle", 3.5 if mode == "absolute" else 0.1, True),
        ):
            rows.append(
                {
                    **HEAD,
                    "feature_set": "baseline",
                    "mode": mode,
                    "model": model,
                    "mae": mae,
                    "rmse": mae,
                    "r2": 0.5,
                    "coverage_90": None,
                    "post_hoc": False,
                    "leaky": leaky,
                    "baseline": True,
                    "headline": False,
                }
            )
    return rows


def _summary() -> dict[str, Any]:
    windows = [
        {
            "id": f"dis_380_{lo}",
            "kind": "discharge",
            "v_high": 3.8,
            "v_low": lo / 100,
            "availability": 1.0,
            "capacity_fraction_mean": frac,
            "absolute": {"mae": 1.0 + frac},
            "relative": {"mae": 0.9 + frac / 2},
        }
        for lo, frac in ((370, 0.1), (350, 0.41))
    ] + [
        {
            "id": "chg_390_419",
            "kind": "charge",
            "v_start": 3.9,
            "v_end": 4.19,
            "availability": 0.9,
            "capacity_fraction_mean": 0.54,
            "absolute": {"mae": 3.5},
            "relative": {"mae": 2.0},
        }
    ]
    conditions = {"B0005": "2A_24C", "B0006": "2A_24C", "B0029": "4A_43C", "B0046": "1A_4C"}
    ext_rows = [
        {
            **HEAD,
            "experiment": "extended_lobo",
            "feature_set": "combined_ext",
            "mode": "relative",
            "model": model,
            "mae": mae,
            "worst_battery": "B0006",
            "worst_mae": 3.0,
            "n_abstained": 0,
            "coverage_90": 0.85,
            "mae_ci": [mae - 0.4, mae + 0.5],
            "baseline": False,
            "per_battery": {
                b: {"mae": mae + i * 0.3, "n": 30, "condition": c}
                for i, (b, c) in enumerate(conditions.items())
            },
        }
        for model, mae in (("ridge", 2.1), ("hgb", 1.4))
    ]
    loco_rows = [
        {
            **row,
            "experiment": "leave_one_condition_out",
            "per_condition": {
                c: {"mae": row["mae"] + k} for k, c in enumerate(("2A_24C", "4A_43C", "1A_4C"))
            },
        }
        for row in ext_rows
    ] + [
        {
            **HEAD,
            "feature_set": "baseline",
            "mode": "relative",
            "model": "mean",
            "mae": 8.0,
            "baseline": True,
            "per_condition": {"2A_24C": {"mae": 9.0}},
        }
    ]
    return {
        "schema_version": "1.0",
        "package_version": "2.0.0",
        "generated_utc": "2026-01-01T00:00:00+00:00",
        "label": "soh_pct",
        "config_sha256": "c" * 64,
        "data_sha256": "d" * 64,
        "runtime_s": 12.0,
        "headline": {
            **HEAD,
            "n_batteries": 4,
            "n_cycles": 160,
            "n": 160,
            "mae": 2.53,
            "rmse": 2.8,
            "r2": 0.88,
            "bias": 1.6,
            "worst_mae": 6.5,
            "worst_battery": "B0006",
            "coverage_90": 0.74,
            "mean_width_90": 9.8,
            "pre_registered": True,
        },
        "pre_registration": {"statement": "Fixed in advance."},
        "baselines": {
            "mean": {"mae": 9.6, "rmse": 11.0, "r2": -0.1, "leaky": False},
            "cycle_only": {"mae": 5.3, "rmse": 5.9, "r2": 0.65, "leaky": False},
            "capacity_oracle": {"mae": 3.47, "rmse": 3.5, "r2": 0.86, "leaky": True},
        },
        "leaderboard": _board(),
        "per_battery": _per_battery(),
        "coverage": _coverage(),
        "secondary": [
            {
                **REL,
                "label": "Relative mode, plain ridge",
                "n": 140,
                "n_abstained": 0,
                "mae": 1.08,
                "r2": 0.98,
                "coverage_90": 0.87,
                "post_hoc": False,
                "leaky": False,
            },
            {
                **HEAD,
                "feature_set": "charge",
                "model": "ridge",
                "label": "Charge window only",
                "n": 152,
                "n_abstained": 8,
                "mae": 4.1,
                "r2": 0.5,
                "coverage_90": 0.6,
                "post_hoc": False,
                "leaky": False,
            },
        ],
        "relative_headline": {
            **REL,
            "per_battery": _per_battery(0.3),
            "coverage": _coverage(),
            "reference_cycles_excluded": 20,
        },
        "selection_stability": {
            "headline": {
                "n_folds": 4,
                "mean_pairwise_jaccard": 0.5,
                "frequency": {"dis_q_380_350_ah": 1.0, "chg_cv_time_s": 0.5},
                "folds": [
                    {
                        "held_out": b,
                        "selected": ["dis_q_380_350_ah"],
                        "alpha": 0.1,
                        "inner_mae": 1.0,
                    }
                    for b in CELLS
                ],
            },
            "specs": {
                "core_lobo/combined/absolute/ridge_nfs": {
                    "always_selected": ["dis_q_380_350_ah"],
                    "never_selected": [],
                    "mean_pairwise_jaccard": 0.5,
                }
            },
        },
        "leakage_checks": {
            "core_lobo/combined": {
                "passed": True,
                "max_within_battery_abs_r_with_label": 0.99,
                "capacity_reconstruction_lobo_r2": 0.94,
            }
        },
        "window_sweep": {
            "status": "ok",
            "note": "Exploratory.",
            "windows": windows,
            "lowest_mae_window": {"absolute": "dis_380_370", "relative": "dis_380_370"},
        },
        "extended": {
            "conditions": conditions,
            "n_qc_valid_cycles": 120,
            "relative_keep_absolute": ["dis_temp_rise_win_c"],
            "results": ext_rows,
            "wilcoxon": [
                {
                    "feature_set": "combined_ext",
                    "mode": "relative",
                    "reference_model": "ridge",
                    "best_candidate": "hgb",
                    "mae_reference": 2.1,
                    "mae_best_candidate": 1.4,
                    "reference_better_count": 1,
                    "n_pairs": 4,
                    "p_value": 0.2,
                    "selection_note": "favours the candidate",
                }
            ],
        },
        "loco": {
            "conditions": {"2A_24C": [], "4A_43C": [], "1A_4C": []},
            "note": "LOCO.",
            "results": loco_rows,
        },
        "notes": ["Note one."],
        "provenance": {"config_path": "configs/benchmark.yaml", "python": "3.11", "seed": 42},
    }


def _predictions() -> pd.DataFrame:
    rows = []
    for spec, offset in ((HEAD, 1.0), (REL, -0.3)):
        for b in CELLS:
            for k in range(40):
                y = 100.0 - 0.6 * k - (10 if b == "B0006" else 0) * k / 40
                rows.append(
                    {
                        **spec,
                        "battery_id": b,
                        "cycle_index": k,
                        "y": y,
                        "yhat": y + offset,
                        "lo_90": y + offset - 3,
                        "hi_90": y + offset + 3,
                        "scored": k >= 5 or spec is HEAD,
                    }
                )
    return pd.DataFrame(rows)


def _audit() -> dict[str, Any]:
    ids = list(style.AUDIT_LABELS)
    return {
        "generated_utc": "2026-01-01T00:00:00+00:00",
        "rows": [
            {
                "id": rid,
                "label": rid.replace("_", " "),
                "category": "v2" if rid.startswith("v2") else "honest",
                "mae": 1.0 + i * 0.4,
                "r2": 0.9,
                "leaky": not rid.startswith("v2"),
                "protocol": "LOBO",
            }
            for i, rid in enumerate(ids)
        ],
        "contaminated_test_overlap": {"overlapping_rows": 284, "test_rows": 336},
        "soh_vs_capacity_within_battery_r2": 1.0,
        "capacity_reconstruction_r2": 0.995,
    }


def _bundle(mode: str, features: list[str], coef: list[float]) -> dict[str, Any]:
    candidates = ["chg_cv_time_s", "chg_ica_peak_height_ah_per_v", "dis_q_380_350_ah", "dis_r0_ohm"]
    return {
        "model_id": f"combined-{mode}-ridge",
        "feature_set": "combined",
        "mode": mode,
        "features": features,
        "estimator": {"type": "ridge", "alpha": 0.1, "coef": coef, "intercept": 85.0},
        "selection": {"candidates": candidates},
        "training": {"n_cycles": 150},
        "conformal": {"quantiles": {"0.8": 4.0, "0.9": 6.0, "0.95": 9.0}},
        "metrics": {"mean": {"mae": 2.5}, "coverage": {"0.9": 0.74}},
        "model_card": f"model_cards/combined-{mode}-ridge.md",
    }


def _write_tree(root: Path) -> ProjectPaths:
    paths = ProjectPaths(root=root)
    bench = root / "reports" / "benchmark"
    bench.mkdir(parents=True)
    (bench / "summary.json").write_text(json.dumps(_summary()), encoding="utf-8")
    _predictions().to_csv(bench / "predictions.csv", index=False)
    (root / "reports" / "audit").mkdir(parents=True)
    (root / "reports" / "audit" / "audit.json").write_text(json.dumps(_audit()), encoding="utf-8")
    models = root / "artifacts" / "models"
    models.mkdir(parents=True)
    for b in (
        _bundle("absolute", ["chg_ica_peak_height_ah_per_v", "dis_q_380_350_ah"], [-2.0, 11.5]),
        _bundle("relative", ["chg_cv_time_s", "dis_q_380_350_ah"], [1.8, 12.0]),
    ):
        (models / f"{b['model_id']}.json").write_text(json.dumps(b), encoding="utf-8")
    write_manifest(models, created_utc="2026-01-01T00:00:00+00:00")
    (root / "README.md").write_bytes(README.encode("utf-8"))
    return paths


def test_bundles_failing_their_manifest_are_left_out(tmp_path: Path) -> None:
    paths = _write_tree(tmp_path)
    (paths.models_dir / "combined-absolute-ridge.json").write_text("{}", encoding="utf-8")
    inputs = load_inputs(paths)
    assert inputs.bundles == ()
    assert inputs.manifest is None


@pytest.fixture(scope="module")
def rendered(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[tuple[ProjectPaths, list[Path], dict[str, bytes]]]:
    """One full render (and the bytes of every output) of the fixture tree."""
    with pytest.MonkeyPatch.context() as mp:
        for var in ("SOH_REPORTS_DIR", "SOH_ARTIFACTS_DIR", "SOH_RAW_DIR", "SOH_FRONTEND_DIR"):
            mp.delenv(var, raising=False)
        paths = _write_tree(tmp_path_factory.mktemp("report"))
        written = render_all(paths)
        snapshot = {p.relative_to(paths.root).as_posix(): p.read_bytes() for p in written}
        yield paths, written, snapshot


def _png_size(data: bytes) -> tuple[int, int]:
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack(">II", data[16:24])
    return width, height


# --------------------------------------------------------------------------- render_all


def test_every_available_figure_is_a_readme_width_png(rendered: Any) -> None:
    _, _, snapshot = rendered
    produced = sorted(n.rsplit("/", 1)[1] for n in snapshot if n.endswith(".png"))
    # No feature table in the fixture tree: only the indicator scatter is skipped.
    assert produced == sorted(set(FIGURES) - {"indicators_vs_soh.png"})
    for name in produced:
        width, height = _png_size(snapshot[f"reports/figures/{name}"])
        assert width == round(style.WIDTH_IN * style.DPI), name
        assert 300 < height < 1600, name


def test_results_document_is_generated_and_honest(rendered: Any) -> None:
    _, _, snapshot = rendered
    text = snapshot["docs/RESULTS.md"].decode("utf-8")
    assert text.startswith("<!-- generated by `soh report`")
    assert "do not edit" in text
    assert "\r\n" not in text
    assert "**2.53**" in text  # the pre-specified headline, as recorded
    assert "B0006 has MAE 6.50" in text  # the cell that dominates it is named
    assert "## Pre-specified headline" in text
    assert "pre-registered" not in text.lower()
    assert "not as a blinded pre-registration" in text  # what the config hash does not show
    # Secondary rows: which cycles each is scored on, computed from the summary.
    assert (
        "Same cells and folds; scored cycles differ (headline 160; relative 140, reference "
        "cycles excluded; `charge` abstain on 8 cycles where none of their inputs was measured)."
    ) in text
    assert "Same cells, folds and rows" not in text
    assert "| R2 (pooled) |" in text  # pooled R2 is labelled as pooled
    assert "Relative mode, plain Ridge (`combined` / relative / `ridge`)" in text
    # The sweep's winner is named as the best of N windows on the same cells.
    assert "Lowest MAE: absolute 1.10 (discharge 3.80 -> 3.70 V)" in text
    assert "best of the 3 windows scored on the same cells" in text
    assert "deterministic timestamp 2026-01-01T00:00:00+00:00" in text
    assert "(post hoc)" in text
    assert "**LEAKY**" in text
    assert "_Figure `indicators_vs_soh.png` not rendered" in text
    for name in set(FIGURES) - {"indicators_vs_soh.png"}:
        assert f"](../reports/figures/{name})" in text
    # The config hash recorded by the benchmark is compared with the file on disk.
    assert "c" * 64 in text


def test_readme_blocks_replace_only_the_marked_regions(rendered: Any) -> None:
    _, _, snapshot = rendered
    readme = snapshot["README.md"].decode("utf-8")
    before, rest = readme.split("<!-- BEGIN:RESULTS -->", 1)
    assert before == "# Project\r\n\r\nIntro text stays.\r\n\r\n"
    assert "old results" not in readme
    assert readme.endswith("<!-- END:AUDIT -->\r\nTail.\r\n")
    assert "\r\nMiddle.\r\n" in rest
    assert "Pre-specified headline" in readme
    assert "**LEAKY:** Capacity oracle" in readme
    assert readme.count("\n") == readme.count("\r\n")  # line endings preserved


def test_text_outputs_are_idempotent(rendered: Any) -> None:
    paths, _, snapshot = rendered
    inputs = load_inputs(paths)
    figs = {n: paths.figures_dir / n for n in FIGURES if (paths.figures_dir / n).is_file()}
    assert render_results_markdown(inputs, figs).encode("utf-8") == snapshot["docs/RESULTS.md"]
    assert not update_readme(paths.root / "README.md", inputs.summary, inputs.audit)
    assert (paths.root / "README.md").read_bytes() == snapshot["README.md"]


def test_figures_are_byte_identical_across_renders(rendered: Any, tmp_path: Path) -> None:
    _, _, snapshot = rendered
    again = figures.coverage(_summary(), tmp_path / "coverage.png")
    assert again.read_bytes() == snapshot["reports/figures/coverage.png"]
    again = figures.leaderboard(_summary(), tmp_path / "leaderboard.png")
    assert again.read_bytes() == snapshot["reports/figures/leaderboard.png"]


def test_missing_summary_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="soh benchmark"):
        render_all(ProjectPaths(root=tmp_path))


# --------------------------------------------------------------------------- markers


def test_marker_replacement_rules() -> None:
    text = "a\n<!-- BEGIN:X -->\nold\n<!-- END:X -->\nb\n"
    assert (
        replace_marked_block(text, "X", "new\n") == "a\n<!-- BEGIN:X -->\nnew\n<!-- END:X -->\nb\n"
    )
    assert replace_marked_block("no markers\n", "X", "new") == "no markers\n"
    with pytest.raises(ValueError, match="exactly one"):
        replace_marked_block(text + "<!-- BEGIN:X -->", "X", "new")
    with pytest.raises(ValueError, match="comes before"):
        replace_marked_block("<!-- END:X -->\n<!-- BEGIN:X -->", "X", "new")


def test_readme_untouched_without_markers(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    original = b"# Readme\r\nnothing generated here\r\n"
    readme.write_bytes(original)
    stamp = readme.stat().st_mtime_ns
    assert not update_readme(readme, _summary(), _audit())
    assert readme.read_bytes() == original
    assert readme.stat().st_mtime_ns == stamp
    assert not update_readme(tmp_path / "missing.md", _summary(), None)


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def test_readme_audit_block_is_three_short_columns() -> None:
    audit = _audit()
    block = readme_audit_block(audit)
    lines = [line for line in block.splitlines() if line.startswith("| ")]
    assert _cells(lines[0]) == ["Evaluation", "MAE", "R\N{SUPERSCRIPT TWO}"]
    assert _cells(lines[1]) == ["---", "---:", "---:"]
    rows = [_cells(line) for line in lines[2:]]
    assert len(rows) == len(audit["rows"]) == len(style.AUDIT_LABELS)
    for cells, row in zip(rows, audit["rows"], strict=True):
        assert len(cells) == 3
        short = style.AUDIT_LABELS[row["id"]]
        assert cells[0] == short + (" \N{DAGGER}" if row["leaky"] else "")
        assert len(cells[0]) <= 50  # fits GitHub's README column next to two numbers
        assert cells[1] == f"{row['mae']:.2f}"
        assert cells[2] == f"{row['r2']:.3f}"
    # The dagger is explained once, under the table, with a link to the full table.
    footnote = block.splitlines()[-1]
    assert footnote.startswith("\N{DAGGER} Leaky:")
    assert "docs/RESULTS.md#leakage-audit-v1-vs-v2" in footnote
    # Unknown rows fall back to their own label.
    extra = {"rows": [{"id": "new_row", "label": "New row", "mae": 1.0, "r2": 0.5}]}
    assert _cells(readme_audit_block(extra).splitlines()[2]) == ["New row", "1.00", "0.500"]


def test_full_audit_table_stays_in_the_results_document(rendered: Any) -> None:
    _, _, snapshot = rendered
    readme = snapshot["README.md"].decode("utf-8")
    results = snapshot["docs/RESULTS.md"].decode("utf-8")
    audit_block = readme.split("<!-- BEGIN:AUDIT -->", 1)[1].split("<!-- END:AUDIT -->", 1)[0]
    assert "| Evaluation | MAE | R\N{SUPERSCRIPT TWO} |" in audit_block
    assert "| Category |" not in audit_block
    assert "| Protocol |" not in audit_block
    assert "| Evaluation | Category | MAE | R2 (pooled) | Leaky | Protocol |" in results


def _config_hash_tree(root: Path, edit: Any = None) -> ProjectPaths:
    """Fixture tree plus a real config whose scoped hashes the results and bundles record."""
    paths = _write_tree(root)
    config = root / "configs" / "benchmark.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_bytes((get_paths().configs / "benchmark.yaml").read_bytes())
    hashes = config_hashes(config)
    summary_path = paths.benchmark_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["config_sha256"] = hashes["benchmark"]
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    for bundle_path in sorted(paths.models_dir.glob("*-ridge.json")):
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        bundle["provenance"] = {
            "benchmark_config_sha256": hashes["benchmark"],
            "production_config_sha256": hashes["production"],
        }
        bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
    write_manifest(paths.models_dir, created_utc="2026-01-01T00:00:00+00:00")
    if edit is not None:
        raw = yaml.safe_load(config.read_text(encoding="utf-8"))
        edit(raw)
        config.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return paths


def _provenance_rows(paths: ProjectPaths) -> dict[str, str]:
    inputs = load_inputs(paths)
    text = render_results_markdown(inputs, {})
    section = text.split("## Provenance", 1)[1].split("\n## ", 1)[0]
    return {
        cells[0]: cells[1]
        for cells in (_cells(line) for line in section.splitlines() if line.startswith("| "))
    }


def test_provenance_shows_both_config_hashes_and_their_freshness(tmp_path: Path) -> None:
    paths = _config_hash_tree(tmp_path / "fresh")
    hashes = config_hashes(paths.configs / "benchmark.yaml")
    rows = _provenance_rows(paths)
    bench = rows["Config SHA-256, benchmark sections"]
    production = rows["Config SHA-256, production section"]
    assert (
        bench == f"`{hashes['benchmark']}` (matches the benchmark sections of the config on disk)"
    )
    assert production == f"`{hashes['production']}` (matches the production section on disk)"
    assert "trained on this benchmark run's results" in rows["Model bundles"]


def test_production_edit_flags_only_the_bundles(tmp_path: Path) -> None:
    paths = _config_hash_tree(
        tmp_path / "prod", lambda c: c["production"].update(created_utc="2030-01-01T00:00:00Z")
    )
    rows = _provenance_rows(paths)
    assert "(matches the benchmark sections" in rows["Config SHA-256, benchmark sections"]
    assert "re-run `soh train`" in rows["Config SHA-256, production section"]


def test_benchmark_edit_flags_the_benchmark(tmp_path: Path) -> None:
    paths = _config_hash_tree(tmp_path / "bench", lambda c: c.update(seed=7))
    rows = _provenance_rows(paths)
    assert "re-run `soh benchmark`" in rows["Config SHA-256, benchmark sections"]
    assert "(matches the production section" in rows["Config SHA-256, production section"]


def test_bundles_from_another_benchmark_run_are_flagged(tmp_path: Path) -> None:
    paths = _config_hash_tree(tmp_path / "other")
    summary_path = paths.benchmark_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["config_sha256"] = "0" * 64
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    rows = _provenance_rows(paths)
    assert "not this run's - re-run `soh train`" in rows["Model bundles"]


_REPO = get_paths()


@pytest.mark.skipif(
    not (_REPO.benchmark_dir / "summary.json").is_file()
    or not (_REPO.docs / "RESULTS.md").is_file(),
    reason="no committed results",
)
def test_committed_generated_documents_are_up_to_date() -> None:
    inputs = load_inputs(_REPO)
    figs = {n: _REPO.figures_dir / n for n in FIGURES if (_REPO.figures_dir / n).is_file()}
    committed = (_REPO.docs / "RESULTS.md").read_bytes().replace(b"\r\n", b"\n")
    assert committed == render_results_markdown(inputs, figs).encode("utf-8"), "run `soh report`"
    readme = (_REPO.root / "README.md").read_text(encoding="utf-8")
    blocks = {"RESULTS": readme_results_block(inputs.summary)}
    if inputs.audit is not None:
        blocks["AUDIT"] = readme_audit_block(inputs.audit)
    for name, block in blocks.items():
        assert replace_marked_block(readme, name, block) == readme, f"README {name} block"


def test_readme_results_block_lists_headline_relative_baselines_and_oracle() -> None:
    block = readme_results_block(_summary())
    lines = [line for line in block.splitlines() if line.startswith("| ")][2:]
    assert lines[0].startswith("| Pre-specified headline")
    assert any("Relative mode, nested selection" in line for line in lines)
    assert "| R2 (pooled) |" in block
    assert any(line.startswith("| **LEAKY:** Capacity oracle") for line in lines)
    assert "docs/RESULTS.md" in block


# --------------------------------------------------------------------------- figure helpers


def test_strongest_indicators_ranks_by_within_cell_correlation() -> None:
    rng = np.random.default_rng(0)
    rows = []
    for b, offset in (("B0005", 0.0), ("B0006", 50.0)):
        soh = np.linspace(100, 70, 60)
        rows.append(
            pd.DataFrame(
                {
                    "battery_id": b,
                    "qc_valid": True,
                    "soh_pct": soh,
                    "a": soh
                    + offset
                    + rng.normal(0, 0.1, 60),  # strong within each cell despite offset
                    "b": -soh + rng.normal(0, 5.0, 60),  # weaker
                    "c": rng.normal(0, 1.0, 60),  # unrelated
                }
            )
        )
    table = pd.concat(rows, ignore_index=True)
    ranked = figures.strongest_indicators(table, ["c", "b", "a"], batteries=["B0005", "B0006"], n=2)
    assert [name for name, _ in ranked] == ["a", "b"]
    assert ranked[0][1] > 0.99


def test_indicator_scatter_renders(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    soh = np.tile(np.linspace(100, 70, 30), 4)
    table = pd.DataFrame(
        {
            "battery_id": np.repeat(CELLS, 30),
            "qc_valid": True,
            "soh_pct": soh,
            "dis_q_380_350_ah": soh / 100 + rng.normal(0, 0.01, soh.size),
            "chg_cv_time_s": 9000 - 50 * soh + rng.normal(0, 30, soh.size),
        }
    )
    out = figures.indicators_vs_soh(
        table, ["dis_q_380_350_ah", "chg_cv_time_s"], tmp_path / "x.png", batteries=CELLS
    )
    assert _png_size(out.read_bytes())[0] == 900


def test_reference_labels_never_overlap() -> None:
    with style.context():
        fig, axes = style.figure(height_in=2.0)
        ax = axes[0][0]
        ax.set_xlim(0, 10)
        ax.set_ylim(1, -2)
        fig.draw_without_rendering()
        lines = [
            (3.4, "Capacity oracle (leaky) 3.47", "red"),
            (5.3, "Cycle count only 5.30", "grey"),
            (9.6, "Training mean 9.61", "grey"),
            (9.8, "Another baseline 9.80", "grey"),
        ]
        rows = figures._place_reference_labels(ax, lines, rows=(-1.0, -1.6))
        assert 1 <= rows <= 2
        texts = list(ax.texts)
        assert len(texts) == len(lines)
        extents = [(style.data_extent(ax, t), t.get_position()[1]) for t in texts]
        for i, ((a0, a1), ya) in enumerate(extents):
            assert a0 >= 0
            assert a1 <= 10
            for (b0, b1), yb in extents[i + 1 :]:
                if ya == yb:
                    assert a1 <= b0 or b1 <= a0  # same row -> disjoint


# --------------------------------------------------------------------------- model card


def _card_bundle() -> dict[str, Any]:
    per = {
        b: {
            "n": 40,
            "mae": MAE[b],
            "rmse": MAE[b] * 1.1,
            "r2": 0.9,
            "bias": 6.42 if b == "B0006" else -0.1,
            "coverage": {"0.8": 0.7, "0.9": 0.0 if b == "B0006" else 0.97, "0.95": 1.0},
            "mean_width": {"0.9": 6.0},
        }
        for b in CELLS
    }
    return {
        "model_id": "combined-absolute-ridge",
        "feature_set": "combined",
        "mode": "absolute",
        "n_reference_cycles": 5,
        "features": ["dis_q_380_350_ah", "chg_cv_time_s"],
        "estimator": {"type": "ridge", "alpha": 0.1, "coef": [11.9, -1.0], "intercept": 84.0},
        "conformal": {
            "method": "group conformal (cv)",
            "n_scores": 160,
            "quantiles": {"0.8": 4.6, "0.9": 6.9, "0.95": None},
        },
        "domain": {
            "feature_min": [0.3, 3500.0],
            "feature_max": [1.0, 5500.0],
            "margin": 0.1,
            "mahalanobis_threshold": 4.8,
            "mahalanobis_quantile": 0.99,
        },
        "training": {"batteries": list(CELLS), "n_cycles": 160, "features_sha256": "f" * 64},
        "label": {"name": "soh_pct", "definition": "100 * capacity / C0"},
        "metrics": {
            "spec": HEAD,
            "per_battery": per,
            "mean": {"mae": 2.53, "n_cycles": 160},
            "coverage": {"0.9": 0.74},
            "coverage_min_battery": {"0.9": 0.0},
        },
        "selection": {
            "candidates": ["dis_q_380_350_ah", "chg_cv_time_s", "dis_r0_ohm"],
            "final": {
                "selected": ["dis_q_380_350_ah", "chg_cv_time_s"],
                "alpha": 0.1,
                "inner_mae": 1.0,
                "path": [
                    {"removed": None, "n_features": 3},
                    {"removed": "dis_r0_ohm", "n_features": 2},
                ],
            },
            "lobo_folds": [{"held_out": "B0006", "selected": ["chg_cv_time_s"], "alpha": 0.1}],
        },
        "training_protocol": {
            "label_range_per_battery": {
                "B0005": [70.1, 101.2],
                "B0006": [57.3, 101.1],
                "B0007": [74.5, 100.6],
                "B0018": [72.9, 100.8],
            }
        },
        "provenance": {"config": "configs/benchmark.yaml"},
    }


def test_model_card_states_the_uncomfortable_numbers() -> None:
    text = render_model_card(_card_bundle())
    for section in (
        "## Intended use",
        "## Not intended for",
        "## Training data",
        "## Label",
        "## Inputs",
        "## Feature selection",
        "## Evaluation (leave-one-battery-out)",
        "## Uncertainty",
        "## Domain of applicability",
        "## Caveats and limitations",
        "## Ethical and safety considerations",
        "## Provenance",
    ):
        assert section in text, section
    assert "B0006 is covered 0% of the time" in text
    assert "over-estimates its SOH by 6.42 points" in text
    assert "B0006 fades to 57.3% SOH, below every other training cell (lowest 70.1%)" in text
    assert "not a safety-certified battery-management function" in text
    assert "| 95% | unbounded |" in text  # an infinite quantile is shown, not hidden
    assert "the per-cell models that hold out B0006 keep it; the shipped model uses it" in text
    assert "\r" not in text
    assert text.endswith("\n")


def test_model_card_claims_measured_not_guaranteed_coverage() -> None:
    text = render_model_card(_card_bundle())
    assert "Research and demonstration on laboratory cells like the training cells" in text
    assert (
        "battery-level conformal intervals whose coverage is measured on held-out cells, "
        "not guaranteed"
    ) in text
    assert (
        "Even for an exchangeable cell this fixed half-width has no finite-sample guarantee; "
        "its coverage is measured"
    ) in text
    assert "calibrated" not in text
    assert "conformal guarantee" not in text
    assert "fleet-level" not in text.lower()
    assert "| Timestamp (UTC) | n/a (deterministic: the feature table's build time" in text


def test_relative_spec_is_named_plain_ridge_in_figures() -> None:
    assert figures._spec_text(REL) == "combined / relative / plain Ridge"
    assert figures._spec_text(HEAD) == "combined / absolute / Ridge + nested selection"
    assert "plain Ridge" in style.AUDIT_LABELS["v2_relative_mode"]


def test_report_rendering_does_not_import_scikit_learn() -> None:
    code = (
        "import sys\n"
        "import battery_soh.reporting.render\n"
        "from battery_soh.evaluation.benchmark import config_hashes\n"
        "assert callable(config_hashes)\n"
        "print(sorted(m for m in sys.modules if m.split('.')[0] == 'sklearn'))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=600
    )
    assert out.stdout.strip() == "[]"
