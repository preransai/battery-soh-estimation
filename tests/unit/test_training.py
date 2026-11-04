"""Unit tests of the production-training configuration, row rules and input checks.

The end-to-end path (benchmark -> train -> serve) is covered by
``tests/integration/test_training_bundles.py``; these tests pin the rules that
decide *what* gets trained on and when training must refuse to run.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from battery_soh.config import get_paths
from battery_soh.data.protocols import CORE_BATTERIES
from battery_soh.evaluation.benchmark import config_hashes, load_config
from battery_soh.training import (
    TrainingError,
    _FeatureTable,
    _load_benchmark_run,
    _load_features,
    load_production_config,
    training_frame,
)

CONFIG = get_paths().configs / "benchmark.yaml"


def _raw_config() -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    return data


def _write(tmp_path: Path, data: dict[str, Any]) -> Path:
    path = tmp_path / "benchmark.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- config


def test_committed_config_defines_the_four_served_bundles() -> None:
    pcfg = load_production_config(CONFIG)
    ids = [s.model_id for s in pcfg.specs]
    assert sorted(ids) == sorted(
        [
            "charge-absolute-ridge",
            "discharge-absolute-ridge",
            "combined-absolute-ridge",
            "combined-relative-ridge",
        ]
    )
    # One procedure for every bundle: the pre-registered headline model.
    assert {s.model for s in pcfg.specs} == {pcfg.benchmark.headline.model} == {"ridge_nfs"}
    assert pcfg.batteries == CORE_BATTERIES
    assert {s.experiment for s in pcfg.specs} == {"core_lobo"}
    # Two scoped hashes: the production section, and the benchmark sections it builds on.
    hashes = config_hashes(CONFIG)
    assert pcfg.sha256 == hashes["production"]
    assert pcfg.benchmark.sha256 == hashes["benchmark"]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda c: c["production"].update(extra=1), "unknown key"),
        (
            lambda c: c["production"]["specs"][0].update(feature_set="capacity_oracle"),
            "leaky|proxy|label",
        ),
        (lambda c: c["production"]["specs"][0].update(model="hgb"), "must be one of"),
        (lambda c: c["production"]["specs"][0].update(feature_set="charge_ext"), "not evaluated"),
        (lambda c: c["production"]["specs"][0].update(mode="sideways"), "mode must be"),
        (lambda c: c["production"]["specs"].append(dict(c["production"]["specs"][0])), "duplicate"),
        (lambda c: c["production"].update(specs=[]), "non-empty list"),
        (lambda c: c.pop("production"), "production must be a mapping"),
        (
            lambda c: c["production"].update(created_utc=datetime(2026, 1, 1, tzinfo=UTC)),
            "created_utc must be a string",
        ),
        (
            lambda c: c["experiments"]["core_lobo"].update(
                relative_keep_absolute=["dis_temp_rise_win_c"]
            ),
            "API relativizes",
        ),
    ],
)
def test_invalid_production_sections_are_rejected(
    tmp_path: Path, mutate: Any, message: str
) -> None:
    data = _raw_config()
    mutate(data)
    with pytest.raises(ValueError, match=message):
        load_production_config(_write(tmp_path, data))


# --------------------------------------------------------------------------- rows


def _table() -> pd.DataFrame:
    """Two cells x 9 cycles with one QC failure, one missing label, one unmeasured cycle."""
    rows = []
    for b, scale in (("B0005", 1.0), ("B0006", 2.0)):
        for k in range(9):
            rows.append(
                {
                    "battery_id": b,
                    "cycle_index": k,
                    "qc_valid": not (b == "B0005" and k == 1),
                    "soh_pct": np.nan if (b == "B0006" and k == 8) else 100.0 - k,
                    "dis_q_380_350_ah": np.nan if (b == "B0006" and k == 0) else scale * (10 - k),
                    "dis_r0_ohm": np.nan if (b == "B0006" and k == 0) else 0.07 + 0.001 * k,
                    "ctx_cycle_index": float(k),
                }
            )
    # Shuffle so the function has to sort.
    return pd.DataFrame(rows).sample(frac=1.0, random_state=0).reset_index(drop=True)


def test_training_frame_absolute_applies_the_benchmark_row_rules() -> None:
    frame = training_frame(
        _table(),
        batteries=("B0005", "B0006"),
        features=("dis_q_380_350_ah", "dis_r0_ohm"),
        mode="absolute",
    )
    keys = list(zip(frame["battery_id"], frame["cycle_index"], strict=True))
    assert ("B0005", 1) not in keys  # QC-invalid
    assert ("B0006", 8) not in keys  # no label
    assert ("B0006", 0) not in keys  # no input measured
    assert keys == sorted(keys)
    assert len(frame) == 18 - 3
    assert not frame["is_reference_cycle"].any()


def test_training_frame_relative_divides_by_the_first_valid_values() -> None:
    frame = training_frame(
        _table(),
        batteries=("B0005", "B0006"),
        features=("dis_q_380_350_ah", "ctx_cycle_index"),
        mode="relative",
        n_ref=3,
    )
    b5 = frame[frame["battery_id"] == "B0005"].set_index("cycle_index")
    # B0005 reference: QC-valid cycles 0, 2, 3 -> values 10, 8, 7 -> mean 25/3.
    np.testing.assert_allclose(b5["dis_q_380_350_ah"], (10 - b5.index.to_numpy()) / (25 / 3))
    assert b5["is_reference_cycle"].to_dict() == {k: k in (0, 2, 3) for k in b5.index}
    b6 = frame[frame["battery_id"] == "B0006"].set_index("cycle_index")
    # B0006 cycle 0 is unmeasured: the reference is cycles 1-3 (values 18, 16, 14).
    np.testing.assert_allclose(b6["dis_q_380_350_ah"], 2 * (10 - b6.index.to_numpy()) / 16.0)
    np.testing.assert_array_equal(b6["ctx_cycle_index"], b6.index.to_numpy(float))  # context


def test_training_frame_rejects_unknown_cells_and_columns() -> None:
    with pytest.raises(ValueError, match="not in the feature table"):
        training_frame(_table(), batteries=("B0042",), features=("dis_r0_ohm",), mode="absolute")
    with pytest.raises(ValueError, match="lacks columns"):
        training_frame(_table(), batteries=("B0005",), features=("eis_re_ohm",), mode="absolute")


# --------------------------------------------------------------------------- inputs


def _features(sha: str = "f" * 64) -> _FeatureTable:
    return _FeatureTable(table=pd.DataFrame(), sha256=sha, created_utc="x", file_name="f.csv")


def _summary(tmp_path: Path, **fields: Any) -> Path:
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(fields), encoding="utf-8")
    return path


def test_consumed_benchmark_run_must_match_config_and_data(tmp_path: Path) -> None:
    cfg = load_config(CONFIG)
    good = {"config_sha256": cfg.sha256, "data_sha256": "f" * 64}
    run = _load_benchmark_run(_summary(tmp_path, **good), cfg=cfg, features=_features())
    assert run.config_sha256 == cfg.sha256 == config_hashes(CONFIG)["benchmark"]

    with pytest.raises(TrainingError, match="other benchmark settings"):
        _load_benchmark_run(
            _summary(tmp_path, **{**good, "config_sha256": "0" * 64}),
            cfg=cfg,
            features=_features(),
        )
    with pytest.raises(TrainingError, match="another feature table"):
        _load_benchmark_run(_summary(tmp_path, **good), cfg=cfg, features=_features("e" * 64))
    with pytest.raises(TrainingError, match="run `soh benchmark` first"):
        _load_benchmark_run(tmp_path / "absent.json", cfg=cfg, features=_features())
    (tmp_path / "bad.json").write_text("[1]", encoding="utf-8")
    with pytest.raises(TrainingError, match="JSON object"):
        _load_benchmark_run(tmp_path / "bad.json", cfg=cfg, features=_features())


def test_stale_feature_sidecar_is_fatal(tmp_path: Path) -> None:
    csv = tmp_path / "features.csv"
    csv.write_bytes(b"battery_id\nB0005\n")
    meta = {"sha256": hashlib.sha256(b"something else").hexdigest(), "created_utc": "x"}
    (tmp_path / "features.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(TrainingError, match="does not match the sha256"):
        _load_features(csv)
    (tmp_path / "features.meta.json").unlink()
    with pytest.raises(TrainingError, match="is missing"):
        _load_features(csv)
