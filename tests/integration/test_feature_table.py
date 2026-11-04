"""Integrity of the committed feature table ``data/processed/nasa_features.csv``.

The CSV is the input of the benchmark, the audit and the CI reproduction job, so these tests run
without raw data: schema, row order, label consistency, the sidecar checksum, the leakage guards
and the documentation. Tests marked ``raw_data`` rebuild rows from the ``.mat`` files and demand
byte-identical output (skipped when the raw files are absent).
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from battery_soh.config import get_paths
from battery_soh.data.nasa import NASA_BATTERY_IDS
from battery_soh.data.protocols import CORE_BATTERIES, EXTENDED_BATTERIES, PROTOCOLS
from battery_soh.evaluation.leakage import LeakageError, check_no_leakage
from battery_soh.evaluation.metrics import regression_metrics
from battery_soh.evaluation.splits import leave_one_group_out
from battery_soh.features.build import (
    FEATURE_TABLE_COLUMNS,
    build_feature_table,
    load_feature_table,
    metadata_path,
    write_feature_table,
)
from battery_soh.features.charge import CHARGE_FEATURES
from battery_soh.features.extract import FEATURE_EXTRACTOR_VERSION
from battery_soh.features.registry import FEATURES, get_feature_set
from battery_soh.labels import label_battery
from battery_soh.models.zoo import make_model
from battery_soh.qc import QC_REASONS

CSV = get_paths().features_csv
README = CSV.parent / "README.md"
#: Discharge operations per core cell in the official archive.
CORE_ROWS = {"B0005": 168, "B0006": 168, "B0007": 168, "B0018": 132}
TOTAL_ROWS = 2794
#: Relative tolerance of values stored with 6 significant digits (half a unit in the 6th digit,
#: twice for a ratio of two rounded numbers).
RTOL_6G = 1.1e-5


@pytest.fixture(scope="module")
def table() -> pd.DataFrame:
    return load_feature_table(CSV)


@pytest.fixture(scope="module")
def meta() -> dict[str, Any]:
    meta: dict[str, Any] = json.loads(metadata_path(CSV).read_text(encoding="utf-8"))
    return meta


@pytest.fixture(scope="module")
def core_valid(table: pd.DataFrame) -> pd.DataFrame:
    rows = table[table["qc_valid"] & table["battery_id"].isin(CORE_BATTERIES)]
    return rows.reset_index(drop=True)


# --------------------------------------------------------------------------- file & sidecar


def test_file_is_small_lf_utf8_and_matches_its_sidecar(meta: dict[str, Any]) -> None:
    data = CSV.read_bytes()
    assert len(data) < 3_000_000  # must stay small enough to live in git
    assert b"\r" not in data
    data.decode("utf-8")  # raises if not UTF-8
    assert meta["sha256"] == hashlib.sha256(data).hexdigest(), "rebuild with `soh data build`"
    assert meta["bytes"] == len(data)
    assert meta["file"] == CSV.name
    assert meta["columns"] == list(FEATURE_TABLE_COLUMNS)
    # A bumped extractor version means the definitions changed: the table must be rebuilt.
    assert meta["feature_extractor_version"] == FEATURE_EXTRACTOR_VERSION


def test_header_and_dtypes(table: pd.DataFrame) -> None:
    assert CSV.read_text(encoding="utf-8").split("\n", 1)[0] == ",".join(FEATURE_TABLE_COLUMNS)
    assert tuple(table.columns) == FEATURE_TABLE_COLUMNS
    assert table["cycle_index"].dtype == np.int64
    assert table["has_charge"].dtype == bool
    assert table["qc_valid"].dtype == bool
    assert str(table["start_time"].dtype) == "datetime64[ms]"
    assert all(table[name].dtype == np.float64 for name in FEATURES)


def test_sidecar_counts_and_nan_rates_match_the_table(
    table: pd.DataFrame, meta: dict[str, Any]
) -> None:
    assert meta["n_rows"] == len(table)
    assert meta["n_columns"] == table.shape[1]
    assert meta["row_counts"] == table["battery_id"].value_counts().sort_index().to_dict()
    valid = table[table["qc_valid"]]["battery_id"].value_counts()
    assert meta["qc_valid_counts"] == {b: int(valid.get(b, 0)) for b in meta["row_counts"]}
    for name in FEATURES:
        assert meta["nan_rate"][name] == pytest.approx(table[name].isna().mean(), abs=5e-5), name


# --------------------------------------------------------------------------- rows & keys


def test_every_battery_and_discharge_is_present(table: pd.DataFrame) -> None:
    counts = table["battery_id"].value_counts()
    assert tuple(sorted(counts.index)) == NASA_BATTERY_IDS
    assert len(table) == TOTAL_ROWS
    for battery_id, n in CORE_ROWS.items():
        assert counts[battery_id] == n, battery_id
    core = table[table["battery_id"].isin(CORE_BATTERIES)]
    assert core["qc_valid"].all()  # every core discharge passes QC


def test_rows_are_sorted_unique_and_contiguous(table: pd.DataFrame) -> None:
    keys = table[["battery_id", "cycle_index"]]
    assert not keys.duplicated().any()
    assert keys.equals(keys.sort_values(["battery_id", "cycle_index"], ignore_index=True))
    for battery_id, group in table.groupby("battery_id"):
        assert group["cycle_index"].tolist() == list(range(len(group))), battery_id
        assert group["start_time"].notna().all(), battery_id
        assert group["start_time"].is_monotonic_increasing, battery_id
    assert (table["ctx_cycle_index"] == table["cycle_index"]).all()


def test_metadata_columns(table: pd.DataFrame) -> None:
    groups = table.groupby("battery_id")["group"].unique()
    assert {b: list(g) for b, g in groups.items()} == {
        b: [PROTOCOLS[b].group] for b in groups.index
    }
    np.testing.assert_array_equal(table["ambient_c"], table["ctx_ambient_c"])
    np.testing.assert_array_equal(table["discharge_current_a"], table["ctx_discharge_current_a"])
    core = table[table["battery_id"].isin(CORE_BATTERIES)]
    assert (core["ambient_c"] == 24.0).all()
    np.testing.assert_allclose(core["discharge_current_a"], 2.0, atol=0.05)


# --------------------------------------------------------------------------- labels


def test_labels_follow_their_definitions(table: pd.DataFrame) -> None:
    assert table.groupby("battery_id")["c0_ah"].nunique().eq(1).all()
    for battery_id, group in table.groupby("battery_id"):
        first_valid = group.loc[group["qc_valid"], "capacity_ah"].iloc[:5]
        # The median of an odd count is one of the (identically rounded) values: exact.
        assert group["c0_ah"].iloc[0] == float(np.median(first_valid)), battery_id
    rows = table[table["capacity_ah"].notna()]
    np.testing.assert_allclose(
        rows["soh_pct"], 100.0 * rows["capacity_ah"] / rows["c0_ah"], rtol=RTOL_6G
    )
    np.testing.assert_allclose(rows["soh_rated_pct"], 50.0 * rows["capacity_ah"], rtol=RTOL_6G)
    assert table.loc[table["capacity_ah"].isna(), "soh_pct"].isna().all()
    np.testing.assert_array_equal(table["leg_capacity_ah"], table["capacity_ah"])


def test_qc_flags_are_reproducible_from_the_capacities(table: pd.DataFrame) -> None:
    assert set(table["qc_reason"]) <= set(QC_REASONS)
    assert (table["qc_valid"] == (table["qc_reason"] == "ok")).all()
    assert table.loc[table["qc_valid"], "soh_pct"].notna().all()
    for battery_id, group in table.groupby("battery_id"):
        labels = label_battery(group["capacity_ah"].reset_index(drop=True))
        assert labels["qc_reason"].tolist() == group["qc_reason"].tolist(), battery_id


def test_core_labels_start_near_100_and_fade(table: pd.DataFrame) -> None:
    for battery_id in CORE_BATTERIES:
        soh = table.loc[table["battery_id"] == battery_id, "soh_pct"]
        assert soh.iloc[0] == pytest.approx(100.0, abs=1.5), battery_id
        assert 55.0 < soh.min() < 80.0, battery_id
        assert soh.max() < 102.0, battery_id
    assert table.loc[table["battery_id"].isin(CORE_BATTERIES), "c0_ah"].between(1.83, 2.02).all()


# --------------------------------------------------------------------------- join rules


def test_charge_features_need_a_paired_charge(table: pd.DataFrame) -> None:
    unpaired = table.loc[~table["has_charge"], list(CHARGE_FEATURES)]
    assert len(unpaired) > 0
    assert unpaired.isna().all().all()
    core = table[table["battery_id"].isin(CORE_BATTERIES)]
    assert core["has_charge"].mean() > 0.99


def test_eis_is_an_as_of_join(table: pd.DataFrame) -> None:
    """Never back-filled (NaN only before the first measurement) and never reset afterwards."""
    for battery_id, group in table.groupby("battery_id"):
        observed = group["eis_re_ohm"].notna().to_numpy()
        assert (group["eis_rct_ohm"].notna().to_numpy() == observed).all(), battery_id
        first = int(np.argmax(observed)) if observed.any() else len(observed)
        assert not observed[:first].any(), battery_id
        assert observed[first:].all(), battery_id
    first_eis = table[table["eis_re_ohm"].notna()].groupby("battery_id")["cycle_index"].min()
    assert first_eis[["B0005", "B0006", "B0007"]].tolist() == [19, 19, 19]
    assert first_eis["B0018"] == 0  # B0018's first EIS precedes its first discharge


def test_usable_cells_stay_within_the_registry_ranges(table: pd.DataFrame) -> None:
    """QC-valid rows of usable cells hold only values the API would accept."""
    rows = table[table["qc_valid"] & table["battery_id"].isin(EXTENDED_BATTERIES)]
    for name, spec in FEATURES.items():
        values = rows[name].dropna()
        low, high = spec.plausible_range
        assert values.between(low, high).all(), name


# --------------------------------------------------------------------------- signal & leakage


def test_health_indicators_track_capacity_in_every_core_cell(core_valid: pd.DataFrame) -> None:
    for battery_id, group in core_valid.groupby("battery_id"):
        r = group[["dis_q_380_350_ah", "chg_cc_time_390_419_s", "chg_ica_peak_height_ah_per_v"]]
        r = r.corrwith(group["capacity_ah"])
        assert r["dis_q_380_350_ah"] > 0.99, battery_id
        assert r["chg_cc_time_390_419_s"] > 0.98, battery_id
        assert r["chg_ica_peak_height_ah_per_v"] > 0.97, battery_id


@pytest.mark.parametrize("feature_set", ["charge", "discharge", "combined", "combined_eis"])
def test_partial_window_sets_pass_the_leakage_check(
    core_valid: pd.DataFrame, feature_set: str
) -> None:
    report = check_no_leakage(core_valid, get_feature_set(feature_set))
    assert not report["flagged"].any()


def test_v1_inputs_are_flagged_as_leaky(core_valid: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="label-side/proxy"):
        get_feature_set("legacy_v1")
    with pytest.raises(LeakageError, match="leg_capacity_ah"):
        check_no_leakage(core_valid, get_feature_set("legacy_v1", allow_leaky=True))


def test_discharge_window_supports_the_headline_benchmark(core_valid: pd.DataFrame) -> None:
    """Coarse guard against scrambled rows: LOBO Ridge on the 3 discharge HIs beats cycle count."""
    y = core_valid["soh_pct"]

    def lobo_mae(features: list[str]) -> float:
        maes = []
        for train, test, _ in leave_one_group_out(core_valid["battery_id"]):
            model = make_model("ridge").fit(core_valid.loc[train, features], y[train])
            maes.append(
                regression_metrics(y[test], model.predict(core_valid.loc[test, features]))["mae"]
            )
        return float(np.mean(maes))

    discharge = lobo_mae(list(get_feature_set("discharge")))
    assert discharge < 2.0
    assert discharge < 0.5 * lobo_mae(list(get_feature_set("cycle_only")))


# --------------------------------------------------------------------------- documentation


def _readme_rows(header: str) -> list[list[str]]:
    """Cells of the Markdown table whose header row starts with ``header``."""
    lines = README.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(header))
    rows = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        cells = [c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", line)[1:-1]]
        rows.append(cells)
    return rows


def test_readme_documents_every_column_as_the_registry_does() -> None:
    rows = _readme_rows("| # | column |")
    assert [r[1].strip("`") for r in rows] == list(FEATURE_TABLE_COLUMNS)
    for _, name, unit, family, availability, risk, meaning in rows:
        spec = FEATURES.get(name.strip("`"))
        if spec is not None:
            assert (unit, family, availability, risk, meaning) == (
                spec.unit,
                spec.family,
                spec.availability,
                spec.leakage_risk,
                spec.description,
            ), spec.name


def test_readme_row_counts_match_the_sidecar(meta: dict[str, Any]) -> None:
    rows = {r[0]: r for r in _readme_rows("| battery | group |") if r[0].startswith("B")}
    counts, valid = meta["row_counts"], meta["qc_valid_counts"]
    assert list(rows) == list(counts)
    for battery_id, (_, group, _use, n_rows, n_valid) in rows.items():
        assert group.strip("`") == PROTOCOLS[battery_id].group
        assert (int(n_rows), int(n_valid)) == (counts[battery_id], valid[battery_id]), battery_id


# --------------------------------------------------------------------------- reproducibility


@pytest.mark.raw_data
def test_rebuilding_b0005_reproduces_the_committed_rows(nasa_raw_dir: Path, tmp_path: Path) -> None:
    rebuilt = write_feature_table(build_feature_table(nasa_raw_dir, ["B0005"]), tmp_path / "b.csv")
    lines = rebuilt.read_text(encoding="utf-8").splitlines()
    committed = CSV.read_text(encoding="utf-8").splitlines()
    assert lines[0] == committed[0]
    assert lines[1:] == [line for line in committed if line.startswith("B0005,")]


@pytest.mark.raw_data
@pytest.mark.slow
def test_full_rebuild_is_byte_identical(nasa_raw_dir: Path, tmp_path: Path) -> None:
    rebuilt = write_feature_table(build_feature_table(nasa_raw_dir), tmp_path / CSV.name)
    assert rebuilt.read_bytes() == CSV.read_bytes(), "committed CSV is stale: run `soh data build`"
