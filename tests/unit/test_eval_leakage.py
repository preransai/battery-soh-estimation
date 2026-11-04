"""Tests for target-leakage detection: v1-style leaks are flagged, honest health indicators pass."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from battery_soh.evaluation import errors, splits
from battery_soh.evaluation.leakage import (
    LABEL_SIDE_COLUMNS,
    SET_ROW,
    LeakageError,
    check_no_leakage,
    leakage_report,
    lobo_r2,
    reconstruction_r2,
)

N_BATTERIES = 5
N_CYCLES = 120


@pytest.fixture(scope="module")
def table() -> pd.DataFrame:
    """Synthetic fleet whose label is exactly ``100 * capacity / C0`` per battery (as in NASA)."""
    rng = np.random.default_rng(42)
    parts = []
    for b in range(N_BATTERIES):
        t = np.arange(N_CYCLES)
        c0 = rng.uniform(1.80, 2.05)
        fade = rng.uniform(0.20, 0.32)
        capacity = c0 * (1 - fade * (t / N_CYCLES) ** 1.2) + rng.normal(0, 0.004, N_CYCLES)
        # v1 record statistics: the whole-record mean current times the record
        # length is a coulomb count, i.e. the capacity in disguise.
        i_mean = rng.uniform(1.5, 2.5, N_CYCLES)
        t_rec = capacity * 3600 / i_mean
        # An honest partial-window health indicator: tracks capacity (r ~ 0.99
        # within a cell) with measurement noise and a cell-specific gain.
        gain = rng.uniform(0.40, 0.50)
        hi = gain * capacity + rng.normal(0, 0.008, N_CYCLES)
        r0 = 0.07 + 0.05 * (1 - capacity / c0) + rng.normal(0, 0.003, N_CYCLES)
        parts.append(
            pd.DataFrame(
                {
                    "battery_id": f"B{b:04d}",
                    "cycle_index": t,
                    "capacity_ah": capacity,
                    "c0_ah": c0,
                    "soh_pct": 100 * capacity / c0,
                    "leg_capacity_ah": capacity,
                    "cap_copy": capacity,
                    "coulomb_proxy": i_mean * t_rec / 3600 * (1 + rng.normal(0, 1e-4, N_CYCLES)),
                    "i_mean": i_mean,
                    "t_rec": t_rec,
                    "dis_q_380_350_ah": hi,
                    "dis_r0_ohm": r0,
                    "ctx_cycle_index": t.astype(float),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def _row(report: pd.DataFrame, feature: str) -> pd.Series:
    return report.set_index("feature").loc[feature]


def test_v1_capacity_input_is_flagged(table: pd.DataFrame) -> None:
    report = leakage_report(table, ["capacity_ah", "leg_capacity_ah", "dis_r0_ohm"])
    cap = _row(report, "capacity_ah")
    assert cap["flagged"]
    assert "label numerator" in cap["reasons"]
    assert cap["max_within_group_abs_r"] == pytest.approx(1.0)
    assert cap["capacity_reconstruction_r2"] == pytest.approx(1.0)
    leg = _row(report, "leg_capacity_ah")
    assert leg["registry_risk"] == "label"
    assert "registry: label-side feature" in leg["reasons"]
    assert _row(report, SET_ROW)["flagged"]
    with pytest.raises(LeakageError, match="capacity_ah"):
        check_no_leakage(table, ["capacity_ah", "dis_r0_ohm"])


def test_unregistered_exact_copy_is_flagged_by_the_data(table: pd.DataFrame) -> None:
    row = _row(leakage_report(table, ["cap_copy"]), "cap_copy")
    assert row["registry_risk"] == "unregistered"
    assert row["flagged"]
    assert "within-battery |r|" in row["reasons"]
    assert "rebuilds capacity_ah" in row["reasons"]


def test_coulomb_count_proxy_is_flagged(table: pd.DataFrame) -> None:
    row = _row(leakage_report(table, ["coulomb_proxy"]), "coulomb_proxy")
    assert row["max_within_group_abs_r"] >= 0.999
    assert row["capacity_reconstruction_r2"] >= 0.995
    assert row["flagged"]
    with pytest.raises(LeakageError, match="coulomb_proxy"):
        check_no_leakage(table, ["coulomb_proxy"])


def test_product_proxy_split_over_two_columns_is_flagged(table: pd.DataFrame) -> None:
    """current * time rebuilds capacity even when neither column alone does."""
    report = leakage_report(table, ["i_mean", "t_rec"])
    assert not _row(report, "i_mean")["flagged"]
    assert not _row(report, "t_rec")["flagged"]
    groups = table["battery_id"].to_numpy().astype(object)
    X = table[["i_mean", "t_rec"]].to_numpy()
    cap = table["capacity_ah"].to_numpy()
    assert lobo_r2(X, cap, groups) < 0.995  # a linear fit misses the product ...
    assert lobo_r2(X, cap, groups, log=True) > 0.9999  # ... a log-linear fit does not
    set_row = _row(report, SET_ROW)
    assert set_row["flagged"]
    assert "feature set rebuilds capacity_ah" in set_row["reasons"]
    with pytest.raises(LeakageError, match="feature set"):
        check_no_leakage(table, ["i_mean", "t_rec"])


def test_legitimate_health_indicator_passes(table: pd.DataFrame) -> None:
    features = ["dis_q_380_350_ah", "dis_r0_ohm", "ctx_cycle_index"]
    report = check_no_leakage(table, features)  # does not raise
    hi = _row(report, "dis_q_380_350_ah")
    # It is a strong health indicator: r ~ 0.99 with SOH inside every battery ...
    for _, part in table.groupby("battery_id"):
        assert abs(np.corrcoef(part["dis_q_380_350_ah"], part["soh_pct"])[0, 1]) > 0.98
    assert 0.98 < hi["max_within_group_abs_r"] < 0.999
    # ... yet it does not reconstruct capacity across batteries.
    assert hi["capacity_reconstruction_r2"] < 0.995
    assert not report["flagged"].any()
    assert list(report["feature"]) == [*features, SET_ROW]
    assert set(report.columns) == {
        "feature",
        "registry_risk",
        "availability",
        "n",
        "max_within_group_abs_r",
        "lobo_r2",
        "capacity_reconstruction_r2",
        "flagged",
        "reasons",
    }
    assert hi["registry_risk"] == "none"
    assert hi["availability"] == "partial_discharge"


@pytest.mark.parametrize("column", sorted(LABEL_SIDE_COLUMNS))
def test_label_side_table_columns_are_flagged(table: pd.DataFrame, column: str) -> None:
    label = "soh_rated_pct" if column == "soh_pct" else "soh_pct"
    df = table.assign(soh_rated_pct=table["soh_pct"] * table["c0_ah"] / 2.0)
    with pytest.raises(LeakageError, match=column):
        check_no_leakage(df, [column, "dis_r0_ohm"], label=label)


def test_registry_proxies_need_explicit_opt_in(table: pd.DataFrame) -> None:
    rng = np.random.default_rng(0)
    df = table.assign(leg_temp_max_c=rng.normal(32.0, 1.0, len(table)))
    with pytest.raises(LeakageError, match="label proxy"):
        check_no_leakage(df, ["leg_temp_max_c"])
    report = check_no_leakage(df, ["leg_temp_max_c"], allow_proxy=True)
    assert _row(report, "leg_temp_max_c")["registry_risk"] == "proxy"
    assert _row(report, SET_ROW)["registry_risk"] == "proxy"


def test_missing_capacity_column_skips_reconstruction(table: pd.DataFrame) -> None:
    report = leakage_report(table.drop(columns="capacity_ah"), ["dis_q_380_350_ah"])
    assert report["capacity_reconstruction_r2"].isna().all()
    assert not report["flagged"].any()


def test_nan_features_and_labels_are_tolerated(table: pd.DataFrame) -> None:
    df = table.copy()
    df.loc[::5, "dis_q_380_350_ah"] = np.nan
    df.loc[::17, "soh_pct"] = np.nan
    report = check_no_leakage(df, ["dis_q_380_350_ah"])
    assert _row(report, "dis_q_380_350_ah")["n"] == int(
        (df["soh_pct"].notna() & df["dis_q_380_350_ah"].notna()).sum()
    )


def test_input_validation(table: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="empty"):
        leakage_report(table, [])
    with pytest.raises(ValueError, match="not found"):
        leakage_report(table, ["nope"])
    with pytest.raises(ValueError, match="unique"):
        leakage_report(table, ["dis_r0_ohm", "dis_r0_ohm"])


def test_lobo_r2_edge_cases() -> None:
    groups = np.array(["a"] * 5 + ["b"] * 5 + ["c"] * 5, dtype=object)
    x = np.arange(15.0)
    assert lobo_r2(x, 2 * x + 1, groups) == pytest.approx(1.0)
    assert math.isnan(lobo_r2(x, np.ones(15), groups))  # constant target
    assert math.isnan(lobo_r2(x, x, np.full(15, "a", dtype=object)))  # one group
    assert math.isnan(lobo_r2(x, x - 5, groups, log=True))  # non-positive target in log space
    assert reconstruction_r2(x, x - 5, groups) == pytest.approx(1.0)  # falls back to linear
    y = x.copy()
    y[[0, 7]] = np.nan  # rows with unknown target are dropped
    assert lobo_r2(x, y, groups) == pytest.approx(1.0)


def test_leakage_error_is_shared_and_not_a_value_error() -> None:
    assert LeakageError is errors.LeakageError is splits.LeakageError
    assert not issubclass(LeakageError, ValueError)
