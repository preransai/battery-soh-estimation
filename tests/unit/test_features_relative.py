"""Commissioning-referenced features: relativize (tables) and apply_reference (serving)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from battery_soh.features.relative import (
    REFERENCE_FLAG_COLUMN,
    apply_reference,
    is_context_feature,
    reference_table,
    relativize,
)

NAN = float("nan")


@pytest.fixture
def table() -> pd.DataFrame:
    """Two cells; B1 has an invalid cycle and a NaN early on, B2 is clean."""
    return pd.DataFrame(
        {
            "battery_id": ["B1"] * 6 + ["B2"] * 4,
            "cycle_index": [0, 1, 2, 3, 4, 5, 0, 1, 2, 3],
            "qc_valid": [True, False, True, True, True, True, True, True, True, True],
            "dis_q_380_350_ah": [1.0, 9.0, NAN, 3.0, 4.0, 2.0, 2.0, 2.0, 4.0, 1.0],
            "chg_cv_time_s": [10.0, 99.0, 20.0, 30.0, 40.0, 50.0, 5.0, 5.0, 5.0, 5.0],
            "ctx_cycle_index": [0.0, 1, 2, 3, 4, 5, 0, 1, 2, 3],
        }
    )


FEATURES = ["dis_q_380_350_ah", "chg_cv_time_s", "ctx_cycle_index"]


def test_reference_is_the_mean_of_the_first_valid_non_nan_values(table: pd.DataFrame) -> None:
    ref = reference_table(table, FEATURES, n_ref=2)
    assert list(ref.columns) == ["dis_q_380_350_ah", "chg_cv_time_s"]  # context skipped
    # B1 dis_q: cycle 1 invalid, cycle 2 NaN -> cycles 0 and 3.
    assert ref.loc["B1", "dis_q_380_350_ah"] == pytest.approx((1.0 + 3.0) / 2)
    # B1 cv time: cycle 1 invalid -> cycles 0 and 2 (selection is per feature).
    assert ref.loc["B1", "chg_cv_time_s"] == pytest.approx((10.0 + 20.0) / 2)
    assert ref.loc["B2", "dis_q_380_350_ah"] == pytest.approx(2.0)


def test_relativize_divides_and_flags_reference_rows(table: pd.DataFrame) -> None:
    out = relativize(table, FEATURES, n_ref=2)
    np.testing.assert_allclose(
        out["dis_q_380_350_ah"], [0.5, 4.5, NAN, 1.5, 2.0, 1.0, 1.0, 1.0, 2.0, 0.5]
    )
    np.testing.assert_allclose(
        out["chg_cv_time_s"][:6], [10 / 15, 99 / 15, 20 / 15, 2, 40 / 15, 50 / 15]
    )
    # Reference rows: union over features (B1 cycles 0, 2, 3; B2 cycles 0, 1).
    assert out[REFERENCE_FLAG_COLUMN].tolist() == [
        True,
        False,
        True,
        True,
        False,
        False,
        True,
        True,
        False,
        False,
    ]
    assert out[REFERENCE_FLAG_COLUMN].dtype == bool


def test_context_features_and_other_columns_are_untouched(table: pd.DataFrame) -> None:
    original = table.copy()
    out = relativize(table, FEATURES)
    pd.testing.assert_frame_equal(table, original)  # input not mutated
    pd.testing.assert_series_equal(out["ctx_cycle_index"], table["ctx_cycle_index"])
    pd.testing.assert_series_equal(out["qc_valid"], table["qc_valid"])
    assert list(out.columns) == [*table.columns, REFERENCE_FLAG_COLUMN]
    assert is_context_feature("ctx_ambient_c")
    assert is_context_feature("ctx_unknown_extra")
    assert not is_context_feature("dis_r0_ohm")


def test_reference_values_average_to_one(table: pd.DataFrame) -> None:
    out = relativize(table, ["chg_cv_time_s"], n_ref=3)
    ref_rows = out[out[REFERENCE_FLAG_COLUMN]]
    assert ref_rows.groupby("battery_id")["chg_cv_time_s"].mean().tolist() == pytest.approx(
        [1.0, 1.0]
    )


def test_order_follows_cycle_index_not_row_order(table: pd.DataFrame) -> None:
    shuffled = table.sample(frac=1.0, random_state=0)
    out = relativize(shuffled, FEATURES, n_ref=2)
    assert out.index.tolist() == shuffled.index.tolist()  # row order preserved
    expected = relativize(table, FEATURES, n_ref=2)
    pd.testing.assert_frame_equal(out.sort_index(), expected)


def test_fewer_valid_values_than_n_ref_and_none_at_all(table: pd.DataFrame) -> None:
    ref = reference_table(table, FEATURES, n_ref=50)
    assert ref.loc["B2", "dis_q_380_350_ah"] == pytest.approx(9.0 / 4)  # all four values
    table.loc[table.battery_id == "B2", "qc_valid"] = False
    out = relativize(table, FEATURES)
    assert out.loc[table.battery_id == "B2", "dis_q_380_350_ah"].isna().all()
    assert not out.loc[table.battery_id == "B2", REFERENCE_FLAG_COLUMN].any()


def test_zero_reference_gives_nan_not_inf() -> None:
    df = pd.DataFrame(
        {
            "battery_id": ["A", "A"],
            "cycle_index": [0, 1],
            "qc_valid": [True, True],
            "chg_temp_rise_cc_c": [0.0, 1.0],
        }
    )
    out = relativize(df, ["chg_temp_rise_cc_c"], n_ref=1)
    assert out["chg_temp_rise_cc_c"].isna().all()


def test_optional_validity_and_order_columns(table: pd.DataFrame) -> None:
    bare = table.drop(columns=["qc_valid", "cycle_index"])
    out = relativize(bare, ["dis_q_380_350_ah"], n_ref=2, valid_col=None, order_col=None)
    # Without a QC column the (invalid) 9.0 Ah row enters B1's reference: (1 + 9) / 2.
    assert out.loc[0, "dis_q_380_350_ah"] == pytest.approx(1.0 / 5.0)
    assert out.loc[1, REFERENCE_FLAG_COLUMN]
    with pytest.raises(ValueError, match="missing"):
        relativize(bare, ["dis_q_380_350_ah"])


@pytest.mark.parametrize("n_ref", [0, -1, True, 2.5, "5"])
def test_invalid_n_ref_raises(table: pd.DataFrame, n_ref: object) -> None:
    with pytest.raises(ValueError, match="n_ref"):
        relativize(table, FEATURES, n_ref=n_ref)  # type: ignore[arg-type]


def test_invalid_columns_raise(table: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="missing"):
        relativize(table, ["not_a_column"])
    with pytest.raises(ValueError, match="sequence"):
        relativize(table, "dis_q_380_350_ah")
    with pytest.raises(ValueError, match="duplicate"):
        relativize(table, ["chg_cv_time_s", "chg_cv_time_s"])
    with pytest.raises(ValueError, match="numeric"):
        relativize(table.assign(chg_cv_time_s="x"), FEATURES)
    with pytest.raises(ValueError, match="missing"):
        relativize(table, FEATURES, group_col="cell")


# ----------------------------------------------------------------------------- serving


def test_apply_reference_matches_relativize_row_by_row(table: pd.DataFrame) -> None:
    """Train/serve parity: the API path gives exactly the training-table values."""
    out = relativize(table, FEATURES, n_ref=2)
    ref = reference_table(table, FEATURES, n_ref=2)
    for idx, row in table.iterrows():
        served = apply_reference(
            {f: float(row[f]) for f in FEATURES}, ref.loc[row["battery_id"]].to_dict()
        )
        for f in FEATURES:
            np.testing.assert_equal(served[f], out.loc[idx, f])


def test_apply_reference_edge_cases() -> None:
    served = apply_reference(
        {
            "dis_q_380_350_ah": 0.9,
            "chg_cv_time_s": 600.0,
            "dis_r0_ohm": 0.1,
            "ctx_ambient_c": 24.0,
            "eis_re_ohm": 0.05,
        },
        {"dis_q_380_350_ah": 1.0, "chg_cv_time_s": 0.0, "dis_r0_ohm": NAN, "ctx_ambient_c": 2.0},
    )
    assert served["dis_q_380_350_ah"] == pytest.approx(0.9)
    assert math.isnan(served["chg_cv_time_s"])  # zero reference
    assert math.isnan(served["dis_r0_ohm"])  # NaN reference
    assert math.isnan(served["eis_re_ohm"])  # missing reference
    assert served["ctx_ambient_c"] == 24.0  # context passes through
    with pytest.raises(ValueError, match="real numbers"):
        apply_reference({"dis_q_380_350_ah": "a"}, {"dis_q_380_350_ah": 1.0})  # type: ignore[dict-item]


def test_nullable_and_infinite_values_never_enter_a_reference() -> None:
    df = pd.DataFrame(
        {
            "battery_id": ["A"] * 4,
            "cycle_index": [0, 1, 2, 3],
            "qc_valid": [True] * 4,
            "dis_r0_ohm": pd.array([np.inf, None, 0.1, 0.2], dtype="Float64"),
        }
    )
    out = relativize(df, ["dis_r0_ohm"], n_ref=1)
    np.testing.assert_allclose(out["dis_r0_ohm"], [NAN, NAN, 1.0, 2.0])
    assert out[REFERENCE_FLAG_COLUMN].tolist() == [False, False, True, False]
