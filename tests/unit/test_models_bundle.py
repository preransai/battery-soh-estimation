"""Tests for the JSON model bundle: sklearn parity, attribution, intervals, domain, manifest."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from battery_soh.models.linear_bundle import (
    MANIFEST_NAME,
    BundleError,
    LinearBundle,
    ManifestError,
    export_linear_bundle,
    save_bundle,
    sha256_file,
    verify_manifest,
    write_manifest,
    write_text_lf,
)
from battery_soh.models.zoo import make_model

FEATURES = ["dis_q_380_350_ah", "dis_r0_ohm", "chg_cv_time_s"]
QUANTILES = {"0.8": 1.5, "0.9": 2.0, "0.95": 2.6}
CREATED = "2026-01-01T00:00:00+00:00"


def _training_frame(seed: int = 0, n_per: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for b in range(4):
        soh = np.linspace(100.0, 75.0 + 3 * b, n_per) + rng.normal(0, 0.3, n_per)
        q = 0.0045 * soh + rng.normal(0, 0.004, n_per)  # strongly correlated with SOH
        r0 = 0.12 - 0.0004 * soh + rng.normal(0, 0.002, n_per)
        cv = 1500 + 10 * (100 - soh) + rng.normal(0, 30, n_per)
        parts.append(
            pd.DataFrame(
                {
                    "battery_id": f"B{b:04d}",
                    "cycle_index": np.arange(n_per),
                    "soh_pct": soh,
                    "dis_q_380_350_ah": q,
                    "dis_r0_ohm": r0,
                    "chg_cv_time_s": cv,
                }
            )
        )
    df = pd.concat(parts, ignore_index=True)
    df.loc[[3, 50, 97], "chg_cv_time_s"] = np.nan  # some missing charge features
    return df


def _export(pipe: Any, df: pd.DataFrame, **kwargs: Any) -> dict[str, Any]:
    params: dict[str, Any] = {
        "model_id": "combined-absolute-ridge",
        "feature_set": "combined",
        "mode": "absolute",
        "features": FEATURES,
        "training_df": df,
        "conformal_quantiles": QUANTILES,
        "metrics": {"mean": {"mae": 1.23, "r2": float("nan")}},
        "n_conformal_scores": len(df),
        "created_utc": CREATED,
    }
    params.update(kwargs)
    return export_linear_bundle(pipe, **params)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return _training_frame()


@pytest.fixture(scope="module")
def pipeline(frame: pd.DataFrame) -> Any:
    return make_model("ridge", alpha=1.0).fit(frame[FEATURES], frame["soh_pct"])


@pytest.fixture(scope="module")
def bundle_path(
    tmp_path_factory: pytest.TempPathFactory, frame: pd.DataFrame, pipeline: Any
) -> Path:
    path = tmp_path_factory.mktemp("models") / "combined-absolute-ridge.json"
    return save_bundle(_export(pipeline, frame), path)


@pytest.fixture(scope="module")
def bundle(bundle_path: Path) -> LinearBundle:
    return LinearBundle.from_json(bundle_path)


def _new_inputs(seed: int = 9, n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "dis_q_380_350_ah": rng.uniform(0.3, 0.5, n),
            "dis_r0_ohm": rng.uniform(0.07, 0.09, n),
            "chg_cv_time_s": rng.uniform(1400, 1900, n),
        }
    )
    X.iloc[::7, 2] = np.nan
    X.iloc[::11, 0] = np.nan
    return X


# --------------------------------------------------------------------- round trip & parity
def test_json_round_trip_matches_sklearn(
    bundle: LinearBundle, pipeline: Any, frame: pd.DataFrame
) -> None:
    for X in (frame[FEATURES], _new_inputs()):
        expected = pipeline.predict(X)
        np.testing.assert_allclose(bundle.predict_raw_array(X.to_numpy()), expected, rtol=1e-9)
        for i in range(0, len(X), 5):
            row = {k: (None if math.isnan(v) else float(v)) for k, v in X.iloc[i].items()}
            assert bundle.predict(row) == pytest.approx(expected[i], rel=1e-9)


def test_document_follows_schema(bundle_path: Path, frame: pd.DataFrame) -> None:
    doc = json.loads(bundle_path.read_text(encoding="utf-8"))  # strict JSON, no NaN
    for key in (
        "schema_version",
        "model_id",
        "created_utc",
        "package_version",
        "feature_set",
        "mode",
        "n_reference_cycles",
        "features",
        "label",
        "preprocessing",
        "estimator",
        "output_clip",
        "conformal",
        "domain",
        "training",
        "metrics",
        "explanation",
    ):
        assert key in doc, key
    assert doc["estimator"]["type"] == "ridge"
    assert doc["estimator"]["alpha"] == 1.0
    assert doc["conformal"]["quantiles"] == QUANTILES
    assert doc["metrics"]["mean"]["r2"] is None  # NaN -> null
    assert doc["training"]["batteries"] == ["B0000", "B0001", "B0002", "B0003"]
    assert doc["training"]["n_cycles"] == len(frame)
    assert len(doc["training"]["features_sha256"]) == 64
    assert doc["explanation"]["base_value"] == doc["estimator"]["intercept"]
    assert "NaN" not in bundle_path.read_text(encoding="utf-8")


def test_export_is_deterministic(frame: pd.DataFrame, pipeline: Any) -> None:
    a = json.dumps(_export(pipeline, frame), sort_keys=True)
    b = json.dumps(_export(pipeline, frame), sort_keys=True)
    assert a == b


# That the runtime (and the app) never imports scikit-learn is checked in a fresh interpreter by
# tests/api/test_api_imports.py, one subprocess for both.


@pytest.mark.parametrize("name", ["pls", "elasticnet"])
def test_other_linear_zoo_models_export_exactly(name: str, frame: pd.DataFrame) -> None:
    pipe = make_model(name).fit(frame[FEATURES], frame["soh_pct"])
    doc = _export(pipe, frame, model_id=f"combined-absolute-{name}")
    runtime = LinearBundle.from_dict(json.loads(json.dumps(doc)))
    X = _new_inputs()
    np.testing.assert_allclose(runtime.predict_raw_array(X.to_numpy()), pipe.predict(X), rtol=1e-9)
    assert doc["estimator"]["type"] == name


# --------------------------------------------------------------------- attribution
def test_contributions_are_exact_and_additive(bundle: LinearBundle, pipeline: Any) -> None:
    x = {"dis_q_380_350_ah": 0.41, "dis_r0_ohm": 0.083, "chg_cv_time_s": 1650.0}
    contributions = bundle.contributions(x)
    assert [c[0] for c in contributions] == FEATURES
    total = bundle.base_value + sum(c for _, _, c in contributions)
    assert total == pytest.approx(bundle.predict(x), rel=1e-12)
    # Each contribution is coef_j * z_j with sklearn's own standardisation.
    row = pd.DataFrame([x])[FEATURES]
    z = pipeline.named_steps["scale"].transform(pipeline.named_steps["impute"].transform(row))[0]
    coef = pipeline.named_steps["model"].coef_
    np.testing.assert_allclose([c for _, _, c in contributions], coef * z, rtol=1e-9, atol=1e-12)
    assert bundle.base_value == pytest.approx(pipeline.named_steps["model"].intercept_)


def test_missing_feature_is_median_imputed_and_reported(
    bundle: LinearBundle, pipeline: Any
) -> None:
    x = {"dis_q_380_350_ah": 0.41, "dis_r0_ohm": 0.083}  # chg_cv_time_s missing
    median = pipeline.named_steps["impute"].statistics_[2]
    contributions = {name: value for name, value, _ in bundle.contributions(x)}
    assert contributions["chg_cv_time_s"] == pytest.approx(median)
    assert bundle.missing_features(x) == ("chg_cv_time_s",)
    assert bundle.missing_features({**x, "chg_cv_time_s": float("nan")}) == ("chg_cv_time_s",)
    assert bundle.domain_check(x).missing == ("chg_cv_time_s",)
    expected = pipeline.predict(pd.DataFrame([{**x, "chg_cv_time_s": np.nan}])[FEATURES])[0]
    assert bundle.predict(x) == pytest.approx(expected, rel=1e-9)
    # Extra keys are ignored.
    assert bundle.predict({**x, "unrelated": 5.0}) == bundle.predict(x)


@pytest.mark.parametrize("bad", ["0.4", True, float("inf"), [0.4]])
def test_invalid_feature_values_raise(bundle: LinearBundle, bad: Any) -> None:
    with pytest.raises(ValueError, match="dis_q_380_350_ah"):
        bundle.predict({"dis_q_380_350_ah": bad})


def test_output_clip_is_reported_by_explain(bundle: LinearBundle) -> None:
    far = {"dis_q_380_350_ah": 5.0, "dis_r0_ohm": 0.0, "chg_cv_time_s": 0.0}
    explanation = bundle.explain(far)
    assert explanation.raw_prediction > bundle.output_clip[1]
    assert explanation.clipped
    assert explanation.prediction == bundle.output_clip[1]
    assert bundle.base_value + sum(c for _, _, c in explanation.contributions) == pytest.approx(
        explanation.raw_prediction, rel=1e-12
    )


# --------------------------------------------------------------------- intervals
def test_interval_uses_stored_half_widths(bundle: LinearBundle) -> None:
    for key, q in QUANTILES.items():
        assert bundle.interval(85.0, float(key)) == pytest.approx((85.0 - q, 85.0 + q))
        assert bundle.half_width(float(key)) == q
    assert bundle.confidences == (0.8, 0.9, 0.95)
    lo, hi = bundle.interval(119.5, 0.95)  # clipped at the physical bound
    assert (lo, hi) == pytest.approx((119.5 - 2.6, 120.0))
    with pytest.raises(ValueError, match="no conformal quantile"):
        bundle.interval(85.0, 0.5)
    with pytest.raises(ValueError, match="finite"):
        bundle.interval(float("nan"), 0.9)


def test_infinite_quantile_round_trips_and_is_explicit(
    tmp_path: Path, frame: pd.DataFrame, pipeline: Any
) -> None:
    doc = _export(pipeline, frame, conformal_quantiles={0.8: 1.0, 0.9: 2.0, 0.95: math.inf})
    assert doc["conformal"]["quantiles"]["0.95"] is None
    loaded = LinearBundle.from_json(save_bundle(doc, tmp_path / "b.json"))
    assert loaded.half_width(0.95) == math.inf
    assert loaded.interval(90.0, 0.95) == loaded.output_clip
    assert loaded.interval(90.0, 0.9) == (88.0, 92.0)


# --------------------------------------------------------------------- domain
def test_training_rows_are_in_domain(bundle: LinearBundle, frame: pd.DataFrame) -> None:
    flags = [
        bundle.domain_check(
            {k: (None if pd.isna(v) else float(v)) for k, v in row.items()}
        ).in_domain
        for row in frame[FEATURES].to_dict(orient="records")
    ]
    assert np.mean(flags) >= 0.97  # threshold = 99th percentile of training distances
    distances = bundle.mahalanobis_array(frame[FEATURES].to_numpy())
    assert np.quantile(distances, 0.99, method="higher") == pytest.approx(
        bundle.mahalanobis_threshold
    )


def test_out_of_range_feature_is_flagged(bundle: LinearBundle, frame: pd.DataFrame) -> None:
    hi = float(frame["dis_r0_ohm"].max())
    span = hi - float(frame["dis_r0_ohm"].min())
    x = {"dis_q_380_350_ah": 0.4, "dis_r0_ohm": hi + 0.5 * span, "chg_cv_time_s": 1600.0}
    check = bundle.domain_check(x)
    assert check.out_of_range == ("dis_r0_ohm",)
    assert not check.in_domain
    # Within the 10 % margin: not out of range.
    assert bundle.domain_check({**x, "dis_r0_ohm": hi + 0.05 * span}).out_of_range == ()
    payload = check.to_dict()
    assert payload["out_of_range"] == ["dis_r0_ohm"]
    assert isinstance(payload["mahalanobis"], float)


def test_mahalanobis_flags_broken_correlation(bundle: LinearBundle, frame: pd.DataFrame) -> None:
    # Every feature inside its own training range, but the joint pattern (healthy
    # charge window, end-of-life discharge window) never occurs in training.
    q_lo = float(frame["dis_q_380_350_ah"].quantile(0.02))
    r_lo = float(frame["dis_r0_ohm"].quantile(0.02))
    cv_hi = float(frame["chg_cv_time_s"].quantile(0.02))
    x = {"dis_q_380_350_ah": q_lo, "dis_r0_ohm": r_lo, "chg_cv_time_s": cv_hi}
    check = bundle.domain_check(x)
    assert check.out_of_range == ()
    assert check.mahalanobis > check.threshold
    assert not check.in_domain


# --------------------------------------------------------------------- export validation
def test_export_rejects_column_permutation(frame: pd.DataFrame, pipeline: Any) -> None:
    with pytest.raises(BundleError, match="fitted on columns"):
        _export(pipeline, frame, features=list(reversed(FEATURES)))


def test_export_rejects_non_linear_pipeline(frame: pd.DataFrame) -> None:
    pipe = make_model("knn").fit(frame[FEATURES], frame["soh_pct"])
    with pytest.raises(BundleError):
        _export(pipe, frame)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"mode": "both"}, "mode"),
        ({"conformal_quantiles": {0.9: float("nan")}}, "quantile"),
        ({"conformal_quantiles": {1.5: 1.0}}, "confidence"),
        ({"mahalanobis_quantile": 0.0}, "mahalanobis_quantile"),
        ({"extra": {"metrics": 1}}, "overwrite"),
        ({"features": [*FEATURES[:2], FEATURES[0]]}, "unique"),
    ],
)
def test_export_validates_arguments(
    frame: pd.DataFrame, pipeline: Any, kwargs: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _export(pipeline, frame, **kwargs)


def test_numpy_fitted_pipeline_exports(frame: pd.DataFrame) -> None:
    pipe = make_model("ridge").fit(frame[FEATURES].to_numpy(), frame["soh_pct"].to_numpy())
    doc = _export(pipe, frame)
    assert LinearBundle.from_dict(doc).predict({"dis_q_380_350_ah": 0.4}) == pytest.approx(
        pipe.predict(np.array([[0.4, np.nan, np.nan]]))[0]
    )


# --------------------------------------------------------------------- load validation
def _mutated(doc: dict[str, Any], path: tuple[str, ...], value: Any) -> dict[str, Any]:
    out = json.loads(json.dumps(doc))
    target = out
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return out


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("schema_version",), "2.0"),
        (("mode",), "sideways"),
        (("features",), []),
        (("preprocessing", "scaler_scale"), [1.0, 0.0, 1.0]),
        (("preprocessing", "impute_median"), [1.0, 2.0]),
        (("estimator", "coef"), [1.0, "x", 2.0]),
        (("estimator", "intercept"), None),
        (("conformal", "quantiles"), {"0.9": -1.0}),
        (("domain", "precision_matrix"), [[1.0]]),
        (("domain", "mahalanobis_threshold"), None),
        (("output_clip",), [120.0, 0.0]),
        (("n_reference_cycles",), 0),
        # Wrong JSON types must surface as BundleError, not TypeError/ValueError, so that
        # the registry skips the file instead of crashing at startup.
        (("domain", "margin"), None),
        (("domain", "margin"), "abc"),
        (("domain", "margin"), [0.1]),
        (("domain", "margin"), True),
        (("domain", "precision_matrix"), [[1.0], [1.0, 2.0]]),
        (("domain", "precision_matrix"), "abc"),
        (("domain", "precision_matrix"), {"a": 1}),
        (("domain", "mahalanobis_threshold"), True),
        (("estimator", "intercept"), False),
        (("conformal", "quantiles"), {"0.9": True}),
        (("metrics",), [1, 2]),
        (("metrics",), "x"),
    ],
)
def test_corrupt_bundles_fail_at_load(
    bundle: LinearBundle, path: tuple[str, ...], value: Any
) -> None:
    with pytest.raises(BundleError):
        LinearBundle.from_dict(_mutated(bundle.to_dict(), path, value))


@pytest.mark.parametrize("doc", [None, [1, 2], "bundle"])
def test_non_object_bundle_fails_at_load(doc: Any) -> None:
    with pytest.raises(BundleError, match="JSON object"):
        LinearBundle.from_dict(doc)


def test_invalid_json_file(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(BundleError, match="not valid JSON"):
        LinearBundle.from_json(bad)
    bad.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(BundleError, match="JSON object"):
        LinearBundle.from_json(bad)


def test_card_omits_precision_matrix(bundle: LinearBundle) -> None:
    card = bundle.to_card()
    assert "precision_matrix" not in card["domain"]
    assert "precision_matrix" in bundle.to_dict()["domain"]


# --------------------------------------------------------------------- manifest
def test_manifest_round_trip_and_tamper_detection(tmp_path: Path, bundle: LinearBundle) -> None:
    for name in ("a-absolute-ridge.json", "b-relative-ridge.json"):
        save_bundle(bundle.to_dict(), tmp_path / name)
    manifest = write_manifest(tmp_path, created_utc=CREATED)
    content = json.loads(manifest.read_text(encoding="utf-8"))
    assert manifest.name == MANIFEST_NAME
    assert content["created_utc"] == CREATED
    assert sorted(content["files"]) == ["a-absolute-ridge.json", "b-relative-ridge.json"]
    raw = (tmp_path / "a-absolute-ridge.json").read_bytes()
    assert content["files"]["a-absolute-ridge.json"] == hashlib.sha256(raw).hexdigest()
    assert sha256_file(tmp_path / "a-absolute-ridge.json") == hashlib.sha256(raw).hexdigest()
    assert verify_manifest(tmp_path) == content["files"]

    # One flipped byte is detected.
    (tmp_path / "a-absolute-ridge.json").write_bytes(raw.replace(b'"ridge"', b'"ridgf"', 1))
    with pytest.raises(ManifestError, match="sha256 mismatch"):
        verify_manifest(tmp_path)
    (tmp_path / "a-absolute-ridge.json").write_bytes(raw)
    verify_manifest(tmp_path)

    # A missing file and an unlisted extra model are detected.
    (tmp_path / "b-relative-ridge.json").rename(tmp_path / "c-extra.json")
    with pytest.raises(ManifestError, match="missing"):
        verify_manifest(tmp_path)
    (tmp_path / "c-extra.json").rename(tmp_path / "b-relative-ridge.json")
    save_bundle(bundle.to_dict(), tmp_path / "d-unlisted.json")
    verify_manifest(tmp_path)  # tolerated by default
    with pytest.raises(ManifestError, match="not covered"):
        verify_manifest(tmp_path, allow_unlisted=False)


def _git_eol_lf_checkout(directory: Path) -> None:
    """What git does to text files on checkout with ``* text=auto eol=lf``: CRLF -> LF."""
    for path in directory.rglob("*"):
        if path.is_file():
            path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))


def test_bundle_and_manifest_are_lf_and_survive_a_git_checkout(
    tmp_path: Path, bundle: LinearBundle
) -> None:
    # Regression: Path.write_text writes CRLF on Windows, so a manifest written there recorded
    # digests that stopped matching once git normalised the files to LF on a Linux checkout.
    doc = bundle.to_dict()
    path = save_bundle(doc, tmp_path / "a-absolute-ridge.json")
    expected = (json.dumps(doc, indent=2, allow_nan=False) + "\n").encode("utf-8")
    assert path.read_bytes() == expected  # byte for byte, on every OS
    (tmp_path / "card.md").write_bytes(b"# card\n\ntext\n")
    manifest = write_manifest(tmp_path, ["a-absolute-ridge.json", "card.md"], created_utc=CREATED)
    assert b"\r" not in manifest.read_bytes()
    recorded = json.loads(manifest.read_bytes())["files"]
    assert recorded["a-absolute-ridge.json"] == hashlib.sha256(expected).hexdigest()

    _git_eol_lf_checkout(tmp_path)
    assert verify_manifest(tmp_path, allow_unlisted=False) == recorded


def test_crlf_artifacts_are_flagged_when_hashed_and_explained_on_mismatch(
    tmp_path: Path, bundle: LinearBundle, caplog: pytest.LogCaptureFixture
) -> None:
    text = json.dumps(bundle.to_dict(), indent=2) + "\n"
    path = tmp_path / "a-absolute-ridge.json"
    path.write_bytes(text.replace("\n", "\r\n").encode("utf-8"))  # Path.write_text on Windows
    with caplog.at_level("WARNING", logger="battery_soh.models.linear_bundle"):
        write_manifest(tmp_path, created_utc=CREATED)
    assert "CRLF" in caplog.text
    _git_eol_lf_checkout(tmp_path)  # the digest of the CRLF bytes no longer matches
    with pytest.raises(ManifestError, match="sha256 mismatch"):
        verify_manifest(tmp_path)

    # The opposite failure (an LF artifact converted to CRLF, e.g. core.autocrlf) is explained.
    save_bundle(bundle.to_dict(), path)
    write_manifest(tmp_path, created_utc=CREATED)
    path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    with pytest.raises(ManifestError, match="CRLF line endings"):
        verify_manifest(tmp_path)


def test_write_text_lf_normalises_and_hashes_the_written_bytes(tmp_path: Path) -> None:
    target = tmp_path / "sub" / "x.txt"
    digest = write_text_lf(target, "a\r\nb\rc\n")
    assert target.read_bytes() == b"a\nb\nc\n"
    assert digest == hashlib.sha256(b"a\nb\nc\n").hexdigest() == sha256_file(target)


@pytest.mark.parametrize(
    "name", ["../escape.json", "a/../../b.json", "/abs.json", "sub\\x.json", "C:evil.json", ""]
)
def test_manifest_rejects_path_traversal(tmp_path: Path, name: str) -> None:
    (tmp_path / MANIFEST_NAME).write_text(
        json.dumps({"files": {name: "0" * 64}, "created_utc": CREATED}), encoding="utf-8"
    )
    with pytest.raises(ManifestError, match="illegal"):
        verify_manifest(tmp_path)


def test_manifest_errors(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="no MANIFEST"):
        verify_manifest(tmp_path)
    (tmp_path / MANIFEST_NAME).write_text("{", encoding="utf-8")
    with pytest.raises(ManifestError, match="not valid JSON"):
        verify_manifest(tmp_path)
    (tmp_path / MANIFEST_NAME).write_text(json.dumps({"files": []}), encoding="utf-8")
    with pytest.raises(ManifestError, match="files"):
        verify_manifest(tmp_path)
    with pytest.raises(ManifestError, match="missing"):
        write_manifest(tmp_path, ["nope.json"])
    outside = tmp_path.parent / "outside.json"
    with pytest.raises(ManifestError, match="not inside"):
        write_manifest(tmp_path, [outside])
