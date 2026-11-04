"""Tests for the model zoo: every model fits, predicts, is seeded and has the documented shape."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor

from battery_soh.models.zoo import (
    LINEAR_MODEL_NAMES,
    MODEL_NAMES,
    MODEL_SPECS,
    PLSRegressor,
    _pls_x_mean,
    final_estimator,
    make_model,
)


def _problem(seed: int = 0, n: int = 80) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3)) * [1.0, 50.0, 0.01] + [0.0, 1500.0, 0.08]  # mixed scales
    y = 90 + 4 * X[:, 0] - 0.05 * (X[:, 1] - 1500) + rng.normal(0, 0.5, n)
    X[::9, 2] = np.nan  # missing values must be handled by every model
    return X, y


def test_model_names() -> None:
    assert MODEL_NAMES == (
        "ridge",
        "pls",
        "elasticnet",
        "gpr",
        "random_forest",
        "hgb",
        "mlp",
        "knn",
    )
    assert LINEAR_MODEL_NAMES == ("ridge", "pls", "elasticnet")


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_model_fits_and_predicts(name: str) -> None:
    X, y = _problem()
    model = make_model(name).fit(X, y)
    pred = model.predict(X)
    assert pred.shape == (len(y),)
    assert np.all(np.isfinite(pred))
    # A clearly learnable signal: every model must beat the mean in-sample.
    assert r2_score(y, pred) > 0.8
    X_new, _ = _problem(seed=1, n=10)
    assert model.predict(X_new).shape == (10,)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_pipeline_shape(name: str) -> None:
    steps = [step for step, _ in make_model(name).steps]
    expected = ["impute", "model"] if not MODEL_SPECS[name].scaled else ["impute", "scale", "model"]
    assert steps == expected
    assert make_model(name).named_steps["impute"].strategy == "median"
    assert MODEL_SPECS[name].scaled == (name not in ("random_forest", "hgb"))


@pytest.mark.parametrize("name", ["elasticnet", "gpr", "random_forest", "hgb", "mlp"])
def test_models_are_seeded(name: str) -> None:
    X, y = _problem()
    params = {"n_estimators": 30} if name == "random_forest" else {}  # keep the test fast
    a = make_model(name, random_state=7, **params).fit(X, y).predict(X)
    b = clone(make_model(name, random_state=7, **params)).fit(X, y).predict(X)
    np.testing.assert_array_equal(a, b)


def test_parameter_overrides_and_validation() -> None:
    model = make_model("ridge", alpha=10.0)
    assert model.named_steps["model"].alpha == 10.0
    assert make_model("random_forest", n_estimators=5).named_steps["model"].n_estimators == 5
    with pytest.raises(ValueError, match="Unknown model"):
        make_model("xgboost")
    with pytest.raises(ValueError, match="Invalid parameter"):
        make_model("ridge", depth=3)


def test_final_estimator_unwraps_target_transform() -> None:
    X, y = _problem()
    mlp = make_model("mlp").fit(X, y)
    assert isinstance(mlp.named_steps["model"], TransformedTargetRegressor)
    assert isinstance(final_estimator(mlp), MLPRegressor)
    ridge = make_model("ridge")
    assert final_estimator(ridge) is ridge.named_steps["model"]


def test_pls_is_affine_and_caps_components() -> None:
    X, y = _problem()
    X = np.nan_to_num(X)
    pls = PLSRegressor(n_components=5).fit(X[:, :2], y)
    assert pls.n_components_ == 2  # capped at the number of features
    np.testing.assert_allclose(pls.predict(X[:, :2]), X[:, :2] @ pls.coef_ + pls.intercept_)
    one = PLSRegressor(n_components=3).fit(X[:, :1], y)
    assert one.n_components_ == 1
    with pytest.raises(ValueError, match="n_components"):
        PLSRegressor(n_components=0).fit(X, y)


def test_pls_training_means_still_come_from_sklearn() -> None:
    # PLSRegressor reads the private PLSRegression._x_mean (no public equivalent): if a
    # scikit-learn release renames or redefines it, this fails instead of a wrong intercept.
    X, y = _problem()
    X = np.nan_to_num(X)
    pls = PLSRegression(n_components=2, scale=False).fit(X, y)
    x_mean = _pls_x_mean(pls)
    np.testing.assert_allclose(x_mean, X.mean(axis=0), rtol=1e-12)
    coef = np.asarray(pls.coef_).reshape(-1)
    expected = (X - x_mean) @ coef + np.asarray(pls.intercept_).reshape(-1)[0]
    np.testing.assert_allclose(np.asarray(pls.predict(X)).reshape(-1), expected, rtol=1e-12)
