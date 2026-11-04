"""Model zoo: seeded scikit-learn pipelines with small-data defaults.

Every model is a :class:`~sklearn.pipeline.Pipeline` with the same named steps
so downstream code (bundle export, permutation importance) can rely on them::

    impute  SimpleImputer(strategy="median", keep_empty_features=True)
    scale   StandardScaler()                      # omitted for tree ensembles
    model   the estimator

Hyper-parameters are fixed, conservative defaults chosen for a few hundred
rows from a handful of batteries, not tuned on the evaluation batteries:
tuning inside leave-one-battery-out would need a nested grouped search, and
with four core cells that search would mostly fit noise. Override any default
with keyword arguments to :func:`make_model`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted, validate_data

from battery_soh._typing import FloatArray
from battery_soh.config import RANDOM_SEED

__all__ = [
    "LINEAR_MODEL_NAMES",
    "MODEL_NAMES",
    "MODEL_SPECS",
    "ModelSpec",
    "PLSRegressor",
    "final_estimator",
    "make_model",
]


def _pls_x_mean(pls: PLSRegression) -> FloatArray:
    """Column means of the training inputs that a fitted ``PLSRegression`` subtracts.

    scikit-learn exposes no public attribute for them, so this reads the private
    ``_x_mean``: the package's only access to a private scikit-learn attribute.
    ``tests/unit/test_models_zoo.py`` checks it against the training data, so a
    release that renames or redefines it fails there instead of silently giving
    PLS bundles a wrong intercept.
    """
    return np.asarray(pls._x_mean, dtype=np.float64)


class PLSRegressor(RegressorMixin, BaseEstimator):  # type: ignore[misc]
    """Partial least squares regression for a single target.

    Thin wrapper around :class:`sklearn.cross_decomposition.PLSRegression`
    that (a) caps ``n_components`` at what the data supports, so the same
    configuration runs on 1-feature and 10-feature sets, (b) always returns
    1-D predictions, and (c) exposes the fitted model as an affine map
    ``predict(X) == X @ coef_ + intercept_`` so it can be exported like any
    other linear model. Inputs are expected to be standardised already (the
    zoo pipeline does that), hence ``scale=False``.

    Args:
        n_components: Requested number of latent components (upper bound).
        max_iter: NIPALS iteration limit.
        tol: NIPALS convergence tolerance.

    Attributes:
        coef_: Shape ``(n_features,)`` slope on the (scaled) inputs.
        intercept_: Scalar offset.
        n_components_: Components actually used.
    """

    def __init__(self, n_components: int = 2, max_iter: int = 500, tol: float = 1e-6) -> None:
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X: ArrayLike, y: ArrayLike) -> PLSRegressor:
        """Fit the PLS model; ``n_components_`` records the components actually used."""
        if self.n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {self.n_components}")
        X_arr, y_arr = validate_data(self, X, y, y_numeric=True)
        cap = min(X_arr.shape[1], max(X_arr.shape[0] - 1, 1))
        self.n_components_ = int(min(self.n_components, cap))
        self.pls_ = PLSRegression(
            n_components=self.n_components_, scale=False, max_iter=self.max_iter, tol=self.tol
        ).fit(X_arr, y_arr)
        self.n_iter_ = np.asarray(self.pls_.n_iter_, dtype=np.intp)
        # sklearn predicts (X - x_mean) @ coef.T + y_mean; fold the centring into the offset.
        coef = np.asarray(self.pls_.coef_, dtype=np.float64).reshape(-1)
        x_mean = _pls_x_mean(self.pls_)
        self.coef_ = coef
        self.intercept_ = float(np.asarray(self.pls_.intercept_).reshape(-1)[0] - x_mean @ coef)
        return self

    def predict(self, X: ArrayLike) -> FloatArray:
        """Predict with the fitted PLS model (1-D output)."""
        check_is_fitted(self, "pls_")
        X_arr = validate_data(self, X, reset=False)
        return np.asarray(self.pls_.predict(X_arr), dtype=np.float64).reshape(-1)


@dataclass(frozen=True)
class ModelSpec:
    """Metadata and factory of one zoo model.

    Attributes:
        name: Registry key accepted by :func:`make_model`.
        description: One-line human-readable description (ASCII).
        linear: The fitted model is an affine function of the standardised
            inputs (exportable as an auditable JSON bundle).
        scaled: The pipeline standardises inputs before the estimator.
        defaults: Default constructor arguments of the estimator.
        build: ``(random_state, params) -> estimator``.
    """

    name: str
    description: str
    linear: bool
    scaled: bool
    defaults: Mapping[str, Any]
    build: Callable[[int, dict[str, Any]], Any]


def _gpr_kernel() -> Any:
    return ConstantKernel(1.0, (1e-3, 1e3)) * RBF(
        length_scale=1.0, length_scale_bounds=(1e-2, 1e3)
    ) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-6, 1e1))


def _build_mlp(random_state: int, params: dict[str, Any]) -> Any:
    # Scaling the target matters for an MLP: SOH is ~70-100, far from the
    # unit scale the default initialisation and learning rate assume.
    return TransformedTargetRegressor(
        regressor=MLPRegressor(random_state=random_state, **params),
        transformer=StandardScaler(),
    )


_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        "ridge",
        "Ridge regression (L2), the production model",
        linear=True,
        scaled=True,
        defaults={"alpha": 1.0},
        build=lambda _rs, p: Ridge(**p),
    ),
    ModelSpec(
        "pls",
        "Partial least squares, up to 2 latent components",
        linear=True,
        scaled=True,
        defaults={"n_components": 2},
        build=lambda _rs, p: PLSRegressor(**p),
    ),
    ModelSpec(
        "elasticnet",
        "Elastic net (L1 + L2)",
        linear=True,
        scaled=True,
        defaults={"alpha": 0.05, "l1_ratio": 0.5, "max_iter": 50_000},
        build=lambda rs, p: ElasticNet(random_state=rs, **p),
    ),
    ModelSpec(
        "gpr",
        "Gaussian process, constant * RBF + white noise kernel",
        linear=False,
        scaled=True,
        # One restart guards against the noise-only optimum at a fraction of
        # the cost of more (each restart refactorises an n x n kernel matrix).
        defaults={"normalize_y": True, "n_restarts_optimizer": 1},
        build=lambda rs, p: GaussianProcessRegressor(
            **{"kernel": _gpr_kernel(), **p}, random_state=rs
        ),
    ),
    ModelSpec(
        "random_forest",
        "Random forest, 300 trees, min 3 samples per leaf",
        linear=False,
        scaled=False,
        defaults={"n_estimators": 300, "min_samples_leaf": 3, "max_features": 1.0, "n_jobs": 1},
        build=lambda rs, p: RandomForestRegressor(random_state=rs, **p),
    ),
    ModelSpec(
        "hgb",
        "Histogram gradient boosting, shallow trees",
        linear=False,
        scaled=False,
        defaults={
            "max_iter": 200,
            "learning_rate": 0.07,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 10,
        },
        build=lambda rs, p: HistGradientBoostingRegressor(random_state=rs, **p),
    ),
    ModelSpec(
        "mlp",
        "Multi-layer perceptron (32-16), early stopping, scaled target",
        linear=False,
        scaled=True,
        defaults={
            "hidden_layer_sizes": (32, 16),
            "alpha": 1e-3,
            "learning_rate_init": 1e-3,
            "max_iter": 2000,
            "early_stopping": True,
            "validation_fraction": 0.15,
            "n_iter_no_change": 30,
        },
        build=_build_mlp,
    ),
    ModelSpec(
        "knn",
        "k-nearest neighbours (k=10, distance-weighted)",
        linear=False,
        scaled=True,
        defaults={"n_neighbors": 10, "weights": "distance"},
        build=lambda _rs, p: KNeighborsRegressor(**p),
    ),
)

#: Registry of zoo models keyed by name.
MODEL_SPECS: Mapping[str, ModelSpec] = MappingProxyType({s.name: s for s in _SPECS})

#: Names accepted by :func:`make_model`, in benchmark display order.
MODEL_NAMES: tuple[str, ...] = tuple(s.name for s in _SPECS)

#: Models whose fitted form is affine in the standardised inputs.
LINEAR_MODEL_NAMES: tuple[str, ...] = tuple(s.name for s in _SPECS if s.linear)


def make_model(name: str, *, random_state: int = RANDOM_SEED, **params: Any) -> Pipeline:
    """Build an unfitted, seeded model pipeline.

    Args:
        name: One of :data:`MODEL_NAMES`.
        random_state: Seed for every stochastic component.
        **params: Overrides of the estimator's default constructor arguments
            (e.g. ``make_model("ridge", alpha=10.0)``).

    Returns:
        ``Pipeline([("impute", ...), ("scale", ...), ("model", ...)])``; tree
        ensembles have no ``scale`` step.

    Raises:
        ValueError: If ``name`` is unknown or a parameter is not accepted by
            the estimator.
    """
    try:
        spec = MODEL_SPECS[name]
    except KeyError:
        raise ValueError(f"Unknown model {name!r}; available: {', '.join(MODEL_NAMES)}") from None
    merged = {**spec.defaults, **params}
    try:
        estimator = spec.build(random_state, merged)
    except TypeError as exc:
        raise ValueError(f"Invalid parameter for model {name!r}: {exc}") from exc
    steps: list[tuple[str, Any]] = [
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True))
    ]
    if spec.scaled:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


def final_estimator(pipeline: Pipeline) -> Any:
    """Return the innermost estimator of a zoo pipeline.

    Unwraps the ``model`` step and a :class:`TransformedTargetRegressor`
    (used by ``mlp``), preferring the fitted ``regressor_`` when available.
    """
    estimator = pipeline.named_steps["model"]
    if isinstance(estimator, TransformedTargetRegressor):
        return getattr(estimator, "regressor_", estimator.regressor)
    return estimator
