"""Conformal prediction intervals with exact finite-sample ranks and battery-grouped calibration.

Split conformal in one paragraph
--------------------------------
Given ``n`` calibration scores ``s_i = |y_i - f(x_i)|`` that are exchangeable
with the score of a new point, the interval ``f(x) +/- q`` with
``q = s_(k)``, the ``k``-th smallest score and ``k = ceil((n + 1)(1 - alpha))``,
covers the new label with probability at least ``1 - alpha`` (and at most
``1 - alpha + 1/(n + 1)`` without ties). When ``k > n`` (too few scores for the
requested level, i.e. ``n < 1/alpha - 1``) no finite interval has the
guarantee and ``q = +inf``. Returning ``max(scores)`` instead, as the v1 code
did, silently breaks the guarantee (simulated coverage 0.835 at ``n = 5`` for
a 0.90 target).

Grouped calibration
-------------------
Batteries, not cycles, are the exchangeable units here: rows of one cell are
strongly dependent, and a deployed model always meets a *new* cell.
:class:`GroupConformalRegressor` therefore produces its calibration scores by
an inner leave-one-group-out loop over the **training** batteries: each
battery's residuals come from a model that never saw that battery, so they
reflect the cross-battery error the model will make on a new cell. Two
interval constructions are offered:

* ``"cv_plus"`` - CV+ of Barber, Candes, Ramdas & Tibshirani (2021,
  "Predictive inference with the jackknife+", Ann. Statist. 49(1)) with
  battery folds: the interval combines every fold model's prediction at
  ``x`` with that fold's residuals. Its worst-case guarantee is
  ``1 - 2 alpha`` (minus a small K-fold term); in practice it is close to
  ``1 - alpha``. The interval need not be centred on the full-data prediction.
* ``"cv"`` - the full-data model's prediction ``+/- q`` with ``q`` the exact
  conformal quantile of the pooled out-of-battery residuals. This is the form
  stored in the production JSON bundle (a fixed half-width per confidence
  level), so the benchmark can report the coverage of exactly what ships. It
  has no formal finite-sample guarantee of its own (the full model is not one
  of the fold models), which is why the benchmark measures its coverage on
  held-out batteries.

What is (not) guaranteed in this project
-----------------------------------------
The finite-sample statements above are about exchangeable *scores*. Here the
scores are the cycles of a few calibration batteries, and the cycles of one
cell are strongly dependent, so the row-level statement does not transfer to
a new cell: what a deployment needs is a guarantee over *batteries*, and the
unit count is the number of calibration batteries, not the number of rows.
Treating each battery as one exchangeable unit, a finite ``1 - alpha``
interval needs at least :func:`min_calibration_size` units: 9 batteries for
90 %, 19 for 95 %. With the four core cells, every outer leave-one-battery-out
fold calibrates on three training cells (hundreds of residuals, but three
units), and the production bundles on four, so **no battery-level
finite-sample coverage guarantee exists for any interval reported here**; the
``cv`` half-width has none even in the exchangeable row-level setting. The
coverage the benchmark reports on held-out cells is a measurement, from four
cells, not a guarantee. Any guarantee would further assume the new battery is
exchangeable with the training batteries (same chemistry, protocol and test
rig); nothing here covers a cell from a different population.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from fractions import Fraction
from typing import Any, Literal, Protocol, Self

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from battery_soh._typing import FloatArray
from battery_soh.evaluation.splits import leave_one_group_out

__all__ = [
    "DEFAULT_CONFIDENCES",
    "ConformalMethod",
    "GroupConformalRegressor",
    "Regressor",
    "confidence_key",
    "confidence_quantiles",
    "conformal_quantile",
    "conformal_rank",
    "min_calibration_size",
]

logger = logging.getLogger(__name__)

ConformalMethod = Literal["cv_plus", "cv"]

#: Confidence levels served by the API and stored in model bundles.
DEFAULT_CONFIDENCES: tuple[float, ...] = (0.8, 0.9, 0.95)

# alpha is read as the nearest fraction with a denominator up to this bound and
# ranks are computed in exact rational arithmetic, so they never depend on
# binary rounding. In particular a level derived as ``1 - 0.9`` (the float
# 0.09999999999999998) means exactly 1/10, as intended. Any alpha written with
# up to six decimals is represented exactly.
_ALPHA_MAX_DENOMINATOR = 1_000_000
# Upper bound on the (calibration x test) matrix materialised by CV+ at once.
_CV_PLUS_MAX_CELLS = 4_000_000


class Regressor(Protocol):
    """Anything with scikit-learn's ``fit``/``predict`` regression interface."""

    def fit(self, X: Any, y: Any) -> Any: ...

    def predict(self, X: Any) -> Any: ...


def _check_alpha(alpha: float) -> float:
    alpha = float(alpha)
    if math.isnan(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    return alpha


def _alpha_fraction(alpha: float) -> Fraction:
    exact = Fraction(_check_alpha(alpha))
    nearest = exact.limit_denominator(_ALPHA_MAX_DENOMINATOR)
    # Extreme levels (alpha < 5e-7 or > 1 - 5e-7) keep their exact binary value.
    return nearest if 0 < nearest < 1 else exact


def conformal_rank(n: int, alpha: float) -> int:
    """Rank ``k = ceil((n + 1)(1 - alpha))`` of the split-conformal quantile.

    Computed in exact rational arithmetic (see ``_ALPHA_MAX_DENOMINATOR``).

    Args:
        n: Number of calibration scores (``>= 0``).
        alpha: Target miscoverage in (0, 1).

    Returns:
        The 1-based rank; a value ``> n`` means the quantile is ``+inf``.

    Raises:
        ValueError: If ``n < 0`` or ``alpha`` is outside (0, 1).
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    return math.ceil((n + 1) * (1 - _alpha_fraction(alpha)))


def _lower_rank(n: int, alpha: float) -> int:
    """``floor(alpha (n + 1))``, the CV+ lower-bound rank (0 means ``-inf``)."""
    return math.floor(_alpha_fraction(alpha) * (n + 1))


def min_calibration_size(alpha: float) -> int:
    """Smallest number of scores for which the ``1 - alpha`` quantile is finite.

    ``k <= n`` holds exactly when ``n >= 1/alpha - 1``, so this is
    ``ceil(1/alpha - 1)``: 4 for alpha = 0.2, 9 for 0.1, 19 for 0.05.
    """
    frac = _alpha_fraction(alpha)
    return math.ceil((1 - frac) / frac)


def conformal_quantile(scores: ArrayLike, alpha: float) -> float:
    """Exact finite-sample split-conformal quantile.

    Args:
        scores: Calibration non-conformity scores (e.g. absolute residuals),
            exchangeable with the score of the point to be predicted.
        alpha: Target miscoverage in (0, 1); ``0.1`` gives a 90 % interval.

    Returns:
        ``sorted(scores)[k - 1]`` with ``k = ceil((n + 1)(1 - alpha))``, or
        ``+inf`` when ``k > n`` (including ``n = 0``): with fewer than
        :func:`min_calibration_size` scores no finite interval is valid.

    Raises:
        ValueError: If ``alpha`` is outside (0, 1), or ``scores`` is not 1-D
            or contains NaN.
    """
    arr = np.asarray(scores, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"scores must be one-dimensional, got shape {arr.shape}")
    if np.isnan(arr).any():
        raise ValueError("scores must not contain NaN")
    k = conformal_rank(arr.size, alpha)
    if k > arr.size:
        return math.inf
    return float(np.partition(arr, k - 1)[k - 1])


def confidence_key(confidence: float | str) -> str:
    """Canonical string key of a confidence level: ``0.9 -> "0.9"``, ``"0.950" -> "0.95"``.

    This is the key format of ``conformal.quantiles`` in model bundles.

    Raises:
        ValueError: If ``confidence`` is not a number in (0, 1).
    """
    value = float(confidence)
    if math.isnan(value) or not 0.0 < value < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence!r}")
    return f"{round(value, 6):g}"


def confidence_quantiles(
    scores: ArrayLike, confidences: Sequence[float] = DEFAULT_CONFIDENCES
) -> dict[str, float]:
    """Conformal half-widths keyed by confidence string, e.g. ``{"0.9": 2.1}``.

    The value for confidence ``c`` is ``conformal_quantile(scores, 1 - c)``
    (``+inf`` when there are too few scores). This is the form
    :func:`battery_soh.models.linear_bundle.export_linear_bundle` accepts.

    Args:
        scores: Calibration scores, e.g. pooled leave-one-battery-out absolute
            residuals of the production model.
        confidences: Nominal coverage levels in (0, 1).
    """
    out: dict[str, float] = {}
    for c in confidences:
        key = confidence_key(c)
        out[key] = conformal_quantile(scores, 1.0 - float(key))
    return out


def _subset(X: Any, idx: NDArray[np.intp]) -> Any:
    """Row subset that keeps DataFrames as DataFrames (column-aware pipelines)."""
    if isinstance(X, pd.DataFrame | pd.Series):
        return X.iloc[idx]
    return np.asarray(X)[idx]


def _n_rows(X: Any) -> int:
    return int(X.shape[0]) if hasattr(X, "shape") else len(X)


def _predict(model: Regressor, X: Any) -> FloatArray:
    return np.asarray(model.predict(X), dtype=np.float64).reshape(-1)


class GroupConformalRegressor:
    """Regression intervals calibrated by leave-one-group-out residuals.

    Calibration uses only the groups passed to :meth:`fit`: each training
    battery is held out once, a fresh model is fitted on the others, and its
    absolute residuals on the held-out battery become calibration scores
    (group CV+). No held-out *evaluation* battery is ever touched, so an outer
    leave-one-battery-out benchmark stays honest.

    Note:
        The rank-based guarantee is over exchangeable rows. Over batteries it
        can hold, on average over new batteries, only if batteries are
        exchangeable (same population, protocol and rig) *and* there are at
        least :func:`min_calibration_size` training groups (9 for a 90 %
        interval); with the three or four cells used in this project it does
        not hold, so measure coverage on held-out batteries. It is never a
        per-battery guarantee and says nothing about a cell from a different
        population.

    Args:
        estimator_factory: Zero-argument callable returning a fresh, unfitted
            regressor (e.g. ``functools.partial(make_model, "ridge")``). It is
            called once per training group plus once for the final model.
        alphas: Miscoverage levels whose quantiles are pre-computed in
            :attr:`quantiles_` (default 80/90/95 % intervals). Any other level
            can still be requested from :meth:`predict_interval`.
        method: ``"cv_plus"`` (default) or ``"cv"``; see the module docstring.

    Attributes:
        estimator_: Model fitted on all training rows (point predictions).
        fold_estimators_: ``{group: model fitted without that group}``.
        groups_: Sorted training group labels.
        scores_: Out-of-group absolute residuals, one per training row.
        oof_predictions_: Out-of-group predictions, one per training row.
        score_groups_: Group label of each score.
        quantiles_: ``{alpha: conformal_quantile(scores_, alpha)}``.
    """

    def __init__(
        self,
        estimator_factory: Callable[[], Regressor],
        alphas: Sequence[float] = (0.2, 0.1, 0.05),
        *,
        method: ConformalMethod = "cv_plus",
    ) -> None:
        if method not in ("cv_plus", "cv"):
            raise ValueError(f"method must be 'cv_plus' or 'cv', got {method!r}")
        if not alphas:
            raise ValueError("alphas must not be empty")
        self.estimator_factory = estimator_factory
        self.alphas = tuple(_check_alpha(a) for a in alphas)
        self.method: ConformalMethod = method

    # ------------------------------------------------------------------ fitting
    def fit(self, X: Any, y: ArrayLike, groups: pd.Series | ArrayLike) -> Self:
        """Fit fold models on all-but-one training group, then the full model.

        Args:
            X: Training features (array or DataFrame); NaN is allowed if the
                estimator handles it.
            y: Training labels (finite).
            groups: Battery id of each training row (at least two distinct).

        Returns:
            ``self``.

        Raises:
            ValueError: On length mismatch, non-finite labels or < 2 groups.
        """
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
        group_arr = (
            groups.to_numpy() if isinstance(groups, pd.Series) else np.asarray(groups)
        ).astype(object)
        n = _n_rows(X)
        if not (y_arr.size == n == group_arr.size):
            raise ValueError(
                f"X, y and groups must have equal length, got {n}, {y_arr.size}, {group_arr.size}"
            )
        if not np.all(np.isfinite(y_arr)):
            raise ValueError("y must be finite")

        oof = np.full(n, np.nan)
        fold_estimators: dict[str, Regressor] = {}
        fold_of_row = np.empty(n, dtype=object)
        for train_idx, cal_idx, held_out in leave_one_group_out(group_arr):
            model = self.estimator_factory()
            model.fit(_subset(X, train_idx), y_arr[train_idx])
            oof[cal_idx] = _predict(model, _subset(X, cal_idx))
            fold_estimators[held_out] = model
            fold_of_row[cal_idx] = held_out
        if not np.all(np.isfinite(oof)):
            raise ValueError("a fold model produced non-finite predictions")

        self.estimator_ = self.estimator_factory()
        self.estimator_.fit(X, y_arr)
        self.fold_estimators_ = fold_estimators
        self.groups_ = tuple(fold_estimators)
        self.oof_predictions_ = oof
        self.scores_ = np.abs(y_arr - oof)
        self.score_groups_ = fold_of_row.astype(str)
        self.quantiles_ = {a: conformal_quantile(self.scores_, a) for a in self.alphas}
        infinite = [a for a, q in self.quantiles_.items() if math.isinf(q)]
        if infinite:
            logger.warning(
                "conformal quantile is +inf for alpha=%s (n_scores=%d); intervals are unbounded",
                infinite,
                self.scores_.size,
            )
        return self

    @property
    def n_scores_(self) -> int:
        """Number of calibration scores."""
        self._check_fitted()
        return int(self.scores_.size)

    def _check_fitted(self) -> None:
        if not hasattr(self, "estimator_"):
            raise RuntimeError("GroupConformalRegressor is not fitted; call fit() first")

    # ------------------------------------------------------------------ prediction
    def predict(self, X: Any) -> FloatArray:
        """Point predictions of the full-data model."""
        self._check_fitted()
        return _predict(self.estimator_, X)

    def quantile(self, alpha: float) -> float:
        """Conformal half-width of the ``"cv"`` interval at miscoverage ``alpha``."""
        self._check_fitted()
        alpha = _check_alpha(alpha)
        cached = self.quantiles_.get(alpha)
        return cached if cached is not None else conformal_quantile(self.scores_, alpha)

    def predict_interval(self, X: Any, alpha: float) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Point prediction and ``1 - alpha`` prediction interval.

        Args:
            X: Features of the rows to predict.
            alpha: Miscoverage in (0, 1).

        Returns:
            ``(yhat, lower, upper)``; bounds may be ``-inf``/``+inf`` when there
            are too few calibration scores for the requested level.
        """
        self._check_fitted()
        alpha = _check_alpha(alpha)
        yhat = self.predict(X)
        if self.method == "cv":
            q = self.quantile(alpha)
            return yhat, yhat - q, yhat + q
        lower, upper = self._cv_plus_bounds(X, alpha)
        return yhat, lower, upper

    def predict_intervals(self, X: Any) -> dict[float, tuple[FloatArray, FloatArray]]:
        """Intervals for every level in :attr:`alphas` (fold predictions computed once)."""
        self._check_fitted()
        if self.method == "cv":
            yhat = self.predict(X)
            return {a: (yhat - self.quantiles_[a], yhat + self.quantiles_[a]) for a in self.alphas}
        per_fold, score_fold = self._fold_predictions(X)
        return {a: self._cv_plus_from_folds(per_fold, score_fold, a) for a in self.alphas}

    def _fold_predictions(self, X: Any) -> tuple[FloatArray, NDArray[np.intp]]:
        """Per-fold predictions ``(n_groups, n_test)`` and the fold of each score.

        Row ``score_fold[i]`` of the matrix holds the predictions of the model
        that did *not* see calibration row ``i``'s battery.
        """
        fold_index = {g: i for i, g in enumerate(self.fold_estimators_)}
        per_fold = np.vstack([_predict(m, X) for m in self.fold_estimators_.values()])
        score_fold = np.array([fold_index[g] for g in self.score_groups_], dtype=np.intp)
        return np.asarray(per_fold, dtype=np.float64), score_fold

    def _cv_plus_bounds(self, X: Any, alpha: float) -> tuple[FloatArray, FloatArray]:
        return self._cv_plus_from_folds(*self._fold_predictions(X), alpha)

    def _cv_plus_from_folds(
        self, per_fold: FloatArray, score_fold: NDArray[np.intp], alpha: float
    ) -> tuple[FloatArray, FloatArray]:
        """CV+ bounds (Barber et al. 2021, eq. 6) with exact order statistics.

        lower = floor(alpha (n+1))-th smallest of  mu_{-g(i)}(x) - R_i  (-inf if rank 0)
        upper = ceil((1-alpha)(n+1))-th smallest of mu_{-g(i)}(x) + R_i  (+inf if rank > n)
        """
        n = self.scores_.size
        m = per_fold.shape[1]
        k_lo = _lower_rank(n, alpha)
        k_hi = conformal_rank(n, alpha)
        lower = np.full(m, -np.inf)
        upper = np.full(m, np.inf)
        if k_lo < 1 and k_hi > n:
            return lower, upper
        scores = self.scores_[:, None]
        chunk = max(1, _CV_PLUS_MAX_CELLS // max(n, 1))
        for start in range(0, m, chunk):
            cols = slice(start, min(start + chunk, m))
            block = per_fold[:, cols][score_fold]  # (n_scores, chunk)
            if k_lo >= 1:
                lower[cols] = np.partition(block - scores, k_lo - 1, axis=0)[k_lo - 1]
            if k_hi <= n:
                upper[cols] = np.partition(block + scores, k_hi - 1, axis=0)[k_hi - 1]
        return lower, upper

    def quantiles_by_confidence(self) -> dict[str, float]:
        """Pre-computed ``"cv"`` half-widths keyed by confidence string (``"0.9"``).

        This is the form stored in the JSON model bundle (``+inf`` = unbounded).
        """
        self._check_fitted()
        return {confidence_key(1.0 - a): q for a, q in self.quantiles_.items()}
