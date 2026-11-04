"""Nested feature selection for Ridge: greedy backward elimination scored by inner LOBO.

Why this exists
---------------
With four core cells, choosing features by looking at leave-one-battery-out
(LOBO) *test* scores would leak the test batteries into the model. The
pre-specified headline model (``ridge_nfs``) therefore selects its features
and its ridge penalty *inside* every outer training fold, using only the
training batteries:

1. Start from every feature of the set. Score a feature subset ``S`` by the
   inner leave-one-battery-out MAE over the training batteries, minimised over
   the penalty grid (default ``alpha in {0.1, 1, 10}``). The inner MAE is the
   mean over inner folds of each held-out battery's MAE, so every battery
   counts once, like the headline metric.
2. Try removing each remaining feature; keep the removal with the lowest
   score if it is strictly better (by more than ``tol``) than the current
   subset, and repeat. Stop when no removal improves the inner MAE or
   ``min_features`` remain.
3. Refit :func:`battery_soh.models.zoo.make_model` ``("ridge", alpha=...)`` on
   all training rows with the selected features.

The outer held-out battery never enters steps 1-3, so the outer LOBO error of
the whole procedure is a nearly unbiased estimate of what selection-plus-fitting
does on a new cell (Varma & Simon 2006, "Bias in error estimation when using
cross-validation for model selection", BMC Bioinformatics 7:91), the
selection step included. It is conditional on design choices made with
knowledge of these cells (the candidate feature sets and indicator windows,
the penalty grid, absolute vs relative mode), which nested selection does not
undo; and with four outer folds the estimate is also very noisy.

Speed
-----
The pipeline (median imputation, standardisation, ridge) acts column by
column before the solve, so the imputed and standardised matrices of each
inner fold are computed once for all features; a subset's ridge solution is
then a ``k x k`` solve on the fold's Gram matrix. The scores equal those of
the scikit-learn pipeline up to floating-point rounding (unit-tested), and a
full elimination over 8 features costs milliseconds.

Groups
------
Scikit-learn's ``fit(X, y)`` has no ``groups`` argument, and
:class:`~battery_soh.uncertainty.conformal.GroupConformalRegressor` calls
``fit(X, y)`` on its fold models. :class:`NestedSelectionRidge` therefore reads
the battery id of each row from a column of ``X`` (default
:data:`GROUP_COLUMN`) and never uses that column as a feature.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from battery_soh._typing import FloatArray
from battery_soh.evaluation.splits import leave_one_group_out
from battery_soh.models.zoo import make_model

__all__ = [
    "DEFAULT_ALPHAS",
    "DEFAULT_TOL",
    "FALLBACK_ALPHA",
    "GROUP_COLUMN",
    "InnerLoboScorer",
    "NestedSelectionRidge",
    "SelectionResult",
    "SelectionStep",
    "backward_elimination",
]

logger = logging.getLogger(__name__)


#: Ridge penalties searched by the nested selection (pre-specified grid).
DEFAULT_ALPHAS: tuple[float, ...] = (0.1, 1.0, 10.0)
#: A removal must lower the inner MAE by more than this (SOH points) to be accepted,
#: so floating-point ties never drive the selection.
DEFAULT_TOL = 1e-9
#: Penalty used when selection is impossible (fewer than two training groups).
FALLBACK_ALPHA = 1.0
#: Column of ``X`` that carries the group (battery id) of each row.
GROUP_COLUMN = "_group"

# Same rule as sklearn's StandardScaler: a (near-)constant column keeps scale 1.
_ZERO_SCALE = 10.0 * np.finfo(np.float64).eps


@dataclass(frozen=True)
class SelectionStep:
    """One accepted state of the backward elimination.

    Attributes:
        features: Feature subset after this step.
        alpha: Best penalty for this subset.
        inner_mae: Inner LOBO MAE of this subset at ``alpha``.
        removed: Feature removed to reach this subset (``None`` for the start).
    """

    features: tuple[str, ...]
    alpha: float
    inner_mae: float
    removed: str | None


@dataclass(frozen=True)
class SelectionResult:
    """Outcome of :func:`backward_elimination`.

    Attributes:
        selected: Selected features, in the original column order.
        dropped: Removed features, in removal order.
        alpha: Selected ridge penalty.
        inner_mae: Inner LOBO MAE of the selected subset.
        path: Accepted steps, starting with the full set.
        n_groups: Number of training groups used for the inner folds.
        fallback: True if selection was skipped (fewer than two groups).
    """

    selected: tuple[str, ...]
    dropped: tuple[str, ...]
    alpha: float
    inner_mae: float
    path: tuple[SelectionStep, ...]
    n_groups: int
    fallback: bool = False

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly representation (NaN inner MAE becomes ``None``)."""
        return {
            "selected": list(self.selected),
            "dropped": list(self.dropped),
            "alpha": self.alpha,
            "inner_mae": None if math.isnan(self.inner_mae) else self.inner_mae,
            "n_groups": self.n_groups,
            "fallback": self.fallback,
            "path": [
                {
                    "removed": s.removed,
                    "n_features": len(s.features),
                    "alpha": s.alpha,
                    "inner_mae": s.inner_mae,
                }
                for s in self.path
            ],
        }


@dataclass(frozen=True)
class _Fold:
    gram: FloatArray  # Z_train' Z_train, (p, p)
    cross: FloatArray  # Z_train' (y_train - mean), (p,)
    y_mean: float
    z_test: FloatArray  # (m, p)
    y_test: FloatArray  # (m,)


def _impute_scale(X_train: FloatArray, X_test: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Median-impute and standardise with training statistics, column by column.

    Mirrors the zoo pipeline: ``SimpleImputer(strategy="median",
    keep_empty_features=True)`` (an all-missing column becomes 0) followed by
    ``StandardScaler()`` (population standard deviation, scale 1 for constant
    columns).
    """
    observed = np.isfinite(X_train)
    median = np.zeros(X_train.shape[1])
    for j in range(X_train.shape[1]):
        col = X_train[observed[:, j], j]
        if col.size:
            median[j] = float(np.median(col))
    train = np.where(observed, X_train, median)
    test = np.where(np.isfinite(X_test), X_test, median)
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale = np.where(scale < _ZERO_SCALE * np.maximum(1.0, np.abs(mean)), 1.0, scale)
    return (train - mean) / scale, (test - mean) / scale


class InnerLoboScorer:
    """Inner leave-one-group-out MAE of a ridge model on any feature subset.

    The folds are prepared once; :meth:`score` then costs one small linear
    solve per inner fold.

    Args:
        X: Training features, shape ``(n, p)``; NaN allowed (median-imputed per fold).
        y: Training labels, shape ``(n,)``, finite.
        groups: Group label (battery id) of each row; at least two distinct.

    Raises:
        ValueError: On shape mismatch, non-finite labels or fewer than two groups.
    """

    def __init__(self, X: ArrayLike, y: ArrayLike, groups: ArrayLike) -> None:
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
        if X_arr.ndim != 2 or X_arr.shape[0] != y_arr.size:
            raise ValueError(
                f"X must be (n, p) with n = len(y), got {X_arr.shape} and {y_arr.size}"
            )
        if not np.all(np.isfinite(y_arr)):
            raise ValueError("y must be finite")
        group_arr = np.asarray(groups, dtype=object).reshape(-1)
        if group_arr.size != y_arr.size:
            raise ValueError("groups must have one label per row")
        folds: list[_Fold] = []
        for train_idx, test_idx, _ in leave_one_group_out(group_arr):
            z_train, z_test = _impute_scale(X_arr[train_idx], X_arr[test_idx])
            y_train = y_arr[train_idx]
            y_mean = float(y_train.mean())
            folds.append(
                _Fold(
                    gram=z_train.T @ z_train,
                    cross=z_train.T @ (y_train - y_mean),
                    y_mean=y_mean,
                    z_test=z_test,
                    y_test=y_arr[test_idx],
                )
            )
        self.n_features = X_arr.shape[1]
        self.n_groups = len(folds)
        self._folds = tuple(folds)

    def predictions(self, subset: Sequence[int], alpha: float) -> list[FloatArray]:
        """Out-of-group predictions of every inner fold (fold order = sorted groups)."""
        if alpha <= 0 or not math.isfinite(alpha):
            raise ValueError(f"alpha must be a positive finite number, got {alpha}")
        idx = np.asarray(subset, dtype=np.intp)
        if idx.size == 0:
            raise ValueError("subset must contain at least one feature")
        out: list[FloatArray] = []
        eye = np.eye(idx.size)
        for fold in self._folds:
            gram = fold.gram[np.ix_(idx, idx)] + alpha * eye
            coef = np.linalg.solve(gram, fold.cross[idx])
            out.append(fold.z_test[:, idx] @ coef + fold.y_mean)
        return out

    def score(self, subset: Sequence[int], alpha: float) -> float:
        """Mean over inner folds of the held-out group's MAE."""
        preds = self.predictions(subset, alpha)
        maes = [
            float(np.mean(np.abs(p - fold.y_test)))
            for p, fold in zip(preds, self._folds, strict=True)
        ]
        return float(np.mean(maes))

    def best_alpha(self, subset: Sequence[int], alphas: Sequence[float]) -> tuple[float, float]:
        """``(alpha, score)`` minimising the score; the first alpha wins exact ties."""
        best_alpha, best_score = float(alphas[0]), math.inf
        for alpha in alphas:
            value = self.score(subset, float(alpha))
            if value < best_score:
                best_alpha, best_score = float(alpha), value
        return best_alpha, best_score


def _check_alphas(alphas: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(a) for a in alphas)
    if not values:
        raise ValueError("alphas must not be empty")
    if any(not (math.isfinite(a) and a > 0) for a in values):
        raise ValueError(f"alphas must be positive finite numbers, got {values}")
    return values


def backward_elimination(
    X: pd.DataFrame,
    y: ArrayLike,
    groups: ArrayLike,
    *,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    min_features: int = 1,
    tol: float = DEFAULT_TOL,
) -> SelectionResult:
    """Greedy backward elimination of features (and penalty) by inner LOBO MAE.

    Deterministic: candidate removals are tried in column order and the first
    of equally good candidates wins; a removal is accepted only if it lowers
    the inner MAE by more than ``tol``.

    Args:
        X: Training features with column names (NaN allowed).
        y: Training labels (finite).
        groups: Group (battery id) of each row. With fewer than two distinct
            groups no inner fold exists: every feature is kept with
            :data:`FALLBACK_ALPHA` and the result is flagged ``fallback``.
        alphas: Ridge penalties to search.
        min_features: Never remove below this many features.
        tol: Minimum improvement of the inner MAE to accept a removal.

    Returns:
        The :class:`SelectionResult`.

    Raises:
        ValueError: On invalid arguments or data (see :class:`InnerLoboScorer`).
    """
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a pandas DataFrame with feature names")
    names = [str(c) for c in X.columns]
    if not names:
        raise ValueError("X has no feature columns")
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate feature names in {names}")
    if min_features < 1:
        raise ValueError(f"min_features must be >= 1, got {min_features}")
    if not (math.isfinite(tol) and tol >= 0):
        raise ValueError(f"tol must be a finite number >= 0, got {tol}")
    alpha_grid = _check_alphas(alphas)
    group_arr = np.asarray(groups, dtype=object).reshape(-1)
    n_groups = len(pd.unique(group_arr))
    if n_groups < 2:
        logger.warning(
            "nested selection needs >= 2 training groups, got %d; keeping all %d features "
            "with alpha=%g",
            n_groups,
            len(names),
            FALLBACK_ALPHA,
        )
        return SelectionResult(
            selected=tuple(names),
            dropped=(),
            alpha=FALLBACK_ALPHA,
            inner_mae=math.nan,
            path=(),
            n_groups=n_groups,
            fallback=True,
        )

    scorer = InnerLoboScorer(X.to_numpy(dtype=np.float64), y, group_arr)
    current = list(range(len(names)))
    alpha, score = scorer.best_alpha(current, alpha_grid)
    path = [SelectionStep(tuple(names), alpha, score, None)]
    dropped: list[str] = []
    while len(current) > min_features:
        best: tuple[int, float, float] | None = None  # (position in current, alpha, score)
        for pos in range(len(current)):
            candidate = current[:pos] + current[pos + 1 :]
            cand_alpha, cand_score = scorer.best_alpha(candidate, alpha_grid)
            if best is None or cand_score < best[2]:
                best = (pos, cand_alpha, cand_score)
        assert best is not None
        pos, cand_alpha, cand_score = best
        if not cand_score < score - tol:
            break
        removed = names[current.pop(pos)]
        dropped.append(removed)
        alpha, score = cand_alpha, cand_score
        path.append(SelectionStep(tuple(names[i] for i in current), alpha, score, removed))
        logger.debug("dropped %s -> inner MAE %.4f (alpha=%g)", removed, score, alpha)
    return SelectionResult(
        selected=tuple(names[i] for i in current),
        dropped=tuple(dropped),
        alpha=alpha,
        inner_mae=score,
        path=tuple(path),
        n_groups=scorer.n_groups,
    )


class NestedSelectionRidge(RegressorMixin, BaseEstimator):  # type: ignore[misc]
    """Ridge whose features and penalty are chosen by nested inner-LOBO backward elimination.

    ``X`` must be a DataFrame holding the candidate features plus the group
    column ``group_col`` (battery id of each row), which is used only to form
    the inner folds. See the module docstring for the procedure.

    Args:
        alphas: Ridge penalties searched (default ``(0.1, 1, 10)``).
        group_col: Column of ``X`` with the group label; never a feature.
        min_features: Minimum number of features kept.
        tol: Minimum inner-MAE improvement to accept a removal.

    Attributes:
        selection_: The :class:`SelectionResult` of the last fit.
        selected_features_: Features used by the final model.
        alpha_: Penalty of the final model.
        inner_mae_: Inner LOBO MAE of the selected subset (NaN on fallback).
        pipeline_: Fitted zoo ``ridge`` pipeline on the selected features.
        feature_names_in_: Candidate feature names seen during ``fit``.
    """

    def __init__(
        self,
        alphas: Sequence[float] = DEFAULT_ALPHAS,
        *,
        group_col: str = GROUP_COLUMN,
        min_features: int = 1,
        tol: float = DEFAULT_TOL,
    ) -> None:
        self.alphas = alphas
        self.group_col = group_col
        self.min_features = min_features
        self.tol = tol

    def _split(self, X: Any) -> tuple[pd.DataFrame, NDArray[np.object_] | None]:
        if not isinstance(X, pd.DataFrame):
            raise ValueError(
                "NestedSelectionRidge needs a DataFrame with feature names and the "
                f"group column {self.group_col!r}"
            )
        if self.group_col not in X.columns:
            return X, None
        groups = X[self.group_col].to_numpy(dtype=object)
        return X.drop(columns=[self.group_col]), groups

    def fit(self, X: pd.DataFrame, y: ArrayLike) -> NestedSelectionRidge:
        """Select features and penalty on ``X``'s groups, then fit the final ridge."""
        features, groups = self._split(X)
        if groups is None:
            raise ValueError(f"X lacks the group column {self.group_col!r}")
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
        result = backward_elimination(
            features,
            y_arr,
            groups,
            alphas=self.alphas,
            min_features=self.min_features,
            tol=self.tol,
        )
        self.selection_ = result
        self.selected_features_ = list(result.selected)
        self.alpha_ = result.alpha
        self.inner_mae_ = result.inner_mae
        self.feature_names_in_ = np.asarray(features.columns, dtype=object)
        self.n_features_in_ = int(features.shape[1])
        self.pipeline_: Pipeline = make_model("ridge", alpha=self.alpha_).fit(
            features[self.selected_features_], y_arr
        )
        return self

    def predict(self, X: pd.DataFrame) -> FloatArray:
        """Predict with the selected features (the group column, if present, is ignored)."""
        check_is_fitted(self, "pipeline_")
        features, _ = self._split(X)
        missing = [c for c in self.selected_features_ if c not in features.columns]
        if missing:
            raise ValueError(f"X lacks selected feature(s) {missing}")
        pred = self.pipeline_.predict(features[self.selected_features_])
        return np.asarray(pred, dtype=np.float64).reshape(-1)

    def __sklearn_tags__(self) -> Any:
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags
