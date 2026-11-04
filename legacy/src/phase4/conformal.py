"""Conformal prediction for calibrated SOH intervals.

Provides distribution-free, finite-sample coverage guarantees for the
cross-attention model's SOH point predictions. Two variants:

    SplitConformalRegressor
        Standard split-conformal. Calibrate on a held-out set, then
        emit symmetric intervals f(x) ± q with marginal coverage 1-alpha.

    MondrianConformalRegressor
        Per-group calibration (e.g. per-battery or per-dataset). Gives
        conditional coverage so guarantees hold within each group, not
        only on average.

References:
    Vovk, Gammerman, Shafer (2005). Algorithmic Learning in a Random World.
    Lei et al. (2018). Distribution-Free Predictive Inference for Regression.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np


@dataclass
class ConformalInterval:
    """SOH point estimate plus a calibrated interval at confidence 1-alpha."""

    point: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    alpha: float
    q_hat: float | np.ndarray

    @property
    def width(self) -> np.ndarray:
        return self.upper - self.lower


def _finite_sample_quantile(scores: np.ndarray, alpha: float) -> float:
    """The (1 - alpha) * (1 + 1/n) empirical quantile used by split-conformal.

    The 1/n inflation is what gives the exact finite-sample coverage guarantee
    rather than the asymptotic one a plain quantile would provide.
    """
    n = len(scores)
    if n == 0:
        raise ValueError("Need at least one calibration score")
    level = np.ceil((n + 1) * (1 - alpha)) / n
    level = float(min(max(level, 0.0), 1.0))
    return float(np.quantile(scores, level, method="higher"))


class SplitConformalRegressor:
    """Wrap any point regressor to emit calibrated prediction intervals.

    The base predictor is treated as a black box: only its predictions on the
    calibration set are used. No retraining, no architectural changes.
    """

    def __init__(self, predict_fn: Callable[[np.ndarray], np.ndarray], alpha: float = 0.1):
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        self.predict_fn = predict_fn
        self.alpha = alpha
        self.q_hat: Optional[float] = None
        self._n_calib: int = 0

    def calibrate(self, x_calib, y_calib: np.ndarray) -> "SplitConformalRegressor":
        preds = np.asarray(self.predict_fn(x_calib)).reshape(-1)
        y_calib = np.asarray(y_calib).reshape(-1)
        if preds.shape != y_calib.shape:
            raise ValueError(
                f"Calibration prediction shape {preds.shape} != target shape {y_calib.shape}"
            )
        residuals = np.abs(y_calib - preds)
        self.q_hat = _finite_sample_quantile(residuals, self.alpha)
        self._n_calib = len(residuals)
        return self

    def predict_interval(self, x) -> ConformalInterval:
        if self.q_hat is None:
            raise RuntimeError("Call .calibrate() before .predict_interval().")
        preds = np.asarray(self.predict_fn(x)).reshape(-1)
        return ConformalInterval(
            point=preds,
            lower=preds - self.q_hat,
            upper=preds + self.q_hat,
            alpha=self.alpha,
            q_hat=self.q_hat,
        )


class MondrianConformalRegressor:
    """Per-group split-conformal. Each group keeps its own quantile.

    A natural use here is to calibrate separately per battery or per dataset
    (NASA vs Mendeley) so coverage guarantees hold within each subpopulation
    instead of only marginally.
    """

    def __init__(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        alpha: float = 0.1,
        fallback_to_global: bool = True,
    ):
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        self.predict_fn = predict_fn
        self.alpha = alpha
        self.fallback_to_global = fallback_to_global
        self.q_by_group: Dict[str, float] = {}
        self.global_q: Optional[float] = None
        self.group_sizes: Dict[str, int] = {}

    def calibrate(
        self,
        x_calib,
        y_calib: np.ndarray,
        groups_calib: Sequence,
    ) -> "MondrianConformalRegressor":
        preds = np.asarray(self.predict_fn(x_calib)).reshape(-1)
        y_calib = np.asarray(y_calib).reshape(-1)
        groups = np.asarray(groups_calib).reshape(-1)
        if not (len(preds) == len(y_calib) == len(groups)):
            raise ValueError("preds, y_calib, groups must all have the same length")

        residuals = np.abs(y_calib - preds)
        self.global_q = _finite_sample_quantile(residuals, self.alpha)
        self.q_by_group = {}
        self.group_sizes = {}
        for g in np.unique(groups):
            mask = groups == g
            grp_residuals = residuals[mask]
            self.group_sizes[str(g)] = int(mask.sum())
            self.q_by_group[str(g)] = _finite_sample_quantile(grp_residuals, self.alpha)
        return self

    def predict_interval(self, x, groups: Sequence) -> ConformalInterval:
        if self.global_q is None:
            raise RuntimeError("Call .calibrate() before .predict_interval().")
        preds = np.asarray(self.predict_fn(x)).reshape(-1)
        groups = np.asarray(groups).reshape(-1)
        if len(preds) != len(groups):
            raise ValueError("preds and groups must have the same length")
        q_hat = np.empty(len(preds), dtype=float)
        for i, g in enumerate(groups):
            key = str(g)
            if key in self.q_by_group:
                q_hat[i] = self.q_by_group[key]
            elif self.fallback_to_global:
                q_hat[i] = self.global_q
            else:
                raise KeyError(f"Group {key!r} not seen during calibration")
        return ConformalInterval(
            point=preds,
            lower=preds - q_hat,
            upper=preds + q_hat,
            alpha=self.alpha,
            q_hat=q_hat,
        )


def empirical_coverage(
    interval: ConformalInterval, y_true: np.ndarray
) -> Dict[str, float]:
    """Empirical coverage and width statistics for a held-out test set."""
    y_true = np.asarray(y_true).reshape(-1)
    if y_true.shape != interval.point.shape:
        raise ValueError("y_true shape must match the interval predictions")
    inside = (y_true >= interval.lower) & (y_true <= interval.upper)
    return {
        "target_coverage": 1 - interval.alpha,
        "empirical_coverage": float(inside.mean()),
        "mean_width": float(interval.width.mean()),
        "median_width": float(np.median(interval.width)),
        "n": int(len(y_true)),
    }


def grouped_empirical_coverage(
    interval: ConformalInterval,
    y_true: np.ndarray,
    groups: Sequence,
) -> Dict[str, Dict[str, float]]:
    """Per-group coverage. Useful to verify Mondrian calibration worked."""
    y_true = np.asarray(y_true).reshape(-1)
    groups = np.asarray(groups).reshape(-1)
    out: Dict[str, Dict[str, float]] = {}
    for g in np.unique(groups):
        mask = groups == g
        if not mask.any():
            continue
        sub = ConformalInterval(
            point=interval.point[mask],
            lower=interval.lower[mask],
            upper=interval.upper[mask],
            alpha=interval.alpha,
            q_hat=interval.q_hat if np.isscalar(interval.q_hat) else np.asarray(interval.q_hat)[mask],
        )
        out[str(g)] = empirical_coverage(sub, y_true[mask])
    return out
