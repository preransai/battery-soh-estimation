"""Conformalised Quantile Regression (CQR) for adaptive SOH intervals.

Split-conformal prediction (see ``conformal.py``) produces intervals of a
fixed width q_hat. That's easy to calibrate but wastes width on easy samples
and can still under-cover on hard ones. CQR fits two quantile regressors at
alpha/2 and 1-alpha/2, then conformally adjusts their predictions so the
coverage guarantee holds while the width varies per input.

References
    Romano, Patterson, Candes (2019), "Conformalized Quantile Regression."

API
    CQRRegressor(predict_quantile, alpha=0.10).calibrate(X_cal, y_cal)
    CQRRegressor.predict_interval(X)  -> AdaptiveInterval

    Use ``train_gradient_boosting_quantile_pair`` to build the predict function
    from raw training data without wiring anything else.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from .conformal import _finite_sample_quantile  # reuse finite-sample quantile helper


@dataclass
class AdaptiveInterval:
    """Per-sample intervals with varying widths."""

    point: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    alpha: float
    q_hat: float

    @property
    def width(self) -> np.ndarray:
        return self.upper - self.lower


class CQRRegressor:
    """Wraps a (low, high) quantile predictor and conformally adjusts its output.

    ``predict_quantile`` is a callable taking x and returning ``(q_lo, q_hi)``
    as two arrays of shape (n,). The caller supplies any model fitted at the
    alpha/2 and 1-alpha/2 levels; this class only handles calibration.
    """

    def __init__(
        self,
        predict_quantile: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
        alpha: float = 0.10,
        point_predict: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        self.predict_quantile = predict_quantile
        self.point_predict = point_predict
        self.alpha = alpha
        self.q_hat: Optional[float] = None
        self.n_calib: int = 0

    def calibrate(self, x_calib, y_calib: np.ndarray) -> "CQRRegressor":
        q_lo, q_hi = self.predict_quantile(x_calib)
        q_lo = np.asarray(q_lo).reshape(-1)
        q_hi = np.asarray(q_hi).reshape(-1)
        y = np.asarray(y_calib).reshape(-1)
        if not (len(q_lo) == len(q_hi) == len(y)):
            raise ValueError("Shape mismatch between predictions and targets.")
        # Non-conformity score: how far outside [q_lo, q_hi] is the target.
        scores = np.maximum(q_lo - y, y - q_hi)
        self.q_hat = _finite_sample_quantile(scores, self.alpha)
        self.n_calib = len(scores)
        return self

    def predict_interval(self, x) -> AdaptiveInterval:
        if self.q_hat is None:
            raise RuntimeError("Call .calibrate() before .predict_interval().")
        q_lo, q_hi = self.predict_quantile(x)
        q_lo = np.asarray(q_lo).reshape(-1)
        q_hi = np.asarray(q_hi).reshape(-1)
        lower = q_lo - self.q_hat
        upper = q_hi + self.q_hat
        if self.point_predict is not None:
            point = np.asarray(self.point_predict(x)).reshape(-1)
        else:
            point = (q_lo + q_hi) / 2.0
        return AdaptiveInterval(
            point=point,
            lower=lower,
            upper=upper,
            alpha=self.alpha,
            q_hat=float(self.q_hat),
        )


def train_gradient_boosting_quantile_pair(
    X_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float = 0.10,
    n_estimators: int = 400,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    random_state: int = 42,
) -> Tuple[Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]], Callable[[np.ndarray], np.ndarray]]:
    """Train two quantile GradientBoostingRegressor models and a central point
    regressor. Returns (predict_quantile_fn, point_predict_fn)."""
    from sklearn.ensemble import GradientBoostingRegressor

    lo_alpha = alpha / 2
    hi_alpha = 1 - alpha / 2
    kwargs = dict(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
    )
    lo = GradientBoostingRegressor(loss="quantile", alpha=lo_alpha, **kwargs)
    hi = GradientBoostingRegressor(loss="quantile", alpha=hi_alpha, **kwargs)
    point = GradientBoostingRegressor(loss="quantile", alpha=0.5, **kwargs)
    lo.fit(X_train, y_train)
    hi.fit(X_train, y_train)
    point.fit(X_train, y_train)

    def predict_quantile(X):
        return lo.predict(X), hi.predict(X)

    def point_predict(X):
        return point.predict(X)

    return predict_quantile, point_predict


def empirical_cqr_metrics(interval: AdaptiveInterval, y_true: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y_true).reshape(-1)
    if y.shape != interval.point.shape:
        raise ValueError("y shape must match interval predictions")
    inside = (y >= interval.lower) & (y <= interval.upper)
    widths = interval.width
    return {
        "target_coverage": 1 - interval.alpha,
        "empirical_coverage": float(inside.mean()),
        "mean_width": float(widths.mean()),
        "median_width": float(np.median(widths)),
        "p10_width": float(np.quantile(widths, 0.10)),
        "p90_width": float(np.quantile(widths, 0.90)),
        "adaptivity_ratio": float(np.quantile(widths, 0.90) / np.quantile(widths, 0.10))
            if np.quantile(widths, 0.10) > 1e-8 else 0.0,
        "n": int(len(y)),
    }
