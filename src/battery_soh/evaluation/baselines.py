"""Reference baselines every reported model must beat (or, for the oracle, explain).

* ``mean`` - predict the training-label mean. Floor of usefulness.
* ``cycle_only`` - Ridge on the cycle count alone. A model that cannot beat
  "older cells are less healthy" has not learned anything from its health
  indicators.
* ``capacity_oracle`` - Ridge on the measured full-discharge capacity, i.e.
  on the label's own numerator. **LEAKY - sanity check only.** It is not even
  perfect under leave-one-battery-out, because ``SOH = 100 * capacity / C0``
  and the held-out cell's ``C0`` is unknown; that gap is exactly why v1's
  near-perfect numbers came from within-battery evaluation.

Baselines are sklearn-compatible estimators. Feature selection is the
caller's job: :func:`baseline_features` says which feature columns a baseline
expects (``None`` = it ignores its inputs).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted, validate_data

from battery_soh._typing import FloatArray
from battery_soh.config import RANDOM_SEED
from battery_soh.features.registry import FEATURE_SETS
from battery_soh.models.zoo import make_model

__all__ = [
    "BASELINES",
    "BASELINE_NAMES",
    "BaselineSpec",
    "MeanBaseline",
    "baseline_features",
    "make_baseline",
    "make_capacity_oracle",
    "make_cycle_only",
]


class MeanBaseline(RegressorMixin, BaseEstimator):  # type: ignore[misc]
    """Predict the mean training label for every input.

    Inputs are validated for shape only; missing values are allowed because
    the baseline never looks at them.
    """

    def fit(self, X: ArrayLike, y: ArrayLike) -> MeanBaseline:
        """Store the mean of ``y``."""
        _, y_arr = validate_data(self, X, y, y_numeric=True, ensure_all_finite="allow-nan")
        self.mean_ = float(np.mean(y_arr))
        return self

    def predict(self, X: ArrayLike) -> FloatArray:
        """Return the stored mean, once per row of ``X``."""
        check_is_fitted(self, "mean_")
        X_arr = validate_data(self, X, reset=False, ensure_all_finite="allow-nan")
        return np.full(X_arr.shape[0], self.mean_, dtype=np.float64)

    def __sklearn_tags__(self) -> Any:
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.regressor_tags.poor_score = True  # a constant cannot reach R2 > 0.5
        return tags


def make_cycle_only(*, random_state: int = RANDOM_SEED) -> Pipeline:
    """Ridge on ``ctx_cycle_index`` only (feed it the ``cycle_only`` feature set)."""
    return make_model("ridge", random_state=random_state)


def make_capacity_oracle(*, random_state: int = RANDOM_SEED) -> Pipeline:
    """LEAKY - sanity check only: Ridge on the label numerator ``leg_capacity_ah``."""
    return make_model("ridge", random_state=random_state)


@dataclass(frozen=True)
class BaselineSpec:
    """Description of a baseline.

    Attributes:
        name: Registry key.
        label: Display name; leaky baselines are labelled as such.
        feature_set: Name of the :data:`FEATURE_SETS` entry it consumes, or
            ``None`` if it ignores its inputs.
        leaky: True if it uses label-side information (never a real model).
        description: One-line explanation (ASCII).
    """

    name: str
    label: str
    feature_set: str | None
    leaky: bool
    description: str


_BASELINES: tuple[BaselineSpec, ...] = (
    BaselineSpec(
        "mean",
        "Training mean",
        None,
        leaky=False,
        description="Predicts the mean SOH of the training batteries.",
    ),
    BaselineSpec(
        "cycle_only",
        "Cycle count only",
        "cycle_only",
        leaky=False,
        description="Ridge on the number of completed discharge cycles.",
    ),
    BaselineSpec(
        "capacity_oracle",
        "Capacity oracle (LEAKY - sanity check only)",
        "capacity_oracle",
        leaky=True,
        description=(
            "Ridge on the measured full-discharge capacity, the label numerator. "
            "Not a valid model; shows the cost of the unknown per-cell C0."
        ),
    ),
)

#: Baseline registry keyed by name.
BASELINES: Mapping[str, BaselineSpec] = MappingProxyType({b.name: b for b in _BASELINES})

#: Names accepted by :func:`make_baseline`.
BASELINE_NAMES: tuple[str, ...] = tuple(b.name for b in _BASELINES)


def _spec(name: str) -> BaselineSpec:
    try:
        return BASELINES[name]
    except KeyError:
        raise ValueError(
            f"Unknown baseline {name!r}; available: {', '.join(BASELINE_NAMES)}"
        ) from None


def make_baseline(name: str, *, random_state: int = RANDOM_SEED) -> Any:
    """Build an unfitted baseline estimator.

    Args:
        name: One of :data:`BASELINE_NAMES`.
        random_state: Seed (kept for a uniform interface; baselines are deterministic).

    Raises:
        ValueError: If ``name`` is unknown.
    """
    spec = _spec(name)
    if spec.name == "mean":
        return MeanBaseline()
    if spec.name == "cycle_only":
        return make_cycle_only(random_state=random_state)
    return make_capacity_oracle(random_state=random_state)


def baseline_features(name: str) -> tuple[str, ...] | None:
    """Feature columns a baseline must be fed, or ``None`` if it ignores its inputs.

    Raises:
        ValueError: If ``name`` is unknown.
    """
    spec = _spec(name)
    return None if spec.feature_set is None else FEATURE_SETS[spec.feature_set]
