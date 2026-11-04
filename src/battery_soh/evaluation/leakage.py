"""Target-leakage diagnostics for a feature set.

The v1 model predicted ``SOH = 100 * capacity / C0`` from inputs that
included ``capacity`` itself and full-discharge statistics that rebuild it
(``current_mean * record_time`` is a coulomb count, within-battery r >= 0.9997
with capacity). This module makes that failure detectable by construction:

1. **Registry / label-side rule** - a feature whose registry entry is
   ``leakage_risk == "label"`` (or ``"proxy"``, unless explicitly allowed) is
   flagged no matter what the data say; so is a column that *is* the label,
   its numerator or another label-side column of the feature table
   (:data:`LABEL_SIDE_COLUMNS`).
2. **Within-battery correlation rule** - ``|r| >= 0.9995`` between a feature and
   the label inside any single battery. Within a battery ``C0`` is a
   constant, so the label is an affine function of capacity and any exact
   capacity reconstruction shows up as ``|r| ~ 1``.
3. **Reconstruction rule** - leave-one-battery-out R² of a near-OLS model
   rebuilding the label numerator (``capacity_ah``) from the feature(s)
   ``>= 0.995``, both per feature and for the whole set. The R² is the better
   of a linear fit and a log-linear fit (``log`` of strictly positive
   columns and of the target), so multiplicative proxies such as
   ``mean current * duration`` are caught even when supplied as two columns.

Why these thresholds: a *health indicator* is supposed to track capacity -
that is its job - so high correlation alone is not leakage. The thresholds
target near-exact reconstructions. Measured on the four core NASA cells
(B0005/6/7/18) in ``data/processed/nasa_features.csv``: the strongest legitimate
partial-window indicator (Ah delivered between 3.80 and 3.50 V) reaches
within-battery ``|r|`` 0.998 with SOH on all QC-valid rows and up to 0.9992 on
smaller row subsets, and no partial-window set rebuilds capacity with LOBO R²
above 0.95. The v1 coulomb-count proxy reaches ``|r|`` >= 0.9997 and LOBO R²
0.998, the v1 full-record set without capacity 0.995, and capacity itself 1.0.
The correlation margin between a strong health indicator and an exact proxy is
thin (0.9992 vs 0.9997), which is why the reconstruction rule (3) is the primary
guard and the correlation rule only catches near-identities.

The checks need the label and, for rule 3, the capacity column - they are
offline audit tools for training data, not something to run at inference.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from battery_soh._typing import FloatArray
from battery_soh.evaluation.errors import LeakageError
from battery_soh.evaluation.splits import leave_one_group_out
from battery_soh.features.registry import FEATURES

__all__ = [
    "LABEL_CORRELATION_THRESHOLD",
    "LABEL_SIDE_COLUMNS",
    "MIN_GROUP_ROWS",
    "RECONSTRUCTION_R2_THRESHOLD",
    "SET_ROW",
    "LeakageError",
    "check_no_leakage",
    "leakage_report",
    "lobo_r2",
    "reconstruction_r2",
]

logger = logging.getLogger(__name__)


#: Within-battery |pearson r| with the label at or above which a feature is flagged.
LABEL_CORRELATION_THRESHOLD = 0.9995
#: LOBO R² of rebuilding the label numerator at or above which features are flagged.
RECONSTRUCTION_R2_THRESHOLD = 0.995
#: Batteries with fewer paired observations are ignored by the correlation rule
#: (with 2 points every correlation is +/-1).
MIN_GROUP_ROWS = 10
#: ``feature`` value of the report row that describes the whole feature set.
SET_ROW = "(feature set)"
#: Feature-table columns derived from the measured capacity (label side). Never inputs.
LABEL_SIDE_COLUMNS: frozenset[str] = frozenset({"capacity_ah", "c0_ah", "soh_pct", "soh_rated_pct"})

_RISK_ORDER = {"none": 0, "unregistered": 1, "proxy": 2, "label": 3}
# Ridge penalty of the reconstruction fit (on standardised inputs): just enough
# to keep collinear designs solvable. The question is whether the features
# *contain* the target, not whether a regularised model generalises well.
_RECONSTRUCTION_RIDGE = 1e-6


def _fit_predict(X_train: FloatArray, y_train: FloatArray, X_test: FloatArray) -> FloatArray:
    """Median-impute, standardise, fit a near-OLS ridge on the training rows; predict ``X_test``.

    Imputation and scaling statistics come from the training rows only.
    """
    observed = np.isfinite(X_train)
    median = np.array(
        [
            float(np.median(col[ok])) if ok.any() else 0.0
            for col, ok in zip(X_train.T, observed.T, strict=True)
        ],
        dtype=np.float64,
    )
    Xtr = np.where(observed, X_train, median)
    Xte = np.where(np.isfinite(X_test), X_test, median)
    mean = Xtr.mean(axis=0)
    scale = Xtr.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    Ztr = (Xtr - mean) / scale
    Zte = (Xte - mean) / scale
    y_mean = float(y_train.mean())
    gram = Ztr.T @ Ztr + _RECONSTRUCTION_RIDGE * np.eye(Ztr.shape[1])
    coef = np.linalg.lstsq(gram, Ztr.T @ (y_train - y_mean), rcond=None)[0]
    return np.asarray(Zte @ coef + y_mean, dtype=np.float64)


def _log_design(X: FloatArray) -> FloatArray:
    """Log-transform the columns whose observed values are all strictly positive."""
    out = X.copy()
    for j in range(X.shape[1]):
        col = X[:, j]
        observed = np.isfinite(col)
        if observed.any() and bool(np.all(col[observed] > 0)):
            out[observed, j] = np.log(col[observed])
    return out


def lobo_r2(
    X: FloatArray, y: FloatArray, groups: NDArray[np.object_], *, log: bool = False
) -> float:
    """Pooled out-of-fold R² of a near-OLS linear model under leave-one-group-out.

    Args:
        X: Features, shape ``(n,)`` or ``(n, p)``; NaN is median-imputed per fold.
        y: Target, shape ``(n,)``; rows with a non-finite target are dropped.
        groups: Group label per row.
        log: Fit in log space: strictly positive feature columns and the
            target are log-transformed and predictions are exponentiated
            back, so R² is still measured on the original scale.

    Returns:
        ``1 - SS_res / SS_tot`` on the original scale, or NaN with fewer than
        two groups, fewer than three rows, a constant target, or (``log``) a
        target that is not strictly positive.
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    X = np.asarray(X, dtype=np.float64).reshape(y.size, -1)
    keep = np.isfinite(y)
    X, y, groups = X[keep], y[keep], np.asarray(groups, dtype=object)[keep]
    if len(pd.unique(groups)) < 2 or y.size < 3:
        return math.nan
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot == 0.0:
        return math.nan
    if log and np.any(y <= 0):
        return math.nan
    design = _log_design(X) if log else X
    target = np.log(y) if log else y
    oof = np.empty_like(y)
    for train_idx, test_idx, _ in leave_one_group_out(groups):
        oof[test_idx] = _fit_predict(design[train_idx], target[train_idx], design[test_idx])
    if log:
        with np.errstate(over="ignore"):
            oof = np.exp(oof)
    if not np.all(np.isfinite(oof)):
        return math.nan
    return 1.0 - float(np.sum((y - oof) ** 2)) / ss_tot


def reconstruction_r2(X: FloatArray, y: FloatArray, groups: NDArray[np.object_]) -> float:
    """Best LOBO R² of rebuilding ``y`` from ``X``: the max of the linear and log-linear fits.

    Undefined fits (NaN) are ignored; NaN if neither is defined.
    """
    values = [lobo_r2(X, y, groups), lobo_r2(X, y, groups, log=True)]
    finite = [v for v in values if not math.isnan(v)]
    return max(finite) if finite else math.nan


def _max_within_group_abs_r(
    x: FloatArray, y: FloatArray, groups: NDArray[np.object_], min_rows: int
) -> float:
    best = math.nan
    for label in pd.unique(groups):
        mask = (groups == label) & np.isfinite(x) & np.isfinite(y)
        if mask.sum() < min_rows:
            continue
        xs, ys = x[mask], y[mask]
        if np.ptp(xs) == 0.0 or np.ptp(ys) == 0.0:
            continue
        r = abs(float(np.corrcoef(xs, ys)[0, 1]))
        best = r if math.isnan(best) else max(best, r)
    return best


def _registry_risk(feature: str) -> tuple[str, str]:
    spec = FEATURES.get(feature)
    if spec is None:
        return "unregistered", ""
    return spec.leakage_risk, spec.availability


def _label_side_reason(feature: str, label: str, capacity_col: str) -> str | None:
    if feature == label:
        return "feature is the label"
    if feature == capacity_col:
        return "feature is the label numerator"
    if feature in LABEL_SIDE_COLUMNS:
        return "feature is a label-side column of the feature table"
    return None


def leakage_report(
    df: pd.DataFrame,
    features: Sequence[str],
    label: str = "soh_pct",
    group_col: str = "battery_id",
    *,
    capacity_col: str = "capacity_ah",
    r_threshold: float = LABEL_CORRELATION_THRESHOLD,
    reconstruction_threshold: float = RECONSTRUCTION_R2_THRESHOLD,
    allow_proxy: bool = False,
    min_group_rows: int = MIN_GROUP_ROWS,
) -> pd.DataFrame:
    """Per-feature leakage diagnostics plus one row for the whole set.

    Args:
        df: Feature table (one row per cycle) with the features, the label,
            the group column and, ideally, ``capacity_col``. Filter QC-invalid
            rows beforehand if they should not count.
        features: Feature columns to audit.
        label: Label column.
        group_col: Battery id column.
        capacity_col: Label numerator; if absent, the reconstruction columns are NaN.
        r_threshold: Correlation-rule threshold.
        reconstruction_threshold: Reconstruction-rule threshold.
        allow_proxy: Do not flag features the registry marks as ``"proxy"``.
        min_group_rows: Minimum paired rows for a battery to enter the correlation rule.

    Returns:
        DataFrame with columns ``feature, registry_risk, availability, n,
        max_within_group_abs_r, lobo_r2, capacity_reconstruction_r2, flagged,
        reasons``. ``lobo_r2`` is the (linear) LOBO R² of predicting the label
        from the row's feature(s) - univariate for feature rows, multivariate
        for the final :data:`SET_ROW` row; ``capacity_reconstruction_r2`` is
        :func:`reconstruction_r2` of ``capacity_col`` likewise.

    Raises:
        ValueError: If columns are missing, ``features`` is empty or has duplicates.
    """
    features = list(features)
    if not features:
        raise ValueError("features must not be empty")
    if len(set(features)) != len(features):
        raise ValueError("features must be unique")
    required = [*features, label, group_col]
    missing = [c for c in dict.fromkeys(required) if c not in df.columns]
    if missing:
        raise ValueError(f"columns {missing} not found in table")
    data = df[df[label].notna() & df[group_col].notna()]
    groups = data[group_col].to_numpy().astype(object)
    y = data[label].to_numpy(dtype=np.float64)
    has_capacity = capacity_col in data.columns
    if not has_capacity:
        logger.warning("column %r not found; capacity reconstruction check skipped", capacity_col)
    cap = data[capacity_col].to_numpy(dtype=np.float64) if has_capacity else None
    X = data[features].to_numpy(dtype=np.float64)

    rows: list[dict[str, object]] = []
    correlations: list[float] = []
    risks: list[str] = []
    for j, feature in enumerate(features):
        risk, availability = _registry_risk(feature)
        x = X[:, j]
        r = _max_within_group_abs_r(x, y, groups, min_group_rows)
        correlations.append(r)
        risks.append(risk)
        cap_r2 = reconstruction_r2(x, cap, groups) if cap is not None else math.nan
        reasons: list[str] = []
        side = _label_side_reason(feature, label, capacity_col)
        if side:
            reasons.append(side)
        if risk == "label":
            reasons.append("registry: label-side feature")
        elif risk == "proxy" and not allow_proxy:
            reasons.append("registry: label proxy (full-discharge statistic)")
        if r >= r_threshold:
            reasons.append(f"within-battery |r| with {label} = {r:.5f} >= {r_threshold}")
        if cap_r2 >= reconstruction_threshold:
            reasons.append(
                f"rebuilds {capacity_col} alone "
                f"(LOBO R2 = {cap_r2:.5f} >= {reconstruction_threshold})"
            )
        rows.append(
            {
                "feature": feature,
                "registry_risk": risk,
                "availability": availability,
                "n": int(np.isfinite(x).sum()),
                "max_within_group_abs_r": r,
                "lobo_r2": lobo_r2(x, y, groups),
                "capacity_reconstruction_r2": cap_r2,
                "flagged": bool(reasons),
                "reasons": "; ".join(reasons),
            }
        )

    set_cap_r2 = reconstruction_r2(X, cap, groups) if cap is not None else math.nan
    set_reasons = []
    if set_cap_r2 >= reconstruction_threshold:
        set_reasons.append(
            f"feature set rebuilds {capacity_col} (LOBO R2 = {set_cap_r2:.5f} "
            f">= {reconstruction_threshold})"
        )
    finite_r = [r for r in correlations if not math.isnan(r)]
    rows.append(
        {
            "feature": SET_ROW,
            "registry_risk": max(risks, key=_RISK_ORDER.__getitem__),
            "availability": "",
            "n": len(data),
            "max_within_group_abs_r": max(finite_r) if finite_r else math.nan,
            "lobo_r2": lobo_r2(X, y, groups),
            "capacity_reconstruction_r2": set_cap_r2,
            "flagged": bool(set_reasons),
            "reasons": "; ".join(set_reasons),
        }
    )
    return pd.DataFrame(rows)


def check_no_leakage(
    df: pd.DataFrame,
    features: Sequence[str],
    label: str = "soh_pct",
    group_col: str = "battery_id",
    *,
    capacity_col: str = "capacity_ah",
    r_threshold: float = LABEL_CORRELATION_THRESHOLD,
    reconstruction_threshold: float = RECONSTRUCTION_R2_THRESHOLD,
    allow_proxy: bool = False,
    min_group_rows: int = MIN_GROUP_ROWS,
) -> pd.DataFrame:
    """Raise :class:`LeakageError` if :func:`leakage_report` flags anything.

    Same arguments as :func:`leakage_report`.

    Returns:
        The (clean) report, for logging or persisting alongside results.

    Raises:
        LeakageError: Listing every flagged feature and the rule(s) it broke.
    """
    report = leakage_report(
        df,
        features,
        label,
        group_col,
        capacity_col=capacity_col,
        r_threshold=r_threshold,
        reconstruction_threshold=reconstruction_threshold,
        allow_proxy=allow_proxy,
        min_group_rows=min_group_rows,
    )
    flagged = report[report["flagged"]]
    if not flagged.empty:
        details = "\n".join(f"  - {row.feature}: {row.reasons}" for row in flagged.itertuples())
        raise LeakageError(f"target leakage detected for label {label!r}:\n{details}")
    logger.info("leakage check passed for %d feature(s) on label %s", len(features), label)
    return report
