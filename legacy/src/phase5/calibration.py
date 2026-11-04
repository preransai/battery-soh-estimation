"""Calibration metrics for SOH regression.

Point-error metrics (MAE, RMSE, R^2) only tell us how close the predicted
SOH is to the truth. For an operational tool, we also need to know whether
the model's uncertainty is *honest*: when it says "I'm 90% confident the
SOH is between 82% and 88%", how often is the truth actually in that
range?

This module reports:

    ECE         Expected Calibration Error (Naeini et al. 2015)
    MCE         Maximum Calibration Error (worst-bin gap)
    sharpness   Mean interval width (lower = more useful when calibrated)
    reliability Per-bin observed-vs-expected coverage curve

For regression with a bootstrap ensemble (you have 20 Ridge bootstraps),
we build intervals at a sweep of nominal levels and compare the fraction
of truth that falls inside at each level. A perfectly calibrated model
traces the diagonal of the reliability diagram.

A conformal predictor should be calibrated by construction *on average*,
but not necessarily per-bin or per-group. We re-check that here on the
project's held-out test set.

References
----------
Kuleshov, Fenner, Ermon (2018) "Accurate Uncertainties for Deep Learning
    Using Calibrated Regression." ICML.
Naeini, Cooper, Hauskrecht (2015) "Obtaining well calibrated probabilities
    using Bayesian binning." AAAI.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Core metrics
# --------------------------------------------------------------------------- #


@dataclass
class ReliabilityBin:
    nominal: float      # target coverage for this bin, e.g. 0.8
    empirical: float    # actual fraction of y_true inside the band
    mean_width: float
    count: int


@dataclass
class CalibrationReport:
    ece: float
    mce: float
    sharpness: float         # mean interval width across nominal levels
    under_coverage: float    # max(nominal - empirical)
    over_coverage: float     # max(empirical - nominal)
    bins: List[ReliabilityBin] = field(default_factory=list)
    picp: Optional[float] = None  # coverage at a specific nominal (e.g. 0.9)
    rmse: Optional[float] = None

    def to_dict(self) -> Dict:
        return {
            "ece": round(self.ece, 4),
            "mce": round(self.mce, 4),
            "sharpness": round(self.sharpness, 4),
            "under_coverage": round(self.under_coverage, 4),
            "over_coverage": round(self.over_coverage, 4),
            "picp_at_90": round(self.picp, 4) if self.picp is not None else None,
            "rmse": round(self.rmse, 4) if self.rmse is not None else None,
            "bins": [
                {
                    "nominal": round(b.nominal, 4),
                    "empirical": round(b.empirical, 4),
                    "mean_width": round(b.mean_width, 4),
                    "count": b.count,
                }
                for b in self.bins
            ],
        }


def reliability_curve(
    y_true: np.ndarray,
    lower_fn: Callable[[float], np.ndarray],
    upper_fn: Callable[[float], np.ndarray],
    nominals: Sequence[float] = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99),
) -> List[ReliabilityBin]:
    """Compute reliability bins for a set of nominal coverage levels.

    ``lower_fn(alpha)`` and ``upper_fn(alpha)`` must return the (1-alpha)
    prediction interval bounds on the held-out set.
    """
    y_true = np.asarray(y_true).reshape(-1)
    bins: List[ReliabilityBin] = []
    for nominal in nominals:
        alpha = 1.0 - nominal
        lo = np.asarray(lower_fn(alpha)).reshape(-1)
        hi = np.asarray(upper_fn(alpha)).reshape(-1)
        inside = (y_true >= lo) & (y_true <= hi)
        bins.append(
            ReliabilityBin(
                nominal=float(nominal),
                empirical=float(inside.mean()),
                mean_width=float((hi - lo).mean()),
                count=int(len(y_true)),
            )
        )
    return bins


def _summary_from_bins(bins: Sequence[ReliabilityBin]) -> Tuple[float, float, float, float, float]:
    gaps = np.array([abs(b.empirical - b.nominal) for b in bins])
    widths = np.array([b.mean_width for b in bins])
    ece = float(gaps.mean()) if len(gaps) else 0.0
    mce = float(gaps.max()) if len(gaps) else 0.0
    sharpness = float(widths.mean()) if len(widths) else 0.0
    under = float(max((b.nominal - b.empirical) for b in bins))
    over = float(max((b.empirical - b.nominal) for b in bins))
    return ece, mce, sharpness, under, over


def calibration_report(
    y_true: np.ndarray,
    lower_fn: Callable[[float], np.ndarray],
    upper_fn: Callable[[float], np.ndarray],
    nominals: Sequence[float] = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99),
    y_pred: Optional[np.ndarray] = None,
) -> CalibrationReport:
    bins = reliability_curve(y_true, lower_fn, upper_fn, nominals=nominals)
    ece, mce, sharp, under, over = _summary_from_bins(bins)

    picp = None
    if 0.9 in nominals:
        picp = next(b.empirical for b in bins if abs(b.nominal - 0.9) < 1e-9)

    rmse = None
    if y_pred is not None:
        y_pred = np.asarray(y_pred).reshape(-1)
        rmse = float(np.sqrt(((y_pred - y_true) ** 2).mean()))

    return CalibrationReport(
        ece=ece, mce=mce, sharpness=sharp,
        under_coverage=under, over_coverage=over,
        bins=bins, picp=picp, rmse=rmse,
    )


# --------------------------------------------------------------------------- #
# Bootstrap-ensemble calibration
# --------------------------------------------------------------------------- #


def bootstrap_interval_fns(
    predictions_by_model: np.ndarray,
) -> Tuple[np.ndarray, Callable[[float], np.ndarray], Callable[[float], np.ndarray]]:
    """Turn an (n_models, n_samples) prediction matrix into calibration-ready fns.

    Returns
    -------
    y_hat
        Mean prediction (used as the point estimate).
    lower_fn, upper_fn
        Callables that take alpha and return the (alpha/2, 1-alpha/2)
        empirical quantile across the ensemble.
    """
    preds = np.asarray(predictions_by_model)
    if preds.ndim != 2:
        raise ValueError("predictions_by_model must be 2-D (n_models, n_samples)")
    y_hat = preds.mean(axis=0)

    def lower(alpha: float) -> np.ndarray:
        return np.quantile(preds, alpha / 2.0, axis=0)

    def upper(alpha: float) -> np.ndarray:
        return np.quantile(preds, 1.0 - alpha / 2.0, axis=0)

    return y_hat, lower, upper


# --------------------------------------------------------------------------- #
# Project glue - run against the bootstrap ensemble already on disk
# --------------------------------------------------------------------------- #


def _default_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_project_artifacts(project_root: Path):
    """Load both (a) the 19-feature production Ridge with its scaler and
    (b) the 6-feature bootstrap ensemble with its scaler. The two were
    trained on different feature sets in the current project so the
    calibration report needs both pairs to run correctly.
    """
    import pandas as pd
    with open(project_root / "models" / "best_model.pkl", "rb") as f:
        best = pickle.load(f)
    with open(project_root / "models" / "scaler.pkl", "rb") as f:
        scaler_main = pickle.load(f)
    with open(project_root / "models" / "feature_cols.pkl", "rb") as f:
        cols_main = list(pickle.load(f))

    bootstraps = None
    scaler_boot = None
    cols_boot: List[str] = []
    bpath = project_root / "models" / "bootstrap_models.pkl"
    if bpath.exists():
        with open(bpath, "rb") as f:
            bootstraps = pickle.load(f)
        sb = project_root / "models" / "simple_scaler.pkl"
        cb = project_root / "models" / "simple_feature_cols.pkl"
        if sb.exists() and cb.exists():
            with open(sb, "rb") as f:
                scaler_boot = pickle.load(f)
            with open(cb, "rb") as f:
                cols_boot = list(pickle.load(f))

    test = pd.read_csv(project_root / "data" / "phase2" / "harmonized" / "test_features.csv")
    return {
        "best": best,
        "scaler_main": scaler_main,
        "cols_main": cols_main,
        "bootstraps": bootstraps,
        "scaler_boot": scaler_boot,
        "cols_boot": cols_boot,
        "test": test,
    }


def run_project_calibration(
    project_root: Optional[Path] = None,
    nominals: Sequence[float] = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99),
    calibration_seed: int = 42,
) -> Dict:
    """Calibrate intervals with split-conformal on a held-out fold.

    Why this rewrite. The earlier version wrapped the bootstrap-ensemble
    percentile band straight into the reliability check. Bootstrap
    variance captures only model-on-model disagreement and is far
    smaller than residual variance, so empirical coverage at the 90%
    nominal level was a catastrophic 0.08 with ECE 0.705. The fix is
    a textbook split-conformal calibration (Lei et al., 2018): the
    test set is split 50/50 into a calibration fold and a final-test
    fold; we compute absolute residuals on the calibration fold and
    take the empirical (1-alpha) quantile as the interval half-width;
    the final-test fold is then used to measure empirical coverage.
    The marginal-coverage guarantee says PICP at level 1-alpha will
    sit close to 1-alpha on the final-test fold, and ECE drops to a
    small fraction.
    """
    project_root = Path(project_root or _default_root())
    art = load_project_artifacts(project_root)
    test = art["test"]
    y_test_full = test["soh"].to_numpy(dtype=float)

    # ------------------------------------------------------------------ point estimates
    cols_main = [c for c in art["cols_main"] if c in test.columns]
    missing_main = set(art["cols_main"]) - set(test.columns)
    X_main = art["scaler_main"].transform(
        test[cols_main].to_numpy(dtype=float, na_value=0.0)
    ) if len(cols_main) == len(art["cols_main"]) else None

    y_hat_main = None
    if X_main is not None and X_main.shape[1] == art["best"].n_features_in_:
        y_hat_main = np.asarray(art["best"].predict(X_main))

    # ------------------------------------------------------------------ pick the predictor we will calibrate
    if y_hat_main is not None:
        y_hat = y_hat_main
        mode_label = "split_conformal_main_ridge"
        n_features_used = int(X_main.shape[1])
    elif art["bootstraps"] is not None and art["scaler_boot"] is not None:
        cols_boot = [c for c in art["cols_boot"] if c in test.columns]
        X_boot = art["scaler_boot"].transform(
            test[cols_boot].to_numpy(dtype=float, na_value=0.0)
        )
        preds = np.stack([np.asarray(m.predict(X_boot)) for m in art["bootstraps"]])
        y_hat = preds.mean(axis=0)
        mode_label = "split_conformal_bootstrap_mean"
        n_features_used = int(X_boot.shape[1])
    else:
        raise RuntimeError("No usable predictor for calibration")

    # ------------------------------------------------------------------ split test into calibration / final-test
    rng = np.random.default_rng(calibration_seed)
    n = len(y_test_full)
    perm = rng.permutation(n)
    cut = n // 2
    cal_idx, fin_idx = perm[:cut], perm[cut:]

    y_cal, y_fin = y_test_full[cal_idx], y_test_full[fin_idx]
    yhat_cal, yhat_fin = y_hat[cal_idx], y_hat[fin_idx]

    # absolute residuals on the calibration fold
    abs_res = np.abs(y_cal - yhat_cal)

    # ------------------------------------------------------------------ split-conformal interval builders
    # q_alpha is the empirical (1-alpha) quantile of |y_cal - yhat_cal|.
    # Using the standard Romano/Lei finite-sample correction:
    #     q = Quantile_{ceil((n_cal+1)(1-alpha))/n_cal}(|residual|)
    n_cal = len(abs_res)

    def q_alpha(alpha: float) -> float:
        if n_cal == 0:
            return float("nan")
        order = int(np.ceil((n_cal + 1) * (1.0 - alpha))) - 1
        order = max(0, min(order, n_cal - 1))
        return float(np.sort(abs_res)[order])

    def lower_fn(alpha: float) -> np.ndarray:
        return yhat_fin - q_alpha(alpha)

    def upper_fn(alpha: float) -> np.ndarray:
        return yhat_fin + q_alpha(alpha)

    report = calibration_report(
        y_fin, lower_fn, upper_fn, nominals=nominals, y_pred=yhat_fin,
    )
    out = {
        "mode": mode_label,
        "n_calibration": int(n_cal),
        "n_final_test": int(len(fin_idx)),
        "n_features": n_features_used,
        "missing_main_cols": sorted(missing_main),
        "calibration_seed": calibration_seed,
        **report.to_dict(),
    }

    out_dir = project_root / "results" / "phase5"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "calibration.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


# --------------------------------------------------------------------------- #
# Plotting (optional; only imports matplotlib when called)
# --------------------------------------------------------------------------- #


def save_reliability_diagram(
    report_dict: Dict,
    out_path: Path,
    title: str = "Reliability diagram (SOH regression)",
) -> None:
    import matplotlib.pyplot as plt

    bins = report_dict.get("bins", [])
    if not bins:
        return
    xs = [b["nominal"] for b in bins]
    ys = [b["empirical"] for b in bins]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect calibration")
    ax.plot(xs, ys, "o-", label="empirical")
    for b in bins:
        ax.annotate(
            f"{b['nominal']:.2f}", xy=(b['nominal'], b['empirical']),
            xytext=(5, -10), textcoords="offset points", fontsize=8,
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _format(out: Dict) -> str:
    lines = ["Calibration report", "=" * 60,
             f"Mode      : {out.get('mode', '?')}",
             f"ECE       : {out.get('ece'):.3f}  (0 = perfect)",
             f"MCE       : {out.get('mce'):.3f}",
             f"Sharpness : {out.get('sharpness'):.2f}  (mean interval width, SOH%)",
             f"PICP @ 90 : {out.get('picp_at_90')}",
             f"Under-cov : {out.get('under_coverage'):.3f}  (largest nominal gap)",
             f"Over-cov  : {out.get('over_coverage'):.3f}",
             f"RMSE      : {out.get('rmse')}",
             "",
             "Per-nominal bins:"]
    for b in out.get("bins", []):
        gap = b["empirical"] - b["nominal"]
        flag = "under" if gap < 0 else "over "
        lines.append(
            f"  nominal={b['nominal']:.2f}  empirical={b['empirical']:.2f}  "
            f"width={b['mean_width']:.2f}  {flag} by {gap:+.2f}"
        )
    return "\n".join(lines)


def main() -> int:
    out = run_project_calibration()
    print(_format(out))
    try:
        save_reliability_diagram(
            out,
            _default_root() / "results" / "phase5" / "reliability_diagram.png",
        )
        print("\nSaved reliability diagram -> results/phase5/reliability_diagram.png")
    except Exception as exc:  # pragma: no cover
        print(f"\nPlot skipped: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
