"""Weighted conformal prediction for covariate shift.

Standard split-conformal (see ``src/phase4/conformal.py``) guarantees
marginal coverage *only when the test distribution matches the calibration
distribution*. This project trains on a mixture of NASA and Mendeley
cycles, so a test battery drawn from a different source violates
exchangeability and coverage silently drifts.

Tibshirani, Foygel Barber, Candes, Ramdas (NeurIPS 2019) showed that
conformal prediction can be repaired under covariate shift if you know
the likelihood-ratio (LR)

    w(x) = p_test(x) / p_train(x)

The corrected quantile weights each calibration residual by its LR
(normalised by the sum over calibration + the test point's weight), so
the resulting interval has the right coverage on the shifted test
distribution.

Estimating w(x) in closed form is usually hopeless; the standard trick
(Sugiyama et al. 2007) is to train a classifier to separate train from
test and derive

    w(x) = p(test | x) / p(train | x)
         = prob_test(x) / (1 - prob_test(x))

which is exactly the source classifier built for the leakage probe in
``src/phase5/leakage.py`` - so we get covariate-shift-aware conformal
intervals as a direct byproduct of the leakage diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Likelihood-ratio estimation
# --------------------------------------------------------------------------- #


def estimate_likelihood_ratio(
    x_source: np.ndarray,
    x_target: np.ndarray,
    clip: float = 10.0,
    random_state: int = 42,
) -> Callable[[np.ndarray], np.ndarray]:
    """Fit a logistic classifier and return a callable w(x) = p(target|x)/p(source|x).

    The classifier is trained on features only; it never sees labels.
    Clipping the ratio at ``clip`` prevents a few extreme points from
    dominating the quantile.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    X = np.vstack([np.asarray(x_source), np.asarray(x_target)]).astype(float)
    y = np.concatenate([np.zeros(len(x_source)), np.ones(len(x_target))])
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=random_state)),
    ])
    pipe.fit(X, y)

    def w(x: np.ndarray) -> np.ndarray:
        p_target = pipe.predict_proba(np.asarray(x, dtype=float))[:, 1]
        p_target = np.clip(p_target, 1e-6, 1 - 1e-6)
        ratio = p_target / (1.0 - p_target)
        # re-balance because p_target's prior is n_target / (n_source + n_target)
        prior_ratio = len(x_source) / max(len(x_target), 1)
        ratio = ratio * prior_ratio
        return np.clip(ratio, 1e-3, clip)

    return w


# --------------------------------------------------------------------------- #
# Weighted conformal quantile
# --------------------------------------------------------------------------- #


def weighted_quantile(values: np.ndarray, weights: np.ndarray, level: float) -> float:
    """Empirical (1-alpha) quantile weighted by ``weights`` (unnormalised)."""
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if v.shape != w.shape:
        raise ValueError("values and weights must have the same shape")
    w = np.clip(w, 0.0, None)
    total = w.sum()
    if total <= 0:
        return float(np.quantile(v, level, method="higher"))
    order = np.argsort(v)
    v, w = v[order], w[order]
    cdf = np.cumsum(w) / total
    idx = int(np.searchsorted(cdf, level, side="left"))
    idx = min(idx, len(v) - 1)
    return float(v[idx])


@dataclass
class WeightedConformalResult:
    lower: np.ndarray
    upper: np.ndarray
    q_hat: np.ndarray
    alpha: float


class WeightedConformalRegressor:
    """Split-conformal with Tibshirani-et-al weights per test point.

    Each test point x gets its own q_hat(x) because the likelihood ratio
    changes with x. This makes predictions adaptively wider in regions
    where the test distribution over-samples relative to training.
    """

    def __init__(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        lr_fn: Callable[[np.ndarray], np.ndarray],
        alpha: float = 0.1,
    ):
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        self.predict_fn = predict_fn
        self.lr_fn = lr_fn
        self.alpha = alpha
        self.calib_residuals: Optional[np.ndarray] = None
        self.calib_weights: Optional[np.ndarray] = None

    def calibrate(self, x_calib: np.ndarray, y_calib: np.ndarray) -> "WeightedConformalRegressor":
        preds = np.asarray(self.predict_fn(x_calib)).reshape(-1)
        y_calib = np.asarray(y_calib).reshape(-1)
        self.calib_residuals = np.abs(y_calib - preds)
        self.calib_weights = self.lr_fn(x_calib).reshape(-1)
        return self

    def predict_interval(self, x: np.ndarray) -> WeightedConformalResult:
        if self.calib_residuals is None:
            raise RuntimeError("Call .calibrate(...) before .predict_interval(...)")
        preds = np.asarray(self.predict_fn(x)).reshape(-1)
        w_x = self.lr_fn(x).reshape(-1)
        n = len(self.calib_residuals)
        level = min(1.0, (np.ceil((n + 1) * (1 - self.alpha)) / n))
        # Use a finite sentinel at the max calibration residual (padded 10%) so
        # the test point always has *some* value it can land on when its
        # weight dominates. Tibshirani et al. use +inf; we cap for numerical
        # sanity and so the reported interval stays interpretable.
        sentinel = float(self.calib_residuals.max()) * 1.1
        q_hat = np.empty(len(preds), dtype=float)
        for i, wx in enumerate(w_x):
            weights = np.concatenate([self.calib_weights, [wx]])
            vals = np.concatenate([self.calib_residuals, [sentinel]])
            q_hat[i] = weighted_quantile(vals, weights, level)
        return WeightedConformalResult(
            lower=preds - q_hat,
            upper=preds + q_hat,
            q_hat=q_hat,
            alpha=self.alpha,
        )


def coverage(result: WeightedConformalResult, y_true: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).reshape(-1)
    inside = (y_true >= result.lower) & (y_true <= result.upper)
    return {
        "target": 1 - result.alpha,
        "empirical": float(inside.mean()),
        "mean_width": float((result.upper - result.lower).mean()),
        "median_width": float(np.median(result.upper - result.lower)),
        "n": int(len(y_true)),
    }


# --------------------------------------------------------------------------- #
# Project demo: split NASA vs Mendeley as source / target
# --------------------------------------------------------------------------- #


def _default_root() -> Path:
    return Path(__file__).resolve().parents[2]


def demo_nasa_to_mendeley(project_root: Optional[Path] = None) -> Dict:
    """Calibrate on NASA cycles, predict on Mendeley; compare plain vs weighted conformal."""
    import json
    import pickle
    import pandas as pd

    from src.phase4.conformal import SplitConformalRegressor, empirical_coverage

    project_root = Path(project_root or _default_root())

    df = pd.read_csv(project_root / "data" / "phase3" / "combined_nasa_mendeley.csv")
    if "source" not in df.columns:
        from src.phase5.leakage import infer_source
        df["source"] = df["battery_id"].map(infer_source)

    with open(project_root / "models" / "simple_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(project_root / "models" / "simple_feature_cols.pkl", "rb") as f:
        cols = list(pickle.load(f))
    with open(project_root / "models" / "simple_model.pkl", "rb") as f:
        model = pickle.load(f)

    df = df.dropna(subset=cols + ["soh"])
    X_all = scaler.transform(df[cols].to_numpy(dtype=float))
    y_all = df["soh"].to_numpy(dtype=float)
    src = df["source"].to_numpy()

    nasa_mask = (src == "NASA")
    mend_mask = (src == "MENDELEY")
    if nasa_mask.sum() < 40 or mend_mask.sum() < 40:
        raise RuntimeError(f"Need >=40 cycles per source; got NASA={nasa_mask.sum()} MENDELEY={mend_mask.sum()}")

    rng = np.random.default_rng(42)
    nasa_idx = np.where(nasa_mask)[0]
    rng.shuffle(nasa_idx)
    # 60% NASA -> train/calibrate, 40% Mendeley -> target test
    n_calib = int(0.6 * len(nasa_idx))
    calib_idx = nasa_idx[:n_calib]

    X_calib, y_calib = X_all[calib_idx], y_all[calib_idx]
    X_target, y_target = X_all[mend_mask], y_all[mend_mask]

    def predict(x):
        return model.predict(x)

    out: Dict[str, Dict] = {}

    # ---- Plain split-conformal
    plain = SplitConformalRegressor(predict_fn=predict, alpha=0.1).calibrate(X_calib, y_calib)
    plain_int = plain.predict_interval(X_target)
    out["plain_conformal"] = empirical_coverage(plain_int, y_target)

    # ---- Weighted conformal
    lr = estimate_likelihood_ratio(X_calib, X_target)
    weighted = WeightedConformalRegressor(predict_fn=predict, lr_fn=lr, alpha=0.1).calibrate(X_calib, y_calib)
    weighted_int = weighted.predict_interval(X_target)
    out["weighted_conformal"] = coverage(weighted_int, y_target)

    # ---- LR sanity
    out["likelihood_ratio_stats"] = {
        "mean": float(lr(X_target).mean()),
        "median": float(np.median(lr(X_target))),
        "max": float(lr(X_target).max()),
        "min": float(lr(X_target).min()),
    }
    out["setup"] = {
        "calibration_source": "NASA (60% of NASA cycles)",
        "test_target": "all MENDELEY cycles",
        "alpha": 0.1,
        "n_calib": int(len(X_calib)),
        "n_target": int(len(X_target)),
    }

    out_dir = project_root / "results" / "phase5"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "weighted_conformal.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _format(out: Dict) -> str:
    lines = ["Weighted conformal under covariate shift", "=" * 60,
             f"Setup          : {out['setup']}",
             "",
             "Plain split-conformal:",
             f"  target={out['plain_conformal']['target_coverage']}  "
             f"empirical={out['plain_conformal']['empirical_coverage']:.3f}  "
             f"width={out['plain_conformal']['mean_width']:.3f}",
             "",
             "Weighted (LR-corrected) conformal:",
             f"  target={out['weighted_conformal']['target']}  "
             f"empirical={out['weighted_conformal']['empirical']:.3f}  "
             f"width={out['weighted_conformal']['mean_width']:.3f}",
             "",
             "Likelihood ratio diagnostics:",
             f"  mean={out['likelihood_ratio_stats']['mean']:.2f}  "
             f"median={out['likelihood_ratio_stats']['median']:.2f}  "
             f"min={out['likelihood_ratio_stats']['min']:.3f}  "
             f"max={out['likelihood_ratio_stats']['max']:.2f}",
             "",
             "Interpretation:",
             "  * If plain < target and weighted ~= target, LR reweighting closed the gap.",
             "  * If weighted is wider than plain, the method correctly flags under-coverage",
             "    the naive method was hiding."]
    return "\n".join(lines)


def main() -> int:
    out = demo_nasa_to_mendeley()
    print(_format(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
