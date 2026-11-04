"""Fit Phase 4 calibration artefacts and dump them as a single pickle.

The API container loads ``models/phase4_state.pkl`` at startup and uses it
to serve the four Tier-A novelty endpoints (conformal, RUL, RAP, info).
Running this script offline keeps the runtime container slim: no torch,
no SBERT, no training data shipped.

Inputs
    - models/simple_model.pkl, simple_scaler.pkl, simple_feature_cols.pkl
      (the existing sklearn predictor exposed by /predict)
    - data/phase2/harmonized/all_features.csv (NASA harmonised corpus)

Outputs
    - models/phase4_state.pkl
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "phase2"))
sys.path.insert(0, str(ROOT / "src" / "phase4"))

from feature_to_text import FeatureToText  # type: ignore
from phase4.conformal import _finite_sample_quantile  # type: ignore
from phase4.survival_head import compute_rul_targets  # type: ignore


SIMPLE_FEATURES = ["capacity", "cycle_number", "voltage_min", "voltage_max", "temp_mean", "discharge_time"]


def load_simple_predictor():
    with open(ROOT / "models" / "simple_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(ROOT / "models" / "simple_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(ROOT / "models" / "simple_feature_cols.pkl", "rb") as f:
        feat_cols = pickle.load(f)
    return model, scaler, feat_cols


def predict_simple(model, scaler, feat_cols, frame: pd.DataFrame) -> np.ndarray:
    X = np.array([[row.get(c, 0.0) for c in feat_cols] for row in frame.to_dict("records")])
    X = np.nan_to_num(X)
    return np.clip(model.predict(scaler.transform(X)), 0.0, 100.0)


# --------------------------------------------------------------------------- #
# Weibull MLE: estimate a global shape parameter from observed RULs.
# --------------------------------------------------------------------------- #


def fit_global_weibull_shape(rul_df: pd.DataFrame) -> float:
    """Maximum-likelihood Weibull shape over all uncensored RUL observations.

    With censoring-aware Weibull MLE (constant scale across the corpus this
    would be implausible, but we use this *only* to set the per-prediction
    shape parameter; the per-prediction scale is computed at runtime from
    the existing RUL point estimate). This avoids shipping torch in the
    runtime container.
    """
    obs = rul_df[(rul_df["event"] == 1) & (rul_df["rul"] > 0)]
    if len(obs) < 10:
        return 1.5  # sensible default for early-life wear-out
    t = obs["rul"].values.astype(float)

    # Estimate scale + shape jointly via Newton on the Weibull MLE equations.
    # Iterating shape (k); scale (lambda) follows in closed form given k.
    k = 1.5
    log_t = np.log(t)
    for _ in range(60):
        t_pow_k = np.power(t, k)
        sum_t_pow_k = t_pow_k.sum()
        sum_t_pow_k_log_t = (t_pow_k * log_t).sum()
        # 1/k + mean(log t) - sum(t^k log t) / sum(t^k) = 0
        f = 1.0 / k + log_t.mean() - sum_t_pow_k_log_t / sum_t_pow_k
        # df/dk
        sum_t_pow_k_log2_t = (t_pow_k * log_t * log_t).sum()
        df = (
            -1.0 / (k * k)
            - (sum_t_pow_k_log2_t * sum_t_pow_k - sum_t_pow_k_log_t ** 2) / (sum_t_pow_k ** 2)
        )
        step = f / df
        k_new = max(1e-3, k - step)
        if abs(k_new - k) < 1e-6:
            k = k_new
            break
        k = k_new
    return float(np.clip(k, 0.5, 10.0))


# --------------------------------------------------------------------------- #
# Conformal: marginal + Mondrian (per-battery) quantiles.
# --------------------------------------------------------------------------- #


def fit_conformal(preds: np.ndarray, y: np.ndarray, groups: np.ndarray, alpha: float) -> dict:
    residuals = np.abs(y - preds)
    marginal_q = _finite_sample_quantile(residuals, alpha)
    per_battery = {}
    for g in np.unique(groups):
        mask = groups == g
        if mask.sum() >= 5:
            per_battery[str(g)] = _finite_sample_quantile(residuals[mask], alpha)
    return {
        "alpha": alpha,
        "marginal_q": float(marginal_q),
        "per_battery_q": per_battery,
        "n_calibration": int(len(residuals)),
    }


def fit_cqr(train_df: pd.DataFrame, cal_df: pd.DataFrame, feat_cols: list, alpha: float) -> dict:
    """Train quantile regressors at alpha/2 and 1-alpha/2, conformally adjust on cal.

    Returns a dict holding the three fitted sklearn estimators directly (they
    pickle cleanly; the closures returned by ``train_gradient_boosting_quantile_pair``
    don't).
    """
    from sklearn.ensemble import GradientBoostingRegressor  # type: ignore
    from phase4.cqr import CQRRegressor, empirical_cqr_metrics  # type: ignore

    X_train = np.nan_to_num(train_df[feat_cols].values).astype(np.float32)
    y_train = train_df["soh"].values.astype(np.float32)
    X_cal = np.nan_to_num(cal_df[feat_cols].values).astype(np.float32)
    y_cal = cal_df["soh"].values.astype(np.float32)

    kwargs = dict(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42)
    lo = GradientBoostingRegressor(loss="quantile", alpha=alpha / 2, **kwargs).fit(X_train, y_train)
    hi = GradientBoostingRegressor(loss="quantile", alpha=1 - alpha / 2, **kwargs).fit(X_train, y_train)
    med = GradientBoostingRegressor(loss="quantile", alpha=0.5, **kwargs).fit(X_train, y_train)

    cqr = CQRRegressor(
        predict_quantile=lambda X: (lo.predict(X), hi.predict(X)),
        alpha=alpha,
        point_predict=med.predict,
    )
    cqr.calibrate(X_cal, y_cal)
    cov = empirical_cqr_metrics(cqr.predict_interval(X_cal), y_cal)
    return {
        "alpha": alpha,
        "q_hat": float(cqr.q_hat),
        "n_calibration": int(cqr.n_calib),
        "feature_cols": list(feat_cols),
        "model_lo": lo,
        "model_hi": hi,
        "model_median": med,
        "cal_coverage": cov,
    }


# --------------------------------------------------------------------------- #
# Retriever corpus: features + descriptions for the K-NN index.
# --------------------------------------------------------------------------- #


RETRIEVAL_FEATURE_COLS = [
    "capacity", "cycle_number", "voltage_min", "voltage_max",
    "voltage_mean", "voltage_std", "temp_mean", "temp_max", "temp_delta",
    "ica_peak_voltage", "ica_peak_magnitude", "dcir", "discharge_time",
]


def build_retriever_corpus(frame: pd.DataFrame) -> dict:
    cols = [c for c in RETRIEVAL_FEATURE_COLS if c in frame.columns]
    converter = FeatureToText()
    converter.set_reference_values(frame.iloc[0].to_dict())
    descriptions = [converter.convert_single(row).full_description for row in frame.to_dict("records")]

    X = np.nan_to_num(frame[cols].values.astype(np.float32))
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-8] = 1.0
    Xz = (X - mean) / std
    norms = np.linalg.norm(Xz, axis=1, keepdims=True)
    Xz_unit = Xz / np.clip(norms, 1e-8, None)

    metadata = frame[["battery_id", "cycle_number", "soh"]].copy().reset_index(drop=True)
    return {
        "feature_cols": cols,
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "matrix": Xz_unit.astype(np.float32),
        "raw_matrix": X.astype(np.float32),  # needed for KNN-imputation at inference time
        "descriptions": descriptions,
        "metadata": metadata.to_dict("records"),
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    print("Loading existing simple predictor...")
    model, scaler, feat_cols = load_simple_predictor()
    print(f"  feature cols: {feat_cols}")

    csv_path = ROOT / "data" / "phase2" / "harmonized" / "all_features.csv"
    print(f"Loading harmonised corpus from {csv_path}")
    df = pd.read_csv(csv_path)

    # Per-battery time-ordered split: 60% history, 25% calibration, 15% retained for end-user testing.
    cal_parts, train_parts, test_parts = [], [], []
    for bat_id, g in df.groupby("battery_id"):
        g = g.sort_values("cycle_number").reset_index(drop=True)
        n = len(g)
        i_train, i_cal = int(n * 0.6), int(n * 0.85)
        train_parts.append(g.iloc[:i_train])
        cal_parts.append(g.iloc[i_train:i_cal])
        test_parts.append(g.iloc[i_cal:])
    train_df = pd.concat(train_parts, ignore_index=True)
    cal_df = pd.concat(cal_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)
    print(f"  splits: train={len(train_df)} cal={len(cal_df)} test={len(test_df)}")

    print("Computing simple-model predictions on calibration + test sets...")
    cal_preds = predict_simple(model, scaler, feat_cols, cal_df)
    cal_y = cal_df["soh"].values.astype(float)
    test_preds = predict_simple(model, scaler, feat_cols, test_df)
    test_y = test_df["soh"].values.astype(float)

    print("Fitting conformal calibrators (alpha=0.10, 0.05)...")
    conformal = {
        "alpha_0_10": fit_conformal(cal_preds, cal_y, cal_df["battery_id"].values, alpha=0.10),
        "alpha_0_05": fit_conformal(cal_preds, cal_y, cal_df["battery_id"].values, alpha=0.05),
    }
    test_residuals = np.abs(test_preds - test_y)
    test_in = (test_residuals <= conformal["alpha_0_10"]["marginal_q"]).mean()
    print(f"  marginal q_hat (alpha=0.10) = {conformal['alpha_0_10']['marginal_q']:.3f}")
    print(f"  empirical coverage on test  = {test_in:.3f} (target 0.90)")

    print("Fitting Conformalized Quantile Regression (CQR) at alpha=0.10...")
    cqr = fit_cqr(train_df, cal_df, feat_cols, alpha=0.10)
    print(f"  CQR q_hat = {cqr['q_hat']:.3f}  n_cal={cqr['n_calibration']}")
    print(f"  CQR calibration-set coverage = {cqr['cal_coverage']['empirical_coverage']:.3f}, "
          f"mean width = {cqr['cal_coverage']['mean_width']:.2f}")

    print("Computing RUL targets and fitting global Weibull shape...")
    rul_df = compute_rul_targets(df, eol_threshold=80.0)
    shape = fit_global_weibull_shape(rul_df)
    print(f"  Weibull shape k = {shape:.3f}")

    print("Building retriever corpus from training cycles...")
    retriever = build_retriever_corpus(train_df.reset_index(drop=True))
    print(f"  retriever: {len(retriever['descriptions'])} docs over {len(retriever['feature_cols'])} dims")

    state = {
        "version": "phase4_v2",
        "conformal": conformal,
        "cqr": cqr,
        "weibull": {
            "shape": shape,
            "eol_threshold_pct": 80.0,
            "shape_estimation_n": int(((rul_df["event"] == 1) & (rul_df["rul"] > 0)).sum()),
        },
        "retriever": retriever,
        "pinn_info": {
            "loss_weights": {"mse": 1.0, "monotonicity": 0.05, "arrhenius": 0.02, "capacity_resistance": 0.05},
            "described_in": "src/phase4/pinn_loss.py",
            "summary": (
                "Composite training-time loss: MSE + soft penalties for "
                "non-monotonic SOH, anti-Arrhenius temperature/fade ranking, "
                "and capacity–DCIR coupling violations."
            ),
        },
        "diagnostics": {
            "test_marginal_coverage": float(test_in),
            "n_test": int(len(test_y)),
            "n_calibration": int(len(cal_y)),
            "n_train": int(len(train_df)),
            "test_mae": float(np.mean(np.abs(test_preds - test_y))),
        },
    }

    out_path = ROOT / "models" / "phase4_state.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(state, f)
    size_kb = os.path.getsize(out_path) / 1024
    print(f"\nWrote {out_path} ({size_kb:.1f} KB)")

    # --- Also emit cross-attention scaler statistics for the runtime API. ---
    # CRITICAL: these stats must be computed on the SAME train split the model was
    # trained on (phase4_tier_a_novelty.ipynb uses per-battery 60/15/10/15 time
    # splits and fits StandardScaler on the train portion). Using full-corpus
    # stats instead shifts z-scores and moves predictions by several %.
    # (Feature list kept in sync with api/cross_attention_service.CROSS_ATTENTION_FEATURES.)
    CROSS_ATTENTION_FEATURES = [
        "capacity", "voltage_mean", "voltage_std", "voltage_min", "voltage_max", "voltage_range",
        "current_mean", "current_std",
        "temp_mean", "temp_max", "temp_delta", "temp_std",
        "ica_peak_voltage", "ica_peak_magnitude", "ica_mean", "ica_std",
        "dcir", "discharge_time",
    ]

    # The cross-attention checkpoint was trained in phase3_step2_cross_attention.ipynb
    # using a 70/15/15 per-battery time split. Use the SAME split here so the scaler
    # z-scores match what the model was trained on. Using the 60/15/10/15 phase4
    # split instead gives a slightly different mean/std and shifts predictions.
    ca_csv_path = ROOT / "data" / "phase3" / "combined_nasa_mendeley.csv"
    if not ca_csv_path.exists():
        ca_csv_path = csv_path  # fallback
    full_df = pd.read_csv(ca_csv_path)
    train_parts_ca = []
    for bat_id, g in full_df.groupby("battery_id"):
        g = g.sort_values("cycle_number").reset_index(drop=True)
        n = len(g)
        train_parts_ca.append(g.iloc[:int(n * 0.70)])
    train_df_ca = pd.concat(train_parts_ca, ignore_index=True)

    X_train = np.nan_to_num(
        np.asarray(
            [[row.get(c, 0.0) for c in CROSS_ATTENTION_FEATURES] for row in train_df_ca.to_dict("records")],
            dtype=np.float32,
        )
    )
    X_full = np.nan_to_num(
        np.asarray(
            [[row.get(c, 0.0) for c in CROSS_ATTENTION_FEATURES] for row in full_df.to_dict("records")],
            dtype=np.float32,
        )
    )
    ca_mean = X_train.mean(axis=0)
    ca_std = X_train.std(axis=0)
    ca_std[ca_std < 1e-8] = 1.0
    # Imputation means use the full corpus so out-of-training queries still get a reasonable value.
    full_mean = X_full.mean(axis=0)
    ca_stats = {
        "feature_cols": CROSS_ATTENTION_FEATURES,
        "mean": ca_mean.astype(np.float32),
        "std": ca_std.astype(np.float32),
        "impute_means": {c: float(full_mean[i]) for i, c in enumerate(CROSS_ATTENTION_FEATURES)},
        "reference_row": train_df_ca.iloc[0].to_dict(),
        "train_split_n": int(len(train_df_ca)),
        "full_n": int(len(full_df)),
    }
    ca_stats_path = ROOT / "models" / "cross_attention_stats.pkl"
    with open(ca_stats_path, "wb") as f:
        pickle.dump(ca_stats, f)
    print(f"Wrote {ca_stats_path} ({os.path.getsize(ca_stats_path)/1024:.1f} KB)  "
          f"train_n={len(train_df_ca)}  full_n={len(full_df)}")


if __name__ == "__main__":
    main()
