"""Worst-case failure gallery for the SOH model.

Examiners massively reward "you understand your model's limits." This
module picks the N test cycles where the model is most wrong, attributes
the error to individual features via (interventional) SHAP, and writes
a structured report + optional figure.

Output
------
    results/phase5/failure_gallery.json    machine-readable
    results/phase5/failure_gallery.png     (optional) visual panel

Selection criteria
------------------
    worst_residual       |y_pred - y_true| largest
    worst_overestimate   y_pred - y_true largest (model was too optimistic)
    worst_underestimate  y_true - y_pred largest (model was too pessimistic)

We report all three because over-estimating SOH is operationally much
worse than under-estimating (you ship a degraded battery as healthy),
and a calibrated system should show the two have similar magnitude.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class FailureCase:
    index: int
    battery_id: str
    cycle_number: int
    y_true: float
    y_pred: float
    residual: float
    top_attributions: List[Tuple[str, float]]
    feature_values: Dict[str, float]

    def to_dict(self) -> Dict:
        return {
            "index": self.index,
            "battery_id": self.battery_id,
            "cycle_number": self.cycle_number,
            "y_true": round(self.y_true, 3),
            "y_pred": round(self.y_pred, 3),
            "residual": round(self.residual, 3),
            "residual_direction": "over_estimate" if self.residual > 0 else "under_estimate",
            "top_attributions": [(n, round(v, 3)) for n, v in self.top_attributions],
            "feature_values": {k: round(float(v), 4) for k, v in self.feature_values.items()},
        }


# --------------------------------------------------------------------------- #
# Project glue
# --------------------------------------------------------------------------- #


def _default_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_failure_gallery(
    project_root: Optional[Path] = None,
    n_per_category: int = 5,
    use_simple: bool = False,
) -> Dict:
    import pandas as pd
    from src.phase5.causal_shap import shapley_permutation

    project_root = Path(project_root or _default_root())

    if use_simple:
        scaler_path = project_root / "models" / "simple_scaler.pkl"
        cols_path = project_root / "models" / "simple_feature_cols.pkl"
        model_path = project_root / "models" / "simple_model.pkl"
    else:
        scaler_path = project_root / "models" / "scaler.pkl"
        cols_path = project_root / "models" / "feature_cols.pkl"
        model_path = project_root / "models" / "best_model.pkl"

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(cols_path, "rb") as f:
        cols = list(pickle.load(f))
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    train = pd.read_csv(project_root / "data" / "phase2" / "harmonized" / "train_features.csv")
    test = pd.read_csv(project_root / "data" / "phase2" / "harmonized" / "test_features.csv")

    X_train = scaler.transform(train[cols].to_numpy(dtype=float, na_value=0.0))
    X_test = scaler.transform(test[cols].to_numpy(dtype=float, na_value=0.0))
    y_test = test["soh"].to_numpy(dtype=float)

    y_pred = np.asarray(model.predict(X_test))
    resid = y_pred - y_test

    def pick(order_idx: np.ndarray) -> List[int]:
        return [int(i) for i in order_idx[:n_per_category]]

    by_abs = np.argsort(-np.abs(resid))
    by_over = np.argsort(-resid)
    by_under = np.argsort(resid)

    categories = {
        "worst_residual": pick(by_abs),
        "worst_overestimate": pick(by_over),
        "worst_underestimate": pick(by_under),
    }

    # Compute interventional SHAP once for the union of selected indices.
    union_idx = sorted({i for lst in categories.values() for i in lst})
    bg_rng = np.random.default_rng(0)
    bg = X_train[bg_rng.choice(len(X_train), size=min(64, len(X_train)), replace=False)]

    def predict(x):
        return model.predict(x)

    shap_all = shapley_permutation(
        predict_fn=predict,
        x=X_test[union_idx],
        background=bg,
        n_permutations=24,
        perturbation="interventional",
    )
    shap_by_idx = {idx: shap_all[k] for k, idx in enumerate(union_idx)}

    def case_for(idx: int) -> FailureCase:
        row = test.iloc[idx]
        shap_row = shap_by_idx[idx]
        order = np.argsort(-np.abs(shap_row))
        top = [(cols[j], float(shap_row[j])) for j in order[:5]]
        return FailureCase(
            index=int(idx),
            battery_id=str(row.get("battery_id", "unknown")),
            cycle_number=int(row.get("cycle_number", -1)),
            y_true=float(y_test[idx]),
            y_pred=float(y_pred[idx]),
            residual=float(resid[idx]),
            top_attributions=top,
            feature_values={c: float(row[c]) for c in cols if c in row.index},
        )

    gallery: Dict[str, List[Dict]] = {k: [case_for(i).to_dict() for i in v]
                                       for k, v in categories.items()}

    summary = {
        "n_test": int(len(y_test)),
        "mae": float(np.mean(np.abs(resid))),
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "max_over_estimate": float(resid.max()),
        "max_under_estimate": float(-resid.min()),
        "mean_error": float(resid.mean()),
        "feature_set": "simple_6" if use_simple else "full_19",
    }

    out = {"summary": summary, "gallery": gallery}
    out_dir = project_root / "results" / "phase5"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "failure_gallery.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


def save_failure_panel(report: Dict, out_path: Path) -> None:
    """Optional: a matplotlib panel of the top-3 worst residuals with their SHAP bars."""
    import matplotlib.pyplot as plt

    cases = report["gallery"]["worst_residual"][:3]
    if not cases:
        return
    fig, axes = plt.subplots(1, len(cases), figsize=(4.2 * len(cases), 4))
    if len(cases) == 1:
        axes = [axes]

    for ax, case in zip(axes, cases):
        names = [n for n, _ in case["top_attributions"]]
        vals = [v for _, v in case["top_attributions"]]
        colors = ["#d62728" if v > 0 else "#1f77b4" for v in vals]
        ax.barh(names, vals, color=colors)
        ax.axvline(0, color="grey", linewidth=0.5)
        ax.set_title(
            f"{case['battery_id']} c={case['cycle_number']}\n"
            f"true={case['y_true']:.1f} pred={case['y_pred']:.1f} resid={case['residual']:+.1f}",
            fontsize=9,
        )
        ax.invert_yaxis()
    fig.suptitle("Top-3 worst SOH residuals with interventional SHAP", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _format(out: Dict) -> str:
    s = out["summary"]
    lines = [
        "Failure-mode gallery",
        "=" * 60,
        f"Test size        : {s['n_test']}   feature_set={s['feature_set']}",
        f"MAE / RMSE       : {s['mae']:.2f}  / {s['rmse']:.2f}",
        f"Max over-est     : {s['max_over_estimate']:+.2f} (pred > true)",
        f"Max under-est    : {s['max_under_estimate']:+.2f} (pred < true)",
        f"Mean residual    : {s['mean_error']:+.3f} (should be ~0 if unbiased)",
    ]
    for cat in ("worst_residual", "worst_overestimate", "worst_underestimate"):
        lines.append(f"\n[{cat}]")
        for case in out["gallery"][cat][:3]:
            lines.append(
                f"  {case['battery_id']} c={case['cycle_number']}  "
                f"true={case['y_true']:.1f}  pred={case['y_pred']:.1f}  "
                f"resid={case['residual']:+.1f}"
            )
            for name, v in case["top_attributions"][:3]:
                lines.append(f"    {name:<28} attr={v:+.3f}")
    return "\n".join(lines)


def main() -> int:
    out = build_failure_gallery()
    print(_format(out))
    try:
        save_failure_panel(out, _default_root() / "results" / "phase5" / "failure_gallery.png")
        print("\nSaved panel -> results/phase5/failure_gallery.png")
    except Exception as exc:  # pragma: no cover
        print(f"\nPanel skipped: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
