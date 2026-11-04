"""DiCE-style minimal-change counterfactuals for SOH predictions.

Given a battery whose predicted SOH is, say, 72%, what's the smallest
feasible change to its operating conditions that would push the
prediction back above 80%? That question maps directly to BMS actions
(cool the pack, limit discharge rate, drop cycle depth) in a way SHAP
cannot.

We implement a simplified DiCE objective (Mothilal, Sharma, Tan 2020)
that searches for a counterfactual ``x'`` minimising

    L(x') = | f(x') - target_soh |        # validity
          + lambda_prox * || x' - x ||_1 / MAD    # proximity (MAD-scaled L1)
          + lambda_sparsity * ||x' - x||_0        # sparsity (few changes)

subject to monotone range constraints on each actionable feature
(e.g. temp_mean >= 15 and <= 45, capacity cannot increase past rated).

We use a random-restart coordinate-descent search over a discrete grid
because the underlying model (Ridge regression) is linear, so the
global optimum is unique and cheap to locate by grid search per feature.

For the (later) cross-attention model, swap this for a gradient-based
DiCE. The API wraps the search in a pure function so the caller can
hot-swap estimators.

References
----------
Mothilal, Sharma, Tan (FAccT 2020) "Explaining machine learning
    classifiers through diverse counterfactual explanations."
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Public dataclasses
# --------------------------------------------------------------------------- #


@dataclass
class FeatureSpec:
    """Allowed operating range + semantic direction for one feature.

    ``direction`` is either:
        +1   a larger value is beneficial (e.g. capacity)
        -1   a smaller value is beneficial (e.g. temp_delta)
         0   ambiguous / any direction allowed
    """
    name: str
    lo: float
    hi: float
    direction: int = 0
    actionable: bool = True


@dataclass
class Counterfactual:
    original: Dict[str, float]
    modified: Dict[str, float]
    deltas: Dict[str, float]
    predicted_original: float
    predicted_counterfactual: float
    target: float
    n_features_changed: int
    l1_normalised: float

    def to_dict(self) -> Dict:
        return {
            "predicted_original": round(float(self.predicted_original), 3),
            "predicted_counterfactual": round(float(self.predicted_counterfactual), 3),
            "target": round(float(self.target), 3),
            "n_features_changed": int(self.n_features_changed),
            "l1_normalised": round(float(self.l1_normalised), 4),
            "changes": {
                k: {
                    "from": round(float(self.original[k]), 4),
                    "to": round(float(self.modified[k]), 4),
                    "delta": round(float(v), 4),
                }
                for k, v in self.deltas.items()
                if abs(v) > 1e-9
            },
        }


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #


def _mad(X: np.ndarray) -> np.ndarray:
    """Median Absolute Deviation per column, floored so we don't divide by zero."""
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0)
    return np.maximum(mad, 1e-6)


def find_counterfactual(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    feature_specs: Sequence[FeatureSpec],
    target_soh: float,
    background: np.ndarray,
    n_grid: int = 25,
    lambda_prox: float = 0.5,
    lambda_sparsity: float = 0.1,
    max_features_changed: int = 3,
    random_state: int = 42,
) -> Optional[Counterfactual]:
    """Coordinate-descent-style search for a single minimal counterfactual.

    Parameters
    ----------
    predict_fn
        Regressor; takes (n, d) -> (n,).
    x
        (d,) instance to explain (pre-scaled, matching predict_fn's input).
    feature_specs
        Per-feature bounds / actionability. Must match ``x``'s order.
    target_soh
        Desired SOH (e.g. 80.0 if the battery is below EoL and we want it
        back above EoL).
    background
        (n_bg, d) reference sample used to compute MAD for the proximity
        term. Should be on the same pre-scaled scale as ``x``.
    n_grid
        Number of grid points per feature when scanning.
    max_features_changed
        Hard cap on the number of simultaneously perturbed features. The
        search picks the single best feature, then the best pair, etc.,
        up to this cap.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    if len(x) != len(feature_specs):
        raise ValueError("x and feature_specs must have the same length")

    mad = _mad(np.asarray(background, dtype=float))
    current_pred = float(np.asarray(predict_fn(x.reshape(1, -1)))[0])

    def score(x_cand: np.ndarray) -> Tuple[float, float]:
        pred = float(np.asarray(predict_fn(x_cand.reshape(1, -1)))[0])
        validity = abs(pred - target_soh)
        prox = float(np.sum(np.abs(x_cand - x) / mad))
        sparsity = float((np.abs(x_cand - x) > 1e-6).sum())
        return validity + lambda_prox * prox + lambda_sparsity * sparsity, pred

    # Greedy additive: pick one feature at a time that lowers objective most.
    best_x = x.copy()
    best_obj, best_pred = score(best_x)
    changed: List[int] = []
    rng = np.random.default_rng(random_state)

    for _ in range(max_features_changed):
        best_gain = 0.0
        best_feat = -1
        best_val = None
        for j, spec in enumerate(feature_specs):
            if j in changed or not spec.actionable:
                continue
            grid = np.linspace(spec.lo, spec.hi, n_grid)
            # If a direction is prescribed, bias toward that direction.
            if spec.direction == +1:
                grid = grid[grid >= x[j]]
            elif spec.direction == -1:
                grid = grid[grid <= x[j]]
            if len(grid) == 0:
                continue
            for v in grid:
                trial = best_x.copy()
                trial[j] = v
                obj, pred = score(trial)
                gain = best_obj - obj
                if gain > best_gain + 1e-6:
                    best_gain = gain
                    best_feat = j
                    best_val = float(v)
        if best_feat < 0 or best_val is None:
            break
        best_x[best_feat] = best_val
        best_obj, best_pred = score(best_x)
        changed.append(best_feat)

    if not changed:
        return None

    deltas = {
        feature_specs[j].name: float(best_x[j] - x[j])
        for j in changed
    }
    original_map = {spec.name: float(x[i]) for i, spec in enumerate(feature_specs)}
    modified_map = {spec.name: float(best_x[i]) for i, spec in enumerate(feature_specs)}
    return Counterfactual(
        original=original_map,
        modified=modified_map,
        deltas=deltas,
        predicted_original=current_pred,
        predicted_counterfactual=best_pred,
        target=target_soh,
        n_features_changed=len(changed),
        l1_normalised=float(np.sum(np.abs(best_x - x) / mad)),
    )


# --------------------------------------------------------------------------- #
# Project glue
# --------------------------------------------------------------------------- #


DEFAULT_DIRECTIONS: Dict[str, int] = {
    # Higher is better for these.
    "capacity": +1,
    "voltage_max": +1,
    # Lower is better for these.
    "temp_mean": -1,
    "temp_max": -1,
    "temp_delta": -1,
    "temp_std": -1,
    "dcir": -1,
    "current_std": -1,
    "voltage_std": -1,
}


def _default_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_project_specs(
    project_root: Path,
    cols: Sequence[str],
    scaler,
    X_bg: np.ndarray,
) -> List[FeatureSpec]:
    """Build FeatureSpecs in the *scaled* feature space.

    We take the min/max of each column in the (pre-scaled) background as
    the feasibility band. Directions come from DEFAULT_DIRECTIONS.
    """
    lo = X_bg.min(axis=0)
    hi = X_bg.max(axis=0)
    specs: List[FeatureSpec] = []
    for i, name in enumerate(cols):
        specs.append(
            FeatureSpec(
                name=name,
                lo=float(lo[i]),
                hi=float(hi[i]),
                direction=DEFAULT_DIRECTIONS.get(name, 0),
                actionable=True,
            )
        )
    return specs


def run_project_counterfactuals(
    project_root: Optional[Path] = None,
    n_examples: int = 5,
    target_soh: float = 85.0,
    use_simple: bool = True,
) -> Dict:
    """Find counterfactuals for the lowest-SOH test samples."""
    import pandas as pd

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

    specs = build_project_specs(project_root, cols, scaler, X_train)

    def predict(x):
        return model.predict(x)

    # Pick the N lowest-SOH test cycles.
    order = np.argsort(y_test)
    selected = order[:n_examples]

    results: List[Dict] = []
    for idx in selected:
        cf = find_counterfactual(
            predict_fn=predict,
            x=X_test[idx],
            feature_specs=specs,
            target_soh=target_soh,
            background=X_train,
        )
        if cf is None:
            results.append({
                "cycle_index": int(idx),
                "true_soh": float(y_test[idx]),
                "status": "no_feasible_counterfactual",
            })
            continue
        item = cf.to_dict()
        item["cycle_index"] = int(idx)
        item["true_soh"] = round(float(y_test[idx]), 3)
        results.append(item)

    out = {
        "feature_set": "simple_6" if use_simple else "full_19",
        "target_soh": target_soh,
        "n_examples": n_examples,
        "examples": results,
    }
    out_dir = project_root / "results" / "phase5"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "counterfactuals.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _format(out: Dict) -> str:
    lines = ["Counterfactual explanations for low-SOH cycles", "=" * 60,
             f"Feature set : {out['feature_set']}",
             f"Target SOH  : {out['target_soh']}",
             ""]
    for ex in out["examples"]:
        lines.append(f"Cycle #{ex.get('cycle_index')}  true_soh={ex.get('true_soh')}")
        if ex.get("status") == "no_feasible_counterfactual":
            lines.append("  no feasible counterfactual found")
            lines.append("")
            continue
        lines.append(
            f"  predicted: {ex['predicted_original']:.2f} -> "
            f"{ex['predicted_counterfactual']:.2f}  (target {ex['target']:.1f})"
        )
        lines.append(f"  features changed: {ex['n_features_changed']}  "
                     f"L1/MAD={ex['l1_normalised']:.2f}")
        for name, info in ex["changes"].items():
            lines.append(f"    {name:<28} {info['from']:+.3f} -> {info['to']:+.3f}  "
                         f"(d={info['delta']:+.3f})")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    out = run_project_counterfactuals()
    print(_format(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
