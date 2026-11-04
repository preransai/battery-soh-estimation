"""Causal (Shapley + do-intervention) feature attribution.

Standard SHAP values answer "how much did feature X contribute to this
prediction, given that every other feature is held at its observed
value?". That is an *observational* question; it conflates X's direct
effect on the target with X's correlation with other features that also
drive the target.

For battery SOH this matters: ``cycle_number`` is strongly correlated
with ``capacity``, and a naive SHAP explanation hands most of the credit
to whichever of the two the background set happens to expose first.

Causal Shapley (Heskes et al. 2020, Janzing et al. 2020) replaces the
observational marginalisation with a *do-intervention*: when computing
a Shapley coalition, the features NOT in the coalition are redrawn from
their marginal distribution (independent of the features IN the
coalition) rather than from their conditional distribution. This
corresponds to intervening on the coalition via ``do()`` and asking how
the prediction changes.

Implementation note
-------------------
We do not need the full causal graph. We use the "interventional SHAP"
variant that Janzing et al. derived: it's equivalent to ``shap.
KernelExplainer`` with ``feature_perturbation="interventional"`` but we
reimplement the sampling loop directly so the output is transparent and
we can mix it with the project's existing Ridge/CrossAttention models.
Exact Shapley for N=19 features is tractable with 2^N = 524288 coalitions,
but we sample ``n_permutations`` random orderings (the Castro et al. 2009
permutation estimator) because 524288 * len(background) would be slow.

References
----------
Heskes, Sijben, Bucur, Claassen (NeurIPS 2020) "Causal Shapley Values."
Janzing, Minorics, Bloebaum (AISTATS 2020) "Feature relevance
    quantification in explainable AI: A causal problem."
Castro, Gomez, Tejada (2009) "Polynomial calculation of the Shapley value."
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class AttributionResult:
    feature_names: List[str]
    observational_shap: np.ndarray   # (n_instances, n_features)
    interventional_shap: np.ndarray  # (n_instances, n_features)
    disagreement: np.ndarray         # |obs - int| per (instance, feature)
    mean_abs_observational: np.ndarray
    mean_abs_interventional: np.ndarray
    mean_abs_disagreement: np.ndarray

    def rankings(self) -> Dict[str, List[Tuple[str, float]]]:
        order_obs = np.argsort(-self.mean_abs_observational)
        order_int = np.argsort(-self.mean_abs_interventional)
        return {
            "observational": [
                (self.feature_names[i], float(self.mean_abs_observational[i])) for i in order_obs
            ],
            "interventional": [
                (self.feature_names[i], float(self.mean_abs_interventional[i])) for i in order_int
            ],
        }


# --------------------------------------------------------------------------- #
# Permutation-based Shapley estimator with selectable perturbation
# --------------------------------------------------------------------------- #


def shapley_permutation(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    background: np.ndarray,
    n_permutations: int = 32,
    perturbation: str = "interventional",
    random_state: int = 42,
) -> np.ndarray:
    """Estimate Shapley values for one or more instances.

    Parameters
    ----------
    predict_fn
        Scalar regressor. ``predict_fn(X)`` takes (n, d) and returns (n,).
    x
        (n_instances, d) instances to explain.
    background
        (n_bg, d) reference sample used for coalition marginalisation.
    perturbation
        ``"interventional"``  -> features outside the coalition are
                                 drawn from the marginal of that column
                                 in ``background`` (independently per
                                 column). This implements Janzing/Heskes
                                 causal Shapley.
        ``"observational"``   -> features outside the coalition are
                                 replaced by a *single* full row from
                                 ``background``. This is the TreeSHAP /
                                 sampling-KernelSHAP default and
                                 conflates direct effect with
                                 correlation.
    n_permutations
        Number of random feature orderings per instance. Variance falls as
        1/sqrt(n_permutations).

    Returns
    -------
    (n_instances, d) array of Shapley estimates.
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    background = np.asarray(background, dtype=float)
    n_inst, d = x.shape
    n_bg = len(background)
    if n_bg < 2:
        raise ValueError("Need >=2 background rows")

    rng = np.random.default_rng(random_state)
    shap = np.zeros((n_inst, d), dtype=float)

    for inst_idx in range(n_inst):
        xi = x[inst_idx]
        for _ in range(n_permutations):
            order = rng.permutation(d)

            # Pick a background row once per permutation for observational
            # marginalisation; for interventional we'll re-sample per
            # feature inside the loop.
            bg_row = background[rng.integers(0, n_bg)]

            # Start from a fully-masked instance.
            if perturbation == "observational":
                current = bg_row.copy()
            elif perturbation == "interventional":
                current = np.array(
                    [background[rng.integers(0, n_bg), j] for j in range(d)]
                )
            else:
                raise ValueError(f"unknown perturbation: {perturbation}")

            # Evaluate incrementally: reveal feature-by-feature in the
            # permutation order and record each marginal contribution.
            pred_prev = float(np.asarray(predict_fn(current.reshape(1, -1)))[0])
            for feat in order:
                current_next = current.copy()
                current_next[feat] = xi[feat]
                pred_next = float(np.asarray(predict_fn(current_next.reshape(1, -1)))[0])
                shap[inst_idx, feat] += (pred_next - pred_prev) / n_permutations
                current = current_next
                pred_prev = pred_next

    return shap


# --------------------------------------------------------------------------- #
# Side-by-side observational vs interventional
# --------------------------------------------------------------------------- #


def compare_attributions(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    background: np.ndarray,
    feature_names: Sequence[str],
    n_permutations: int = 32,
    random_state: int = 42,
) -> AttributionResult:
    obs = shapley_permutation(
        predict_fn, x, background,
        n_permutations=n_permutations,
        perturbation="observational",
        random_state=random_state,
    )
    inter = shapley_permutation(
        predict_fn, x, background,
        n_permutations=n_permutations,
        perturbation="interventional",
        random_state=random_state + 1,
    )
    disagree = np.abs(obs - inter)
    return AttributionResult(
        feature_names=list(feature_names),
        observational_shap=obs,
        interventional_shap=inter,
        disagreement=disagree,
        mean_abs_observational=np.abs(obs).mean(axis=0),
        mean_abs_interventional=np.abs(inter).mean(axis=0),
        mean_abs_disagreement=disagree.mean(axis=0),
    )


# --------------------------------------------------------------------------- #
# Project glue
# --------------------------------------------------------------------------- #


def _default_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run_project_causal_shap(
    project_root: Optional[Path] = None,
    n_instances: int = 20,
    n_background: int = 64,
    n_permutations: int = 24,
    use_simple: bool = True,
) -> Dict:
    """Run observational vs interventional Shapley on a small test subset.

    Set ``use_simple=True`` to explain the 6-feature Ridge (fast and the
    feature names are the interpretable ones). ``use_simple=False``
    switches to the 19-feature production Ridge.
    """
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

    rng = np.random.default_rng(0)
    bg_idx = rng.choice(len(X_train), size=min(n_background, len(X_train)), replace=False)
    test_idx = rng.choice(len(X_test), size=min(n_instances, len(X_test)), replace=False)

    def predict(x):
        return model.predict(x)

    result = compare_attributions(
        predict_fn=predict,
        x=X_test[test_idx],
        background=X_train[bg_idx],
        feature_names=cols,
        n_permutations=n_permutations,
    )

    # Pack up: rankings + per-feature disagreement magnitude.
    out = {
        "feature_set": "simple_6" if use_simple else "full_19",
        "n_instances": int(result.observational_shap.shape[0]),
        "n_background": int(len(bg_idx)),
        "n_permutations": int(n_permutations),
        "rankings": result.rankings(),
        "mean_abs_disagreement": {
            cols[i]: round(float(result.mean_abs_disagreement[i]), 4)
            for i in range(len(cols))
        },
    }

    out_dir = project_root / "results" / "phase5"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "causal_shap.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _format(out: Dict) -> str:
    lines = [
        "Observational vs Causal (Interventional) Shapley",
        "=" * 60,
        f"Feature set  : {out['feature_set']}",
        f"Instances    : {out['n_instances']}",
        f"Background   : {out['n_background']}",
        f"Permutations : {out['n_permutations']}",
        "",
        "Mean |SHAP| (observational) - top 6:",
    ]
    for name, val in out["rankings"]["observational"][:6]:
        lines.append(f"  {name:<28} {val:+.3f}")
    lines.append("")
    lines.append("Mean |SHAP| (interventional / causal) - top 6:")
    for name, val in out["rankings"]["interventional"][:6]:
        lines.append(f"  {name:<28} {val:+.3f}")
    lines.append("")
    lines.append("Disagreement (|obs - interventional|, larger = more correlation-driven):")
    items = sorted(out["mean_abs_disagreement"].items(), key=lambda kv: -kv[1])
    for name, val in items[:6]:
        lines.append(f"  {name:<28} {val:+.3f}")
    return "\n".join(lines)


def main() -> int:
    out = run_project_causal_shap()
    print(_format(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
