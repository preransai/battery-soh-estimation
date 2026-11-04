"""Cross-dataset leakage probe.

We combine NASA (B0005/B0006/B0007) and Mendeley forklift cycles into a
single training corpus. If the downstream model secretly learns ``source
== Mendeley`` as a shortcut instead of the underlying degradation physics,
performance numbers are inflated and generalization to a held-out
chemistry collapses.

The probe trains a lightweight source classifier (logistic regression) on
each feature representation:

    raw        Raw harmonized feature vector
    scaled     Ridge / cross-attention input (StandardScaler applied)
    embedding  Optional: the pooled cross-attention embedding (if available)

and reports the classifier's ROC-AUC and balanced accuracy. Chance is
0.5 / 0.5. A clean representation is close to chance; a leaky one is
close to 1.0. We also report a permutation baseline so you can see what
"chance" looks like given the class imbalance.

References
----------
Ganin et al. (2016) "Domain-Adversarial Training of Neural Networks."
    Uses the same classifier-as-probe idea to detect (and remove) source
    signal, here we use it purely diagnostically.
"""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Source inference
# --------------------------------------------------------------------------- #

NASA_BATTERY_IDS = {"B0005", "B0006", "B0007", "B0018"}


def infer_source(battery_id: str) -> str:
    """Heuristic: NASA uses B#### IDs, Mendeley uses Cell# or similar.

    We keep this isolated so a future dataset only needs a new rule here.
    """
    bid = str(battery_id).strip()
    if bid in NASA_BATTERY_IDS or bid.startswith("B00"):
        return "NASA"
    return "MENDELEY"


# --------------------------------------------------------------------------- #
# Core probe
# --------------------------------------------------------------------------- #


@dataclass
class ProbeResult:
    representation: str
    roc_auc: float
    balanced_accuracy: float
    permutation_roc_auc: float
    n_samples: int
    n_nasa: int
    n_mendeley: int
    top_features: List[Tuple[str, float]] = field(default_factory=list)
    verdict: str = ""

    def to_dict(self) -> Dict:
        return {
            "representation": self.representation,
            "roc_auc": round(self.roc_auc, 4),
            "balanced_accuracy": round(self.balanced_accuracy, 4),
            "permutation_roc_auc": round(self.permutation_roc_auc, 4),
            "n_samples": self.n_samples,
            "n_nasa": self.n_nasa,
            "n_mendeley": self.n_mendeley,
            "top_features": [(n, round(w, 4)) for n, w in self.top_features],
            "verdict": self.verdict,
        }


def _verdict(roc_auc: float) -> str:
    if roc_auc < 0.60:
        return "CLEAN (source signal ~= chance)"
    if roc_auc < 0.75:
        return "MILD leakage"
    if roc_auc < 0.90:
        return "SIGNIFICANT leakage"
    return "SEVERE leakage (representation encodes source identity)"


def run_probe(
    X: np.ndarray,
    sources: Sequence[str],
    feature_names: Optional[Sequence[str]] = None,
    representation: str = "raw",
    n_folds: int = 5,
    random_state: int = 42,
) -> ProbeResult:
    """Logistic-regression source classifier with 5-fold CV.

    Metrics are averaged across folds. The permutation baseline shuffles
    labels inside each fold to estimate what an uninformative classifier
    would score, which matters when classes are imbalanced.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score
    from sklearn.preprocessing import StandardScaler

    X = np.asarray(X, dtype=float)
    y = np.asarray([1 if s == "MENDELEY" else 0 for s in sources])

    if len(np.unique(y)) < 2:
        raise ValueError(
            "Leakage probe needs both NASA and Mendeley cycles in the input; "
            f"got only source={set(sources)}"
        )

    aucs, baccs, perm_aucs = [], [], []
    coef_accumulator = np.zeros(X.shape[1])

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    for tr, te in skf.split(X, y):
        scaler = StandardScaler().fit(X[tr])
        Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])

        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=random_state)
        clf.fit(Xtr, y[tr])
        prob = clf.predict_proba(Xte)[:, 1]
        pred = (prob >= 0.5).astype(int)

        aucs.append(roc_auc_score(y[te], prob))
        baccs.append(balanced_accuracy_score(y[te], pred))
        coef_accumulator += np.abs(clf.coef_.ravel())

        # Permutation baseline: shuffle train labels, measure test AUC.
        rng = np.random.default_rng(random_state + int(tr[0]))
        y_perm = rng.permutation(y[tr])
        clf_perm = LogisticRegression(max_iter=1000, C=1.0, random_state=random_state)
        clf_perm.fit(Xtr, y_perm)
        prob_perm = clf_perm.predict_proba(Xte)[:, 1]
        perm_aucs.append(roc_auc_score(y[te], prob_perm))

    coef_accumulator /= n_folds
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(X.shape[1])]
    order = np.argsort(-coef_accumulator)
    top = [(str(feature_names[i]), float(coef_accumulator[i])) for i in order[:10]]

    auc = float(np.mean(aucs))
    return ProbeResult(
        representation=representation,
        roc_auc=auc,
        balanced_accuracy=float(np.mean(baccs)),
        permutation_roc_auc=float(np.mean(perm_aucs)),
        n_samples=len(y),
        n_nasa=int((y == 0).sum()),
        n_mendeley=int((y == 1).sum()),
        top_features=top,
        verdict=_verdict(auc),
    )


# --------------------------------------------------------------------------- #
# Convenience wrappers that load the project's own artifacts
# --------------------------------------------------------------------------- #


def load_combined_dataset(project_root: Path) -> pd.DataFrame:
    """Return the harmonized NASA + Mendeley corpus used by the cross-attention model."""
    combined = project_root / "data" / "phase3" / "combined_nasa_mendeley.csv"
    if combined.exists():
        return pd.read_csv(combined)
    fallback = project_root / "data" / "phase2" / "harmonized" / "train_features.csv"
    if fallback.exists():
        return pd.read_csv(fallback)
    raise FileNotFoundError(
        "Expected combined_nasa_mendeley.csv under data/phase3/ or "
        "train_features.csv under data/phase2/harmonized/"
    )


def _load_scaler_and_cols(project_root: Path) -> Tuple[object, List[str]]:
    with open(project_root / "models" / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(project_root / "models" / "feature_cols.pkl", "rb") as f:
        cols = list(pickle.load(f))
    return scaler, cols


def run_project_probe(project_root: Optional[Path] = None) -> Dict[str, Dict]:
    """Run the probe on the current project's harmonized corpus.

    Returns a dict with one entry per representation. Writes a JSON copy
    to ``results/phase5/leakage.json`` so downstream reports can pick it
    up without re-running.
    """
    project_root = Path(project_root or _default_root())
    df = load_combined_dataset(project_root)

    if "source" not in df.columns:
        df = df.copy()
        df["source"] = df["battery_id"].map(infer_source)

    # Feature columns: use the cached feature_cols.pkl if present, otherwise
    # fall back to every numeric column except ID / target / source.
    try:
        _, cols = _load_scaler_and_cols(project_root)
    except Exception:
        reserved = {"battery_id", "source", "soh", "capacity", "cycle_number",
                    "original_index", "data_quality", "has_spikes"}
        cols = [c for c in df.columns
                if c not in reserved and pd.api.types.is_numeric_dtype(df[c])]

    cols = [c for c in cols if c in df.columns]
    X = df[cols].to_numpy(dtype=float, na_value=0.0)
    sources = df["source"].tolist()

    results: Dict[str, Dict] = {}
    results["raw"] = run_probe(X, sources, feature_names=cols, representation="raw").to_dict()

    try:
        scaler, _ = _load_scaler_and_cols(project_root)
        Xs = scaler.transform(X)
        results["scaled"] = run_probe(Xs, sources, feature_names=cols, representation="scaled").to_dict()
    except Exception as exc:  # pragma: no cover - optional path
        results["scaled"] = {"skipped": True, "reason": str(exc)}

    out_dir = project_root / "results" / "phase5"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "leakage.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results


def _default_root() -> Path:
    return Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _format_report(results: Dict[str, Dict]) -> str:
    lines = ["Leakage probe results", "=" * 60]
    for rep, data in results.items():
        lines.append(f"\nRepresentation: {rep}")
        if data.get("skipped"):
            lines.append(f"  skipped: {data.get('reason')}")
            continue
        lines.append(f"  n            : {data['n_samples']} "
                     f"(NASA={data['n_nasa']}, Mendeley={data['n_mendeley']})")
        lines.append(f"  ROC-AUC      : {data['roc_auc']:.3f}")
        lines.append(f"  Balanced-acc : {data['balanced_accuracy']:.3f}")
        lines.append(f"  Permuted AUC : {data['permutation_roc_auc']:.3f}  (baseline ~ 0.5)")
        lines.append(f"  Verdict      : {data['verdict']}")
        if data["top_features"]:
            lines.append("  Top-5 most source-predictive features:")
            for name, weight in data["top_features"][:5]:
                lines.append(f"    {name:<28} |coef|={weight:.3f}")
    return "\n".join(lines)


def main() -> int:
    results = run_project_probe()
    print(_format_report(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
