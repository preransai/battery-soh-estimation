"""Leave-one-battery-out cross-validation with multi-seed error bars.

Phase 2 and Phase 3 reported results on a per-battery *time-ordered*
split, which is the easiest possible generalisation test. This module
produces the LOBO number a reviewer will actually ask for:

    For each battery b in {B0005, B0006, B0007, ...}:
        train Ridge on all OTHER batteries
        test on battery b
    Repeat for seeds {42, 43, 44}.
    Report mean +- std of MAE / R^2 per held-out battery.

Ridge fits in milliseconds, so 3 seeds * 3 held-out batteries = 9 fits
is essentially free. We also run a "joint" variant that includes the
Mendeley corpus in training, and compare the within-group MAE on NASA
batteries under the two training sets - a clean way to quantify the
gain (or loss) from adding heterogeneous data.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _default_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass
class FoldResult:
    held_out: str
    train_n: int
    test_n: int
    mae: float
    rmse: float
    r2: float
    seed: int


def _fit_ridge(X_tr, y_tr, X_te, y_te, alpha=1.0, seed=42):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, r2_score
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)
    model = Ridge(alpha=alpha, random_state=seed).fit(X_tr_s, y_tr)
    pred = model.predict(X_te_s)
    mae = float(mean_absolute_error(y_te, pred))
    rmse = float(np.sqrt(np.mean((pred - y_te) ** 2)))
    r2 = float(r2_score(y_te, pred))
    return mae, rmse, r2


def run_lobo(
    project_root: Optional[Path] = None,
    seeds: Tuple[int, ...] = (42, 43, 44),
    include_mendeley: bool = True,
) -> Dict:
    project_root = Path(project_root or _default_root())
    with open(project_root / "models" / "feature_cols.pkl", "rb") as f:
        cols = list(pickle.load(f))

    # Load both harmonised NASA data and (if requested) combined corpus.
    all_path = project_root / "data" / "phase2" / "harmonized" / "all_features.csv"
    df_nasa = pd.read_csv(all_path)
    df_combined: Optional[pd.DataFrame] = None
    if include_mendeley:
        combined_path = project_root / "data" / "phase3" / "combined_nasa_mendeley.csv"
        if combined_path.exists():
            df_combined = pd.read_csv(combined_path)
    cols = [c for c in cols if c in df_nasa.columns]

    nasa_bats = sorted(df_nasa["battery_id"].astype(str).unique().tolist())

    folds_nasa_only: List[FoldResult] = []
    folds_combined: List[FoldResult] = []

    for seed in seeds:
        for held in nasa_bats:
            # NASA-only training
            tr = df_nasa[df_nasa["battery_id"].astype(str) != held]
            te = df_nasa[df_nasa["battery_id"].astype(str) == held]
            if tr.empty or te.empty:
                continue
            X_tr = tr[cols].to_numpy(dtype=float, na_value=0.0)
            y_tr = tr["soh"].to_numpy(dtype=float)
            X_te = te[cols].to_numpy(dtype=float, na_value=0.0)
            y_te = te["soh"].to_numpy(dtype=float)
            mae, rmse, r2 = _fit_ridge(X_tr, y_tr, X_te, y_te, seed=seed)
            folds_nasa_only.append(FoldResult(held, len(tr), len(te), mae, rmse, r2, seed))

            # Combined NASA + Mendeley training (test set still this NASA battery)
            if df_combined is not None:
                cols_c = [c for c in cols if c in df_combined.columns]
                tr_c = df_combined[df_combined["battery_id"].astype(str) != held]
                te_c = df_combined[df_combined["battery_id"].astype(str) == held]
                if not te_c.empty:
                    X_tr2 = tr_c[cols_c].to_numpy(dtype=float, na_value=0.0)
                    y_tr2 = tr_c["soh"].to_numpy(dtype=float)
                    X_te2 = te_c[cols_c].to_numpy(dtype=float, na_value=0.0)
                    y_te2 = te_c["soh"].to_numpy(dtype=float)
                    mae, rmse, r2 = _fit_ridge(X_tr2, y_tr2, X_te2, y_te2, seed=seed)
                    folds_combined.append(FoldResult(held, len(tr_c), len(te_c), mae, rmse, r2, seed))

    def aggregate(folds: List[FoldResult]) -> Dict:
        if not folds:
            return {}
        by_bat: Dict[str, Dict] = {}
        for b in sorted({f.held_out for f in folds}):
            sub = [f for f in folds if f.held_out == b]
            maes = np.array([f.mae for f in sub])
            r2s = np.array([f.r2 for f in sub])
            by_bat[b] = {
                "mae_mean": round(float(maes.mean()), 4),
                "mae_std":  round(float(maes.std(ddof=0)), 4),
                "r2_mean":  round(float(r2s.mean()), 4),
                "r2_std":   round(float(r2s.std(ddof=0)), 4),
                "n_seeds":  len(sub),
                "test_n":   sub[0].test_n,
            }
        all_maes = np.array([f.mae for f in folds])
        return {
            "per_battery": by_bat,
            "overall_mae_mean": round(float(all_maes.mean()), 4),
            "overall_mae_std":  round(float(all_maes.std(ddof=0)), 4),
            "n_folds": len(folds),
        }

    out = {
        "seeds": list(seeds),
        "nasa_batteries": nasa_bats,
        "nasa_only": aggregate(folds_nasa_only),
        "combined_nasa_mendeley": aggregate(folds_combined),
        "note": (
            "Leave-one-battery-out: for each NASA battery, the remaining "
            "batteries form the training set. The 'combined' variant adds "
            "Mendeley cycles to training but still tests on the held-out "
            "NASA battery. All folds use Ridge(alpha=1.0) on the 19-feature "
            "vector with train-only standard scaling."
        ),
    }
    out_dir = project_root / "results" / "phase6"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "lobo_cv.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


def _format(out: Dict) -> str:
    lines = ["Leave-one-battery-out CV (Ridge)", "=" * 60]
    lines.append(f"Seeds               : {out['seeds']}")
    lines.append(f"NASA batteries held : {out['nasa_batteries']}")

    for label, key in [("NASA only", "nasa_only"),
                       ("NASA + Mendeley training", "combined_nasa_mendeley")]:
        block = out[key]
        if not block:
            continue
        lines.append(f"\n{label}:")
        lines.append(f"  Overall MAE : {block['overall_mae_mean']:.3f} ± {block['overall_mae_std']:.3f}  "
                     f"(n={block['n_folds']} folds)")
        for bid, m in block["per_battery"].items():
            lines.append(
                f"  {bid:<6} MAE {m['mae_mean']:.3f} ± {m['mae_std']:.3f}  "
                f"R^2 {m['r2_mean']:.3f} ± {m['r2_std']:.3f}"
            )
    lines.append("\n" + out["note"])
    return "\n".join(lines)


def main() -> int:
    out = run_lobo()
    print(_format(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
