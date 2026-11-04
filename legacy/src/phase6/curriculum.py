"""Curriculum-learning comparison.

The SOH regression problem has a clear difficulty gradient: healthy
mid-life cycles are easy (SOH ~ 0.9, linear in capacity), end-of-life
cycles are hard (non-linear knee, thermal noise, fewer samples). The
curriculum-learning hypothesis (Bengio et al. 2009) is that training on
easy samples first and progressively exposing the model to harder ones
yields a better final fit than random-order SGD.

We implement the simplest testable version:

    - easy-first: sort training rows by |SOH - 1.0| ascending (healthy
      cycles first), feed through a tiny MLP one epoch per split
    - hard-first: same but descending
    - random:  uniform shuffling each epoch (standard SGD)

Metric: test MAE overall AND test MAE restricted to the hard-tail
(SOH <= 0.8 on NASA). A curriculum win should show up preferentially in
the tail.

Ridge has a closed-form solution and ignores presentation order, so we
do the comparison with a small MLP. Ten seeds, 30 epochs each, ~3 min
wall time on CPU. If the mean tail-MAE gap between curricula is
smaller than the seed std, we conclude no benefit and report that
honestly.
"""

from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn


def _default_root() -> Path:
    return Path(__file__).resolve().parents[2]


class TinyMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


@dataclass
class TrainResult:
    schedule: str
    seed: int
    test_mae: float
    tail_mae: float  # MAE restricted to y_true <= 0.8
    epochs: int


def _schedule_order(y: np.ndarray, rng: np.random.Generator, mode: str) -> np.ndarray:
    if mode == "random":
        return rng.permutation(len(y))
    difficulty = np.abs(y - 1.0)
    if mode == "easy_first":
        return np.argsort(difficulty)
    if mode == "hard_first":
        return np.argsort(-difficulty)
    raise ValueError(mode)


def _fit_one(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    schedule: str, seed: int, epochs: int = 30,
    batch_size: int = 64,
) -> TrainResult:
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    X_tr_t = torch.from_numpy(X_tr).float()
    y_tr_t = torch.from_numpy(y_tr).float()
    X_te_t = torch.from_numpy(X_te).float()

    model = TinyMLP(in_dim=X_tr.shape[1])
    with torch.no_grad():
        final = list(model.net.modules())[-1]
        if isinstance(final, nn.Linear):
            final.bias.fill_(float(y_tr.mean()))
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)

    for ep in range(epochs):
        # Re-compute order each epoch for the random schedule; keep
        # fixed order for curriculum schedules but still shuffle the
        # minibatches within the easy->hard sweep.
        if schedule == "random":
            order = rng.permutation(len(y_tr))
        else:
            # Deterministic curriculum ordering (once), then walk through
            # in mini-batches so each epoch sees easy->hard once.
            order = _schedule_order(y_tr, rng, schedule)
        model.train()
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            preds = model(X_tr_t[idx])
            loss = nn.functional.mse_loss(preds, y_tr_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        pred_te = model(X_te_t).numpy()
    test_mae = float(np.mean(np.abs(pred_te - y_te)))
    tail_mask = y_te <= 0.8
    if tail_mask.any():
        tail_mae = float(np.mean(np.abs(pred_te[tail_mask] - y_te[tail_mask])))
    else:
        tail_mae = float("nan")
    return TrainResult(schedule=schedule, seed=seed,
                       test_mae=round(test_mae, 5),
                       tail_mae=round(tail_mae, 5),
                       epochs=epochs)


def run_curriculum(project_root: Optional[Path] = None,
                   seeds: tuple = (0, 1, 2, 3, 4),
                   epochs: int = 30) -> Dict:
    project_root = Path(project_root or _default_root())
    train = pd.read_csv(project_root / "data" / "phase2" / "harmonized" / "train_features.csv")
    test  = pd.read_csv(project_root / "data" / "phase2" / "harmonized" / "test_features.csv")
    with open(project_root / "models" / "feature_cols.pkl", "rb") as f:
        cols = [c for c in pickle.load(f) if c in train.columns]

    from sklearn.preprocessing import StandardScaler
    Xtr_raw = train[cols].to_numpy(dtype=float)
    Xte_raw = test[cols].to_numpy(dtype=float)
    ytr = train["soh"].to_numpy(dtype=float)
    yte = test["soh"].to_numpy(dtype=float)
    # SOH is already in [0, 1] in the harmonised CSV? Defensive rescale.
    if ytr.max() > 2:
        ytr = ytr / 100.0
        yte = yte / 100.0

    scaler = StandardScaler().fit(Xtr_raw)
    Xtr = scaler.transform(Xtr_raw)
    Xte = scaler.transform(Xte_raw)

    schedules = ["random", "easy_first", "hard_first"]
    all_runs: List[TrainResult] = []
    t0 = time.perf_counter()
    for s in schedules:
        for sd in seeds:
            all_runs.append(_fit_one(Xtr, ytr, Xte, yte, s, sd, epochs=epochs))
    wall = round(time.perf_counter() - t0, 2)

    def _agg(runs: List[TrainResult]) -> Dict[str, float]:
        maes = np.array([r.test_mae for r in runs])
        tails = np.array([r.tail_mae for r in runs if not np.isnan(r.tail_mae)])
        return {
            "test_mae_mean": round(float(maes.mean()), 5),
            "test_mae_std":  round(float(maes.std(ddof=0)), 5),
            "tail_mae_mean": round(float(tails.mean()), 5) if len(tails) else None,
            "tail_mae_std":  round(float(tails.std(ddof=0)), 5) if len(tails) else None,
            "n_seeds": len(runs),
        }

    agg = {s: _agg([r for r in all_runs if r.schedule == s]) for s in schedules}

    # Verdict: does the best curriculum beat random by more than one seed-std?
    random_tail = agg["random"]["tail_mae_mean"]
    random_tail_std = agg["random"]["tail_mae_std"]
    winners = []
    for s in ("easy_first", "hard_first"):
        if agg[s]["tail_mae_mean"] is None or random_tail is None:
            continue
        gap = random_tail - agg[s]["tail_mae_mean"]
        if gap > random_tail_std:
            winners.append((s, round(gap, 5)))

    verdict = ("no_significant_gain"
               if not winners else
               f"{winners[0][0]} beats random by {winners[0][1]}")

    out = {
        "setup": {
            "n_train": int(len(ytr)),
            "n_test":  int(len(yte)),
            "n_features": len(cols),
            "seeds": list(seeds),
            "epochs": epochs,
            "wall_seconds": wall,
            "tail_threshold_soh": 0.8,
        },
        "aggregates": agg,
        "verdict": verdict,
        "all_runs": [asdict(r) for r in all_runs],
        "note": (
            "Curriculum learning applied to a 2-layer MLP. Difficulty = "
            "|SOH - 1.0| (knee cycles are 'hard'). If the mean tail-MAE "
            "gap between schedules is smaller than the random-baseline "
            "seed std, the experiment is inconclusive and we do not "
            "claim a curriculum benefit."
        ),
    }
    out_dir = project_root / "results" / "phase6"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "curriculum.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


def _format(out: Dict) -> str:
    s = out["setup"]
    lines = [
        "Curriculum learning (MLP)", "=" * 60,
        f"n_train={s['n_train']}  n_test={s['n_test']}  features={s['n_features']}  "
        f"seeds={len(s['seeds'])}  epochs={s['epochs']}  wall={s['wall_seconds']}s",
        "",
        f"{'schedule':<14} {'test MAE':>20} {'tail MAE (SOH<=0.8)':>24}",
    ]
    for sched, a in out["aggregates"].items():
        mae = f"{a['test_mae_mean']:.4f} +- {a['test_mae_std']:.4f}"
        tail = (f"{a['tail_mae_mean']:.4f} +- {a['tail_mae_std']:.4f}"
                if a["tail_mae_mean"] is not None else "(no tail samples)")
        lines.append(f"{sched:<14} {mae:>20} {tail:>24}")
    lines.append("")
    lines.append(f"Verdict : {out['verdict']}")
    lines.append("")
    lines.append(out["note"])
    return "\n".join(lines)


def main() -> int:
    out = run_curriculum()
    print(_format(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
