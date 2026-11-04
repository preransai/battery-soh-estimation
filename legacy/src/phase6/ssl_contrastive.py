"""Self-supervised contrastive pretraining + linear probe.

SimCLR-style idea adapted to tabular battery features:

    1. Unlabelled pretext: for each cycle, build two augmented views
       (gaussian noise + feature dropout) and train a small encoder f()
       so that f(view_a) and f(view_b) of the same cycle agree while
       disagreeing with other cycles in the batch (NT-Xent loss).
    2. Freeze f(), train a linear Ridge probe on top using only the
       original labelled cycles.

Compared against:

    - linear_on_raw   : Ridge on scaled raw features (no pretraining)
    - linear_on_pca   : Ridge on PCA(k=embed_dim) features -- cheap
                        unsupervised baseline that the contrastive head
                        has to beat to justify itself.

All three probes see the same labelled split so the only moving part
is the representation. We don't claim SSL is a win unless contrastive
beats BOTH raw and PCA by more than one seed-std. This is a stricter
bar than most papers apply and it catches the common failure mode
where SSL just reinvents PCA on tabular data.
"""

from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn


def _default_root() -> Path:
    return Path(__file__).resolve().parents[2]


class Encoder(nn.Module):
    def __init__(self, in_dim: int, embed_dim: int = 16, hidden: int = 64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, embed_dim),
        )
        # Projection head used ONLY during contrastive training
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def embed(self, x):
        return self.backbone(x)

    def forward(self, x):
        h = self.backbone(x)
        z = self.projector(h)
        return nn.functional.normalize(z, dim=-1)


def augment(x: torch.Tensor, noise_std: float, drop_p: float,
            rng: torch.Generator) -> torch.Tensor:
    noise = torch.randn(x.shape, generator=rng) * noise_std
    mask = (torch.rand(x.shape, generator=rng) > drop_p).float()
    return (x + noise) * mask


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """InfoNCE (SimCLR) loss for a batch of matched pairs."""
    n = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)                     # (2n, d)
    sim = z @ z.T / temperature                        # (2n, 2n)
    # Mask the diagonal (self-similarity)
    mask_self = torch.eye(2 * n, dtype=torch.bool)
    sim = sim.masked_fill(mask_self, -9e15)
    # Positives: i <-> i+n and i+n <-> i
    targets = torch.arange(2 * n)
    targets = (targets + n) % (2 * n)
    return nn.functional.cross_entropy(sim, targets)


def _pretrain(
    X: np.ndarray,
    embed_dim: int = 16,
    epochs: int = 40,
    batch_size: int = 64,
    noise_std: float = 0.1,
    drop_p: float = 0.2,
    seed: int = 0,
) -> Encoder:
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed)
    model = Encoder(in_dim=X.shape[1], embed_dim=embed_dim)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    X_t = torch.from_numpy(X).float()
    n = len(X_t)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=rng)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            if len(idx) < 4:
                continue
            xb = X_t[idx]
            v1 = augment(xb, noise_std, drop_p, rng)
            v2 = augment(xb, noise_std, drop_p, rng)
            z1 = model(v1)
            z2 = model(v2)
            loss = nt_xent(z1, z2)
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def _ridge_probe(X_tr: np.ndarray, y_tr: np.ndarray,
                 X_te: np.ndarray, y_te: np.ndarray) -> Dict[str, float]:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X_tr)
    model = Ridge(alpha=1.0).fit(sc.transform(X_tr), y_tr)
    pred = model.predict(sc.transform(X_te))
    mae = float(np.mean(np.abs(pred - y_te)))
    r2 = float(1 - np.sum((y_te - pred) ** 2) / max(np.sum((y_te - y_te.mean()) ** 2), 1e-9))
    return {"mae": round(mae, 5), "r2": round(r2, 5)}


def run_ssl(project_root: Optional[Path] = None,
            seeds: tuple = (0, 1, 2),
            embed_dim: int = 16,
            pretrain_epochs: int = 40) -> Dict:
    project_root = Path(project_root or _default_root())
    train = pd.read_csv(project_root / "data" / "phase2" / "harmonized" / "train_features.csv")
    test  = pd.read_csv(project_root / "data" / "phase2" / "harmonized" / "test_features.csv")
    with open(project_root / "models" / "feature_cols.pkl", "rb") as f:
        cols = [c for c in pickle.load(f) if c in train.columns]

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    scaler = StandardScaler().fit(train[cols].to_numpy(dtype=float))
    Xtr = scaler.transform(train[cols].to_numpy(dtype=float))
    Xte = scaler.transform(test[cols].to_numpy(dtype=float))
    ytr = train["soh"].to_numpy(dtype=float)
    yte = test["soh"].to_numpy(dtype=float)

    # Baseline 1: Ridge on raw scaled features
    baseline_raw = _ridge_probe(Xtr, ytr, Xte, yte)

    # Baseline 2: Ridge on PCA(embed_dim)
    pca = PCA(n_components=min(embed_dim, Xtr.shape[1])).fit(Xtr)
    baseline_pca = _ridge_probe(pca.transform(Xtr), ytr, pca.transform(Xte), yte)

    # Contrastive: pretrain on union of train+test features (labels unused),
    # then probe on labelled train -> eval labelled test.
    X_all = np.concatenate([Xtr, Xte], axis=0)

    t0 = time.perf_counter()
    per_seed: List[Dict[str, float]] = []
    for sd in seeds:
        enc = _pretrain(X_all, embed_dim=embed_dim, epochs=pretrain_epochs, seed=sd)
        enc.eval()
        with torch.no_grad():
            Htr = enc.embed(torch.from_numpy(Xtr).float()).numpy()
            Hte = enc.embed(torch.from_numpy(Xte).float()).numpy()
        per_seed.append(_ridge_probe(Htr, ytr, Hte, yte))
    wall = round(time.perf_counter() - t0, 2)

    maes = np.array([r["mae"] for r in per_seed])
    r2s  = np.array([r["r2"]  for r in per_seed])
    contrastive = {
        "mae_mean": round(float(maes.mean()), 5),
        "mae_std":  round(float(maes.std(ddof=0)), 5),
        "r2_mean":  round(float(r2s.mean()), 5),
        "r2_std":   round(float(r2s.std(ddof=0)), 5),
        "per_seed": per_seed,
    }

    # Strict verdict: contrastive MAE must beat BOTH baselines by > mae_std.
    beats_raw = baseline_raw["mae"] - contrastive["mae_mean"] > contrastive["mae_std"]
    beats_pca = baseline_pca["mae"] - contrastive["mae_mean"] > contrastive["mae_std"]
    if beats_raw and beats_pca:
        verdict = "contrastive_beats_both_baselines"
    elif beats_raw or beats_pca:
        verdict = "contrastive_beats_one_baseline_only"
    else:
        verdict = "contrastive_no_benefit_over_linear_baselines"

    out = {
        "setup": {
            "n_unlabelled_pretrain": int(len(X_all)),
            "n_train_probe": int(len(ytr)),
            "n_test_probe":  int(len(yte)),
            "embed_dim": embed_dim,
            "pretrain_epochs": pretrain_epochs,
            "seeds": list(seeds),
            "pretrain_wall_seconds": wall,
        },
        "baseline_raw_ridge": baseline_raw,
        "baseline_pca_ridge": baseline_pca,
        "contrastive_probe": contrastive,
        "verdict": verdict,
        "note": (
            "Contrastive encoder uses NT-Xent loss on (gaussian + dropout) "
            "augmentations. The strict bar for success is beating BOTH "
            "raw-Ridge and PCA-Ridge by at least one seed-std. On 504-row "
            "tabular features this bar is hard to clear, and a negative "
            "result is a meaningful finding (SSL gains need scale)."
        ),
    }
    out_dir = project_root / "results" / "phase6"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "ssl_contrastive.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


def _format(out: Dict) -> str:
    s = out["setup"]
    lines = [
        "Self-supervised contrastive pretraining + linear probe", "=" * 60,
        f"unlabelled={s['n_unlabelled_pretrain']}  train/test probe={s['n_train_probe']}/{s['n_test_probe']}"
        f"  embed={s['embed_dim']}  seeds={len(s['seeds'])}  wall={s['pretrain_wall_seconds']}s",
        "",
        f"{'probe':<28} {'MAE':>12} {'R^2':>10}",
        f"{'Ridge on raw scaled':<28} {out['baseline_raw_ridge']['mae']:>12.4f} {out['baseline_raw_ridge']['r2']:>10.4f}",
        f"{'Ridge on PCA':<28} {out['baseline_pca_ridge']['mae']:>12.4f} {out['baseline_pca_ridge']['r2']:>10.4f}",
        f"{'Ridge on contrastive':<28} "
        f"{out['contrastive_probe']['mae_mean']:>6.4f}+-{out['contrastive_probe']['mae_std']:.4f}  "
        f"{out['contrastive_probe']['r2_mean']:>6.4f}+-{out['contrastive_probe']['r2_std']:.4f}",
        "",
        f"Verdict : {out['verdict']}",
        "",
        out["note"],
    ]
    return "\n".join(lines)


def main() -> int:
    out = run_ssl()
    print(_format(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
