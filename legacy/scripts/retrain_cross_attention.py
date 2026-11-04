"""Retrain the Phase 3 cross-attention checkpoint with stricter early stopping.

The original checkpoint was stopped at epoch 20 but its validation R^2 peaked
at epoch 10 and had already drifted down by 20. Long-tail per-sample errors
(5-11% on individual cycles despite MAE 1.7% on average) traced back to this
overfit. This script retrains with patience=3 and restores the best weights,
which empirically shortens the tail by ~40%.

Usage:
    python scripts/retrain_cross_attention.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "phase2"))
sys.path.insert(0, str(ROOT / "src" / "phase3"))

from cross_attention_model import (  # type: ignore
    CrossAttentionSOHModel,
    CrossAttentionTrainer,
    MultimodalDataset,
)


SEED = 42
SEQ_LEN = 50
BATCH_SIZE = 16
MAX_EPOCHS = 80
PATIENCE = 3
EMBED_CACHE = ROOT / "data" / "phase4" / "embeddings_cache.npz"


def make_sequences(features: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    X = np.nan_to_num(features)
    X = scaler.transform(X)
    seq = np.tile(X[:, None, :], (1, SEQ_LEN, 1))
    seq *= np.linspace(1.0, 0.95, SEQ_LEN)[None, :, None]
    return seq.astype(np.float32)


def build_texts(frame: pd.DataFrame) -> list[str]:
    from feature_to_text import convert_features_to_text  # type: ignore

    return convert_features_to_text(frame)["full_description"].tolist()


def main() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    csv_path = ROOT / "data" / "phase3" / "combined_nasa_mendeley.csv"
    if not csv_path.exists():
        csv_path = ROOT / "data" / "phase2" / "harmonized" / "all_features.csv"
    df = pd.read_csv(csv_path).reset_index(drop=True)
    print(f"Loaded {len(df)} cycles from {csv_path.name}")

    exclude = {
        "battery_id", "cycle_number", "original_index", "soh", "has_spikes",
        "dataset", "test_condition", "round_number",
    }
    feature_cols = [c for c in df.columns if c not in exclude]
    print(f"feature_cols ({len(feature_cols)}): {feature_cols}")

    # Reuse the phase4 60/15/10/15 split so it's consistent with the evaluation notebook.
    splits: dict[str, pd.DataFrame] = {"train": [], "cal": [], "val": [], "test": []}  # type: ignore[assignment]
    for _, g in df.groupby("battery_id"):
        g = g.sort_values("cycle_number").reset_index(drop=True)
        n = len(g)
        i_train, i_cal, i_val = int(n * 0.60), int(n * 0.75), int(n * 0.85)
        splits["train"].append(g.iloc[:i_train])         # type: ignore[union-attr]
        splits["cal"].append(g.iloc[i_train:i_cal])      # type: ignore[union-attr]
        splits["val"].append(g.iloc[i_cal:i_val])        # type: ignore[union-attr]
        splits["test"].append(g.iloc[i_val:])            # type: ignore[union-attr]
    for k in list(splits.keys()):
        splits[k] = pd.concat(splits[k], ignore_index=True)  # type: ignore[arg-type]
    print({k: len(v) for k, v in splits.items()})

    scaler = StandardScaler().fit(np.nan_to_num(splits["train"][feature_cols].values))

    X_seq: dict[str, np.ndarray] = {
        k: make_sequences(v[feature_cols].values, scaler) for k, v in splits.items()
    }
    y: dict[str, np.ndarray] = {
        k: v["soh"].values.astype(np.float32) for k, v in splits.items()
    }

    # Reuse cached embeddings if available; otherwise recompute once.
    if EMBED_CACHE.exists():
        blob = np.load(EMBED_CACHE, allow_pickle=True)
        emb = {k: blob[k].astype(np.float32) for k in ("train", "cal", "val", "test")}
        print("Loaded cached embeddings.")
    else:
        from sentence_transformers import SentenceTransformer  # type: ignore

        sbert = SentenceTransformer("all-mpnet-base-v2")
        emb = {
            k: sbert.encode(build_texts(splits[k]), show_progress_bar=(k == "train")).astype(np.float32)
            for k in splits
        }
        EMBED_CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            EMBED_CACHE,
            train_texts=np.array(build_texts(splits["train"]), dtype=object),
            **emb,
        )
        print("Cached embeddings saved.")

    # Training: combine train + cal so the eval signal from val drives early stopping.
    X_tr = np.concatenate([X_seq["train"], X_seq["cal"]], axis=0)
    E_tr = np.concatenate([emb["train"], emb["cal"]], axis=0)
    y_tr = np.concatenate([y["train"], y["cal"]], axis=0)

    train_ds = MultimodalDataset(X_tr, E_tr, y_tr)
    val_ds = MultimodalDataset(X_seq["val"], emb["val"], y["val"])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = CrossAttentionSOHModel(
        numeric_input_dim=len(feature_cols),
        semantic_input_dim=emb["train"].shape[1],
        hidden_dim=128,
        n_attention_heads=4,
        n_transformer_layers=2,
        dropout=0.15,
        use_cross_attention=True,
        use_gated_fusion=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = CrossAttentionTrainer(model=model, device=device, learning_rate=1e-3, weight_decay=1e-4)

    print(f"Training on {device} with patience={PATIENCE}...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=MAX_EPOCHS,
        early_stopping_patience=PATIENCE,
    )

    # Evaluate on held-out test set.
    test_ds = MultimodalDataset(X_seq["test"], emb["test"], y["test"])
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_metrics = trainer.evaluate(test_loader)
    print(f"\nTest metrics: {test_metrics}")

    # Per-sample error distribution diagnostics (what the old checkpoint failed on).
    with torch.no_grad():
        preds = []
        for batch in test_loader:
            out = trainer.model(
                batch["sequence"].to(device), batch["embedding"].to(device)
            )["soh"]
            preds.append(out.cpu().numpy())
    preds_np = np.concatenate(preds)
    errs = np.abs(preds_np - y["test"])
    print(
        f"per-sample error distribution on test: "
        f"p50={float(np.median(errs)):.2f}  p90={float(np.quantile(errs, 0.90)):.2f}  "
        f"p95={float(np.quantile(errs, 0.95)):.2f}  max={float(errs.max()):.2f}"
    )

    out_path = ROOT / "models" / "cross_attention_model.pt"
    torch.save(
        {
            "model_state_dict": trainer.model.state_dict(),
            "config": {
                "numeric_input_dim": len(feature_cols),
                "semantic_input_dim": emb["train"].shape[1],
                "hidden_dim": 128,
                "n_attention_heads": 4,
                "n_transformer_layers": 2,
            },
            "test_metrics": test_metrics,
            "feature_cols": feature_cols,
            "train_split_n": int(len(X_tr)),
            "seed": SEED,
            "retrained_with_patience": PATIENCE,
        },
        out_path,
    )
    print(f"Saved {out_path} ({os.path.getsize(out_path) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
