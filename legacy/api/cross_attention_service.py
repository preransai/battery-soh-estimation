"""Runtime wrapper around the Phase 3 cross-attention SOH model.

The model in ``models/cross_attention_model.pt`` was trained on 18-dim numeric
feature vectors (turned into a seq_len=50 pseudo-sequence) plus a 768-dim
Sentence-BERT embedding of the diagnostic text. This service makes that model
usable from the FastAPI container:

    1. Loads the PyTorch checkpoint and its config at startup.
    2. Loads the Phase 2 ``FeatureToText`` converter for template text.
    3. Loads Sentence-BERT (mpnet-base-v2) once.
    4. Reads per-feature means + stds from ``models/phase4_state.pkl`` so it
       can impute the ~12 rich features the user interface does not collect.
    5. Exposes a single ``predict(features)`` entrypoint returning a
       float SOH percentage.

If anything fails to load the service flips ``self.loaded`` to False; callers
should fall back to the existing sklearn predictor.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Imputation config — features expected by the cross-attention model but not
# always collected by the web UI. Imputed from training-set statistics.
# --------------------------------------------------------------------------- #

CROSS_ATTENTION_FEATURES = [
    "capacity", "voltage_mean", "voltage_std", "voltage_min", "voltage_max", "voltage_range",
    "current_mean", "current_std",
    "temp_mean", "temp_max", "temp_delta", "temp_std",
    "ica_peak_voltage", "ica_peak_magnitude", "ica_mean", "ica_std",
    "dcir", "discharge_time",
]


def _derive_if_missing(features: Dict[str, Any], means: Dict[str, float]) -> Dict[str, float]:
    """Fill in any missing feature with a sensible derivation or the training mean.

    Priority for fills:
        - voltage_range = voltage_max - voltage_min when both known
        - voltage_mean  = (voltage_min + voltage_max)/2 when both known
        - temp_max      = temp_mean + temp_delta/2 when both known
        - temp_delta    = 2*(temp_max - temp_mean) when both known
        - otherwise, the training mean for that feature.
    """
    out: Dict[str, float] = {}
    for k, v in features.items():
        if v is None:
            continue
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            continue

    if "voltage_min" in out and "voltage_max" in out:
        out.setdefault("voltage_range", out["voltage_max"] - out["voltage_min"])
        out.setdefault("voltage_mean", (out["voltage_min"] + out["voltage_max"]) / 2.0)
    if "temp_mean" in out and "temp_delta" in out:
        out.setdefault("temp_max", out["temp_mean"] + out["temp_delta"] / 2.0)

    for col in CROSS_ATTENTION_FEATURES:
        if col not in out:
            out[col] = float(means.get(col, 0.0))
    return out


# --------------------------------------------------------------------------- #
# Service
# --------------------------------------------------------------------------- #


class CrossAttentionService:
    """Lazy-loaded service holding model, tokenizer, SBERT, and scaler statistics."""

    def __init__(self) -> None:
        self.loaded: bool = False
        self.error: Optional[str] = None
        self._lock = Lock()
        self.model = None
        self.converter = None
        self.sbert = None
        self.scaler_mean: np.ndarray = np.zeros(len(CROSS_ATTENTION_FEATURES), dtype=np.float32)
        self.scaler_std: np.ndarray = np.ones(len(CROSS_ATTENTION_FEATURES), dtype=np.float32)
        self.impute_means: Dict[str, float] = {}
        self.device: str = "cpu"
        # Retriever for per-request KNN imputation — populated from phase4_state.pkl.
        self._ret_cols: list[str] = []
        self._ret_means: np.ndarray | None = None
        self._ret_stds: np.ndarray | None = None
        self._ret_matrix: np.ndarray | None = None  # L2-normalised, standardised training rows
        self._ret_raw_matrix: np.ndarray | None = None  # unstandardised rows for imputation
        self._ret_descriptions: list[str] = []
        self._ret_meta: list[Dict[str, Any]] = []

    # --- lifecycle -------------------------------------------------------- #

    def load(self) -> bool:
        """Load model + SBERT + scaler stats. Safe to call multiple times."""
        if self.loaded:
            return True
        with self._lock:
            if self.loaded:
                return True
            try:
                self._load_impl()
                self.loaded = True
                self.error = None
                return True
            except Exception as exc:  # noqa: BLE001 - we want any failure to degrade gracefully
                self.error = f"{type(exc).__name__}: {exc}"
                print(f"cross-attention service disabled: {self.error}")
                return False

    def _load_impl(self) -> None:
        import sys

        sys.path.insert(0, str(_ROOT / "src"))
        sys.path.insert(0, str(_ROOT / "src" / "phase2"))
        sys.path.insert(0, str(_ROOT / "src" / "phase3"))

        import torch  # type: ignore

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt_path = _ROOT / "models" / "cross_attention_model.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"missing checkpoint at {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        cfg = ckpt["config"]

        from cross_attention_model import CrossAttentionSOHModel  # type: ignore

        model = CrossAttentionSOHModel(
            numeric_input_dim=cfg["numeric_input_dim"],
            semantic_input_dim=cfg["semantic_input_dim"],
            hidden_dim=cfg["hidden_dim"],
            n_attention_heads=cfg["n_attention_heads"],
            n_transformer_layers=cfg["n_transformer_layers"],
        ).to(self.device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        self.model = model
        self._torch = torch

        from feature_to_text import FeatureToText  # type: ignore

        self.converter = FeatureToText()

        # Sentence-BERT: prefer a local pre-cached copy (see Dockerfile) to avoid
        # hitting the HF Hub at runtime.
        from sentence_transformers import SentenceTransformer  # type: ignore

        model_dir = os.environ.get(
            "SBERT_MODEL_DIR",
            str(_ROOT / "models" / "sbert-all-mpnet-base-v2"),
        )
        sbert = (
            SentenceTransformer(model_dir)
            if os.path.isdir(model_dir)
            else SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        )
        self.sbert = sbert

        # Scaler statistics + impute means come from the harmonised training corpus.
        # Prefer a pre-computed file; fall back to computing from CSV if absent.
        stats_path = _ROOT / "models" / "cross_attention_stats.pkl"
        if stats_path.exists():
            with open(stats_path, "rb") as f:
                stats = pickle.load(f)
            self.scaler_mean = np.asarray(stats["mean"], dtype=np.float32)
            self.scaler_std = np.asarray(stats["std"], dtype=np.float32)
            self.impute_means = dict(stats["impute_means"])
            self.converter.set_reference_values(stats["reference_row"])
        else:
            self._bootstrap_from_csv()

        # Retriever for per-request imputation.
        self._load_retriever()

        print(
            f"cross-attention service ready on {self.device} "
            f"(num_features={len(CROSS_ATTENTION_FEATURES)}, sbert_dir={'local' if os.path.isdir(model_dir) else 'hub'})"
        )

    def _load_retriever(self) -> None:
        """Load the Phase 4 retriever so we can KNN-impute the 12 rich features
        the web UI doesn't collect. Without this, every prediction gets the
        global training mean for ICA/DCIR/etc. and the model sees near-identical
        inputs, collapsing to a constant output."""
        phase4_path = _ROOT / "models" / "phase4_state.pkl"
        if not phase4_path.exists():
            print("cross-attention: phase4_state.pkl missing, KNN imputation disabled")
            return
        with open(phase4_path, "rb") as f:
            state = pickle.load(f)
        ret = state.get("retriever")
        if not ret:
            return
        self._ret_cols = list(ret["feature_cols"])
        self._ret_means = np.asarray(ret["mean"], dtype=np.float32)
        self._ret_stds = np.asarray(ret["std"], dtype=np.float32)
        self._ret_matrix = np.asarray(ret["matrix"], dtype=np.float32)
        self._ret_descriptions = list(ret["descriptions"])
        self._ret_meta = list(ret["metadata"])
        # Recover raw (unstandardised) values for imputation: matrix is L2-normalised
        # per row after z-score, so we can't invert it exactly. Store the raw training
        # rows in a parallel matrix reconstructed from the standardised ones.
        # (ret["matrix"] = l2_normalize((X - mean) / std), so we lose norm info. We
        # instead rebuild raw values from the descriptions' metadata — but those are
        # strings, not numbers. Cleanest fix: also persist raw features in fit_phase4
        # and read them here when available.)
        raw_mat = ret.get("raw_matrix")
        if raw_mat is not None:
            self._ret_raw_matrix = np.asarray(raw_mat, dtype=np.float32)
        print(f"cross-attention retriever loaded: {len(self._ret_descriptions)} docs over {len(self._ret_cols)} dims")

    def _bootstrap_from_csv(self) -> None:
        """Fallback: compute scaler + impute means from the harmonised CSV at first run."""
        import pandas as pd

        csv_path = _ROOT / "data" / "phase3" / "combined_nasa_mendeley.csv"
        if not csv_path.exists():
            csv_path = _ROOT / "data" / "phase2" / "harmonized" / "all_features.csv"
        if not csv_path.exists():
            raise FileNotFoundError("no harmonised feature CSV available for bootstrap")
        df = pd.read_csv(csv_path)
        X = np.nan_to_num(
            np.asarray([[row.get(c, 0.0) for c in CROSS_ATTENTION_FEATURES] for row in df.to_dict("records")],
                       dtype=np.float32))
        self.scaler_mean = X.mean(axis=0)
        self.scaler_std = X.std(axis=0); self.scaler_std[self.scaler_std < 1e-8] = 1.0
        self.impute_means = {c: float(X[:, i].mean()) for i, c in enumerate(CROSS_ATTENTION_FEATURES)}
        self.converter.set_reference_values(df.iloc[0].to_dict())

    # --- inference -------------------------------------------------------- #

    def _knn_impute(self, features: Dict[str, Any]) -> tuple[Dict[str, float], Optional[str]]:
        """Find the training cycle whose *provided* features are closest, then borrow
        its full feature vector + description. Returns (imputed_features, description).

        The description is None when the retriever isn't available; callers should
        fall back to the template generator."""
        overlap = [c for c in self._ret_cols if c in features and features[c] is not None]
        have_retriever = (self._ret_matrix is not None and self._ret_raw_matrix is not None and overlap)
        if not have_retriever:
            return _derive_if_missing(features, self.impute_means), None

        # Build a query in the retriever's standardised space, using ONLY the
        # dimensions the user actually provided. Treating missing dims as "at the
        # mean" would make high-variance columns like ICA/DCIR dominate the
        # similarity and return physically-inconsistent neighbours.
        provided_idx: list[int] = []
        provided_q: list[float] = []
        for i, col in enumerate(self._ret_cols):
            v = features.get(col)
            if v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            provided_idx.append(i)
            provided_q.append((fv - float(self._ret_means[i])) / float(self._ret_stds[i]))
        if not provided_idx:
            return _derive_if_missing(features, self.impute_means), None
        q_sub = np.asarray(provided_q, dtype=np.float32)
        q_norm = float(np.linalg.norm(q_sub))
        if q_norm < 1e-8:
            return _derive_if_missing(features, self.impute_means), None
        q_unit = q_sub / q_norm
        # Build a parallel submatrix from raw rows standardised by the retriever scaler,
        # L2-normalised over the SAME subset of dimensions used in the query.
        sub_mat = self._ret_raw_matrix[:, provided_idx]
        sub_z = (sub_mat - self._ret_means[provided_idx]) / self._ret_stds[provided_idx]
        sub_norms = np.linalg.norm(sub_z, axis=1, keepdims=True)
        sub_unit = sub_z / np.clip(sub_norms, 1e-8, None)
        sims = sub_unit @ q_unit

        # Average over the top-K neighbours for the aux features (not just the single
        # nearest). A single cycle's ICA/DCIR can be noisy; averaging ~5 similar cycles
        # gives a much more stable imputation and avoids biasing the downstream model
        # with outlier values. Use similarity-weighted mean so closer neighbours count more.
        K = 5
        top_idx = np.argsort(-sims)[:K]
        top_sims = np.clip(sims[top_idx], 1e-6, None)
        w = top_sims / top_sims.sum()
        neighbour_raw = (self._ret_raw_matrix[top_idx] * w[:, None]).sum(axis=0)
        # The diagnostic description comes from the single nearest neighbour — it's
        # specific, human-readable, and already conditioned on the retrieval.
        nearest = int(top_idx[0])
        neighbour_desc = self._ret_descriptions[nearest] if self._ret_descriptions else None

        # For every feature the UI DID provide, keep the UI value. For every other
        # feature, take the neighbour's raw value. Also derive voltage_range /
        # voltage_mean / temp_max if not explicitly present.
        imputed: Dict[str, float] = {}
        for i, col in enumerate(self._ret_cols):
            imputed[col] = float(neighbour_raw[i])
        for k, v in features.items():
            if v is None:
                continue
            try:
                imputed[k] = float(v)
            except (TypeError, ValueError):
                continue
        if "voltage_min" in imputed and "voltage_max" in imputed:
            imputed.setdefault("voltage_range", imputed["voltage_max"] - imputed["voltage_min"])
            imputed.setdefault("voltage_mean", (imputed["voltage_min"] + imputed["voltage_max"]) / 2.0)
        if "temp_mean" in imputed and "temp_delta" in imputed:
            imputed.setdefault("temp_max", imputed["temp_mean"] + imputed["temp_delta"] / 2.0)
        # Any column still missing falls back to the global training mean.
        for col in CROSS_ATTENTION_FEATURES:
            imputed.setdefault(col, float(self.impute_means.get(col, 0.0)))
        return imputed, neighbour_desc

    def _build_sequence(self, imputed: Dict[str, float]) -> np.ndarray:
        row = np.array([imputed[c] for c in CROSS_ATTENTION_FEATURES], dtype=np.float32)
        row = np.nan_to_num(row)
        z = (row - self.scaler_mean) / self.scaler_std
        seq = np.tile(z[None, None, :], (1, 50, 1)).astype(np.float32)
        seq *= np.linspace(1.0, 0.95, 50)[None, :, None].astype(np.float32)
        return seq

    def _build_embedding(self, text: str) -> np.ndarray:
        vec = self.sbert.encode([text], show_progress_bar=False).astype(np.float32)
        return vec

    def predict(self, features: Dict[str, Any]) -> float:
        """Run a single SOH prediction through the cross-attention model.

        Strategy:
            1. KNN-impute the 12 features the UI doesn't collect from the nearest
               training cycle (per Phase 4 retriever).
            2. Build the 18-dim numeric sequence from the imputed row.
            3. Use the retrieved neighbour's description as the semantic channel;
               if the retriever is unavailable, fall back to the template generator.

        Raises RuntimeError if the service wasn't successfully loaded.
        """
        if not self.loaded:
            raise RuntimeError("cross-attention service not loaded")
        imputed, neighbour_desc = self._knn_impute(features)
        seq = self._build_sequence(imputed)
        if neighbour_desc:
            text = neighbour_desc
        else:
            text = self.converter.convert_single(imputed).full_description
        emb = self._build_embedding(text)
        torch = self._torch
        seq_t = torch.from_numpy(seq).to(self.device)
        emb_t = torch.from_numpy(emb).to(self.device)
        with torch.no_grad():
            out = self.model(seq_t, emb_t)
        soh = float(out["soh"].cpu().item())
        return float(np.clip(soh, 0.0, 100.0))

    # --- metadata --------------------------------------------------------- #

    def info(self) -> Dict[str, Any]:
        return {
            "loaded": self.loaded,
            "device": self.device,
            "error": self.error,
            "num_features": len(CROSS_ATTENTION_FEATURES),
            "semantic_dim": 768,
            "provider": "cross_attention_phase3",
        }


# Module-level singleton used by api.main.
SERVICE = CrossAttentionService()
