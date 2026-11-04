"""Retrieval-Augmented Prognostics (RAP).

Replaces the rule-based ``feature_to_text`` templates with a retrieval-then-
generate pipeline:

    1. Index every training cycle as (numeric feature vector, ground-truth
       SOH, descriptive text). Embeddings are cached on disk.

    2. For a query cycle, retrieve the K nearest training cycles in
       feature space (cosine on standardised features; uses a NumPy linear
       scan if FAISS isn't available).

    3. Build a structured prompt containing the retrieved analogues and
       call an LLM (OpenAI by default; configurable). The LLM emits a
       free-form diagnostic paragraph that is richer than the templates.

    4. Embed that paragraph with Sentence-BERT exactly as Phase 3 already
       does, so the downstream cross-attention model is unchanged.

The whole pipeline degrades gracefully:

    - No FAISS         -> NumPy nearest-neighbours
    - No OpenAI key    -> falls back to the existing template generator
    - Cache hit on disk -> skips the LLM call entirely

This makes the module deterministic and reproducible across runs once the
cache is populated, which matters for ablation studies.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except ImportError:
    faiss = None  # type: ignore
    _HAS_FAISS = False


# --------------------------------------------------------------------------- #
# Retrieval index
# --------------------------------------------------------------------------- #


@dataclass
class RetrievalHit:
    distance: float
    soh: float
    cycle_number: int
    battery_id: str
    description: str
    features: Dict[str, float] = field(default_factory=dict)


class CycleRetriever:
    """K-NN over standardised cycle features.

    Cosine similarity (i.e. inner product after L2 normalisation) gives the
    most stable retrieval here because feature magnitudes vary across
    chemistries. We compute mean/std on the indexed corpus so query-time
    standardisation is consistent.
    """

    def __init__(self, use_faiss: bool = True):
        self.use_faiss = use_faiss and _HAS_FAISS
        self.feature_cols: List[str] = []
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.index = None
        self.metadata: Optional[pd.DataFrame] = None

    @staticmethod
    def _l2_normalize(x: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.clip(norm, 1e-8, None)

    def fit(
        self,
        features_df: pd.DataFrame,
        feature_cols: Sequence[str],
        descriptions: Sequence[str],
    ) -> "CycleRetriever":
        if len(features_df) != len(descriptions):
            raise ValueError("features_df and descriptions must have the same length")
        self.feature_cols = list(feature_cols)
        X = features_df[self.feature_cols].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0)
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std < 1e-8] = 1.0
        Xz = (X - self.mean) / self.std
        Xz = self._l2_normalize(Xz).astype(np.float32)

        meta_cols = [c for c in ("battery_id", "cycle_number", "soh") if c in features_df.columns]
        self.metadata = features_df[meta_cols].copy().reset_index(drop=True)
        self.metadata["description"] = list(descriptions)
        self.metadata["_features"] = [
            features_df.iloc[i][self.feature_cols].to_dict() for i in range(len(features_df))
        ]

        if self.use_faiss:
            self.index = faiss.IndexFlatIP(Xz.shape[1])
            self.index.add(Xz)
        else:
            self.index = Xz  # NumPy fallback: store the matrix and search by hand
        return self

    def _standardise_query(self, features: Dict[str, float]) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Retriever not fitted")
        vec = np.array([features.get(c, 0.0) for c in self.feature_cols], dtype=np.float32)
        vec = np.nan_to_num(vec, nan=0.0)
        z = (vec - self.mean) / self.std
        return self._l2_normalize(z[None, :]).astype(np.float32)

    def query(self, features: Dict[str, float], k: int = 5) -> List[RetrievalHit]:
        if self.metadata is None:
            raise RuntimeError("Retriever not fitted")
        q = self._standardise_query(features)
        if self.use_faiss:
            sims, idx = self.index.search(q, k)
            sims, idx = sims[0], idx[0]
        else:
            sims_all = (self.index @ q.T).reshape(-1)
            idx = np.argsort(-sims_all)[:k]
            sims = sims_all[idx]
        hits: List[RetrievalHit] = []
        for sim, i in zip(sims, idx):
            row = self.metadata.iloc[int(i)]
            hits.append(
                RetrievalHit(
                    distance=float(1.0 - sim),
                    soh=float(row.get("soh", float("nan"))),
                    cycle_number=int(row.get("cycle_number", -1)),
                    battery_id=str(row.get("battery_id", "?")),
                    description=str(row["description"]),
                    features=dict(row.get("_features", {})),
                )
            )
        return hits


# --------------------------------------------------------------------------- #
# LLM diagnosis generation
# --------------------------------------------------------------------------- #


def _format_hits_for_prompt(hits: Sequence[RetrievalHit]) -> str:
    lines = []
    for i, h in enumerate(hits, start=1):
        lines.append(
            f"[Analogue {i}]  battery={h.battery_id}  cycle={h.cycle_number}  "
            f"SOH={h.soh:.2f}%  cosine_dist={h.distance:.3f}\n"
            f"    {h.description.strip()}"
        )
    return "\n\n".join(lines)


def _format_query_features(features: Dict[str, float]) -> str:
    keys = [
        "capacity",
        "voltage_mean",
        "voltage_std",
        "current_mean",
        "temp_mean",
        "temp_delta",
        "ica_peak_voltage",
        "ica_peak_magnitude",
        "dcir",
        "discharge_time",
    ]
    parts = []
    for k in keys:
        if k in features and features[k] is not None:
            parts.append(f"{k}={features[k]:.3f}")
    return ", ".join(parts)


PROMPT_TEMPLATE = """You are a senior battery diagnostic engineer. A new
discharge cycle has been measured. Use it together with K nearest historical
cycles (retrieved from a labelled corpus) to write a concise, technically
precise diagnostic paragraph. The paragraph will be fed to a sentence
embedding model that downstream feeds a regressor, so prioritise discriminative
physical detail (ICA peak shift, DCIR drift, thermal behaviour, capacity fade
direction) over vague language. Do not invent numbers; ground claims in the
features given.

Query cycle features:
{query}

Retrieved analogues (sorted by similarity, lower distance = more similar):
{analogues}

Write 3-5 sentences. Begin with the most likely degradation mode, then justify
it from the features and analogues, then note one operational implication.
"""


@dataclass
class LLMConfig:
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    max_tokens: int = 280


class LLMGenerator:
    """Wraps an LLM call with on-disk caching and a deterministic fallback."""

    def __init__(
        self,
        cache_dir: str | os.PathLike = "data/phase4/rap_cache",
        config: Optional[LLMConfig] = None,
        fallback: Optional[Callable[[Dict[str, float]], str]] = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or LLMConfig()
        self.fallback = fallback
        self._client = None

    def _maybe_init_client(self):
        if self._client is not None:
            return
        if self.config.provider != "openai":
            return
        if not os.environ.get("OPENAI_API_KEY"):
            return
        try:
            from openai import OpenAI  # type: ignore

            self._client = OpenAI()
        except ImportError:
            self._client = None

    def _cache_path(self, prompt: str) -> Path:
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:24]
        return self.cache_dir / f"{digest}.json"

    def _read_cache(self, prompt: str) -> Optional[str]:
        path = self._cache_path(prompt)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8")).get("text")
        except (json.JSONDecodeError, OSError):
            return None

    def _write_cache(self, prompt: str, text: str) -> None:
        path = self._cache_path(prompt)
        payload = {"text": text, "model": self.config.model, "ts": time.time()}
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def _call_openai(self, prompt: str) -> str:
        self._maybe_init_client()
        if self._client is None:
            raise RuntimeError("OpenAI client unavailable")
        resp = self._client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": "You are a battery diagnostics expert."},
                {"role": "user", "content": prompt},
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return resp.choices[0].message.content.strip()

    def generate(self, prompt: str, query_features: Dict[str, float]) -> str:
        cached = self._read_cache(prompt)
        if cached is not None:
            return cached
        try:
            text = self._call_openai(prompt)
        except Exception as exc:  # noqa: BLE001 - fallback by design
            if self.fallback is None:
                raise
            text = self.fallback(query_features)
            text = f"[fallback: {type(exc).__name__}] {text}"
        self._write_cache(prompt, text)
        return text


# --------------------------------------------------------------------------- #
# End-to-end pipeline
# --------------------------------------------------------------------------- #


@dataclass
class RAPDiagnosis:
    text: str
    hits: List[RetrievalHit]
    prompt: str
    used_llm: bool


class RetrievalAugmentedDiagnoser:
    """Glue: retrieve -> prompt -> LLM-generate (with caching and fallback)."""

    def __init__(
        self,
        retriever: CycleRetriever,
        generator: LLMGenerator,
        k: int = 5,
    ):
        self.retriever = retriever
        self.generator = generator
        self.k = int(k)

    def diagnose(self, query_features: Dict[str, float]) -> RAPDiagnosis:
        hits = self.retriever.query(query_features, k=self.k)
        prompt = PROMPT_TEMPLATE.format(
            query=_format_query_features(query_features),
            analogues=_format_hits_for_prompt(hits),
        )
        text = self.generator.generate(prompt, query_features)
        used_llm = not text.startswith("[fallback:")
        return RAPDiagnosis(text=text, hits=hits, prompt=prompt, used_llm=used_llm)

    def diagnose_dataframe(self, features_df: pd.DataFrame) -> List[RAPDiagnosis]:
        out = []
        for _, row in features_df.iterrows():
            out.append(self.diagnose(row.to_dict()))
        return out


def build_rap(
    train_features_df: pd.DataFrame,
    feature_cols: Sequence[str],
    descriptions: Sequence[str],
    k: int = 5,
    cache_dir: str | os.PathLike = "data/phase4/rap_cache",
    llm_config: Optional[LLMConfig] = None,
    template_fallback: Optional[Callable[[Dict[str, float]], str]] = None,
) -> RetrievalAugmentedDiagnoser:
    """Convenience constructor: wire up retriever + generator in one call."""
    retriever = CycleRetriever().fit(train_features_df, feature_cols, descriptions)
    generator = LLMGenerator(cache_dir=cache_dir, config=llm_config, fallback=template_fallback)
    return RetrievalAugmentedDiagnoser(retriever=retriever, generator=generator, k=k)
