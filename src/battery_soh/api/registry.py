"""Model registry: verified loading and selection of the JSON model bundles.

Loading is fail-closed and never executes anything from disk:

1. ``MANIFEST.json`` is verified with :func:`~battery_soh.models.linear_bundle.verify_manifest`
   (every listed file must exist and match its SHA-256; names cannot escape the folder).
   If verification fails, *no* model is served and ``/readyz`` stays 503: a
   modified artifact folder is treated as compromised, not as partially usable.
2. Only top-level ``*.json`` files listed in the manifest are loaded (files on
   disk that the manifest does not list are ignored). Each file is read once,
   its bytes are re-hashed against the manifest (no check-then-read race) and
   parsed as plain JSON into a numpy-only :class:`LinearBundle`. Nothing is
   unpickled or imported.
3. Each bundle must use known, non-leaky registry features and carry a model id
   that is safe in URLs and file names; a bundle failing these checks is
   skipped with an error log.

Selection (:meth:`ModelRegistry.select`) picks, among the bundles of a mode
whose inputs are all available, the one with the lowest leave-one-battery-out
MAE recorded in the bundle (ties broken by model id, so it is deterministic).
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from battery_soh.api.schemas import INPUT_FEATURES, MODEL_ID_PATTERN
from battery_soh.models.linear_bundle import (
    BundleError,
    LinearBundle,
    ManifestError,
    Mode,
    sha256_bytes,
    verify_manifest,
)

__all__ = ["Mode", "ModelEntry", "ModelRegistry", "model_metrics"]

logger = logging.getLogger(__name__)

#: Largest bundle file accepted (bytes); real bundles are a few kB.
MAX_BUNDLE_BYTES = 5_000_000

_MODEL_ID_RE = re.compile(MODEL_ID_PATTERN)
_ALLOWED_FEATURES = frozenset(INPUT_FEATURES)


def _reject_constant(token: str) -> Any:
    """Bundles are strict JSON: ``NaN``/``Infinity`` tokens make the file invalid."""
    raise BundleError(f"non-standard JSON constant {token!r}")


def _finite_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _coverage_at(coverage: Any, level: float) -> float | None:
    if not isinstance(coverage, Mapping):
        return None
    for key, value in coverage.items():
        try:
            if abs(float(key) - level) < 1e-9:
                return _finite_or_none(value)
        except (TypeError, ValueError):
            continue
    return None


def model_metrics(bundle: LinearBundle) -> dict[str, float | None]:
    """Headline LOBO metrics recorded in a bundle (``None`` when absent)."""
    metrics = bundle.metrics
    mean = metrics.get("mean")
    mean = mean if isinstance(mean, Mapping) else {}
    return {
        "lobo_mae": _finite_or_none(mean.get("mae")),
        "lobo_rmse": _finite_or_none(mean.get("rmse")),
        "coverage_90": _coverage_at(metrics.get("coverage"), 0.9),
    }


@dataclass(frozen=True)
class ModelEntry:
    """A loaded, verified bundle plus precomputed metadata."""

    bundle: LinearBundle
    file_name: str
    sha256: str
    metrics: dict[str, float | None] = field(default_factory=dict)

    @property
    def model_id(self) -> str:
        """Bundle identifier."""
        return self.bundle.model_id

    @property
    def mode(self) -> Mode:
        """``"absolute"`` or ``"relative"``."""
        return self.bundle.mode

    @property
    def features(self) -> tuple[str, ...]:
        """Model inputs in bundle order."""
        return self.bundle.features

    @property
    def rank_key(self) -> tuple[float, str]:
        """Sort key: lowest recorded LOBO MAE first (unknown MAE last), then id."""
        mae = self.metrics.get("lobo_mae")
        return (math.inf if mae is None else mae, self.model_id)

    def summary(self) -> dict[str, Any]:
        """``GET /v1/models`` entry."""
        return {
            "model_id": self.model_id,
            "feature_set": self.bundle.feature_set,
            "mode": self.mode,
            "features": list(self.features),
            "metrics": dict(self.metrics),
        }


class ModelRegistry:
    """Immutable set of verified model bundles."""

    def __init__(self, entries: Iterable[ModelEntry] = (), *, error: str | None = None) -> None:
        ordered = sorted(entries, key=lambda e: e.rank_key)
        self._entries: dict[str, ModelEntry] = {e.model_id: e for e in ordered}
        self.error = error

    # ------------------------------------------------------------------ loading
    @classmethod
    def empty(cls, error: str | None = None) -> ModelRegistry:
        """A registry with no models (service not ready)."""
        return cls((), error=error)

    @classmethod
    def load(cls, directory: Path) -> ModelRegistry:
        """Verify ``directory/MANIFEST.json`` and load the bundles it lists.

        Never raises for a missing, empty or tampered folder: the result is then
        empty and :attr:`error` says why (the details are logged).
        """
        directory = Path(directory)
        if not directory.is_dir():
            logger.warning("model directory %s does not exist; no model is loaded", directory)
            return cls.empty("model directory not found")
        try:
            manifest = verify_manifest(directory, allow_unlisted=True)
        except ManifestError as exc:
            logger.error("model manifest verification failed, serving no model: %s", exc)
            return cls.empty("model manifest verification failed")
        except (OSError, ValueError) as exc:  # unreadable folder, undecodable manifest
            logger.error("cannot read the model folder %s: %s", directory, exc)
            return cls.empty("model folder unreadable")

        entries: list[ModelEntry] = []
        for name in sorted(manifest):
            pure = PurePosixPath(name)
            if len(pure.parts) != 1 or pure.suffix != ".json":
                continue  # model cards and other listed artifacts are verified, not loaded
            entry = cls._load_entry(directory / name, name, manifest[name])
            if entry is None:
                continue
            if any(e.model_id == entry.model_id for e in entries):
                logger.error("duplicate model id %r in %s; skipping it", entry.model_id, name)
                continue
            entries.append(entry)
        unlisted = sorted(
            p.name
            for p in directory.glob("*.json")
            if p.name != "MANIFEST.json" and p.name not in manifest
        )
        if unlisted:
            logger.warning("ignoring JSON files not listed in the manifest: %s", unlisted)
        if not entries:
            logger.error("no usable model bundle in %s", directory)
            return cls.empty("no usable model bundle")
        logger.info(
            "loaded %d model bundle(s): %s", len(entries), ", ".join(e.model_id for e in entries)
        )
        return cls(entries)

    @staticmethod
    def _load_entry(path: Path, name: str, expected_sha256: str) -> ModelEntry | None:
        try:
            if path.stat().st_size > MAX_BUNDLE_BYTES:
                logger.error("bundle %s exceeds %d bytes; skipping it", name, MAX_BUNDLE_BYTES)
                return None
            raw = path.read_bytes()
        except OSError as exc:
            logger.error("cannot read bundle %s: %s", name, exc)
            return None
        digest = sha256_bytes(raw)
        if digest != expected_sha256.lower():
            logger.error("bundle %s changed after manifest verification; skipping it", name)
            return None
        try:
            data = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant)
            if not isinstance(data, dict):
                raise BundleError("bundle must be a JSON object")
            bundle = LinearBundle.from_dict(data)
            metrics = model_metrics(bundle)
        # BundleError, UnicodeDecodeError and JSONDecodeError are ValueErrors. TypeError and
        # any other ValueError are caught too: a manifest-valid but malformed bundle must
        # be skipped (and /readyz stay 503 if nothing else loads), never crash the startup.
        except (TypeError, ValueError, RecursionError) as exc:
            logger.error("invalid model bundle %s: %s", name, exc)
            return None
        if not _MODEL_ID_RE.match(bundle.model_id):
            logger.error("bundle %s has an unsafe model id %r; skipping it", name, bundle.model_id)
            return None
        bad = [f for f in bundle.features if f not in _ALLOWED_FEATURES]
        if bad:
            logger.error(
                "bundle %s uses unknown or leaky features %s; refusing to serve it", name, bad
            )
            return None
        return ModelEntry(bundle=bundle, file_name=name, sha256=digest, metrics=metrics)

    # ------------------------------------------------------------------ access
    def __len__(self) -> int:
        return len(self._entries)

    def __bool__(self) -> bool:
        return bool(self._entries)

    def __contains__(self, model_id: object) -> bool:
        return model_id in self._entries

    @property
    def entries(self) -> tuple[ModelEntry, ...]:
        """All entries, best (lowest recorded LOBO MAE) first."""
        return tuple(self._entries.values())

    @property
    def model_ids(self) -> list[str]:
        """Model ids, best first."""
        return list(self._entries)

    def get(self, model_id: str) -> ModelEntry | None:
        """Entry by id, or ``None``."""
        return self._entries.get(model_id)

    def has_mode(self, mode: Mode) -> bool:
        """True if at least one bundle of ``mode`` is loaded."""
        return any(e.mode == mode for e in self._entries.values())

    def select(
        self,
        available_features: Collection[str],
        mode: Mode | None = None,
        *,
        allow_partial: bool = False,
    ) -> ModelEntry | None:
        """Pick the best bundle for the available inputs.

        Args:
            available_features: Names of the features the caller can provide.
            mode: Restrict to one mode (``None``: any mode). Callers must only
                consider ``"relative"`` bundles when a commissioning reference
                is available for the features.
            allow_partial: If no bundle has all its inputs, fall back to the
                bundle with the most available inputs (the rest are imputed).

        Returns:
            The bundle with the lowest recorded LOBO MAE among those whose
            inputs are all available; with ``allow_partial`` the best-covered
            bundle instead when none is complete. ``None`` if nothing qualifies
            (a partial match needs at least one available input).
        """
        available = frozenset(available_features)
        pool = [e for e in self._entries.values() if mode is None or e.mode == mode]
        complete = [e for e in pool if available.issuperset(e.features)]
        if complete:
            return min(complete, key=lambda e: e.rank_key)
        if not allow_partial:
            return None
        scored = [(sum(f in available for f in e.features), e) for e in pool]
        scored = [(n, e) for n, e in scored if n > 0]
        if not scored:
            return None
        return min(scored, key=lambda item: (-item[0], *item[1].rank_key))[1]
