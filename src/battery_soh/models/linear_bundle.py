"""Auditable JSON export of linear models, and a numpy-only runtime for serving.

The production model is ``SimpleImputer(median) -> StandardScaler -> Ridge``.
Instead of pickling it (opaque, version-fragile, unsafe to load), training
exports every number the model uses into a human-readable JSON *bundle*
(format: ``docs/MODEL_CARD.md``, section "Bundle format"). :class:`LinearBundle`
reloads it and predicts with plain numpy, so serving never imports
scikit-learn and the artifact can be reviewed in a pull request.

What a bundle answers, for one cycle's health indicators ``x``:

* **prediction** ``yhat = b + sum_j w_j z_j`` with ``z_j = (x_j - mu_j) / s_j``
  (missing ``x_j`` replaced by the training median, exactly as the
  ``SimpleImputer`` did in training), clipped to a physical range;
* **exact attribution** ``w_j z_j`` per feature, which sums with the base
  value ``b`` (the prediction for an average training cycle) to the
  prediction - no sampling, no approximation. The only exception is an
  out-of-range output that the clip changed, which :class:`Explanation`
  reports (``clipped``);
* **uncertainty** ``yhat +/- q(confidence)`` from battery-grouped conformal
  calibration (:mod:`battery_soh.uncertainty.conformal`); ``q`` may be
  ``+inf`` (too few calibration scores), stored as JSON ``null``;
* **applicability** - is ``x`` inside the training envelope (per-feature range
  with margin) and not far from the training cloud (Mahalanobis distance
  under a Ledoit-Wolf shrunk covariance, inverted with a pseudo-inverse,
  compared with a high quantile of the training distances)?

In ``"relative"`` mode the inputs must already be divided by the cell's
commissioning reference (:func:`battery_soh.features.relative.apply_reference`);
the bundle itself is mode-agnostic arithmetic.

The module also provides SHA-256 helpers and the ``MANIFEST.json``
writer/verifier used by training and by the API model registry. It imports
scikit-learn only inside the export path, so serving needs numpy only.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from battery_soh._typing import FloatArray
from battery_soh.config import DEFAULT_N_REFERENCE_CYCLES
from battery_soh.errors import BatterySohError
from battery_soh.uncertainty.conformal import confidence_key

__all__ = [
    "DEFAULT_CONFORMAL_METHOD",
    "DEFAULT_OUTPUT_CLIP",
    "LABEL_DEFINITION",
    "MANIFEST_NAME",
    "SCHEMA_VERSION",
    "BundleError",
    "DomainCheck",
    "Explanation",
    "LinearBundle",
    "ManifestError",
    "export_linear_bundle",
    "save_bundle",
    "sha256_bytes",
    "sha256_file",
    "verify_manifest",
    "write_manifest",
    "write_text_lf",
]

logger = logging.getLogger(__name__)

Mode = Literal["absolute", "relative"]

SCHEMA_VERSION = "1.0"
MANIFEST_NAME = "MANIFEST.json"
DEFAULT_OUTPUT_CLIP: tuple[float, float] = (0.0, 120.0)
LABEL_DEFINITION = "100 * capacity / C0; C0 = median of first 5 QC-valid discharge capacities"
DEFAULT_CONFORMAL_METHOD = (
    "group conformal: symmetric half-width = exact conformal quantile of "
    "leave-one-battery-out absolute residuals"
)
ATTRIBUTION_METHOD = "exact linear attribution on standardized features"

# Export is verified against scikit-learn; allow for summation-order rounding only.
_EXPORT_RTOL = 1e-9
_ESTIMATOR_TYPES = {
    "Ridge": "ridge",
    "RidgeCV": "ridge",
    "PLSRegressor": "pls",
    "LinearRegression": "linear",
    "Lasso": "lasso",
    "LassoCV": "lasso",
    "ElasticNet": "elasticnet",
    "ElasticNetCV": "elasticnet",
}


class BundleError(ValueError, BatterySohError):
    """A bundle cannot be exported or loaded (wrong pipeline shape, corrupt JSON)."""


class ManifestError(BatterySohError):
    """A model artifact is missing or does not match its recorded SHA-256."""


# ====================================================================== hashing / manifest
def sha256_bytes(data: bytes) -> str:
    """Hex SHA-256 digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """Hex SHA-256 digest of a file, streamed in chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _lf_bytes(text: str) -> bytes:
    """UTF-8 encoding of ``text`` with every line ending normalised to LF."""
    return text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")


def write_text_lf(path: Path, text: str) -> str:
    """Write ``text`` as UTF-8 with LF line endings on every OS.

    ``Path.write_text`` translates ``"\\n"`` to the platform separator, so on
    Windows it writes CRLF. The repository normalises text files to LF
    (``.gitattributes``: ``* text=auto eol=lf``), so a digest of CRLF bytes
    recorded in ``MANIFEST.json`` would stop matching after a checkout on
    Linux (CI, Docker). Writing the LF bytes explicitly makes the file
    identical on every OS and in every checkout.

    Args:
        path: Destination (parent folders are created).
        text: Content; ``\\r\\n`` and lone ``\\r`` become ``\\n``.

    Returns:
        Hex SHA-256 of exactly the bytes written.
    """
    data = _lf_bytes(text)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return sha256_bytes(data)


def _has_crlf(path: Path, *, probe: int = 8000) -> bool:
    """True for a text file (no NUL byte in its head, git's rule) containing CRLF."""
    data = path.read_bytes()
    return b"\0" not in data[:probe] and b"\r\n" in data


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _safe_member(directory: Path, name: str) -> Path:
    """Resolve a manifest entry, refusing anything that could escape ``directory``.

    Names are POSIX-style relative paths; backslashes and colons (Windows
    separators and drive prefixes such as ``C:x``) are rejected outright.
    """
    pure = PurePosixPath(name)
    if pure.is_absolute() or ".." in pure.parts or "\\" in name or ":" in name or not pure.parts:
        raise ManifestError(f"illegal file name in manifest: {name!r}")
    return directory.joinpath(*pure.parts)


def write_manifest(
    directory: Path,
    files: Iterable[str | Path] | None = None,
    *,
    created_utc: str | None = None,
) -> Path:
    """Write ``MANIFEST.json`` with the SHA-256 of each artifact in ``directory``.

    Args:
        directory: Folder holding the artifacts.
        files: Files to include (paths or names relative to ``directory``).
            Defaults to every ``*.json`` file in ``directory`` except the
            manifest itself.
        created_utc: Timestamp override (ISO 8601) for reproducible output.

    Returns:
        Path of the written manifest.

    Raises:
        ManifestError: If a listed file does not exist or lies outside ``directory``.
    """
    directory = Path(directory)
    if files is None:
        names = sorted(
            p.name for p in directory.glob("*.json") if p.is_file() and p.name != MANIFEST_NAME
        )
    else:
        names = []
        for item in files:
            path = Path(item)
            if path.is_absolute():
                try:
                    path = path.relative_to(directory)
                except ValueError:
                    raise ManifestError(f"{item} is not inside {directory}") from None
            names.append(path.as_posix())
        names = sorted(set(names))
    entries: dict[str, str] = {}
    for name in names:
        member = _safe_member(directory, name)
        if not member.is_file():
            raise ManifestError(f"cannot hash missing file {member}")
        if _has_crlf(member):
            logger.warning(
                "%s has CRLF line endings: git (eol=lf) rewrites it on checkout and the "
                "recorded sha256 will not match there; write it with write_text_lf",
                name,
            )
        entries[name] = sha256_file(member)
    manifest = {"files": entries, "created_utc": created_utc or _utc_now()}
    target = directory / MANIFEST_NAME
    write_text_lf(target, json.dumps(manifest, indent=2) + "\n")
    logger.info("wrote manifest for %d file(s) to %s", len(entries), target)
    return target


def verify_manifest(directory: Path, *, allow_unlisted: bool = True) -> dict[str, str]:
    """Check every file listed in ``directory/MANIFEST.json`` against its SHA-256.

    Args:
        directory: Folder containing ``MANIFEST.json`` and the artifacts.
        allow_unlisted: If false, a ``*.json`` file in ``directory`` that the
            manifest does not list is an error too (use this when every JSON
            file in the folder is loaded as a model).

    Returns:
        The verified ``{file name: sha256}`` mapping.

    Raises:
        ManifestError: If the manifest is missing or malformed, a file is
            missing, a name escapes ``directory``, a digest does not match, or
            (``allow_unlisted=False``) an unlisted JSON file is present.
    """
    directory = Path(directory)
    manifest_path = directory / MANIFEST_NAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ManifestError(f"no {MANIFEST_NAME} in {directory}") from None
    except json.JSONDecodeError as exc:
        raise ManifestError(f"{manifest_path} is not valid JSON: {exc}") from exc
    files = manifest.get("files") if isinstance(manifest, dict) else None
    if not isinstance(files, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in files.items()
    ):
        raise ManifestError(f"{manifest_path} must contain a 'files' object of name -> sha256")
    for name, expected in files.items():
        member = _safe_member(directory, name)
        if not member.is_file():
            raise ManifestError(f"{name} is listed in the manifest but missing")
        actual = sha256_file(member)
        if actual != expected.lower():
            hint = (
                " (the file has CRLF line endings: was it checked out or saved with eol=crlf?)"
                if _has_crlf(member)
                else ""
            )
            raise ManifestError(
                f"sha256 mismatch for {name}: expected {expected}, got {actual}{hint}"
            )
    if not allow_unlisted:
        unlisted = sorted(
            p.name
            for p in directory.glob("*.json")
            if p.is_file() and p.name != MANIFEST_NAME and p.name not in files
        )
        if unlisted:
            raise ManifestError(f"files not covered by {MANIFEST_NAME}: {unlisted}")
    return dict(files)


# ====================================================================== runtime objects
@dataclass(frozen=True)
class DomainCheck:
    """Whether an input lies inside the bundle's training domain.

    Attributes:
        in_domain: No provided feature is out of range and the Mahalanobis
            distance is within the threshold.
        out_of_range: Provided features outside ``[min, max]`` widened by the margin.
        mahalanobis: Distance of the (imputed, standardised) input from the
            training centre under the shrunk training covariance.
        threshold: Training-distance quantile above which inputs are flagged.
        missing: Features that were not provided (imputed with the training median).
    """

    in_domain: bool
    out_of_range: tuple[str, ...]
    mahalanobis: float
    threshold: float
    missing: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready representation (lists instead of tuples)."""
        return {
            "in_domain": self.in_domain,
            "out_of_range": list(self.out_of_range),
            "mahalanobis": self.mahalanobis,
            "threshold": self.threshold,
            "missing": list(self.missing),
        }


@dataclass(frozen=True)
class Explanation:
    """Exact additive decomposition of one linear prediction.

    ``base_value + sum(c for _, _, c in contributions) == raw_prediction``
    up to floating-point rounding; ``prediction`` is ``raw_prediction`` clipped
    to the bundle's output range.
    """

    base_value: float
    contributions: tuple[tuple[str, float, float], ...]
    raw_prediction: float
    prediction: float

    @property
    def clipped(self) -> bool:
        """True if the output clip changed the prediction."""
        return self.prediction != self.raw_prediction


def _finite_vector(values: Any, name: str, n: int | None = None) -> FloatArray:
    try:
        arr = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise BundleError(f"{name} must be a list of numbers") from exc
    if arr.ndim != 1 or (n is not None and arr.size != n):
        raise BundleError(f"{name} must be a list of length {n}, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise BundleError(f"{name} contains non-finite values")
    return arr


def _optional_vector(values: Any, name: str, n: int) -> FloatArray:
    """List of numbers or nulls (null -> NaN = no information)."""
    if not isinstance(values, list) or len(values) != n:
        raise BundleError(f"{name} must be a list of length {n}")
    try:
        out = np.array([np.nan if v is None else v for v in values], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise BundleError(f"{name} must contain numbers or nulls") from exc
    if out.ndim != 1 or np.any(np.isinf(out)):
        raise BundleError(f"{name} contains infinite values")
    return out


def _require(data: Mapping[str, Any], key: str, kind: type | tuple[type, ...]) -> Any:
    value = data.get(key)
    if not isinstance(value, kind):
        raise BundleError(f"bundle field {key!r} is missing or has the wrong type")
    return value


def _is_real(value: Any) -> bool:
    """A JSON number (``bool`` is an ``int`` subclass in Python, but not a number here)."""
    return isinstance(value, int | float) and not isinstance(value, bool)


def _finite_matrix(values: Any, name: str, n: int) -> FloatArray:
    """A finite ``n x n`` matrix; ragged or non-numeric input is a :class:`BundleError`."""
    try:
        arr = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise BundleError(f"{name} must be a finite {n}x{n} matrix") from exc
    if arr.shape != (n, n) or not np.all(np.isfinite(arr)):
        raise BundleError(f"{name} must be a finite {n}x{n} matrix")
    return arr


class LinearBundle:
    """Numpy-only runtime for an exported linear model bundle.

    Construct with :meth:`from_json` or :meth:`from_dict`; the constructor
    validates the whole document so a corrupt bundle fails at load time, not
    on the first request. Every malformed field raises :class:`BundleError`
    (never a bare ``TypeError``/``ValueError``), so a loader that catches
    ``BundleError`` can skip the file instead of crashing.
    """

    def __init__(self, data: Mapping[str, Any]) -> None:
        if not isinstance(data, Mapping):
            raise BundleError("bundle must be a JSON object")
        version = str(data.get("schema_version", ""))
        if version.split(".")[0] != SCHEMA_VERSION.split(".")[0]:
            raise BundleError(f"unsupported bundle schema_version {version!r}")
        self.model_id: str = _require(data, "model_id", str)
        self.feature_set: str = _require(data, "feature_set", str)
        mode = _require(data, "mode", str)
        if mode not in ("absolute", "relative"):
            raise BundleError(f"mode must be 'absolute' or 'relative', got {mode!r}")
        self.mode: Mode = mode
        features = _require(data, "features", list)
        if not features or not all(isinstance(f, str) for f in features):
            raise BundleError("features must be a non-empty list of names")
        if len(set(features)) != len(features):
            raise BundleError("features must be unique")
        self.features: tuple[str, ...] = tuple(features)
        p = len(self.features)
        n_ref = data.get("n_reference_cycles", DEFAULT_N_REFERENCE_CYCLES)
        if isinstance(n_ref, bool) or not isinstance(n_ref, int) or n_ref < 1:
            raise BundleError("n_reference_cycles must be a positive integer")
        self.n_reference_cycles = n_ref

        pre = _require(data, "preprocessing", dict)
        self.impute_median = _finite_vector(pre.get("impute_median"), "impute_median", p)
        self.scaler_mean = _finite_vector(pre.get("scaler_mean"), "scaler_mean", p)
        self.scaler_scale = _finite_vector(pre.get("scaler_scale"), "scaler_scale", p)
        if np.any(self.scaler_scale <= 0):
            raise BundleError("scaler_scale must be strictly positive")

        est = _require(data, "estimator", dict)
        self.estimator_type = str(est.get("type", "linear"))
        self.coef = _finite_vector(est.get("coef"), "coef", p)
        intercept = est.get("intercept")
        if not _is_real(intercept) or not math.isfinite(intercept):
            raise BundleError("estimator.intercept must be a finite number")
        self.intercept = float(intercept)

        clip = data.get("output_clip", list(DEFAULT_OUTPUT_CLIP))
        clip_arr = _finite_vector(clip, "output_clip", 2)
        if not clip_arr[0] < clip_arr[1]:
            raise BundleError("output_clip must be [low, high] with low < high")
        self.output_clip: tuple[float, float] = (float(clip_arr[0]), float(clip_arr[1]))

        conformal = _require(data, "conformal", dict)
        quantiles = _require(conformal, "quantiles", dict)
        self.quantiles: dict[str, float] = {}
        for key, value in quantiles.items():
            q = math.inf if value is None else value
            if not _is_real(q) or math.isnan(q) or q < 0:
                raise BundleError(f"conformal quantile {key!r} must be a number >= 0 or null")
            try:
                self.quantiles[confidence_key(key)] = float(q)
            except ValueError as exc:
                raise BundleError(f"invalid conformal confidence key {key!r}") from exc
        self.conformal_method = str(conformal.get("method", ""))
        n_scores = conformal.get("n_scores", 0)
        if isinstance(n_scores, bool) or not isinstance(n_scores, int) or n_scores < 0:
            raise BundleError("conformal.n_scores must be a non-negative integer")
        self.n_scores = n_scores

        dom = _require(data, "domain", dict)
        self.feature_min = _optional_vector(dom.get("feature_min"), "feature_min", p)
        self.feature_max = _optional_vector(dom.get("feature_max"), "feature_max", p)
        margin = dom.get("margin", 0.1)
        if not _is_real(margin) or not math.isfinite(margin) or margin < 0:
            raise BundleError("domain.margin must be a finite number >= 0")
        self.margin = float(margin)
        threshold = dom.get("mahalanobis_threshold")
        if not _is_real(threshold) or not math.isfinite(threshold):
            raise BundleError("domain.mahalanobis_threshold must be a finite number")
        self.mahalanobis_threshold = float(threshold)
        self.precision = _finite_matrix(dom.get("precision_matrix"), "domain.precision_matrix", p)
        center = dom.get("center")
        self.domain_center = (
            np.zeros(p) if center is None else _finite_vector(center, "domain.center", p)
        )
        if not isinstance(data.get("metrics", {}), Mapping):
            raise BundleError("bundle field 'metrics' must be an object")
        self._data = copy.deepcopy(dict(data))

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LinearBundle:
        """Validate and wrap an in-memory bundle document."""
        return cls(data)

    @classmethod
    def from_json(cls, path: Path) -> LinearBundle:
        """Load and validate a bundle file.

        Raises:
            BundleError: If the file is not valid JSON or fails validation.
        """
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise BundleError(f"{path} is not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise BundleError(f"{path} must contain a JSON object")
        return cls(data)

    def to_dict(self) -> dict[str, Any]:
        """Deep copy of the full bundle document."""
        return copy.deepcopy(self._data)

    def to_card(self) -> dict[str, Any]:
        """Bundle document without the large domain matrix (for model-card endpoints)."""
        card = self.to_dict()
        card.get("domain", {}).pop("precision_matrix", None)
        return card

    # ------------------------------------------------------------------ properties
    @property
    def n_features(self) -> int:
        """Number of model inputs."""
        return len(self.features)

    @property
    def base_value(self) -> float:
        """Prediction for an input at the training mean (all ``z_j = 0``)."""
        return self.intercept

    @property
    def metrics(self) -> dict[str, Any]:
        """Evaluation metrics recorded at training time."""
        return copy.deepcopy(dict(self._data.get("metrics", {})))

    @property
    def confidences(self) -> tuple[float, ...]:
        """Confidence levels with a stored conformal quantile, ascending."""
        return tuple(sorted(float(k) for k in self.quantiles))

    # ------------------------------------------------------------------ input handling
    def _vector(self, features: Mapping[str, float | None]) -> tuple[FloatArray, tuple[str, ...]]:
        """Raw input in model order (NaN = missing) and the names of missing inputs."""
        x = np.full(self.n_features, np.nan)
        for j, name in enumerate(self.features):
            value = features.get(name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(
                value, int | float | np.integer | np.floating
            ):
                raise ValueError(f"feature {name!r} must be a number, got {type(value).__name__}")
            if math.isinf(float(value)):
                raise ValueError(f"feature {name!r} must be finite, got {value}")
            x[j] = float(value)
        missing = tuple(n for n, v in zip(self.features, x, strict=True) if np.isnan(v))
        return x, missing

    def missing_features(self, features: Mapping[str, float | None]) -> tuple[str, ...]:
        """Model inputs absent (or NaN/None) in ``features``; they would be median-imputed."""
        return self._vector(features)[1]

    def transform(self, X: ArrayLike) -> FloatArray:
        """Impute missing values and standardise a raw design matrix (rows x features)."""
        arr = np.array(X, dtype=np.float64, ndmin=2)
        if arr.shape[1] != self.n_features:
            raise ValueError(f"expected {self.n_features} columns, got {arr.shape[1]}")
        if np.any(np.isinf(arr)):
            raise ValueError("inputs must not be infinite (NaN marks a missing value)")
        arr = np.where(np.isnan(arr), self.impute_median, arr)
        return np.asarray((arr - self.scaler_mean) / self.scaler_scale, dtype=np.float64)

    # ------------------------------------------------------------------ prediction
    def predict_raw_array(self, X: ArrayLike) -> FloatArray:
        """Unclipped predictions for a raw design matrix (columns in :attr:`features` order)."""
        return np.asarray(self.transform(X) @ self.coef + self.intercept, dtype=np.float64)

    def predict_array(self, X: ArrayLike) -> FloatArray:
        """Clipped predictions for a raw design matrix (NaN = missing feature)."""
        return np.clip(self.predict_raw_array(X), *self.output_clip)

    def predict(self, features: Mapping[str, float | None]) -> float:
        """SOH prediction (percent) for one input mapping; extra keys are ignored."""
        return self.explain(features).prediction

    def explain(self, features: Mapping[str, float | None]) -> Explanation:
        """Prediction with its exact per-feature decomposition.

        Each contribution is ``coef_j * z_j`` where ``z_j`` is the standardised
        (and, if missing, median-imputed) input; the reported value is the raw
        input actually used (the training median for a missing feature).
        """
        x, _ = self._vector(features)
        used = np.where(np.isnan(x), self.impute_median, x)
        z = (used - self.scaler_mean) / self.scaler_scale
        parts = self.coef * z
        raw = float(self.intercept + parts.sum())
        lo, hi = self.output_clip
        return Explanation(
            base_value=self.intercept,
            contributions=tuple(
                (name, float(v), float(c))
                for name, v, c in zip(self.features, used, parts, strict=True)
            ),
            raw_prediction=raw,
            prediction=float(min(max(raw, lo), hi)),
        )

    def contributions(self, features: Mapping[str, float | None]) -> list[tuple[str, float, float]]:
        """Exact linear attribution ``[(feature, value used, coef_j * z_j)]``.

        ``base_value + sum(contributions) == predict(features)`` up to float
        rounding, unless the output clip changed the prediction (see
        :meth:`explain`, whose ``raw_prediction`` the sum always equals).
        Missing features contribute at their imputed (median) value.
        """
        return list(self.explain(features).contributions)

    def half_width(self, confidence: float = 0.9) -> float:
        """Stored conformal half-width ``q`` for ``confidence``; ``+inf`` means unbounded.

        An infinite half-width means the calibration had too few scores for
        that level (``n < confidence / (1 - confidence)``): no finite conformal
        quantile exists there. Callers should surface that, not hide it.

        Raises:
            ValueError: If no quantile is stored for ``confidence``.
        """
        key = confidence_key(confidence)
        if key not in self.quantiles:
            available = ", ".join(sorted(self.quantiles))
            raise ValueError(f"no conformal quantile for confidence {key}; available: {available}")
        return self.quantiles[key]

    def interval(self, prediction: float, confidence: float = 0.9) -> tuple[float, float]:
        """Conformal prediction interval ``prediction +/- q(confidence)``.

        Bounds are clipped to the output range: the label is physically
        bounded, so clipping never reduces coverage, and it keeps the result
        finite (JSON-safe). In particular an unbounded quantile
        (:meth:`half_width` is ``+inf``) yields the whole output range
        ``output_clip`` - an explicit "no information" interval, which callers
        should flag using :meth:`half_width`.

        Raises:
            ValueError: If no quantile is stored for ``confidence`` or
                ``prediction`` is not finite.
        """
        if not math.isfinite(prediction):
            raise ValueError(f"prediction must be finite, got {prediction}")
        q = self.half_width(confidence)
        lo, hi = self.output_clip
        if math.isinf(q):
            return lo, hi
        return float(min(max(prediction - q, lo), hi)), float(max(min(prediction + q, hi), lo))

    # ------------------------------------------------------------------ applicability
    def mahalanobis_array(self, X: ArrayLike) -> FloatArray:
        """Mahalanobis distance of each (imputed, standardised) row from the training centre."""
        diff = self.transform(X) - self.domain_center
        d2 = np.einsum("ij,jk,ik->i", diff, self.precision, diff)
        return np.asarray(np.sqrt(np.maximum(d2, 0.0)), dtype=np.float64)

    def domain_check(self, features: Mapping[str, float | None]) -> DomainCheck:
        """Flag inputs outside the training envelope (range with margin, Mahalanobis)."""
        x, missing = self._vector(features)
        span = self.feature_max - self.feature_min
        # A feature constant in training (e.g. one ambient temperature) gets a
        # tolerance relative to its magnitude instead of a zero-width range.
        span = np.where(span > 0, span, np.maximum(np.abs(self.feature_max), 1.0))
        low = self.feature_min - self.margin * span
        high = self.feature_max + self.margin * span
        with np.errstate(invalid="ignore"):
            bad = ~np.isnan(x) & ~np.isnan(low) & ((x < low) | (x > high))
        out_of_range = tuple(n for n, flag in zip(self.features, bad, strict=True) if flag)
        distance = float(self.mahalanobis_array(x[None, :])[0])
        return DomainCheck(
            in_domain=not out_of_range and distance <= self.mahalanobis_threshold,
            out_of_range=out_of_range,
            mahalanobis=distance,
            threshold=self.mahalanobis_threshold,
            missing=missing,
        )

    def __repr__(self) -> str:
        return (
            f"LinearBundle(model_id={self.model_id!r}, feature_set={self.feature_set!r}, "
            f"mode={self.mode!r}, n_features={self.n_features})"
        )


# ====================================================================== export
def _json_safe(value: Any) -> Any:
    """Recursively convert numpy scalars/arrays and non-finite floats (-> None) for strict JSON."""
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple | np.ndarray):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.bool_ | bool):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        return float(value) if math.isfinite(value) else None
    return value


def _unpack_pipeline(pipeline: Any, features: Sequence[str]) -> tuple[Any, Any, Any]:
    """Return (imputer, scaler, estimator) after checking the pipeline shape.

    If the pipeline was fitted on a DataFrame, its recorded column names must
    equal ``features`` in order: a silent column permutation would export a
    model that multiplies every coefficient with the wrong input.
    """
    n_features = len(features)
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    steps = getattr(pipeline, "steps", None)
    if not steps or len(steps) != 3:
        raise BundleError("expected a fitted Pipeline(SimpleImputer -> StandardScaler -> linear)")
    imputer, scaler, estimator = (step for _, step in steps)
    if not isinstance(imputer, SimpleImputer) or imputer.strategy != "median":
        raise BundleError("first pipeline step must be SimpleImputer(strategy='median')")
    if imputer.add_indicator:
        raise BundleError("SimpleImputer(add_indicator=True) is not supported")
    if not isinstance(scaler, StandardScaler) or not (scaler.with_mean and scaler.with_std):
        raise BundleError("second pipeline step must be StandardScaler(with_mean, with_std)")
    if not hasattr(estimator, "coef_") or not hasattr(estimator, "intercept_"):
        raise BundleError("final step must be a fitted linear model with coef_ and intercept_")
    if getattr(imputer, "n_features_in_", None) != n_features:
        raise BundleError(
            f"pipeline was fitted on {getattr(imputer, 'n_features_in_', '?')} features, "
            f"but {n_features} feature names were given"
        )
    if np.asarray(imputer.statistics_).size != n_features or scaler.n_features_in_ != n_features:
        raise BundleError(
            "the imputer dropped an all-missing column; fit with keep_empty_features=True"
        )
    fitted_names = getattr(imputer, "feature_names_in_", None)
    if fitted_names is not None and [str(c) for c in fitted_names] != list(features):
        raise BundleError(
            f"pipeline was fitted on columns {list(fitted_names)}, not on {list(features)}"
        )
    return imputer, scaler, estimator


def _estimator_section(estimator: Any, n_features: int) -> dict[str, Any]:
    coef = np.asarray(estimator.coef_, dtype=np.float64).reshape(-1)
    intercept = np.asarray(estimator.intercept_, dtype=np.float64).reshape(-1)
    if coef.size != n_features or intercept.size != 1:
        raise BundleError("only single-target linear models can be exported")
    name = type(estimator).__name__
    section: dict[str, Any] = {"type": _ESTIMATOR_TYPES.get(name, name.lower())}
    params = estimator.get_params()
    for key in ("alpha", "l1_ratio"):
        value = getattr(estimator, f"{key}_", params.get(key))
        if isinstance(value, int | float | np.floating) and not isinstance(value, bool):
            section[key] = float(value)
    n_components = getattr(estimator, "n_components_", None)
    if isinstance(n_components, int | np.integer):
        section["n_components"] = int(n_components)
    section["coef"] = coef.tolist()
    section["intercept"] = float(intercept[0])
    return section


def _training_hash(df: pd.DataFrame, columns: Sequence[str]) -> str:
    """SHA-256 of a canonical CSV rendering of the training columns."""
    text = df.loc[:, list(columns)].to_csv(index=False, float_format="%.17g", lineterminator="\n")
    return sha256_bytes(text.encode("utf-8"))


def _domain_section(
    bundle_core: Mapping[str, Any],
    X_raw: FloatArray,
    *,
    margin: float,
    quantile: float,
) -> dict[str, Any]:
    """Range envelope plus a Ledoit-Wolf Mahalanobis model of the training inputs."""
    from sklearn.covariance import ledoit_wolf

    n, p = X_raw.shape
    # Columns with no observed value get no range (NaN -> null): nothing to compare against.
    all_missing = np.all(np.isnan(X_raw), axis=0)
    filled = np.where(all_missing, 0.0, X_raw)
    fmin = np.where(all_missing, np.nan, np.nanmin(filled, axis=0))
    fmax = np.where(all_missing, np.nan, np.nanmax(filled, axis=0))
    pre = bundle_core["preprocessing"]
    median = np.asarray(pre["impute_median"])
    Z = (np.where(np.isnan(X_raw), median, X_raw) - np.asarray(pre["scaler_mean"])) / np.asarray(
        pre["scaler_scale"]
    )
    center = Z.mean(axis=0)
    if n > 1:
        covariance, shrinkage = ledoit_wolf(Z - center, assume_centered=True)
    else:
        covariance, shrinkage = np.eye(p), 1.0
    precision = np.linalg.pinv(np.asarray(covariance), hermitian=True)
    diff = Z - center
    distances = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", diff, precision, diff), 0.0))
    threshold = float(np.quantile(distances, quantile, method="higher"))
    return {
        "feature_min": [None if np.isnan(v) else float(v) for v in fmin],
        "feature_max": [None if np.isnan(v) else float(v) for v in fmax],
        "margin": float(margin),
        "center": center.tolist(),
        "mahalanobis_threshold": threshold,
        "mahalanobis_quantile": float(quantile),
        "covariance_shrinkage": float(shrinkage),
        "precision_matrix": precision.tolist(),
    }


def export_linear_bundle(
    pipeline: Any,
    *,
    model_id: str,
    feature_set: str,
    mode: Mode,
    features: Sequence[str],
    training_df: pd.DataFrame,
    conformal_quantiles: Mapping[float | str, float],
    metrics: Mapping[str, Any],
    n_conformal_scores: int = 0,
    conformal_method: str = DEFAULT_CONFORMAL_METHOD,
    label_col: str = "soh_pct",
    label_definition: str = LABEL_DEFINITION,
    group_col: str = "battery_id",
    n_reference_cycles: int = DEFAULT_N_REFERENCE_CYCLES,
    output_clip: tuple[float, float] = DEFAULT_OUTPUT_CLIP,
    domain_margin: float = 0.1,
    mahalanobis_quantile: float = 0.99,
    features_sha256: str | None = None,
    created_utc: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Export a fitted ``SimpleImputer(median) -> StandardScaler -> linear`` pipeline.

    The returned document is verified before it is returned: a
    :class:`LinearBundle` built from it must reproduce ``pipeline.predict`` on
    the training rows to within ``1e-9`` (relative).

    Args:
        pipeline: Fitted scikit-learn pipeline (e.g. ``make_model("ridge")``).
        model_id: Stable identifier, e.g. ``"combined-absolute-ridge"``.
        feature_set: Registry feature-set name.
        mode: ``"absolute"`` or ``"relative"`` (inputs divided by the cell's
            commissioning reference before prediction).
        features: Model input names in pipeline column order.
        training_df: The rows the pipeline was fitted on (features in the same
            mode as the model, plus ``group_col``); used for domain statistics
            and the training record.
        conformal_quantiles: ``{confidence: half-width}``, e.g. ``{0.9: 2.1}``
            (``inf`` is stored as ``null`` and means unbounded).
        metrics: Evaluation metrics to embed (copied verbatim, NaN -> null).
        n_conformal_scores: Number of calibration residuals behind the quantiles.
        conformal_method: Human-readable description of the calibration.
        label_col: Name of the label.
        label_definition: Human-readable label definition.
        group_col: Battery id column of ``training_df``.
        n_reference_cycles: Commissioning cycles used in relative mode.
        output_clip: Physical range of the label.
        domain_margin: Fractional widening of each feature's training range.
        mahalanobis_quantile: Quantile of training distances used as threshold.
        features_sha256: SHA-256 of the source feature table; when omitted a
            hash of the training columns is recorded instead.
        created_utc: Timestamp override for reproducible exports.
        extra: Additional top-level fields (must not shadow schema fields).

    Returns:
        The bundle as a JSON-ready ``dict`` (strict JSON: no NaN/inf).

    Raises:
        BundleError: If the pipeline has the wrong shape or the export does
            not reproduce scikit-learn's predictions.
        ValueError: On invalid arguments.
    """
    features = list(features)
    if not features or len(set(features)) != len(features):
        raise ValueError("features must be a non-empty list of unique names")
    if mode not in ("absolute", "relative"):
        raise ValueError(f"mode must be 'absolute' or 'relative', got {mode!r}")
    missing_cols = [c for c in (*features, group_col) if c not in training_df.columns]
    if missing_cols:
        raise ValueError(f"training_df lacks columns {missing_cols}")
    if not 0.0 < mahalanobis_quantile <= 1.0:
        raise ValueError(f"mahalanobis_quantile must be in (0, 1], got {mahalanobis_quantile}")
    if domain_margin < 0:
        raise ValueError(f"domain_margin must be >= 0, got {domain_margin}")
    if training_df.empty:
        raise ValueError("training_df is empty")

    imputer, scaler, estimator = _unpack_pipeline(pipeline, features)
    quantiles: dict[str, float | None] = {}
    for key, value in conformal_quantiles.items():
        q = float(value)
        if math.isnan(q) or q < 0:
            raise ValueError(f"conformal quantile for {key} must be >= 0, got {value}")
        quantiles[confidence_key(key)] = q if math.isfinite(q) else None

    from battery_soh import __version__

    X_raw = training_df.loc[:, features].to_numpy(dtype=np.float64)
    core: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "model_id": model_id,
        "created_utc": created_utc or _utc_now(),
        "package_version": __version__,
        "feature_set": feature_set,
        "mode": mode,
        "n_reference_cycles": int(n_reference_cycles),
        "features": features,
        "label": {"name": label_col, "definition": label_definition},
        "preprocessing": {
            "impute_median": np.asarray(imputer.statistics_, dtype=np.float64).tolist(),
            "scaler_mean": np.asarray(scaler.mean_, dtype=np.float64).tolist(),
            "scaler_scale": np.asarray(scaler.scale_, dtype=np.float64).tolist(),
        },
        "estimator": _estimator_section(estimator, len(features)),
        "output_clip": [float(output_clip[0]), float(output_clip[1])],
        "conformal": {
            "method": conformal_method,
            "quantiles": quantiles,
            "n_scores": int(n_conformal_scores),
        },
    }
    core["domain"] = _domain_section(
        core, X_raw, margin=domain_margin, quantile=mahalanobis_quantile
    )
    core["training"] = {
        "batteries": sorted(str(b) for b in training_df[group_col].unique()),
        "n_cycles": len(training_df),
        "features_sha256": features_sha256 or _training_hash(training_df, [group_col, *features]),
    }
    core["metrics"] = _json_safe(dict(metrics))
    core["explanation"] = {
        "method": ATTRIBUTION_METHOD,
        "base_value": core["estimator"]["intercept"],
    }
    for key, value in (extra or {}).items():
        if key in core:
            raise ValueError(f"extra field {key!r} would overwrite a schema field")
        core[key] = _json_safe(value)

    # Self-check: the numpy runtime must reproduce scikit-learn on the training rows.
    runtime = LinearBundle.from_dict(json.loads(json.dumps(core, allow_nan=False)))
    fitted_on_frame = getattr(imputer, "feature_names_in_", None) is not None
    reference_input = training_df.loc[:, features] if fitted_on_frame else X_raw
    expected = np.asarray(pipeline.predict(reference_input), dtype=np.float64).reshape(-1)
    actual = runtime.predict_raw_array(X_raw)
    tolerance = _EXPORT_RTOL * max(1.0, float(np.max(np.abs(expected))))
    worst = float(np.max(np.abs(actual - expected)))
    if worst > tolerance:
        raise BundleError(f"exported bundle deviates from the pipeline by {worst:.3g}")
    logger.info(
        "exported bundle %s (%d features, %d training cycles)", model_id, len(features), len(X_raw)
    )
    return core


def save_bundle(bundle: Mapping[str, Any], path: Path) -> Path:
    """Write a bundle as strict, pretty-printed JSON with LF line endings (validated first).

    The bytes are identical on every OS (see :func:`write_text_lf`), so the
    SHA-256 that :func:`write_manifest` records for the file still matches
    after a git checkout anywhere.

    Raises:
        BundleError: If the document is not a valid bundle.
    """
    LinearBundle.from_dict(bundle)
    path = Path(path)
    digest = write_text_lf(path, json.dumps(bundle, indent=2, allow_nan=False) + "\n")
    logger.debug("saved bundle %s (sha256 %s)", path.name, digest[:12])
    return path
