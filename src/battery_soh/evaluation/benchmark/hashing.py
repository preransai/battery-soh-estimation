"""Scoped SHA-256 hashes of ``configs/benchmark.yaml`` (no scikit-learn import).

``summary.json`` records ``config_sha256``, a hash of exactly the sections that
can change what ``soh benchmark`` writes; model bundles record a separate hash
of the ``production`` section that only ``soh train`` reads. Every top-level
key is in exactly one scope (:data:`BENCHMARK_HASH_SECTIONS`,
:data:`PRODUCTION_HASH_SECTIONS` or :data:`UNHASHED_SECTIONS`), so editing a
section never makes an artifact of the other step look stale.

This module only parses YAML and hashes JSON, so the report renderer and the
training code can compare hashes without importing the model zoo. It also holds
the two validation helpers (:func:`require_mapping`, :func:`check_keys`) shared
by the benchmark and the production sections of the configuration.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from datetime import date
from pathlib import Path
from typing import Any, Final, cast

import yaml

__all__ = [
    "BENCHMARK_HASH_SECTIONS",
    "PRODUCTION_HASH_SECTIONS",
    "TOP_KEYS",
    "UNHASHED_SECTIONS",
    "check_keys",
    "config_hashes",
    "config_sections_sha256",
    "read_raw_config",
    "require_mapping",
]

#: Every top-level key ``benchmark.yaml`` may contain (unknown keys are errors).
TOP_KEYS: Final = frozenset(
    {
        "schema_version",
        "description",
        "seed",
        "label",
        "n_reference_cycles",
        "confidence_levels",
        "n_jobs",
        "features_csv",
        "raw_dir",
        "conformal",
        "nested_selection",
        "headline",
        "secondary",
        "predictions",
        "experiments",
        "window_sweep",
        "quick",
        "production",
    }
)

#: Top-level sections of ``benchmark.yaml`` that determine what ``soh benchmark``
#: writes (results.csv, predictions.csv, summary.json, window_sweep.json, and the
#: quick-mode outputs). ``config_sha256`` hashes exactly these, so editing any
#: other section does not make a recorded benchmark look stale.
BENCHMARK_HASH_SECTIONS: Final[tuple[str, ...]] = (
    "seed",
    "label",
    "n_reference_cycles",
    "confidence_levels",
    "conformal",
    "nested_selection",
    "headline",
    "secondary",
    "predictions",
    "experiments",
    "window_sweep",
    "quick",
)
#: The section read only by ``soh train``; training records its own hash of it.
PRODUCTION_HASH_SECTIONS: Final[tuple[str, ...]] = ("production",)
#: Top-level keys in neither hash, and why the benchmark's results do not depend on them.
UNHASHED_SECTIONS: Final[Mapping[str, str]] = {
    "schema_version": "version of the file layout; validated by the key checks, not a setting",
    "description": "free text",
    "n_jobs": "worker processes; results are identical for any value",
    "features_csv": "location only; the table's content is hashed as data_sha256",
    "raw_dir": "location of the raw files read by the window sweep",
}
#: Free-text keys ignored inside hashed sections when their value is a string (never
#: echoed in the outputs); a key of that name holding settings is still hashed.
_UNHASHED_TEXT_KEYS: Final = frozenset({"description"})


def require_mapping(value: Any, where: str) -> dict[str, Any]:
    """``value`` as a dict, or a ``ValueError`` naming ``where``."""
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be a mapping, got {type(value).__name__}")
    return cast(dict[str, Any], value)


def check_keys(section: Mapping[str, Any], allowed: Iterable[str], where: str) -> None:
    """Raise a ``ValueError`` naming ``where`` if ``section`` has a key outside ``allowed``."""
    allowed_keys = set(allowed)
    unknown = sorted(set(section) - allowed_keys)
    if unknown:
        raise ValueError(f"unknown key(s) {unknown} in {where}; allowed: {sorted(allowed_keys)}")


def read_raw_config(config_path: Path) -> dict[str, Any]:
    """The parsed YAML document (UTF-8), which must be a mapping.

    Raises:
        FileNotFoundError: If the config does not exist.
        ValueError: If it is not a YAML mapping.
    """
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"benchmark config not found: {config_path}")
    return require_mapping(
        yaml.safe_load(config_path.read_bytes().decode("utf-8")), "benchmark config"
    )


def _hashable(value: Any) -> Any:
    """``value`` without free-text entries, with YAML dates as ISO strings."""
    if isinstance(value, Mapping):
        return {
            str(k): _hashable(v)
            for k, v in value.items()
            if not (str(k) in _UNHASHED_TEXT_KEYS and isinstance(v, str))
        }
    if isinstance(value, list | tuple):
        return [_hashable(v) for v in value]
    if isinstance(value, date):  # also datetime
        return value.isoformat()
    return value


def config_sections_sha256(raw: Mapping[str, Any], sections: Iterable[str]) -> str:
    """SHA-256 of a canonical serialisation of some top-level sections of a parsed config.

    Canonical form: the parsed YAML values of ``sections`` (absent sections are
    omitted; nested free-text ``description`` strings are dropped) as JSON with
    sorted keys, no whitespace and ASCII escapes. The digest is therefore insensitive to
    comments, formatting, key order, quoting style and edits of other sections,
    and sensitive to every value that can change a result (including list order).

    Args:
        raw: The parsed YAML document.
        sections: Top-level keys to include.

    Returns:
        Hex SHA-256 digest.

    Raises:
        ValueError: If a value is not JSON-serialisable or not finite.
    """
    subset = {name: _hashable(raw[name]) for name in sorted(set(sections)) if name in raw}
    try:
        text = json.dumps(
            subset, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"config sections {sorted(subset)} cannot be hashed: {exc}") from exc
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def config_hashes(config_path: Path) -> dict[str, str]:
    """The two scoped hashes of ``benchmark.yaml``.

    Returns:
        ``{"benchmark": <hash of BENCHMARK_HASH_SECTIONS>, "production": <hash
        of PRODUCTION_HASH_SECTIONS>}``; the first is what ``summary.json``
        records as ``config_sha256``, the second what model bundles record as
        ``production_config_sha256``.

    Raises:
        FileNotFoundError: If the config does not exist.
        ValueError: If it is not a YAML mapping.
    """
    raw = read_raw_config(config_path)
    return {
        "benchmark": config_sections_sha256(raw, BENCHMARK_HASH_SECTIONS),
        "production": config_sections_sha256(raw, PRODUCTION_HASH_SECTIONS),
    }
