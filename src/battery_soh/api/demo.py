"""Bundled demo data: a few real NASA cycles, small enough to ship with the image.

:func:`build_demo_assets` (CLI: ``soh demo-assets``) reads the raw ``.mat`` files of
the four core cells and writes ``demo_cycles.json.gz`` with, per cell:

* the first ``n_ref`` QC-valid cycles (the commissioning reference that
  relative-mode models need) plus every ``every``-th QC-valid cycle and the last
  one;
* for each of them the charge that preceded the discharge and the discharge
  itself as raw time series, downsampled to at most ``max_points`` samples. The
  measurement windows (3.85-4.20 V of the CC charge, 3.85-3.45 V under load)
  keep all their samples whenever the budget allows; transitions such as
  load-on, CC->CV and the end of the CV phase are always kept;
* the health indicators, the true SOH label and the QC flag exactly as the
  feature table has them (:func:`battery_soh.features.build.cycle_rows`, i.e.
  :func:`~battery_soh.features.extract.features_from_segments` on the *full
  resolution* arrays, with the table's charge pairing and EIS as-of rules);
* the cell's commissioning reference per indicator (mean over its first
  ``n_ref`` QC-valid non-NaN values, :func:`battery_soh.features.relative.reference_table`).

Label-side/proxy (``leg_*``) values are never written. The output is
byte-for-byte deterministic (fixed key order, fixed float rounding of the raw
arrays, gzip header without timestamp or file name).

:func:`get_demo_store` loads that file once (cached per file version) for the
``/v1/demo`` endpoints and the static-site export; each response document is
serialized once per store (:meth:`DemoStore.battery_list_json`,
:meth:`DemoStore.cycle_json`).
"""

from __future__ import annotations

import gzip
import io
import json
import logging
import math
import os
import re
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import TypeAdapter, ValidationError

from battery_soh import __version__
from battery_soh.api.http_cache import CachedJson, to_cached_json
from battery_soh.api.schemas import BATTERY_ID_PATTERN, INPUT_FEATURES, DemoBattery, DemoCycle
from battery_soh.config import DEFAULT_N_REFERENCE_CYCLES
from battery_soh.data.schema import BatteryRecord, Segment
from battery_soh.errors import BatterySohError

__all__ = [
    "DEMO_FILE_NAME",
    "DemoDataError",
    "DemoStore",
    "build_demo_assets",
    "demo_document",
    "downsample_indices",
    "get_demo_store",
    "load_demo_store",
    "write_demo_assets",
]

logger = logging.getLogger(__name__)

DEMO_FILE_NAME = "demo_cycles.json.gz"
DEMO_SCHEMA_VERSION = "1.0"
#: Refuse to decompress more than this (bytes) when loading.
MAX_DECOMPRESSED_BYTES = 64 * 1024 * 1024

_DECIMALS = {"time_s": 3, "voltage_v": 5, "current_a": 5, "temperature_c": 3}
_CHARGE_WINDOW_V = (3.85, 4.20)
_CHARGE_CV_V = 4.19
_CV_END_A = 0.1
_DISCHARGE_WINDOW_V = (3.45, 3.85)
_LOAD_ON_A = 0.5
_REFERENCE_FEATURES = tuple(f for f in INPUT_FEATURES if not f.startswith("ctx_"))


class DemoDataError(ValueError, BatterySohError):
    """The demo asset is missing required content or is malformed."""


# ====================================================================== downsampling
def _evenly(indices: NDArray[np.intp], count: int) -> NDArray[np.intp]:
    """``count`` evenly spaced members of ``indices`` (all of them if it is small)."""
    if count <= 0 or indices.size == 0:
        return np.empty(0, dtype=np.intp)
    if count >= indices.size:
        return indices
    picks = np.unique(np.round(np.linspace(0, indices.size - 1, count)).astype(np.intp))
    return indices[picks]


def _landmarks(seg: Segment) -> tuple[set[int], NDArray[np.bool_]]:
    """Indices that must survive downsampling, and the measurement-window mask."""
    n = seg.n_samples
    v, i = seg.voltage_v, seg.current_a
    idx = np.arange(n)
    must: set[int] = {0, n - 1, int(np.argmin(v)), int(np.argmax(v))}
    window = np.zeros(n, dtype=bool)
    if seg.kind == "charge":
        on = np.flatnonzero(i > _LOAD_ON_A)
        if on.size:
            k_on = int(on[0])
            must |= {k_on - 1, k_on, k_on + 1}
            reached = np.flatnonzero((idx >= k_on) & (v >= _CHARGE_CV_V))
            k_cv = int(reached[0]) if reached.size else n - 1
            must |= {k_cv - 1, k_cv, k_cv + 1}
            low, high = _CHARGE_WINDOW_V
            window = (idx >= k_on) & (idx <= k_cv + 2) & (v >= low) & (v <= high)
            ended = np.flatnonzero((idx > k_cv) & (i < _CV_END_A))
            if ended.size:
                k_end = int(ended[0])
                must |= {k_end - 1, k_end}
    else:
        loaded = np.flatnonzero(np.abs(i) > _LOAD_ON_A)
        if loaded.size:
            k_on, k_off = int(loaded[0]), int(loaded[-1])
            must |= {k_on - 1, k_on, k_on + 1, k_off, k_off + 1}
            low, high = _DISCHARGE_WINDOW_V
            window = (idx >= k_on) & (idx <= k_off) & (v >= low) & (v <= high)
    return {k for k in must if 0 <= k < n}, window


def downsample_indices(seg: Segment, max_points: int = 400) -> NDArray[np.intp]:
    """Sorted sample indices to keep, at most ``max_points`` of them.

    Landmarks (record ends, voltage extrema, current switch-on/off, CC->CV and
    end of CV) are always kept; the measurement window gets the remaining
    budget first (at least a quarter stays for the rest of the record), so
    window samples are kept at full resolution whenever they fit.

    Raises:
        ValueError: If ``max_points`` < 16.
    """
    if max_points < 16:
        raise ValueError(f"max_points must be >= 16, got {max_points}")
    n = seg.n_samples
    if n <= max_points:
        return np.arange(n, dtype=np.intp)
    must, window = _landmarks(seg)
    keep = np.zeros(n, dtype=bool)
    keep[list(must)] = True
    priority = np.flatnonzero(window & ~keep)
    rest = np.flatnonzero(~window & ~keep)
    budget = max_points - int(keep.sum())
    n_rest = min(rest.size, max(budget // 4, budget - priority.size))
    n_priority = min(priority.size, budget - n_rest)
    n_rest = min(rest.size, budget - n_priority)
    keep[_evenly(priority, n_priority)] = True
    keep[_evenly(rest, n_rest)] = True
    out = np.flatnonzero(keep).astype(np.intp)
    return out[:max_points] if out.size > max_points else out


def _segment_arrays(seg: Segment, max_points: int) -> dict[str, list[float]]:
    keep = downsample_indices(seg, max_points)
    out: dict[str, list[float]] = {}
    for name, decimals in _DECIMALS.items():
        values = np.round(np.asarray(getattr(seg, name), dtype=np.float64)[keep], decimals)
        if not np.all(np.isfinite(values)):
            raise DemoDataError(f"non-finite {name} values in a {seg.kind} record")
        out[name] = [float(v) for v in values]
    return out


# ====================================================================== building
def _paired_segments(record: BatteryRecord) -> list[tuple[Segment | None, Segment]]:
    """``(charge, discharge)`` per discharge, with the pairing rule of the feature table.

    Mirrors :func:`battery_soh.features.build.cycle_rows`: a discharge is paired
    with the charge that is the last charge/discharge operation before it
    (impedance measurements in between are ignored); an empty charge counts as none.
    """
    out: list[tuple[Segment | None, Segment]] = []
    pending: Segment | None = None
    for seg in record.segments:
        if seg.kind == "charge":
            pending = seg
        elif seg.kind == "discharge":
            out.append((pending if pending is not None and pending.n_samples else None, seg))
            pending = None
    return out


def _json_number(value: Any) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def _battery_document(
    record: BatteryRecord,
    *,
    every: int,
    max_points: int,
    n_ref: int,
) -> dict[str, Any]:
    """Demo entry of one cell; features, labels and QC come from the feature-table code."""
    from battery_soh.features.build import cycle_rows
    from battery_soh.features.relative import reference_table
    from battery_soh.labels import label_battery

    rows = cycle_rows(record)
    segments = _paired_segments(record)
    if not rows:
        raise DemoDataError(f"{record.battery_id} has no discharge cycles")
    paired = [charge is not None for charge, _ in segments]
    if paired != [bool(row["has_charge"]) for row in rows]:  # pragma: no cover - drift guard
        raise DemoDataError(f"{record.battery_id}: segment pairing does not match cycle_rows")
    labels = label_battery(pd.Series([row["capacity_ah"] for row in rows], dtype=np.float64))
    table = pd.DataFrame(
        [{name: float(row[name]) for name in INPUT_FEATURES} for row in rows],
        columns=list(INPUT_FEATURES),
    )
    table["battery_id"] = record.battery_id
    table["cycle_index"] = np.arange(len(rows))
    table["qc_valid"] = labels["qc_valid"].to_numpy(dtype=bool)
    soh = labels["soh_pct"].to_numpy(dtype=np.float64)
    reference_row = reference_table(table, _REFERENCE_FEATURES, n_ref=n_ref).loc[record.battery_id]

    valid = [k for k in range(len(rows)) if table["qc_valid"].iat[k] and math.isfinite(soh[k])]
    if not valid:
        raise DemoDataError(f"{record.battery_id} has no QC-valid cycle")
    reference_cycles = set(valid[:n_ref])
    selected = sorted(reference_cycles | set(valid[::every]) | {valid[-1]})

    entries = []
    for k in selected:
        charge, discharge = segments[k]
        features = {name: _json_number(table[name].iat[k]) for name in INPUT_FEATURES}
        eis_doc = {"re_ohm": features["eis_re_ohm"], "rct_ohm": features["eis_rct_ohm"]}
        entries.append(
            {
                "cycle_index": k,
                "soh_pct": _json_number(soh[k]),
                "qc_valid": True,
                "is_reference": k in reference_cycles,
                "features": features,
                "charge": None if charge is None else _segment_arrays(charge, max_points),
                "discharge": _segment_arrays(discharge, max_points),
                "eis": None if all(v is None for v in eis_doc.values()) else eis_doc,
            }
        )
    return {
        "battery_id": record.battery_id,
        "n_cycles": len(rows),
        "reference": {name: _json_number(reference_row[name]) for name in _REFERENCE_FEATURES},
        "cycles": entries,
    }


def demo_document(
    records: Iterable[BatteryRecord],
    *,
    every: int = 8,
    max_points: int = 400,
    n_ref: int = DEFAULT_N_REFERENCE_CYCLES,
) -> dict[str, Any]:
    """Build the demo document from parsed battery records.

    Args:
        records: Cells to include (parsed NASA records).
        every: Keep every ``every``-th QC-valid cycle (plus references and the last one).
        max_points: Largest number of samples per downsampled record.
        n_ref: Number of commissioning reference cycles.

    Returns:
        The JSON-ready document (see the module docstring).

    Raises:
        ValueError: On invalid arguments or a cell without usable cycles.
    """
    from battery_soh.features.extract import FEATURE_EXTRACTOR_VERSION

    if every < 1:
        raise ValueError(f"every must be >= 1, got {every}")
    if n_ref < 1:
        raise ValueError(f"n_ref must be >= 1, got {n_ref}")
    if max_points < 16:
        raise ValueError(f"max_points must be >= 16, got {max_points}")
    batteries = []
    for record in records:
        doc = _battery_document(record, every=every, max_points=max_points, n_ref=n_ref)
        logger.info("demo: %s, %d cycle(s) selected", record.battery_id, len(doc["cycles"]))
        batteries.append(doc)
    return {
        "schema_version": DEMO_SCHEMA_VERSION,
        "package_version": __version__,
        "feature_extractor_version": FEATURE_EXTRACTOR_VERSION,
        "source": "NASA PCoE Li-ion Battery Aging Data Set (B. Saha, K. Goebel)",
        "max_points": max_points,
        "n_reference_cycles": n_ref,
        "batteries": batteries,
    }


def write_demo_assets(document: Mapping[str, Any], out_dir: Path) -> Path:
    """Write ``document`` deterministically to ``out_dir/demo_cycles.json.gz`` (atomic).

    Raises:
        DemoDataError: If the document is not a valid demo asset.
    """
    raw = json.dumps(document, separators=(",", ":"), allow_nan=False).encode("utf-8")
    _parse_store(json.loads(raw))  # validate before publishing
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0, compresslevel=9) as gz:
        gz.write(raw)
    target = Path(out_dir) / DEMO_FILE_NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=".demo-", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(buffer.getvalue())
        Path(tmp).replace(target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    logger.info("demo assets written to %s (%d bytes)", target, target.stat().st_size)
    return target


def build_demo_assets(
    raw_dir: Path,
    out_dir: Path,
    *,
    battery_ids: Sequence[str] | None = None,
    every: int = 8,
    max_points: int = 400,
    n_ref: int = DEFAULT_N_REFERENCE_CYCLES,
) -> Path:
    """Write ``out_dir/demo_cycles.json.gz`` from the raw NASA files.

    Args:
        raw_dir: Folder with the NASA ``.mat`` files (searched recursively).
        out_dir: Output folder (created if needed).
        battery_ids: Cells to include (default: the four core cells).
        every: Keep every ``every``-th QC-valid cycle (plus references and the last one).
        max_points: Largest number of samples per downsampled record.
        n_ref: Number of commissioning reference cycles.

    Returns:
        Path of the written file.

    Raises:
        FileNotFoundError: If ``raw_dir`` or a requested cell's file is missing.
        ValueError: On invalid arguments or unusable data.
    """
    from battery_soh.data.nasa import find_mat_files, load_battery
    from battery_soh.data.protocols import CORE_BATTERIES

    ids = tuple(battery_ids) if battery_ids is not None else tuple(CORE_BATTERIES)
    files = find_mat_files(Path(raw_dir))
    missing = [b for b in ids if b not in files]
    if missing:
        raise FileNotFoundError(f"no .mat file for {missing} under {raw_dir}")
    records = (load_battery(files[battery_id], battery_id) for battery_id in ids)
    document = demo_document(records, every=every, max_points=max_points, n_ref=n_ref)
    return write_demo_assets(document, Path(out_dir))


# ====================================================================== loading
_BATTERY_LIST: TypeAdapter[list[DemoBattery]] = TypeAdapter(list[DemoBattery])


@dataclass(frozen=True)
class DemoStore:
    """Validated, in-memory demo data.

    The documents are read-only: the ``*_json`` methods serialize each one once
    (the HTTP endpoints send those bytes); the other accessors return fresh copies.
    """

    batteries: tuple[dict[str, Any], ...]
    cycles: Mapping[tuple[str, int], dict[str, Any]]
    _json: dict[object, CachedJson] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def battery_list_json(self) -> CachedJson:
        """``GET /v1/demo/batteries`` body, validated against the response schema once.

        Raises:
            pydantic.ValidationError: If the summaries do not match :class:`DemoBattery`.
        """
        cached = self._json.get("batteries")
        if cached is None:
            validated = _BATTERY_LIST.validate_python(list(self.batteries))
            cached = to_cached_json(_BATTERY_LIST.dump_python(validated, mode="json"))
            self._json["batteries"] = cached
        return cached

    def cycle_json(self, battery_id: str, cycle_index: int) -> CachedJson | None:
        """``GET /v1/demo/batteries/{id}/cycles/{k}`` body, or ``None`` for an unknown cycle."""
        key = (battery_id, cycle_index)
        cached = self._json.get(key)
        if cached is None:
            found = self.cycles.get(key)
            if found is None:
                return None
            cached = to_cached_json(found)  # validated as DemoCycle when loaded
            self._json[key] = cached
        return cached

    def battery_list(self) -> list[dict[str, Any]]:
        """``GET /v1/demo/batteries`` payload (a fresh copy)."""
        return list(json.loads(self.battery_list_json().body))

    def cycle(self, battery_id: str, cycle_index: int) -> dict[str, Any] | None:
        """``GET /v1/demo/batteries/{id}/cycles/{k}`` payload (a fresh copy), or ``None``."""
        cached = self.cycle_json(battery_id, cycle_index)
        return None if cached is None else dict(json.loads(cached.body))


def _reference_value(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise DemoDataError("reference values must be numbers or null")
    return _json_number(value)


def _parse_store(document: Any) -> DemoStore:
    if not isinstance(document, dict) or not isinstance(document.get("batteries"), list):
        raise DemoDataError("demo document must be an object with a 'batteries' list")
    id_re = re.compile(BATTERY_ID_PATTERN)
    summaries: list[dict[str, Any]] = []
    cycles: dict[tuple[str, int], dict[str, Any]] = {}
    for battery in document["batteries"]:
        if not isinstance(battery, dict):
            raise DemoDataError("battery entries must be objects")
        battery_id = battery.get("battery_id")
        if not isinstance(battery_id, str) or not id_re.match(battery_id):
            raise DemoDataError(f"invalid battery id {battery_id!r}")
        n_cycles = battery.get("n_cycles")
        reference = battery.get("reference", {})
        entries = battery.get("cycles")
        if isinstance(n_cycles, bool) or not isinstance(n_cycles, int):
            raise DemoDataError(f"{battery_id}: n_cycles must be an integer")
        if not isinstance(reference, dict) or not isinstance(entries, list):
            raise DemoDataError(f"{battery_id}: malformed reference or cycles")
        listed = []
        for entry in entries:
            if not isinstance(entry, dict):
                raise DemoDataError(f"{battery_id}: cycle entries must be objects")
            doc = {"battery_id": battery_id, **entry}
            try:
                cycle = DemoCycle.model_validate(doc).model_dump(mode="json")
            except ValidationError as exc:
                raise DemoDataError(f"{battery_id}: invalid demo cycle: {exc}") from None
            key = (battery_id, cycle["cycle_index"])
            if key in cycles:
                raise DemoDataError(f"{battery_id}: duplicate cycle {key[1]}")
            cycles[key] = cycle
            listed.append(
                {
                    "cycle_index": cycle["cycle_index"],
                    "soh_pct": cycle["soh_pct"],
                    "is_reference": cycle["is_reference"],
                }
            )
        summaries.append(
            {
                "battery_id": battery_id,
                "n_cycles": n_cycles,
                "reference": {str(k): _reference_value(v) for k, v in reference.items()},
                "cycles": sorted(listed, key=lambda c: c["cycle_index"]),
            }
        )
    store = DemoStore(batteries=tuple(summaries), cycles=cycles)
    try:
        store.battery_list_json()  # validate (and serialize) the battery list up front
    except (ValidationError, ValueError) as exc:
        raise DemoDataError(f"invalid demo battery list: {exc}") from None
    return store


def load_demo_store(path: Path) -> DemoStore:
    """Read and validate a demo asset.

    Raises:
        FileNotFoundError: If the file does not exist.
        DemoDataError: If it is not a valid demo asset (or decompresses too large).
    """
    try:
        with gzip.open(Path(path), "rb") as handle:
            raw = handle.read(MAX_DECOMPRESSED_BYTES + 1)
    except (OSError, EOFError) as exc:
        if isinstance(exc, FileNotFoundError):
            raise
        raise DemoDataError(f"cannot read demo asset: {exc}") from None
    if len(raw) > MAX_DECOMPRESSED_BYTES:
        raise DemoDataError("demo asset is too large")
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DemoDataError(f"demo asset is not valid JSON: {exc}") from None
    return _parse_store(document)


@lru_cache(maxsize=4)
def _cached_store(path: str, mtime_ns: int, size: int) -> DemoStore:
    del mtime_ns, size  # cache keys: a rewritten file is reloaded
    return load_demo_store(Path(path))


def get_demo_store(demo_dir: Path) -> DemoStore | None:
    """The demo data in ``demo_dir`` (cached per file version), or ``None`` if absent.

    Raises:
        DemoDataError: If the file exists but is invalid.
    """
    path = Path(demo_dir) / DEMO_FILE_NAME
    try:
        stat = path.stat()
    except OSError:
        return None
    return _cached_store(str(path.resolve()), stat.st_mtime_ns, stat.st_size)
