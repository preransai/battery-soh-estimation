"""Build the per-cycle feature table from the raw NASA files.

The table has one row per discharge operation. Each row holds

* metadata: the cell, its campaign group, the 0-based discharge count, the start time and the
  test conditions;
* labels: the measured capacity, the reference capacity ``C0``, both SOH labels and the
  capacity QC flag (label side - never model inputs, see :mod:`battery_soh.labels`);
* every registry feature (:data:`battery_soh.features.registry.FEATURES`), in registry order.

Joining rules (they decide what information a row may contain, so they are leakage rules):

* **Charge pairing.** A discharge is paired with a charge only when the operation right
  before it, ignoring impedance measurements, is a charge. If a discharge directly follows
  another discharge, its charge features are NaN rather than borrowed from an older charge.
  Two consecutive charges (an interrupted charge that was restarted) pair the discharge with
  the second one, the charge that actually conditioned it.
* **EIS as-of join.** The ``eis_*`` features come from the most recent impedance measurement
  recorded *before* the discharge. Cycles before the first measurement get NaN (no
  back-filling from the future). A measurement whose fitted ``Re`` or ``Rct`` is missing,
  non-positive or outside the registry's plausible range (diverged equivalent-circuit fits:
  up to 1e15 ohm in B0050 and B0052) is treated as a failed measurement and skipped, so the
  previous valid measurement stays in effect.
* **Context.** ``ctx_cycle_index`` is the 0-based discharge count (QC-invalid discharges
  included, so it is a time axis). ``ctx_ambient_c`` is the chamber temperature recorded with
  the discharge, falling back to the protocol registry's nominal value. The metadata columns
  ``ambient_c`` and ``discharge_current_a`` duplicate ``ctx_ambient_c`` and the measured
  ``ctx_discharge_current_a``.

Labels and QC are computed per battery over all its discharges in recorded order
(:func:`battery_soh.labels.label_battery`). Feature values come from
:func:`battery_soh.features.extract.features_from_segments` (re-exported here, where the build
specification lists it), the same function the API calls, so training data and served features
are computed identically.

The committed CSV (``data/processed/nasa_features.csv``) is written by
:func:`write_feature_table` with 6 significant digits, LF line endings and a deterministic row
order, next to a ``.meta.json`` sidecar recording its SHA-256.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from battery_soh import __version__
from battery_soh.config import get_paths
from battery_soh.data.nasa import NASA_BATTERY_IDS, iter_batteries
from battery_soh.data.protocols import PROTOCOLS
from battery_soh.data.schema import BatteryRecord, Segment
from battery_soh.features._signal import finite_or_nan
from battery_soh.features.extract import (
    FEATURE_EXTRACTOR_VERSION,
    METADATA_FEATURES,
    features_from_segments,
)
from battery_soh.features.impedance import EIS_FEATURES, impedance_values
from battery_soh.features.registry import FEATURES
from battery_soh.labels import label_battery

logger = logging.getLogger(__name__)

__all__ = [
    "BOOL_COLUMNS",
    "CYCLE_ROW_COLUMNS",
    "FEATURE_TABLE_COLUMNS",
    "FLOAT_FORMAT",
    "KEY_COLUMNS",
    "LABEL_COLUMNS",
    "MAX_CSV_BYTES",
    "METADATA_COLUMNS",
    "META_SCHEMA_VERSION",
    "STRING_COLUMNS",
    "UNREGISTERED_GROUP",
    "battery_table",
    "build_feature_table",
    "cycle_rows",
    "features_from_segments",
    "load_feature_table",
    "metadata_path",
    "write_feature_table",
]

#: Identification and test-condition columns (first block of the table).
METADATA_COLUMNS: Final[tuple[str, ...]] = (
    "battery_id",
    "group",
    "cycle_index",
    "start_time",
    "ambient_c",
    "discharge_current_a",
    "has_charge",
)
#: Label-side columns (second block). Derived from the measured capacity: never inputs.
LABEL_COLUMNS: Final[tuple[str, ...]] = (
    "capacity_ah",
    "c0_ah",
    "soh_pct",
    "soh_rated_pct",
    "qc_valid",
    "qc_reason",
)
#: The feature-table contract: exact column order of the table and of the CSV.
FEATURE_TABLE_COLUMNS: Final[tuple[str, ...]] = (
    *METADATA_COLUMNS,
    *LABEL_COLUMNS,
    *FEATURES,
)
#: Keys of :func:`cycle_rows` dicts (everything except ``group`` and the derived labels).
CYCLE_ROW_COLUMNS: Final[tuple[str, ...]] = (
    *(c for c in METADATA_COLUMNS if c != "group"),
    "capacity_ah",
    *FEATURES,
)
#: Primary key of the table; rows are sorted by it.
KEY_COLUMNS: Final[tuple[str, str]] = ("battery_id", "cycle_index")
#: Columns stored as strings / booleans; ``start_time`` is a timestamp, ``cycle_index`` an
#: integer and every other column a float.
STRING_COLUMNS: Final[tuple[str, ...]] = ("battery_id", "group", "qc_reason")
BOOL_COLUMNS: Final[tuple[str, ...]] = ("has_charge", "qc_valid")
_TIME_COLUMN: Final = "start_time"
_INT_COLUMN: Final = "cycle_index"
_FLOAT_COLUMNS: Final[tuple[str, ...]] = tuple(
    c
    for c in FEATURE_TABLE_COLUMNS
    if c not in (*STRING_COLUMNS, *BOOL_COLUMNS, _TIME_COLUMN, _INT_COLUMN)
)

#: ``group`` of a cell that is not in the protocol registry.
UNREGISTERED_GROUP: Final = "unregistered"
#: Float format of the CSV: 6 significant digits (far below sensor resolution).
FLOAT_FORMAT: Final = "%.6g"
#: Version of the ``.meta.json`` sidecar layout.
META_SCHEMA_VERSION: Final = "1.0"
#: The committed CSV must stay small enough to live in git.
MAX_CSV_BYTES: Final = 3_000_000

_NAN: Final = math.nan


# --------------------------------------------------------------------------- per-cycle rows


def _plausible_eis(seg: Segment) -> tuple[float, float] | None:
    """``(re_ohm, rct_ohm)`` of an impedance segment, or None for a failed fit."""
    values = impedance_values(seg)  # NaN for missing / non-finite / non-positive values
    for name, value in zip(EIS_FEATURES, values, strict=True):
        low, high = FEATURES[name].plausible_range
        if not low <= value <= high:  # also False for NaN
            return None
    return values


def cycle_rows(record: BatteryRecord) -> list[dict[str, Any]]:
    """Turn one battery's operations into one row per discharge.

    See the module docstring for the pairing, as-of and context rules.

    Args:
        record: A parsed battery (operations in recorded order).

    Returns:
        One dict per discharge, in recorded order, with the keys of
        :data:`CYCLE_ROW_COLUMNS`: ``battery_id``, ``cycle_index`` (int), ``start_time``
        (``datetime`` or None), ``ambient_c``, ``discharge_current_a``, ``has_charge``
        (bool), ``capacity_ah`` (label numerator; NaN when not recorded) and every registry
        feature as a float (NaN when not computable).
    """
    protocol = PROTOCOLS.get(record.battery_id)
    nominal_ambient = protocol.ambient_c if protocol is not None else _NAN

    rows: list[dict[str, Any]] = []
    pending_charge: Segment | None = None  # set iff the last non-impedance op is a charge
    eis: tuple[float, float] | None = None
    n_failed_eis = 0
    for seg in record.segments:
        if seg.kind == "impedance":
            measured = _plausible_eis(seg)
            if measured is None:
                n_failed_eis += 1
            else:
                eis = measured
            continue
        if seg.kind == "charge":
            pending_charge = seg
            continue

        charge = pending_charge if pending_charge is not None and pending_charge.n_samples else None
        pending_charge = None
        cycle_index = len(rows)
        ambient = finite_or_nan(seg.ambient_c)
        if math.isnan(ambient):
            ambient = nominal_ambient
        measured_features = features_from_segments(charge, seg, eis=eis)
        context = {"ctx_cycle_index": float(cycle_index), "ctx_ambient_c": ambient}
        row: dict[str, Any] = {
            "battery_id": record.battery_id,
            "cycle_index": cycle_index,
            "start_time": seg.start_time,
            "ambient_c": ambient,
            "discharge_current_a": measured_features["ctx_discharge_current_a"],
            "has_charge": charge is not None,
            "capacity_ah": finite_or_nan(seg.capacity_ah),
        }
        for name in FEATURES:
            row[name] = context[name] if name in METADATA_FEATURES else measured_features[name]
        rows.append(row)

    if n_failed_eis:
        logger.debug(
            "%s: skipped %d impedance measurements with implausible fits",
            record.battery_id,
            n_failed_eis,
        )
    return rows


# --------------------------------------------------------------------------- tables


def _coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with every contract column in its canonical dtype.

    Raises:
        ValueError: If a flag column has missing values or a numeric column holds text.
    """
    out = df.copy()
    for name in STRING_COLUMNS:
        out[name] = out[name].astype(str)
    for name in BOOL_COLUMNS:
        if out[name].isna().any():  # astype(bool) would silently turn NaN into True
            raise ValueError(f"column {name!r} has missing values")
        out[name] = out[name].astype(bool)
    out[_INT_COLUMN] = out[_INT_COLUMN].astype(np.int64)
    out[_TIME_COLUMN] = pd.to_datetime(out[_TIME_COLUMN]).astype("datetime64[ms]")
    for name in _FLOAT_COLUMNS:
        out[name] = pd.to_numeric(out[name], errors="raise").astype(np.float64)
    return out


def _empty_table() -> pd.DataFrame:
    return _coerce_dtypes(pd.DataFrame({c: pd.Series(dtype=object) for c in FEATURE_TABLE_COLUMNS}))


def battery_table(record: BatteryRecord) -> pd.DataFrame:
    """Feature table of one battery: :func:`cycle_rows` plus group, QC and labels.

    Args:
        record: A parsed battery.

    Returns:
        A DataFrame with exactly :data:`FEATURE_TABLE_COLUMNS`, one row per discharge in cycle
        order. Labels use only this battery's capacities (``C0`` = median of its first five
        QC-valid discharges). A cell missing from the protocol registry gets
        ``group = "unregistered"``.
    """
    rows = cycle_rows(record)
    if not rows:
        logger.warning(
            "%s: no discharge operations; the cell contributes no rows", record.battery_id
        )
        return _empty_table()
    frame = pd.DataFrame(rows, columns=list(CYCLE_ROW_COLUMNS))
    protocol = PROTOCOLS.get(record.battery_id)
    if protocol is None:
        logger.warning(
            "%s is not in the protocol registry; group set to %r",
            record.battery_id,
            UNREGISTERED_GROUP,
        )
    frame["group"] = protocol.group if protocol is not None else UNREGISTERED_GROUP
    labels = label_battery(frame["capacity_ah"])
    for name in LABEL_COLUMNS:
        frame[name] = labels[name]
    return _coerce_dtypes(frame.loc[:, list(FEATURE_TABLE_COLUMNS)])


def build_feature_table(raw_dir: Path, battery_ids: Sequence[str] | None = None) -> pd.DataFrame:
    """Build the feature table of the NASA cells found under ``raw_dir``.

    Args:
        raw_dir: Directory searched recursively for ``B00XX.mat`` files.
        battery_ids: Cells to include (aliases such as ``"b5"`` are accepted). ``None`` builds
            every cell found - all 34 for the complete archive.

    Returns:
        A DataFrame with exactly :data:`FEATURE_TABLE_COLUMNS`, sorted by
        ``(battery_id, cycle_index)`` with a fresh ``RangeIndex``.

    Raises:
        FileNotFoundError: If ``raw_dir`` is missing, holds no battery files, or a requested
            battery has no file.
        ValueError: If ``battery_ids`` is empty or an identifier is malformed.
    """
    if battery_ids is not None and len(battery_ids) == 0:
        raise ValueError("battery_ids must not be empty (pass None to build every cell)")
    frames: list[pd.DataFrame] = []
    for record in iter_batteries(Path(raw_dir), battery_ids):
        table = battery_table(record)
        logger.debug(
            "%s: %d cycles (%d QC-valid)",
            record.battery_id,
            len(table),
            int(table["qc_valid"].sum()),
        )
        frames.append(table)
    if not frames:
        raise FileNotFoundError(f"no NASA battery files (B00XX.mat) found under {raw_dir}")
    if battery_ids is None:
        built = {str(f["battery_id"].iloc[0]) for f in frames if len(f)}
        missing = [b for b in NASA_BATTERY_IDS if b not in built]
        if missing:
            logger.warning(
                "%d registry batteries are absent from %s: %s", len(missing), raw_dir, missing
            )
    table = pd.concat([f for f in frames if len(f)] or [_empty_table()], ignore_index=True)
    table = table.sort_values(list(KEY_COLUMNS), kind="stable", ignore_index=True)
    logger.info(
        "Built feature table: %d cycles (%d QC-valid) from %d batteries",
        len(table),
        int(table["qc_valid"].sum()),
        table["battery_id"].nunique(),
    )
    return table


# --------------------------------------------------------------------------- persistence


def metadata_path(csv_path: Path) -> Path:
    """Path of the ``.meta.json`` sidecar of a feature CSV (``x.csv`` -> ``x.meta.json``)."""
    csv_path = Path(csv_path)
    return csv_path.with_name(f"{csv_path.stem}.meta.json")


def _validate_contract(df: pd.DataFrame) -> None:
    columns = list(df.columns)
    if len(set(columns)) != len(columns):
        raise ValueError("feature table has duplicate column names")
    missing = [c for c in FEATURE_TABLE_COLUMNS if c not in df.columns]
    extra = [c for c in columns if c not in FEATURE_TABLE_COLUMNS]
    if missing or extra:
        raise ValueError(
            f"feature table does not match the column contract (missing={missing}, extra={extra})"
        )
    if df[list(KEY_COLUMNS)].isna().any().any():
        raise ValueError("feature table has missing battery_id or cycle_index values")
    duplicated = df.duplicated(list(KEY_COLUMNS))
    if duplicated.any():
        keys = df.loc[duplicated, list(KEY_COLUMNS)].head(5).to_records(index=False).tolist()
        raise ValueError(f"duplicate (battery_id, cycle_index) keys, e.g. {keys}")


def _format_timestamps(values: pd.Series) -> pd.Series:
    stamps = pd.to_datetime(values)
    text = stamps.dt.strftime("%Y-%m-%dT%H:%M:%S.%f").str.slice(0, 23)  # millisecond precision
    return text.where(stamps.notna(), "")


def _atomic_write(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` via a temporary file in the same directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _counts(df: pd.DataFrame, mask: pd.Series | None = None) -> dict[str, int]:
    ids = df["battery_id"] if mask is None else df.loc[mask, "battery_id"]
    counts = ids.value_counts()
    return {str(b): int(counts.get(b, 0)) for b in sorted(df["battery_id"].unique())}


def _nan_rates(df: pd.DataFrame) -> dict[str, float | None]:
    if df.empty:
        return dict.fromkeys(FEATURES)
    return {name: round(float(df[name].isna().mean()), 4) for name in FEATURES}


def _metadata(df: pd.DataFrame, *, file_name: str, sha256: str, n_bytes: int) -> dict[str, Any]:
    return {
        "schema_version": META_SCHEMA_VERSION,
        "file": file_name,
        "sha256": sha256,
        "bytes": n_bytes,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "package_version": __version__,
        "feature_extractor_version": FEATURE_EXTRACTOR_VERSION,
        "float_format": FLOAT_FORMAT,
        "n_rows": len(df),
        "n_columns": df.shape[1],
        "columns": list(df.columns),
        "row_counts": _counts(df),
        "qc_valid_counts": _counts(df, df["qc_valid"].astype(bool)),
        "nan_rate": _nan_rates(df),
    }


def write_feature_table(df: pd.DataFrame, path: Path) -> Path:
    """Write the feature table as a deterministic CSV plus a ``.meta.json`` sidecar.

    The CSV has the :data:`FEATURE_TABLE_COLUMNS` order, rows sorted by
    ``(battery_id, cycle_index)``, floats with 6 significant digits, NaN as an empty field,
    ``start_time`` as ISO 8601 local laboratory time with millisecond precision, UTF-8 and LF
    line endings, so the same input always produces byte-identical output. The sidecar
    (:func:`metadata_path`) records the CSV's SHA-256 and size, the package and extractor
    versions, per-battery row and QC-valid counts, the NaN rate of every feature and the
    build time (UTC). Both files are written atomically.

    Args:
        df: A table from :func:`build_feature_table` (exactly the contract columns, in any
            order).
        path: Target CSV path; parent directories are created.

    Returns:
        ``path``.

    Raises:
        ValueError: If the columns do not match the contract or a key is missing/duplicated.
    """
    path = Path(path)
    _validate_contract(df)
    table = _coerce_dtypes(df.loc[:, list(FEATURE_TABLE_COLUMNS)])
    table = table.sort_values(list(KEY_COLUMNS), kind="stable", ignore_index=True)
    out = table.copy()
    out[_TIME_COLUMN] = _format_timestamps(out[_TIME_COLUMN])
    text = out.to_csv(index=False, float_format=FLOAT_FORMAT, lineterminator="\n")
    data = text.encode("utf-8")
    if len(data) > MAX_CSV_BYTES:
        logger.warning(
            "Feature CSV is %.1f MB (limit for the committed file: %.1f MB)",
            len(data) / 1e6,
            MAX_CSV_BYTES / 1e6,
        )
    _atomic_write(path, data)
    meta = _metadata(
        table, file_name=path.name, sha256=hashlib.sha256(data).hexdigest(), n_bytes=len(data)
    )
    meta_text = json.dumps(meta, indent=2, allow_nan=False) + "\n"
    _atomic_write(metadata_path(path), meta_text.encode("utf-8"))
    logger.info("Wrote %d rows to %s (sha256 %s)", len(table), path, meta["sha256"][:12])
    return path


def _parse_bool(values: pd.Series, name: str) -> pd.Series:
    mapping: Mapping[str, bool] = {"True": True, "False": False}
    parsed = values.map(mapping)
    bad = parsed.isna()
    if bad.any():
        examples = values[bad].unique()[:3].tolist()
        raise ValueError(f"column {name!r} holds non-boolean values {examples}")
    return parsed.astype(bool)


def load_feature_table(path: Path | None = None) -> pd.DataFrame:
    """Read a feature CSV written by :func:`write_feature_table`, restoring dtypes.

    Args:
        path: CSV path; defaults to ``data/processed/nasa_features.csv``
            (:attr:`battery_soh.config.ProjectPaths.features_csv`).

    Returns:
        The table: ``battery_id``, ``group`` and ``qc_reason`` as strings, ``cycle_index`` as
        int64, ``has_charge`` and ``qc_valid`` as bool, ``start_time`` as naive
        ``datetime64`` (NaT when unknown) and every other column as float64.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the header does not match :data:`FEATURE_TABLE_COLUMNS` (for example a
            table built before a registry change: rebuild it with ``soh data build``) or a
            value cannot be parsed.
    """
    path = Path(path) if path is not None else get_paths().features_csv
    if not path.is_file():
        raise FileNotFoundError(f"feature table not found: {path} (run `soh data build`)")
    header = tuple(pd.read_csv(path, nrows=0, encoding="utf-8").columns)
    if header != FEATURE_TABLE_COLUMNS:
        missing = [c for c in FEATURE_TABLE_COLUMNS if c not in header]
        extra = [c for c in header if c not in FEATURE_TABLE_COLUMNS]
        raise ValueError(
            f"{path} does not match the feature-table contract (missing={missing}, "
            f"extra={extra}, or a different order); rebuild it with `soh data build`"
        )
    # Only empty fields are missing values, and only in float/time columns: a string such as
    # "NA" or "None" must never be turned into NaN behind the caller's back.
    dtypes: dict[str, Any] = dict.fromkeys(_FLOAT_COLUMNS, "float64")
    dtypes.update(dict.fromkeys((*STRING_COLUMNS, *BOOL_COLUMNS, _TIME_COLUMN), str))
    dtypes[_INT_COLUMN] = "int64"
    na_values = {name: [""] for name in (*_FLOAT_COLUMNS, _TIME_COLUMN)}
    table = pd.read_csv(
        path, dtype=dtypes, keep_default_na=False, na_values=na_values, encoding="utf-8"
    )
    for name in BOOL_COLUMNS:
        table[name] = _parse_bool(table[name], name)
    table[_TIME_COLUMN] = pd.to_datetime(table[_TIME_COLUMN], format="ISO8601").astype(
        "datetime64[ms]"
    )
    return table
