"""Reader for the NASA PCoE Li-ion "Battery Data Set" (MATLAB ``.mat`` files).

Every file ``B00XX.mat`` holds one MATLAB variable named after the cell. Its layout is::

    mat["B0005"][0, 0]["cycle"][0]        # 1 x N struct array, one entry per operation
        .type                             # "charge" | "discharge" | "impedance"
        .ambient_temperature              # chamber temperature, deg C
        .time                             # start time as a MATLAB datevec [Y M D h m s]
        .data[0, 0]                       # measurements
            charge/discharge: Voltage_measured, Current_measured, Temperature_measured,
                              Time (s from operation start), Capacity (discharge only, Ah)
            impedance:        Re, Rct (Ohm) plus raw spectra that are not used here

The reader turns each operation into a :class:`~battery_soh.data.schema.Segment` in
recorded order. It is deliberately strict about structure (a malformed file raises
:class:`MatFormatError` naming the battery and operation) and tolerant about values that the
public files are known to contain:

* About 25 charge records contain non-finite samples (1-2 in most, 57 in B0041-B0044); a
  sample is dropped when any of its four channels is non-finite, so the arrays stay aligned.
* Empty ``Capacity`` fields (4 discharges of B0050, 21 of B0052) become ``None``.
* EIS fits are stored as ``abs(real(x))``, the convention used by the audit. Some fits of
  B0050 and B0052 diverged (``|Re|`` up to 1e15 Ohm); plausibility is judged downstream.

B0025-B0028 ship twice in the official archive (byte-identical copies in two folders);
:func:`find_mat_files` keeps one copy deterministically.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Iterable, Iterator
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import scipy.io
from scipy.io.matlab import MatReadError

from battery_soh._typing import FloatArray
from battery_soh.data.schema import BatteryRecord, Segment, SegmentKind
from battery_soh.errors import BatterySohError

logger = logging.getLogger(__name__)

__all__ = [
    "NASA_BATTERY_IDS",
    "MatFormatError",
    "datevec_to_datetime",
    "find_mat_files",
    "iter_batteries",
    "load_battery",
    "normalize_battery_id",
]

#: The 34 distinct cells in the official NASA PCoE "Battery Data Set" archive.
NASA_BATTERY_IDS: Final[tuple[str, ...]] = (
    "B0005",
    "B0006",
    "B0007",
    "B0018",
    *(f"B00{n}" for n in range(25, 35)),
    "B0036",
    *(f"B00{n}" for n in range(38, 57)),
)

_BATTERY_ID_RE: Final = re.compile(r"^[Bb]?0*(\d{1,4})$")
_FILE_STEM_RE: Final = re.compile(r"^B\d{4}$", re.IGNORECASE)
_SEGMENT_KINDS: Final[frozenset[str]] = frozenset({"charge", "discharge", "impedance"})
_TIMESERIES_FIELDS: Final[tuple[str, ...]] = (
    "Time",
    "Voltage_measured",
    "Current_measured",
    "Temperature_measured",
)
_HASH_CHUNK_BYTES: Final = 1 << 20


class MatFormatError(ValueError, BatterySohError):
    """Raised when a ``.mat`` file does not have the NASA PCoE battery structure."""


def normalize_battery_id(value: str) -> str:
    """Return the canonical ``B0005``-style identifier.

    Args:
        value: An identifier such as ``"B0005"``, ``"b0005"``, ``"B5"`` or ``"5"``.

    Returns:
        The upper-case, zero-padded identifier.

    Raises:
        ValueError: If ``value`` is not a battery number.
    """
    match = _BATTERY_ID_RE.match(value.strip())
    if match is None:
        raise ValueError(f"not a NASA battery id: {value!r} (expected e.g. 'B0005')")
    return f"B{int(match.group(1)):04d}"


def datevec_to_datetime(vec: object) -> datetime | None:
    """Convert a MATLAB date vector ``[Y, M, D, h, m, s]`` to a naive ``datetime``.

    Seconds may be fractional. Hours, minutes and seconds are added as a ``timedelta`` so
    values such as ``s = 60.0`` (a rounding artefact some loggers produce) roll over
    correctly instead of raising.

    Args:
        vec: Anything array-like holding six numbers.

    Returns:
        The timestamp (local laboratory time, no timezone), or ``None`` when the vector is
        missing, has the wrong length, contains non-finite values or is not a valid date.
    """
    try:
        arr = np.asarray(vec, dtype=np.float64).ravel()
    except (TypeError, ValueError):
        return None
    if arr.size != 6 or not np.all(np.isfinite(arr)):
        return None
    year, month, day = (round(float(x)) for x in arr[:3])
    try:
        midnight = datetime(year, month, day)  # naive: the lab clock's timezone is not recorded
    except ValueError:
        return None
    return midnight + timedelta(hours=float(arr[3]), minutes=float(arr[4]), seconds=float(arr[5]))


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_mat_files(root: Path) -> dict[str, Path]:
    """Discover NASA battery files below ``root``.

    Files are matched recursively by name (``B0005.mat``, case-insensitive); anything else,
    such as macOS ``._B0005.mat`` resource forks, is ignored, and so are hidden folders (for
    example the staging folder of an interrupted download). When the same battery appears
    more than once (the official archive ships B0025-B0028 in two folders) the first path in
    sorted POSIX order wins; the copies are compared by MD5 and a differing copy is reported
    as a warning.

    Args:
        root: Directory to search.

    Returns:
        Mapping ``battery_id -> path``, sorted by battery id.

    Raises:
        FileNotFoundError: If ``root`` is not an existing directory.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(
            f"NASA data directory not found: {root} (run `soh data download` to fetch it)"
        )
    candidates = sorted(
        (
            p
            for p in root.rglob("*")
            if p.suffix.lower() == ".mat"
            and not any(part.startswith(".") for part in p.relative_to(root).parts[:-1])
            and p.is_file()
        ),
        key=lambda p: p.as_posix(),
    )
    found: dict[str, Path] = {}
    for path in candidates:
        if not _FILE_STEM_RE.match(path.stem):
            logger.debug("Ignoring non-battery .mat file %s", path)
            continue
        battery_id = path.stem.upper()
        kept = found.get(battery_id)
        if kept is None:
            found[battery_id] = path
            continue
        kept_md5, dup_md5 = _md5(kept), _md5(path)
        if kept_md5 == dup_md5:
            logger.debug(
                "Duplicate %s: %s is byte-identical to %s (md5 %s); keeping the first",
                battery_id,
                path,
                kept,
                kept_md5,
            )
        else:
            logger.warning(
                "Conflicting copies of %s: keeping %s (md5 %s), ignoring %s (md5 %s)",
                battery_id,
                kept,
                kept_md5,
                path,
                dup_md5,
            )
    return dict(sorted(found.items()))


# ---------------------------------------------------------------------------- parsing helpers


def _field(struct: Any, name: str, context: str) -> Any:
    names = struct.dtype.names or ()
    if name not in names:
        raise MatFormatError(f"{context}: missing field {name!r} (available: {list(names)})")
    return struct[name]


def _first_string(value: Any) -> str:
    arr = np.ravel(np.asarray(value))
    return str(arr[0]).strip().lower() if arr.size else ""


def _first_float(value: Any) -> float | None:
    arr = np.ravel(np.asarray(value))
    if arr.size == 0:
        return None
    try:
        out = float(np.real(arr[0]))
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _abs_real(value: Any) -> float | None:
    out = _first_float(value)
    return None if out is None else abs(out)


def _timeseries(data: Any, context: str) -> tuple[FloatArray, ...]:
    arrays = tuple(
        np.asarray(_field(data, name, context), dtype=np.float64).ravel()
        for name in _TIMESERIES_FIELDS
    )
    sizes = {name: arr.size for name, arr in zip(_TIMESERIES_FIELDS, arrays, strict=True)}
    if len(set(sizes.values())) > 1:
        raise MatFormatError(f"{context}: arrays have unequal lengths {sizes}")
    finite = np.logical_and.reduce([np.isfinite(arr) for arr in arrays])
    if not finite.all():
        logger.debug("%s: dropping %d non-finite samples", context, int((~finite).sum()))
        arrays = tuple(arr[finite] for arr in arrays)
    if arrays[0].size > 1 and np.any(np.diff(arrays[0]) < 0):
        logger.warning("%s: time vector is not monotone non-decreasing", context)
    return arrays


def _parse_operation(op: Any, context: str) -> Segment | None:
    kind = _first_string(_field(op, "type", context))
    if kind not in _SEGMENT_KINDS:
        logger.warning("%s: skipping operation of unknown type %r", context, kind)
        return None
    seg_kind = cast(SegmentKind, kind)  # checked against _SEGMENT_KINDS above
    start_time = datevec_to_datetime(_field(op, "time", context))
    ambient_c = _first_float(_field(op, "ambient_temperature", context))
    data_field = np.asarray(_field(op, "data", context))
    if data_field.size == 0:
        logger.debug("%s: empty data struct", context)
        return Segment(kind=seg_kind, start_time=start_time, ambient_c=ambient_c)
    data = data_field.flat[0]
    if seg_kind == "impedance":
        return Segment(
            kind="impedance",
            start_time=start_time,
            ambient_c=ambient_c,
            re_ohm=_abs_real(_field(data, "Re", context)),
            rct_ohm=_abs_real(_field(data, "Rct", context)),
        )
    time_s, voltage_v, current_a, temperature_c = _timeseries(data, context)
    capacity_ah = None
    if seg_kind == "discharge" and "Capacity" in (data.dtype.names or ()):
        capacity_ah = _first_float(data["Capacity"])
    return Segment(
        kind=seg_kind,
        time_s=time_s,
        voltage_v=voltage_v,
        current_a=current_a,
        temperature_c=temperature_c,
        start_time=start_time,
        ambient_c=ambient_c,
        capacity_ah=capacity_ah,
    )


def _select_variable(mat: dict[str, Any], preferred: str, path: Path) -> str:
    keys = [k for k in mat if not k.startswith("__")]
    for key in keys:
        if key.upper() == preferred.upper():
            return key
    if len(keys) == 1:
        logger.debug("%s: using variable %r (requested %r)", path, keys[0], preferred)
        return keys[0]
    raise MatFormatError(f"{path}: no variable named {preferred!r} (variables: {keys})")


def load_battery(path: Path, battery_id: str | None = None) -> BatteryRecord:
    """Parse one NASA ``.mat`` file into a :class:`BatteryRecord`.

    Args:
        path: Path to ``B00XX.mat``.
        battery_id: Identifier of the cell; defaults to the file name. If the file holds a
            single variable with a different name, that variable is used.

    Returns:
        The record with one segment per operation, in recorded order.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        MatFormatError: If the file cannot be read or lacks the expected structure.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"battery file not found: {path}")
    try:
        mat: dict[str, Any] = scipy.io.loadmat(str(path))
    except NotImplementedError as exc:  # MATLAB v7.3 (HDF5) files
        raise MatFormatError(f"{path}: unsupported MATLAB file version ({exc})") from exc
    except (MatReadError, ValueError, OSError, TypeError) as exc:
        raise MatFormatError(f"{path}: cannot read MATLAB file ({exc})") from exc

    variable = _select_variable(mat, battery_id or path.stem, path)
    if battery_id is not None:
        record_id = normalize_battery_id(battery_id)
    elif _FILE_STEM_RE.match(variable):
        record_id = variable.upper()
    else:
        record_id = path.stem.upper()

    top = np.asarray(mat[variable])
    if top.dtype.names is None or top.size == 0:
        raise MatFormatError(f"{path}: variable {variable!r} is not a MATLAB struct")
    cycles = np.asarray(_field(top.flat[0], "cycle", f"{record_id}")).ravel()
    if cycles.dtype.names is None:
        raise MatFormatError(f"{path}: {record_id}.cycle is not a struct array")

    segments: list[Segment] = []
    for index, op in enumerate(cycles):
        segment = _parse_operation(op, f"{record_id} operation {index}")
        if segment is not None:
            segments.append(segment)
    counts = {kind: sum(s.kind == kind for s in segments) for kind in sorted(_SEGMENT_KINDS)}
    logger.debug("Loaded %s from %s: %s", record_id, path, counts)
    return BatteryRecord(battery_id=record_id, segments=tuple(segments), source_path=path)


def iter_batteries(root: Path, battery_ids: Iterable[str] | None = None) -> Iterator[BatteryRecord]:
    """Load batteries one at a time (memory stays bounded by the largest file).

    Args:
        root: Directory searched recursively with :func:`find_mat_files`.
        battery_ids: Cells to load, in this order. ``None`` loads every cell found, sorted
            by id. Identifiers are normalised (``"b5"`` -> ``"B0005"``).

    Returns:
        An iterator of records.

    Raises:
        FileNotFoundError: If ``root`` is missing or a requested battery has no file. This is
            checked eagerly, before the first record is loaded.
        ValueError: If a requested identifier is malformed.
    """
    files = find_mat_files(root)
    if battery_ids is None:
        wanted = list(files)
    else:
        wanted = list(dict.fromkeys(normalize_battery_id(b) for b in battery_ids))
        missing = [b for b in wanted if b not in files]
        if missing:
            raise FileNotFoundError(
                f"no .mat file for {missing} under {root} (found {len(files)} batteries)"
            )

    def _generate() -> Iterator[BatteryRecord]:
        for battery_id in wanted:
            yield load_battery(files[battery_id], battery_id)

    return _generate()
