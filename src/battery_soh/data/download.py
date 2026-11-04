"""Download the NASA PCoE "Battery Data Set" from its official source.

The archive published by the NASA Prognostics Center of Excellence is a zip file that contains
one nested zip per test campaign. :func:`download_nasa` streams it to a temporary file while
hashing it, verifies the SHA-256 against :data:`KNOWN_SHA256`, extracts the nested archives
safely (no path traversal, no symlinks, bounded size) and installs the campaign folders into
``dest``. Re-running it is cheap: when ``dest`` already holds all 34 cells nothing is
downloaded.

This is the only module in the package that touches the network, and it uses the standard
library only.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import stat
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from http.client import HTTPException
from pathlib import Path, PurePosixPath
from typing import Final

from battery_soh import __version__
from battery_soh.data.nasa import NASA_BATTERY_IDS, find_mat_files
from battery_soh.errors import BatterySohError

logger = logging.getLogger(__name__)

__all__ = [
    "KNOWN_SHA256",
    "NASA_URL",
    "ChecksumMismatchError",
    "DownloadError",
    "download_nasa",
    "extract_zip_recursive",
]

#: Official location of the NASA PCoE "Battery Data Set" (Saha & Goebel, 2007).
NASA_URL: Final = "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip"

#: SHA-256 of the archive at :data:`NASA_URL`, recorded after a verified download.
KNOWN_SHA256: Final[str | None] = "82302a7db4fc1b34e0b6676326610438d43b816bdf11a69d1d012a464ef2f92e"

_ALLOWED_SCHEMES: Final = frozenset({"https", "file"})
_CHUNK_BYTES: Final = 1 << 20
_MAX_UNCOMPRESSED_BYTES: Final = 4 << 30  # the real archive expands to ~0.4 GB
_MAX_NESTING_DEPTH: Final = 4


class DownloadError(RuntimeError, BatterySohError):
    """Raised when the dataset cannot be downloaded or installed."""


class ChecksumMismatchError(DownloadError):
    """Raised when the downloaded archive does not match the expected SHA-256."""


def _has_all_batteries(directory: Path) -> bool:
    if not directory.is_dir():
        return False
    return set(NASA_BATTERY_IDS) <= set(find_mat_files(directory))


def _validate_url(url: str) -> None:
    scheme = urllib.parse.urlsplit(url).scheme.lower()
    if scheme not in _ALLOWED_SCHEMES:
        raise ValueError(f"unsupported URL scheme {scheme!r} in {url!r}; use https")


def _stream_once(url: str, target: Path, *, timeout: float) -> str:
    request = urllib.request.Request(  # noqa: S310 - scheme validated by _validate_url
        url, headers={"User-Agent": f"battery-soh/{__version__} (+urllib)"}
    )
    digest = hashlib.sha256()
    received = 0
    started = time.monotonic()
    with (
        urllib.request.urlopen(request, timeout=timeout) as response,  # noqa: S310
        target.open("wb") as sink,
    ):
        length_header = response.headers.get("Content-Length")
        total = int(length_header) if length_header and length_header.isdigit() else None
        next_report = 0.1
        last_report_bytes = 0
        while chunk := response.read(_CHUNK_BYTES):
            sink.write(chunk)
            digest.update(chunk)
            received += len(chunk)
            if total:
                if received / total >= next_report:
                    elapsed = max(time.monotonic() - started, 1e-6)
                    logger.info(
                        "Downloaded %3.0f%% (%.1f / %.1f MB, %.1f MB/s)",
                        100.0 * received / total,
                        received / 1e6,
                        total / 1e6,
                        received / 1e6 / elapsed,
                    )
                    next_report = (int(10 * received / total) + 1) / 10
            elif received - last_report_bytes >= 25 * _CHUNK_BYTES:
                logger.info("Downloaded %.1f MB", received / 1e6)
                last_report_bytes = received
    if total is not None and received != total:
        raise DownloadError(f"incomplete download: got {received} of {total} bytes")
    logger.info("Download finished: %.1f MB in %.1f s", received / 1e6, time.monotonic() - started)
    return digest.hexdigest()


def _fetch(url: str, target: Path, *, timeout: float, retries: int) -> str:
    """Stream ``url`` to ``target`` and return its SHA-256, retrying transient failures."""
    attempt = 0
    while True:
        attempt += 1
        try:
            return _stream_once(url, target, timeout=timeout)
        except urllib.error.HTTPError as exc:
            if exc.code < 500 or attempt > retries:
                raise DownloadError(f"HTTP {exc.code} while downloading {url}") from exc
            reason: object = f"HTTP {exc.code}"
        except (urllib.error.URLError, HTTPException, TimeoutError, ConnectionError) as exc:
            if attempt > retries:
                raise DownloadError(f"cannot download {url}: {exc}") from exc
            reason = exc
        except DownloadError as exc:
            if attempt > retries:
                raise
            reason = exc
        backoff = 2.0 ** (attempt - 1)
        logger.warning(
            "Download attempt %d failed (%s); retrying in %.0f s", attempt, reason, backoff
        )
        time.sleep(backoff)


# ------------------------------------------------------------------------------- extraction


def _is_symlink(info: zipfile.ZipInfo) -> bool:
    return stat.S_ISLNK(info.external_attr >> 16)


def _safe_members(archive: zipfile.ZipFile, target: Path) -> list[zipfile.ZipInfo]:
    """Return the members to extract, rejecting anything that could escape ``target``."""
    root = target.resolve()
    members: list[zipfile.ZipInfo] = []
    total = 0
    for info in archive.infolist():
        name = PurePosixPath(info.filename.replace("\\", "/"))
        if not name.parts or name.parts[0] == "__MACOSX" or name.name.startswith("._"):
            continue
        # ":" catches drive letters and NTFS alternate data streams on Windows.
        if name.is_absolute() or ".." in name.parts or any(":" in part for part in name.parts):
            raise DownloadError(f"unsafe path in archive: {info.filename!r}")
        if _is_symlink(info):
            raise DownloadError(f"symbolic link in archive: {info.filename!r}")
        resolved = (root / Path(*name.parts)).resolve()
        if not resolved.is_relative_to(root):
            raise DownloadError(f"archive member escapes target: {info.filename!r}")
        total += info.file_size
        if total > _MAX_UNCOMPRESSED_BYTES:
            raise DownloadError("archive expands beyond the size limit (possible zip bomb)")
        members.append(info)
    return members


def _strip_single_folder(directory: Path, expected_name: str) -> Path:
    """Return ``directory/expected_name`` when it is the only entry, else ``directory``."""
    entries = list(directory.iterdir())
    if len(entries) == 1 and entries[0].is_dir() and entries[0].name == expected_name:
        return entries[0]
    return directory


def extract_zip_recursive(archive: Path, target: Path, *, _depth: int = 0) -> Path:
    """Extract ``archive`` into ``target`` and expand any zip files it contains.

    A nested ``X.zip`` is replaced by a folder ``X`` holding its contents (a redundant
    top-level ``X/`` inside the nested archive is collapsed).

    Args:
        archive: Zip file to extract.
        target: Destination directory (created if needed).

    Returns:
        ``target``.

    Raises:
        DownloadError: If the archive is corrupt, unsafe or nested too deeply.
    """
    if _depth > _MAX_NESTING_DEPTH:
        raise DownloadError(f"zip nesting deeper than {_MAX_NESTING_DEPTH} levels")
    target.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(target, members=_safe_members(zf, target))
    except zipfile.BadZipFile as exc:
        raise DownloadError(f"corrupt zip archive {archive.name}: {exc}") from exc
    for nested in sorted(target.rglob("*.zip")):
        destination = nested.with_suffix("")
        staging = Path(tempfile.mkdtemp(prefix=".unzip-", dir=nested.parent))
        try:
            extract_zip_recursive(nested, staging, _depth=_depth + 1)
            content = _strip_single_folder(staging, destination.name)
            if destination.exists():
                _merge_tree(content, destination)
            else:
                content.rename(destination)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        nested.unlink()
        logger.debug("Expanded nested archive %s", nested.name)
    return target


def _merge_tree(source: Path, destination: Path) -> None:
    """Move every file under ``source`` into ``destination``, replacing existing files."""
    destination.mkdir(parents=True, exist_ok=True)
    for entry in sorted(source.iterdir()):
        target = destination / entry.name
        if entry.is_dir():
            if target.exists() and not target.is_dir():
                target.unlink()
            if target.exists():
                _merge_tree(entry, target)
            else:
                entry.rename(target)
        else:
            if target.is_dir():
                shutil.rmtree(target)
            entry.replace(target)


def _content_root(extracted: Path) -> Path:
    """Descend through single-folder wrappers (e.g. ``5. Battery Data Set/``)."""
    current = extracted
    while True:
        entries = [p for p in current.iterdir() if not p.name.startswith(".")]
        if len(entries) == 1 and entries[0].is_dir():
            current = entries[0]
        else:
            return current


# ------------------------------------------------------------------------------- public API


def download_nasa(
    dest: Path,
    *,
    force: bool = False,
    url: str = NASA_URL,
    expected_sha256: str | None = KNOWN_SHA256,
    timeout: float = 60.0,
    retries: int = 3,
) -> Path:
    """Download, verify and extract the NASA battery dataset into ``dest``.

    Args:
        dest: Target directory. Campaign folders (``1. BatteryAgingARC-FY08Q4/`` ...) are
            placed directly inside it; unrelated files already there are left untouched.
        force: Download again even if ``dest`` already holds all 34 batteries.
        url: Archive location (``https://`` or, for offline tests, ``file://``).
        expected_sha256: Hex digest the archive must match. ``None`` skips the check and logs
            the computed digest instead.
        timeout: Socket timeout in seconds for each network operation.
        retries: Extra attempts after a transient network failure (exponential backoff).

    Returns:
        ``dest``, the directory under which the ``.mat`` files now live.

    Raises:
        ValueError: If the URL scheme or another argument is invalid.
        ChecksumMismatchError: If the archive does not match ``expected_sha256``.
        DownloadError: If the download, extraction or final validation fails.
    """
    dest = Path(dest)
    _validate_url(url)
    if retries < 0 or timeout <= 0:
        raise ValueError("retries must be >= 0 and timeout > 0")
    if not force and _has_all_batteries(dest):
        logger.info("NASA dataset already present in %s; skipping download", dest)
        return dest

    dest.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading NASA battery dataset from %s", url)
    with tempfile.TemporaryDirectory(prefix=".soh-download-", dir=dest) as tmp:
        workdir = Path(tmp)
        archive = workdir / "nasa_battery_dataset.zip"
        sha256 = _fetch(url, archive, timeout=timeout, retries=retries)
        if expected_sha256 is None:
            logger.warning("No expected checksum given; archive sha256=%s", sha256)
        elif sha256.lower() != expected_sha256.lower():
            raise ChecksumMismatchError(
                f"sha256 mismatch for {url}: expected {expected_sha256}, got {sha256}"
            )
        else:
            logger.info("Checksum verified (sha256=%s)", sha256)

        extracted = extract_zip_recursive(archive, workdir / "extracted")
        archive.unlink()
        _merge_tree(_content_root(extracted), dest)

    found = find_mat_files(dest)
    missing = sorted(set(NASA_BATTERY_IDS) - set(found))
    if missing:
        raise DownloadError(f"archive installed but batteries are missing: {missing}")
    logger.info("NASA dataset ready: %d batteries in %s", len(found), dest)
    return dest
