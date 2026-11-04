"""Offline tests for the verified NASA dataset download.

A fake archive with the real nesting (outer zip -> one zip per campaign -> ``.mat`` files,
B0025-B0028 duplicated in two campaigns) is served through ``file://`` URLs or a patched
``urlopen``; nothing here touches the network.
"""

from __future__ import annotations

import hashlib
import io
import logging
import stat
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable
from email.message import Message
from pathlib import Path
from typing import Any

import pytest

from battery_soh.data import download as dl
from battery_soh.data.download import (
    KNOWN_SHA256,
    NASA_URL,
    ChecksumMismatchError,
    DownloadError,
    download_nasa,
    extract_zip_recursive,
)
from battery_soh.data.nasa import NASA_BATTERY_IDS, find_mat_files

_CAMPAIGNS: dict[str, tuple[str, ...]] = {
    "1. BatteryAgingARC-FY08Q4": ("B0005", "B0006", "B0007", "B0018"),
    "2. BatteryAgingARC_25_26_27_28_P1": ("B0025", "B0026", "B0027", "B0028"),
    "3. BatteryAgingARC_25-44": tuple(
        b for b in NASA_BATTERY_IDS if "B0025" <= b <= "B0044"
    ),  # includes the duplicates of B0025-B0028
    "4. BatteryAgingARC_45_46_47_48": ("B0045", "B0046", "B0047", "B0048"),
    "5. BatteryAgingARC_49_50_51_52": ("B0049", "B0050", "B0051", "B0052"),
    "6. BatteryAgingARC_53_54_55_56": ("B0053", "B0054", "B0055", "B0056"),
}


def _mat_bytes(battery_id: str) -> bytes:
    return f"MATLAB 5.0 MAT-file stub for {battery_id}".encode()


def _campaign_zip(name: str, battery_ids: tuple[str, ...], *, wrap: bool) -> bytes:
    buffer = io.BytesIO()
    prefix = f"{name}/" if wrap else ""
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for battery_id in battery_ids:
            zf.writestr(f"{prefix}{battery_id}.mat", _mat_bytes(battery_id))
        zf.writestr(f"{prefix}README.txt", f"README for {name}")
        zf.writestr(f"__MACOSX/{prefix}._README.txt", b"resource fork")
    return buffer.getvalue()


def build_fake_archive(
    path: Path, campaigns: dict[str, tuple[str, ...]] = _CAMPAIGNS
) -> tuple[Path, str]:
    """Write a fake NASA archive and return ``(path, sha256)``."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as outer:
        for k, (name, ids) in enumerate(campaigns.items()):
            # Odd campaigns repeat their folder name inside the nested zip (both layouts exist).
            payload = _campaign_zip(name, ids, wrap=k % 2 == 1)
            outer.writestr(f"5. Battery Data Set/{name}.zip", payload)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def archive(tmp_path: Path) -> tuple[Path, str]:
    return build_fake_archive(tmp_path / "src" / "battery.zip")


@pytest.fixture(autouse=True)
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    sleeps: list[float] = []
    monkeypatch.setattr(dl.time, "sleep", sleeps.append)
    return sleeps


@pytest.fixture(autouse=True)
def _create_src_dir(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir(exist_ok=True)


# --------------------------------------------------------------------------------- constants


def test_constants() -> None:
    assert NASA_URL.startswith("https://phm-datasets.s3.amazonaws.com/NASA/")
    assert NASA_URL.endswith("Battery+Data+Set.zip")
    assert KNOWN_SHA256 is not None
    assert len(KNOWN_SHA256) == 64
    assert int(KNOWN_SHA256, 16) >= 0
    assert KNOWN_SHA256.lower() == KNOWN_SHA256


# ---------------------------------------------------------------------------- happy path


def test_download_installs_all_batteries(tmp_path: Path, archive: tuple[Path, str]) -> None:
    path, sha = archive
    dest = tmp_path / "nasa"
    out = download_nasa(dest, url=path.as_uri(), expected_sha256=sha)
    assert out == dest
    found = find_mat_files(dest)
    assert tuple(found) == NASA_BATTERY_IDS
    for name, ids in _CAMPAIGNS.items():
        for battery_id in ids:
            mat = dest / name / f"{battery_id}.mat"
            assert mat.read_bytes() == _mat_bytes(battery_id)
        assert (dest / name / "README.txt").is_file()
    # No archives, staging folders or macOS metadata are left behind.
    assert not list(dest.rglob("*.zip"))
    assert not [p for p in dest.rglob("*") if p.name.startswith(".")]
    assert not list(dest.rglob("__MACOSX"))


def test_download_checksum_is_case_insensitive(tmp_path: Path, archive: tuple[Path, str]) -> None:
    path, sha = archive
    download_nasa(tmp_path / "nasa", url=path.as_uri(), expected_sha256=sha.upper())
    assert len(find_mat_files(tmp_path / "nasa")) == 34


def test_download_without_expected_checksum_logs_digest(
    tmp_path: Path, archive: tuple[Path, str], caplog: pytest.LogCaptureFixture
) -> None:
    path, sha = archive
    with caplog.at_level(logging.INFO, logger="battery_soh.data.download"):
        download_nasa(tmp_path / "nasa", url=path.as_uri(), expected_sha256=None)
    assert any(sha in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)
    assert any("Downloaded 100%" in r.getMessage() for r in caplog.records)  # progress log


def test_download_keeps_unrelated_files(tmp_path: Path, archive: tuple[Path, str]) -> None:
    path, sha = archive
    dest = tmp_path / "nasa"
    dest.mkdir()
    (dest / "my_notes.txt").write_text("keep me")
    download_nasa(dest, url=path.as_uri(), expected_sha256=sha)
    assert (dest / "my_notes.txt").read_text() == "keep me"


# ----------------------------------------------------------------------- checksum mismatch


def test_checksum_mismatch_raises_and_installs_nothing(
    tmp_path: Path, archive: tuple[Path, str]
) -> None:
    path, _ = archive
    dest = tmp_path / "nasa"
    with pytest.raises(ChecksumMismatchError, match="sha256 mismatch"):
        download_nasa(dest, url=path.as_uri(), expected_sha256="0" * 64)
    assert list(dest.iterdir()) == []  # temp files cleaned up, nothing installed


def test_default_checksum_rejects_a_different_archive(
    tmp_path: Path, archive: tuple[Path, str]
) -> None:
    path, _ = archive
    with pytest.raises(ChecksumMismatchError):
        download_nasa(tmp_path / "nasa", url=path.as_uri())  # KNOWN_SHA256 of the real archive


# -------------------------------------------------------------------------------- idempotency


def _forbid_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_: Any, **__: Any) -> Any:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(urllib.request, "urlopen", fail)


def test_second_call_does_not_download(
    tmp_path: Path, archive: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    path, sha = archive
    dest = tmp_path / "nasa"
    download_nasa(dest, url=path.as_uri(), expected_sha256=sha)
    before = {p: p.stat().st_mtime_ns for p in dest.rglob("*")}
    _forbid_network(monkeypatch)
    assert download_nasa(dest, url=path.as_uri(), expected_sha256=sha) == dest
    assert download_nasa(dest) == dest  # default URL: still no network
    assert {p: p.stat().st_mtime_ns for p in dest.rglob("*")} == before


def test_force_downloads_again(
    tmp_path: Path, archive: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    path, sha = archive
    dest = tmp_path / "nasa"
    download_nasa(dest, url=path.as_uri(), expected_sha256=sha)
    (dest / "1. BatteryAgingARC-FY08Q4" / "B0005.mat").write_bytes(b"corrupted")
    calls: list[str] = []
    real_urlopen = urllib.request.urlopen

    def counting(request: urllib.request.Request, **kwargs: Any) -> Any:
        calls.append(request.full_url)
        return real_urlopen(request, **kwargs)

    monkeypatch.setattr(urllib.request, "urlopen", counting)
    download_nasa(dest, url=path.as_uri(), expected_sha256=sha, force=True)
    assert calls == [path.as_uri()]
    restored = dest / "1. BatteryAgingARC-FY08Q4" / "B0005.mat"
    assert restored.read_bytes() == _mat_bytes("B0005")


def test_partial_install_triggers_download(tmp_path: Path, archive: tuple[Path, str]) -> None:
    path, sha = archive
    dest = tmp_path / "nasa"
    (dest / "1. BatteryAgingARC-FY08Q4").mkdir(parents=True)
    (dest / "1. BatteryAgingARC-FY08Q4" / "B0005.mat").write_bytes(b"x")
    download_nasa(dest, url=path.as_uri(), expected_sha256=sha)
    assert len(find_mat_files(dest)) == 34


# --------------------------------------------------------------------------- argument checks


@pytest.mark.parametrize(
    "url", ["http://example.com/data.zip", "ftp://example.com/data.zip", "data.zip"]
)
def test_rejects_insecure_or_unknown_schemes(tmp_path: Path, url: str) -> None:
    with pytest.raises(ValueError, match="unsupported URL scheme"):
        download_nasa(tmp_path / "nasa", url=url)


@pytest.mark.parametrize(("timeout", "retries"), [(0.0, 1), (-1.0, 1), (10.0, -1)])
def test_rejects_bad_timeout_or_retries(tmp_path: Path, timeout: float, retries: int) -> None:
    with pytest.raises(ValueError, match="retries"):
        download_nasa(tmp_path / "nasa", url=NASA_URL, timeout=timeout, retries=retries)


# --------------------------------------------------------------------------- network failures


class _FakeResponse(io.BytesIO):
    def __init__(self, payload: bytes, *, declared_length: int | None = None) -> None:
        super().__init__(payload)
        self.headers = Message()
        if declared_length is not None:
            self.headers["Content-Length"] = str(declared_length)


def _scripted_urlopen(
    monkeypatch: pytest.MonkeyPatch, script: list[Callable[[], Any]]
) -> list[int]:
    attempts: list[int] = []

    def fake(request: urllib.request.Request, *, timeout: float) -> Any:
        attempts.append(len(attempts) + 1)
        assert timeout > 0
        assert request.get_header("User-agent", "").startswith("battery-soh/")
        return script[len(attempts) - 1]()

    monkeypatch.setattr(urllib.request, "urlopen", fake)
    return attempts


def _raise(exc: BaseException) -> Callable[[], Any]:
    def thrower() -> Any:
        raise exc

    return thrower


def test_transient_errors_are_retried_with_backoff(
    tmp_path: Path,
    archive: tuple[Path, str],
    monkeypatch: pytest.MonkeyPatch,
    sleeps: list[float],
) -> None:
    path, sha = archive
    payload = path.read_bytes()
    attempts = _scripted_urlopen(
        monkeypatch,
        [
            _raise(urllib.error.URLError("connection reset")),
            lambda: _FakeResponse(payload[:100], declared_length=len(payload)),  # truncated
            lambda: _FakeResponse(payload, declared_length=len(payload)),
        ],
    )
    download_nasa(tmp_path / "nasa", url=NASA_URL, expected_sha256=sha, retries=2)
    assert attempts == [1, 2, 3]
    assert sleeps == [1.0, 2.0]
    assert len(find_mat_files(tmp_path / "nasa")) == 34


def test_gives_up_after_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    attempts = _scripted_urlopen(monkeypatch, [_raise(TimeoutError("timed out")) for _ in range(3)])
    with pytest.raises(DownloadError, match="cannot download"):
        download_nasa(tmp_path / "nasa", url=NASA_URL, retries=2)
    assert attempts == [1, 2, 3]
    assert sleeps == [1.0, 2.0]


def test_client_errors_are_not_retried(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    error = urllib.error.HTTPError(NASA_URL, 404, "Not Found", Message(), None)
    attempts = _scripted_urlopen(monkeypatch, [_raise(error)])
    with pytest.raises(DownloadError, match="HTTP 404"):
        download_nasa(tmp_path / "nasa", url=NASA_URL, retries=3)
    assert attempts == [1]


def test_server_errors_are_retried(
    tmp_path: Path, archive: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    path, sha = archive
    error = urllib.error.HTTPError(NASA_URL, 503, "Unavailable", Message(), None)
    attempts = _scripted_urlopen(
        monkeypatch, [_raise(error), lambda: _FakeResponse(path.read_bytes())]
    )
    download_nasa(tmp_path / "nasa", url=NASA_URL, expected_sha256=sha, retries=1)
    assert attempts == [1, 2]


def test_incomplete_download_without_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _scripted_urlopen(monkeypatch, [lambda: _FakeResponse(b"abc", declared_length=10)])
    with pytest.raises(DownloadError, match="incomplete download: got 3 of 10 bytes"):
        download_nasa(tmp_path / "nasa", url=NASA_URL, retries=0)


def test_missing_local_file_raises_download_error(tmp_path: Path) -> None:
    url = (tmp_path / "does_not_exist.zip").as_uri()
    with pytest.raises(DownloadError):
        download_nasa(tmp_path / "nasa", url=url, expected_sha256=None, retries=0)


# --------------------------------------------------------------------------------- extraction


def test_archive_without_all_batteries_is_rejected(tmp_path: Path) -> None:
    campaigns = {"1. BatteryAgingARC-FY08Q4": ("B0005", "B0006")}
    path, sha = build_fake_archive(tmp_path / "src" / "partial.zip", campaigns)
    with pytest.raises(DownloadError, match="batteries are missing"):
        download_nasa(tmp_path / "nasa", url=path.as_uri(), expected_sha256=sha)


def test_corrupt_archive_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "src" / "corrupt.zip"
    path.write_bytes(b"PK\x03\x04 definitely not a zip file")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(DownloadError, match="corrupt zip archive"):
        download_nasa(tmp_path / "nasa", url=path.as_uri(), expected_sha256=sha)


def _zip_with(path: Path, members: list[tuple[zipfile.ZipInfo | str, bytes]]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for info, data in members:
            zf.writestr(info, data)
    return path


@pytest.mark.parametrize(
    "member",
    [
        "../evil.txt",
        "a/../../evil.txt",
        "/abs/evil.txt",
        "C:/evil.txt",
        "a\\..\\..\\e",
        "dir/file.txt:stream",  # NTFS alternate data stream
    ],
)
def test_path_traversal_is_rejected(tmp_path: Path, member: str) -> None:
    archive = _zip_with(tmp_path / "evil.zip", [(member, b"x")])
    with pytest.raises(DownloadError, match=r"unsafe path|escapes target"):
        extract_zip_recursive(archive, tmp_path / "out")
    assert not (tmp_path / "evil.txt").exists()


def test_symlink_member_is_rejected(tmp_path: Path) -> None:
    info = zipfile.ZipInfo("link")
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    archive = _zip_with(tmp_path / "link.zip", [(info, b"/etc/passwd")])
    with pytest.raises(DownloadError, match="symbolic link"):
        extract_zip_recursive(archive, tmp_path / "out")


def test_nested_traversal_is_rejected(tmp_path: Path) -> None:
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as zf:
        zf.writestr("../../escape.txt", b"x")
    archive = _zip_with(tmp_path / "outer.zip", [("campaign.zip", inner.getvalue())])
    with pytest.raises(DownloadError, match="unsafe path"):
        extract_zip_recursive(archive, tmp_path / "out")
    assert not list(tmp_path.rglob("escape.txt"))


def test_zip_bomb_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dl, "_MAX_UNCOMPRESSED_BYTES", 1000)
    archive = _zip_with(tmp_path / "big.zip", [("a.bin", b"\0" * 600), ("b.bin", b"\0" * 600)])
    with pytest.raises(DownloadError, match="size limit"):
        extract_zip_recursive(archive, tmp_path / "out")


def test_nesting_depth_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dl, "_MAX_NESTING_DEPTH", 1)
    payload = b"leaf"
    for level in range(3):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            zf.writestr(f"level{level}.zip" if level else "leaf.txt", payload)
        payload = buffer.getvalue()
    archive = tmp_path / "deep.zip"
    archive.write_bytes(payload)
    with pytest.raises(DownloadError, match="nesting deeper"):
        extract_zip_recursive(archive, tmp_path / "out")


def test_extract_expands_nested_zips_into_folders(tmp_path: Path) -> None:
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as zf:
        zf.writestr("campaign/B0005.mat", b"m")  # redundant top-level folder is collapsed
    archive = _zip_with(tmp_path / "outer.zip", [("top/campaign.zip", inner.getvalue())])
    out = extract_zip_recursive(archive, tmp_path / "out")
    assert (out / "top" / "campaign" / "B0005.mat").read_bytes() == b"m"
    assert not list(out.rglob("*.zip"))
