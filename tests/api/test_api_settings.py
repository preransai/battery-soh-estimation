"""Service settings: environment variables only, strict validation."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from pydantic import ValidationError

from battery_soh.api.settings import Settings
from battery_soh.config import get_paths


def _clear_soh_env(monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    for name in list(os.environ):
        if name.upper().startswith("SOH_") and name.upper() not in (
            "SOH_PROJECT_ROOT",
            "SOH_RAW_DIR",
            "SOH_ARTIFACTS_DIR",
            "SOH_REPORTS_DIR",
            "SOH_FRONTEND_DIR",
        ):
            monkeypatch.delenv(name)


def test_defaults_follow_project_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_soh_env(monkeypatch)
    settings = Settings()
    paths = get_paths()
    assert settings.models_dir == paths.models_dir
    assert settings.demo_dir == paths.demo_dir
    assert settings.benchmark_dir == paths.benchmark_dir
    assert settings.frontend_dir == paths.frontend
    assert settings.cors_origins == []
    assert settings.max_body_bytes == 1_000_000
    assert settings.max_samples == 20_000
    assert settings.max_batch_items == 100
    assert settings.enable_metrics is True
    assert settings.log_level == "INFO"
    assert settings.log_level_number == logging.INFO


def test_environment_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _clear_soh_env(monkeypatch)
    monkeypatch.setenv("SOH_MODELS_DIR", str(tmp_path))
    monkeypatch.setenv("SOH_MAX_BODY_BYTES", "2048")
    monkeypatch.setenv("SOH_LOG_LEVEL", "debug")
    monkeypatch.setenv("SOH_LOG_JSON", "true")
    monkeypatch.setenv("SOH_ENABLE_METRICS", "false")
    monkeypatch.setenv("SOH_MAX_BATCH_ITEMS", "1000")
    monkeypatch.setenv("SOH_SOMETHING_UNKNOWN", "ignored")
    settings = Settings()
    assert settings.models_dir == tmp_path
    assert settings.max_body_bytes == 2048
    assert settings.log_level == "DEBUG"
    assert settings.log_json is True
    assert settings.enable_metrics is False
    assert settings.max_batch_items == 1000


def test_dotenv_files_are_not_read(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _clear_soh_env(monkeypatch)
    (tmp_path / ".env").write_text(
        "SOH_MAX_BODY_BYTES=4096\nOPENAI_API_KEY=sk-not-ours\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    assert Settings().max_body_bytes == 1_000_000


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('["https://a.example", "http://localhost:5173"]', ["https://a.example", "http://localhost:5173"]),
        ("https://a.example, https://b.example/", ["https://a.example", "https://b.example"]),
        ("", []),
        ("*", ["*"]),
    ],
)  # fmt: skip
def test_cors_origins_parsing(
    monkeypatch: pytest.MonkeyPatch, raw: str, expected: list[str]
) -> None:
    _clear_soh_env(monkeypatch)
    monkeypatch.setenv("SOH_CORS_ORIGINS", raw)
    assert Settings().cors_origins == expected


@pytest.mark.parametrize(
    "raw", ["javascript:alert(1)", "https://a.example/path", '["ftp://x"]', "[not json", "null"]
)
def test_invalid_cors_origins_are_rejected(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    _clear_soh_env(monkeypatch)
    monkeypatch.setenv("SOH_CORS_ORIGINS", raw)
    with pytest.raises(ValidationError):
        Settings()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("log_level", "LOUD"),
        ("max_body_bytes", 10),
        ("max_samples", 20_001),
        ("max_samples", 1),
        ("max_batch_items", 0),
        ("max_batch_items", 1001),
    ],
)
def test_invalid_values_are_rejected(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        Settings(**{field: value})  # type: ignore[arg-type]
