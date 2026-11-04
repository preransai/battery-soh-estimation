"""Every exception the package defines shares one base and keeps its built-in base."""

from __future__ import annotations

import importlib
import inspect
import pkgutil

import pytest

import battery_soh
from battery_soh.api.demo import DemoDataError
from battery_soh.api.errors import ProblemError
from battery_soh.data.download import ChecksumMismatchError, DownloadError
from battery_soh.data.nasa import MatFormatError
from battery_soh.errors import BatterySohError
from battery_soh.evaluation.leakage import LeakageError
from battery_soh.models.linear_bundle import BundleError, ManifestError
from battery_soh.training import TrainingError


@pytest.mark.parametrize(
    ("exc", "builtin"),
    [
        (DemoDataError, ValueError),
        (BundleError, ValueError),
        (MatFormatError, ValueError),
        (DownloadError, RuntimeError),
        (ChecksumMismatchError, RuntimeError),
        (TrainingError, RuntimeError),
        (ManifestError, Exception),
        (LeakageError, Exception),
        (ProblemError, Exception),
    ],
)
def test_exceptions_keep_their_builtin_base(exc: type[Exception], builtin: type[Exception]) -> None:
    assert issubclass(exc, BatterySohError)
    assert issubclass(exc, builtin)
    if builtin is not Exception:  # the built-in base stays the nearest one in the MRO
        mro = inspect.getmro(exc)
        assert mro.index(builtin) < mro.index(BatterySohError)


def test_leakage_error_is_not_a_value_error() -> None:
    # A broken experiment must not be swallowed by an `except ValueError` for bad arguments.
    assert not issubclass(LeakageError, ValueError)


def test_every_exception_defined_in_the_package_derives_from_the_base() -> None:
    defined = []
    for info in pkgutil.walk_packages(battery_soh.__path__, prefix="battery_soh."):
        module = importlib.import_module(info.name)
        for obj in vars(module).values():
            if (
                inspect.isclass(obj)
                and issubclass(obj, BaseException)
                and obj.__module__ == module.__name__
            ):
                defined.append(obj)
    assert BatterySohError in defined
    assert len(defined) >= 10
    assert [e.__qualname__ for e in defined if not issubclass(e, BatterySohError)] == []
