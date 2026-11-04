"""Leakage-free Li-ion battery state-of-health (SOH) estimation.

The package turns raw NASA PCoE cycling data into partial-window health
indicators, benchmarks models with leave-one-battery-out evaluation, adds
battery-level conformal uncertainty, and serves the result through a FastAPI
service. See ``docs/METHODOLOGY.md`` for the task definition and
``docs/LEAKAGE_AUDIT.md`` for why capacity is never a model input.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("battery-soh")
except PackageNotFoundError:  # running from a source checkout without install
    __version__ = "0.0.0+local"

__all__ = ["__version__"]
