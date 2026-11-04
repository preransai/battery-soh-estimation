"""The common base class of every exception the package defines.

Each exception raised deliberately by ``battery_soh`` derives from
:class:`BatterySohError`, so a caller can catch all of them with one clause.
Every subclass also keeps the built-in base that describes its kind of failure,
listed first so it stays the nearest one: ``ValueError`` for bad input or a
malformed file, ``RuntimeError`` for an environment that cannot produce a
result. Existing ``except ValueError`` and ``except RuntimeError`` handlers
therefore behave exactly as before.

========================  =========================================  ================
Exception                 Raised when                                Built-in base
========================  =========================================  ================
``MatFormatError``        a ``.mat`` file lacks the NASA structure   ``ValueError``
``DownloadError``         the dataset cannot be downloaded           ``RuntimeError``
``LeakageError``          evaluation data reached the training side  (none)
``BundleError``           a model bundle cannot be exported/loaded   ``ValueError``
``ManifestError``         an artifact is missing or was modified     (none)
``TrainingError``         production training cannot be trusted      ``RuntimeError``
``DemoDataError``         the demo asset is missing or malformed     ``ValueError``
``ProblemError``          an API request fails in an expected way    (none)
========================  =========================================  ================

This module imports nothing, so any module (including the scikit-learn-free
serving path) can depend on it.
"""

from __future__ import annotations

__all__ = ["BatterySohError"]


class BatterySohError(Exception):
    """Base class of every exception defined by :mod:`battery_soh`."""
