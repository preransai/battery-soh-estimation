"""Exceptions shared by the evaluation modules.

:class:`LeakageError` lives in its own module so that :mod:`.splits` and
:mod:`.leakage` can both raise it without importing each other. It is
re-exported from :mod:`battery_soh.evaluation.leakage`, which is where callers
are expected to import it from.
"""

from __future__ import annotations

from battery_soh.errors import BatterySohError

__all__ = ["LeakageError"]


class LeakageError(BatterySohError):
    """Information from the evaluation side reached the training side.

    Raised when train and test rows overlap, when a feature is label-side by
    construction (per the feature registry), or when a feature set rebuilds
    the label almost exactly. It is deliberately *not* a ``ValueError``: it
    signals a broken experiment, not a bad argument, and should never be
    caught and ignored.
    """
