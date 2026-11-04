"""Type aliases shared across the package (NumPy only, no other dependency)."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

__all__ = ["FloatArray"]

#: A NumPy ``float64`` array of any shape.
FloatArray: TypeAlias = NDArray[np.float64]
