"""Partial-window health indicators (HIs) and their registry.

Submodules are imported explicitly so that ``import battery_soh.features.registry``
stays cheap (no SciPy/pandas import):

* :mod:`~battery_soh.features.registry` - feature metadata and named feature sets.
* :mod:`~battery_soh.features.extract` - :func:`features_from_segments`, the single
  raw-arrays-to-features function shared by training and serving.
* :mod:`~battery_soh.features.charge`, :mod:`~battery_soh.features.discharge`,
  :mod:`~battery_soh.features.ica`, :mod:`~battery_soh.features.impedance` - the
  individual extractors (window parameters are keyword arguments).
* :mod:`~battery_soh.features.legacy` - the leaky v1 inputs, for the audit only.
* :mod:`~battery_soh.features.relative` - commissioning-referenced features.
"""
