"""Raw-data access for the NASA PCoE battery dataset.

Submodules (imported explicitly, so ``import battery_soh.data`` stays cheap):

* :mod:`battery_soh.data.schema` - typed containers (``Segment``, ``BatteryRecord``).
* :mod:`battery_soh.data.nasa` - ``.mat`` discovery and parsing.
* :mod:`battery_soh.data.protocols` - per-battery test protocol registry and usability evidence.
* :mod:`battery_soh.data.download` - verified download from the official NASA source.
"""
