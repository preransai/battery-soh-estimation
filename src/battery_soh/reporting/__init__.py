"""Report rendering: figures, ``docs/RESULTS.md``, README result blocks and model cards.

``soh report`` (:func:`battery_soh.reporting.render.render_all`) renders the
figures and documents from the benchmark/audit JSON and the model bundles;
``soh train`` uses :func:`battery_soh.reporting.model_card.render_model_card`.
Importing this package does not import matplotlib (only ``figures``/``style`` do).
"""
