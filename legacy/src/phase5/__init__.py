"""Phase 5 - research rigor additions.

Modules in this package close the evaluation / calibration / attribution gaps
that an external examiner would reasonably flag on top of the Phase 1-4
pipeline. Each module is self-contained and usable as a library import or
a ``python -m`` CLI entry point.

    leakage             Cross-dataset source-classifier probe
    calibration         ECE, MCE, reliability diagrams, sharpness
    weighted_conformal  Covariate-shift-aware conformal intervals
    survival_synth      Right-censored synthetic holdout for the Weibull head
    causal_shap         Causal (Shapley + do-intervention) feature attribution
    counterfactuals     DiCE-style minimal-change counterfactuals for the API
    failure_gallery     Worst-N test-set cases with SHAP attribution
    carbon              codecarbon instrumentation for training runs
"""

from __future__ import annotations

__all__ = [
    "leakage",
    "calibration",
    "weighted_conformal",
    "survival_synth",
    "causal_shap",
    "counterfactuals",
    "failure_gallery",
    "carbon",
]
