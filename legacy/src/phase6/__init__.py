"""Phase 6 - deferred items.

Modules in this package implement the Tier-1 / Tier-2 items that were
listed as deferred when Phase 5 shipped. Each module is self-contained
and writes a JSON sidecar under ``results/phase6/``.

    rap_benchmark          #3   real-LLM vs template retrieval-augmented diagnosis
    llm_feature_discovery  #14  LLM proposes new features, we validate with SHAP
    pinn_sweep             #2   small Bayesian-ish sweep over the Phase-4 PINN weights
    lobo_cv                #4   leave-one-battery-out with multi-seed error bars
    tent_adapt             #10  test-time normalisation adaptation under shift
    curriculum             #12  order-of-training comparison (random vs SOH-sorted)
    ssl_contrastive        #13  SimCLR-style pretraining + linear probe on SOH

Tier-3 items (foundation model, neural ODE, multi-task EIS, federated
learning) are explicitly deferred as out-of-scope; see
``PHASE6_DEFERRED_REPORT.md`` for the rationale and what each would
need.
"""

from __future__ import annotations

__all__ = [
    "rap_benchmark",
    "llm_feature_discovery",
    "pinn_sweep",
    "lobo_cv",
    "tent_adapt",
    "curriculum",
    "ssl_contrastive",
]
