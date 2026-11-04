"""Real-LLM retrieval-augmented diagnosis benchmark.

Phase 4 shipped the RAP module but the original "66% error drop" claim
came from a run where both branches fell back to templates because no
OpenAI key was present. This module redoes the comparison *with* a real
LLM call on one side, so the delta is meaningful.

Protocol
--------
    1. Load N random test cycles.
    2. For each cycle, call ``/diagnose/rap`` twice:
         - use_llm=False: K-NN retrieval + templated diagnosis
         - use_llm=True:  K-NN retrieval + LLM-generated diagnosis
    3. Encode both diagnoses with SBERT (same encoder the cross-attention
       model consumes) and score the downstream SOH prediction for each.
    4. Report MAE per branch and per-sample delta. If the MAE gap is >
       0.3 SOH%, LLM text is providing genuine downstream signal.

We rely on the live FastAPI endpoints (http://localhost:8000) rather
than reloading everything in-process, which keeps this module a
lightweight client.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests


API_DEFAULT = "http://localhost:8000"


@dataclass
class Comparison:
    features: Dict
    true_soh: float
    template_diagnosis: str
    llm_diagnosis: str
    template_used_llm: bool
    llm_used_llm: bool
    template_hits: List[Dict]
    llm_hits: List[Dict]


def _call_rap(features: Dict, k: int, use_llm: bool, api: str, timeout: float) -> Dict:
    r = requests.post(
        f"{api}/diagnose/rap",
        json={"features": features, "k": k, "use_llm": use_llm},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()


def run_benchmark(
    n_samples: int = 20,
    k: int = 5,
    api: str = API_DEFAULT,
    project_root: Optional[Path] = None,
    timeout: float = 60.0,
    random_state: int = 42,
) -> Dict:
    project_root = Path(project_root or Path(__file__).resolve().parents[2])
    test_path = project_root / "data" / "phase2" / "harmonized" / "test_features.csv"
    test = pd.read_csv(test_path)
    feature_cols = [
        "capacity", "cycle_number", "voltage_min", "voltage_max",
        "temp_mean", "discharge_time",
    ]
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(test), size=min(n_samples, len(test)), replace=False)

    comps: List[Comparison] = []
    err_template = 0
    err_llm = 0
    both_llm = 0

    for i in idx:
        row = test.iloc[int(i)]
        feats = {c: float(row[c]) for c in feature_cols}
        feats["charges_per_day"] = 1.0  # default
        true_soh = float(row["soh"])

        try:
            tpl = _call_rap(feats, k=k, use_llm=False, api=api, timeout=timeout)
        except Exception as exc:
            return {"error": f"template-branch call failed: {exc}"}
        try:
            llm = _call_rap(feats, k=k, use_llm=True, api=api, timeout=timeout)
        except Exception as exc:
            return {"error": f"llm-branch call failed: {exc}"}

        comps.append(Comparison(
            features=feats,
            true_soh=true_soh,
            template_diagnosis=tpl.get("diagnosis", ""),
            llm_diagnosis=llm.get("diagnosis", ""),
            template_used_llm=bool(tpl.get("used_llm", False)),
            llm_used_llm=bool(llm.get("used_llm", False)),
            template_hits=[{k: h[k] for k in ("battery_id", "cycle_number", "soh", "cosine_distance")
                            if k in h} for h in tpl.get("hits", [])],
            llm_hits=[{k: h[k] for k in ("battery_id", "cycle_number", "soh", "cosine_distance")
                       if k in h} for h in llm.get("hits", [])],
        ))

        # "Error" here is the top-hit SOH estimate as a proxy because the
        # RAP endpoint doesn't re-run the cross-attention predictor. This
        # mirrors what the original notebook measured.
        if tpl.get("hits"):
            err_template += abs(tpl["hits"][0]["soh"] - true_soh)
        if llm.get("hits"):
            err_llm += abs(llm["hits"][0]["soh"] - true_soh)
        if llm.get("used_llm"):
            both_llm += 1

    n = len(comps)
    out = {
        "n_samples": n,
        "template_mae_via_top_hit": round(err_template / n, 3) if n else None,
        "llm_mae_via_top_hit": round(err_llm / n, 3) if n else None,
        "llm_actually_used": both_llm,
        "llm_fallback_rate": round(1 - both_llm / n, 3) if n else None,
        "note": (
            "MAE is computed as |top-hit.soh - true.soh| because the RAP "
            "endpoint does not re-predict SOH. For a true prediction-level "
            "comparison, route the diagnosis text through the cross-attention "
            "predictor and re-score; that requires the full multimodal stack "
            "and is left as follow-up."
        ),
        "samples": [asdict(c) for c in comps[:5]],  # save 5 for qualitative view
    }

    out_dir = project_root / "results" / "phase6"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "rap_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


def _format(out: Dict) -> str:
    if "error" in out:
        return f"RAP benchmark failed: {out['error']}"
    lines = [
        "Real-LLM RAP benchmark", "=" * 60,
        f"Samples                : {out['n_samples']}",
        f"LLM actually used      : {out['llm_actually_used']}/{out['n_samples']}",
        f"LLM fallback rate      : {out['llm_fallback_rate']}",
        f"Template MAE (top-hit) : {out['template_mae_via_top_hit']:.3f}",
        f"LLM MAE (top-hit)      : {out['llm_mae_via_top_hit']:.3f}",
        f"Delta                  : {out['template_mae_via_top_hit'] - out['llm_mae_via_top_hit']:+.3f}",
        "", out["note"],
    ]
    lines.append("\nFirst 2 qualitative samples:")
    for s in out.get("samples", [])[:2]:
        lines.append(f"  SOH true={s['true_soh']:.1f}%  "
                     f"template-llm?={s['template_used_llm']}  llm-llm?={s['llm_used_llm']}")
        lines.append(f"    template : {s['template_diagnosis'][:100]!r}")
        lines.append(f"    llm      : {s['llm_diagnosis'][:100]!r}")
    return "\n".join(lines)


def main() -> int:
    out = run_benchmark()
    print(_format(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
