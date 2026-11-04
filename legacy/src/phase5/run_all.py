"""End-to-end runner for the Phase 5 rigor suite.

Invokes each sub-module, wraps the call in a codecarbon-tracked block,
and writes a single consolidated manifest under
``results/phase5/manifest.json``. Makes ``make rigor`` a one-shot
command.
"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List

from src.phase5 import (
    calibration,
    carbon,
    causal_shap,
    counterfactuals,
    failure_gallery,
    leakage,
    survival_synth,
    weighted_conformal,
)


STEPS: List[Dict[str, Any]] = [
    {"key": "leakage",            "runner": leakage.run_project_probe},
    {"key": "calibration",        "runner": calibration.run_project_calibration},
    {"key": "weighted_conformal", "runner": weighted_conformal.demo_nasa_to_mendeley},
    {"key": "survival_synth",     "runner": survival_synth.run_project_survival_synth},
    {"key": "causal_shap",        "runner": causal_shap.run_project_causal_shap},
    {"key": "counterfactuals",    "runner": counterfactuals.run_project_counterfactuals},
    {"key": "failure_gallery",    "runner": failure_gallery.build_failure_gallery},
]


def _default_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run_all() -> Dict[str, Any]:
    root = _default_root()
    out_dir = root / "results" / "phase5"
    out_dir.mkdir(parents=True, exist_ok=True)
    compute_dir = out_dir / "compute"

    manifest: Dict[str, Any] = {
        "started_at": time.time(),
        "steps": {},
    }

    for spec in STEPS:
        key = spec["key"]
        runner: Callable[..., Dict[str, Any]] = spec["runner"]
        t0 = time.perf_counter()
        status = "ok"
        payload: Any
        try:
            with carbon.track(f"phase5_{key}", out_dir=compute_dir) as _:
                payload = runner()
        except Exception as exc:
            status = f"error: {exc}"
            payload = {"error": str(exc), "traceback": traceback.format_exc()}
        manifest["steps"][key] = {
            "status": status,
            "duration_sec": round(time.perf_counter() - t0, 3),
            "output_file": f"results/phase5/{key}.json",
            "summary": _summary(key, payload),
        }

    manifest["ended_at"] = time.time()
    manifest["total_duration_sec"] = round(manifest["ended_at"] - manifest["started_at"], 3)

    # Compute aggregate codecarbon
    compute_summary = carbon.aggregate_reports(compute_dir)
    manifest["compute"] = {
        "n_runs": compute_summary.get("n_runs"),
        "total_duration_sec": compute_summary.get("total_duration_sec"),
        "total_emissions_kg_co2": compute_summary.get("total_emissions_kg_co2"),
        "total_energy_kwh": compute_summary.get("total_energy_kwh"),
    }

    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def _summary(key: str, payload: Any) -> Dict[str, Any]:
    """Pull a tiny per-step digest for the manifest without re-dumping full JSON."""
    if not isinstance(payload, dict):
        return {}
    if key == "leakage":
        raw = payload.get("raw", {})
        return {"raw_roc_auc": raw.get("roc_auc"), "verdict": raw.get("verdict")}
    if key == "calibration":
        return {
            "mode": payload.get("mode"),
            "ece": payload.get("ece"),
            "picp_at_90": payload.get("picp_at_90"),
            "rmse": payload.get("rmse"),
        }
    if key == "weighted_conformal":
        plain = payload.get("plain_conformal", {})
        wc = payload.get("weighted_conformal", {})
        return {
            "plain_cov": plain.get("empirical_coverage"),
            "weighted_cov": wc.get("empirical"),
            "mean_width_plain": plain.get("mean_width"),
            "mean_width_weighted": wc.get("mean_width"),
        }
    if key == "survival_synth":
        pit = payload.get("pit", {})
        return {
            "n_uncensored": pit.get("n_uncensored"),
            "chi2_p": pit.get("chi2_p_value"),
            "c_index_baseline": payload.get("c_index_baseline"),
        }
    if key == "causal_shap":
        ranks = payload.get("rankings", {})
        return {
            "top_obs": [n for n, _ in ranks.get("observational", [])[:3]],
            "top_int": [n for n, _ in ranks.get("interventional", [])[:3]],
        }
    if key == "counterfactuals":
        return {"n_examples": payload.get("n_examples")}
    if key == "failure_gallery":
        return payload.get("summary", {})
    return {}


def main() -> int:
    manifest = run_all()
    print("Phase 5 rigor suite complete")
    print(f"  total duration : {manifest['total_duration_sec']} s")
    comp = manifest.get("compute", {})
    print(f"  codecarbon     : {comp.get('total_emissions_kg_co2')} kg CO2, "
          f"{comp.get('total_energy_kwh')} kWh across {comp.get('n_runs')} tracked calls")
    print("  per-step status:")
    for k, meta in manifest["steps"].items():
        print(f"    {k:<22} {meta['status']:<10} ({meta['duration_sec']}s)  -> {meta['output_file']}")
    print("\nManifest written to results/phase5/manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
