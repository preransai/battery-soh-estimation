"""Run every Phase-6 deferred module in sequence and emit a manifest.

Modules (in order):
    1. rap_benchmark         (requires FastAPI at :8000; skipped if down)
    2. llm_feature_discovery (requires GROQ_API_KEY; degrades cleanly)
    3. pinn_sweep
    4. lobo_cv
    5. tent_adapt
    6. curriculum
    7. ssl_contrastive

Writes results/phase6/manifest.json summarising which modules ran and
their headline numbers, so downstream tooling (the frontend's Rigor
tab, the paper's Phase-6 table) has a single place to look.
"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Dict

import requests


def _default_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _api_up(url: str = "http://localhost:8000/health", timeout: float = 1.5) -> bool:
    try:
        r = requests.get(url, timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def run_all(project_root: Path | None = None) -> Dict:
    project_root = Path(project_root or _default_root())
    results_dir = project_root / "results" / "phase6"
    results_dir.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Dict] = {"started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                 "modules": {}}
    t_global = time.perf_counter()

    # 1. RAP benchmark
    if _api_up():
        try:
            from src.phase6.rap_benchmark import run_benchmark
            t = time.perf_counter()
            out = run_benchmark(n_samples=10)
            manifest["modules"]["rap_benchmark"] = {
                "status": "error" if "error" in out else "ok",
                "seconds": round(time.perf_counter() - t, 2),
                "head": {k: out.get(k) for k in (
                    "n_samples", "template_mae_via_top_hit", "llm_mae_via_top_hit",
                    "llm_actually_used", "llm_fallback_rate",
                )},
            }
        except Exception as exc:
            manifest["modules"]["rap_benchmark"] = {
                "status": "exception", "error": str(exc),
                "trace": traceback.format_exc()[-600:],
            }
    else:
        manifest["modules"]["rap_benchmark"] = {
            "status": "skipped", "reason": "FastAPI not responding on :8000",
        }

    # 2. LLM feature discovery
    try:
        from src.phase6.llm_feature_discovery import run_discovery
        t = time.perf_counter()
        out = run_discovery()
        manifest["modules"]["llm_feature_discovery"] = {
            "status": "error" if "error" in out else "ok",
            "seconds": round(time.perf_counter() - t, 2),
            "head": {k: out.get(k) for k in ("baseline", "n_proposals", "n_evaluated")},
        }
    except Exception as exc:
        manifest["modules"]["llm_feature_discovery"] = {
            "status": "exception", "error": str(exc),
            "trace": traceback.format_exc()[-600:],
        }

    # 3. PINN weight sweep
    try:
        from src.phase6.pinn_sweep import run_sweep
        t = time.perf_counter()
        out = run_sweep()
        manifest["modules"]["pinn_sweep"] = {
            "status": "ok",
            "seconds": round(time.perf_counter() - t, 2),
            "head": {
                "n_configurations": out["n_configurations"],
                "top1": out["top3_composite"][0] if out["top3_composite"] else None,
                "baseline_no_penalty": out["baseline_no_penalties"],
            },
        }
    except Exception as exc:
        manifest["modules"]["pinn_sweep"] = {
            "status": "exception", "error": str(exc),
            "trace": traceback.format_exc()[-600:],
        }

    # 4. LOBO CV
    try:
        from src.phase6.lobo_cv import run_lobo
        t = time.perf_counter()
        out = run_lobo()
        manifest["modules"]["lobo_cv"] = {
            "status": "ok",
            "seconds": round(time.perf_counter() - t, 2),
            "head": {
                "seeds": out["seeds"],
                "nasa_only_overall_mae": out["nasa_only"].get("overall_mae_mean"),
                "combined_overall_mae":  out["combined_nasa_mendeley"].get("overall_mae_mean"),
            },
        }
    except Exception as exc:
        manifest["modules"]["lobo_cv"] = {
            "status": "exception", "error": str(exc),
            "trace": traceback.format_exc()[-600:],
        }

    # 5. TENT
    try:
        from src.phase6.tent_adapt import run_tent
        t = time.perf_counter()
        out = run_tent()
        manifest["modules"]["tent_adapt"] = {
            "status": "error" if "error" in out else "ok",
            "seconds": round(time.perf_counter() - t, 2),
            "head": {k: out.get(k) for k in (
                "baseline_no_adapt", "tent_scaler_only", "tent_full_self_train",
                "oracle_labelled_target", "shift_diagnostics",
            )},
        }
    except Exception as exc:
        manifest["modules"]["tent_adapt"] = {
            "status": "exception", "error": str(exc),
            "trace": traceback.format_exc()[-600:],
        }

    # 6. Curriculum
    try:
        from src.phase6.curriculum import run_curriculum
        t = time.perf_counter()
        out = run_curriculum()
        manifest["modules"]["curriculum"] = {
            "status": "ok",
            "seconds": round(time.perf_counter() - t, 2),
            "head": {"verdict": out["verdict"], "aggregates": out["aggregates"]},
        }
    except Exception as exc:
        manifest["modules"]["curriculum"] = {
            "status": "exception", "error": str(exc),
            "trace": traceback.format_exc()[-600:],
        }

    # 7. SSL contrastive
    try:
        from src.phase6.ssl_contrastive import run_ssl
        t = time.perf_counter()
        out = run_ssl()
        manifest["modules"]["ssl_contrastive"] = {
            "status": "ok",
            "seconds": round(time.perf_counter() - t, 2),
            "head": {"verdict": out["verdict"],
                     "raw": out["baseline_raw_ridge"],
                     "pca": out["baseline_pca_ridge"],
                     "contrastive": {k: out["contrastive_probe"][k] for k in (
                         "mae_mean", "mae_std", "r2_mean", "r2_std")}},
        }
    except Exception as exc:
        manifest["modules"]["ssl_contrastive"] = {
            "status": "exception", "error": str(exc),
            "trace": traceback.format_exc()[-600:],
        }

    manifest["total_seconds"] = round(time.perf_counter() - t_global, 2)
    manifest["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(results_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main() -> int:
    m = run_all()
    print("Phase 6 deferred run complete")
    print("=" * 60)
    for name, info in m["modules"].items():
        status = info["status"]
        secs = info.get("seconds", "-")
        print(f"  {name:<24} [{status:<9}] {secs}s")
    print(f"\nTotal : {m['total_seconds']}s  manifest -> results/phase6/manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
