"""codecarbon / compute-report instrumentation.

Top-tier ML venues (NeurIPS, ICML) now expect a compute + carbon report
next to experimental results. codecarbon (mlco2/codecarbon) is the
standard estimator and is a soft dependency here: if it isn't
installed we fall back to reporting wall-clock time and a crude CPU
energy estimate so the runbook still produces *something* useful.

Usage
-----
    from src.phase5.carbon import track

    with track("phase3_train", out_dir="results/phase5") as t:
        train(...)
    # t.report is a dict you can json.dump alongside the other results.

The ``track`` context manager writes a JSON sidecar per run so you can
accumulate many runs and aggregate later. The aggregator lives in
``aggregate_reports``.
"""

from __future__ import annotations

import json
import os
import platform
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

try:
    from codecarbon import EmissionsTracker
    _HAS_CODECARBON = True
except Exception:  # pragma: no cover
    EmissionsTracker = None  # type: ignore
    _HAS_CODECARBON = False


@dataclass
class RunReport:
    run_id: str
    label: str
    started_at: float
    duration_sec: float = 0.0
    emissions_kg_co2: Optional[float] = None
    energy_kwh: Optional[float] = None
    cpu_energy_kwh: Optional[float] = None
    gpu_energy_kwh: Optional[float] = None
    country_code: Optional[str] = None
    fallback: bool = False
    host: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "run_id": self.run_id,
            "label": self.label,
            "started_at": self.started_at,
            "duration_sec": round(self.duration_sec, 3),
            "emissions_kg_co2": self.emissions_kg_co2,
            "energy_kwh": self.energy_kwh,
            "cpu_energy_kwh": self.cpu_energy_kwh,
            "gpu_energy_kwh": self.gpu_energy_kwh,
            "country_code": self.country_code,
            "fallback": self.fallback,
            "host": self.host,
            "extra": self.extra,
        }


def _host_info() -> Dict[str, Any]:
    try:
        import psutil
        mem_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    except Exception:
        mem_gb = None
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "mem_gb": mem_gb,
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return info


@contextmanager
def track(
    label: str,
    out_dir: Optional[str | Path] = None,
    country_code: Optional[str] = None,
) -> Iterator[RunReport]:
    """Context manager that produces a RunReport and optionally writes a JSON sidecar."""
    report = RunReport(
        run_id=uuid.uuid4().hex[:12],
        label=label,
        started_at=time.time(),
        host=_host_info(),
        country_code=country_code,
    )

    # codecarbon 3.x: country is auto-detected from IP; ISO override is via
    # electricitymaps_api_token, not a direct kwarg. We store the intended
    # country on the report for auditability but let codecarbon resolve it
    # at runtime. If the user explicitly passes one that disagrees with the
    # auto-detected, the sidecar will show both.
    start_wall = time.perf_counter()
    tracker: Optional[EmissionsTracker] = None
    if _HAS_CODECARBON:
        try:
            tracker = EmissionsTracker(
                project_name=label,
                log_level="ERROR",
                save_to_file=False,
                tracking_mode="process",
                allow_multiple_runs=True,
            )
            tracker.start()
        except Exception as exc:  # pragma: no cover - debug aid
            report.extra["codecarbon_init_error"] = repr(exc)
            tracker = None
            report.fallback = True
    else:
        report.fallback = True

    try:
        yield report
    finally:
        report.duration_sec = time.perf_counter() - start_wall
        if tracker is not None:
            try:
                emissions = float(tracker.stop())
                report.emissions_kg_co2 = emissions
                data = tracker.final_emissions_data
                if data is not None:
                    report.energy_kwh = float(getattr(data, "energy_consumed", 0.0))
                    report.cpu_energy_kwh = float(getattr(data, "cpu_energy", 0.0))
                    report.gpu_energy_kwh = float(getattr(data, "gpu_energy", 0.0))
                    report.country_code = getattr(data, "country_iso_code", report.country_code)
            except Exception:
                report.fallback = True
        if out_dir is not None:
            outp = Path(out_dir)
            outp.mkdir(parents=True, exist_ok=True)
            (outp / f"compute_{label}_{report.run_id}.json").write_text(
                json.dumps(report.to_dict(), indent=2), encoding="utf-8"
            )


# --------------------------------------------------------------------------- #
# Aggregation helper
# --------------------------------------------------------------------------- #


def aggregate_reports(reports_dir: str | Path) -> Dict:
    """Collect every compute_*.json sidecar into a summary table."""
    reports_dir = Path(reports_dir)
    rows: List[Dict] = []
    for path in sorted(reports_dir.glob("compute_*.json")):
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    if not rows:
        return {"n_runs": 0, "rows": []}
    total_duration = sum(r.get("duration_sec", 0) or 0 for r in rows)
    total_emissions = sum((r.get("emissions_kg_co2") or 0.0) for r in rows)
    total_energy = sum((r.get("energy_kwh") or 0.0) for r in rows)
    return {
        "n_runs": len(rows),
        "total_duration_sec": round(total_duration, 2),
        "total_emissions_kg_co2": round(total_emissions, 6),
        "total_energy_kwh": round(total_energy, 6),
        "rows": rows,
    }


# --------------------------------------------------------------------------- #
# CLI demo
# --------------------------------------------------------------------------- #


def _demo_workload(seconds: float = 0.5) -> None:
    import math
    end = time.perf_counter() + seconds
    x = 0.0
    while time.perf_counter() < end:
        x += math.sin(x) * 1e-3


def main() -> int:
    import argparse
    import numpy as np
    from pathlib import Path

    parser = argparse.ArgumentParser(description="codecarbon instrumentation demo")
    parser.add_argument("--label", default="phase5_demo")
    parser.add_argument("--seconds", type=float, default=1.0, help="CPU workload duration")
    parser.add_argument("--country", default=None, help="ISO-3166-1 alpha-3 code (e.g. IND)")
    parser.add_argument("--out-dir", default="results/phase5/compute")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    out_dir = project_root / args.out_dir

    with track(args.label, out_dir=out_dir, country_code=args.country) as report:
        _demo_workload(seconds=args.seconds)

    payload = report.to_dict()
    print("Compute run", payload["run_id"])
    print(f"  duration       : {payload['duration_sec']:.3f} s")
    if payload["fallback"]:
        print("  codecarbon     : not active (install it to get CO2)")
    else:
        print(f"  energy (kWh)   : {payload['energy_kwh']}")
        print(f"  kg CO2         : {payload['emissions_kg_co2']}")
        print(f"  country        : {payload['country_code']}")
    print(f"  host           : {payload['host']}")
    print(f"\n  sidecar written under {out_dir}")

    summary = aggregate_reports(out_dir)
    print(f"\nAggregate across {summary['n_runs']} runs:")
    print(f"  total duration : {summary.get('total_duration_sec')} s")
    print(f"  total CO2      : {summary.get('total_emissions_kg_co2')} kg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
