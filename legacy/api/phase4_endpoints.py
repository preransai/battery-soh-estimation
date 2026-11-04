"""Phase 4 (Tier-A novelty) endpoints for the SOH API.

Exposes the four Tier-A contributions over HTTP:

    POST /predict/conformal   conformal-calibrated SOH interval (90% / 95%)
    POST /predict/rul         Weibull RUL distribution (median + p10/p90)
    POST /diagnose/rap        retrieval-augmented LLM diagnosis with K analogues
    GET  /phase4/info         summary of the four contributions and their state

The endpoints rely on a pre-fit ``models/phase4_state.pkl`` artefact produced
by ``scripts/fit_phase4.py``. That keeps the runtime container slim: no torch,
no SBERT, no training data.
"""

from __future__ import annotations

import math
import os
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# State object loaded once at startup.
# --------------------------------------------------------------------------- #


class Phase4State:
    """In-memory holder of the calibration artefacts and the upstream predictor."""

    def __init__(self) -> None:
        self.loaded: bool = False
        self.state: Dict[str, Any] = {}
        self.predict_simple: Optional[Callable[[Dict[str, float]], float]] = None
        self.feature_labels: Dict[str, str] = {}
        self.openai_client = None
        self.openai_model: str = "gpt-4o-mini"
        self.llm_provider: Optional[str] = None

    def install(
        self,
        predict_fn: Callable[[Dict[str, float]], float],
        feature_labels: Dict[str, str],
        openai_client=None,
        openai_model: str = "gpt-4o-mini",
        llm_provider: Optional[str] = None,
    ) -> bool:
        self.predict_simple = predict_fn
        self.feature_labels = feature_labels
        self.openai_client = openai_client
        self.openai_model = openai_model
        self.llm_provider = llm_provider
        path = Path(__file__).resolve().parent.parent / "models" / "phase4_state.pkl"
        if not path.exists():
            print(f"Phase4: state file missing at {path} (endpoints will return 503)")
            return False
        with open(path, "rb") as f:
            self.state = pickle.load(f)
        self.loaded = True
        cov = self.state.get("diagnostics", {}).get("test_marginal_coverage")
        n_cal = self.state.get("conformal", {}).get("alpha_0_10", {}).get("n_calibration")
        print(f"Phase4 ready: cal_n={n_cal}  test_marginal_coverage={cov}")
        return True


PHASE4 = Phase4State()


# --------------------------------------------------------------------------- #
# Pydantic schemas (kept simple to mirror the existing /predict request).
# --------------------------------------------------------------------------- #


class FeaturesIn(BaseModel):
    capacity: float = Field(..., description="Battery capacity in Ah")
    cycle_number: int = Field(..., ge=0)
    voltage_min: float
    voltage_max: float
    temp_mean: float = 25.0
    discharge_time: float = 3600.0
    charges_per_day: float = Field(1.0, ge=0.1, le=10.0)
    # Optional rich features for retrieval; unused by the simple SOH predictor.
    voltage_mean: Optional[float] = None
    voltage_std: Optional[float] = None
    temp_max: Optional[float] = None
    temp_delta: Optional[float] = None
    ica_peak_voltage: Optional[float] = None
    ica_peak_magnitude: Optional[float] = None
    dcir: Optional[float] = None


class ConformalRequest(BaseModel):
    features: FeaturesIn
    confidence_pct: int = Field(90, ge=80, le=99, description="Target coverage (e.g. 90 or 95)")
    use_per_battery: bool = Field(False, description="Use Mondrian per-battery quantiles when available")
    battery_id: Optional[str] = None


class CQRRequest(BaseModel):
    features: FeaturesIn


class CQRResponse(BaseModel):
    soh_point: float
    lower: float
    upper: float
    width: float
    q_hat: float
    alpha: float
    calibration_n: int
    adaptive: bool = Field(True, description="Width varies per input, unlike split-conformal")
    note: str


class ConformalResponse(BaseModel):
    soh: float
    lower: float
    upper: float
    width: float
    target_coverage: float
    empirical_coverage: float
    q_hat: float
    calibration_n: int
    method: str  # "marginal" or "mondrian:<battery>"


class RULRequest(BaseModel):
    features: FeaturesIn
    horizon_days: int = Field(365, ge=30, le=3650)


class RULResponse(BaseModel):
    soh_now: float
    median_remaining_cycles: float
    p10_remaining_cycles: float
    p90_remaining_cycles: float
    median_remaining_days: float
    p10_remaining_days: float
    p90_remaining_days: float
    weibull_scale: float
    weibull_shape: float
    eol_threshold_pct: float
    note: str


class RAPHit(BaseModel):
    battery_id: str
    cycle_number: int
    soh: float
    cosine_distance: float
    description: str


class RAPRequest(BaseModel):
    features: FeaturesIn
    k: int = Field(5, ge=1, le=20)
    use_llm: bool = Field(True, description="If false, skip OpenAI and use the closest analogue's text.")


class RAPResponse(BaseModel):
    diagnosis: str
    used_llm: bool
    k: int
    hits: List[RAPHit]


class Phase4Info(BaseModel):
    loaded: bool
    contributions: List[Dict[str, Any]]
    diagnostics: Dict[str, Any]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _ensure_loaded() -> None:
    if not PHASE4.loaded or PHASE4.predict_simple is None:
        raise HTTPException(503, "Phase 4 state not loaded — run scripts/fit_phase4.py")


def _retriever_query(features: Dict[str, float], k: int) -> List[RAPHit]:
    ret = PHASE4.state.get("retriever")
    if not ret:
        return []
    cols = ret["feature_cols"]
    vec = np.array([float(features.get(c, 0.0) or 0.0) for c in cols], dtype=np.float32)
    vec = np.nan_to_num(vec)
    z = (vec - ret["mean"]) / ret["std"]
    norm = float(np.linalg.norm(z))
    if norm < 1e-8:
        return []
    z_unit = (z / norm).astype(np.float32)
    sims = ret["matrix"] @ z_unit
    idx = np.argsort(-sims)[:k]
    out: List[RAPHit] = []
    for i in idx:
        meta = ret["metadata"][int(i)]
        out.append(
            RAPHit(
                battery_id=str(meta["battery_id"]),
                cycle_number=int(meta["cycle_number"]),
                soh=float(meta["soh"]),
                cosine_distance=float(1.0 - sims[int(i)]),
                description=ret["descriptions"][int(i)],
            )
        )
    return out


PROMPT_TEMPLATE = (
    "You are a senior Li-ion battery diagnostic engineer. A query cycle has been measured "
    "and the K nearest historical cycles (retrieved from a labelled corpus by cosine "
    "similarity over standardised features) are listed below. Write a tight 3-5 sentence "
    "diagnostic paragraph: lead with the most likely degradation mode, justify it from the "
    "query features and the analogues, and close with one operational implication. Do not "
    "invent numbers; ground claims in the data given.\n\n"
    "Query features: {query}\n\n"
    "Retrieved analogues (lower distance = more similar):\n{analogues}\n"
)


def _format_query(features: Dict[str, float]) -> str:
    keys = ["capacity", "cycle_number", "voltage_min", "voltage_max", "temp_mean", "discharge_time",
            "voltage_mean", "ica_peak_voltage", "ica_peak_magnitude", "dcir", "temp_delta"]
    parts = []
    for k in keys:
        v = features.get(k)
        if v is None:
            continue
        parts.append(f"{k}={float(v):.3f}")
    return ", ".join(parts)


def _format_hits(hits: List[RAPHit]) -> str:
    return "\n\n".join(
        f"[Analogue {i+1}]  battery={h.battery_id}  cycle={h.cycle_number}  "
        f"SOH={h.soh:.2f}%  cos_dist={h.cosine_distance:.3f}\n    {h.description.strip()}"
        for i, h in enumerate(hits)
    )


def _llm_diagnose(features: Dict[str, float], hits: List[RAPHit]) -> Optional[str]:
    if PHASE4.openai_client is None or not hits:
        return None
    prompt = PROMPT_TEMPLATE.format(query=_format_query(features), analogues=_format_hits(hits))
    try:
        resp = PHASE4.openai_client.chat.completions.create(
            model=PHASE4.openai_model,
            temperature=0.2,
            max_tokens=320,
            messages=[
                {"role": "system", "content": "You are a battery diagnostics expert."},
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:  # noqa: BLE001 - any LLM failure should fall back gracefully
        print(f"RAP LLM failed: {exc}")
        return None


def _weibull_quantile(scale: float, shape: float, q: float) -> float:
    return float(scale * (-math.log(1.0 - q)) ** (1.0 / shape))


def _weibull_median(scale: float, shape: float) -> float:
    return _weibull_quantile(scale, shape, 0.5)


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #


router = APIRouter(tags=["phase4"])


@router.get("/phase4/info", response_model=Phase4Info)
async def phase4_info() -> Phase4Info:
    if not PHASE4.loaded:
        return Phase4Info(loaded=False, contributions=[], diagnostics={})
    s = PHASE4.state
    contributions = [
        {
            "id": "conformal",
            "name": "Conformal Prediction (fixed width)",
            "description": "Calibrated SOH intervals with finite-sample coverage guarantees.",
            "endpoint": "POST /predict/conformal",
            "alphas_supported": [0.05, 0.10],
            "calibration_n": s["conformal"]["alpha_0_10"]["n_calibration"],
            "marginal_q_hat_90": s["conformal"]["alpha_0_10"]["marginal_q"],
            "marginal_q_hat_95": s["conformal"]["alpha_0_05"]["marginal_q"],
            "per_battery_q_90": s["conformal"]["alpha_0_10"]["per_battery_q"],
        },
        {
            "id": "cqr",
            "name": "Conformalized Quantile Regression (adaptive)",
            "description": "Per-input interval width via quantile regression + conformal correction.",
            "endpoint": "POST /predict/cqr",
            "alpha": s.get("cqr", {}).get("alpha"),
            "q_hat": s.get("cqr", {}).get("q_hat"),
            "calibration_n": s.get("cqr", {}).get("n_calibration"),
            "available": "cqr" in s,
        },
        {
            "id": "weibull_rul",
            "name": "Weibull Survival RUL",
            "description": "Probabilistic time-to-EoL distribution (median + p10/p90).",
            "endpoint": "POST /predict/rul",
            "shape": s["weibull"]["shape"],
            "eol_threshold_pct": s["weibull"]["eol_threshold_pct"],
            "shape_estimation_n": s["weibull"]["shape_estimation_n"],
        },
        {
            "id": "rap",
            "name": "Retrieval-Augmented Prognostics",
            "description": "K-NN over training cycles + LLM-generated diagnosis with template fallback.",
            "endpoint": "POST /diagnose/rap",
            "corpus_size": len(s["retriever"]["descriptions"]),
            "feature_dims": len(s["retriever"]["feature_cols"]),
            "llm_provider": PHASE4.llm_provider,
            "llm_model": PHASE4.openai_model if PHASE4.openai_client else None,
            "llm_available": PHASE4.openai_client is not None,
            "openai_available": PHASE4.openai_client is not None,  # kept for back-compat
        },
        {
            "id": "pinn",
            "name": "Physics-Informed Loss (training-time)",
            "description": s["pinn_info"]["summary"],
            "endpoint": None,
            "loss_weights": s["pinn_info"]["loss_weights"],
            "described_in": s["pinn_info"]["described_in"],
        },
    ]
    return Phase4Info(loaded=True, contributions=contributions, diagnostics=s.get("diagnostics", {}))


@router.post("/predict/cqr", response_model=CQRResponse)
async def predict_cqr(req: CQRRequest) -> CQRResponse:
    """Conformalized Quantile Regression - width adapts per input.

    Unlike /predict/conformal which gives a fixed +/- q interval, CQR returns a
    tight interval for easy samples (near the training distribution) and a wide
    one for hard samples. Both have the same distribution-free 90% coverage guarantee.
    """
    _ensure_loaded()
    cqr_state = PHASE4.state.get("cqr")
    if not cqr_state:
        raise HTTPException(503, "CQR calibrator missing - re-run scripts/fit_phase4.py")
    features = req.features.dict()
    feat_cols = cqr_state["feature_cols"]
    row = np.array([[float(features.get(c) or 0.0) for c in feat_cols]], dtype=np.float32)
    row = np.nan_to_num(row)
    q_lo = cqr_state["model_lo"].predict(row)
    q_hi = cqr_state["model_hi"].predict(row)
    point = cqr_state["model_median"].predict(row)
    q_hat = float(cqr_state["q_hat"])
    lower = float(q_lo[0]) - q_hat
    upper = float(q_hi[0]) + q_hat
    return CQRResponse(
        soh_point=round(float(point[0]), 3),
        lower=round(max(0.0, lower), 3),
        upper=round(min(100.0, upper), 3),
        width=round(upper - lower, 3),
        q_hat=round(q_hat, 3),
        alpha=cqr_state["alpha"],
        calibration_n=int(cqr_state["n_calibration"]),
        adaptive=True,
        note=(
            "Interval width varies by input. Coverage guarantee of 1 - alpha "
            "holds marginally over exchangeable data."
        ),
    )


@router.post("/predict/conformal", response_model=ConformalResponse)
async def predict_conformal(req: ConformalRequest) -> ConformalResponse:
    _ensure_loaded()
    feats = req.features.dict()
    soh = float(PHASE4.predict_simple(feats))

    alpha_key = "alpha_0_10" if req.confidence_pct == 90 else "alpha_0_05" if req.confidence_pct == 95 else None
    if alpha_key is None:
        raise HTTPException(400, "confidence_pct must be 90 or 95 (matches pre-fit calibrators)")
    cinfo = PHASE4.state["conformal"][alpha_key]

    method = "marginal"
    q = cinfo["marginal_q"]
    if req.use_per_battery and req.battery_id:
        per = cinfo["per_battery_q"].get(req.battery_id)
        if per is not None:
            q = per
            method = f"mondrian:{req.battery_id}"

    return ConformalResponse(
        soh=round(soh, 3),
        lower=round(max(0.0, soh - q), 3),
        upper=round(min(100.0, soh + q), 3),
        width=round(2 * q, 3),
        target_coverage=round(1.0 - cinfo["alpha"], 3),
        empirical_coverage=round(PHASE4.state["diagnostics"]["test_marginal_coverage"], 3),
        q_hat=round(q, 3),
        calibration_n=cinfo["n_calibration"],
        method=method,
    )


@router.post("/predict/rul", response_model=RULResponse)
async def predict_rul(req: RULRequest) -> RULResponse:
    _ensure_loaded()
    feats = req.features.dict()
    soh = float(PHASE4.predict_simple(feats))

    weibull = PHASE4.state["weibull"]
    shape = float(weibull["shape"])
    eol_pct = float(weibull["eol_threshold_pct"])

    # Scale = expected cycles until SOH crosses EoL, derived from a simple linear-fade
    # extrapolation. The Weibull then puts uncertainty around that point estimate using the
    # global shape parameter estimated at fit time (constant-shape AFT-style).
    soh_gap = max(0.5, soh - eol_pct)
    fade_per_cycle = max(0.005, 0.04 + 0.0005 * float(feats.get("cycle_number", 0)))
    if float(feats.get("temp_mean", 25)) > 25:
        fade_per_cycle *= 2.0 ** ((float(feats["temp_mean"]) - 25.0) / 10.0)
    scale_cycles = float(np.clip(soh_gap / fade_per_cycle, 1.0, 5000.0))

    if soh <= eol_pct:
        return RULResponse(
            soh_now=round(soh, 3), median_remaining_cycles=0.0, p10_remaining_cycles=0.0,
            p90_remaining_cycles=0.0, median_remaining_days=0.0, p10_remaining_days=0.0,
            p90_remaining_days=0.0, weibull_scale=scale_cycles, weibull_shape=shape,
            eol_threshold_pct=eol_pct, note="Already at or below End-of-Life threshold.",
        )

    median = _weibull_median(scale_cycles, shape)
    p10 = _weibull_quantile(scale_cycles, shape, 0.10)
    p90 = _weibull_quantile(scale_cycles, shape, 0.90)
    cpd = max(0.1, float(feats.get("charges_per_day", 1.0)))

    return RULResponse(
        soh_now=round(soh, 3),
        median_remaining_cycles=round(median, 1),
        p10_remaining_cycles=round(p10, 1),
        p90_remaining_cycles=round(p90, 1),
        median_remaining_days=round(median / cpd, 1),
        p10_remaining_days=round(p10 / cpd, 1),
        p90_remaining_days=round(p90 / cpd, 1),
        weibull_scale=round(scale_cycles, 2),
        weibull_shape=round(shape, 3),
        eol_threshold_pct=eol_pct,
        note=(
            f"Weibull(scale={scale_cycles:.1f} cycles, shape={shape:.2f}). "
            "Width of the p10-p90 band reflects the uncertainty of the wear-out distribution, "
            "not the upstream point estimate's noise."
        ),
    )


@router.post("/diagnose/rap", response_model=RAPResponse)
async def diagnose_rap(req: RAPRequest) -> RAPResponse:
    _ensure_loaded()
    feats = req.features.dict()
    hits = _retriever_query(feats, req.k)
    diagnosis: Optional[str] = None
    used_llm = False
    if req.use_llm:
        diagnosis = _llm_diagnose(feats, hits)
        used_llm = diagnosis is not None
    if diagnosis is None and hits:
        # Template fallback: nearest analogue's diagnosis, plus a short framing line.
        nearest = hits[0]
        diagnosis = (
            f"Closest historical analogue: battery {nearest.battery_id} at cycle "
            f"{nearest.cycle_number} (SOH {nearest.soh:.1f}%, cosine distance "
            f"{nearest.cosine_distance:.3f}). Reading: {nearest.description.strip()}"
        )
    if diagnosis is None:
        diagnosis = "No analogues retrieved (retriever index unavailable)."
    return RAPResponse(diagnosis=diagnosis, used_llm=used_llm, k=req.k, hits=hits)


def mount(app, predict_fn: Callable[[Dict[str, float]], float], feature_labels: Dict[str, str],
          openai_client=None, openai_model: str = "gpt-4o-mini") -> bool:
    """Attach Phase 4 endpoints to ``app`` and load the calibration artefact."""
    ok = PHASE4.install(predict_fn, feature_labels, openai_client=openai_client, openai_model=openai_model)
    app.include_router(router)
    return ok
