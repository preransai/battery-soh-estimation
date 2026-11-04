"""
Battery SOH Prediction API v3 — Full Novel Feature Set

Endpoints:
- POST /predict          — SOH + RUL + risk + uncertainty + SHAP + AI guidance
- POST /simulate         — What-If Digital Twin simulator
- POST /chat             — Conversational Battery Doctor
- GET  /report/pdf/{id}  — PDF diagnostic report download
- GET  /model/generalization — Cross-battery transfer learning metrics
- GET  /health, /model/info, /disposal/ranges
"""

import os, sys, json, io, time, pickle
from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, '.env'))
except Exception:
    pass

from api.disposal_guidance import all_bands_public, guidance_dict
from api.phase4_endpoints import PHASE4 as PHASE4_STATE, router as phase4_router
from api.cross_attention_service import SERVICE as XATTN_SERVICE

# =================== Global State ===================
model = None
scaler = None
feature_cols = None
bootstrap_models = None
# Split-conformal calibration: dict of {nominal_coverage: half_width}, e.g.
# {0.5: 0.42, 0.6: 0.55, 0.7: 0.66, 0.8: 0.78, 0.9: 0.93, 0.95: 1.05, 0.99: 1.45}.
# Computed once at startup from absolute residuals on a calibration fold.
conformal_quantiles = None
shap_explainer = None
rul_params = None
cross_battery_metrics = None
openai_client = None
openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
llm_provider: Optional[str] = None  # "groq" or "openai" once init_llm() picks one

# Prediction cache for PDF reports (last 20)
prediction_cache: Dict[str, dict] = {}

FEATURE_LABELS = {
    'capacity': 'Battery Capacity (Ah)',
    'cycle_number': 'Charge Cycles Used',
    'voltage_min': 'Min Voltage (V)',
    'voltage_max': 'Max Voltage (V)',
    'temp_mean': 'Avg Temperature (C)',
    'discharge_time': 'Discharge Time (s)',
}

# =================== Pydantic Models ===================

class SimpleFeatures(BaseModel):
    capacity: float = Field(..., description="Battery capacity in Ah")
    cycle_number: int = Field(..., ge=0)
    voltage_min: float
    voltage_max: float
    temp_mean: float = 25.0
    discharge_time: float = 3600.0
    charges_per_day: float = Field(1.0, ge=0.1, le=10.0)

class PredictionRequest(BaseModel):
    features: SimpleFeatures
    include_explanation: bool = True

class ShapFeature(BaseModel):
    feature: str
    label: str
    value: float
    shap_value: float

class ShapExplanation(BaseModel):
    base_value: float
    features: List[ShapFeature]

class RULEstimate(BaseModel):
    remaining_cycles: int
    remaining_days: int
    eol_threshold: float
    confidence_note: str

class RiskScore(BaseModel):
    overall_risk: int = Field(..., description="0-100 risk score")
    risk_level: str = Field(..., description="Low/Medium/High/Critical")
    thermal_risk: int
    voltage_risk: int
    capacity_risk: int
    cycle_risk: int
    risk_factors: List[str]

class UncertaintyBand(BaseModel):
    soh_mean: float
    soh_lower: float
    soh_upper: float
    confidence_pct: int = 90

class DisposalGuidanceResponse(BaseModel):
    band_id: str
    label: str
    soh_min: float
    soh_max: float
    summary: str
    disposal_methods: List[str]
    analysis_precautions: List[str]
    transport_storage_precautions: List[str]
    is_low_soh: bool

class PredictionResponse(BaseModel):
    prediction_id: str
    soh: float
    confidence: Optional[float] = None
    uncertainty: Optional[UncertaintyBand] = None
    risk: Optional[RiskScore] = None
    rul: Optional[RULEstimate] = None
    explanation: Optional[str] = None
    shap_explanation: Optional[ShapExplanation] = None
    recommendations: List[str] = Field(default_factory=list)
    llm_recommendation: Optional[str] = None
    disposal_guidance: Optional[DisposalGuidanceResponse] = None
    processing_time_ms: float

class SimulateRequest(BaseModel):
    base_features: SimpleFeatures
    scenarios: List[dict] = Field(..., description="List of {charges_per_day, temp_mean} overrides")
    horizon_days: int = Field(365, ge=30, le=1825)

class SimulatePoint(BaseModel):
    day: int
    cycle: int
    soh: float
    capacity: float

class SimulateScenario(BaseModel):
    label: str
    charges_per_day: float
    temp_mean: float
    trajectory: List[SimulatePoint]
    eol_day: Optional[int] = None

class SimulateResponse(BaseModel):
    scenarios: List[SimulateScenario]

class ChatRequest(BaseModel):
    message: str
    battery_context: Optional[SimpleFeatures] = None

class ChatResponse(BaseModel):
    reply: str
    suggestions: List[str] = Field(default_factory=list)

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    features_available: List[str]
    timestamp: str
    version: str

# =================== FastAPI App ===================

app = FastAPI(
    title="Battery SOH Prediction API",
    description="Li-ion SOH prediction with SHAP, risk scoring, digital twin, chat, PDF reports",
    version="3.0.0",
    docs_url="/docs", redoc_url="/redoc",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])
app.include_router(phase4_router)

# =================== Model Loading ===================

def load_all():
    global model, scaler, feature_cols, rul_params, bootstrap_models, cross_battery_metrics, conformal_quantiles

    for name, fname in [('model','simple_model.pkl'),('scaler','simple_scaler.pkl'),
                        ('feature_cols','simple_feature_cols.pkl')]:
        path = os.path.join(PROJECT_ROOT, 'models', fname)
        if not os.path.exists(path):
            print(f"FATAL: {fname} not found"); return False
        with open(path, 'rb') as f:
            globals()[name] = pickle.load(f)

    model = globals()['model']; scaler = globals()['scaler']; feature_cols = globals()['feature_cols']

    rul_path = os.path.join(PROJECT_ROOT, 'models', 'rul_params.pkl')
    if os.path.exists(rul_path):
        with open(rul_path, 'rb') as f: rul_params = pickle.load(f)
    else:
        rul_params = {'degradation_coeffs':[0,-0.004,1.9],'rated_capacity':2.0,
                      'eol_threshold_pct':70,'eol_capacity':1.4}

    bs_path = os.path.join(PROJECT_ROOT, 'models', 'bootstrap_models.pkl')
    if os.path.exists(bs_path):
        with open(bs_path, 'rb') as f: bootstrap_models = pickle.load(f)
        print(f"Loaded {len(bootstrap_models)} bootstrap models for uncertainty")

    cb_path = os.path.join(PROJECT_ROOT, 'models', 'cross_battery_metrics.json')
    if os.path.exists(cb_path):
        with open(cb_path) as f: cross_battery_metrics = json.load(f)
        print(f"Loaded cross-battery metrics: {list(cross_battery_metrics.keys())}")

    # Split-conformal calibration on the held-out test set.
    # The absolute residuals |y - y_hat| on the production model are
    # gathered and turned into a dict of {nominal: half-width} quantiles
    # at the standard nominal levels. compute_uncertainty() then looks
    # up the half-width for the requested confidence level.
    try:
        import pandas as pd
        test_csv = os.path.join(PROJECT_ROOT, 'data', 'phase2', 'harmonized', 'test_features.csv')
        if os.path.exists(test_csv):
            df = pd.read_csv(test_csv).dropna(subset=feature_cols + ['soh'])
            if len(df) >= 20:
                X_t = scaler.transform(df[feature_cols].to_numpy(dtype=float))
                yhat = np.clip(model.predict(X_t), 0, 100)
                y = df['soh'].to_numpy(dtype=float)
                residuals = np.abs(y - yhat)
                n = len(residuals)
                conformal_quantiles = {}
                for nominal in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99):
                    alpha = 1.0 - nominal
                    order = int(np.ceil((n + 1) * (1.0 - alpha))) - 1
                    order = max(0, min(order, n - 1))
                    q = float(np.sort(residuals)[order])
                    conformal_quantiles[round(nominal, 2)] = q
                print(f"Conformal quantiles ready: {conformal_quantiles}")
    except Exception as exc:  # noqa: BLE001
        print(f"Conformal calibration skipped: {exc}")
        conformal_quantiles = None

    print(f"Model ready: {type(model).__name__}, {len(feature_cols)} features")
    _init_shap()
    return True


def _init_shap():
    global shap_explainer
    try:
        import shap
        bg = np.random.RandomState(42).rand(100, len(feature_cols))
        shap_explainer = shap.KernelExplainer(model.predict, bg)
        print("SHAP ready")
    except Exception as e:
        print(f"SHAP failed: {e}"); shap_explainer = None


def init_llm():
    """Pick a Groq key if present, else OpenAI. Either uses the openai-SDK wire format."""
    global openai_client, openai_model, llm_provider
    groq_key = os.getenv("GROQ_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    placeholder = {"your_key_here", "your_groq_key_here", "", None}

    try:
        from openai import OpenAI
    except Exception as e:
        print(f"openai SDK unavailable: {e}")
        return False

    if groq_key and groq_key not in placeholder:
        openai_client = OpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1")
        openai_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        llm_provider = "groq"
        print(f"LLM ready: groq ({openai_model})")
        return True
    if openai_key and openai_key not in placeholder:
        openai_client = OpenAI(api_key=openai_key)
        openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        llm_provider = "openai"
        print(f"LLM ready: openai ({openai_model})")
        return True

    print("LLM not configured: set GROQ_API_KEY or OPENAI_API_KEY")
    return False


# Back-compat alias (external callers may still import init_openai)
init_openai = init_llm

# =================== Core Computations ===================

def predict_soh(features: dict):
    """Primary predictor. Uses the simple Ridge model — it has tight per-sample
    accuracy across the entire training distribution. The cross-attention model
    has a better *average* MAE but a longer tail on individual cycles; it's
    exposed on /predict/crossattention for users who want to probe it directly.
    Returns (soh, scaled_simple_feature_row)."""
    X = np.array([[features.get(f, 0.0) for f in feature_cols]])
    X_s = scaler.transform(X)
    soh = float(np.clip(model.predict(X_s), 0, 100)[0])
    return soh, X_s


# Kept for the legacy /predict/simple route which calls it explicitly.
predict_soh_simple = predict_soh


def predict_soh_crossattention(features: dict) -> Optional[float]:
    """Run the Phase 3 cross-attention model if available. Returns None on failure."""
    if not XATTN_SERVICE.loaded:
        return None
    try:
        return float(XATTN_SERVICE.predict(features))
    except Exception as exc:  # noqa: BLE001
        print(f"cross-attention predict failed: {exc}")
        return None


def compute_uncertainty(X_scaled: np.ndarray, soh: float,
                        nominal: float = 0.90) -> Optional[UncertaintyBand]:
    """Split-conformal 90% interval centred on the production point.

    Uses the calibrated half-width that was computed at server start-up
    from absolute residuals on the held-out test set (Lei et al. 2018).
    On exchangeable data this gives marginal coverage of at least the
    nominal level — that is, ~90% of true SOH values fall inside the
    band over many predictions. If the calibration step failed at
    start-up we fall back to a centred bootstrap-spread band, which is
    less rigorous but never returns nonsense.
    """
    # Preferred path: split-conformal half-width from the calibration table.
    if conformal_quantiles is not None:
        half_width = conformal_quantiles.get(round(nominal, 2))
        if half_width is None:
            keys = sorted(conformal_quantiles.keys())
            nearest = min(keys, key=lambda k: abs(k - nominal))
            half_width = conformal_quantiles[nearest]
        return UncertaintyBand(
            soh_mean=round(soh, 2),
            soh_lower=round(max(0.0, soh - half_width), 2),
            soh_upper=round(min(100.0, soh + half_width), 2),
            confidence_pct=int(round(nominal * 100)),
        )

    # Fallback: bootstrap-spread band centred on the production point.
    if bootstrap_models is None:
        return None
    preds = np.array(
        [float(np.clip(m.predict(X_scaled), 0, 100)[0]) for m in bootstrap_models]
    )
    half_width = float(np.percentile(np.abs(preds - soh), nominal * 100))
    return UncertaintyBand(
        soh_mean=round(soh, 2),
        soh_lower=round(max(0.0, soh - half_width), 2),
        soh_upper=round(min(100.0, soh + half_width), 2),
        confidence_pct=int(round(nominal * 100)),
    )


def compute_risk(features: dict, soh: float, rul: Optional[RULEstimate]) -> RiskScore:
    factors = []
    cap = features.get('capacity', 2.0)
    v_min = features.get('voltage_min', 3.0)
    v_max = features.get('voltage_max', 4.2)
    temp = features.get('temp_mean', 25)
    cycles = features.get('cycle_number', 0)

    # Thermal risk (0-100)
    if temp > 45: thermal = 95; factors.append(f"Extreme temperature ({temp:.0f}C) - thermal runaway risk")
    elif temp > 40: thermal = 75; factors.append(f"High temperature ({temp:.0f}C) - accelerated degradation")
    elif temp > 35: thermal = 50; factors.append(f"Elevated temperature ({temp:.0f}C)")
    elif temp < 5: thermal = 60; factors.append(f"Low temperature ({temp:.0f}C) - lithium plating risk")
    else: thermal = 10

    # Voltage risk
    if v_min < 2.0: voltage = 95; factors.append(f"Critically low voltage ({v_min:.2f}V) - deep discharge damage")
    elif v_min < 2.5: voltage = 70; factors.append(f"Low voltage ({v_min:.2f}V) - cell damage risk")
    elif v_max > 4.3: voltage = 80; factors.append(f"Overcharge detected ({v_max:.2f}V)")
    elif v_max > 4.25: voltage = 50; factors.append(f"Voltage near limit ({v_max:.2f}V)")
    else: voltage = 10

    # Capacity risk
    fade = (2.0 - cap) / 2.0 * 100
    if fade > 40: capacity_r = 90; factors.append(f"Severe capacity fade ({fade:.0f}% lost)")
    elif fade > 30: capacity_r = 70; factors.append(f"Significant capacity fade ({fade:.0f}% lost)")
    elif fade > 20: capacity_r = 45; factors.append(f"Moderate capacity fade ({fade:.0f}% lost)")
    else: capacity_r = 10

    # Cycle risk
    if cycles > 300: cycle_r = 80; factors.append(f"High cycle count ({cycles})")
    elif cycles > 200: cycle_r = 55; factors.append(f"Elevated cycle count ({cycles})")
    elif cycles > 100: cycle_r = 30
    else: cycle_r = 10

    # Bonus risk factors from RUL
    if rul and rul.remaining_days == 0:
        factors.append("Battery has reached end-of-life threshold")

    overall = int(thermal * 0.3 + voltage * 0.25 + capacity_r * 0.25 + cycle_r * 0.2)
    if overall >= 75: level = "Critical"
    elif overall >= 50: level = "High"
    elif overall >= 25: level = "Medium"
    else: level = "Low"

    if not factors:
        factors.append("All parameters within normal operating range")

    return RiskScore(overall_risk=overall, risk_level=level, thermal_risk=thermal,
                     voltage_risk=voltage, capacity_risk=capacity_r, cycle_risk=cycle_r,
                     risk_factors=factors)


def compute_rul(cap, cycle, cpd) -> Optional[RULEstimate]:
    if not rul_params: return None
    a, b, c = rul_params['degradation_coeffs']
    eol_cap = rul_params['eol_capacity']
    eol_pct = rul_params['eol_threshold_pct']
    if cap <= eol_cap:
        return RULEstimate(remaining_cycles=0, remaining_days=0, eol_threshold=eol_pct,
                           confidence_note="Battery has reached end-of-life threshold.")
    rate = 2*a*cycle + b
    if rate >= 0: rate = b
    rem_cyc = max(0, min(5000, int((cap - eol_cap) / abs(rate))))
    rem_days = max(0, int(rem_cyc / cpd))
    if cycle < 10: note = "Low confidence: few cycles recorded."
    elif cycle < 50: note = "Moderate confidence: based on early trend."
    else: note = "Good confidence: established degradation pattern."
    return RULEstimate(remaining_cycles=rem_cyc, remaining_days=rem_days,
                       eol_threshold=eol_pct, confidence_note=note)


def compute_shap(X_s, raw) -> Optional[ShapExplanation]:
    if not shap_explainer: return None
    try:
        sv = shap_explainer.shap_values(X_s, nsamples=200).flatten().tolist()
        bv = float(shap_explainer.expected_value)
        if hasattr(shap_explainer.expected_value, '__len__'):
            bv = float(shap_explainer.expected_value[0])
        feats = [ShapFeature(feature=c, label=FEATURE_LABELS.get(c,c),
                             value=round(float(raw.get(c,0)),4), shap_value=round(sv[i],4))
                 for i,c in enumerate(feature_cols)]
        feats.sort(key=lambda f: abs(f.shap_value), reverse=True)
        return ShapExplanation(base_value=round(bv,4), features=feats)
    except Exception as e:
        print(f"SHAP error: {e}"); return None


def generate_explanation(feat, soh, rul):
    if soh >= 90: s = f"Battery is in excellent condition (SOH: {soh:.1f}%)."
    elif soh >= 80: s = f"Battery is in good condition (SOH: {soh:.1f}%)."
    elif soh >= 70: s = f"Battery shows moderate degradation (SOH: {soh:.1f}%)."
    else: s = f"Battery is significantly degraded (SOH: {soh:.1f}%). Consider replacement."
    s += f" Capacity: {feat.get('capacity',0):.3f} Ah after {int(feat.get('cycle_number',0))} cycles."
    if rul and rul.remaining_days > 0:
        s += f" Estimated {rul.remaining_days} days of useful life remaining."
    return s


def generate_recommendations(soh, feat):
    recs = []
    if soh >= 90:
        recs += ["Battery health is excellent. Continue normal operation.",
                 "Maintain charging between 20%-80% to minimize aging."]
    elif soh >= 80:
        recs += ["Battery health is good. Keep routine maintenance.",
                 "Track SOH every 50 cycles to catch degradation early."]
    elif soh >= 70:
        recs += ["Moderate degradation detected. Increase monitoring.",
                 "Plan a capacity test in next maintenance window."]
    elif soh >= 50:
        recs += ["Significant degradation. Begin replacement planning.",
                 "Avoid high-current loads and monitor for voltage sag."]
    else:
        recs += ["Battery near end-of-life. Remove from critical service.",
                 "Follow hazardous e-waste disposal regulations."]
    if feat.get('capacity',2) < 1.2:
        recs.append("Low capacity: validate with controlled discharge test.")
    if feat.get('temp_mean',25) > 35:
        recs.append(f"High operating temp ({feat['temp_mean']:.0f}C): improve cooling.")
    return recs


def build_disposal(soh, feat=None, rul_est=None):
    static = guidance_dict(soh)
    if not feat: return DisposalGuidanceResponse(**static)
    cap=feat.get('capacity',0); cyc=int(feat.get('cycle_number',0))
    v_min=feat.get('voltage_min',3); temp=feat.get('temp_mean',25)
    days=rul_est.remaining_days if rul_est else None

    if soh>=90: sm=f"Battery at {soh:.1f}% SOH ({cap:.2f} Ah, {cyc} cycles) in excellent health."
    elif soh>=80: sm=f"At {soh:.1f}% SOH ({cap:.2f} Ah, {cyc} cycles), healthy but showing early wear."
    elif soh>=70: sm=f"Battery at {soh:.1f}% SOH ({cap:.2f} Ah, {cyc} cycles) — moderate degradation."
    else: sm=f"At {soh:.1f}% SOH ({cap:.2f} Ah, {cyc} cycles) — significantly degraded."
    if days and days>0: sm+=f" ~{days} days remaining."
    elif days==0: sm+=" End-of-life reached."

    disp=list(static['disposal_methods'])
    if cap<1.4: disp.insert(0,f"Capacity {cap:.2f} Ah (<70% rated) — route to recycling.")
    prec=list(static['analysis_precautions'])
    if temp>35: prec.insert(0,f"Operating at {temp:.0f}C — monitor for thermal events.")
    if v_min<2.5: prec.insert(0,f"Min voltage {v_min:.2f}V dangerously low — check before recharge.")
    trans=list(static['transport_storage_precautions'])
    if days is not None and 0<days<30:
        trans.append(f"~{days} days left — schedule recycler pickup now.")

    static.update(summary=sm, disposal_methods=disp[:4], analysis_precautions=prec[:4],
                  transport_storage_precautions=trans[:3])
    return DisposalGuidanceResponse(**static)


def ai_disposal(soh, feat, rul_est):
    if not openai_client:
        return build_disposal(soh, feat, rul_est)
    static = guidance_dict(soh)
    rul_info = f"~{rul_est.remaining_days}d ({rul_est.remaining_cycles} cycles) remaining." if rul_est else ""
    try:
        r = openai_client.chat.completions.create(
            model=openai_model, temperature=0.3, max_tokens=500,
            messages=[{"role":"system","content":
                "You are a Li-ion battery safety expert. Given battery data, return ONLY valid JSON:\n"
                '{"summary":"...","disposal_methods":["...","...","..."],'
                '"analysis_precautions":["...","...","..."],'
                '"transport_storage_precautions":["...","..."]}\n'
                "Reference actual values. Safety-first. No markdown."},
                {"role":"user","content":
                f"SOH:{soh:.1f}%, Cap:{feat.get('capacity',0):.3f}Ah, "
                f"Cycles:{int(feat.get('cycle_number',0))}, "
                f"V:{feat.get('voltage_min',0):.2f}-{feat.get('voltage_max',0):.2f}V, "
                f"Temp:{feat.get('temp_mean',25):.0f}C, {rul_info} Band:{static['label']}"}])
        txt = r.choices[0].message.content.strip()
        if txt.startswith("```"): txt=txt.split("\n",1)[1]
        if txt.endswith("```"): txt=txt[:-3]
        d = json.loads(txt.strip())
        return DisposalGuidanceResponse(
            band_id=static["band_id"], label=static["label"],
            soh_min=static["soh_min"], soh_max=static["soh_max"],
            summary=d.get("summary",static["summary"]),
            disposal_methods=d.get("disposal_methods",static["disposal_methods"]),
            analysis_precautions=d.get("analysis_precautions",static["analysis_precautions"]),
            transport_storage_precautions=d.get("transport_storage_precautions",
                                                 static["transport_storage_precautions"]),
            is_low_soh=soh<50)
    except Exception as e:
        print(f"AI disposal failed: {e}")
        return build_disposal(soh, feat, rul_est)


def llm_recommend(soh, feat, rules):
    if not openai_client: return None
    try:
        r = openai_client.chat.completions.create(
            model=openai_model, temperature=0.2, max_tokens=220,
            messages=[{"role":"system","content":
                "Battery maintenance advisor. 4-6 sentences, practical, safety-first."},
                {"role":"user","content":
                f"SOH:{soh:.1f}%, Features:{dict((k,round(float(feat.get(k,0)),3)) for k in feature_cols)}\n"
                f"Baseline:{rules}\nRefine into concise recommendation block."}])
        return r.choices[0].message.content.strip() if r.choices else None
    except: return None


# =================== Simulator (Digital Twin) ===================

def simulate_degradation(base_feat: dict, cpd: float, temp: float, horizon_days: int):
    a, b, c = rul_params['degradation_coeffs']
    eol_cap = rul_params['eol_capacity']

    # Temperature acceleration factor (Arrhenius-inspired)
    # Reference temp = 25C, every 10C doubles degradation rate
    temp_factor = 2.0 ** ((temp - 25.0) / 10.0)

    cap = base_feat['capacity']
    cycle = int(base_feat['cycle_number'])
    trajectory = []
    eol_day = None

    for day in range(0, horizon_days + 1, max(1, horizon_days // 100)):
        cycles_today = cpd * day
        current_cycle = cycle + int(cycles_today)
        rate = abs(2 * a * current_cycle + b) * temp_factor
        current_cap = cap - rate * cycles_today
        current_cap = max(0.5, current_cap)
        soh_val = (current_cap / rul_params['rated_capacity']) * 100
        soh_val = max(0, min(100, soh_val))
        trajectory.append(SimulatePoint(day=day, cycle=current_cycle,
                                        soh=round(soh_val, 2), capacity=round(current_cap, 4)))
        if current_cap <= eol_cap and eol_day is None:
            eol_day = day

    return trajectory, eol_day


# =================== PDF Report ===================

def _sanitize(text):
    """Replace unicode chars that latin-1 can't encode."""
    if not isinstance(text, str): return str(text)
    return (text.replace('\u2013', '-').replace('\u2014', '-')
                .replace('\u2018', "'").replace('\u2019', "'")
                .replace('\u201c', '"').replace('\u201d', '"')
                .replace('\u2026', '...').replace('\u00b0', 'deg')
                .replace('\u2022', '-').replace('\u00e2\u20ac\u201c', '-'))


def generate_pdf_report(pred_data: dict) -> bytes:
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    W = pdf.w - pdf.l_margin - pdf.r_margin

    def heading(text, size=12):
        pdf.set_font('Helvetica', 'B', size)
        pdf.cell(0, 8, _sanitize(text), ln=True)

    def body(text):
        pdf.set_font('Helvetica', '', 10)
        pdf.multi_cell(W, 6, _sanitize(text))

    def bullet(text):
        pdf.set_font('Helvetica', '', 10)
        pdf.multi_cell(W, 6, _sanitize(f"  - {text}"))

    # Title
    pdf.set_font('Helvetica', 'B', 18)
    pdf.cell(0, 12, 'Battery Health Diagnostic Report', ln=True, align='C')
    pdf.set_font('Helvetica', '', 9)
    pdf.cell(0, 6, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  |  '
                    f'ID: {pred_data.get("prediction_id", "N/A")}', ln=True, align='C')
    pdf.ln(4)

    soh = pred_data['soh']
    pdf.set_font('Helvetica', 'B', 16)
    pdf.set_fill_color(240, 248, 255)
    pdf.cell(0, 12, f'State of Health: {soh:.1f}%', ln=True, fill=True, align='C')
    pdf.ln(4)

    # Risk
    risk = pred_data.get('risk')
    if risk:
        heading(f'Risk Assessment: {risk["risk_level"]} ({risk["overall_risk"]}/100)')
        pdf.set_font('Helvetica', '', 9)
        pdf.cell(0, 5, f'Thermal: {risk["thermal_risk"]}  |  Voltage: {risk["voltage_risk"]}  |  '
                       f'Capacity: {risk["capacity_risk"]}  |  Cycle: {risk["cycle_risk"]}', ln=True)
        for f in risk.get('risk_factors', []):
            bullet(f)
        pdf.ln(3)

    # Uncertainty
    unc = pred_data.get('uncertainty')
    if unc:
        heading('Prediction Confidence')
        body(f'SOH: {unc["soh_lower"]:.1f}% - {unc["soh_upper"]:.1f}% '
             f'({unc["confidence_pct"]}% confidence interval)')
        pdf.ln(2)

    # RUL
    rul = pred_data.get('rul')
    if rul:
        heading('Remaining Useful Life')
        body(f'Days: {rul["remaining_days"]}  |  Cycles: {rul["remaining_cycles"]}  '
             f'|  EOL at {rul["eol_threshold"]:.0f}% SOH')
        pdf.set_font('Helvetica', 'I', 9)
        pdf.cell(0, 5, rul["confidence_note"], ln=True)
        pdf.ln(2)

    # SHAP
    shap_data = pred_data.get('shap_explanation')
    if shap_data:
        heading(f'SHAP Feature Analysis (baseline: {shap_data["base_value"]:.1f}%)')
        pdf.set_font('Helvetica', '', 9)
        for sf in shap_data['features']:
            sign = '+' if sf['shap_value'] >= 0 else ''
            pdf.cell(0, 5, f'  {sf["label"]}: {sf["value"]}  ->  {sign}{sf["shap_value"]:.3f}% impact', ln=True)
        pdf.ln(2)

    # Explanation
    expl = pred_data.get('explanation')
    if expl:
        heading('Analysis Summary')
        body(expl)
        pdf.ln(2)

    # Recommendations
    recs = pred_data.get('recommendations', [])
    if recs:
        heading('Recommendations')
        for r in recs:
            bullet(r)
        pdf.ln(2)

    # Disposal
    disp = pred_data.get('disposal_guidance')
    if disp:
        heading(f'Safety & Disposal ({_sanitize(disp["label"])})')
        body(disp.get('summary', ''))
        for section, key in [('Disposal', 'disposal_methods'),
                             ('Precautions', 'analysis_precautions'),
                             ('Transport', 'transport_storage_precautions')]:
            pdf.set_font('Helvetica', 'B', 9)
            pdf.cell(0, 6, f'{section}:', ln=True)
            for item in disp.get(key, []):
                bullet(item)

    pdf.ln(6)
    pdf.set_font('Helvetica', 'I', 8)
    pdf.cell(0, 5, 'Generated by Battery SOH Prediction System v3.0', ln=True, align='C')

    return pdf.output()


# =================== Endpoints ===================

@app.on_event("startup")
async def startup():
    load_all()
    init_llm()
    XATTN_SERVICE.load()  # best-effort; if it fails, predict_soh falls back to Ridge
    PHASE4_STATE.install(
        predict_fn=lambda feats: predict_soh(feats)[0],
        feature_labels=FEATURE_LABELS,
        openai_client=openai_client,
        openai_model=openai_model,
        llm_provider=llm_provider,
    )


@app.get("/", include_in_schema=False)
async def root():
    p = os.path.join(PROJECT_ROOT, 'frontend', 'index.html')
    return FileResponse(p) if os.path.exists(p) else {"message": "API v3", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="healthy", model_loaded=model is not None,
                          features_available=["predict","simulate","chat","report","generalization",
                                              "risk_score","uncertainty","shap"],
                          timestamp=datetime.now().isoformat(), version="3.0.0")


@app.get("/disposal/ranges")
async def disposal_ranges():
    return {"bands": all_bands_public()}


@app.get("/model/info")
async def model_info():
    predictor = {
        "active": "simple_ridge",  # /predict uses Ridge for reliable per-sample predictions
        "simple_ridge": {"model_type": type(model).__name__ if model else "N/A",
                          "features": feature_cols or []},
        "cross_attention": XATTN_SERVICE.info(),  # available via /predict/crossattention
        "cross_attention_endpoint": "/predict/crossattention",
    }
    return {"model_type": type(model).__name__ if model else "N/A",
            "features": feature_cols or [], "dataset": "NASA + Mendeley",
            "metrics": {"r2": 0.987, "mae": 0.17}, "version": "3.0.0",
            "predictor": predictor,
            "llm": {"provider": llm_provider, "model": openai_model if openai_client else None},
            "novel_features": ["shap_explainability","risk_scoring","uncertainty_bands",
                               "what_if_simulator","battery_doctor_chat","pdf_reports",
                               "cross_battery_generalization",
                               "cross_attention_multimodal","conformal_intervals",
                               "weibull_rul","retrieval_augmented_prognostics",
                               "physics_informed_loss"]}


# ---- Feature 7: Cross-Battery Generalization ----
@app.get("/model/generalization")
async def generalization():
    if not cross_battery_metrics:
        raise HTTPException(404, "Cross-battery metrics not available")
    return {"description": "Model trained on B0005, tested on unseen batteries (zero-shot transfer)",
            "batteries": cross_battery_metrics}


# ---- Main Predict (Features 5,6 + existing) ----
@app.post("/predict", response_model=PredictionResponse)
async def predict(req: PredictionRequest):
    start = time.time()
    if not model: raise HTTPException(503, "Model not loaded")

    feat = req.features.dict()
    soh, X_s = predict_soh(feat)

    pred_id = f"pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{int(soh*100)}"

    uncertainty = compute_uncertainty(X_s, soh)
    rul_est = compute_rul(feat['capacity'], int(feat['cycle_number']), req.features.charges_per_day)
    risk = compute_risk(feat, soh, rul_est)
    shap_exp = compute_shap(X_s, feat)
    explanation = generate_explanation(feat, soh, rul_est) if req.include_explanation else None
    recs = generate_recommendations(soh, feat)
    llm_rec = llm_recommend(soh, feat, recs)
    disposal = ai_disposal(soh, feat, rul_est)

    result = PredictionResponse(
        prediction_id=pred_id, soh=soh, confidence=0.95, uncertainty=uncertainty,
        risk=risk, rul=rul_est, explanation=explanation, shap_explanation=shap_exp,
        recommendations=recs, llm_recommendation=llm_rec, disposal_guidance=disposal,
        processing_time_ms=(time.time()-start)*1000)

    # Cache for PDF
    prediction_cache[pred_id] = result.dict()
    if len(prediction_cache) > 20:
        oldest = next(iter(prediction_cache))
        del prediction_cache[oldest]

    return result


# ---- Cross-attention (Phase 3 multimodal) explicit route ----
@app.post("/predict/crossattention")
async def predict_crossattention(req: PredictionRequest):
    if not XATTN_SERVICE.loaded:
        raise HTTPException(503, "Cross-attention model not loaded")
    feat = req.features.dict()
    soh = predict_soh_crossattention(feat)
    if soh is None:
        raise HTTPException(500, "Cross-attention inference failed")
    ridge_soh, _ = predict_soh_simple(feat)
    return {
        "soh_cross_attention": round(soh, 3),
        "soh_ridge": round(ridge_soh, 3),
        "delta": round(soh - ridge_soh, 3),
        "note": (
            "Cross-attention uses 18 features with KNN-imputation for the 12 that "
            "the UI doesn't collect. Individual-cycle error can reach 5-11% on the "
            "tail of the training distribution despite a 1.7% average MAE; the "
            "Ridge predictor is more reliable per-sample and is what /predict uses."
        ),
    }


# ---- Legacy / fast predictor (Ridge, 10ms) ----
@app.post("/predict/simple", response_model=PredictionResponse)
async def predict_simple_route(req: PredictionRequest):
    start = time.time()
    if not model:
        raise HTTPException(503, "Simple model not loaded")
    feat = req.features.dict()
    soh, X_s = predict_soh_simple(feat)
    pred_id = f"pred_simple_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{int(soh*100)}"
    uncertainty = compute_uncertainty(X_s, soh)
    rul_est = compute_rul(feat['capacity'], int(feat['cycle_number']), req.features.charges_per_day)
    risk = compute_risk(feat, soh, rul_est)
    recs = generate_recommendations(soh, feat)
    result = PredictionResponse(
        prediction_id=pred_id, soh=soh, confidence=0.9,
        uncertainty=uncertainty, risk=risk, rul=rul_est,
        explanation=generate_explanation(feat, soh, rul_est) if req.include_explanation else None,
        recommendations=recs,
        processing_time_ms=(time.time()-start)*1000,
    )
    return result


# ---- Feature 1: What-If Simulator ----
@app.post("/simulate", response_model=SimulateResponse)
async def simulate(req: SimulateRequest):
    if not rul_params: raise HTTPException(503, "RUL params not loaded")
    base = req.base_features.dict()
    scenarios = []
    for sc in req.scenarios:
        cpd = sc.get('charges_per_day', req.base_features.charges_per_day)
        temp = sc.get('temp_mean', req.base_features.temp_mean)
        label = sc.get('label', f'{cpd}x/day at {temp}C')
        traj, eol = simulate_degradation(base, cpd, temp, req.horizon_days)
        scenarios.append(SimulateScenario(label=label, charges_per_day=cpd,
                                          temp_mean=temp, trajectory=traj, eol_day=eol))
    return SimulateResponse(scenarios=scenarios)


# ---- Feature 3: Battery Doctor Chat ----
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    # Build battery context
    ctx = ""
    if req.battery_context:
        f = req.battery_context.dict()
        soh, _ = predict_soh(f)
        rul_est = compute_rul(f['capacity'], int(f['cycle_number']), req.battery_context.charges_per_day)
        risk = compute_risk(f, soh, rul_est)
        ctx = (f"Battery state: SOH={soh:.1f}%, Capacity={f['capacity']:.3f}Ah, "
               f"Cycles={f['cycle_number']}, Voltage={f['voltage_min']:.2f}-{f['voltage_max']:.2f}V, "
               f"Temp={f['temp_mean']:.0f}C, Discharge={f['discharge_time']:.0f}s, "
               f"Risk={risk.risk_level} ({risk.overall_risk}/100)")
        if rul_est:
            ctx += f", Days remaining={rul_est.remaining_days}, Cycles remaining={rul_est.remaining_cycles}"

    if openai_client:
        try:
            r = openai_client.chat.completions.create(
                model=openai_model, temperature=0.4, max_tokens=400,
                messages=[
                    {"role": "system", "content":
                     "You are Battery Doctor, an AI expert on lithium-ion batteries. "
                     "You answer questions about battery health, safety, charging, storage, "
                     "degradation, and replacement. Be helpful, specific, and safety-conscious. "
                     "If battery data is provided, reference it in your answers. "
                     "End with 1-2 follow-up question suggestions."},
                    {"role": "user", "content":
                     f"{'[Battery Data: '+ctx+']' if ctx else ''}\n\nUser question: {req.message}"}
                ])
            reply = r.choices[0].message.content.strip()
            return ChatResponse(reply=reply, suggestions=[
                "What can I do to extend battery life?",
                "Is my battery safe for travel?",
                "When should I replace it?"])
        except Exception as e:
            print(f"Chat LLM error: {e}")

    # Fallback: rule-based responses
    msg = req.message.lower()
    if any(w in msg for w in ['safe','danger','risk']):
        reply = f"Based on your battery data{' ('+ctx+')' if ctx else ''}, "
        reply += "ensure you follow standard safety protocols. Monitor for swelling, unusual heat, or voltage drops."
    elif any(w in msg for w in ['replace','new','buy']):
        reply = "Consider replacement when SOH drops below 70-80%. "
        if ctx: reply += f"Your current battery: {ctx}"
    elif any(w in msg for w in ['charge','charg']):
        reply = "For Li-ion batteries, charge between 20-80% SOH for longevity. Avoid extreme temperatures while charging."
    elif any(w in msg for w in ['store','storage']):
        reply = "Store at 40-60% charge in a cool, dry place (15-25C). Avoid full charge for long storage."
    else:
        reply = f"I can help with battery health questions. {'Your battery: '+ctx if ctx else 'Add your battery data for personalized advice.'}"

    return ChatResponse(reply=reply, suggestions=[
        "Is my battery safe to use?", "How can I extend battery life?", "When should I replace it?"])


# ---- Feature 2: PDF Report ----
@app.get("/report/pdf/{prediction_id}")
async def get_pdf_report(prediction_id: str):
    if prediction_id not in prediction_cache:
        raise HTTPException(404, "Prediction not found. Make a prediction first.")
    pdf_bytes = generate_pdf_report(prediction_cache[prediction_id])
    return StreamingResponse(
        io.BytesIO(pdf_bytes), media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=battery_report_{prediction_id}.pdf"})


# =====================================================================
# Endpoints added for the v4 frontend (Phase-5 dashboard, CSV upload,
# SSE simulator stream, dataset trajectory, about/model card, badge).
# All are read-mostly and additive; the v3 contract above is unchanged.
# =====================================================================


@app.get("/api/about")
async def about():
    """Model card + project metadata for the About modal in the UI."""
    import platform
    info: Dict[str, Any] = {
        "name": "Battery SOH Predictor",
        "version": "4.0.0",
        "description": ("Physics-guided multimodal lithium-ion State-of-Health "
                        "estimation with conformal intervals, Weibull RUL, "
                        "retrieval-augmented diagnostics, and a Phase-5 "
                        "rigor layer."),
        "model": {
            "production": type(model).__name__ if model else "N/A",
            "n_features_production": int(getattr(model, "n_features_in_", 0)) if model else 0,
            "bootstrap_models": len(bootstrap_models) if bootstrap_models else 0,
            "cross_attention_loaded": bool(getattr(XATTN_SERVICE, "loaded", False)),
            "llm_provider": llm_provider,
            "llm_model": openai_model if openai_client else None,
        },
        "datasets": {
            "nasa_b0005_b0006_b0007_cycles": 504,
            "mendeley_cycles": 2738,
            "test_split_cycles": 336,
        },
        "metrics_headline": {
            "ridge_19feat_test_mae_pct": 0.90,
            "ridge_19feat_test_rmse_pct": 1.21,
            "ridge_19feat_test_r2": 0.96,
            "phase3_xattn_r2": 0.9595,
            "phase3_xattn_mae_pct": 3.33,
        },
        "host": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
        },
        "phase5_modules": [
            "leakage", "calibration", "weighted_conformal", "survival_synth",
            "causal_shap", "counterfactuals", "failure_gallery", "carbon",
        ],
    }
    return info


@app.get("/api/phase5/manifest")
async def phase5_manifest():
    """Serve the Phase-5 rigor-suite manifest so the dashboard tab can render it."""
    p = os.path.join(PROJECT_ROOT, "results", "phase5", "manifest.json")
    if not os.path.exists(p):
        raise HTTPException(404, "Phase-5 results not found. Run `make rigor` first.")
    with open(p, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    # Inline the per-step JSON sidecars so the UI only needs one fetch.
    detail: Dict[str, Any] = {}
    for key in manifest.get("steps", {}):
        sidecar = os.path.join(PROJECT_ROOT, "results", "phase5", f"{key}.json")
        if os.path.exists(sidecar):
            try:
                with open(sidecar, "r", encoding="utf-8") as f:
                    detail[key] = json.load(f)
            except Exception:
                detail[key] = {"error": "could not read sidecar"}
    return {"manifest": manifest, "detail": detail}


@app.get("/docs/phase6-report")
async def phase6_report():
    """Serve PHASE6_DEFERRED_REPORT.md as plain text for the Rigor tab link."""
    p = os.path.join(PROJECT_ROOT, "PHASE6_DEFERRED_REPORT.md")
    if not os.path.exists(p):
        raise HTTPException(404, "PHASE6_DEFERRED_REPORT.md not found")
    return FileResponse(p, media_type="text/markdown; charset=utf-8")


@app.get("/api/phase6/manifest")
async def phase6_manifest():
    """Serve the Phase-6 deferred-suite manifest for the dashboard tab."""
    p = os.path.join(PROJECT_ROOT, "results", "phase6", "manifest.json")
    if not os.path.exists(p):
        raise HTTPException(404, "Phase-6 results not found. Run `make deferred` first.")
    with open(p, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    detail: Dict[str, Any] = {}
    for key in manifest.get("modules", {}):
        sidecar = os.path.join(PROJECT_ROOT, "results", "phase6", f"{key}.json")
        if os.path.exists(sidecar):
            try:
                with open(sidecar, "r", encoding="utf-8") as f:
                    detail[key] = json.load(f)
            except Exception:
                detail[key] = {"error": "could not read sidecar"}
    return {"manifest": manifest, "detail": detail}


@app.get("/api/phase5/figure/{name}")
async def phase5_figure(name: str):
    """Serve the Phase-5 PNG outputs (reliability_diagram, failure_gallery)."""
    safe = "".join(c for c in name if c.isalnum() or c in {"_", "-"})
    p = os.path.join(PROJECT_ROOT, "results", "phase5", f"{safe}.png")
    if not os.path.exists(p):
        raise HTTPException(404, f"Figure {safe}.png not found")
    return FileResponse(p, media_type="image/png")


@app.get("/api/dataset/trajectory")
async def dataset_trajectory(battery: str = "B0005"):
    """Per-cycle SOH trajectory for the timeline-scrubber tab."""
    import pandas as pd
    p = os.path.join(PROJECT_ROOT, "data", "phase2", "harmonized", "all_features.csv")
    if not os.path.exists(p):
        raise HTTPException(404, "Harmonized features CSV missing")
    df = pd.read_csv(p)
    bid = battery.strip()
    if bid not in df["battery_id"].astype(str).unique():
        available = sorted(df["battery_id"].astype(str).unique().tolist())
        raise HTTPException(404, f"Battery {bid!r} not found; have {available}")
    g = df[df["battery_id"].astype(str) == bid].sort_values("cycle_number")
    rows = []
    for _, r in g.iterrows():
        rows.append({
            "cycle": int(r["cycle_number"]),
            "soh": float(r["soh"]),
            "capacity": float(r["capacity"]),
            "voltage_min": float(r["voltage_min"]),
            "voltage_max": float(r["voltage_max"]),
            "temp_mean": float(r["temp_mean"]),
            "discharge_time": float(r["discharge_time"]),
        })
    return {"battery_id": bid, "n_cycles": len(rows), "rows": rows}


@app.get("/api/dataset/list")
async def dataset_list():
    """Available battery IDs for the trajectory scrubber."""
    import pandas as pd
    p = os.path.join(PROJECT_ROOT, "data", "phase2", "harmonized", "all_features.csv")
    if not os.path.exists(p):
        return {"batteries": []}
    df = pd.read_csv(p)
    out = []
    for bid, g in df.groupby("battery_id"):
        out.append({
            "battery_id": str(bid),
            "n_cycles": int(len(g)),
            "soh_initial": float(g["soh"].iloc[0]),
            "soh_final": float(g["soh"].iloc[-1]),
        })
    return {"batteries": out}


@app.get("/api/ica/{battery}/{cycle}")
async def ica_curve(battery: str, cycle: int):
    """Reconstruct an approximate dQ/dV (incremental capacity) curve for a cycle."""
    import pandas as pd
    p = os.path.join(PROJECT_ROOT, "data", "phase2", "harmonized", "all_features.csv")
    if not os.path.exists(p):
        raise HTTPException(404, "Harmonized features CSV missing")
    df = pd.read_csv(p)
    bid = battery.strip()
    sub = df[(df["battery_id"].astype(str) == bid) & (df["cycle_number"] == cycle)]
    if sub.empty:
        raise HTTPException(404, f"({bid}, {cycle}) not found")
    row = sub.iloc[0]
    # Rebuild a smooth synthetic curve anchored on the per-cycle ICA features
    # the harmoniser actually persists. This is a *reconstruction* for display,
    # not the raw dQ/dV signal (we only ship features, not raw cycle traces).
    v_min = float(row["voltage_min"])
    v_max = float(row["voltage_max"])
    v_peak = float(row["ica_peak_voltage"])
    peak_mag = float(row["ica_peak_magnitude"])
    width = max(0.04, 0.10 * (v_max - v_min))
    voltages = np.linspace(v_min, v_max, 80)
    base = float(row["ica_mean"])
    curve = base + peak_mag * np.exp(-((voltages - v_peak) ** 2) / (2 * width ** 2))
    return {
        "battery_id": bid,
        "cycle": int(cycle),
        "voltage": voltages.tolist(),
        "dqdv": curve.tolist(),
        "peak_voltage": v_peak,
        "peak_magnitude": peak_mag,
        "soh": float(row["soh"]),
        "note": ("Reconstructed from cached ICA features. "
                 "For the original raw curve re-run the segmentation on .mat data."),
    }


def _csv_to_features(content_text: str) -> List[Dict[str, Any]]:
    """Parse an uploaded CSV into the feature-dict list our predictor consumes."""
    import csv
    rows: List[Dict[str, Any]] = []
    reader = csv.DictReader(io.StringIO(content_text))
    if not reader.fieldnames:
        return rows
    # Aliases for forgiving column matching
    alias = {
        "capacity": ["capacity", "capacity_ah", "cap_ah"],
        "cycle_number": ["cycle_number", "cycle", "cycle_count"],
        "voltage_min": ["voltage_min", "v_min"],
        "voltage_max": ["voltage_max", "v_max"],
        "temp_mean": ["temp_mean", "temperature_c", "temp_c"],
        "discharge_time": ["discharge_time", "discharge_seconds", "discharge_s"],
        "charges_per_day": ["charges_per_day", "cpd"],
    }
    cols_lower = {c.lower(): c for c in reader.fieldnames}
    def pick(keys: List[str]) -> Optional[str]:
        for k in keys:
            if k.lower() in cols_lower:
                return cols_lower[k.lower()]
        return None
    pickers = {std: pick(opts) for std, opts in alias.items()}
    for raw in reader:
        try:
            row = {
                "capacity": float(raw[pickers["capacity"]]) if pickers["capacity"] else 1.8,
                "cycle_number": int(float(raw[pickers["cycle_number"]])) if pickers["cycle_number"] else 0,
                "voltage_min": float(raw[pickers["voltage_min"]]) if pickers["voltage_min"] else 2.7,
                "voltage_max": float(raw[pickers["voltage_max"]]) if pickers["voltage_max"] else 4.2,
                "temp_mean": float(raw[pickers["temp_mean"]]) if pickers["temp_mean"] else 32.0,
                "discharge_time": float(raw[pickers["discharge_time"]]) if pickers["discharge_time"] else 3300.0,
                "charges_per_day": float(raw[pickers["charges_per_day"]]) if pickers["charges_per_day"] else 1.0,
            }
            rows.append(row)
        except (ValueError, TypeError, KeyError):
            continue
    return rows


@app.post("/predict/csv")
async def predict_csv(payload: Dict[str, Any]):
    """Batch-predict from CSV text. Accepts {csv_text: "..."} JSON body."""
    if not model:
        raise HTTPException(503, "Model not loaded")
    text = payload.get("csv_text") or payload.get("csv") or ""
    if not isinstance(text, str) or not text.strip():
        raise HTTPException(400, "Body must include non-empty csv_text")
    rows = _csv_to_features(text)
    if not rows:
        raise HTTPException(400, "No valid rows parsed from CSV. Need at least 'capacity' and 'cycle_number'.")
    if len(rows) > 500:
        rows = rows[:500]
    out: List[Dict[str, Any]] = []
    for feat in rows:
        soh, X_s = predict_soh(feat)
        rul_est = compute_rul(feat["capacity"], int(feat["cycle_number"]),
                              feat.get("charges_per_day", 1.0))
        risk = compute_risk(feat, soh, rul_est)
        out.append({
            "input": feat,
            "soh": float(soh),
            "rul_days": rul_est.get("days") if isinstance(rul_est, dict) else None,
            "rul_cycles": rul_est.get("cycles") if isinstance(rul_est, dict) else None,
            "risk_band": risk.get("band") if isinstance(risk, dict) else None,
            "risk_score": risk.get("score") if isinstance(risk, dict) else None,
        })
    summary = {
        "n": len(out),
        "soh_mean": float(np.mean([r["soh"] for r in out])),
        "soh_min": float(np.min([r["soh"] for r in out])),
        "soh_max": float(np.max([r["soh"] for r in out])),
    }
    return {"summary": summary, "rows": out}


@app.get("/simulate/stream")
async def simulate_stream(
    cpd: float = 1.0,
    temp: float = 25.0,
    horizon_days: int = 365,
    initial_soh: float = 100.0,
):
    """Server-Sent Events stream of a degradation simulation, one frame per cycle."""
    def event_iter():
        yield ("event: meta\n"
               f"data: {json.dumps({'cpd': cpd, 'temp': temp, 'horizon_days': horizon_days})}\n\n")
        cycles_total = max(1, int(horizon_days * cpd))
        # Simple thermal-accelerated capacity-fade model: percent-per-cycle
        # depending on temperature (Arrhenius-ish), bounded so the demo doesn't
        # collapse to zero immediately.
        base_loss = 0.02 + max(0.0, (temp - 25.0)) * 0.002
        soh = initial_soh
        for k in range(cycles_total):
            # loss per cycle drifts up as soh falls (knee acceleration)
            knee = 1.0 + max(0.0, (90 - soh) / 30.0) * 0.6
            soh = max(50.0, soh - base_loss * knee)
            payload = {"cycle": k + 1, "day": (k + 1) / max(cpd, 1e-3), "soh": round(soh, 3)}
            yield f"data: {json.dumps(payload)}\n\n"
            if k % 5 == 0:
                time.sleep(0.005)  # keep frames small but smooth
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(event_iter(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/badge/{soh_int}.svg")
async def health_badge(soh_int: int):
    """Shareable SVG health badge (right-click save)."""
    soh = max(0, min(100, int(soh_int)))
    if soh >= 90:
        color, label = "#10b981", "EXCELLENT"
    elif soh >= 80:
        color, label = "#3b82f6", "GOOD"
    elif soh >= 70:
        color, label = "#f59e0b", "MODERATE"
    elif soh >= 60:
        color, label = "#f97316", "POOR"
    else:
        color, label = "#ef4444", "CRITICAL"
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="220" height="44">'
        '<linearGradient id="bg" x1="0" x2="0" y1="0" y2="1">'
        '<stop offset="0" stop-color="#fff" stop-opacity=".3"/>'
        '<stop offset="1" stop-opacity=".3"/></linearGradient>'
        '<mask id="m"><rect width="220" height="44" rx="6" fill="#fff"/></mask>'
        '<g mask="url(#m)">'
        '<rect width="100" height="44" fill="#374151"/>'
        f'<rect x="100" width="120" height="44" fill="{color}"/>'
        '<rect width="220" height="44" fill="url(#bg)"/></g>'
        '<g fill="#fff" font-family="Verdana,Geneva,sans-serif" font-size="13" '
        'font-weight="600" text-anchor="middle">'
        '<text x="50" y="20">Battery SOH</text>'
        f'<text x="50" y="34" font-size="10" font-weight="400">PHASE-5 v4</text>'
        f'<text x="160" y="20">{soh}%</text>'
        f'<text x="160" y="34" font-size="10" font-weight="400">{label}</text>'
        '</g></svg>'
    )
    return StreamingResponse(io.BytesIO(svg.encode("utf-8")), media_type="image/svg+xml")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
