"""
SOH-based disposal routes, transport handling, and laboratory/field analysis precautions.

Guidance is informational; always follow local regulations, OEM instructions, and SDS.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class SOHBand:
    band_id: str
    label: str
    soh_min: float
    soh_max: float
    summary: str
    disposal_methods: Tuple[str, ...]
    analysis_precautions: Tuple[str, ...]
    transport_storage_precautions: Tuple[str, ...]


# Ordered from highest to lowest SOH; first matching band wins.
SOH_BANDS: Tuple[SOHBand, ...] = (
    SOHBand(
        band_id="excellent",
        label="Excellent (90–100%)",
        soh_min=90.0,
        soh_max=100.0,
        summary="Healthy cell; normal service life and standard end-of-life recycling paths.",
        disposal_methods=(
            "Return to manufacturer or retailer take-back where available.",
            "Drop at municipal or retailer e-waste / battery recycling bins for Li-ion.",
            "Do not place in general household trash or metal scrap.",
        ),
        analysis_precautions=(
            "Use standard lab PPE (safety glasses, nitrile gloves) and a ventilated bench.",
            "Verify polarity and current limits before cycling; use a battery-specific test rig.",
            "Log temperature during tests; stop if cell exceeds OEM temperature limits.",
        ),
        transport_storage_precautions=(
            "Ship at partial state-of-charge per carrier rules; pad terminals against short circuits.",
            "Store in a cool, dry place away from combustibles.",
        ),
    ),
    SOHBand(
        band_id="good",
        label="Good (80–90%)",
        soh_min=80.0,
        soh_max=90.0,
        summary="Still fit for primary use; plan for certified recycling at retirement.",
        disposal_methods=(
            "Prefer certified electronics / Li-ion battery recyclers over landfill.",
            "Trade-in or refurbishment programs if voltage and impedance remain stable.",
            "If removed from service, tape terminals and use original or fire-resistant packaging.",
        ),
        analysis_precautions=(
            "Inspect for swelling, vent rupture, or electrolyte odor before any load test.",
            "Use fused, current-limited equipment; avoid parallel unknown cells.",
            "Compare capacity fade trend across cycles rather than a single snapshot.",
        ),
        transport_storage_precautions=(
            "Label as used Li-ion; comply with local rules for used battery shipment.",
            "Avoid stacking heavy objects on packs; prevent case deformation.",
        ),
    ),
    SOHBand(
        band_id="moderate",
        label="Moderate (70–80%)",
        soh_min=70.0,
        soh_max=80.0,
        summary="Accelerated aging; tighter monitoring and earlier recycling planning.",
        disposal_methods=(
            "Schedule retirement to a certified Li-ion recycler within your maintenance plan.",
            "Second-life (e.g. stationary storage) only after impedance and thermal screening by qualified staff.",
            "Do not mix with new cells in the same string without a full BMS review.",
        ),
        analysis_precautions=(
            "Increase thermal monitoring frequency; watch for hot spots during charge/discharge.",
            "Reduce max C-rate versus new-cell guidelines; shorten test duration if temperature rises.",
            "Document resting voltage drift; unstable OCV may indicate imbalance or side reactions.",
        ),
        transport_storage_precautions=(
            "Store separately from healthy stock; clearly mark as degraded.",
            "Keep fire suppression suitable for Class B/C fires accessible in the test area.",
        ),
    ),
    SOHBand(
        band_id="degraded",
        label="Degraded (50–70%)",
        soh_min=50.0,
        soh_max=70.0,
        summary="Significant fade; prioritize safety over performance extraction in testing.",
        disposal_methods=(
            "Route to a certified battery recycling or hazardous-waste facility—never general landfill.",
            "If regulations require, bring pack to low state-of-charge using approved equipment before shipping.",
            "Retain chain-of-custody paperwork where required for industrial packs.",
        ),
        analysis_precautions=(
            "Run tests in a fire-rated or well-ventilated area with no unattended high-power cycling.",
            "Use insulated tools; pre-check for soft casing, vent noise, or voltage collapse under light load.",
            "Avoid destructive teardown unless in a qualified lab with thermal runaway controls.",
        ),
        transport_storage_precautions=(
            "Pad and isolate terminals; use UN-compliant packaging when mandated for damaged or large packs.",
            "Quarantine from flammables; post warning signage in storage.",
        ),
    ),
    SOHBand(
        band_id="poor",
        label="Poor (30–50%)",
        soh_min=30.0,
        soh_max=50.0,
        summary="Near end-of-life; high thermal and electrical risk during misuse.",
        disposal_methods=(
            "Hand off to professional Li-ion / hazardous-waste handlers; do not attempt home disposal.",
            "Do not incinerate or crush; follow local rules for Class 9 (lithium battery) transport if applicable.",
            "If the pack is damaged, consult emergency response or specialist disposal immediately.",
        ),
        analysis_precautions=(
            "Minimize stress tests; prefer low-rate characterization with continuous temperature watch.",
            "Work with a second person for high-energy packs; keep Class D dry powder or approved media nearby per policy.",
            "Do not fast-charge or fast-discharge without engineering sign-off and containment.",
        ),
        transport_storage_precautions=(
            "Treat as high-risk: shorted terminals can ignite; use non-conductive terminal covers.",
            "Store in a metal cabinet or sand-filled container if required by site safety policy.",
        ),
    ),
    SOHBand(
        band_id="critical",
        label="Critical (0–30%)",
        soh_min=0.0,
        soh_max=30.0,
        summary="Assume unstable; prioritize personnel safety over data collection.",
        disposal_methods=(
            "Use only licensed hazardous e-waste / battery recycling with Li-ion capability.",
            "Never discard in household bins, metal recycling, or open-air piles.",
            "Follow incident-specific guidance for bloated, hissing, or smoking cells (evacuate, emergency services).",
        ),
        analysis_precautions=(
            "Avoid cycling unless in a controlled facility; skip destructive analysis on vented or leaking cells.",
            "Do not pierce, twist, or hammer cells; assume flammable gas and corrosive electrolyte risk.",
            "If any test is required, use minimal energy throughput and abort on any temperature or voltage anomaly.",
        ),
        transport_storage_precautions=(
            "Do not ship damaged or smoking packs through normal carriers—use emergency procedures.",
            "Isolate in a fire-resistant container; keep away from personnel traffic until collected by professionals.",
        ),
    ),
)


def resolve_band(soh: float) -> SOHBand:
    """Return the guidance band for a given SOH percentage (0–100).

    Boundaries are top-down so 90% maps to Excellent, 30% to Poor, etc.
    """
    s = max(0.0, min(100.0, float(soh)))
    if s >= 90:
        return SOH_BANDS[0]
    if s >= 80:
        return SOH_BANDS[1]
    if s >= 70:
        return SOH_BANDS[2]
    if s >= 50:
        return SOH_BANDS[3]
    if s >= 30:
        return SOH_BANDS[4]
    return SOH_BANDS[5]


def guidance_dict(soh: float) -> Dict[str, Any]:
    """Structured dict suitable for JSON / Pydantic responses."""
    s = max(0.0, min(100.0, float(soh)))
    b = resolve_band(soh)
    return {
        "band_id": b.band_id,
        "label": b.label,
        "soh_min": b.soh_min,
        "soh_max": b.soh_max,
        "summary": b.summary,
        "disposal_methods": list(b.disposal_methods),
        "analysis_precautions": list(b.analysis_precautions),
        "transport_storage_precautions": list(b.transport_storage_precautions),
        "is_low_soh": s < 50.0,
    }


def all_bands_public() -> List[Dict[str, Any]]:
    """All bands for documentation endpoints (no computed fields)."""
    out: List[Dict[str, Any]] = []
    for b in SOH_BANDS:
        out.append(
            {
                "band_id": b.band_id,
                "label": b.label,
                "soh_min": b.soh_min,
                "soh_max": b.soh_max,
                "summary": b.summary,
                "disposal_methods": list(b.disposal_methods),
                "analysis_precautions": list(b.analysis_precautions),
                "transport_storage_precautions": list(b.transport_storage_precautions),
            }
        )
    return out
