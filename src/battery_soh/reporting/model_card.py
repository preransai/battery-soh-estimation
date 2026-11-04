"""Markdown model cards rendered from production bundle documents.

The layout follows Mitchell et al., "Model Cards for Model Reporting" (FAT*
2019): model details, intended use, factors/data, metrics, evaluation,
caveats and ethical considerations. Every number in a card is read from the
bundle JSON it describes (written by :func:`battery_soh.training.train_bundles`),
so a card cannot drift from its artifact. Statements about individual cells
(under-covered cells, extrapolation, a feature with a cell-specific offset)
are generated from the recorded numbers, not written by hand.

This module only formats text: it imports neither matplotlib nor scikit-learn.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Final

from battery_soh.config import DEFAULT_N_REFERENCE_CYCLES
from battery_soh.data.protocols import PROTOCOLS
from battery_soh.features.registry import FEATURES
from battery_soh.features.relative import is_context_feature

__all__ = ["render_model_card"]

_PROCEDURES: Final[Mapping[str, str]] = {
    "ridge": "Ridge regression on every input of the feature set",
    "ridge_nfs": "Ridge regression with nested backward feature selection and penalty choice",
}
_DISCHARGE_WINDOW: Final = "a discharge whose loaded voltage starts above 3.80 V and reaches 3.50 V"
#: What a deployment must record for each input to be measurable.
_REQUIREMENTS: Final[Mapping[str, str]] = {
    "chg_cc_time_390_419_s": "a CC charge that crosses 3.90 V (starts below about 3.85 V)",
    "chg_temp_rise_cc_c": "a CC charge that crosses 3.90 V (starts below about 3.85 V)",
    "chg_cc_time_400_419_s": "a CC charge that crosses 4.00 V (starts below about 3.97 V)",
    "chg_ica_peak_height_ah_per_v": "a CC charge covering 3.90-4.18 V",
    "chg_ica_peak_voltage_v": "a CC charge covering 3.90-4.18 V",
    "chg_cv_time_s": "a complete CV phase (until the current falls below 0.1 A)",
    "dis_q_380_350_ah": _DISCHARGE_WINDOW,
    "dis_time_380_350_s": _DISCHARGE_WINDOW,
    "dis_temp_rise_win_c": _DISCHARGE_WINDOW,
    "dis_r0_ohm": "the load step at the start of a discharge",
    "eis_re_ohm": "an impedance (EIS) measurement",
    "eis_rct_ohm": "an impedance (EIS) measurement",
}
_DISCHARGE_MODES: Final[Mapping[str, str]] = {
    "cc": "constant-current",
    "square_wave": "square-wave",
    "mixed": "mixed-load",
}
_LEVEL: Final = "0.9"


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None or isinstance(value, bool):
        return "n/a"
    number = float(value)
    return f"{number:.{digits}f}" if math.isfinite(number) else "n/a"


def _signed(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):+.{digits}f}"


def _pct(value: Any, digits: int = 0) -> str:
    if value is None:
        return "n/a"
    number = float(value)
    return f"{100.0 * number:.{digits}f}%" if math.isfinite(number) else "n/a"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]], align: str = "") -> list[str]:
    """Markdown table; ``align`` holds one ``l``/``r`` per column (default left)."""
    spec = align.ljust(len(header), "l")
    out = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---:" if a == "r" else "---" for a in spec) + " |",
    ]
    out += ["| " + " | ".join(row) + " |" for row in rows]
    return out


class _Card:
    """Read-only view of the bundle fields a card needs (missing fields become empty)."""

    def __init__(self, bundle: Mapping[str, Any]) -> None:
        self.bundle = bundle
        self.model_id = str(bundle["model_id"])
        self.relative = str(bundle["mode"]) == "relative"
        self.features = [str(f) for f in bundle["features"]]
        self.estimator: Mapping[str, Any] = bundle["estimator"]
        self.metrics: Mapping[str, Any] = bundle.get("metrics") or {}
        self.per_battery: Mapping[str, Any] = self.metrics.get("per_battery") or {}
        self.selection: Mapping[str, Any] = bundle.get("selection") or {}
        self.protocol: Mapping[str, Any] = bundle.get("training_protocol") or {}
        self.provenance: Mapping[str, Any] = bundle.get("provenance") or {}
        self.training: Mapping[str, Any] = bundle.get("training") or {}
        self.domain: Mapping[str, Any] = bundle.get("domain") or {}
        self.spec: Mapping[str, Any] = self.metrics.get("spec") or {}
        self.batteries = [str(b) for b in self.training.get("batteries", [])]
        self.n_ref = int(bundle.get("n_reference_cycles", DEFAULT_N_REFERENCE_CYCLES))
        self.ranges: Mapping[str, Sequence[Any]] = (
            self.protocol.get("label_range_per_battery") or {}
        )


def _header(card: _Card) -> list[str]:
    scope = " and divided by the cell's commissioning reference" if card.relative else ""
    procedure = str(card.spec.get("model", card.protocol.get("procedure", "ridge")))
    candidates = card.selection.get("candidates", card.features)
    mode = (
        f"relative: each input divided by the mean of the cell's first {card.n_ref} QC-valid values"
        if card.relative
        else "absolute: raw indicator values"
    )
    rows = [
        ["Model id", f"`{card.model_id}`"],
        ["Procedure", _PROCEDURES.get(procedure, procedure)],
        [
            "Exported estimator",
            "median imputation -> standardisation -> ridge "
            f"(alpha = {_fmt(card.estimator.get('alpha'), 3)})",
        ],
        ["Candidate inputs", f"feature set `{card.bundle['feature_set']}` ({len(candidates)})"],
        ["Inputs used", f"{len(card.features)}: " + ", ".join(f"`{f}`" for f in card.features)],
        ["Mode", mode],
        ["Output", f"SOH in %, clipped to {card.bundle.get('output_clip', [0, 120])}"],
        [
            "Timestamp (UTC)",
            f"{card.bundle.get('created_utc') or 'n/a'} (deterministic: the feature table's "
            "build time or a fixed config value, not when `soh train` ran)",
        ],
        ["Package version", str(card.bundle.get("package_version", ""))],
        ["Artifact", f"`{card.model_id}.json` (SHA-256 in `MANIFEST.json`)"],
    ]
    return [
        f"# Model card: `{card.model_id}`",
        "",
        "<!-- Generated by `soh train` from the bundle JSON; do not edit by hand. -->",
        "",
        "Estimates the state of health (SOH, % of the cell's own initial capacity) of an "
        "18650 Li-ion cell from partial-window health indicators measured during routine "
        f"operation{scope}. The capacity itself, or any statistic of a complete discharge, "
        "is never an input.",
        "",
        "## Model details",
        "",
        *_table(["Field", "Value"], rows),
    ]


def _conditions(batteries: Sequence[str]) -> str:
    labels = []
    for b in batteries:
        proto = PROTOCOLS.get(b)
        if proto is not None:
            kind = _DISCHARGE_MODES.get(proto.discharge_mode, proto.discharge_mode)
            labels.append(
                f"{proto.discharge_current_a:g} A {kind} discharge, {proto.ambient_c:g} degC"
            )
    return "; ".join(dict.fromkeys(labels)) or "see the data README"


def _use(card: _Card) -> list[str]:
    return [
        "",
        "## Intended use",
        "",
        f"- Research and demonstration on laboratory cells like the training cells "
        f"({_conditions(card.batteries)}), where a partial charge and/or a partial discharge "
        "window is recorded but a full capacity test is not.",
        "- An auditable, leakage-free baseline with battery-level conformal intervals whose "
        "coverage is measured on held-out cells, not guaranteed, and an explicit "
        "domain-of-applicability check.",
        "",
        "## Not intended for",
        "",
        "- Safety-critical decisions (charge or discharge limits, thermal protection, warranty "
        "or second-life grading). This is not a safety-certified battery-management function.",
        "- Fleet or in-service health monitoring without first validating on the target cells: "
        "a handful of laboratory cells cannot establish field accuracy.",
        "- Other chemistries, form factors, temperatures, load currents or test rigs: the "
        "training data cover one cell type under one protocol. Inputs outside the training "
        "envelope are flagged (`domain.in_domain = false`), but in-range inputs from another "
        "population are not detected.",
        "- Remaining-useful-life forecasting: the model estimates current SOH only.",
    ]


def _data(card: _Card) -> list[str]:
    excluded = (
        f", {card.protocol.get('n_reference_cycles_excluded', 0)} reference cycles excluded"
        if card.relative
        else ""
    )
    lines = [
        "",
        "## Training data",
        "",
        f"- Source: NASA Prognostics Center of Excellence Battery Data Set, cells "
        f"{', '.join(card.batteries)} ({_conditions(card.batteries)}).",
        f"- Rows: {card.protocol.get('rows', 'QC-valid cycles')}; "
        f"{card.training.get('n_cycles', 'n/a')} cycles used for fitting{excluded}.",
        f"- Feature table SHA-256: `{card.training.get('features_sha256', 'n/a')}`.",
    ]
    if card.ranges:
        cycles = card.protocol.get("cycles_per_battery") or {}
        lines += ["", "SOH range of each training cell (fitted cycles):", ""]
        lines += _table(
            ["Cell", "Cycles", "SOH min (%)", "SOH max (%)"],
            [
                [b, str(cycles.get(b, "")), _fmt(r[0], 1), _fmt(r[1], 1)]
                for b, r in card.ranges.items()
            ],
            "lrrr",
        )
    label = card.bundle.get("label") or {}
    lines += [
        "",
        "## Label",
        "",
        f"`{label.get('name', 'soh_pct')}` = {label.get('definition', '')}. Capacity enters "
        "only the label, never the inputs (enforced by the feature registry and the leakage "
        "checks).",
    ]
    return lines


def _inputs(card: _Card) -> list[str]:
    coef = card.estimator.get("coef", [])
    fmin = card.domain.get("feature_min", [])
    fmax = card.domain.get("feature_max", [])
    rows = []
    for j, name in enumerate(card.features):
        spec = FEATURES.get(name)
        unit = (
            "ratio"
            if card.relative and not is_context_feature(name)
            else (spec.unit if spec else "")
        )
        lo = fmin[j] if j < len(fmin) else None
        hi = fmax[j] if j < len(fmax) else None
        rows.append(
            [
                f"`{name}`",
                unit,
                spec.availability if spec else "",
                spec.description if spec else "",
                _signed(coef[j], 3) if j < len(coef) else "n/a",
                f"{_fmt(lo, 4)} to {_fmt(hi, 4)}",
            ]
        )
    return [
        "",
        "## Inputs",
        "",
        *_table(
            [
                "Input",
                "Unit",
                "Availability",
                "Definition",
                "Coef. (SOH pts / SD)",
                "Training range",
            ],
            rows,
            "lllllr",
        ),
        "",
        f"Prediction = {_fmt(card.estimator.get('intercept'), 3)} + sum of coefficient x "
        "(input - training mean) / training SD. A missing input is replaced by its training "
        "median and reported as a warning. Contributions are exact (linear model).",
    ]


def _selection(card: _Card) -> list[str]:
    final = card.selection.get("final")
    if not final:
        return []
    lines = [
        "",
        "## Feature selection",
        "",
        f"{card.selection.get('procedure', '')}. Penalty grid {card.selection.get('alphas')}.",
        "",
        f"Selected on all {len(card.batteries)} cells: "
        + ", ".join(f"`{f}`" for f in final.get("selected", []))
        + f" (alpha = {_fmt(final.get('alpha'), 3)}, inner LOBO MAE "
        f"{_fmt(final.get('inner_mae'), 3)}). Elimination trace:",
        "",
        *_table(
            ["Step", "Removed", "Inputs left", "alpha", "Inner LOBO MAE"],
            [
                [
                    str(i),
                    f"`{s['removed']}`" if s.get("removed") else "(start)",
                    str(s.get("n_features")),
                    _fmt(s.get("alpha"), 3),
                    _fmt(s.get("inner_mae"), 3),
                ]
                for i, s in enumerate(final.get("path", []))
            ],
            "rlrrr",
        ),
    ]
    folds = card.selection.get("lobo_folds") or []
    if folds:
        lines += [
            "",
            "Selection stability: the same procedure inside each leave-one-battery-out fold "
            "(these per-cell models produce the metrics and the conformal residuals):",
            "",
            *_table(
                ["Held-out cell", "Selected inputs", "alpha"],
                [
                    [
                        str(f["held_out"]),
                        ", ".join(f"`{x}`" for x in f["selected"]),
                        _fmt(f.get("alpha"), 3),
                    ]
                    for f in folds
                ],
                "llr",
            ),
        ]
        freq = card.selection.get("lobo_frequency") or {}
        if freq:
            lines += [
                "",
                "Share of folds selecting each candidate: "
                + ", ".join(f"`{k}` {_fmt(v, 2)}" for k, v in freq.items())
                + ".",
            ]
    return lines


def _evaluation(card: _Card) -> list[str]:
    m = card.metrics
    mean = m.get("mean") or {}
    coverage = m.get("coverage") or {}
    spec = card.spec
    rows = [
        [
            b,
            str(r.get("n", "")),
            _fmt(r.get("mae")),
            _fmt(r.get("rmse")),
            _fmt(r.get("r2"), 3),
            _signed(r.get("bias")),
            _pct((r.get("coverage") or {}).get(_LEVEL)),
            _fmt((r.get("mean_width") or {}).get(_LEVEL)),
        ]
        for b, r in card.per_battery.items()
    ]
    rows.append(
        [
            "**mean**",
            str(mean.get("n_cycles", "")),
            f"**{_fmt(mean.get('mae'))}**",
            _fmt(mean.get("rmse")),
            f"{_fmt(mean.get('r2'), 3)} (pooled)",
            _signed(mean.get("bias")),
            _pct(coverage.get(_LEVEL)),
            _fmt((m.get("mean_width") or {}).get(_LEVEL)),
        ]
    )
    return [
        "",
        "## Evaluation (leave-one-battery-out)",
        "",
        f"Source: `{m.get('source', 'reports/benchmark/results.csv')}`, spec "
        f"`{spec.get('experiment')}/{spec.get('feature_set')}/{spec.get('mode')}/"
        f"{spec.get('model')}`. Each row is a cell the procedure never saw (fitting, "
        "selection, penalty and calibration all use the other cells). The shipped model is "
        "the same procedure refitted on every cell, so it has no held-out score of its own; "
        "these numbers estimate how the procedure does on a new cell of the same population.",
        "",
        *_table(
            ["Held-out cell", "Cycles", "MAE", "RMSE", "R2", "Bias", "90% coverage", "90% width"],
            rows,
            "lrrrrrrr",
        ),
        "",
        "MAE, RMSE, bias and widths in SOH percentage points; bias = prediction - truth.",
    ]


def _uncertainty(card: _Card) -> list[str]:
    conformal = card.bundle.get("conformal") or {}
    coverage = card.metrics.get("coverage") or {}
    worst = card.metrics.get("coverage_min_battery") or {}
    quantiles: Mapping[str, Any] = conformal.get("quantiles") or {}
    lines = [
        "",
        "## Uncertainty",
        "",
        f"Method: {conformal.get('method', '')}. Calibration residuals: "
        f"{conformal.get('n_scores', 'n/a')}.",
        "",
        *_table(
            ["Confidence", "Half-width (SOH pts)", "LOBO coverage (mean over cells)", "Worst cell"],
            [
                [
                    _pct(float(k)),
                    "unbounded" if v is None else f"+/- {_fmt(v)}",
                    _pct(coverage.get(k), 1),
                    _pct(worst.get(k), 1),
                ]
                for k, v in sorted(quantiles.items(), key=lambda kv: float(kv[0]))
            ],
            "lrrr",
        ),
        "",
    ]
    under = []
    for b, r in card.per_battery.items():
        cov = (r.get("coverage") or {}).get(_LEVEL)
        if cov is not None and float(cov) < float(_LEVEL) - 0.05:
            text = f"{b} is covered {_pct(cov)} of the time"
            if r.get("bias") is not None:
                bias = float(r["bias"])
                direction = "over" if bias > 0 else "under"
                text += (
                    f" (the model {direction}-estimates its SOH by {abs(bias):.2f} points "
                    "on average)"
                )
            under.append(text)
    if under:
        lines.append(
            "Coverage is not uniform across cells. At 90% nominal, "
            + "; ".join(under)
            + ". Even for an exchangeable cell this fixed half-width has no finite-sample "
            "guarantee; its coverage is measured, and with "
            f"{len(card.per_battery)} cells one unusual cell is enough to pull it far below the "
            "nominal level."
        )
    else:
        lines.append("At 90% nominal every held-out cell is covered within 5 points of the target.")
    lows = {b: float(r[0]) for b, r in card.ranges.items() if r and r[0] is not None}
    for b, low in lows.items():
        others = [v for k, v in lows.items() if k != b]
        if others and low < min(others) - 2.0:
            lines += [
                "",
                f"{b} fades to {low:.1f}% SOH, below every other training cell (lowest "
                f"{min(others):.1f}%): when it is held out, the model must extrapolate.",
            ]
    return lines


def _domain(card: _Card) -> list[str]:
    d = card.domain
    return [
        "",
        "## Domain of applicability",
        "",
        "An input is flagged out of domain when an indicator leaves its training range widened "
        f"by {_pct(d.get('margin', 0.1))} of the range, or when its Mahalanobis distance "
        "(Ledoit-Wolf covariance of the standardised training inputs) exceeds "
        f"{_fmt(d.get('mahalanobis_threshold'))}, the {_pct(d.get('mahalanobis_quantile', 0.99))} "
        "quantile of the training distances. Out-of-domain estimates are still returned, with a "
        "warning.",
    ]


def _caveats(card: _Card) -> list[str]:
    needs = list(dict.fromkeys(_REQUIREMENTS[f] for f in card.features if f in _REQUIREMENTS))
    campaigns = {PROTOCOLS[b].group for b in card.batteries if b in PROTOCOLS}
    origin = "one test campaign" if len(campaigns) == 1 else f"{len(campaigns)} test campaigns"
    lines = [
        "",
        "## Caveats and limitations",
        "",
        f"- {len(card.batteries)} cells from {origin} of one laboratory. Every metric above "
        f"averages {len(card.per_battery)} held-out cells; differences of a few tenths of a "
        "point between models are within cell-to-cell noise.",
        "- The label uses each cell's own initial capacity (median of its first five QC-valid "
        "capacities), so every cell starts near 100% whatever its nameplate deviation.",
    ]
    if needs:
        lines.append(
            "- The inputs must be measurable: this model needs "
            + "; ".join(needs)
            + ". A missing input is median-imputed and reported in the response warnings."
        )
    lines.append(
        "- The inputs are strongly correlated (they all track the same capacity fade), so "
        "individual coefficients, including their signs, are not physical effects; the "
        "contributions explain this model's arithmetic, not the cell's chemistry."
    )
    candidates = [str(c) for c in card.selection.get("candidates", card.features)]
    if not card.relative and "chg_cv_time_s" in candidates:
        keepers = [
            str(f["held_out"])
            for f in card.selection.get("lobo_folds") or []
            if "chg_cv_time_s" in f.get("selected", [])
        ]
        text = (
            "- `chg_cv_time_s` (constant-voltage phase duration) carries a large cell-specific "
            "offset in absolute units"
        )
        if keepers:
            text += f"; the per-cell models that hold out {', '.join(keepers)} keep it"
        used = "uses" if "chg_cv_time_s" in card.features else "does not use"
        lines.append(f"{text}; the shipped model {used} it.")
    if card.relative:
        lines.append(
            f"- Relative mode needs the cell's own reference values from its first {card.n_ref} "
            "QC-valid cycles in service; without them use an absolute-mode bundle."
        )
    lines.append(
        "- Interval coverage on a new cell is measured, not guaranteed: see the per-cell table."
    )
    return lines


def _ethics_and_provenance(card: _Card) -> list[str]:
    p = card.provenance
    rows = [["Config", f"`{p.get('config', '')}`"]]
    if p.get("benchmark_config_sha256") or p.get("production_config_sha256"):
        rows += [
            [
                "Benchmark sections of the config",
                f"sha256 `{p.get('benchmark_config_sha256', '')}` (the benchmark run's "
                "`config_sha256`)",
            ],
            [
                "Production section of the config",
                f"sha256 `{p.get('production_config_sha256', '')}`",
            ],
        ]
    elif p.get("config_sha256"):  # bundles written before the hashes were scoped
        rows[0][1] += f" (sha256 `{p['config_sha256']}`)"
    rows += [
        [
            "Benchmark results",
            f"`{p.get('benchmark_results', '')}` "
            f"(sha256 `{p.get('benchmark_results_sha256', '')}`)",
        ],
        [
            "Feature table",
            f"`{p.get('features_csv', '')}` (sha256 `{card.training.get('features_sha256', '')}`)",
        ],
        [
            "Environment",
            f"python {p.get('python', '')}, numpy {p.get('numpy', '')}, pandas "
            f"{p.get('pandas', '')}, scikit-learn {p.get('sklearn', '')}",
        ],
    ]
    return [
        "",
        "## Ethical and safety considerations",
        "",
        "Over-estimating SOH can keep a degraded cell in service. This model is a research "
        "artifact, not a safety-certified battery-management function: do not use it for "
        "protection, charging-limit or other safety decisions. Report every estimate together "
        "with its interval and domain flag.",
        "",
        "## Provenance",
        "",
        *_table(["Item", "Value"], rows),
        "",
        'Reference: Mitchell et al., "Model Cards for Model Reporting", FAT* 2019.',
    ]


def render_model_card(bundle: Mapping[str, Any]) -> str:
    """Markdown model card for a production bundle document.

    Args:
        bundle: Bundle document as written by :func:`battery_soh.training.train_bundles`
            (the optional ``selection``, ``training_protocol`` and ``provenance``
            sections enrich the card; without them those parts are left out).

    Returns:
        The card text with LF line endings and a trailing newline.
    """
    card = _Card(bundle)
    lines = [
        *_header(card),
        *_use(card),
        *_data(card),
        *_inputs(card),
        *_selection(card),
        *_evaluation(card),
        *_uncertainty(card),
        *_domain(card),
        *_caveats(card),
        *_ethics_and_provenance(card),
    ]
    return "\n".join(lines) + "\n"
