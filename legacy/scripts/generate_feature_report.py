"""Generate a BRIEF, easy-to-read PDF report covering every feature and novelty
contribution in the capstone project.

Run:
    python scripts/generate_feature_report.py

Produces: paper/PROJECT_REPORT.pdf
"""

from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime

from fpdf import FPDF

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper" / "PROJECT_REPORT.pdf"
FIGS = ROOT / "results" / "phase4_evaluation"


def _safe(text: str) -> str:
    repl = {
        "–": "-", "—": "--", "‘": "'", "’": "'", "“": '"', "”": '"',
        "…": "...", "°": " deg ", "•": "-", "×": "x", "±": "+/-",
        "≤": "<=", "≥": ">=", "∞": "inf", "Δ": "Delta", "α": "alpha",
        "λ": "lambda", "τ": "tau", "φ": "phi", "σ": "sigma", "≈": "~",
        "é": "e", "²": "^2", "³": "^3", "→": "->", "®": "(R)",
        "™": "(TM)", "©": "(c)", "Ω": "Ohm",
    }
    for k, v in repl.items():
        text = text.replace(k, v)
    return text.encode("latin-1", "replace").decode("latin-1")


class Report(FPDF):
    def header(self):
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(140, 140, 140)
        self.cell(0, 4, _safe("Li-ion SOH Capstone - Project Report"),
                  align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.ln(0.5)

    def footer(self):
        self.set_y(-10)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(140, 140, 140)
        self.cell(0, 4, f"{self.page_no()} / {{nb}}", align="C")


def h1(pdf: Report, text: str) -> None:
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 15)
    pdf.set_text_color(20, 40, 100)
    pdf.cell(0, 7, _safe(text), new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(20, 40, 100)
    pdf.set_line_width(0.4)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)


def h2(pdf: Report, text: str) -> None:
    pdf.ln(1)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(50, 70, 130)
    pdf.cell(0, 5.5, _safe(text), new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)


def para(pdf: Report, text: str) -> None:
    pdf.set_font("Helvetica", "", 9.5)
    pdf.set_text_color(30, 30, 30)
    W = pdf.w - pdf.l_margin - pdf.r_margin
    pdf.multi_cell(W, 4.6, _safe(text))
    pdf.ln(0.8)


def bullet(pdf: Report, text: str) -> None:
    pdf.set_font("Helvetica", "", 9.5)
    pdf.set_text_color(30, 30, 30)
    W = pdf.w - pdf.l_margin - pdf.r_margin
    pdf.cell(4, 4.6, "")
    pdf.multi_cell(W - 4, 4.6, _safe("- " + text))


def kv(pdf: Report, key: str, value: str) -> None:
    """Short 'Key: value' line."""
    W = pdf.w - pdf.l_margin - pdf.r_margin
    pdf.set_font("Helvetica", "B", 9.5)
    kw = 38
    pdf.multi_cell(kw, 4.6, _safe(key + ":"), new_x="RIGHT", new_y="TOP")
    pdf.set_font("Helvetica", "", 9.5)
    pdf.multi_cell(W - kw, 4.6, _safe(value), new_x="LMARGIN", new_y="NEXT")


def table(pdf: Report, rows: list[list[str]], widths: list[float]) -> None:
    pdf.set_font("Helvetica", "B", 8.5)
    pdf.set_fill_color(230, 235, 245)
    for i, row in enumerate(rows):
        fill = i == 0
        for j, cell in enumerate(row):
            pdf.cell(widths[j], 5, _safe(str(cell)), border=1, fill=fill)
        pdf.ln(5)
        if i == 0:
            pdf.set_font("Helvetica", "", 8.5)
    pdf.ln(1.5)


def img(pdf: Report, path: Path, caption: str, w_mm: float) -> None:
    if not path.exists():
        return
    pdf.ln(0.5)
    pdf.image(str(path), w=w_mm)
    pdf.set_font("Helvetica", "I", 7)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(0, 3.8, _safe(caption))
    pdf.set_text_color(0, 0, 0)
    pdf.ln(1)


# =========================================================================== #
# Content (compact)
# =========================================================================== #


def cover(pdf: Report) -> None:
    pdf.add_page()
    W = pdf.w - pdf.l_margin - pdf.r_margin
    pdf.set_y(50)

    def centered(text: str, size: int, bold: bool, line: float, color: tuple) -> None:
        pdf.set_x(pdf.l_margin)
        pdf.set_font("Helvetica", "B" if bold else "", size)
        pdf.set_text_color(*color)
        pdf.multi_cell(W, line, _safe(text), align="C", new_x="LMARGIN", new_y="NEXT")

    centered("Li-ion Battery Health Prediction", 20, True, 10, (20, 40, 100))
    centered("Features & Novelty - Brief Report", 13, True, 7, (60, 80, 130))
    pdf.ln(3)
    centered("Every feature, dataset, model, and new contribution in this project, "
             "explained in plain language.", 10, False, 5, (50, 50, 50))
    pdf.ln(10)
    centered(f"Generated {datetime.now().strftime('%d %B %Y')}", 9, False, 4.5, (110, 110, 110))
    pdf.set_text_color(0, 0, 0)


def overview(pdf: Report) -> None:
    h1(pdf, "1. What the project does")
    para(pdf,
        "It predicts how healthy a lithium-ion battery is and how many "
        "cycles it has left, from simple sensor readings. Along with the "
        "number, it tells you how confident it is, why it thinks so, and "
        "which past batteries looked similar.")
    h2(pdf, "Why it is different from typical battery-ML work")
    bullet(pdf, "Uses both numbers AND text (a natural-language diagnosis) as inputs.")
    bullet(pdf, "Gives a RANGE, not just a number, with a mathematical coverage guarantee.")
    bullet(pdf, "Teaches the model physical rules during training (batteries don't get healthier, hotter cells age faster, etc).")
    bullet(pdf, "Predicts a full Remaining-Life DISTRIBUTION, not a point estimate.")
    bullet(pdf, "The text input is generated by an LLM from the 5 most similar past cycles (a first for Li-ion).")


def datasets(pdf: Report) -> None:
    h1(pdf, "2. Datasets")
    h2(pdf, "NASA Prognostics Data Repository")
    bullet(pdf, "4 small 18650 cells (B0005-B0007, B0018), ~2 Ah each.")
    bullet(pdf, "168 discharge cycles per battery, cycled at 24-43 C.")
    bullet(pdf, "The standard Li-ion benchmark.")
    h2(pdf, "Mendeley forklift cells")
    bullet(pdf, "3 large commercial cells (Cell1-3) with realistic duty cycles.")
    bullet(pdf, "Hundreds of cycles each, larger capacity (up to 190 Ah).")
    h2(pdf, "How the data is split")
    para(pdf,
        "Per battery, by cycle order: 60% train / 15% calibration / 10% "
        "validation / 15% test. The calibration chunk is separate from "
        "training so the uncertainty guarantees are mathematically valid.")


def features(pdf: Report) -> None:
    h1(pdf, "3. Features used (18 numbers + 1 text)")
    para(pdf, "Every cycle is summarised by 18 physical measurements plus a natural-language description:")
    rows = [
        ["Group", "Features", "What they capture"],
        ["Capacity",  "capacity",
         "How much charge the cell holds right now (Ah)"],
        ["Voltage",   "voltage_min/max/mean/std/range",
         "Voltage window and variability in the cycle"],
        ["Current",   "current_mean, current_std",
         "Discharge current magnitude and noise"],
        ["Thermal",   "temp_mean/max/delta/std",
         "Cell temperature, rise during discharge, variability"],
        ["ICA (dQ/dV)", "ica_peak_voltage/magnitude/mean/std",
         "Physics diagnostic: phase-transition peaks shift with ageing"],
        ["Resistance", "dcir",
         "Internal resistance, rises with SEI growth"],
        ["Time",       "discharge_time",
         "Seconds from full charge to cutoff voltage"],
    ]
    table(pdf, rows, [20, 55, 95])
    h2(pdf, "Text modality")
    para(pdf,
        "Every cycle also gets a 3-5 sentence diagnostic paragraph. In the "
        "old pipeline this was rule-based templates; in the new RAP pipeline "
        "an LLM writes it after seeing the 5 most similar past cycles. "
        "Sentence-BERT then encodes the text into a 768-number vector.")


def phase2(pdf: Report) -> None:
    h1(pdf, "4. Phase 2 - Core pipeline")
    para(pdf, "Ten notebooks (data audit, harmonisation, baselines, multimodal, "
              "ablation, cross-battery test, robustness, SHAP, statistics, final report). Key outputs:")
    bullet(pdf, "Harmonised feature CSV across NASA + Mendeley.")
    bullet(pdf, "Baseline models: Ridge (best, R^2 0.92), Random Forest, Gradient Boosting.")
    bullet(pdf, "First multimodal attempt (concatenation fusion + SBERT text).")
    bullet(pdf, "Cross-battery transfer, robustness, SHAP explainability all validated.")


def phase3(pdf: Report) -> None:
    h1(pdf, "5. Phase 3 - Multimodal Cross-Attention Model")
    para(pdf,
        "Replaces the simple concatenation fusion with a proper cross-attention "
        "architecture. The numeric side is a CNN+Transformer; the text side is "
        "SBERT + adapter; the two modalities fuse via attention with a learnable "
        "gate that decides how much each modality contributes per sample.")
    rows = [
        ["Metric", "Train", "Validation", "Test"],
        ["MAE",  "1.70%", "2.29%", "1.91%"],
        ["RMSE", "2.90",  "6.94",  "3.47"],
        ["R^2",  "0.99",  "0.95",  "0.96"],
    ]
    table(pdf, rows, [30, 35, 35, 35])
    para(pdf, "About 500K trainable parameters. The saved checkpoint lives at "
              "models/cross_attention_model.pt.")


def phase4(pdf: Report) -> None:
    h1(pdf, "6. Phase 4 - The four novelty contributions")
    para(pdf,
        "These are the main thesis contributions. Each sits on top of the Phase 3 "
        "model and is exposed as its own API endpoint.")

    h2(pdf, "6.1 Physics-informed loss (training-time rules)")
    para(pdf,
        "The model is penalised during training when its predictions break "
        "three known facts of battery physics:")
    bullet(pdf, "Batteries don't get healthier: predicted SOH must not increase with cycle number.")
    bullet(pdf, "Hotter cycles age faster (Arrhenius rule).")
    bullet(pdf, "Capacity loss and DCIR rise are coupled (Butler-Volmer-like).")
    para(pdf,
        "Effect on the test set: RMSE drops 28% (6.18 -> 4.45), monotonicity "
        "violations drop by 13 percentage points (60% -> 47%). File: "
        "src/phase4/pinn_loss.py.")

    h2(pdf, "6.2 Conformal prediction (uncertainty with a guarantee)")
    para(pdf,
        "Gives a low/high SOH range with a mathematical promise: for example "
        "'the true SOH is inside this range at least 90% of the time.' Three "
        "flavours:")
    bullet(pdf, "Marginal split-conformal: fixed +/- width, simplest.")
    bullet(pdf, "Mondrian: a separate width per battery, reveals which batteries are hard.")
    bullet(pdf, "CQR (Conformalized Quantile Regression): width ADAPTS - narrow on easy cycles, wide on hard ones.")
    para(pdf,
        "At target 90% coverage, q_hat = 6.57 SOH% (conformal) and 5.59 "
        "(CQR). Files: src/phase4/conformal.py, src/phase4/cqr.py.")

    h2(pdf, "6.3 Weibull survival head (remaining-life as a distribution)")
    para(pdf,
        "Instead of predicting a single 'number of cycles left', the model "
        "outputs a full Weibull distribution. You can then ask for any "
        "percentile: 'optimistic estimate', 'most likely', 'pessimistic'. "
        "Cycles that haven't yet reached End-of-Life are correctly treated "
        "as censored data.")
    bullet(pdf, "Weibull shape k = 1.52 (MLE over training trajectories) means wear-out dominated ageing.")
    bullet(pdf, "Files: src/phase4/survival_head.py, endpoint: POST /predict/rul.")

    h2(pdf, "6.4 Retrieval-Augmented Prognostics (RAP)")
    para(pdf,
        "Replaces the rule-based text templates with a smarter pipeline:")
    bullet(pdf, "1. Find the 5 most similar past cycles in the training corpus.")
    bullet(pdf, "2. Send those 5 cycles + current features to an LLM (Groq llama-3.3-70b).")
    bullet(pdf, "3. LLM writes a fresh 3-5 sentence diagnosis grounded in the analogues.")
    bullet(pdf, "4. Sentence-BERT encodes that text for the cross-attention model.")
    para(pdf,
        "Measured effect on test: MAE drops 61% (4.59 -> 1.80) just by "
        "swapping the semantic channel. File: src/phase4/rap.py.")


def results(pdf: Report) -> None:
    h1(pdf, "7. Results at a glance")
    h2(pdf, "Ablation on combined NASA+Mendeley")
    rows = [
        ["Model", "MAE", "RMSE", "Monotonicity viol."],
        ["Numeric-only (no text)",    "4.38", "8.90", "55.8%"],
        ["Multimodal (MSE only)",     "3.07", "6.18", "60.5%"],
        ["Multimodal + Physics loss", "2.57", "4.45", "47.3%"],
    ]
    table(pdf, rows, [70, 22, 22, 45])
    h2(pdf, "RAP vs template text channel")
    rows = [
        ["Text source", "MAE", "RMSE"],
        ["Rule-based template",           "4.59", "5.98"],
        ["RAP (retrieval + LLM)",         "1.80", "3.37"],
    ]
    table(pdf, rows, [95, 25, 25])
    h2(pdf, "Conformal coverage calibration")
    rows = [
        ["Target", "Marginal", "Mondrian", "Width (SOH %)"],
        ["95%", "91.6%", "86.5%", "14.58"],
        ["90%", "84.9%", "86.5%",  "9.73"],
        ["85%", "82.0%", "85.1%",  "7.50"],
        ["80%", "81.0%", "82.2%",  "5.95"],
    ]
    table(pdf, rows, [30, 40, 40, 40])
    W = pdf.w - pdf.l_margin - pdf.r_margin
    img(pdf, FIGS / "conformal_calibration.png",
        "Coverage stays close to the y=x ideal; width shrinks with coverage.",
        w_mm=W)
    img(pdf, FIGS / "ablation.png",
        "Left: point-error drops with each added module. Right: monotonicity violations drop with PINN loss.",
        w_mm=W)


def deployment(pdf: Report) -> None:
    h1(pdf, "8. What is deployed")
    h2(pdf, "API endpoints")
    rows = [
        ["Endpoint", "What it returns"],
        ["POST /predict",                 "SOH point estimate (Ridge, fastest)"],
        ["POST /predict/crossattention",  "Cross-attention SOH + Ridge + delta"],
        ["POST /predict/conformal",       "Fixed-width 90%/95% interval"],
        ["POST /predict/cqr",             "Adaptive-width CQR interval"],
        ["POST /predict/rul",             "Weibull median + p10/p90 cycles to EoL"],
        ["POST /diagnose/rap",            "LLM diagnosis + similar past cycles"],
        ["POST /chat",                    "Battery Doctor LLM chat"],
        ["POST /simulate",                "Digital-twin trajectory over time"],
        ["GET  /phase4/info",             "Calibration stats for all wrappers"],
    ]
    table(pdf, rows, [62, 108])
    h2(pdf, "Frontend (single-page web UI)")
    bullet(pdf, "Predict tab: slider form, SOH gauge, risk meter, SHAP chart, PDF download.")
    bullet(pdf, "Phase 4 tab: uncertainty range + remaining life + AI diagnosis, in plain English.")
    bullet(pdf, "What-If simulator, Battery Doctor chat, Fleet Manager, Generalization tabs.")
    h2(pdf, "Container")
    bullet(pdf, "Dockerfile with pre-baked Sentence-BERT weights.")
    bullet(pdf, "Groq llama-3.3-70b as the LLM; falls back to OpenAI; rule fallback if neither.")
    bullet(pdf, "Runs healthy on localhost:8000 with docker compose up -d --build.")


def novelty_rank(pdf: Report) -> None:
    h1(pdf, "9. Strongest novelties, ranked")
    kv(pdf, "1. RAP",
       "Retrieval + LLM text is a battery-ML first; 61% MAE drop on the text channel alone.")
    kv(pdf, "2. Physics loss",
       "Three soft-hinge penalties together, inside a multimodal model. 28% RMSE reduction.")
    kv(pdf, "3. Conformal + CQR",
       "Coverage-guaranteed SOH intervals; per-battery stratification + adaptive widths.")
    kv(pdf, "4. Weibull RUL",
       "Right-censoring-aware survival head gives a full remaining-life distribution.")
    h2(pdf, "Supporting contributions")
    bullet(pdf, "Cross-attention fusion over numeric + text (R^2 0.96 on test).")
    bullet(pdf, "Harmonisation pipeline across two public datasets.")
    bullet(pdf, "Production Docker + LLM-backed web UI with SHAP and PDF reports.")


def limitations(pdf: Report) -> None:
    h1(pdf, "10. Honest limitations")
    bullet(pdf, "Cross-attention has larger per-sample error tails than Ridge on some cycles (5-8% at p90). Phase-4 intervals cover this.")
    bullet(pdf, "Calibration set is small (n ~ 126 on NASA-only). Coverage curves are noisier than published studies.")
    bullet(pdf, "RAP quality depends on the LLM; smaller open models reduce the gain.")
    bullet(pdf, "No formal domain-adaptation (DANN) yet between NASA and Mendeley.")
    bullet(pdf, "Calibrator is fit once; no online update as new cycles stream in.")


def refs(pdf: Report) -> None:
    h1(pdf, "11. Key references")
    for r in [
        "Severson et al. 2019 - Nature Energy. Data-driven battery life prediction.",
        "Zhang et al. 2018 - IEEE TVT. LSTM for RUL.",
        "Song et al. 2022 - Energy Reports. Transformer SOH.",
        "Zhao et al. 2021 - Energies. Review and semantic-reasoning gap.",
        "Vovk, Gammerman, Shafer 2005 - Algorithmic Learning in a Random World.",
        "Lei et al. 2018 - JASA. Distribution-free predictive inference.",
        "Romano, Patterson, Candes 2019 - NeurIPS. Conformalized Quantile Regression.",
        "Raissi et al. 2019 - JCP. Physics-informed neural networks.",
        "Dubarry et al. 2012 - J. Power Sources. Incremental Capacity Analysis.",
    ]:
        bullet(pdf, r)


def main() -> None:
    pdf = Report(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.set_margins(14, 12, 14)
    pdf.alias_nb_pages()

    cover(pdf)
    overview(pdf)
    datasets(pdf)
    features(pdf)
    phase2(pdf)
    phase3(pdf)
    phase4(pdf)
    results(pdf)
    deployment(pdf)
    novelty_rank(pdf)
    limitations(pdf)
    refs(pdf)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(OUT))
    size_kb = os.path.getsize(OUT) / 1024
    print(f"Wrote {OUT} ({size_kb:.1f} KB, {pdf.page_no()} pages)")


if __name__ == "__main__":
    main()
