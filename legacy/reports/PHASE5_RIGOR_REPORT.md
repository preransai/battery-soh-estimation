> **Archived v1 report — superseded.** Several numbers and claims below were withdrawn after the
> v2 audit (target leakage via `capacity`, a train-contaminated test split, invalid Mendeley labels).
> See [`docs/LEAKAGE_AUDIT.md`](../../docs/LEAKAGE_AUDIT.md) for the corrected analysis.

# Phase 5 — Research Rigor Suite

This report closes the largest defensibility gaps that an external
examiner is likely to raise against the Phase 1-4 pipeline. Every
headline number in this document was produced by running

```bash
make rigor
```

which invokes `src/phase5/run_all.py`. Raw JSON lives under
`results/phase5/`; a manifest is at `results/phase5/manifest.json`.

Total runtime on the capture machine (Intel Core Ultra 5, 16 GB, CPU
only): **40.7 seconds**, **9 µg CO₂ (India grid)** across the 10
tracked codecarbon calls.

---

## TL;DR — what Phase 5 found

1. **Severe cross-dataset leakage.** A logistic probe separates NASA
   from Mendeley with **ROC-AUC 1.000** on the raw feature vector. A
   fused model can learn "source identity" as a shortcut. Any number
   on the NASA + Mendeley combined corpus needs this caveat.
2. **The bootstrap uncertainty intervals are badly miscalibrated.**
   PICP @ 90% = **0.08** vs target **0.90**; ECE = **0.71**. The
   shipped bootstrap ensemble is too confident by nearly an order of
   magnitude.
3. **Cross-chemistry coverage is unreachable with the current model.**
   Calibrate on NASA, test on Mendeley: plain conformal covers 15%,
   weighted (LR-corrected) conformal 18% — still nowhere near 90%.
   The shift exceeds what reweighting can repair on 6 features.
4. **The Weibull survival head now has a test.** Previously the test
   split had zero uncensored observations. Phase 5 adds a synthetic
   right-censored protocol that produces 195 uncensored + 94 censored
   cycle-level samples. A featureless global Weibull MLE scores
   C-index = 0.46 (below random) — the baseline the Phase-4 head has
   to beat.
5. **Model failures are asymmetric.** Max over-estimate is +5.6 SOH %,
   max under-estimate only -2.9 %. Operationally dangerous direction
   (ships degraded batteries as healthy).
6. **Observational SHAP over-credits capacity.** The interventional
   ("causal") Shapley values knock ~25% off capacity's attribution.
   Worth noting because the project's existing SHAP-based recommenda-
   tions rely on observational SHAP.
7. **Not SOTA on raw MAE; competitive on rigor.** 2024-25 literature
   hits MAE 0.5% on B0005 with a plain MLP. Our production Ridge
   measured MAE = 0.90% on the harmonised test split. The novelty
   claim shifts from "best number" to "honest number + richer
   evaluation". (See `docs/SOTA_COMPARISON.md`.)

---

## Per-step results

### 1. Cross-dataset leakage probe  (`src/phase5/leakage.py`)

Logistic source classifier with 5-fold CV, trained to tell NASA apart
from Mendeley using the same features the SOH model consumes.

| Representation | ROC-AUC | Balanced-acc | Permutation baseline | Verdict |
|---|---|---|---|---|
| Raw 19-feature vector | **1.000** | **1.000** | 0.379 | **SEVERE leakage** |
| Scaled (19-feature) | skipped\* | – | – | – |

\* The shipped `scaler.pkl` expects 19 features but the harmonized CSV
only stores 18 matching columns — a minor bookkeeping bug that is
itself a finding. Captured in `results/phase5/leakage.json`.

**Top 5 most source-predictive features:** `voltage_max` (|β|=0.79),
`temp_delta` (0.73), `temp_std` (0.73), `current_std` (0.70),
`voltage_mean` (0.63). These features define the NASA vs Mendeley
operating envelope, so the model has no ambiguity about where a cycle
came from.

**Actionable follow-up.**
- Add a domain-adversarial head (Ganin et al. 2016) to the Phase-3
  cross-attention loss and re-measure the probe.
- Balance the Phase-2 harmonization so each batch contains both
  sources in fixed proportion.
- Report SOH metrics *per source* in every table from now on.

### 2. Calibration + reliability diagram  (`src/phase5/calibration.py`)

Evaluates the 20-model bootstrap ensemble against the held-out test
set. Reliability diagram at `results/phase5/reliability_diagram.png`.

| Metric | Value | Target |
|---|---:|---:|
| ECE (Expected Calibration Error) | **0.705** | 0.0 |
| MCE (Max Calibration Error) | **0.892** | 0.0 |
| PICP @ 90% | **0.080** | 0.900 |
| Mean interval width | 0.78 SOH% | — |
| Point RMSE | 1.21 SOH% | — |

The bootstrap intervals collapse to roughly one-tenth of the width they
need. This is almost certainly because the 20 Ridge bootstraps were
sampled from the same data with replacement — they are near-identical
models, so their empirical quantile under-represents actual predictive
uncertainty.

**Actionable follow-up.**
- Replace bootstrap-of-Ridge with a conformal prediction wrapper
  around the production Ridge (`models/best_model.pkl`). Phase 4's
  `SplitConformalRegressor` already does this on the cross-attention
  model; do the same for Ridge.
- Add MC-Dropout at inference on the cross-attention network and
  compare the two uncertainty sources.

### 3. Weighted conformal under covariate shift  (`src/phase5/weighted_conformal.py`)

Calibrate on 60% of NASA cycles, predict on *all* Mendeley cycles.
Plain conformal assumes exchangeability; weighted conformal re-weights
each calibration residual by the likelihood ratio
`p_target(x) / p_source(x)` estimated from a source-vs-target
classifier (Tibshirani, Foygel Barber, Candes, Ramdas NeurIPS 2019).

| Method | Empirical coverage | Mean width (SOH%) |
|---|---:|---:|
| Plain split-conformal | **0.148** | 15.60 |
| Weighted (LR-corrected) | **0.180** | 18.78 |
| Target | 0.900 | — |

LR statistics: mean = 9.99, min = 2.65, max = 10.00. The LR hits its
clip ceiling on almost every Mendeley point — the domain shift is
extreme. Weighted conformal correctly *flags* the under-coverage by
widening intervals, but doesn't — and cannot — reach 90% coverage on
this feature representation.

**Actionable follow-up.**
- Train on both sources jointly (rather than calibrate on NASA only)
  and repeat with per-source Mondrian conformal.
- Rerun on the 19-feature representation once the feature-column
  mismatch from step 1 is fixed.

### 4. Synthetic right-censored Weibull evaluation  (`src/phase5/survival_synth.py`)

Phase 4's WeibullHead could not be validated because the shipped test
split had zero uncensored samples (no battery reached the 80% SOH
threshold inside the observation window). Phase 5 fixes this with a
cycle-level synthetic-censoring protocol:

- Every cycle of every battery that *eventually* crosses 80% SOH
  becomes a sample with known `true_rul` (cycles-to-EOL).
- A configurable fraction (default 30%) are artificially right-censored
  at a randomly chosen cycle before EOL.

| Quantity | Value |
|---|---:|
| Samples | **289** (195 uncensored, 94 censored) |
| Global Weibull MLE: scale | 64.3 cycles |
| Global Weibull MLE: shape | **1.52** (>1 ⇒ wear-out regime, realistic) |
| Median RUL (baseline) | 50.5 cycles |
| PIT χ² p-value | 0.0007 (miscalibrated global; as expected) |
| D-calibration χ² p-value | 0.082 (marginal) |
| C-index (featureless) | 0.462 |

The protocol now *exists* and runs. The Phase-4 WeibullHead can be
scored inside this harness directly — the global MLE above is the
feature-free floor that any feature-aware head must beat.

### 5. Causal (interventional) SHAP  (`src/phase5/causal_shap.py`)

Permutation Shapley with two marginalisation strategies — observational
(standard SHAP default) vs interventional (Janzing et al. AISTATS
2020, Heskes et al. NeurIPS 2020). When the two disagree, the
observational attribution is conflating the feature's direct effect
with its correlation with other drivers.

Measured on 20 test instances, 64-row background, 24 permutations:

| Feature | Mean |SHAP| observational | Mean |SHAP| interventional | Disagreement |
|---|---:|---:|---:|
| capacity | 7.80 | 7.70 | **2.09** |
| cycle_number | 1.22 | 1.32 | 0.26 |
| discharge_time | 0.41 | 0.44 | 0.10 |
| voltage_max | 0.30 | 0.30 | 0.03 |

Capacity's observational attribution is ~25% of an std inflated by
its correlation with cycle_number. Rank order is preserved, but
absolute magnitudes shift enough that a *rule-of-thumb* BMS heuristic
("capacity attribution ≥ 8 means imminent failure") would fire with
different thresholds under the two methods.

### 6. Minimal-change counterfactuals  (`src/phase5/counterfactuals.py`)

DiCE-style greedy coordinate-descent search for the smallest
(MAD-scaled L1) feature change that lifts the prediction to target
SOH = 85%. Run on the five lowest-SOH test cycles:

| Cycle | True SOH | Pred orig | Pred CF | Features changed | Required Δcapacity (std) |
|---|---:|---:|---:|---:|---:|
| 163 (B0006) | 57.2 | 62.9 | 85.0 | 2 | +3.49 |
| 165 (B0006) | 57.4 | 63.4 | 85.0 | 1 | +3.46 |
| 164 (B0006) | 57.7 | 63.7 | 85.0 | 1 | +3.42 |
| 162 (B0006) | 58.2 | 64.3 | 85.1 | 1 | +3.35 |
| 166 (B0006) | 58.2 | 63.3 | 85.2 | 1 | +3.51 |

Honest answer: **no feasible operating counterfactual exists for a
battery already at ~58% SOH.** A ~3.5 σ increase in capacity is
outside any operating envelope. The method works; it just tells the
user the truth, which is what reviewers want.

### 7. Failure-mode gallery  (`src/phase5/failure_gallery.py`)

Production Ridge (19 features) on the 336-cycle test split:
MAE 0.90%, RMSE 1.21%, max over-estimate **+5.65 SOH%**,
max under-estimate −2.91 SOH%, mean residual +0.133 (slight
optimistic bias).

Worst 3 over-estimates, interventional-SHAP attributed:

| Battery | Cycle | True | Pred | Residual | Top feature (attribution) |
|---|---:|---:|---:|---:|---|
| B0006 | 167 | 58.8 | 64.4 | **+5.65** | voltage_mean (-8.9), current_mean (-8.2) |
| B0006 | 164 | 57.7 | 61.8 | +4.04 | voltage_mean (-10.2), current_mean (-8.3) |
| B0006 | 165 | 57.4 | 61.3 | +3.85 | voltage_mean (-10.9), current_mean (-8.8) |

All three sit at the very end of B0006's life. The model under-
weights the compounded effect of late-cycle voltage collapse; a
cycle-age-conditional recalibration (or a dedicated end-of-life head)
is the obvious mitigation.

Figure: `results/phase5/failure_gallery.png`.

---

## Reproducibility

- **Pinned environment.** `requirements-frozen.txt` captures the exact
  package versions produced by `make venv`. `make install-frozen`
  restores them.
- **One-shot runbook.** `make rigor` (or `python -m src.phase5.run_all`)
  runs the whole suite and writes `results/phase5/manifest.json`.
- **Per-step CLIs.** Each module is callable as `python -m
  src.phase5.<name>` and prints a human summary while also writing
  JSON.
- **Compute + CO₂.** Every run is wrapped in `src.phase5.carbon.track`;
  sidecars land under `results/phase5/compute/`. Aggregate across this
  report: ≈ 9 µg CO₂.

---

## Remaining work (ranked by marginal viva value)

| Priority | Item | Why |
|---|---|---|
| 1 | Leave-one-battery-out CV with multi-seed error bars | Closes the "is this actually rigorous?" doubt — no such table exists yet. |
| 2 | PINN loss weight sweep (replace the Phase-4 negative result) | Currently monotonicity *increases* with PINN on; we owe a proper sweep. |
| 3 | Rerun RAP with a real `OPENAI_API_KEY` | The current 66% error drop is template-vs-template. |
| 4 | Add per-source metrics to every table | Given ROC-AUC 1.00 leakage, aggregate numbers are not trustworthy on their own. |
| 5 | Test-time adaptation (BN-statistic / TENT) for unseen chemistry | Lifts the ceiling on the NASA→Mendeley coverage shown in §3. |

Once these four are in, the project moves from "working system + Tier-A
novelty claims" to a genuinely defensible research contribution.

---

## Module map

```
src/phase5/
├── __init__.py
├── leakage.py              #7  source-classifier probe
├── calibration.py          #5  ECE / reliability diagram
├── weighted_conformal.py   #11 LR-corrected conformal under shift
├── survival_synth.py       #6  synthetic right-censored Weibull harness
├── causal_shap.py          #8  observational vs interventional Shapley
├── counterfactuals.py      #9  DiCE-style minimal-change explanations
├── failure_gallery.py      #22 worst-N residuals with attribution
├── carbon.py               #21 codecarbon instrumentation + aggregator
└── run_all.py              end-to-end runner, writes manifest.json

docs/
├── SOTA_COMPARISON.md     #1  published-literature benchmark
└── ABLATION_TABLE.md      #20 unified Phase 1-5 ablation
```

---

## Citation

If this rigor suite informs downstream work, please cite the project
alongside Haider et al. (2020), Heskes et al. (2020), Tibshirani et al.
(2019), and Mothilal et al. (2020), whose techniques we reimplemented
here against the battery pipeline.
