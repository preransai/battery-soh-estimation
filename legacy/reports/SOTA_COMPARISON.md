> **Archived v1 report — superseded.** Several numbers and claims below were withdrawn after the
> v2 audit (target leakage via `capacity`, a train-contaminated test split, invalid Mendeley labels).
> See [`docs/LEAKAGE_AUDIT.md`](../../docs/LEAKAGE_AUDIT.md) for the corrected analysis.

# State-of-the-Art Comparison

Published numbers on the NASA PCoE B0005 / B0006 / B0007 batteries,
gathered in 2026 from peer-reviewed venues (Nature Scientific Reports,
Wiley IJER, MDPI Applied Sciences, ScienceDirect JPS). The point of
this table is to sanity-check where our project sits versus the
literature and to be explicit about where we lead, trail, or measure
something nobody else is measuring.

## Reporting conventions

All prior-art numbers are on a SOH scale in **[0, 1]** (fractional SOH)
as reported in the source papers. Our project reports SOH as a
percentage (e.g. `soh=96.0`). Multiply their MAE / RMSE by 100 to
compare. Where a paper reports on a specific battery we note it; where
they average over B0005-B0007 we note that too.

## Point-prediction benchmark

| # | Method                               | Year | Data                    | R²         | MAE (SOH%) | RMSE (SOH%) | Notes                                               |
|---|--------------------------------------|------|-------------------------|------------|------------|-------------|-----------------------------------------------------|
| 1 | MLP (Günes et al., Sci Rep)          | 2025 | B0005                   | 0.9955     | 0.49       | 0.69        | Cycle-features only; strong numeric baseline        |
| 2 | TCN (Günes et al., Sci Rep)          | 2025 | B0005                   | 0.9951     | ≈0.50      | 0.71        | Temporal conv over raw cycle trace                  |
| 3 | LSTM (MDPI App Sci)                  | 2025 | B0005/6/7 avg.          | 0.980      | 1.18       | 1.45        | Straight recurrent baseline                         |
| 4 | CNN+ATT (Taylor & Francis)           | 2025 | B0005/6/7               | >0.90      | <2.0       | <2.0        | Attention on cycle windows                          |
| 5 | CEEMDAN+SVR+LSTM (Sci Rep hybrid)    | 2025 | B0005/6/7               | >0.96      | <0.60      | <0.86       | Signal decomposition + two learners                 |
| 6 | WOA-GRU-TCN-SVR (WGTS)               | 2025 | NASA composite          | –          | –          | 0.20        | Ensemble with metaheuristic tuning                  |
| 7 | SA-PI-AGRU (physics-informed GRU)    | 2024 | B0005                   | 0.9961     | –          | –           | Best-reported single-model R² on B0005              |
| 8 | ELM variant (Wiley IJER)             | 2024 | B0005/6/7               | >0.98      | <1.5       | <1.8        | Extreme Learning Machine                            |
| — | **This project — Ridge (full-19)**   | 2026 | B0005/6/7 test split    | 0.96\*     | 0.90       | 1.21        | `models/best_model.pkl`, 336-cycle holdout          |
| — | **This project — Phase-3 CrossAttn** | 2026 | B0005/6/7 + Mendeley    | 0.9595     | 3.33       | –           | README headline figure (different test partition)   |
| — | **This project — Phase-3 bootstrap** | 2026 | B0005/6/7 test split    | n/a        | –          | 1.21        | Ensemble mean + interval; see calibration report    |

\* The 0.96 vs 0.9595 difference comes from different test partitions:
the cross-attention paper-ready split in `PHASE3_FINAL_REPORT.md`
vs. the production Ridge measured on
`data/phase2/harmonized/test_features.csv`. We keep both rows honest.

### Where we stand

- **Not SOTA on raw point error.** 2024-2025 Sci-Rep-tier papers
  achieve MAE ≈ 0.5% on B0005 with an MLP; our 0.9% on the full-19
  Ridge is competitive with 2022-2023 baselines but trails the
  current frontier by ~0.4 SOH%.
- **On par with hybrid baselines.** CEEMDAN+SVR+LSTM (#5) reports
  MAE <0.60 with a 2-stage signal-decomposition pipeline; this
  project's MAE=0.90 from a *linear* model suggests most of the
  predictable variance is captured by simple features.
- **Physics-informed leader.** SA-PI-AGRU (#7) at R²=0.9961 is the
  reference point for any Phase-4 PINN story: our current PINN
  loss does not reach that bar and the weight sweep in Tier-1 #2
  remains open work.

## Where published literature does *not* compete

These are dimensions no paper in the table reports, so they are our
novelty room:

| Dimension                          | Project status                                              | Literature coverage |
|------------------------------------|-------------------------------------------------------------|----------------------|
| Multimodal (numeric + LLM text)    | Phase 3 cross-attention with Sentence-BERT                  | Not represented      |
| Retrieval-Augmented Prognostics    | Phase 4 RAP module (needs real OpenAI key for full rerun)   | Not represented      |
| Conformal prediction (Mondrian)    | Phase 4 shipped; Phase 5 adds weighted / shift-aware variant| 1 paper (2023) uses plain split-conformal |
| Source-classifier leakage probe    | Phase 5, ROC-AUC 1.00 detected — real finding               | Not represented      |
| Counterfactual explanations (DiCE) | Phase 5, minimal-change search over feasible range          | Not represented      |
| Failure-mode gallery + causal SHAP | Phase 5                                                     | Not represented      |
| Compute / carbon report            | Phase 5 (codecarbon, per-run sidecars)                      | Not represented      |

A defensible framing for the viva:
> "We do not target the leaderboard on MAE - the published MLP (#1)
> already lives inside the noise floor of the NASA dataset. Our
> contributions are orthogonal: a multimodal numeric+LLM representation,
> a rigor suite (calibration, leakage detection, weighted conformal,
> synthetic-censored survival, causal SHAP, counterfactuals), and an
> operational wrapping (FastAPI + PDF + what-if) that none of the cited
> papers provide."

## Re-running the comparison

All numbers in the "this project" rows come from the code under
`src/phase5/`:

```bash
make rigor                # runs the whole suite, writes results/phase5/*.json
python -m src.phase5.failure_gallery   # MAE / RMSE on the production test split
python -m src.phase5.calibration       # calibration + bootstrap intervals
```

The prior-art rows should be re-checked against the cited papers before
going into the final report; the references section of
`PHASE5_RIGOR_REPORT.md` links each one.
