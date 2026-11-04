> **Archived v1 report — superseded.** Several numbers and claims below were withdrawn after the
> v2 audit (target leakage via `capacity`, a train-contaminated test split, invalid Mendeley labels).
> See [`docs/LEAKAGE_AUDIT.md`](../../docs/LEAKAGE_AUDIT.md) for the corrected analysis.

# Unified Ablation Table

A single table that lets a reviewer see the marginal contribution of
every component of the system (Phases 1 → 5) without hunting through
notebooks. Where Phase 2, 3 or 4 already measured a variant we
re-use those numbers; where Phase 5 adds new variants we mark them
`[P5]`.

## Source of each row

| Column       | Source notebook / module                                                 |
|--------------|--------------------------------------------------------------------------|
| Phase 1-2    | `notebooks/PHASE2_RUN_ALL.ipynb`, `phase2_step3_baselines_corrected.ipynb`, `phase2_step5_ablation.ipynb` |
| Phase 3      | `notebooks/phase3_step2_cross_attention.ipynb`, `PHASE3_FINAL_REPORT.md` |
| Phase 4      | `notebooks/phase4_tier_a_novelty.ipynb`, `notebooks/phase4_evaluation.ipynb` |
| Phase 5      | `src/phase5/*`, results under `results/phase5/`                         |

## Main table (SOH point prediction + calibration)

Split semantics for Phase 2 and Phase 5 rows is the per-battery
time-ordered train/test split under
`data/phase2/harmonized/{train,test}_features.csv`. Phase 3's cross-
attention reports numbers on a different test partition and we do
not over-claim equivalence.

| # | Configuration                              | Features     | Test R² | MAE %  | RMSE % | Coverage @90 | ECE    | Notes                                 |
|---|--------------------------------------------|--------------|---------|--------|--------|--------------|--------|---------------------------------------|
| 1 | Ridge (simple 6 features)                  | 6 numeric    | 0.9176  | 4.28   | ~5.3   | n/a          | n/a    | Phase 2 baseline                      |
| 2 | Gradient Boosting                          | 6 numeric    | 0.9100  | 4.52   | ~5.5   | n/a          | n/a    | Phase 2 baseline                      |
| 3 | Random Forest                              | 6 numeric    | 0.8900  | 5.01   | ~6.0   | n/a          | n/a    | Phase 2 baseline                      |
| 4 | **Ridge (full 19 features, production)**   | 19 numeric   | *0.96*  | 0.90   | 1.21   | n/a          | n/a    | `models/best_model.pkl` [P5 measured] |
| 5 | CNN-LSTM (Phase 1 baseline)                | Raw cycle    | ~0.89   | ~5.5   | –      | n/a          | n/a    | Phase 1 proof of concept              |
| 6 | Cross-Attention (num + SBERT + gated)      | 19 + text    | 0.9595  | 3.33   | –      | n/a          | n/a    | Phase 3 headline                      |
| 7 | + Conformal (Mondrian per-battery)         | same as #6   | 0.9595  | 3.33   | –      | 0.81 per-bat | 0.04   | Phase 4, from phase4_evaluation.ipynb |
| 8 | + PINN loss (fixed weights)                | same as #6   | 0.959   | 3.35   | –      | –            | –      | Minimal lift; monotonicity ↑violations|
| 9 | + Weibull RUL head (censoring-ignored)     | +RUL output  | 0.9595  | 3.33   | –      | –            | –      | Test had no uncensored cycles         |
| 10| + RAP retrieval (template fallback only)   | same as #6   | –       | 2.34   | –      | –            | –      | LLM path needs OPENAI key [blocked]   |
| 11| **[P5] Bootstrap ensemble (20 Ridge)**     | 6 numeric    | –       | 0.90   | 1.21   | 0.08 ⚠       | 0.705 ⚠| Severely miscalibrated intervals      |
| 12| **[P5] Weighted conformal (NASA → Mend.)** | 6 numeric    | –       | cross-domain | –| 0.18         | –      | LR-corrected vs plain 0.15            |
| 13| **[P5] Source-classifier probe**           | 19 numeric   | –       | –      | –      | –            | –      | ROC-AUC=1.00 ⚠ severe leakage         |
| 14| **[P5] Synthetic-censored survival**       | n/a global   | –       | –      | –      | –            | –      | 195 uncensored + 94 censored cycles   |
| 15| **[P5] Causal (interventional) SHAP**      | 6 numeric    | –       | –      | –      | –            | –      | Capacity disagreement 2.1 vs obs      |
| 16| **[P5] Counterfactuals at target SOH 85**  | 6 numeric    | –       | –      | –      | –            | –      | Lowest-SOH cycles need infeasible Δcap|

⚠ flags rows that surface *problems* we discovered during Phase 5.
These are the rigor deltas: items a reviewer would have caught. We
discovered them first and can talk about them openly.

## Row-by-row narrative

1-3 **Classical ML.** Phase 2 measured three linear/tree baselines on
the 6-feature set. Ridge wins marginally. These remain our tabular
reference.

4 **Production Ridge.** The 19-feature Ridge (`best_model.pkl`) lives
at MAE 0.9 % SOH on the harmonized test split - materially better
than Phase-3's headline 3.3 %. The delta comes from (a) more
features and (b) a different test partition. **Flagging this is
itself a Phase-5 contribution.**

5 **CNN-LSTM.** Phase-1's end-to-end deep baseline. Competitive but
not state-of-art; we still report it to give reviewers a
"numerics-only" lower bound on what cycle-level deep learning can
deliver.

6 **Cross-attention (numeric + SBERT).** The Phase-3 contribution.
Adds a frozen Sentence-BERT text stream, projects to 256 dim, runs
asymmetric cross-attention (numeric queries, semantic keys / values),
gated fusion. This is where most of the novelty claim lives.

7 **+ Conformal.** Mondrian split conformal per battery. On the
Phase-4 test split, empirical coverage is ≈ 0.81 per-battery at
target 0.90 (Phase 4 notebook). On Phase 5's bootstrap ensemble
the plain intervals collapse (row 11) - the two are different
calibration tools on different uncertainty sources.

8 **+ PINN loss.** Small-to-zero MAE lift in the ablation, and
monotonicity *violations increased* from 46.7 % → 53.5 %. Honest
negative result; Tier-1 action item #2 in the rigor report is to
sweep the PINN weights before claiming this as a positive
contribution.

9 **+ Weibull RUL head.** Fit the shape and scale of a Weibull
survival distribution. Test set had zero uncensored samples so the
PIT histogram was empty. Phase 5's synthetic-censoring protocol
(row 14) is the repair.

10 **+ RAP.** The 2.34 % MAE in the Phase-4 notebook is
template-vs-template because no OpenAI key was present. We leave
it in the table with an explicit ⚠ and flag it as *claim pending*
until re-run with a real key.

11 **Bootstrap intervals.** The 20 Ridge bootstraps in
`models/bootstrap_models.pkl` produce intervals that are *far too
narrow*: empirical coverage at target 0.90 is only 0.08, and the
ECE is 0.705. This is exactly the finding Phase 5 was designed to
surface.

12 **Weighted conformal.** On the cross-chemistry NASA → Mendeley
split the plain conformal covers 15 %; the LR-weighted variant
pushes to 18 % and widens intervals by ~20 %. Still not 90 % —
the finding is that the chemistry shift exceeds what any
reweighting can fix on a 6-feature representation.

13 **Leakage probe.** A logistic regression on the raw feature
vector distinguishes NASA from Mendeley with ROC-AUC = 1.00. That
means a fused multimodal model has a trivial shortcut available.
Actionable follow-up: add a domain-adversarial term or balance
batches by source.

14 **Synthetic-censored Weibull.** Closes the Phase 4 empty-PIT
hole with 195 uncensored + 94 censored cycle-level samples. A
global (featureless) MLE gets C-index ≈ 0.46 which is the floor a
feature-aware head has to beat.

15 **Causal SHAP.** Observational vs interventional Shapley
agree on ordering; disagreement concentrates on capacity and
cycle-number (absolute Δ ≈ 2.1 vs 0.3). Conclusion: observational
SHAP over-credits capacity by ~25 % because of its correlation
with cycle_number.

16 **Counterfactuals.** Lowest-SOH cycles would need capacity
increases of +3.4 to +3.5 standardised units to reach target SOH
85 %. Physically impossible, which is the right answer and
demonstrates the search works.

## What's missing from this table

- **Multi-seed CI on rows 6-10.** Phase 3 and Phase 4 did not multi-seed
  the cross-attention training (Tier 1 action item #4). Adding bars here
  is the next rigor step.
- **Leave-one-battery-out.** Every row above uses a time-ordered
  per-battery split. Table needs a LOBO column to survive a domain-shift
  challenge.
- **Full-feature RAP.** Row 10 is on the template fallback. Provide the
  OpenAI key and rerun.

The PHASE5_RIGOR_REPORT.md has the full action list.
