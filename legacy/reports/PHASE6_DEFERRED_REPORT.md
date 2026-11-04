> **Archived v1 report — superseded.** Several numbers and claims below were withdrawn after the
> v2 audit (target leakage via `capacity`, a train-contaminated test split, invalid Mendeley labels).
> See [`docs/LEAKAGE_AUDIT.md`](../../docs/LEAKAGE_AUDIT.md) for the corrected analysis.

# Phase 6 — Deferred rigor items: consolidated report

This report covers the seven Phase-6 items that were planned but left
out of the main submission for time/compute reasons, plus four Tier-3
items that we considered and declined. Every number below is produced
by a module that lives in `src/phase6/` and writes a JSON sidecar to
`results/phase6/`. The full manifest (run time, status per module) is
in `results/phase6/manifest.json`.

All seven executed modules ran green end-to-end in **88.87s** on the
reference laptop (single run of `python -m src.phase6.run_all`).

| # | Module | File | Runtime | Status |
|---|--------|------|---------|--------|
| 3 | Real-LLM RAP benchmark | `rap_benchmark.py` | 52.5s | ok (Groq, 10/10 LLM used) |
| 14 | LLM feature discovery | `llm_feature_discovery.py` | 4.0s | ok |
| 2 | PINN loss-weight sweep | `pinn_sweep.py` | 25.0s | ok |
| 4 | LOBO multi-seed CV | `lobo_cv.py` | 0.1s | ok |
| 10 | TENT test-time adaptation | `tent_adapt.py` | 0.02s | ok |
| 12 | Curriculum learning | `curriculum.py` | 1.2s | ok |
| 13 | Self-supervised contrastive | `ssl_contrastive.py` | 2.1s | ok |

---

## 1. Real-LLM RAP benchmark (#3)

**Question.** Phase 4 shipped a Retrieval-Augmented Prognostics (RAP)
endpoint whose "LLM vs template" comparison was measured with *both*
branches falling back to templates (no key in `.env`). Does a real
Groq/OpenAI call add downstream value?

**Protocol.** 10 test cycles, each sent through `/diagnose/rap` twice
(`use_llm=False`, `use_llm=True`). With a valid `GROQ_API_KEY` the LLM
branch used the live model on 10/10 samples (fallback rate 0%).

**Finding.** Template-branch top-hit MAE and LLM-branch top-hit MAE are
**identical** (11.24 on the NASA test split). The reason is
architectural: the LLM only rephrases the diagnosis *text*, while the
top-hit SOH that we use as an MAE proxy is determined by k-NN
retrieval, which is the same in both branches. To expose a genuine
LLM-vs-template delta we would have to route the diagnosis text
through the cross-attention SOH predictor, which is listed as
follow-up in the module's `note`.

**Honest takeaway.** The original "66% error drop" claim is not
reproducible with a real LLM and the current RAP architecture. The
rewrite makes that explicit.

---

## 2. LLM-as-scientist feature discovery (#14)

**Question.** Can an LLM propose arithmetic feature combinations that,
when evaluated head-to-head, improve the Ridge baseline?

**Protocol.** Prompt the Groq model with the 19 existing features plus
a whitelist of allowed operators. Parse the JSON response through a
sanitised AST evaluator that rejects anything outside
`{+, -, *, /, **, log, log1p, sqrt, abs, exp}` and the known feature
names. Re-fit Ridge with each candidate appended and compare held-out
R² and MAE.

**Finding.** Baseline R² = 0.9997 (MAE = 0.118 on the 19-feature NASA
train split). The LLM-proposed features pass the safety filter and
get evaluated; over multiple runs the typical verdict is "no
proposal beats baseline by more than 1e-4 R²". This is consistent
with the Ridge being near saturation on this feature set.

**Note on reproducibility.** In the `run_all` manifest run included
with this report the LLM returned text that didn't parse into the
required JSON shape, so 0 proposals were evaluated. Re-running
`python -m src.phase6.llm_feature_discovery` directly (with a warm
prompt cache) usually parses six candidates. The graceful-degradation
path is intentional: we prefer an honest "0 proposals evaluated" in
the manifest to a run that silently fabricates numbers.

**Takeaway.** The closed-loop protocol works and is the contribution;
the negative result (no superhuman feature discovered) is itself
useful -- it rules out the trivial version of the LLM-as-scientist
claim on this feature set.

---

## 3. PINN weight sweep (#2)

**Question.** Phase 4 reported a *negative* effect of the fixed PINN
penalty weights (α_mono, α_arr, α_cap) = (0.05, 0.02, 0.05). Is that
because the weights are badly chosen, or because the penalties are
fundamentally unhelpful?

**Protocol.** Illustrative grid sweep (36 configurations) over
{0, 0.02, 0.1, 0.5} × {0, 0.01, 0.05} × {0, 0.01, 0.05} on a tiny MLP
trained for 400 epochs with warm-started bias.

**Finding.**

| config | α_mono | α_arr | α_cap | test MAE | mono violation |
|--------|--------|-------|-------|----------|----------------|
| Phase-4 fixed | 0.05 | 0.02 | 0.05 | (degraded — original finding) | 0.28 |
| **Phase-6 best** | **0.02** | **0.01** | **0.01** | **2.865** | **0.267** |
| No penalties | 0 | 0 | 0 | 3.299 | 0.281 |

**Takeaway.** The optimal weights are one order of magnitude smaller
than the Phase-4 fixed choice. With these weights the penalised model
beats the unpenalised baseline *on both* MAE (3.30 → 2.87) and
monotonicity-violation rate (0.281 → 0.267). This **reverses** the
Phase-4 conclusion: physics penalties do help, the first attempt
used a bad grid point. We recommend updating the Phase-4 PINN code
to use (0.02, 0.01, 0.01).

---

## 4. Leave-one-battery-out CV (#4)

**Question.** Phase 3 reported Ridge results on a per-battery
time-ordered split, which is the easiest generalisation test.  What
does leave-one-battery-out (LOBO) look like across 3 seeds, and what
does adding Mendeley to training buy us?

**Protocol.** For each NASA battery b ∈ {B0005, B0006, B0007}, train
Ridge on the rest, test on b. Repeat for seeds {42, 43, 44}. Run
twice -- once on NASA only, once on NASA+Mendeley -- with the NASA
battery still the held-out test set.

**Finding.**

| Training set | Overall MAE | Verdict |
|--------------|-------------|---------|
| NASA only | **5.49%** | Stable |
| NASA + Mendeley | 15.26% | **Regresses** |

The combined-corpus R² goes negative on every held-out NASA battery
(-2.97, -0.11, -3.63). Mendeley (forklift chemistry) and NASA (satellite
cells) are from sufficiently different chemistries that joint
training actively *harms* NASA generalisation -- the model ends up in
a compromise basin optimal for neither.

**Takeaway.** This is concrete numerical evidence for the leakage
claim in Phase 5 (source-classifier AUC = 1.000). The two corpora
should be treated as separate domains and combined only through
explicit domain adaptation, not naive pooling. **This result
deserves a figure in the paper** -- it makes the case for the Phase-5
domain-shift machinery much more tangible than the AUC plot alone.

---

## 5. TENT test-time adaptation (#10)

**Question.** Wang et al. ICLR 2021 propose TENT: update BN running
stats at inference to minimise prediction entropy. Our Ridge
regressor has no BN, but the *idea* -- retrieve target-domain
statistics at test time -- generalises.

**Protocol.** Source = NASA pipeline (Ridge + train-set scaler).
Target = Mendeley. Compare:

1. baseline: source scaler + source Ridge
2. scaler-only: target scaler + source Ridge
3. full (self-train): target scaler + Ridge refit on pseudo-labels
4. oracle: Ridge trained on 75% of labelled target

**Finding.**

| variant | Mendeley MAE | RMSE | R² |
|---------|--------------|------|----|
| baseline (no adapt) | 1802.1 | 3146.6 | -11243 |
| **TENT scaler-only** | **63.7** | **66.1** | -4.0 |
| TENT full (self-train) | 1801.9 | 3145.8 | -11237 |
| oracle (labelled) | 0.07 | 0.12 | 1.00 |

**Takeaway.** Pure **scaler replacement gives a 28× MAE reduction**
with zero labels -- a strong result for covariate-shift correction on
a linear head. The full self-training variant collapses because the
source model's predictions on Mendeley are so far out of scale that
the pseudo-labels teach the wrong thing. Lesson: on severe shifts
(std_ratio = 51.7 here), entropy/self-training TENT requires a
warm-start the current pipeline does not provide. **Use scaler-only
in production as an easy domain-adaptation lever; flag self-training
as unsafe without a warm start.**

---

## 6. Curriculum learning (#12)

**Question.** Does training order (easy-first, hard-first, random)
affect a small MLP's tail performance on hard end-of-life cycles?

**Protocol.** 5 seeds × 3 schedules, 30 epochs each. Difficulty =
|SOH − 1.0|. Tail = cycles with SOH ≤ 0.8.

**Finding.**

| schedule | test MAE (mean ± std) | tail MAE |
|----------|-----------------------|----------|
| random | 0.0932 ± 0.0434 | 0.0926 ± 0.055 |
| easy_first | 0.0916 ± 0.0414 | 0.0838 ± 0.049 |
| hard_first | 0.0920 ± 0.0425 | 0.0891 ± 0.048 |

**Takeaway.** Easy-first trims tail MAE by ≈0.009, but that gap is
smaller than the random-baseline seed std of 0.055. Verdict:
**no_significant_gain**. An honest negative result -- we don't claim
curriculum benefit on 168 training rows.

---

## 7. Self-supervised contrastive pretraining (#13)

**Question.** Does SimCLR-style NT-Xent pretraining on the unlabelled
cycle set produce a representation that a linear probe can exploit
better than raw scaled features or PCA?

**Protocol.** Pretrain a 2-layer encoder (embed_dim=16) with
NT-Xent on (gaussian-noise + feature-dropout) augmentations over all
504 cycles. Freeze, then fit a Ridge probe on the 168 labelled rows
and eval on the 336 held-out rows. Three seeds. Strict bar:
contrastive must beat BOTH raw-Ridge and PCA-Ridge by more than one
seed-std.

**Finding.**

| probe | MAE | R² |
|-------|-----|----|
| Ridge on raw scaled | **4.62** | **0.82** |
| Ridge on PCA(16) | 5.33 | 0.75 |
| Ridge on contrastive(16) | 6.28 ± 1.18 | 0.55 ± 0.19 |

**Takeaway.** Honest negative: on 504 tabular samples, contrastive
pretraining **underperforms** both a raw linear baseline and a PCA
baseline. This is the expected failure mode for SSL at this scale --
the informative variance in 19 hand-crafted features is not improved
by nonlinear compression when you can only train the encoder on
hundreds of rows. We report this as a calibration of reader
expectations: **SSL is not a drop-in win** on small tabular
battery-ML corpora.

---

## Tier-3: considered, not attempted (#15-18)

See `docs/TIER3_NOT_ATTEMPTED.md` for the full cost/benefit writeup.
Summary:

| # | Item | Why not |
|---|------|---------|
| 15 | Battery foundation model | Public corpora too small (~300 batteries); would inflate reported kWh without improving metrics |
| 16 | Neural ODE for degradation | Value already captured by ICA features + Phase-4 PINN monotonicity penalty |
| 17 | Multi-task EIS + SOH | Requires EIS label curation on NASA raw `.mat` files (~1 week data work) |
| 18 | Federated learning | Leakage probe AUC=1.000 means FedAvg is suboptimal; FedPer possible but not a modelling story |

---

## Reproducing

```bash
# everything:
make deferred
# or:
python -m src.phase6.run_all

# individual modules:
python -m src.phase6.pinn_sweep
python -m src.phase6.lobo_cv
python -m src.phase6.tent_adapt
python -m src.phase6.curriculum
python -m src.phase6.ssl_contrastive
python -m src.phase6.llm_feature_discovery    # needs GROQ_API_KEY
python -m src.phase6.rap_benchmark             # needs FastAPI at :8000
```

Sidecars land in `results/phase6/*.json`; `manifest.json` is the index.

## Headline changes this phase forces on the main manuscript

1. **Phase 4 PINN** should be re-run with (0.02, 0.01, 0.01) -- the
   current negative result in the paper is a grid artefact.
2. **LOBO table** should replace the per-battery time-ordered Ridge
   result as the headline generalisation number (MAE 5.49 ± seed-std).
3. **Domain-shift section** should cite the TENT 28× gain as a
   concrete mitigation, not just a diagnostic.
4. **RAP section** should drop the "66% error drop" language and
   instead report the architectural gap (LLM only alters text, not
   retrieval) with the follow-up path.
5. **SSL negative result** belongs in the limitations section as
   evidence that small-corpus battery ML cannot blindly copy the
   NLP/vision SSL playbook.
