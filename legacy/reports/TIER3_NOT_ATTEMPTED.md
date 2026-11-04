> **Archived v1 report — superseded.** Several numbers and claims below were withdrawn after the
> v2 audit (target leakage via `capacity`, a train-contaminated test split, invalid Mendeley labels).
> See [`docs/LEAKAGE_AUDIT.md`](../../docs/LEAKAGE_AUDIT.md) for the corrected analysis.

# Tier-3 items: considered, not attempted

This document enumerates four directions we sketched during the Phase-6
planning round but deliberately did **not** execute. Listing them with a
concrete cost/benefit is more useful than a hand-wave "future work"
paragraph, and it shows the reader we are aware of the real frontier
rather than claiming to cover it.

---

### 1. Battery Foundation Model (BFM) pretraining

**What it would be.** A transformer or SSM pretrained on millions of
cycle sequences from a union of public datasets (NASA, Stanford-HST,
Sandia, Oxford, Mendeley, MIT-Stanford) with a masked-cycle or
next-cycle objective, then fine-tuned per task.

**Why not now.**
- The union of public sets we can actually obtain is ~300 batteries /
  low-six-figure cycles after QC, which is 2-3 orders of magnitude too
  small for a foundation-scale pretraining run to produce a real
  improvement over the PCA baseline (see Phase-6 SSL result: on 504
  cycles the contrastive probe *underperforms* raw Ridge).
- Compute: a serious BFM pretraining needs days on at least one A100.
  Our carbon logger budget is set at "laptop + 100 epochs" and a BFM
  run would single-handedly dominate the reported kWh.
- The honest ablation this project already ships (causal Shapley +
  weighted conformal + LOBO CV) is more informative for a safety-
  critical SOH product than a foundation model would be.

**If we were to attempt it.** The path is (a) harmonise 6+ public
datasets to the same feature schema (partially done in Phase 3),
(b) self-supervised pretext at the cycle-sequence level with a masked-
step loss, (c) linear probe for SOH, RUL, internal-resistance, EIS
reconstruction. Minimum credible budget: ~2 GPU-weeks.

---

### 2. Neural ODE for continuous-time degradation

**What it would be.** Replace the discrete cycle-to-cycle Ridge with
`dh/dt = f_theta(h, x, t)` where `h` is latent health and `x` are
per-cycle covariates. Trained with torchdiffeq's adjoint backprop.

**Why not now.**
- Our cycles are already aggregated (one row = one cycle) so the
  "continuous time" part would be largely synthetic. The real signal
  an ODE would capture -- within-cycle voltage/current dynamics -- is
  better handled by the ICA / IC-curve features we already extract in
  Phase 2.
- The SOH monotonicity constraint is more naturally enforced as a
  linear penalty (which our Phase-4 PINN already does) than as a sign
  constraint on a learned vector field.

**If we were to attempt it.** Likely as an extension of the Weibull
survival head: replace the time-to-event parametric form with a
learned ODE for hazard, trained on the same right-censored samples.

---

### 3. Multi-task EIS + SOH joint prediction

**What it would be.** Add an auxiliary head that predicts EIS spectra
(Nyquist plot parameters, or raw impedance at selected frequencies)
from the same shared encoder. Task loss = SOH-MSE + lambda *
EIS-spectrum-MSE. Motivated by Zhang et al. 2020, who showed EIS is a
strong aging proxy.

**Why not now.**
- Our combined corpus has no EIS labels. NASA B0005-B0007 does include
  impedance sweeps, but matching them to cycle-level features would
  require re-curating the raw `.mat` files -- roughly a week of data
  work for a result that would only validate on one source.
- The multi-task scaffolding (shared encoder + heads) is already
  present in the cross-attention architecture. Adding EIS is 2 days
  of engineering **once the labels exist**.

**If we were to attempt it.** Start from NASA raw impedance files,
extract R_ohm / R_ct / C_dl from Nyquist fits, train jointly with SOH
on the NASA subset, then probe whether the Mendeley-transfer
performance improves vs the Phase-5 baseline.

---

### 4. Federated learning across fleet operators

**What it would be.** Split NASA vs Mendeley vs simulated-Sandia as
three "operators" with local data, run FedAvg (McMahan 2017) or a
personalised variant (FedPer, Ditto) so each operator keeps a model
adapted to their chemistry without sharing raw cycles.

**Why not now.**
- The Phase-5 leakage probe already shows the three domains are
  trivially distinguishable (source classifier AUC = 1.000), so a
  single federated average will produce a model that is suboptimal on
  every participant. Personalisation partially fixes this but
  introduces a per-operator hyperparameter.
- Federated learning's value proposition is *privacy*, which is a
  regulatory/engineering story, not a modelling one. Our project
  ships as an open research artifact, so the infrastructure cost
  (federation framework, secure aggregation, communication-round
  accounting) pays back no concrete downstream metric.

**If we were to attempt it.** Use Flower (flwr) with the existing
Ridge pipeline, three clients (NASA, Mendeley, a held-out synthetic
shard), compare FedAvg and FedPer against the Phase-5 pooled model
using per-client held-out MAE. Expected outcome: FedPer matches
pooled, FedAvg underperforms on the minority corpus.

---

## How to read this list

These four items are **not** claimed as future work in the paper's
"Limitations" section without this stub behind them. Listing them
here with the honest reason-for-deferral is the difference between
rigor and handwaving, and it is also a useful checklist if the project
is picked up by a future team.
