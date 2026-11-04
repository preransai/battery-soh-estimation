# Methodology

This document defines the estimation task, the labels, the data rules, the health indicators,
the evaluation protocol and the uncertainty method. Numbers quoted here come from
[`reports/benchmark/summary.json`](../reports/benchmark/summary.json) (the benchmark),
[`data/processed/nasa_features.csv`](../data/processed/nasa_features.csv) and its
[`.meta.json`](../data/processed/nasa_features.meta.json) sidecar. The complete generated
tables are in [RESULTS.md](RESULTS.md).

## Contents

1. [Task](#1-task)
2. [Data](#2-data)
3. [Labels](#3-labels)
4. [Capacity quality control](#4-capacity-quality-control)
5. [Which cells are used](#5-which-cells-are-used)
6. [Health indicators](#6-health-indicators)
7. [Absolute and relative mode](#7-absolute-and-relative-mode)
8. [Evaluation protocol](#8-evaluation-protocol)
9. [Nested feature selection](#9-nested-feature-selection)
10. [Conformal prediction intervals](#10-conformal-prediction-intervals)
11. [Statistics](#11-statistics)
12. [Pre-specified headline](#12-pre-specified-headline)
13. [Window sweep](#13-window-sweep)
14. [Reproducibility](#14-reproducibility)

## 1. Task

Estimate the state of health (SOH) of a lithium-ion cell at a given discharge cycle from
**health indicators (HIs) that can be measured during part of a cycle**:

- a voltage window of the constant-current (CC) charge that precedes the discharge, and the
  constant-voltage (CV) phase that follows it;
- the 3.80 → 3.50 V portion of the discharge under load, and the voltage step at load-on;
- the most recent electrochemical impedance spectroscopy (EIS) measurement before the cycle.

The inputs never include the discharge capacity or any statistic of the complete discharge
record. The output is SOH in percent with a prediction interval at 80, 90 or 95 % confidence.

Why partial windows: a deployed cell is rarely discharged completely under controlled
conditions, and when it is, that discharge measures capacity (and therefore SOH) directly.
An estimator is only useful if it works from less than that. The v1 version of this project
used full-discharge inputs; [LEAKAGE_AUDIT.md](LEAKAGE_AUDIT.md) explains what that did to its
numbers.

## 2. Data

The NASA Ames Prognostics Center of Excellence (PCoE) Battery Data Set (Saha and Goebel,
2007): 34 commercial 18650 Li-ion cells with a 2.0 Ah nameplate capacity, cycled in several
campaigns. Every cell was charged the same way (CC 1.5 A to 4.2 V, then CV at 4.2 V until the
current falls to 20 mA; `battery_soh.data.protocols.CHARGE_PROTOCOL`); the campaigns differ
in ambient temperature, discharge load and cut-off voltage. Some cells also have periodic EIS
measurements.

The raw `.mat` files are not redistributed; `soh data download` fetches the official archive
and verifies its SHA-256. The derived per-cycle table is committed (2,794 rows x 33 columns,
one row per discharge). See the [data card](DATA_CARD.md) for the full datasheet.

## 3. Labels

Implementation: [`src/battery_soh/labels.py`](../src/battery_soh/labels.py).

| Column | Definition | Role |
|---|---|---|
| `soh_pct` | `100 * capacity_ah / c0_ah`, with `c0_ah` the **median of the cell's first 5 QC-valid discharge capacities** | headline label |
| `soh_rated_pct` | `100 * capacity_ah / 2.0` (nameplate), the NASA README's convention | reported in the table, not benchmarked |

The own-reference label measures each cell's *fade*. Cells leave the factory with different
capacities: the reference capacities of the four core cells are 1.835, 2.013, 1.881 and
1.840 Ah (B0005, B0006, B0007, B0018). A rated-capacity label would mix that manufacturing
spread into the target. The median (rather than v1's mean) makes `C0` robust to one odd early
cycle. An 80 % end-of-life threshold (`EOL_SOH_PCT`) is used in plots only, never in training.

The label needs a complete discharge. That is acceptable for ground truth in a laboratory
dataset; it is the reason capacity can never be an input.

## 4. Capacity quality control

Implementation: [`src/battery_soh/qc.py`](../src/battery_soh/qc.py). The README files of the
dataset warn about discharges with very low capacity, and the files contain truncated records
and formation cycles. Every discharge gets a validity flag and a reason; the first matching
rule wins:

1. `missing`: no finite capacity recorded.
2. `below_min`: capacity below 0.3 Ah (15 % of nameplate), physically implausible here.
3. `isolated`: fewer than 3 plausible capacities in the centred 9-cycle window, so the value
   cannot be checked against neighbours.
4. `hampel_outlier`: more than 0.15 Ah from the median of the plausible capacities in a
   centred 9-cycle window (a Hampel-style filter with a fixed absolute tolerance; a
   `k * MAD` band would reject the genuine capacity recovery after rest periods because the
   within-window MAD is a few mAh).
5. `ok`.

Labels are computed for every row so the table is complete, but training, calibration and
scoring use `qc_valid` rows only. All 636 discharges of the four core cells pass; 2,516 of the
2,794 rows pass overall.

## 5. Which cells are used

Implementation: [`src/battery_soh/data/protocols.py`](../src/battery_soh/data/protocols.py).
Inclusion is decided by an explicit rule whose quantities are measured on the raw files. A cell
is usable when it passes all five criteria:

| Criterion | Requirement | Threshold |
|---|---|---|
| `protocol` | one constant discharge current at one ambient temperature; the load is never switched off while on | at most 10 % near-zero current samples under load |
| `history` | enough QC-valid discharges | at least 30 |
| `reference` | `C0` is trustworthy: few cycles far above it, and close to the campaign-group median | at most 10 % of valid cycles above 105 % SOH; `C0` within 25 % of the group median |
| `windows` | a partial window is observable | 3.80 → 3.50 V discharge or 4.00 → 4.19 V charge window in at least 50 % of valid cycles |
| `labels` | the SOH trajectory carries signal | robust cycle-to-cycle label noise at most 0.75 SOH points; total fade at least 5 points |

The thresholds sit in gaps of the measured distributions, i.e. they were chosen after
inspecting them; for example, label noise is 0.23-0.54 points for every cell that passes the
other criteria and 0.97 or more for every cell that fails on noise (module docstring). Writing
the rule down makes the inclusion reproducible and auditable, not blind. Result:

- **Core set** (headline): B0005, B0006, B0007, B0018. 2 A constant-current discharge at
  24 °C, 132-168 cycles each.
- **Extended set**: the core cells plus B0036 (2 A, 24 °C), B0029-B0032 (4 A, 43 °C) and
  B0046-B0048 (1 A, 4 °C): 12 cells in three operating conditions, 1,193 QC-valid cycles.
- **Excluded** (22 cells, each with a recorded reason): square-wave loads (B0025-B0028), a
  formation anomaly and erratic capacity at 4 A / 24 °C (B0033, B0034), mixed loads
  (B0038-B0044), an anomalous cell (B0045), a control-software crash after 25 discharges
  (B0049-B0052) and label noise of 1.15-2.31 points at 2 A / 4 °C (B0053-B0056). The
  per-cell reasons are in `PROTOCOLS` and in the
  [processed-data README](../data/processed/README.md#row-counts).

The fade depth differs a lot even inside the core set. The lowest SOH reached is 70.1 %
(B0005), 57.3 % (B0006), 74.5 % (B0007) and 72.9 % (B0018). B0006 therefore covers a range
that no other core cell reaches, which matters for the headline result (section 9).

## 6. Health indicators

Implementation: [`src/battery_soh/features/`](../src/battery_soh/features/). Every feature is
declared in [`registry.py`](../src/battery_soh/features/registry.py) with its unit, family,
availability, leakage risk and a physical plausibility range that the API enforces. Per-cycle
failures return NaN, never a placeholder.

| Feature | Unit | Window | Physical rationale |
|---|---|---|---|
| `chg_cc_time_390_419_s` | s | CC charge, 3.90 → 4.19 V (needs the CC phase to start below 3.85 V) | At constant current the time spent in a voltage window is proportional to the charge stored in it; it shrinks with capacity fade. |
| `chg_cc_time_400_419_s` | s | CC charge, 4.00 → 4.19 V (start below 3.97 V) | Same quantity on a narrower window that more cells reach; used by the extended sets. |
| `chg_cv_time_s` | s | from the first sample at 4.19 V until the current falls below 0.1 A | Higher resistance and lower capacity end the CC phase earlier and lengthen the CV taper. |
| `chg_ica_peak_height_ah_per_v` | Ah/V | dQ/dV of the CC charge on a 5 mV grid, 3.90-4.18 V | The incremental-capacity peak marks an electrode phase transition; its height falls with loss of active material and lithium inventory. |
| `chg_ica_peak_voltage_v` | V | same | The peak shifts with polarisation (resistance growth). |
| `chg_temp_rise_cc_c` | °C | CC charge, 3.90 → 4.19 V | Joule heating grows with resistance. |
| `dis_q_380_350_ah` | Ah | discharge under load, 3.80 → 3.50 V (about 40 % of a 2 A discharge) | Charge delivered in a fixed voltage window scales with usable capacity. |
| `dis_time_380_350_s` | s | same | Duration of that window (in the table; not in the model feature sets). |
| `dis_r0_ohm` | Ω | voltage step at load-on, `dV / dI` across the switching sample pair | Ohmic resistance rises with ageing. |
| `dis_temp_rise_win_c` | °C | discharge, 3.80 → 3.50 V | Heating under load grows with resistance. |
| `eis_re_ohm`, `eis_rct_ohm` | Ω | most recent EIS fit *before* the cycle (as-of join, no back-fill) | Electrolyte and charge-transfer resistance. |
| `ctx_ambient_c`, `ctx_discharge_current_a` | °C, A | protocol metadata, median load current | Operating condition (extended sets only). |
| `ctx_cycle_index` | cycles | count of previous discharges | Used only by the `cycle_only` baseline. |

Implementation details that matter for correctness:

- **Crossings and integrals.** Window boundaries are interpolated crossing times; charge is
  the integral of the measured current between them, so `dis_q_380_350_ah` is exact under
  constant current and still correct under a varying load. Only samples under load enter a
  discharge window, so the voltage relaxation after cut-off cannot leak in.
- **ICA.** Computed on the CC samples only (CC onset to the CC → CV transition). The voltage
  trajectory is made monotone with an isotonic fit, charge is interpolated onto a uniform
  5 mV grid, and the derivative is a Savitzky-Golay filter (window 9, order 2) on that grid.
  The peak is the highest interior local maximum when it is within 3 % of the window maximum,
  otherwise the window maximum, reported at the window edge (censored) rather than as NaN.
- **Charge pairing.** A discharge is paired with a charge only when the operation right before
  it (ignoring EIS) is a charge; otherwise its charge features are NaN.
- **Train/serve parity.** The table builder and the API endpoint
  `POST /v1/soh/estimate-from-cycle` both call
  [`features_from_segments`](../src/battery_soh/features/extract.py), so a deployed model
  never sees differently computed inputs. A synthetic cell simulator with analytic ground truth
  ([`synthetic.py`](../src/battery_soh/synthetic.py)) tests every extractor against exact
  answers.

Feature sets (registry `FEATURE_SETS`): `charge` (5 charge HIs, 3.90 V window), `discharge`
(`dis_q_380_350_ah`, `dis_r0_ohm`, `dis_temp_rise_win_c`), `combined` (both, 8 HIs),
`combined_eis` (plus the two EIS values), `charge_ext` and `combined_ext` (4.00 V window plus
ambient temperature and load current, for the multi-condition sets). `charge_no_cv` and
`combined_no_cv` are **post-hoc** ablations defined after the leave-one-battery-out results
showed the CV-time problem (section 9); they are labelled `post_hoc` wherever they appear.

Availability on the core cells (QC-valid rows): discharge-window features are never missing;
charge features are missing in 1-4 % of cycles (no preceding charge, or a top-up charge that
starts above 4.19 V), except the 3.90 V window in B0006, which is missing in 28 % of cycles
because its CC phase often starts above 3.85 V. Details and the extended cells' gaps are in
the [processed-data README](../data/processed/README.md#coverage-and-known-gaps).

## 7. Absolute and relative mode

**Absolute mode** feeds the HIs as measured.

**Relative mode** divides each HI by the same cell's *commissioning reference*: the mean of
its first five QC-valid, finite values
([`features/relative.py`](../src/battery_soh/features/relative.py)). Context features pass
through unchanged. The reference values are indicators from the cell's first few cycles, never
its capacity, so a reference is available in deployment if the cell is measured when it enters
service; the API takes it as the `reference` field.

Which cycles enter the reference is decided by the `qc_valid` flag, and that flag is computed
from the measured capacity (section 4), which a deployed system would replace by its own
commissioning check. On the four core cells every cycle passes QC, so the headline and
relative-mode results do not depend on it; on the extended cells a cycle rejected by capacity
QC is skipped when the reference is formed.

Rationale: cells differ at birth, and so do their HIs. Dividing by the commissioning value
removes that offset and leaves the change caused by ageing, which is what the own-reference
label measures. It also removes cell-specific offsets such as B0006's longer CV phase
(section 9).

Rules:

- Rows whose values entered a reference are flagged `is_reference_cycle` and are excluded
  from training, calibration and scoring (24 of the 636 core cycles for the `combined` set).
  `mae_non_reference` in `summary.json` gives the absolute-mode error on the same
  non-reference cycles, for a like-for-like comparison.
- Temperature rises are interval-scale and their references can be near zero or negative,
  where a ratio is ill-conditioned: the charge-rise references of the cold cells and B0036 lie
  between -1.47 and -0.16 °C, and the discharge-rise references are +1.39 °C for B0036,
  +0.54 °C for B0046 and within 0.05 °C of zero for B0047 and B0048. In the
  extended experiments they stay absolute (`relative_keep_absolute` in
  [`configs/benchmark.yaml`](../configs/benchmark.yaml)); this was decided from the reference
  values, which are features only, before any extended result existed.
- Relative mode has a hard requirement: a commissioning record under the same protocol. It is
  not available for cells of unknown history.

## 8. Evaluation protocol

Implementation: the `evaluation.benchmark` package:
[`runner.py`](../src/battery_soh/evaluation/benchmark/runner.py) runs the experiments,
[`frames.py`](../src/battery_soh/evaluation/benchmark/frames.py) builds the scored rows and runs
the leakage gate, and [`fitting.py`](../src/battery_soh/evaluation/benchmark/fitting.py) fits
every outer fold and asserts its disjointness. Configuration:
[`configs/benchmark.yaml`](../configs/benchmark.yaml).

**Splits.** Whole groups are held out:

| Experiment | Cells | Held-out unit | Folds |
|---|---|---|---:|
| `core_lobo` (headline) | 4 core cells | battery | 4 |
| `extended_lobo` | 12 extended cells | battery | 12 |
| `leave_one_condition_out` | 12 extended cells | operating condition | 3 |

On every fold `assert_disjoint` checks that no `(battery_id, cycle_index)` key is in both
training and evaluation rows, and `assert_groups_disjoint` checks the groups. Everything
learned from data is learned inside the training fold: median imputation, scaling, feature
selection, the ridge penalty and the conformal calibration. The held-out cell never enters
any of them.

**Scored rows.** A specification (experiment x feature set x mode x model) is fitted and
scored on the QC-valid cycles with a finite label where at least one of its own measured
inputs is present. Other cycles are *abstentions*, counted in `n_abstained` and not imputed
(for example 8 of the 636 core cycles for the `charge` set). In relative mode the reference
cycles are excluded as described above.

**Models.** `ridge`, `ridge_nfs` (Ridge with nested feature selection, section 9), `pls`,
`elasticnet`, `gpr`, `random_forest`, `hgb`, `mlp` and `knn`, each a scikit-learn pipeline
of median imputation, standardisation (skipped for trees) and the estimator
([`models/zoo.py`](../src/battery_soh/models/zoo.py)). Hyper-parameters are fixed small-data
defaults; they are not tuned on the evaluation cells, because a nested grouped search over
four cells would mostly fit noise. Baselines: the training mean, Ridge on the cycle count
(`cycle_only`), and Ridge on capacity (`capacity_oracle`, **leaky, a sanity check only**).

**Metrics.** In SOH percentage points:

- `mae`, `rmse`, `bias`: computed per held-out cell, then averaged over cells, so each cell
  counts once regardless of its cycle count. `worst_mae` is the worst cell. Pooled variants
  (`mae_pooled`) are also recorded.
- `r2`: pooled over all scored out-of-fold predictions.
- Interval `coverage_*` and `mean_width_*` at 80, 90 and 95 %: mean over cells (pooled values
  in the `coverage` table).

MAE in SOH points is the primary metric. R² is reported but is not an accuracy: it depends on
the spread of the label across the pooled cells as much as on the error.

## 9. Nested feature selection

Implementation: [`evaluation/nested.py`](../src/battery_soh/evaluation/nested.py).

With four cells, choosing features by their leave-one-battery-out *test* score would leak the
test cells into the model. `ridge_nfs` therefore selects its features and penalty inside each
outer training fold, using only the training cells:

1. Score a feature subset by the inner leave-one-battery-out MAE over the training cells
   (mean of the per-cell MAEs), minimised over the penalty grid `alpha in {0.1, 1, 10}`.
2. Starting from the full set, try removing each feature; keep the removal with the lowest
   score if it improves the score by more than `1e-9`; repeat until no removal helps (at
   least one feature remains).
3. Refit Ridge on all training rows with the selected features and penalty.

The outer error of the whole procedure is then a nearly unbiased estimate of what
selection-plus-fitting does on a new cell (Varma and Simon, 2006), conditional on the design
choices (indicator windows, feature sets, the penalty grid) that were made with knowledge of
these cells. With four outer folds it is also very noisy.

**What happened in the headline run** (`selection_stability.headline`). The ICA peak height
and `dis_q_380_350_ah` were selected in all four folds, the charge temperature rise in none;
the mean pairwise Jaccard similarity of the selected sets is 0.49. The fold that holds out
B0006 kept six features, including the CV-phase duration `chg_cv_time_s`, with an inner MAE
of 0.66 on B0005, B0007 and B0018, and then scored MAE 6.45 on B0006.

The cause is a cell-specific offset. B0006's commissioning CV duration is 4,111 s against
3,738-3,928 s for the other three cells (`extended.reference_values` in `summary.json`), and
B0006 fades far below the others (section 5). Inner leave-one-battery-out over three cells
that share neither property cannot detect that the CV feature will mislead on the fourth.
A post-hoc ablation without the CV time, labelled as such, reduces the B0006 error of plain
Ridge on the combined set from 5.83 to 2.45 and the mean from 2.60 to 1.65; it is reported
as a diagnosis, not as a result, because the ablation was defined after seeing the outcome.
Relative mode removes the offset without any post-hoc choice (B0006: 1.26 with nested
selection).

## 10. Conformal prediction intervals

Implementation: [`uncertainty/conformal.py`](../src/battery_soh/uncertainty/conformal.py).

**Calibration unit.** Batteries, not cycles, are the exchangeable units: cycles of one cell
are strongly dependent, and a deployed model always meets a new cell.
`GroupConformalRegressor` produces its calibration scores by an inner leave-one-battery-out
loop over the *training* cells, so each score is the absolute error of a model that never saw
that cell.

**Two constructions** are evaluated:

- `cv`: the full-data model's prediction plus or minus `q`, the exact conformal quantile of
  the pooled out-of-battery residuals. This is what the production bundles ship (a fixed
  half-width per confidence level). The full model is not one of the fold models, so this
  construction has no finite-sample guarantee of its own.
- `cv_plus`: group CV+ (Barber, Candes, Ramdas and Tibshirani, 2021), which combines every
  fold model's prediction with that fold's residuals. Its `1 - 2 alpha` worst-case guarantee
  holds for exchangeable units, here cells, and needs enough of them (below).

**Exact ranks.** `q` is the `k`-th smallest score with `k = ceil((n + 1)(1 - alpha))`,
computed in rational arithmetic. When `k > n` there are too few scores for the requested
level and `q = +inf`; v1 returned the largest score instead, which silently understates the
half-width.

**No battery-level guarantee here.** The scores are pooled over cycles, so `n` is in the
hundreds and `k <= n`, but the exchangeable units are cells, not cycles. Counted in cells, a
90 % finite-sample guarantee needs `ceil((n + 1) * 0.9) <= n`, that is at least nine
calibration cells; each fold has three. Neither construction therefore has a battery-level
finite-sample guarantee in this benchmark, even for a new cell that is exchangeable with the
calibration cells (same chemistry, protocol and rig). Coverage is an empirical property,
measured on held-out cells, and the headline shows it failing:

| Held-out cell | 90 % coverage (`cv`) | Mean 90 % width | Why |
|---|---:|---:|---|
| B0005 | 1.00 | 16.38 | B0006's large residuals are in the calibration set and widen the band |
| B0006 | 0.00 | 3.69 | the three calibration cells agree with each other, so the band is narrow exactly where the error is largest |
| B0007 | 1.00 | 10.96 | |
| B0018 | 0.95 | 8.08 | |
| Mean | 0.74 | 9.78 | `cv_plus`: 0.75 |

In relative mode, where the per-cell offset is removed, coverage is closer to nominal: 0.93
at 90 % with nested selection (plain Ridge: 0.87 with `cv`, 0.91 with `cv_plus`), though
B0006 is still the worst-covered cell. Nothing here covers a cell from a different population;
the leave-one-condition-out intervals (two calibration conditions per fold) are too wide to
be informative.

## 11. Statistics

- **Bootstrap over batteries.** Confidence intervals for the extended-set MAE resample the
  per-cell MAEs of whole cells (2,000 resamples, 95 %), because cells, not cycles, are the
  independent units. They describe the between-cell spread conditional on the fitted models
  (nothing is refitted per resample) and ignore that cells cluster by operating condition, so
  they are rough.
- **Paired test.** On the extended set, Ridge (reference) is compared with the better of two
  candidates (GPR, HGB) by a two-sided Wilcoxon signed-rank test on the 12 per-cell MAE pairs.
  Picking the better candidate on the same folds favours it (`selection_note` in the
  summary), and four such comparisons were run (two feature sets x two modes). The summary
  stores uncorrected p-values; this documentation reports a Bonferroni factor of 2 for the
  candidate choice next to them and treats a single p below 0.05 among four tests as
  suggestive, not established. That one comparison (HGB against Ridge, `combined_ext`,
  relative mode: p = 0.042, 0.085 corrected) is also library-sensitive: under scikit-learn
  1.9.1 instead of the pinned 1.8.0, HGB's numerics change and the same test gives p = 0.021
  (0.042 corrected), with Ridge better on 2 cells instead of 4.
- **No significance tests on the core set.** With four paired cells the smallest attainable
  two-sided Wilcoxon p-value is 2/16 = 0.125, so no difference between models on the core
  cells can be called significant. Core-set comparisons are reported as effect sizes with
  per-cell values.

## 12. Pre-specified headline

The headline specification is fixed in the `headline` block of
[`configs/benchmark.yaml`](../configs/benchmark.yaml):

```yaml
# DECLARED HEADLINE (fixed 2026-10-12 before any ridge_nfs result; plain-ridge results for this set/mode were known)
# Do not change it in response to results.
headline:
  experiment: core_lobo
  feature_set: combined
  mode: absolute
  model: ridge_nfs
```

together with the nested-selection procedure (`nested_selection`) and a list of secondary
rows. The summary repeats it under the `pre_registration` key (the name is kept for
compatibility) with the statement that the headline is reported whatever it shows, and
records the SHA-256 of the configuration sections that affect benchmark results
(`config_sha256`; the sections are listed under `provenance.config_sha256_sections`) and of
the feature table (`data_sha256`). Each model bundle records that benchmark hash plus a
separate hash of the `production` section. Feature sets defined after seeing results carry
`post_hoc: true`.

**What was known when the headline was fixed.** `combined` indicators, absolute mode, Ridge
with nested feature selection (`ridge_nfs`): this was written into `configs/benchmark.yaml`
before any `ridge_nfs` result was computed, and it is reported unchanged. It was not chosen
blind to these four cells: the indicator windows, feature sets and relative mode come from an
audit prototype that had already scored plain Ridge leave-one-battery-out on them (MAE about
1.6-1.7), and plain-Ridge results for this same `combined`/absolute specification on the v2
table (2.60; B0006 5.83), the no-CV-time ablation and relative mode were known when the
headline was fixed. The declaration guards against picking the best of the 114 leaderboard
rows afterwards; it does not make the design independent of these cells.

**Limits.** The declaration is inside the repository, not in an external registry with an
independent timestamp. The configuration hash shows which configuration produced
`summary.json`, not when it was written; treat this as a commitment made before the
headline's own result was computed, not as a blinded pre-registration. It covers the headline
row only; the leaderboard is exploratory.
[ADR 0006](adr/0006-pre-specified-headline-and-nested-selection.md) records the decision.

## 13. Window sweep

Implementation: [`evaluation/benchmark/window_sweep.py`](../src/battery_soh/evaluation/benchmark/window_sweep.py).
An exploratory sensitivity analysis (`window_sweep` in the summary): how much of a cycle does
an estimator need? For each voltage window, Ridge on two single-window HIs (charge or time in
the window, and the temperature rise) is evaluated leave-one-battery-out on the core cells.
The x-axis is the mean fraction of the cell's capacity that passes through the window. The
registry windows were fixed before the sweep and are not selected from it. Eleven windows are
scored on the same four cells, so the lowest error among them is optimistic: a best-of-11 on
the evaluation cells, not an estimate for a new cell. The sweep needs the raw `.mat` files;
without them the committed `window_sweep.json` is reused. Seven of the eleven windows:

| Window | Mean capacity fraction | MAE absolute | MAE relative |
|---|---:|---:|---:|
| discharge 3.80 → 3.70 V | 0.10 | 2.15 | 1.61 |
| discharge 3.80 → 3.60 V | 0.23 | 1.28 | 1.33 |
| discharge 3.80 → 3.50 V (registry) | 0.41 | 1.60 | 0.95 |
| discharge 3.80 → 3.40 V | 0.63 | 1.65 | 1.19 |
| discharge 3.90 → 3.50 V | 0.51 | 2.16 | 0.97 |
| CC charge 3.90 → 4.19 V | 0.54 | 3.46 | 2.04 |
| CC charge 4.00 → 4.19 V | 0.36 | 6.70 | 4.47 |

Windows starting at 3.90 V on discharge are observable in 85 % of core cycles (44 % for
B0006), the 3.80 V windows in all of them. In absolute mode the lowest error comes from the
3.80 → 3.60 V window (MAE 1.28, the optimistic best-of-11 above), which covers about a
quarter of the capacity; in relative mode from the registry window. Observing more of the discharge does not reduce the error monotonically. In
absolute mode every charge window (MAE 3.46-6.70) is worse than every discharge window
(1.28-2.64); in relative mode the 3.80 → 4.19 V charge window reaches 1.12, but it is
observable in only 45 % of core cycles.

## 14. Reproducibility

- Seed 42 for everything stochastic (`battery_soh.config.RANDOM_SEED`).
- `summary.json` records the configuration and data hashes, package and library versions
  (Python 3.11.9, NumPy 2.4.4, pandas 3.0.2, SciPy 1.17.1, scikit-learn 1.8.0, Matplotlib 3.10.8)
  and the run time (`runtime_s`: 537 s for the committed full run on a multi-core workstation).
- `requirements/constraints.txt` pins those library versions; install with
  `python -m pip install -e ".[dev]" -c requirements/constraints.txt`. With them, reruns
  produce identical files apart from `generated_utc` and the run time. With other versions
  the linear models and the baselines reproduce exactly, but tree ensembles (HGB, random
  forest) can differ slightly, which is enough to move the one borderline p-value of
  section 11.
- The feature table is rebuilt byte-identically from the same raw files
  (`tests/integration/test_feature_table.py`, raw-data marker).

## References

- B. Saha and K. Goebel (2007). *Battery Data Set*. NASA Ames Prognostics Data Repository,
  NASA Ames Research Center, Moffett Field, CA.
- S. Varma and R. Simon (2006). Bias in error estimation when using cross-validation for model
  selection. *BMC Bioinformatics* 7:91.
- R. F. Barber, E. J. Candes, A. Ramdas and R. J. Tibshirani (2021). Predictive inference with
  the jackknife+. *Annals of Statistics* 49(1).
- V. Vovk, A. Gammerman and G. Shafer (2005). *Algorithmic Learning in a Random World*.
  Springer.
