# Model card: shipped SOH bundles

An overview, after Mitchell et al. (*Model Cards for Model Reporting*, 2019), of the model
bundles the service ships in [`artifacts/models/`](../artifacts/models/). Each bundle also
has its own generated card with its exact inputs, coefficients, per-cell metrics and
interval half-widths:

| Bundle | Card |
|---|---|
| `combined-absolute-ridge` | [model_cards/combined-absolute-ridge.md](../artifacts/models/model_cards/combined-absolute-ridge.md) |
| `discharge-absolute-ridge` | [model_cards/discharge-absolute-ridge.md](../artifacts/models/model_cards/discharge-absolute-ridge.md) |
| `charge-absolute-ridge` | [model_cards/charge-absolute-ridge.md](../artifacts/models/model_cards/charge-absolute-ridge.md) |
| `combined-relative-ridge` | [model_cards/combined-relative-ridge.md](../artifacts/models/model_cards/combined-relative-ridge.md) |

`GET /v1/models/{model_id}` returns the same information from the running service.

## Model details

- **What it estimates.** State of health of a lithium-ion cell in percent of its own initial
  capacity (`soh_pct = 100 * capacity / C0`, `C0` the median of the first five QC-valid
  discharge capacities), with a battery-level conformal interval at 80, 90 or 95 % whose
  coverage is measured on held-out cells, not guaranteed.
- **Model.** Median imputation, standardisation and Ridge regression, exported to JSON and
  evaluated with NumPy ([ADR 0003](adr/0003-json-model-bundles-no-pickle.md)). Inputs and
  the ridge penalty are chosen by the nested backward elimination of the pre-specified
  headline procedure (`ridge_nfs`, `production.model` in
  [`configs/benchmark.yaml`](../configs/benchmark.yaml)), run on all four core cells; each
  card records the selected inputs and the selection path.
- **Training data.** The QC-valid cycles of NASA cells B0005, B0006, B0007 and B0018 (2 A
  constant-current discharge at 24 °C) on which at least one of the bundle's indicators was
  measured. In relative mode the indicators are divided by each cell's commissioning reference
  and the reference cycles are not fitted. See the [data card](DATA_CARD.md).
- **Intervals.** Fixed half-width per confidence level: the exact conformal quantile of the
  pooled leave-one-battery-out residuals of the same procedure on the four cells
  ([ADR 0004](adr/0004-battery-level-conformal-intervals.md)). With four cells there is no
  battery-level finite-sample guarantee; the held-out coverage below is what was measured.
- **Version.** Package 2.0.0, bundle schema 1.0. Bundles are byte-reproducible with
  `soh train` and listed with their SHA-256 in `MANIFEST.json`.
- **Author.** Preran Sai Yarramasu. License: MIT.

## Intended use

- Research and demonstration on laboratory cells like the training cells: SOH estimates for
  cells of the same type and test protocol as the NASA reference cells, from a partial CC
  charge (3.90 → 4.19 V and the CV phase), a partial discharge (3.80 → 3.50 V under load) or
  both, with an interval and an explanation.
- Research and teaching on leakage-free evaluation and on uncertainty for battery health.
- A reference implementation of an auditable model artifact and serving stack.

**Out of scope.** Safety-critical battery management, warranty, second-life grading or any
decision about real cells; cells of other chemistries, formats, chargers or load profiles;
operating conditions outside the training data (the leave-one-condition-out results show
that transfer fails); estimating remaining useful life.

## Factors

Accuracy depends on how far a cell's ageing path and its initial offsets differ from the
training cells. The four core cells share one protocol but fade to different depths (lowest
SOH 57.3 % for B0006, 70.1-74.5 % for the others) and differ at commissioning (for example
B0006's constant-voltage phase is about 4,100 s long against 3,738-3,928 s). Relative mode
removes initial offsets but requires a commissioning record. Ambient temperature and load
current are constant in the core data; they are not inputs of the shipped bundles.

## Metrics

Leave-one-battery-out on the four core cells, reproduced from
[`reports/benchmark/summary.json`](../reports/benchmark/summary.json) for exactly the
procedure each bundle uses. MAE in SOH percentage points, mean over held-out cells; coverage
of the shipped `cv` interval at nominal 90 %, mean over held-out cells.

| Bundle | Candidate inputs | Mode | MAE | RMSE | Worst cell MAE | 90 % coverage |
|---|---|---|---:|---:|---:|---:|
| `combined-absolute-ridge` | 8 charge and discharge indicators | absolute | 2.53 | 2.84 | 6.45 (B0006) | 0.74 |
| `discharge-absolute-ridge` | `dis_q_380_350_ah`, `dis_r0_ohm`, `dis_temp_rise_win_c` | absolute | 1.71 | 1.97 | 2.51 (B0006) | 0.84 |
| `charge-absolute-ridge` | 5 charge indicators | absolute | 4.11 | 4.78 | 5.21 (B0006) | 0.75 |
| `combined-relative-ridge` | 8 indicators / commissioning reference | relative | 1.02 | 1.23 | 1.26 (B0006) | 0.93 |

These rows are the nested-selection procedure the bundles ship. The README's results table
and the audit table also show plain-Ridge rows, which are different models: the
discharge-window row there is plain Ridge (1.65, 84 % coverage) against 1.71 for the shipped
`discharge-absolute-ridge` procedure, and the audit's relative-mode row is plain Ridge (1.08,
87 % coverage) against 1.02 for `combined-relative-ridge`.

What the four bundles contain after the selection was run on all four core cells (from the
bundle files; half-widths in SOH points):

| Bundle | Inputs kept | Ridge alpha | Half-width 80 / 90 / 95 % | Calibration residuals |
|---|---|---:|---:|---:|
| `combined-absolute-ridge` | ICA peak height and voltage, `dis_q_380_350_ah`, `dis_temp_rise_win_c` | 0.1 | 4.57 / 6.86 / 9.62 | 636 |
| `discharge-absolute-ridge` | `dis_q_380_350_ah`, `dis_r0_ohm` | 0.1 | 2.39 / 3.36 / 4.41 | 636 |
| `charge-absolute-ridge` | ICA peak height and voltage, `chg_temp_rise_cc_c` | 10 | 5.77 / 6.65 / 7.68 | 628 |
| `combined-relative-ridge` | `chg_cv_time_s`, `dis_q_380_350_ah`, `dis_temp_rise_win_c` (relative) | 1 | 1.70 / 1.97 / 2.16 | 612 |

`combined-absolute-ridge` is the pre-specified headline specification. With B0006 in the
selection data, the procedure drops the CV-phase duration first (inner MAE 2.55 to 1.65),
which is the feature that misled the fold holding B0006 out. That is in-sample behaviour of
the selection, not evidence of better accuracy: the leave-one-battery-out numbers above are
the estimate for a new cell.

When a client does not name a model, the service picks, among the bundles whose inputs are all
present, the one with the lowest recorded MAE ([API.md](API.md#model-selection)): with
discharge indicators and no reference that is `discharge-absolute-ridge`; with a reference it
is `combined-relative-ridge`.

## Quantitative analyses

Per held-out cell, the headline procedure scores MAE 1.02 (B0005), 6.45 (B0006), 1.42
(B0007) and 1.21 (B0018), with 90 % coverage 1.00, 0.00, 1.00 and 0.95. The B0006 failure
(extrapolation below the training range and a cell-specific offset in the CV-phase duration)
is analysed in the [README](../README.md#failure-analysis-b0006) and
[METHODOLOGY](METHODOLOGY.md#9-nested-feature-selection). The same procedure selects
different inputs in different folds (mean pairwise Jaccard similarity 0.49 for the headline),
so the inputs of a bundle trained on all four cells are one draw from an unstable selection,
not a stable property of the cells.

Baselines under the same protocol: training mean 9.61, cycle count only 5.30, capacity
oracle (leaky, a sanity check) 3.47.

## Ethical considerations and caveats

- **Not a safety function.** An over-estimated SOH can delay the retirement of a degraded
  cell. The headline model over-estimates its worst held-out cell by 6.42 points on average.
  Do not use these models for decisions about real batteries.
- **Intervals can be over-confident on unusual cells.** The half-width comes from the
  training cells' leave-one-battery-out errors; it carries no battery-level guarantee, and its
  coverage on held-out cells was measured at 0.74 for the headline procedure at nominal 90 %
  (0.00 for B0006). The domain check (feature ranges and a Mahalanobis distance) flags inputs
  outside the training envelope, but a cell can be unusual in ways that stay inside it, as
  B0006 shows.
- **Small evaluation base.** Four core cells from one laboratory. No difference between
  models on these cells is statistically significant.
- **Relative mode needs the right reference.** The reference must be the mean of the cell's
  first valid cycles under the same protocol; a wrong reference shifts every estimate.

## Bundle format

A bundle is one JSON document (`schema_version` 1.0), written by
[`models/linear_bundle.py`](../src/battery_soh/models/linear_bundle.py):

| Key | Contents |
|---|---|
| `model_id`, `created_utc`, `package_version`, `feature_set`, `mode`, `n_reference_cycles` | Identity and mode |
| `features` | Input names, in order |
| `label` | Name and definition of the target |
| `preprocessing` | `impute_median`, `scaler_mean`, `scaler_scale` per input |
| `estimator` | Type, penalty, coefficients and intercept |
| `output_clip` | Physical output range, `[0, 120]` |
| `conformal` | Method, half-width per confidence level (`null` = unbounded), number of calibration scores |
| `domain` | Per-input training minimum and maximum, range margin (10 %), centre, Ledoit-Wolf shrinkage, precision matrix, and the Mahalanobis threshold (99 % quantile of the training distances) |
| `training` | Training cells, cycle count, SHA-256 of the feature table |
| `metrics` | Leave-one-battery-out metrics per held-out cell, mean, coverage per confidence level |
| `explanation` | Attribution method and base value |
| `selection`, `training_protocol` | Candidates, elimination trace, per-fold selections; rows and label ranges used |
| `provenance`, `model_card` | Hashes of the config, the benchmark results and the feature table; library versions; path of the Markdown card |

`MANIFEST.json` maps every bundle and model-card file to its SHA-256. The service verifies the manifest
at start-up and serves nothing if any hash differs.
