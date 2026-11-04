# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.0.0] - 2026-10-12

A rewrite. The task, the evaluation protocol, the models, the service and the documentation
are new; the v1 code is not included in this repository as evidence for the leakage audit. Results of
1.0.0 are withdrawn (see `docs/LEAKAGE_AUDIT.md`).

### Added

- Task definition: SOH estimation from health indicators measurable in part of a cycle
  (CC-charge window 3.90 → 4.19 V, ICA peak, CV-phase duration; discharge window
  3.80 → 3.50 V, ohmic resistance at load-on, temperature rises; optional EIS), with an
  optional commissioning-referenced ("relative") mode.
- Feature registry with unit, family, availability, leakage risk and plausible range for
  every input; `get_feature_set` refuses label-side and proxy features.
- NASA PCoE loader for all 34 cells; protocol registry with an explicit five-criterion
  inclusion rule (4 core cells, 12 in the multi-condition set, every exclusion with a recorded
  reason); `soh data download` with SHA-256 verification and safe nested-zip extraction.
- Capacity quality control (missing, below 0.3 Ah, isolated, Hampel-style outlier) and the
  labels `soh_pct` (own initial capacity, median of the first five valid cycles) and
  `soh_rated_pct`.
- Committed per-cycle feature table `data/processed/nasa_features.csv` (2,794 rows x 33
  columns) with a SHA-256 sidecar; deterministic, byte-identical rebuild.
- `features_from_segments`, the single feature path shared by the table builder and the API.
- Synthetic equivalent-circuit cell simulator with analytic ground truth for testing the
  extractors.
- Leave-one-battery-out benchmark (`soh benchmark`): six feature sets, two modes, nine
  models and three baselines on the core cells; a 12-cell multi-condition experiment with a
  battery-level bootstrap and Wilcoxon tests; leave-one-condition-out; an exploratory
  voltage-window sweep.
- Ridge with nested feature selection (`ridge_nfs`) and a pre-specified headline
  specification in `configs/benchmark.yaml`, written before its own result was computed
  (plain-Ridge results on the same cells were already known; see `docs/METHODOLOGY.md`,
  section 12); post-hoc analyses flagged in every output.
- Battery-grouped conformal prediction (`cv` and group CV+) with exact conformal ranks in
  rational arithmetic (an infinite half-width when there are too few scores). With three or
  four calibration cells there is no battery-level finite-sample guarantee; coverage is
  measured on held-out cells and reported per cell.
- Leakage tooling: `check_no_leakage`, `assert_disjoint`, `assert_groups_disjoint`, and
  `soh audit`, which reproduces the v1 failure modes from the committed data.
- JSON model bundles with exact linear attributions, a training-domain check and a SHA-256
  manifest; Markdown model cards; `soh train` reproduces the benchmark's per-cell results or
  fails.
- FastAPI service: `/v1/soh/estimate`, `/v1/soh/estimate-from-cycle`, `/v1/soh/batch`,
  `/v1/models`, `/v1/demo/...`, `/v1/benchmark`, `/livez`, `/readyz`, `/metrics`, OpenAPI
  docs.
- Web UI with Estimate, Benchmark, Leakage audit and Model card tabs, served by the API or
  exported as a static site (`soh export-static`).
- Figures and a generated results document (`soh report`).
- Multi-stage Docker image and Compose file; CI (lint, `mypy --strict`, tests on Ubuntu and
  Windows, reproduction of the audit and the quick benchmark, Docker smoke test), GitHub Pages
  and GHCR release workflows; Makefile; pre-commit hooks; Dependabot.
- An end-to-end test that runs the static site's JavaScript backend in Node.js and requires
  the API's answers, validation errors included.
- `requirements/constraints.txt` with the library versions of the committed results
  (`pip install -e ".[dev]" -c requirements/constraints.txt`); byte-exact reproduction needs
  them, while other versions change only tree-ensemble numbers slightly.
- An executed walkthrough notebook (`notebooks/01_walkthrough.ipynb`).
- Documentation: README, methodology, leakage audit, data card, model card, architecture,
  API, deployment, six architecture decision records, `SECURITY.md`, `CONTRIBUTING.md`,
  `CITATION.cff`.

### Changed

- Evaluation: whole cells are held out; metrics are MAE in SOH points per held-out cell with
  the worst cell and interval coverage reported, instead of R² on rows of seen cells.
- Label reference capacity: the median of the first five QC-valid capacities (v1 averaged the
  first five discharges).
- Ohmic resistance is measured at the located load-on step instead of at the largest current
  jump anywhere in the record.
- Incremental capacity analysis: CC samples only, isotonic monotone voltage, a uniform 5 mV
  grid and a Savitzky-Golay derivative.
- Packaging: one installable package (`battery-soh`, Python 3.11+) with the `soh` CLI;
  paths resolved through `battery_soh.config.get_paths()`.
- The benchmark code is a subpackage (`battery_soh.evaluation.benchmark`) with an unchanged
  public API and byte-identical outputs.
- Wording of results: "pre-specified" instead of "pre-registered", with a disclosure of what
  was known when the headline was fixed; relative-mode and discharge-only results always name
  the model (nested selection or plain Ridge); pooled R² is labelled as pooled; conformal
  intervals are described as measured, not guaranteed; the best window of the sweep is marked
  as a best-of-11 on the same cells; the library sensitivity of the one borderline Wilcoxon
  test is stated.
- Web UI: an estimate for a demo cycle is marked as in-sample (the shipped models were
  trained on all four demo cells), and for the headline model the UI also shows the held-out
  leave-one-battery-out estimate of the same cycle, in API and static mode.
- `GET /v1/benchmark` and the demo endpoints are served from bytes cached per source file,
  with an `ETag`, `Cache-Control: public, max-age=300` and `304 Not Modified` on a matching
  `If-None-Match`.

### Removed

- Capacity and every full-discharge statistic as model inputs (kept only as labelled `leg_*`
  columns for the audit).
- The Mendeley forklift dataset and its pulse-based labels.
- The Sentence-BERT text modality, whose templates contained the SOH label.
- PyTorch models (CNN-LSTM, Transformer, cross-attention, physics-informed variants) and the
  LLM chat endpoints of the v1 API.
- Pickled model artifacts.

### Fixed

- Test rows that were also training rows (284 of 336 in v1's documented test set); every fold
  now asserts disjointness.
- The conformal quantile no longer falls back to the largest score when there are too few
  scores; the half-width is infinite and flagged.
- Early stopping in v1 restored the last epoch (`state_dict().copy()` is a shallow copy); the
  affected training loops are removed.
- A bundle that passes the manifest check but is malformed (for example a non-numeric margin
  or a ragged precision matrix) is skipped with an error instead of crashing the service at
  start-up; `/readyz` stays 503 if no bundle remains.
- `soh audit` prints `n/a` for a missing metric instead of failing.
- Benchmark edge cases: the window-sweep consistency check fails instead of passing when it
  cannot compare values, a configuration without the reported 90 % level fails with a clear
  error, and every convergence warning is counted.
- Web UI: separate request tokens for loading a cycle and refreshing an estimate, a 2xx
  response with unparseable JSON is an error, and one caller's abort no longer cancels a
  shared cached request. The static backend rejects the same inputs as the API (`leg_*`
  features, non-finite references).

### Security

- No pickle: models are JSON evaluated with NumPy; the registry verifies SHA-256 hashes and
  serves nothing on a mismatch.
- Strict request validation (unknown fields rejected, finite numbers, physical ranges, array
  and batch limits), a 413 body limit before parsing, RFC 7807 errors without stack traces.
- Content-Security-Policy (a separate, pinned-CDN policy on `/docs` and `/redoc`) and other
  security headers on every response; CORS closed by default and never with credentials (v1
  allowed every origin with credentials).
- Non-root container with a read-only root filesystem, no capabilities and no compiler; no
  outbound network calls and no API keys at runtime (v1 called external LLM APIs).
- Batch requests are capped at 100 items by default (`SOH_MAX_BATCH_ITEMS`, schema ceiling
  1,000).
- Supply chain: GitHub Actions pinned to commit SHAs; the Pages and release workflows run
  only after CI has passed, and releases use a protected environment; the Docker base image is
  pinned by digest and runtime dependencies are installed from a hashed lock file
  (`--require-hashes`); a gitleaks pre-commit hook.
- The `/docs` and `/redoc` pages load their pinned CDN files with Subresource Integrity under
  a policy scoped to those files; responses carry `Cross-Origin-Resource-Policy`; the static
  Pages build has its own Content-Security-Policy.
- Client-triggered extraction errors are logged without a stack trace (the traceback is
  logged at DEBUG only).

## [1.0.0] - 2026-03-31

The first version, now not included in this repository and not maintained: NASA and Mendeley data,
physics-inspired features, CNN-LSTM, Transformer and cross-attention models with a
Sentence-BERT text branch, conformal and survival extensions, and a FastAPI service. Its
reported metrics (R² about 0.96) are withdrawn: the 2.0.0 audit showed target leakage and a
contaminated test set (`docs/LEAKAGE_AUDIT.md`). The date is that of the last revision of the
v1 README.
