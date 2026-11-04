# 6. Pre-specified headline and nested feature selection

Date: 2026-10-12

## Status

Accepted

## Context

The core benchmark evaluates six feature sets, two modes, nine models and three baselines on
four cells: 114 leaderboard rows. Reporting the best row as "the result" is model selection
on the test cells, and with four cells the optimism of that choice is large. The same
problem appears inside a model: choosing features by their leave-one-battery-out test score
leaks the test cells into the feature set.

The design was not developed blind to these cells. The indicator windows, feature sets and
relative mode come from an audit prototype that had already scored plain Ridge
leave-one-battery-out on the four core cells (MAE about 1.6-1.7), and plain-Ridge results on
the v2 feature table were known before the headline was fixed.

## Decision

- **One pre-specified headline.** The headline is experiment `core_lobo`, feature set
  `combined` (all eight partial-window indicators), mode `absolute`, model `ridge_nfs` (Ridge
  with nested feature selection). It was written into
  [`configs/benchmark.yaml`](../../configs/benchmark.yaml) before any `ridge_nfs` result was
  computed, together with the nested-selection procedure and the list of secondary rows, and
  it is reported unchanged, whatever it shows. It was not chosen blind to these four cells:
  plain-Ridge results for this same `combined`/absolute specification on the v2 table (2.60;
  B0006 5.83), the no-CV-time ablation and relative mode were known when the headline was
  fixed. The declaration guards against picking the best of the 114 leaderboard rows
  afterwards; it does not make the design independent of these cells. `summary.json` repeats
  the declaration under its `pre_registration` key and records the configuration and data
  hashes.
- **Nested feature selection** ([`evaluation/nested.py`](../../src/battery_soh/evaluation/nested.py)).
  Inside every outer training fold: greedy backward elimination over the feature set, scored
  by the inner leave-one-battery-out MAE over the training cells, minimised over the penalty
  grid `alpha in {0.1, 1, 10}`; stop when no removal improves the score. The outer held-out
  cell never enters selection, so the outer error is a nearly unbiased estimate of the whole
  procedure on a new cell (Varma and Simon, 2006), conditional on design choices made with
  knowledge of these cells; with four folds it is also very noisy.
- **Post-hoc analyses are labelled.** Feature sets defined after seeing results
  (`charge_no_cv`, `combined_no_cv`) carry `post_hoc: true` in every output and are presented
  as diagnoses, never as results.
- **Production uses the same procedure.** The shipped bundles run the nested selection on all
  four core cells (`production.model: ridge_nfs`), so the deployed model is the procedure that
  was evaluated.

## Consequences

- The headline (MAE 2.53, 90 % coverage 0.74) is worse than several rows of the leaderboard,
  for example the best absolute-mode row that is not a post-hoc ablation (elastic net on the
  discharge window, 1.50) or relative mode (1.02 with nested selection; the best non-leaky row
  overall is 0.99, relative mode with EIS). Those rows are reported as secondary.
- The failure behind the headline became a documented finding: with B0006 held out, inner
  validation on the other three cells keeps the CV-phase duration, whose cell-specific offset
  then misleads on B0006 (MAE 6.45). The ablation that removes that feature is reported, but
  only as a labelled post-hoc diagnosis.
- Selection is unstable across folds (mean pairwise Jaccard similarity of the selected sets
  0.49), which is itself informative about how little four cells can determine.
- The declaration is inside the repository rather than with an external, timestamped
  registry. The configuration hash shows which configuration produced `summary.json`, not when
  it was written; treat this as a commitment made before the headline's own result was
  computed, not as a blinded pre-registration. Any later change to the hashed configuration
  sections changes that hash, so the benchmark must be re-run for the recorded hash to match
  the file.
