# 4. Battery-level conformal prediction intervals

Date: 2026-10-12

## Status

Accepted

## Context

An SOH estimate without an error bar invites over-confidence. Split conformal prediction
gives distribution-free intervals, but its guarantee assumes the calibration scores are
exchangeable with the score of the new point. Cycles of one cell are strongly dependent, and
a deployed model meets a new cell, so cycle-level calibration on held-out rows of seen cells
would measure the wrong error and produce intervals that are far too narrow.

v1's conformal code also clipped the quantile level at 1
([`legacy/src/phase4/conformal.py`](../../legacy/src/phase4/conformal.py#L43-L54)), returning
the largest score when there were too few scores for the requested level, which silently
breaks the guarantee.

Alternatives considered: row-level split conformal (wrong exchangeable unit), conformalised
quantile regression (same unit problem, more machinery), Gaussian-process predictive
intervals (model-specific and not calibrated to the cross-cell error), and condition-wise
(Mondrian) calibration (too few cells per condition today).

## Decision

- Calibration scores come from an inner leave-one-battery-out loop over the **training**
  cells ([`uncertainty/conformal.py`](../../src/battery_soh/uncertainty/conformal.py)): each
  score is the absolute error of a model that never saw that cell.
- The quantile uses the exact rank `k = ceil((n + 1)(1 - alpha))` in rational arithmetic, and
  is `+inf` when `k > n`.
- Two constructions are evaluated in the benchmark: `cv` (full-model prediction plus or minus
  the quantile of the pooled out-of-cell residuals) and `cv_plus` (group CV+, Barber et al.
  2021). The bundle ships `cv`: one half-width per confidence level (0.8, 0.9, 0.95), and the
  benchmark reports the held-out coverage of exactly that construction, per cell and on
  average.
- The documentation states the limits: even for a new cell exchangeable with the
  calibration cells there is no battery-level finite-sample guarantee with three calibration
  cells per fold (a 90 % guarantee needs at least nine), so coverage is an empirical property
  measured on held-out cells; and nothing covers a cell from a different population.

## Consequences

- Intervals describe the error on a new cell, so they are wide: the headline's mean 90 %
  width is 9.78 SOH points.
- With three calibration cells per fold the assumption fails visibly when one cell is unlike
  the others. In the headline, B0006 gets the narrowest band (3.69 points) and 0.00 coverage,
  while B0005's band (16.38 points) is inflated by B0006's residuals; mean coverage at 90 % is
  0.74. In relative mode, where per-cell offsets are removed, coverage is 0.93 with nested
  selection.
- `cv` has no finite-sample guarantee of its own (the full model is not one of the fold
  models); its coverage is measured, not assumed. `cv_plus` has a worst-case `1 - 2 alpha`
  guarantee for enough exchangeable units, which three cells per fold are not, and it would
  need every fold model at prediction time. The exact ranks prevent v1's silent truncation;
  they do not create a battery-level guarantee.
- With only two calibration conditions per fold, leave-one-condition-out intervals are too
  wide to be informative. Covariate-shift-aware calibration is on the roadmap.
