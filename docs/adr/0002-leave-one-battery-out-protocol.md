# 2. Leave-one-battery-out evaluation protocol

Date: 2026-10-12

## Status

Accepted

## Context

Consecutive cycles of one cell are strongly dependent: a model that has seen cycle 120 of a
cell can "predict" cycle 121 by interpolation. v1 scored its production model on a test file
in which 284 of 336 rows were training rows. Reproduced on the committed data with v1's
inputs, that model scores MAE 0.83 on the rows it was trained on and 3.29 on the rows it had
not seen; an honest within-battery holdout gives 2.48 and leave-one-battery-out 3.22
([LEAKAGE_AUDIT.md](../LEAKAGE_AUDIT.md)).

A deployed estimator always meets a cell it has never seen. The evaluation unit must be the
cell.

The headline data set has four cells (B0005, B0006, B0007, B0018) that share one protocol;
the extended set has twelve cells in three operating conditions.

Alternatives considered:

- *Random K-fold over cycles*: leaks neighbouring cycles of the same cell into training.
- *Within-cell chronological holdout* (train on early life, test on late life): available
  as `temporal_holdout` and used by the audit, but it answers a forecasting question for cells
  whose early life was seen, not the new-cell question.
- *Leave-one-condition-out as the headline*: the harder transfer question, but with three
  conditions it gives three folds of very different sizes; it is reported as a secondary
  experiment.

## Decision

- The headline experiment holds out one core cell at a time (`core_lobo`, 4 folds). The
  extended experiment does the same over 12 cells (`extended_lobo`); a third holds out whole
  operating conditions (`leave_one_condition_out`).
- On every fold, `assert_disjoint` checks that no `(battery_id, cycle_index)` key is in both
  training and evaluation rows, and `assert_groups_disjoint` checks the groups.
- Everything learned from data is learned inside the training fold: imputation medians,
  scaling, feature selection, the ridge penalty and conformal calibration.
- Metrics are computed per held-out cell and averaged, so each cell counts once; the worst
  cell is always reported next to the mean.
- In relative mode, the cycles used to compute a cell's reference are never scored.

## Consequences

- Results describe performance on a new cell of the same population, which is the deployment
  question.
- Four folds give high-variance estimates. With four paired cells the smallest attainable
  two-sided Wilcoxon p-value is 0.125, so no model difference on the core set can be called
  significant; comparisons there are reported as per-cell effect sizes. The extended set
  allows a battery-level bootstrap and a paired test.
- One unusual cell can dominate the mean: B0006 contributes an MAE of 6.45 to a headline of
  2.53. The per-cell table is therefore part of every result, not an appendix.
- Hyper-parameters cannot be tuned on the evaluation cells; models use fixed small-data
  defaults, and the one tuned procedure (`ridge_nfs`) tunes inside the training fold
  ([ADR 0006](0006-pre-specified-headline-and-nested-selection.md)).
