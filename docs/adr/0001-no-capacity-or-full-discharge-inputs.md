# 1. No capacity or full-discharge inputs

Date: 2026-10-12

## Status

Accepted

## Context

The label is `SOH = 100 * capacity / C0`, where `capacity` is the charge delivered by a
complete discharge and `C0` the cell's initial capacity. v1 used `capacity` as a model input,
together with statistics of the whole discharge record (record length, mean current, mean
voltage, maximum temperature). The audit ([LEAKAGE_AUDIT.md](../LEAKAGE_AUDIT.md)) measured
the consequences on the committed data:

- within one cell, SOH against capacity has R² = 1.0;
- without capacity, the remaining v1 inputs rebuild it with leave-one-battery-out R² 0.99498,
  just below the 0.995 threshold of the statistical leakage rule;
- leave-one-battery-out MAE with the v1 inputs is 3.22 with capacity and 3.22 without it.

There is also a usefulness argument independent of statistics: in deployment a complete,
controlled discharge is rarely available, and when it is, it measures capacity, and therefore
SOH, directly. An estimator that needs it has nothing left to estimate.

Health indicators that *are* legitimate also correlate strongly with capacity (the charge
delivered between 3.80 and 3.50 V reaches a within-battery \|r\| of 0.998 with SOH), so a
correlation threshold alone cannot separate an indicator from a proxy.

## Decision

Model inputs are restricted to quantities measurable in part of a cycle or from metadata:
a voltage window of the CC charge and the CV phase, a voltage window of the discharge under
load and the load-on step, a prior EIS measurement, and protocol context.

- Every feature is declared in [`features/registry.py`](../../src/battery_soh/features/registry.py)
  with `availability` (`partial_charge`, `partial_discharge`, `eis`, `metadata`,
  `full_discharge`) and `leakage_risk` (`none`, `proxy`, `label`).
- Full-discharge quantities exist only as `leg_*` columns with risk `label` or `proxy`, to
  reproduce the audit.
- `get_feature_set` raises unless every member has risk `none`; only the audit and the
  explicitly labelled `capacity_oracle` baseline opt in with `allow_leaky=True`.
- The benchmark runs `check_no_leakage` on every feature set and aborts on failure.
- The API rejects `leg_*` fields with 422 and refuses to load a bundle that uses them.

## Consequences

- Reported accuracy is lower than v1 claimed, and honest: the pre-specified headline is MAE
  2.53 under leave-one-battery-out.
- The estimator applies when only part of a cycle is observed, which is the realistic case.
- Indicator windows are tied to the NASA protocol (CC 1.5 A to 4.2 V; constant-load
  discharge). Some cells never reach a window (B0032's loaded voltage starts below 3.80 V), so
  some features are missing for some cells and other protocols need their windows
  re-derived.
- The registry is now the primary leakage guard and must be maintained honestly: adding a
  feature means declaring when it becomes measurable. The statistical checks are a backstop;
  the v1 proxy set sits at R² 0.99498, just below the 0.995 reconstruction threshold, and is
  caught by its registry classification.
