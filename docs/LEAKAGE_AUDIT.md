# The v1 leakage audit

The first version of this project (v1, frozen under [`legacy/`](../legacy/README.md))
reported an R² of about 0.96 for state-of-health (SOH) estimation on the NASA PCoE cells and
described it as "96 % accuracy". An audit of the code and the evaluation found that those
numbers came from how the task was set up and scored, not from anything the models had
learned. This document records each failure, how it was detected, the reproduced numbers, and
the guard in v2 that turns the same mistake into a hard error.

Every number below comes from [`reports/audit/audit.json`](../reports/audit/audit.json)
(table form: [`audit_table.csv`](../reports/audit/audit_table.csv)), which `soh audit`
regenerates from the committed feature table, or from the benchmark summary
[`reports/benchmark/summary.json`](../reports/benchmark/summary.json). Code evidence links
point at the frozen v1 sources in `legacy/`.

## Contents

- [Reproduce it](#reproduce-it)
- [The numbers side by side](#the-numbers-side-by-side)
- [Finding 1: the label was an input](#finding-1-the-label-was-an-input)
- [Finding 2: dropping capacity does not remove the leak](#finding-2-dropping-capacity-does-not-remove-the-leak)
- [Finding 3: the test set overlapped the training set](#finding-3-the-test-set-overlapped-the-training-set)
- [Finding 4: the Mendeley "SOH" labels measured discharge pulses](#finding-4-the-mendeley-soh-labels-measured-discharge-pulses)
- [Finding 5: the text modality contained the label](#finding-5-the-text-modality-contained-the-label)
- [Other v1 defects found on the way](#other-v1-defects-found-on-the-way)
- [Why the leaky oracle is not perfect](#why-the-leaky-oracle-is-not-perfect)
- [Guards: each mistake is now an error](#guards-each-mistake-is-now-an-error)
- [What changed in the results](#what-changed-in-the-results)
- [Lessons](#lessons)

## Reproduce it

```bash
python -m pip install -e . -c requirements/constraints.txt
soh audit        # reads data/processed/nasa_features.csv, writes reports/audit/ in seconds
```

The audit needs only the committed feature table (SHA-256
`f6c4fb709c4810f76a71eaf3ee2b99f963e0fcc94d354dbe636ae8682333d997`, recorded in
`audit.json`); CI runs it on every push to `main` and every pull request (job `reproduce` in
[`.github/workflows/ci.yml`](../.github/workflows/ci.yml)), and
[`tests/integration/test_audit.py`](../tests/integration/test_audit.py) checks its outputs.
Implementation: [`src/battery_soh/evaluation/audit.py`](../src/battery_soh/evaluation/audit.py).
The constraints file pins the library versions that produced the committed results; the
audit fits only linear models, and a check with scikit-learn 1.9.1 reproduced every audit
number exactly.

Scope of the reproduction. The audit does not re-run v1's deep models (CNN-LSTM,
Transformer, cross-attention), which would need v1's pinned PyTorch environment. It
reproduces the *mechanisms* on the QC-valid cycles of the four core cells (B0005, B0006,
B0007, B0018), using the v1 inputs (`legacy_v1`: capacity, record length, whole-record mean
current, mean voltage, maximum temperature and cycle count) with the same Ridge model that v2
uses (alpha = 1 on standardised inputs). Holding the model fixed isolates the effect of the
inputs and of the evaluation protocol.

## The numbers side by side

MAE and RMSE are in SOH percentage points. For the leave-one-battery-out (LOBO) rows they are
the mean over the four held-out cells; R² is pooled over all scored cycles.

| # | Evaluation | Inputs | MAE | R² | Leaky |
|---|---|---|---:|---:|:---:|
| 1 | v1 protocol: train on the first 85 % of B0005-B0007, "test" on all of B0006 and B0007 (284 of 336 test rows were training rows) | v1, incl. capacity | 1.21 | 0.977 | yes |
| 2 | Same model, honest within-battery holdout (last 15 % of each cell, 78 rows) | v1, incl. capacity | 2.48 | 0.801 | yes |
| 3 | Leave-one-battery-out | v1, incl. capacity | 3.22 | 0.878 | yes |
| 4 | Leave-one-battery-out | v1 without capacity | 3.22 | 0.883 | yes |
| 5 | Leave-one-battery-out | capacity alone (oracle) | 3.47 | 0.857 | yes |
| 6 | Leave-one-battery-out, **v2 pre-specified headline** (Ridge with nested selection) | partial-window HIs | 2.53 | 0.881 | no |
| 7 | Leave-one-battery-out, v2 same set, plain Ridge | partial-window HIs | 2.60 | 0.898 | no |
| 8 | Leave-one-battery-out, v2 discharge window only, plain Ridge | 3.80 → 3.50 V HIs | 1.65 | 0.965 | no |
| 9 | Leave-one-battery-out, v2 relative mode, plain Ridge | HIs / commissioning reference | 1.08 | 0.982 | no |

Rows 1-9 are `rows[0..8]` of `audit.json`. Row 9 is scored on 612 non-reference cycles; all
other LOBO rows on 636 cycles. Row 9 is plain Ridge; with nested selection, the procedure of
the shipped relative bundle, relative mode scores 1.02 (`summary.json`, `secondary`). The figure
[`reports/figures/leakage_audit.png`](../reports/figures/leakage_audit.png) plots the same
rows.

Read the table top to bottom: the error grows as soon as the evaluation stops reusing training
rows (row 1 to row 2), grows again when whole batteries are held out (row 3), and does not
change when capacity is removed (row 4). v2's leakage-free inputs, which need only part of a
cycle, do better than v1's leaky inputs under the same protocol (rows 6-9 against rows 3-5).

## Finding 1: the label was an input

**What v1 did.** The label was `SOH = 100 * capacity / C0`, with `C0` the cell's initial
capacity (v1 averaged the first five discharges,
[`legacy/configs/config.yaml`](../legacy/configs/config.yaml#L24-L25)). The v1 baselines then
used every column of the feature table except identifiers and the label as model inputs
([`legacy/src/phase2/baseline_models.py`](../legacy/src/phase2/baseline_models.py#L845-L847),
same rule in
[`data_splits.py`](../legacy/src/phase2/data_splits.py#L135-L138)), and the harmonised table
carried `capacity` as one of those columns
([`data_harmonization.py`](../legacy/src/phase2/data_harmonization.py#L293-L308)).

**How it was detected.** Within one cell `C0` is a constant, so SOH is an exact affine
function of capacity. On the committed table the within-battery R² of `soh_pct` against
`capacity_ah` is **1.0 for all four core cells**, and the largest deviation of
`soh_pct` from `100 * capacity_ah / c0_ah` is 0.0007 SOH points, which is the CSV's
six-significant-digit rounding (`details.soh_vs_capacity_within_battery`).

**Consequence.** An evaluation that scores the model on rows of a cell it has already seen
grades a rescaling of the label. This is the mechanism behind row 1 of the table.

## Finding 2: dropping capacity does not remove the leak

**What v1 did.** v1's other inputs were statistics of the *whole* discharge record: the
record length (`discharge_time`, which includes the rest after the cut-off), the whole-record
mean current, mean voltage and maximum temperature. Mean current times record length is a
coulomb count, that is, a capacity measurement.

**How it was detected.** The audit fits a near-OLS model that predicts `capacity_ah` from the
v1 inputs *without* capacity (`leg_record_time_s`, `leg_current_mean_a`,
`leg_voltage_mean_v`, `leg_temp_max_c`, `ctx_cycle_index`) and scores it leave-one-battery-out:
**R² 0.99498** with a log-linear fit (0.993 linear), `details.capacity_reconstruction`. That
is just below the 0.995 threshold of the statistical leakage rule (see the
[limitation below](#guards-each-mistake-is-now-an-error)). Rows 3
and 4 of the table make the same point from the other side: LOBO MAE is 3.22 with capacity
and 3.22 without it, because the proxies carry the same information.

Per-feature view (`details.leakage_report_legacy_v1`):

| Feature | Registry risk | Max within-battery \|r\| with SOH | LOBO R² rebuilding capacity |
|---|---|---:|---:|
| `leg_capacity_ah` | label | 1.000 | 1.000 |
| `leg_record_time_s` | proxy | 0.993 | 0.852 |
| `leg_current_mean_a` | proxy | 0.989 | 0.716 |
| `leg_voltage_mean_v` | proxy | 0.986 | 0.683 |
| `leg_temp_max_c` | proxy | 0.935 | -0.412 |
| `ctx_cycle_index` | none | 0.988 | 0.778 |

No single proxy rebuilds capacity on its own; together they nearly do (R² 0.99498). There is also a practical
objection that needs no statistics: every one of these inputs requires a complete discharge,
and a complete discharge measures capacity directly. A model that needs it has nothing left to
estimate.

## Finding 3: the test set overlapped the training set

**What v1 did.** The production Ridge was trained on the first 85 % of the cycles of B0005,
B0006 and B0007 (426 rows) and its documented metrics were computed on a test file holding all
cycles of B0006 and B0007 (336 rows). **284 of the 336 test rows were training rows**
(`contaminated_test_overlap`). The same file served as the test set of the v1 baselines
([`legacy/src/phase2/baseline_models.py`](../legacy/src/phase2/baseline_models.py#L878)) and
as the calibration set of the API's conformal intervals
([`legacy/api/main.py`](../legacy/api/main.py#L219-L237)).

**How it was detected.** By joining the two files on `(battery_id, cycle_index)`. v2 does the
same join on every fold automatically (see the guards below), and the audit replays the v1
split through that guard, which raises:

```text
LeakageError: 284 evaluation row(s) also appear in the training data
(e.g. ('B0006', 0), ('B0006', 1), ('B0006', 2), ('B0006', 3), ('B0006', 4))
```

**Reproduced numbers** (`details.v1_split`). With the v1 inputs, the contaminated "test" gives
MAE 1.21 and R² 0.977. Split by provenance, the model scores MAE **0.83** on the 284 rows it
was trained on and **3.29** on the rows it had not seen. Scored honestly on the last 15 % of
each of the same three cells (78 rows, disjoint but from cells it has seen), the same model
gets MAE 2.48 and R² 0.80; under leave-one-battery-out, 3.22.

## Finding 4: the Mendeley "SOH" labels measured discharge pulses

**What v1 did.** v1 added a Mendeley forklift-battery dataset. Its loader treated every
contiguous stretch of negative current with at least 50 samples as a discharge "cycle"
([`legacy/src/phase3/mendeley_loader.py`](../legacy/src/phase3/mendeley_loader.py#L124-L147)),
integrated the current over that stretch as the cycle's `capacity`
([L156-L158](../legacy/src/phase3/mendeley_loader.py#L156-L158)), took the largest of the
first five stretches as the initial capacity
([L188-L190](../legacy/src/phase3/mendeley_loader.py#L188-L190)), and defined
`soh = min(100, 100 * capacity / initial_capacity)`
([L214-L222](../legacy/src/phase3/mendeley_loader.py#L214-L222)).

**Why it is invalid.** In a forklift duty cycle each stretch of negative current is one load
pulse, not a full discharge. The resulting "SOH" measures how much charge one pulse drew
relative to the largest early pulse; it says nothing about ageing, and it is again computed
from an input feature (`capacity`).

**Status in v2.** Not reproduced numerically: v2 does not use the Mendeley data
([ADR 0005](adr/0005-exclude-mendeley-from-v2.md)). The evidence is the loader code above.

## Finding 5: the text modality contained the label

**What v1 did.** v1's "multimodal" models embedded a natural-language description of each
cycle with Sentence-BERT
([`legacy/src/phase2/multimodal_model.py`](../legacy/src/phase2/multimodal_model.py#L87-L109)).
The descriptions were generated from templates
([`legacy/src/phase2/text_templates.yaml`](../legacy/src/phase2/text_templates.yaml#L4-L20))
such as

```yaml
template: "Battery shows excellent health with SOH at {soh:.1f}%."
```

filled with the cycle's SOH value
([`legacy/src/phase2/feature_to_text.py`](../legacy/src/phase2/feature_to_text.py#L102-L112)).

**Consequence.** The text input stated the answer. Any gain the text branch brought was the
label read back.

**Status in v2.** There is no text modality, and the API rejects label-side inputs outright
(see the guards).

## Other v1 defects found on the way

These are not leakage, but they corrupted v1 results and are fixed by construction in v2.

- **Early stopping restored the last epoch, not the best one.** `model.state_dict().copy()`
  ([`legacy/src/phase2/baseline_models.py`](../legacy/src/phase2/baseline_models.py#L551), also
  [`baseline_model.py`](../legacy/src/baseline_model.py#L315),
  [`multimodal_model.py`](../legacy/src/phase2/multimodal_model.py#L725) and
  [`cross_attention_model.py`](../legacy/src/phase3/cross_attention_model.py#L636)) is a
  shallow copy: the dictionary is new, but its tensors are the live parameters that the
  optimizer keeps updating in place. v2 has no iterative deep-learning training.
- **The conformal quantile was capped at the sample maximum.**
  [`legacy/src/phase4/conformal.py`](../legacy/src/phase4/conformal.py#L43-L54) clips the
  quantile level to 1, so with too few calibration scores it returns the largest score instead
  of an infinite half-width, silently breaking the coverage guarantee; the API's own
  calibration clamps the rank the same way
  ([`legacy/api/main.py`](../legacy/api/main.py#L234-L235)). v2 computes the exact
  rank `ceil((n + 1)(1 - alpha))` and returns `+inf` when it exceeds `n`
  ([`uncertainty/conformal.py`](../src/battery_soh/uncertainty/conformal.py)).
- **The ohmic-resistance feature looked for the largest current jump anywhere in the record**
  (`np.argmax(np.abs(current_diff))`,
  [`legacy/src/feature_extraction.py`](../legacy/src/feature_extraction.py#L275)), which can be
  the load removal rather than the load-on step. v2 locates load-on explicitly and returns NaN
  when it cannot ([`features/discharge.py`](../src/battery_soh/features/discharge.py)).

## Why the leaky oracle is not perfect

Row 5 feeds the label's own numerator, capacity, to a Ridge model, and still scores MAE 3.47
(per held-out cell: B0005 3.48, B0006 6.70, B0007 0.74, B0018 2.97;
`details.lobo_per_battery_mae.capacity_oracle`). The reason is the denominator: SOH is
capacity divided by the *cell's own* `C0`, and `C0` differs between cells (1.835, 2.013,
1.881 and 1.840 Ah for B0005, B0006, B0007 and B0018 in the committed table). A model that
sees a new cell's capacity but not its `C0` is off by a per-cell scale factor, which is
largest for B0006, the cell with the largest `C0`.

Two consequences matter for the rest of the project:

1. Within one battery that scale factor is constant, which is exactly why within-battery
   evaluation of v1 looked near-perfect.
2. In *relative* mode, where every input is divided by the cell's own value over its first
   five valid cycles, the same oracle scores MAE **0.11** (`leaderboard`, `capacity_oracle`,
   relative mode, in `summary.json`): capacity divided by its own early value *is* the label.
   Relative mode is therefore only legitimate with inputs that are not capacity, which is what
   the feature registry enforces.

The oracle is a sanity bound for the leak, not an upper bound on achievable accuracy: the
partial-window discharge indicators (row 8) beat it under the same protocol.

## Guards: each mistake is now an error

| v1 mistake | v2 guard | Where | Demonstrated in `audit.json` |
|---|---|---|---|
| Label numerator as an input | Every feature is declared in a registry with its unit, *availability* (`partial_charge`, `partial_discharge`, `eis`, `metadata`, `full_discharge`) and *leakage risk* (`none`, `proxy`, `label`). `get_feature_set` raises unless every member has risk `none`; only the audit and the labelled oracle baseline pass `allow_leaky=True`. | [`features/registry.py`](../src/battery_soh/features/registry.py) | `guards.get_feature_set_legacy_v1_without_opt_in` raised `ValueError` |
| Proxies that rebuild capacity | `check_no_leakage` flags (1) registry label/proxy features and label-side columns, (2) within-battery \|r\| >= 0.9995 with the label, (3) a feature or set that rebuilds capacity with LOBO R² >= 0.995. The benchmark runs it on every feature set of every experiment and aborts on failure. | [`evaluation/leakage.py`](../src/battery_soh/evaluation/leakage.py), [`evaluation/benchmark/frames.py`](../src/battery_soh/evaluation/benchmark/frames.py) | `guards.check_no_leakage_legacy_v1` and `..._no_capacity` raised `LeakageError` |
| Test rows that were training rows | Every split holds out whole groups; `assert_disjoint` on `(battery_id, cycle_index)` and `assert_groups_disjoint` run on every fold. | [`evaluation/splits.py`](../src/battery_soh/evaluation/splits.py) | `guards.assert_disjoint_on_v1_split` raised `LeakageError` (284 rows) |
| Invalid labels | Capacity QC (missing, below 0.3 Ah, Hampel-style outlier) and a five-criterion cell inclusion rule (protocol, history, reference, windows, labels). Mendeley is out of scope. | [`qc.py`](../src/battery_soh/qc.py), [`data/protocols.py`](../src/battery_soh/data/protocols.py) | (inputs to every row of the audit) |
| Label in an input modality | No text modality. The API's request schema rejects any label-side or proxy feature with 422, and the model registry refuses to load a bundle that uses one. | [`api/schemas.py`](../src/battery_soh/api/schemas.py), [`api/registry.py`](../src/battery_soh/api/registry.py) | covered by `tests/api` |

What the benchmark recorded for the v2 sets that did pass (`summary.json`,
`leakage_checks`): the `combined` set reaches a maximum within-battery \|r\| of 0.998 with the
label and rebuilds capacity with LOBO R² 0.944, both below the thresholds. That is expected:
a health indicator is *supposed* to track capacity closely; the thresholds target exact
reconstructions, not strong correlation.

**A limitation of the statistical rules, stated plainly.** The v1 inputs without capacity
rebuild capacity with LOBO R² **0.99498**, just *below* the 0.995 threshold. In
`guards.check_no_leakage_legacy_v1_no_capacity` the error is raised because the registry
classifies those columns as full-discharge proxies, not because of the reconstruction
statistic. The registry, which records *when* a quantity becomes measurable, is the primary
guard; the data-driven rules are a backstop for near-identities. Thresholds on correlation
statistics cannot separate a good health indicator from a proxy reliably, because both
correlate with capacity at 0.99 and above.

## What changed in the results

| Stage | Number | Source |
|---|---|---|
| v1 as documented | R² about 0.96 | [`legacy/README.md`](../legacy/README.md); `details.v1_documented_claim` |
| v1 inputs, v1 protocol, reproduced | MAE 1.21, R² 0.977 | `audit.json` row 1 |
| v1 inputs, leave-one-battery-out | MAE 3.22 | row 3 |
| v2 pre-specified headline | MAE 2.53, 90 % interval coverage 0.74 | row 6; `summary.json` `headline` |
| v2 discharge window only, plain Ridge | MAE 1.65 | row 8 |
| v2 relative mode (plain Ridge / nested selection) | MAE 1.08 / 1.02 | row 9; `summary.json` `secondary` |
| Training-mean baseline | MAE 9.61 | `summary.json` `baselines.mean` |

v2's headline is not a dramatic improvement over the honest v1 number (2.53 against 3.22),
and it should not be read as one. The difference that matters is what the inputs require: v1
needs a complete discharge, which measures capacity directly; v2 needs a voltage window that
covers a fraction of a cycle, or a commissioning record. The
[README](../README.md#failure-analysis-b0006) and [METHODOLOGY](METHODOLOGY.md) discuss why the
pre-specified headline is pulled up by one cell (B0006).

## Lessons

1. **Declare when each input becomes available.** A registry entry with `availability` and
   `leakage_risk` costs one line per feature and catches what correlation statistics miss.
2. **Evaluate at the unit of deployment.** A deployed estimator meets a new cell; cycles of
   one cell are strongly dependent, so row-level random or within-battery splits measure
   memorisation.
3. **Make leakage checks executable.** A disjointness assertion on every fold, a refusing
   feature-set accessor and a schema that rejects label-side inputs are cheaper than any
   review.
4. **Treat near-perfect scores on ageing data as a defect report.** R² near 1 for SOH from
   one cycle's data is far more likely to be a leak than a model.
5. **Pre-specify the headline and publish it when it is unflattering.** v2's headline
   configuration (`combined` indicators, absolute mode, `ridge_nfs`) was written into
   [`configs/benchmark.yaml`](../configs/benchmark.yaml) before any `ridge_nfs` result was
   computed, and it is reported unchanged
   ([ADR 0006](adr/0006-pre-specified-headline-and-nested-selection.md)). It was not chosen
   blind to these four cells: the indicator windows, feature sets and relative mode come from
   an audit prototype that had already scored plain Ridge leave-one-battery-out on them (MAE
   about 1.6-1.7), and plain-Ridge results for this same `combined`/absolute specification on
   the v2 table (2.60; B0006 5.83), the no-CV-time ablation and relative mode were known when
   the headline was fixed. The declaration guards against picking the best of the 114
   leaderboard rows afterwards; it does not make the design independent of these cells.
