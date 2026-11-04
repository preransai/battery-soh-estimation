# Data card: `nasa_features.csv`

A datasheet (after Gebru et al., *Datasheets for Datasets*) for the processed per-cycle table
that every result in this repository is computed from. Column-level documentation and the
build rules are in [`data/processed/README.md`](../data/processed/README.md); this card
covers motivation, provenance, intended use and known problems.

| | |
|---|---|
| File | [`data/processed/nasa_features.csv`](../data/processed/nasa_features.csv) |
| Sidecar | [`nasa_features.meta.json`](../data/processed/nasa_features.meta.json) |
| SHA-256 | `f6c4fb709c4810f76a71eaf3ee2b99f963e0fcc94d354dbe636ae8682333d997` |
| Size | 2,794 rows x 33 columns, 656,344 bytes |
| Built | 2026-10-12 07:39:55 UTC, package 2.0.0, feature extractor 2.0.0 |
| Source | NASA PCoE Battery Data Set (Saha and Goebel, 2007) |
| Builder | `soh data build` ([`features/build.py`](../src/battery_soh/features/build.py)) |
| Format | UTF-8 CSV, LF line endings, floats `%.6g`, NaN as an empty field |

## Motivation

**Why was the table created?** To estimate lithium-ion state of health (SOH) from health
indicators that are measurable in part of a charge or discharge, and to evaluate such
estimators without target leakage. It replaces the v1 feature files, whose inputs included
the label numerator (see [LEAKAGE_AUDIT.md](LEAKAGE_AUDIT.md)).

**Who created it?** Preran Sai Yarramasu, from the publicly released NASA data. The raw
measurements were produced by the NASA Ames Prognostics Center of Excellence.

**Why is a derived table committed?** So that CI, reviewers and readers can reproduce the
benchmark, the audit and the model bundles in minutes without the ~200 MB raw download.

## Composition

**What does a row represent?** One discharge operation of one cell, in recorded order, with
the charge that immediately preceded it (if any) and the latest impedance measurement
recorded before it.

**How many rows?** 2,794 discharges of 34 cells, of which 2,516 pass the capacity quality
control. Per cell: 25 to 197 rows. Role of each cell, from the protocol registry's inclusion
rule:

| Set | Cells | Rows | QC-valid rows |
|---|---|---:|---:|
| Core (headline benchmark) | B0005, B0006, B0007, B0018 | 636 | 636 |
| Extended (multi-condition benchmark, includes core) | core + B0029-B0032, B0036, B0046-B0048 | 1,209 | 1,193 |
| Excluded (kept in the table with reasons) | the other 22 cells | 1,585 | 1,323 |

**What does each row contain?**

- Identifiers and metadata: `battery_id`, campaign `group`, `cycle_index` (0-based discharge
  count, a time axis), `start_time`, ambient temperature, median load current, `has_charge`.
- Labels (label side, never model inputs): `capacity_ah`, `c0_ah`, `soh_pct`,
  `soh_rated_pct`, `qc_valid`, `qc_reason`.
- 20 registry features: 6 charge, 4 discharge, 2 EIS, 3 context, and 5 `leg_*` columns
  that reproduce v1's full-discharge inputs for the audit only.

**Is there a label?** Yes. `soh_pct = 100 * capacity_ah / c0_ah`, where `c0_ah` is the
median of the cell's first five QC-valid capacities; `soh_rated_pct` uses the 2.0 Ah
nameplate instead.

**Is anything missing?** Yes, by design: a feature that cannot be computed is NaN, never a
placeholder. Notable gaps (QC-valid rows):

- Core cells: discharge-window features are complete. Charge features are NaN in 1-4 % of
  cycles (no preceding charge, or a top-up charge starting above 4.19 V); the 3.90 V charge
  window is NaN in 28 % of B0006's cycles because its CC phase often starts above 3.85 V.
  EIS is NaN for the first 19 cycles of B0005-B0007 (before their first impedance
  measurement; no back-filling from the future).
- Extended cells: the discharge window is NaN in 68 % of B0030's and all of B0032's cycles
  (under 4 A at 43 °C the loaded voltage starts below 3.80 V); in B0046-B0048 the CC-window
  and ICA features are NaN in 12-57 % of cycles; B0031's ICA peak voltage sits at the
  3.90 V grid edge in every cycle.
- NaN rates over all rows are in the sidecar (`nan_rate`), for example 43.6 % for
  `chg_cc_time_390_419_s` and 41.3 % for `dis_q_380_350_ah`; most of that comes from
  excluded cells whose protocols never reach the windows.

**Are there errors or noise?** The capacity field contains truncated records, near-zero
values and formation cycles; QC flags them (reason codes `missing`, `below_min`, `isolated`,
`hampel_outlier`). Some equivalent-circuit EIS fits in B0050 and B0052 diverged (up to 1e15
Ω); the EIS join skips fits outside the registry's plausible range. 12 charges of B0042
(mixed protocol) give a charge temperature rise of about -21 °C, outside the plausible
range; that cell is excluded. Labels are at most as accurate as NASA's capacity integration.

**Is the table self-contained?** For the benchmark, audit and training, yes. The window-sweep
experiment and the demo assets need the raw `.mat` files.

**Does it contain personal or sensitive data?** No. It describes commercial 18650 cells in a
laboratory.

**Are there relationships between rows?** Rows of the same cell are strongly dependent
(consecutive cycles of one ageing trajectory). Treat the **cell** as the independent unit:
split, resample and calibrate by `battery_id`, never by row.

## Collection process

The NASA team cycled commercial 18650 cells (2.0 Ah nameplate) in several campaigns (the
recorded discharge start times run from April 2008 to September 2010) with three operation
types: charge (CC 1.5 A to 4.2 V, then CV until 20 mA, identical for every cell), discharge
(constant current, square-wave or mixed loads to a campaign-specific cut-off voltage, at
ambient temperatures between 4 and 44 °C) and impedance (EIS
sweep, with fitted electrolyte and charge-transfer resistances). Each operation records
voltage, current and temperature against time; discharges also store the integrated
capacity. The protocol of every cell, transcribed from the dataset's README files and checked
against the `.mat` contents, is in
[`data/protocols.py`](../src/battery_soh/data/protocols.py); where README and data disagree
(for example B0028's load amplitude), the entry records the discrepancy.

The raw archive is retrieved by `soh data download` from
`https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip` and verified against
the SHA-256 `82302a7db4fc1b34e0b6676326610438d43b816bdf11a69d1d012a464ef2f92e`
([`data/raw/README.md`](../data/raw/README.md)). B0025-B0028 appear in two folders as
byte-identical copies and are de-duplicated.

## Preprocessing, cleaning and labelling

All steps are deterministic code; the same raw files and code give a byte-identical CSV.

1. **Parsing.** Every `.mat` record becomes a typed `Segment` (charge, discharge or
   impedance) with NASA's sign convention (discharge current negative).
2. **Joining.** One row per discharge; charge pairing only when the previous non-EIS
   operation is a non-empty charge (a restarted charge pairs with the second attempt); EIS
   values as of the most recent valid measurement before the discharge.
3. **Features.** `features_from_segments`, the same function the API calls, with the
   registry's default windows. Per-cycle failures are NaN.
4. **Quality control.** Capacity flags as described in
   [METHODOLOGY section 4](METHODOLOGY.md#4-capacity-quality-control).
5. **Labels.** Per cell over all its discharges; `c0_ah` from the first five QC-valid
   capacities. Labels are filled in for invalid rows too, so the table is complete; use
   `qc_valid` rows only.
6. **Writing.** Rows sorted by `(battery_id, cycle_index)`, 6 significant digits, sidecar
   with SHA-256, row counts, QC counts and NaN rates. `load_feature_table()` restores dtypes
   and refuses a file whose header differs from the registry contract.

The raw data is not modified or redistributed.

## Uses

**Intended.** Benchmarking SOH estimators that use partial-window indicators under
leave-one-battery-out evaluation; studying leakage in battery-health modelling; teaching.

**Use with care.**

- The `leg_*` columns and every label column are **not** valid model inputs.
  `get_feature_set` refuses them and `check_no_leakage` flags them; they exist so that the
  v1 failure can be reproduced.
- The four core cells share one protocol; results do not transfer to other chemistries,
  form factors, chargers or load profiles without re-validation.
- Relative-mode features must be computed from the cell's own first QC-valid cycles; those
  cycles must then be excluded from scoring.

**Not intended.** Safety-critical battery management, warranty or second-life decisions
about real cells.

## Distribution

The table is part of this repository under the project's MIT license, as a derived product of
the publicly released NASA data. The raw NASA files are not redistributed; obtain them from
NASA's Prognostics Data Repository and cite the dataset:

> B. Saha and K. Goebel (2007). *Battery Data Set*. NASA Ames Prognostics Data Repository,
> NASA Ames Research Center, Moffett Field, CA.

## Maintenance

- Rebuild with `soh data download && soh data build`. Any change to a feature definition must
  bump `FEATURE_EXTRACTOR_VERSION` in
  [`features/extract.py`](../src/battery_soh/features/extract.py); the sidecar and every model
  bundle record that version and the table's SHA-256, so stale artifacts are detectable.
- `tests/integration/test_feature_table.py` checks the committed file's schema, row order,
  labels, hash and leakage guards in CI, and its byte-identical rebuild when the raw files are
  present.
- Issues and corrections: through the repository's issue tracker.
