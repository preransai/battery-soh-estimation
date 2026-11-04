# Processed data

| File | What it is |
|---|---|
| `nasa_features.csv` | One row per discharge of the 34 NASA PCoE cells: metadata, labels and every registry feature. 2,794 rows x 33 columns, about 0.66 MB. |
| `nasa_features.meta.json` | Sidecar written with the CSV: its SHA-256 and size, package and feature-extractor versions, per-battery row and QC-valid counts, the NaN rate of every feature, and the build time (UTC). |

The table is committed on purpose: CI, the benchmark (`soh benchmark`), the leakage audit
(`soh audit`) and reviewers can reproduce every reported number without the ~200 MB raw
download. Nothing else in `data/processed/` is tracked.

## Regenerate

```bash
soh data download   # once: official NASA archive -> data/raw/nasa (SHA-256 verified)
soh data build      # -> data/processed/nasa_features.csv + nasa_features.meta.json (~40 s)
```

The same from Python:

```python
from battery_soh.config import get_paths
from battery_soh.features.build import build_feature_table, load_feature_table, write_feature_table

table = build_feature_table(get_paths().nasa_raw)          # all 34 cells; or battery_ids=["B0005"]
write_feature_table(table, get_paths().features_csv)
table = load_feature_table()                               # dtypes restored, header checked
```

The build is deterministic: the same raw files and code give a byte-identical CSV (only
`created_utc` in the sidecar changes). `tests/integration/test_feature_table.py` checks this
against the raw files when they are present (`raw_data` marker), and checks the committed file's
integrity (schema, order, labels, SHA-256, leakage guards) in CI without them.

## How a row is built

Code: `battery_soh.features.build`. Every rule below decides what information a row may
contain, so each one is also a leakage rule.

1. **One row per discharge**, in recorded order. `cycle_index` is the 0-based discharge count of
   the cell, QC-invalid discharges included, so it is a time axis.
2. **Charge pairing.** The discharge is paired with a charge only when the operation right before
   it, ignoring impedance measurements, is a (non-empty) charge. A discharge that follows another
   discharge gets NaN charge features instead of borrowing an older charge. When a charge was
   restarted (two charges in a row), the second one - the charge that actually conditioned the
   cell - is used.
3. **EIS as-of join.** `eis_*` come from the most recent impedance measurement recorded *before*
   the discharge. Cycles before the first measurement are NaN: nothing is back-filled from the
   future. Failed equivalent-circuit fits (missing, non-positive, or outside the registry's
   plausible range, such as the 1e15 ohm values of B0050/B0052) are skipped, so the previous
   valid measurement stays in effect.
4. **Features** come from `battery_soh.features.extract.features_from_segments`, the same
   function the API's `POST /v1/soh/estimate-from-cycle` calls (train/serve parity), with the
   registry's default windows. `ctx_cycle_index` and `ctx_ambient_c` are added by the builder.
5. **Labels and QC** are computed per battery over all its discharges (`battery_soh.labels`,
   `battery_soh.qc`): `C0` is the median of the cell's first five QC-valid capacities,
   `soh_pct = 100 * capacity_ah / c0_ah`. Labels are filled in for QC-invalid rows too, so the
   table is complete; training and scoring use `qc_valid` rows only.

Which rows to use: the headline benchmark uses the QC-valid rows of
`battery_soh.data.protocols.CORE_BATTERIES` (B0005, B0006, B0007, B0018); the multi-condition
benchmark uses `EXTENDED_BATTERIES` (`usable` in the protocol registry). Excluded cells stay in
the table, with their reasons in the registry, so the exclusions can be audited.

## File format

* UTF-8, comma-separated, LF line endings, header = the column contract below, in that order.
* Rows sorted by `(battery_id, cycle_index)`; that pair is unique.
* Floats written with 6 significant digits (`%.6g`), far below sensor resolution. NaN (not
  computable) is an empty field - never `0`.
* Booleans are `True` / `False`. `start_time` is ISO 8601 laboratory local time (the dataset
  records no time zone) with millisecond precision.

Read it with `load_feature_table()`, which restores the dtypes (strings, `int64`, `bool`,
`datetime64[ms]`, `float64`) and refuses a file whose header differs from the contract (for
example one built before a registry change; rebuild it).

## Columns

Feature rows (14-33) mirror `battery_soh.features.registry.FEATURES`; the integration tests fail
if this table drifts from the registry. *Leakage risk* follows the registry: `label` columns are
the label or its numerator, `proxy` columns are v1 full-discharge statistics that rebuild it.
Neither may be a model input: `get_feature_set` refuses them and
`battery_soh.evaluation.leakage.check_no_leakage` flags them. Columns 1-13 are metadata and
labels, not features.

| # | column | unit | family | availability | leakage risk | meaning |
|---|---|---|---|---|---|---|
| 1 | `battery_id` | - | id | - | - | NASA cell identifier (`B0005` ... `B0056`). |
| 2 | `group` | - | id | - | - | Test campaign and condition from the protocol registry (`battery_soh.data.protocols`), e.g. `FY08Q4_2A_24C`. |
| 3 | `cycle_index` | cycles | id | - | - | 0-based count of the cell's discharges in recorded order (QC-invalid discharges included). |
| 4 | `start_time` | ISO 8601 | metadata | - | - | Start of the discharge, laboratory local time without time zone, millisecond precision. |
| 5 | `ambient_c` | degC | metadata | - | - | Chamber temperature recorded with the discharge (protocol nominal value if missing). Same as `ctx_ambient_c`. |
| 6 | `discharge_current_a` | A | metadata | - | - | Measured median load current. Same as `ctx_discharge_current_a`. |
| 7 | `has_charge` | bool | metadata | - | - | True when the operation right before the discharge (ignoring EIS) is a non-empty charge; otherwise every `chg_*` value is NaN. |
| 8 | `capacity_ah` | Ah | label | full_discharge | label | Discharge capacity stored in the dataset: the SOH numerator. Never an input. |
| 9 | `c0_ah` | Ah | label | full_discharge | label | Reference capacity: median of the cell's first 5 QC-valid capacities. |
| 10 | `soh_pct` | % | label | full_discharge | label | Headline label, `100 * capacity_ah / c0_ah`. |
| 11 | `soh_rated_pct` | % | label | full_discharge | label | `100 * capacity_ah / 2.0` (nameplate reference). |
| 12 | `qc_valid` | bool | label | - | - | Capacity passed QC (`battery_soh.qc`); train and score on these rows only. |
| 13 | `qc_reason` | - | label | - | - | `ok`, `missing`, `below_min`, `isolated` or `hampel_outlier`. |
| 14 | `chg_cc_time_390_419_s` | s | charge | partial_charge | none | Time for the voltage to rise from 3.90 V to 4.19 V during constant-current charge. |
| 15 | `chg_cc_time_400_419_s` | s | charge | partial_charge | none | Time for the voltage to rise from 4.00 V to 4.19 V during constant-current charge. |
| 16 | `chg_cv_time_s` | s | charge | partial_charge | none | Duration of the constant-voltage phase (V >= 4.19 V until current < 0.1 A). |
| 17 | `chg_ica_peak_height_ah_per_v` | Ah/V | charge | partial_charge | none | Height of the incremental-capacity (dQ/dV) peak on the CC charge segment, 3.90-4.18 V. |
| 18 | `chg_ica_peak_voltage_v` | V | charge | partial_charge | none | Voltage at which the incremental-capacity peak occurs. |
| 19 | `chg_temp_rise_cc_c` | degC | charge | partial_charge | none | Cell temperature rise between 3.90 V and 4.19 V in the CC charge phase. |
| 20 | `dis_q_380_350_ah` | Ah | discharge | partial_discharge | none | Charge delivered while the loaded voltage falls 3.80 V -> 3.50 V (~40% of a discharge). |
| 21 | `dis_time_380_350_s` | s | discharge | partial_discharge | none | Duration of the 3.80 V to 3.50 V discharge window. |
| 22 | `dis_r0_ohm` | ohm | discharge | partial_discharge | none | Ohmic resistance from the voltage step at load-on (dV/dI). |
| 23 | `dis_temp_rise_win_c` | degC | discharge | partial_discharge | none | Cell temperature rise inside the 3.80 V to 3.50 V discharge window. |
| 24 | `eis_re_ohm` | ohm | impedance | eis | none | Electrolyte resistance from the most recent EIS measurement before the cycle. |
| 25 | `eis_rct_ohm` | ohm | impedance | eis | none | Charge-transfer resistance from the most recent EIS measurement before the cycle. |
| 26 | `ctx_cycle_index` | cycles | context | metadata | none | Zero-based count of discharge cycles completed by the cell. |
| 27 | `ctx_ambient_c` | degC | context | metadata | none | Ambient (chamber) temperature of the test protocol. |
| 28 | `ctx_discharge_current_a` | A | context | partial_discharge | none | Median absolute current under load during discharge. |
| 29 | `leg_capacity_ah` | Ah | legacy | full_discharge | label | v1 input: full-discharge capacity. It is the numerator of the SOH label (target leakage). |
| 30 | `leg_record_time_s` | s | legacy | full_discharge | proxy | v1 'discharge_time': record length including the post-cutoff rest (a capacity proxy). |
| 31 | `leg_current_mean_a` | A | legacy | full_discharge | proxy | v1 whole-record mean \|I\|; encodes the load fraction of the record (a capacity proxy). |
| 32 | `leg_voltage_mean_v` | V | legacy | full_discharge | proxy | v1 whole-record mean voltage (needs the full discharge). |
| 33 | `leg_temp_max_c` | degC | legacy | full_discharge | proxy | v1 whole-record maximum temperature (needs the full discharge). |

## Row counts

Rows / QC-valid rows per cell (the sidecar's `row_counts` and `qc_valid_counts`). "Use" is the
cell's role from the protocol registry; excluded cells list the usability criteria they fail.

| battery | group | use | rows | QC-valid |
|---|---|---|---|---|
| B0005 | `FY08Q4_2A_24C` | core | 168 | 168 |
| B0006 | `FY08Q4_2A_24C` | core | 168 | 168 |
| B0007 | `FY08Q4_2A_24C` | core | 168 | 168 |
| B0018 | `FY08Q4_2A_24C` | core | 132 | 132 |
| B0025 | `ARC25-28_SQW4A_24C` | excluded (protocol, history, labels) | 28 | 28 |
| B0026 | `ARC25-28_SQW4A_24C` | excluded (protocol, history, labels) | 28 | 27 |
| B0027 | `ARC25-28_SQW4A_24C` | excluded (protocol, history, labels) | 28 | 28 |
| B0028 | `ARC25-28_SQW4A_24C` | excluded (protocol, history, labels) | 28 | 28 |
| B0029 | `ARC29-32_4A_43C` | extended | 40 | 40 |
| B0030 | `ARC29-32_4A_43C` | extended | 40 | 40 |
| B0031 | `ARC29-32_4A_43C` | extended | 40 | 40 |
| B0032 | `ARC29-32_4A_43C` | extended | 40 | 39 |
| B0033 | `ARC33-36_4A_24C` | excluded (reference, labels) | 197 | 184 |
| B0034 | `ARC33-36_4A_24C` | excluded (labels) | 197 | 194 |
| B0036 | `ARC33-36_2A_24C` | extended | 197 | 194 |
| B0038 | `ARC38-40_MIXED` | excluded (protocol, reference, labels) | 47 | 44 |
| B0039 | `ARC38-40_MIXED` | excluded (protocol, reference, labels) | 47 | 43 |
| B0040 | `ARC38-40_MIXED` | excluded (protocol, reference, labels) | 47 | 45 |
| B0041 | `ARC41-44_MIXED` | excluded (protocol, history, reference, labels) | 67 | 25 |
| B0042 | `ARC41-44_MIXED` | excluded (protocol) | 112 | 65 |
| B0043 | `ARC41-44_MIXED` | excluded (protocol) | 112 | 65 |
| B0044 | `ARC41-44_MIXED` | excluded (protocol) | 112 | 65 |
| B0045 | `ARC45-48_1A_4C` | excluded (reference, labels) | 72 | 69 |
| B0046 | `ARC45-48_1A_4C` | extended | 72 | 68 |
| B0047 | `ARC45-48_1A_4C` | extended | 72 | 68 |
| B0048 | `ARC45-48_1A_4C` | extended | 72 | 68 |
| B0049 | `ARC49-52_2A_4C` | excluded (history, labels) | 25 | 21 |
| B0050 | `ARC49-52_2A_4C` | excluded (history, reference, labels) | 25 | 9 |
| B0051 | `ARC49-52_2A_4C` | excluded (history, windows, labels) | 25 | 22 |
| B0052 | `ARC49-52_2A_4C` | excluded (history, labels) | 25 | 3 |
| B0053 | `ARC53-56_2A_4C` | excluded (labels) | 56 | 55 |
| B0054 | `ARC53-56_2A_4C` | excluded (labels) | 103 | 101 |
| B0055 | `ARC53-56_2A_4C` | excluded (labels) | 102 | 101 |
| B0056 | `ARC53-56_2A_4C` | excluded (labels) | 102 | 101 |
| **total** | | | **2,794** | **2,516** |

Core cells: 636 rows, all QC-valid. Extended set (12 cells, core included): 1,209 rows, 1,193 QC-valid.

## Coverage and known gaps

NaN rates are in the sidecar (`nan_rate`, all rows). On QC-valid rows:

* **Core cells.** Discharge-window features are complete (0 % NaN). Charge features are NaN in
  1-4 % of rows: cycles without a preceding charge (cycle 89 of B0005-B0007) and top-up charges
  that start at or above 4.19 V (cycle 30 of B0005-B0007, cycles 45 and 55 of B0018), which
  have neither a CC window nor a CC-to-CV transition. The exception is
  `chg_cc_time_390_419_s` / `chg_temp_rise_cc_c` in B0006: 28 % NaN because its CC charge
  often starts above 3.85 V (`chg_cc_time_400_419_s` is NaN in only 1.8 %). EIS is NaN for the
  first 19 cycles of B0005-B0007, which precede their first impedance measurement (as-of join,
  no back-fill).
* **Extended cells.** `dis_*` window features are NaN in 68 % of B0030 and all of B0032 cycles
  (under the 4 A load the voltage starts below 3.80 V). In B0046-B0048 the CC-window and ICA
  features are NaN in 12-57 % of cycles (`chg_cv_time_s` is complete). The ICA peak voltage of B0031 sits at the 3.90 V grid edge in every
  cycle (uninformative there).
* **Excluded cells.** 12 charges of B0042 (mixed protocol) give `chg_temp_rise_cc_c` of about
  -21 degC, below the registry's plausible range; B0050 and B0052 contain diverged EIS fits,
  which the as-of join skips.
