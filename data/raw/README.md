# Raw data

This folder holds the raw datasets. **Nothing in it is committed except this file**
(`data/raw/*` is git-ignored): the NASA files are not redistributed by this repository and
must be downloaded from NASA's official source.

## NASA PCoE "Battery Data Set" (used by v2)

### Get it

```bash
soh data download                   # -> data/raw/nasa (about 200 MB download)
soh data download --dest /data/nasa # elsewhere; then set SOH_RAW_DIR=/data
soh data download --force           # download and verify again
```

The command (`battery_soh.data.download.download_nasa`) streams the archive from
`https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip`, verifies its SHA-256
against the recorded value
`82302a7db4fc1b34e0b6676326610438d43b816bdf11a69d1d012a464ef2f92e`, unpacks the nested zip
files with path-traversal and size checks, and installs the campaign folders. It is
idempotent: when all 34 batteries are already present it returns without touching the
network. Only the Python standard library is used for the download.

If you fetched the archive by hand, unzip it (including the six inner zips) so that the
layout below appears under `data/raw/nasa/`; the loader searches recursively, so any folder
names work as long as the files are called `B00XX.mat`.

### Expected layout

```
data/raw/nasa/
  1. BatteryAgingARC-FY08Q4/            B0005, B0006, B0007, B0018 (.mat) + README.txt
  2. BatteryAgingARC_25_26_27_28_P1/    B0025-B0028 + README.txt
  3. BatteryAgingARC_25-44/             B0025-B0034, B0036, B0038-B0044 + README_*.txt
  4. BatteryAgingARC_45_46_47_48/       B0045-B0048 + README_45_46_47_48.txt
  5. BatteryAgingARC_49_50_51_52/       B0049-B0052 + README_49_50_51_52.txt
  6. BatteryAgingARC_53_54_55_56/       B0053-B0056 + README_53_54_55_56.txt
```

That is 34 distinct cells in 38 `.mat` files: B0025-B0028 ship twice (folders 2 and 3). The
copies are byte-identical; `find_mat_files` keeps the first in sorted path order and logs the
MD5 comparison at DEBUG level.

Each `B00XX.mat` holds one MATLAB struct, `B00XX.cycle`, with one entry per operation
(`type` = charge, discharge or impedance; `ambient_temperature`; `time` as a MATLAB date
vector; `data` with `Voltage_measured`, `Current_measured`, `Temperature_measured`, `Time`,
plus `Capacity` for discharges and `Re`/`Rct` for impedance). Discharge current is negative,
charge current positive. `battery_soh.data.nasa.load_battery` parses a file into typed
`Segment` objects; see its module docstring for the known quirks (non-finite samples, empty
capacities, diverged EIS fits).

### Which cells are used

The per-cell test conditions, transcribed from the READMEs and checked against the data,
live in `battery_soh.data.protocols.PROTOCOLS`, together with an evidence-backed usability
decision for every cell:

| Set | Cells | Conditions |
|---|---|---|
| Core (headline benchmark) | B0005, B0006, B0007, B0018 | 2 A CC discharge, 24 deg C |
| Extended (multi-condition) | core + B0036, B0029-B0032, B0046-B0048 | + 4 A / 43 deg C, 1 A / 4 deg C |
| Excluded | the other 22 cells | square-wave or mixed loads, software crashes, formation anomalies, noisy labels |

The rule and each exclusion reason are documented in `src/battery_soh/data/protocols.py`;
`tests/unit/test_data_protocols.py` re-derives every decision from these files when they are
present.

### Citation

B. Saha and K. Goebel, "Battery Data Set", NASA Prognostics Data Repository, NASA Ames
Research Center, Moffett Field, CA, 2007.

Please cite the dataset as above when you use it. This repository contains only the code
that downloads and reads the files.

## Mendeley forklift dataset (`data/raw/mendeley/`) - out of scope for v2

`data/raw/mendeley/` (Cell1-Cell3, per-round `RPT.csv` files) is the forklift-profile
dataset used by the archived v1 pipeline under `legacy/`. v2 does not read it: v1 treated each
forklift load step as a cycle, so its "SOH" labels were discharge-pulse sizes (median 0.6 %)
rather than capacity fade (see `legacy/README.md`). Like the NASA files, it is not committed.
