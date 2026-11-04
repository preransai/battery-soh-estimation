# Notebooks

| Notebook | What it shows |
|---|---|
| [`01_walkthrough.ipynb`](01_walkthrough.ipynb) | A short, executed tour of the project for reviewers |

`01_walkthrough.ipynb` is committed **with its outputs**, so it reads on GitHub without running
anything. Every number in it is computed from files in this repository:
`data/processed/nasa_features.csv`, `reports/benchmark/summary.json`, `reports/audit/audit.json` and
`artifacts/models/`. It does not download the raw NASA data. A full run re-fits the
leave-one-battery-out models live, so it takes from about a minute to a few minutes depending
on the machine.

## What it covers

1. **Setup**: imports from the installed `battery_soh` package; paths from
   `battery_soh.config.get_paths()`; library versions.
2. **The data**: the feature table, the four core cells (B0005, B0006, B0007, B0018) and the
   12-cell extended set, plus the SOH trajectories of the core cells.
3. **The leakage trap, demonstrated**: within one cell, SOH is an exact linear function of capacity
   (R² = 1). A v1-style model that takes capacity as an input looks accurate on v1's contaminated
   split and degrades under leave-one-battery-out. The guards (`get_feature_set`,
   `assert_disjoint`, `check_no_leakage`) reject each v1 mistake.
4. **Partial-window health indicators**: definitions from the feature registry, per-cell
   correlations with SOH, leave-one-battery-out R² of each indicator on its own, and how well
   each one reconstructs capacity.
5. **Honest evaluation**: leave-one-battery-out ridge models on the `discharge` and `combined`
   sets, run live next to the baselines and checked against the committed benchmark. Also the
   pre-specified headline result (with what was known when it was fixed), an analysis of why
   B0006 fails (over-estimation caused by cell-specific offsets), relative mode with plain
   Ridge and with nested selection, and the extended multi-condition results.
6. **Uncertainty**: per-cell coverage of battery-grouped conformal intervals, measured on
   held-out cells (with three calibration cells there is no battery-level guarantee), and why
   B0006 is under-covered.
7. **Using the shipped model**: manifest verification, `LinearBundle` prediction with an
   interval, an exact additive explanation and a domain check, plus the equivalent
   `POST /v1/soh/estimate` request.
8. **Where to go next**: links to the README and the methodology and leakage-audit docs.

## Running it

From the repository root, in a virtual environment:

```bash
python -m pip install -e ".[notebook]" -c requirements/constraints.txt   # matplotlib, JupyterLab, nbconvert
jupyter lab notebooks/01_walkthrough.ipynb
```

`requirements/constraints.txt` pins the library versions that produced the committed results,
so the live numbers match the committed benchmark exactly.

On Windows, installing JupyterLab into a virtual environment under a very long path can fail
with `No such file or directory` on a deeply nested file, because of the 260-character path
limit. Enable long paths (`LongPathsEnabled` in the registry, or the Group Policy "Enable Win32
long paths") or create the environment under a shorter path.

To re-execute it in place, which is how the committed outputs were produced:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_walkthrough.ipynb
```

Section 7 needs the model bundles in `artifacts/models/` (they are committed; `soh train`
rebuilds them). If the bundles are missing, the notebook says so and skips that section.

## v1 notebooks

The notebooks of the first version (v1) are archived, unmaintained, in
`legacy/notebooks/`. Their results are invalid because of target leakage
and contaminated test sets; [`docs/LEAKAGE_AUDIT.md`](../docs/LEAKAGE_AUDIT.md) explains why.
