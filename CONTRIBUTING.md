# Contributing to battery-soh

Thanks for helping. This project estimates Li-ion state of health (SOH) from
**partial-window health indicators** and evaluates every model **leave-one-battery-out**.
One rule outranks all others: nothing derived from capacity or from the full discharge
may become a model input, and nothing from a held-out battery may reach training,
scaling, imputation or calibration. The v1 code under `legacy/` broke that rule; see
`docs/LEAKAGE_AUDIT.md` for what that did to its numbers.

## Development setup

You need Python 3.11 or 3.12 and git; Node.js is optional (the end-to-end tests in
`tests/e2e` skip without it). The NASA raw data (~200 MB) is optional too: the committed
feature table `data/processed/nasa_features.csv` is enough for the tests, the audit, the
benchmark and training.

Install with the constraints file `requirements/constraints.txt`, which pins the library
versions that produced the committed results (they are also recorded under `provenance` in
`reports/benchmark/summary.json`). Byte-exact reproduction of `reports/` needs those
versions; with other versions the linear models and baselines reproduce exactly, but tree
ensembles (HGB, random forest) can differ slightly.

### Windows (PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
python -m pip install -e ".[dev]" -c requirements/constraints.txt
pre-commit install
soh info

# Recommended for every session:
$env:OMP_NUM_THREADS = "1"            # avoids OpenMP oversubscription in scikit-learn
$env:PYTHONIOENCODING = "utf-8"
```

`make` is optional on Windows: every Makefile target is a one-line wrapper around a
command listed below, which you can run directly.

### macOS and Linux

Debian and Ubuntu need the `python3.11-venv` package (or `python3.12-venv`) first.

```bash
python3.11 -m venv .venv              # or python3.12
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]" -c requirements/constraints.txt
pre-commit install
export OMP_NUM_THREADS=1              # avoids OpenMP oversubscription in scikit-learn
soh info
```

`make install` does the same without the constraints file. `soh info` prints the version
and where data, models and reports are expected.

## Everyday commands

| Task | make (POSIX) | Direct command (any OS) |
|---|---|---|
| Lint + format check | `make lint` | `ruff check src tests` and `ruff format --check src tests` |
| Auto-format | `make format` | `ruff check --fix src tests` and `ruff format src tests` |
| Type check | `make typecheck` | `mypy src/battery_soh` |
| Fast tests | `make test-fast` | `pytest -m "not slow and not raw_data"` |
| All tests | `make test` | `pytest` |
| Coverage | `make coverage` | `pytest -m "not raw_data" --cov=battery_soh --cov-report=term-missing` |
| Download + build data | `make data` | `soh data download` then `soh data build` |
| Leakage audit | `make audit` | `soh audit` |
| Benchmark | `make benchmark` | `soh benchmark` (add `--quick` for a smoke run) |
| Train model bundles | `make train` | `soh train` |
| Figures + generated docs | `make report` | `soh report` |
| Demo data for the UI | `make demo-assets` | `soh demo-assets` |
| Static site (GitHub Pages) | `make export-static` | `soh export-static --out-dir site` |
| Everything from raw data | `make reproduce` | the data, audit, benchmark, train, demo-assets and report commands above, in that order |
| Run API + UI | `make serve` | `soh serve --reload` then open http://127.0.0.1:8000 |
| Docker image | `make docker-build docker-run` | `docker compose up --build` |

`make help` lists all targets.

## Tests

- Tests live in four folders and run with `pytest`:
  - `tests/unit`: one module at a time, mostly on synthetic data;
  - `tests/api`: the FastAPI service through its test client;
  - `tests/integration`: the audit, a quick benchmark, the feature table and the training
    bundles on the committed data, including checks that committed artifacts are up to
    date;
  - `tests/e2e`: the static site's JavaScript backend run in Node.js against the API's
    answers and validation errors (skipped when Node.js is not installed).
- They must be deterministic (fixed seeds, `battery_soh.config.RANDOM_SEED`), need no
  network, and assert ground truth: known values from the synthetic cycle generator
  (`battery_soh.synthetic`), exact identities, or hand-computed expectations, not just
  "it runs". Keep the fast suite fast (it takes a few minutes today); anything that runs a
  full benchmark belongs under `slow`.
- Markers (declared in `pyproject.toml`; unknown markers are an error):
  - `slow`: long-running tests, excluded by `-m "not slow"`.
  - `raw_data`: needs the NASA `.mat` files under `data/raw/nasa`; skipped automatically
    when they are absent, and excluded in CI with `-m "not raw_data"`.
- The fast suite is `pytest -m "not slow and not raw_data"`; run one folder with, for
  example, `pytest tests/e2e`.

## Code standards

- `ruff check`, `ruff format --check` and `mypy --strict` must pass (line length 100).
- Type hints everywhere, Google-style docstrings on public functions,
  `from __future__ import annotations`.
- Logging through `logging.getLogger(__name__)` only; no `print` in library code (the CLI
  uses `typer.echo`). Keep log text ASCII.
- Resolve files with `battery_soh.config.get_paths()`; never build paths relative to the
  working directory and never modify `sys.path`.
- Per-cycle feature failures return `float("nan")`; invalid arguments raise `ValueError`
  with a clear message.
- Everything stochastic takes a `random_state` / `seed`.
- No pickle or joblib: models are exported as JSON bundles. No network calls outside
  `battery_soh.data.download`.

### Adding a health indicator or model

1. Register the feature in `src/battery_soh/features/registry.py` with its unit,
   availability (when it can be measured) and `leakage_risk`. Production feature sets
   may only contain `leakage_risk="none"`; `get_feature_set` enforces this.
2. Compute it in the shared feature path (`battery_soh.features`), so training and the
   API's `/v1/soh/estimate-from-cycle` use the same code.
3. Test it on synthetic cycles with known answers, then run `soh audit` and
   `soh benchmark` and report the leave-one-battery-out numbers in the pull request.

## Notebooks

Exploration notebooks are welcome, but commit them without outputs: the `nbstripout`
pre-commit hook strips outputs, execution counts and widget state (outputs embed local
paths and data). Results belong in `reports/`, produced by the CLI. The v1 notebooks in
`legacy/notebooks/` are kept for reference only.

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```text
<type>(<optional scope>): <imperative summary, at most ~72 characters>

<optional body: what and why, wrapped at 72 characters>
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `build`, `ci`, `chore`.
Typical scopes: `data`, `features`, `eval`, `models`, `api`, `ui`, `docker`, `docs`.
Mark breaking changes (API or bundle schema, CLI flags, feature names) with `!` after the
type/scope and a `BREAKING CHANGE:` footer. Examples:

```text
feat(features): add 4.00-4.19 V CC charge time for multi-condition cells
fix(api): return 422 instead of 500 for an empty discharge array
docs: explain why reference cycles are excluded from scoring
```

## Pull requests

- Keep each pull request focused; fill in the template, including before/after numbers
  when results change.
- CI must be green: lint and types, tests on Ubuntu (Python 3.11, 3.12) and Windows
  (3.11), the audit and quick benchmark reproduced from the committed feature table, and
  a Docker build that is started read-only and answers a sample request.
- Regenerated artifacts (`reports/`, `artifacts/models/`, `data/processed/`) are
  committed only together with the code or config change that produced them.

## Releases

Bump `version` in `pyproject.toml`, update `CHANGELOG.md` and `CITATION.cff`, merge,
then push a tag `vX.Y.Z` matching that version. The release workflow publishes the
multi-architecture image to `ghcr.io` with an SBOM and provenance.

## Security

Never open a public issue for a vulnerability; follow [SECURITY.md](SECURITY.md).
