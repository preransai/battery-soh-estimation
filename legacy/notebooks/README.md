> **Archived v1 notebooks.** Outputs were stripped; they document the original pipeline and are
> not maintained. For the current walkthrough see [`notebooks/01_walkthrough.ipynb`](../../notebooks/01_walkthrough.ipynb).

# Running notebooks (Windows, macOS, Linux)

Paths are **not** hardcoded. Each notebook’s first code cell loads `notebook_setup.py`, finds the project root (the folder that contains `src/` and `config.yaml` or `phase2_config.yaml`), runs `os.chdir` there, and adds `src/` (and phase folders) to `sys.path`.

## One-time setup (recommended)

Use a **virtual environment** so package versions match `requirements.txt`.

### Windows (PowerShell or Command Prompt)

```text
cd path\to\unzipped\li_ion_battery
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r legacy/requirements-legacy-frozen.txt   # archived v1 environment (Windows pins)
python -m ipykernel install --user --name=li-ion-battery --display-name="Python (li-ion-battery)"
jupyter notebook
```

### macOS / Linux

```text
cd /path/to/unzipped/li_ion_battery
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r legacy/requirements-legacy-frozen.txt   # archived v1 environment (Windows pins)
python -m ipykernel install --user --name=li-ion-battery --display-name="Python (li-ion-battery)"
jupyter notebook
```

In the Jupyter UI, open notebooks under the `notebooks/` folder and choose the **Python (li-ion-battery)** kernel if prompted.

## Where to start Jupyter from

Either is fine:

- **Project root** (folder that contains `notebooks/`, `src/`, `config.yaml`): e.g. `jupyter notebook` then navigate to `notebooks/`.
- Or: `cd notebooks` then `jupyter notebook` (the setup cell still finds the root).

Do **not** rely on hardcoded `C:\...` or `/Users/...` paths.

## Python version

Use **Python 3.10 or 3.11** for best compatibility with the pinned stack (e.g. `torch==2.1.0` in `requirements.txt`). If you use another version and see install errors, create a new venv with 3.11.

## Phase 3 deployment notebook (FastAPI)

For `phase3_step3_deployment.ipynb`, also install API dependencies:

```text
pip install -r legacy/requirements-legacy-api.txt
```

## Files

| File | Role |
|------|------|
| `notebook_setup.py` | Resolves project root and `sys.path` on any OS |
| `../requirements-legacy-frozen.txt` | Frozen v1 environment (Windows-specific pins) |

## Sanity check (maintainers)

From the repo root, validate notebook JSON and schema (requires `pip install nbformat`):

```text
python3 -c "import nbformat; from nbformat.validator import validate; from pathlib import Path; \
[validate(nbformat.read(p,4)) for p in Path('notebooks').glob('*.ipynb')]; print('OK')"
```

Jupyter’s `.ipynb_checkpoints/` folders are **gitignored** and can be deleted; they are auto-saved copies, not source of truth.
