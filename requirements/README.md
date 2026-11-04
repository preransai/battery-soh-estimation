# Pinned dependencies

`pyproject.toml` declares compatible ranges (`numpy>=1.26`, ...). The two files here fix
exact versions, so that a fresh environment reproduces the committed results and the
Docker image is rebuilt from the same bytes.

| File | What it is | Used by |
|---|---|---|
| `constraints.txt` | Exact versions of every package `pip install -e ".[dev,notebook]"` installs: the environment that produced the committed results (numpy 2.4.4, pandas 3.0.2, scipy 1.17.1, scikit-learn 1.8.0, matplotlib 3.10.8, ...) plus the tool versions (ruff, mypy, pytest). A constraints file installs nothing by itself; it only fixes the versions of whatever an install pulls in. | every `pip install` in CI, and local development |
| `runtime.lock` | The runtime dependencies of the service (core + `[api]` extra) with SHA-256 hashes, resolved for every platform (`uv pip compile --universal`) under `constraints.txt`. | the `Dockerfile` (`pip install --require-hashes -r requirements/runtime.lock`) |

## Installing

```bash
python -m pip install -e ".[dev]" -c requirements/constraints.txt        # development
python -m pip install -e ".[notebook]" -c requirements/constraints.txt   # walkthrough notebook
```

Without `-c`, pip picks the newest releases within the `pyproject.toml` ranges. The code
works with them, but some committed numbers (for example every gradient-boosting benchmark
row) and the rendered figures can differ slightly, and the committed-artifact tests in
`tests/integration` then fail as designed.

## Refreshing

Upgrade on purpose, not by accident: a new numpy, scipy, scikit-learn or matplotlib can move
committed numbers, so refreshing the pins goes together with re-running the pipeline.

1. Upgrade the environment, for example `python -m pip install -U -e ".[dev,notebook]"`.
2. Re-run the pipeline and check the diff of the generated files:
   `soh audit`, `soh benchmark`, `soh train`, `soh report`, then re-execute the notebook
   (`jupyter nbconvert --to notebook --execute --inplace notebooks/01_walkthrough.ipynb`).
3. Rewrite `constraints.txt` from that environment: `python -m pip freeze`, restricted to the
   dependency closure of `battery-soh[dev,notebook]` (no editable install, no unrelated
   packages, no `setuptools`, no Windows-only packages such as `pywinpty`; the one exception
   is `colorama`, which the universal runtime lock contains and which is pinned so that
   regenerating the lock is reproducible).
   Check that it resolves on every supported platform, for example with
   `uv pip compile pyproject.toml --extra dev --extra notebook -c requirements/constraints.txt --python-platform x86_64-unknown-linux-gnu --python-version 3.12 --no-build`
   (repeat for `aarch64-apple-darwin`, `x86_64-pc-windows-msvc` and Python 3.11).
4. Regenerate the lock with the same `uv` version CI uses (pinned in `constraints.txt`):

   ```bash
   python -m pip install uv -c requirements/constraints.txt
   uv pip compile pyproject.toml --extra api --universal --python-version 3.11 \
       --generate-hashes -c requirements/constraints.txt -o requirements/runtime.lock
   ```

   Run it from the repository root and keep the arguments exactly as above: the command is
   recorded in the file header, and CI regenerates the lock with it and fails if the result
   differs from the committed file.
5. Bump `ruff` in `.pre-commit-config.yaml` together with its pin here.

Dependabot proposes updates for the `pyproject.toml` ranges, the GitHub Actions and the base
image; the two files here are refreshed by hand with the steps above.
