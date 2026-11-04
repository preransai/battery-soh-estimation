# 3. JSON model bundles, no pickle

Date: 2026-10-12

## Status

Accepted

## Context

v1's API loaded its models, scalers and auxiliary objects with `pickle.load`
([`legacy/api/main.py`](../../legacy/api/main.py#L186-L207)). Unpickling executes code chosen
by whoever wrote the file, so a model artifact is effectively an executable. Pickles are also
coupled to the exact library versions that wrote them, and they cannot be reviewed in a pull
request.

The production models in v2 are small pipelines: median imputation, standardisation and a
linear estimator over at most eight inputs.

Alternatives considered:

- *pickle / joblib*: rejected for the reasons above.
- *skops*: safer loading, but still tied to scikit-learn's object model and versions, and the
  artifact is not human-readable.
- *ONNX*: would also cover tree ensembles, but adds a runtime and produces an opaque binary
  for a model that is a dot product.

## Decision

- Production models are exported by
  [`models/linear_bundle.py`](../../src/battery_soh/models/linear_bundle.py) to a versioned JSON
  document (`schema_version`): feature names, imputation medians, scaler mean and scale,
  coefficients and intercept, output clip, conformal half-widths per confidence level, the
  training-domain envelope and precision matrix, the training record (cells, cycle count,
  feature-table SHA-256) and the leave-one-battery-out metrics.
- Export is verified against the fitted scikit-learn pipeline (relative tolerance `1e-9`).
- `LinearBundle` evaluates a bundle with NumPy only; the API never imports scikit-learn. The
  web UI's static mode evaluates the same JSON in the browser.
- `MANIFEST.json` lists the SHA-256 of every bundle. The API's model registry verifies it at
  start-up and is fail-closed: a missing manifest, a missing file or one mismatch means no
  model is served and `/readyz` returns 503.
- No pickle or joblib file is produced anywhere in v2; `.gitignore` excludes `*.pkl`,
  `*.pickle` and `*.joblib`.

## Consequences

- Loading a model cannot execute code, and a model change is a reviewable text diff.
- Explanations are exact: each feature's contribution is `coef_j * z_j`, and the
  contributions plus the base value equal the prediction.
- Production models are restricted to linear models. Gradient boosting looks better on the
  multi-condition benchmark (not significantly after correction) but cannot ship without a new
  artifact format; that is an explicit trade-off, revisited if the accuracy gap becomes
  significant.
- A change to the bundle schema needs a `schema_version` bump and re-export (`soh train`).
- `soh train` writes byte-identical bundles for the same inputs (its timestamp comes from the
  feature table's build time), so the manifest hashes are stable across reruns and operating
  systems.
