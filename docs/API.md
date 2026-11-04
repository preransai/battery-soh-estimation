# API reference

The service is a FastAPI application
([`src/battery_soh/api/app.py`](../src/battery_soh/api/app.py)) that serves a versioned JSON
API under `/v1`, health probes, Prometheus metrics, the OpenAPI documents and the static web
UI. It is stateless and read-only, makes no outbound network calls and writes nothing to
disk.

```bash
soh serve                       # http://127.0.0.1:8000 (development)
docker compose up --build -d    # same, hardened container
```

The OpenAPI schema is generated from the code: `/openapi.json`, interactive at `/docs`
(Swagger UI) and `/redoc`. This page explains the contract and the behaviour that a schema
cannot express.

## Contents

- [Endpoints](#endpoints)
- [Estimating SOH](#estimating-soh)
- [Estimating from a raw cycle](#estimating-from-a-raw-cycle)
- [Batch](#batch)
- [Models](#models)
- [Demo data and benchmark](#demo-data-and-benchmark)
- [Health and metrics](#health-and-metrics)
- [Validation rules and limits](#validation-rules-and-limits)
- [Errors](#errors)
- [Headers](#headers)
- [Settings](#settings)
- [Static mode](#static-mode)

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/soh/estimate` | SOH from health-indicator values |
| `POST` | `/v1/soh/estimate-from-cycle` | Extract the indicators from raw arrays, then estimate |
| `POST` | `/v1/soh/batch` | Several estimate requests in one call (atomic; 100 by default, at most 1,000) |
| `GET` | `/v1/models` | Served models with their inputs and leave-one-battery-out metrics |
| `GET` | `/v1/models/{model_id}` | Model card: the bundle without its precision matrix |
| `GET` | `/v1/demo/batteries` | Bundled demo cells and their cycles |
| `GET` | `/v1/demo/batteries/{battery_id}/cycles/{cycle_index}` | One demo cycle: raw arrays, indicators, true SOH |
| `GET` | `/v1/benchmark` | `summary.json` plus headline predictions and the leakage audit |
| `GET` | `/livez` | Liveness |
| `GET` | `/readyz` | Readiness (200 only with verified models) |
| `GET` | `/metrics` | Prometheus text format (when enabled) |
| `GET` | `/docs`, `/redoc`, `/openapi.json` | API documentation (when enabled) |
| `GET` | `/` | Web UI (when `frontend/` is present) |

Handlers that compute (`/v1/soh/*`) are synchronous functions run in FastAPI's thread pool,
so CPU work never blocks the event loop.

## Estimating SOH

`POST /v1/soh/estimate`

```json
{
  "features": {
    "chg_cc_time_390_419_s": 2075.4,
    "chg_cv_time_s": 4429.8,
    "chg_ica_peak_height_ah_per_v": 3.942,
    "chg_ica_peak_voltage_v": 4.008,
    "chg_temp_rise_cc_c": 2.95,
    "dis_q_380_350_ah": 0.6619,
    "dis_r0_ohm": 0.1011,
    "dis_temp_rise_win_c": 5.19
  },
  "confidence": 0.9
}
```

This is [`.github/fixtures/sample_estimate_request.json`](../.github/fixtures/sample_estimate_request.json):
the eight `combined` indicators of NASA cell B0005 at discharge cycle 80. CI posts it to
the running container on every change.

| Field | Type | Required | Meaning |
|---|---|---|---|
| `features` | object | yes | Indicator values keyed by registry name ([table](METHODOLOGY.md#6-health-indicators)). Omitted or `null` means missing. |
| `reference` | object | no | The cell's commissioning reference per indicator (mean over its first five valid cycles). Required by relative-mode models. |
| `model_id` | string | no | Force a model; default is automatic selection (below). |
| `confidence` | number | no | `0.8`, `0.9` (default) or `0.95`. |

Response (`200`, numbers rounded, `method` shortened):

```json
{
  "model_id": "discharge-absolute-ridge",
  "soh_pct": 84.14,
  "interval": {"lower": 80.78, "upper": 87.51, "confidence": 0.9, "method": "group conformal (cv): ..."},
  "base_value": 83.60,
  "contributions": [
    {"feature": "dis_q_380_350_ah", "value": 0.6619, "contribution": -0.10},
    {"feature": "dis_r0_ohm", "value": 0.1011, "contribution": 0.65}
  ],
  "domain": {"in_domain": true, "out_of_range": [], "mahalanobis": 0.72, "threshold": 2.91, "missing": []},
  "warnings": []
}
```

The measured SOH of that cycle is 84.98 %. B0005 is a training cell of the shipped bundles,
so the example illustrates the contract, not out-of-sample accuracy. The same request with
`"model_id": "combined-absolute-ridge"` returns 83.59 [76.73, 90.45]; with the cell's
commissioning `reference` added (from `GET /v1/demo/batteries`) the service selects
`combined-relative-ridge` and returns 84.22 [82.25, 86.19].

| Field | Meaning |
|---|---|
| `model_id` | Bundle that produced the estimate. |
| `soh_pct` | Estimated SOH in percent of the cell's own initial capacity, clipped to [0, 120]. |
| `interval` | `soh_pct` plus or minus the bundle's conformal half-width at `confidence`, clipped to the output range; `method` names the construction. With too few calibration scores the half-width is infinite and the interval spans the whole output range (with a warning). |
| `base_value` | Prediction for an average training cycle (the intercept on standardised inputs). |
| `contributions` | Exact linear attribution per input: `coef_j * (x_j - mean_j) / scale_j`. `base_value + sum(contribution)` equals the unclipped prediction. `value` is the input used (the training median when the input was missing). |
| `domain` | Applicability check: `out_of_range` lists inputs outside the training range widened by a margin; `mahalanobis` is the distance of the standardised input from the training cloud under a Ledoit-Wolf shrunk covariance, compared with `threshold`; `missing` lists imputed inputs. `in_domain` is false if either check fails. |
| `warnings` | Human-readable caveats: imputed inputs, clipped output, infinite interval, out-of-domain input. |

### Model selection

When `model_id` is omitted ([`api/inference.py`](../src/battery_soh/api/inference.py)):

1. Absolute-mode bundles are always eligible. Relative-mode bundles are eligible only when a
   `reference` is supplied, and only for indicators with a finite, non-zero reference.
2. Among bundles whose inputs are *all* available, the one with the lowest recorded
   leave-one-battery-out MAE wins (ties broken by model id).
3. If no bundle is complete, the absolute bundle covering the most provided indicators is
   used and the missing ones are median-imputed (reported in `warnings` and
   `domain.missing`). A relative bundle is never used with imputed inputs unless it is
   named explicitly.
4. If no bundle uses any provided indicator: `422 /problems/no-usable-features`.

### Relative mode

Relative-mode bundles expect each indicator divided by the cell's commissioning value. The
client sends absolute values plus the `reference`; the service divides them with the same
function the training code uses
([`features/relative.py`](../src/battery_soh/features/relative.py)). Context features are not
divided. A missing or zero reference for a required input gives
`422 /problems/reference-required`.

## Estimating from a raw cycle

`POST /v1/soh/estimate-from-cycle` runs the feature extractor that built the training table
(`features_from_segments`, train/serve parity) on raw arrays, then estimates.

```json
{
  "charge":    {"time_s": [...], "voltage_v": [...], "current_a": [...], "temperature_c": [...]},
  "discharge": {"time_s": [...], "voltage_v": [...], "current_a": [...], "temperature_c": [...]},
  "eis": {"re_ohm": 0.0551, "rct_ohm": 0.0805},
  "reference": null,
  "model_id": null,
  "confidence": 0.9
}
```

The demo endpoint returns real NASA cycles in exactly this shape, so a complete request can be
built from it:

```bash
curl -s http://127.0.0.1:8000/v1/demo/batteries/B0005/cycles/80 \
  | jq '{charge, discharge, eis}' \
  | curl -s http://127.0.0.1:8000/v1/soh/estimate-from-cycle \
      -H 'Content-Type: application/json' -d @-
```

For that cycle (arrays downsampled to at most 400 points, measurement windows kept), the
extractor returns `dis_q_380_350_ah` = 0.661888 against 0.661887 in the training table, and
the estimate equals the one above (84.14 [80.78, 87.51]).

A record must span a measurement window (3.90-4.19 V of a CC charge, or 3.80-3.50 V of a
discharge under load).

Response: `{"features": {...}, "estimate": {...}}`, where `features` holds every
non-leaky indicator the extractor measured (`null` when not computable) and `estimate` has
the shape above. Extracted values outside a feature's plausible range are dropped with a
warning; a current with the wrong sign produces a warning. If no indicator can be measured:
`422 /problems/no-health-indicators`.

The web UI's "try your own data" CSV upload uses this endpoint; the demo tab feeds it real
NASA cycles from `/v1/demo`.

## Batch

`POST /v1/soh/batch` with `{"items": [<estimate request>, ...]}` returns
`{"results": [<estimate response>, ...]}` in request order. The batch is atomic: if any item
fails, the request fails with that item's error and its index in `detail`
(`"items[3]: ..."`).

A batch holds 1 to `SOH_MAX_BATCH_ITEMS` items: 100 by default, configurable up to the
schema ceiling of 1,000. Every item is a full estimate, so this setting bounds the CPU one
request can buy. A larger batch is refused before any estimate is computed, with
`422 /problems/validation-error`:

```json
{"type": "/problems/validation-error", "title": "Invalid input", "status": 422,
 "detail": "items: List should have at most 100 items (this server's batch limit), not 250; split the request",
 "request_id": "...", "errors": [{"type": "too_long", "loc": ["body", "items"], "msg": "..."}]}
```

## Models

`GET /v1/models` lists the served bundles, best recorded MAE first:

```json
[
  {
    "model_id": "combined-relative-ridge",
    "feature_set": "combined",
    "mode": "relative",
    "features": ["chg_cv_time_s", "dis_q_380_350_ah", "dis_temp_rise_win_c"],
    "metrics": {"lobo_mae": 1.02138, "lobo_rmse": 1.22625, "coverage_90": 0.927028}
  },
  {
    "model_id": "discharge-absolute-ridge",
    "feature_set": "discharge",
    "mode": "absolute",
    "features": ["dis_q_380_350_ah", "dis_r0_ohm"],
    "metrics": {"lobo_mae": 1.7128, "lobo_rmse": 1.97041, "coverage_90": 0.843479}
  }
]
```

(two of the four entries shown). `features` are the inputs the nested selection kept when the
bundle was trained on all four core cells; the metrics are those of the same procedure under
leave-one-battery-out.

`GET /v1/models/{model_id}` returns the full bundle (preprocessing, coefficients, conformal
quantiles, domain envelope, training record, per-cell metrics) minus the domain precision
matrix. The bundle schema is described in [MODEL_CARD.md](MODEL_CARD.md#bundle-format).

## Demo data and benchmark

`GET /v1/demo/batteries` returns, per core cell, `battery_id`, `n_cycles` (discharges in the
full dataset), the commissioning `reference` per indicator, and the bundled `cycles` as
`{cycle_index, soh_pct, is_reference}`. The bundled asset holds the first five valid cycles
of each cell plus a regular subsample (26 cycles for B0005, B0006 and B0007, 22 for B0018).

`GET /v1/demo/batteries/{battery_id}/cycles/{cycle_index}` returns `battery_id`,
`cycle_index`, the true `soh_pct`, `qc_valid`, `is_reference`, the `features` exactly as the
training table has them, the downsampled raw `charge` and `discharge` arrays (at most 400
points each, measurement windows kept at full resolution where the budget allows) and the
latest `eis` values. Unknown battery or cycle: `404 /problems/demo-cycle-not-found`. The
asset is built by `soh demo-assets`; without it these endpoints return
`404 /problems/demo-unavailable`.

`GET /v1/benchmark` returns [`reports/benchmark/summary.json`](../reports/benchmark/summary.json)
with two additions: `predictions` (the headline specification's out-of-fold per-cycle
predictions from `predictions.csv`) and `audit` (the content of
[`reports/audit/audit.json`](../reports/audit/audit.json)). Files are re-read when they
change; non-finite numbers are returned as `null`. Without a summary:
`404 /problems/benchmark-unavailable`.

These three `GET` endpoints serve data that changes only when its files change. Each
response is serialized once per file version (cache key: path, modification time and size
of every source file) and sent with a weak `ETag` and `Cache-Control: public, max-age=300`.
A request whose `If-None-Match` lists the current tag gets `304 Not Modified` with no body,
so a browser revalidating after five minutes downloads the large benchmark document only
when it changed ([`api/http_cache.py`](../src/battery_soh/api/http_cache.py)).

## Health and metrics

- `GET /livez` returns `{"status": "ok"}` while the process serves HTTP.
- `GET /readyz` returns `{"status": "ready", "models": [...]}` once the model registry has
  verified every bundle against `MANIFEST.json`. Until then, or if verification failed, it
  returns `503 /problems/not-ready`. Verification is fail-closed: a missing manifest, a
  missing file or one hash mismatch means no model is served.
- `GET /metrics` exposes, per application instance:

| Series | Type | Labels |
|---|---|---|
| `soh_http_requests_total` | counter | `method`, `route` (template), `status` |
| `soh_http_request_duration_seconds` | histogram | `method`, `route` |
| `soh_http_requests_in_flight` | gauge | |
| `soh_inference_duration_seconds` | histogram | `model_id` |
| `soh_predictions_total` | counter | `model_id`, `in_domain` |
| `soh_models_loaded` | gauge | |

Route labels are templates (`/v1/models/{model_id}`), so cardinality is bounded. The share of
`in_domain="false"` predictions is an early drift signal. Process and platform collectors are
included.

## Validation rules and limits

Implementation: [`api/schemas.py`](../src/battery_soh/api/schemas.py),
[`api/settings.py`](../src/battery_soh/api/settings.py),
[`api/middleware.py`](../src/battery_soh/api/middleware.py).

- **Strict types.** Unknown fields are rejected (`extra="forbid"`). Numbers must be JSON
  numbers: strings and booleans are refused. `NaN` and `Infinity` are refused.
- **Physical ranges.** Every indicator is bounded by its registry `plausible_range`, for
  example `dis_q_380_350_ah` in [0, 5] Ah, `chg_ica_peak_voltage_v` in [3, 4.5] V,
  `dis_r0_ohm` in [0, 2] ohm. The OpenAPI schema lists every bound.
- **No leaky inputs.** Label-side and proxy features (`leg_*`) are rejected with 422:

  ```json
  {"type": "/problems/validation-error", "title": "Invalid input", "status": 422,
   "detail": "features: Value error, label-side/proxy feature(s) leg_capacity_ah are not accepted (they need the full discharge and leak the SOH label)",
   "request_id": "...", "errors": [{"type": "value_error", "loc": ["body", "features"], "msg": "..."}]}
  ```

- **Raw arrays.** 2 to 20,000 samples per array (lower with `SOH_MAX_SAMPLES`), all four
  arrays of a segment of equal length, `time_s` non-decreasing; time in [0, 1e7] s, voltage
  in [0, 10] V, current in [-100, 100] A, temperature in [-100, 200] degC. EIS resistances in
  [0, 2] ohm.
- **Identifiers.** `model_id` matches `^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$`, `battery_id`
  matches `^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$`.
- **Body size.** Requests larger than `SOH_MAX_BODY_BYTES` (default 1,000,000 bytes) are
  rejected with 413 before parsing, both from `Content-Length` and by counting the bytes of
  chunked bodies.
- **Batch size.** 1 to `SOH_MAX_BATCH_ITEMS` items (default 100; the schema ceiling is
  1,000). Above it: 422 with an error of type `too_long` at `["body", "items"]`.

## Errors

Every 4xx and 5xx response is RFC 7807 problem details with media type
`application/problem+json` and the request id:

```json
{
  "type": "/problems/model-not-found",
  "title": "Model not found",
  "status": 404,
  "detail": "No model bundle with id 'nope'. See GET /v1/models.",
  "request_id": "13b5ab87-e994-49b8-92ce-c43e17267187"
}
```

Validation errors (422) add `errors`, a list of `{type, loc, msg}` (the offending input is
never echoed back; long strings are truncated and the list is capped at 50). Unhandled
exceptions become a generic 500 with no traceback, exception text or path; the details go to
the server log under the same request id.

| Status | `type` | When |
|---|---|---|
| 404 | `/problems/model-not-found` | Unknown `model_id` |
| 404 | `/problems/demo-unavailable` | No demo asset installed |
| 404 | `/problems/demo-cycle-not-found` | Unknown demo battery or cycle |
| 404 | `/problems/benchmark-unavailable` | No `summary.json` |
| 404, 405 | `about:blank` | Unknown path, wrong method (405 carries `Allow`) |
| 413 | `/problems/payload-too-large` | Body above `SOH_MAX_BODY_BYTES` |
| 422 | `/problems/validation-error` | Schema violation (types, ranges, unknown or leaky fields, array rules, batch size) |
| 422 | `/problems/no-usable-features` | No served model uses any provided indicator |
| 422 | `/problems/reference-required` | Relative model without a usable reference |
| 422 | `/problems/unsupported-confidence` | Confidence level the bundle does not carry |
| 422 | `/problems/non-finite-result` | Inputs produce a non-finite model output |
| 422 | `/problems/invalid-record` | The feature extractor rejected the raw arrays |
| 422 | `/problems/no-health-indicators` | No measurement window found and no EIS given |
| 500 | `/problems/internal-error` | Unexpected error (quote the request id) |
| 503 | `/problems/not-ready` | `/readyz` before models are verified |
| 503 | `/problems/no-models` | Estimate requested while no model is loaded |
| 503 | `/problems/demo-unavailable` | Demo asset present but failed validation |

## Headers

Every response carries:

| Header | Value |
|---|---|
| `X-Request-ID` | Echo of a well-formed incoming id (`^[A-Za-z0-9._:-]{1,128}$`), otherwise a new UUID4 |
| `Content-Security-Policy` | `default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'` |
| `X-Content-Type-Options` | `nosniff` |
| `X-Frame-Options` | `DENY` |
| `Referrer-Policy` | `no-referrer` |
| `Permissions-Policy` | every sensitive feature disabled |
| `Cross-Origin-Opener-Policy` | `same-origin` |
| `Cross-Origin-Resource-Policy` | `same-origin` (other sites cannot embed responses as no-cors subresources) |
| `Cache-Control` | `no-store` on API routes and errors, except `public, max-age=300` with an `ETag` on `/v1/benchmark` and `/v1/demo/*` |

`/docs` and `/redoc` are the only pages with a different Content-Security-Policy
([`api/docs.py`](../src/battery_soh/api/docs.py)). Swagger UI and ReDoc load version-pinned
bundles from jsDelivr, so each page allows scripts and styles from exactly one package
directory (`https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.17.14/` for `/docs`,
`https://cdn.jsdelivr.net/npm/redoc@2.1.5/` for `/redoc`), never the whole CDN; every CDN
tag carries a Subresource Integrity hash (`integrity="sha384-..."`,
`crossorigin="anonymous"`), so a modified file is refused by the browser; Swagger's inline
bootstrap script is allowed by its SHA-256 hash, not by `'unsafe-inline'`; styles allow
`'unsafe-inline'` because both UIs set inline styles at runtime. Set
`SOH_ENABLE_DOCS=false` to remove those pages.

CORS is closed by default: no CORS headers are sent unless `SOH_CORS_ORIGINS` lists origins,
and credentials are never allowed. Responses above 1 kB are gzip-compressed when the client
accepts it.

## Settings

All settings are environment variables; nothing is read from a `.env` file, and unknown
`SOH_*` variables are ignored.

| Variable | Default | Meaning |
|---|---|---|
| `SOH_MODELS_DIR` | `artifacts/models` | Model bundles and `MANIFEST.json` |
| `SOH_DEMO_DIR` | `artifacts/demo` | `demo_cycles.json.gz` (optional) |
| `SOH_BENCHMARK_DIR` | `reports/benchmark` | `summary.json` and `predictions.csv` |
| `SOH_AUDIT_DIR` | `reports/audit` | `audit.json` (optional) |
| `SOH_FRONTEND_DIR` | `frontend` | Web UI served at `/` when it contains `index.html` |
| `SOH_CORS_ORIGINS` | empty | JSON list or comma-separated origins (`scheme://host[:port]`) |
| `SOH_MAX_BODY_BYTES` | `1000000` | Request body limit, 1,024 to 50,000,000 |
| `SOH_MAX_SAMPLES` | `20000` | Samples per raw array, 2 to 20,000 |
| `SOH_MAX_BATCH_ITEMS` | `100` | Items per `/v1/soh/batch` request, 1 to 1,000 |
| `SOH_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `SOH_LOG_JSON` | `false` | One JSON object per log line (the Compose file sets `true`) |
| `SOH_ENABLE_METRICS` | `true` | Expose `/metrics` |
| `SOH_ENABLE_DOCS` | `true` | Expose `/docs`, `/redoc`, `/openapi.json` |

Paths default relative to the project root, which `SOH_PROJECT_ROOT` overrides (`/app` in
the image); `SOH_ARTIFACTS_DIR`, `SOH_REPORTS_DIR` and `SOH_RAW_DIR` move those trees. The
container additionally reads `PORT`, `WEB_CONCURRENCY`, `LOG_LEVEL` (uvicorn) and
`FORWARDED_ALLOW_IPS`; see [DEPLOYMENT.md](DEPLOYMENT.md).

Logs: one access-log line per request with method, path, status, duration and request id
(probe paths at DEBUG); application log records carry the same `request_id`.

## Static mode

`soh export-static --out-dir site` writes the web UI plus precomputed responses
([`api/static_export.py`](../src/battery_soh/api/static_export.py)):

| File | Equivalent |
|---|---|
| `data/models.json` | `GET /v1/models` |
| `data/models/<model_id>.json` | the full bundle, including the precision matrix the browser needs for the domain check |
| `data/demo/batteries.json` | `GET /v1/demo/batteries` |
| `data/demo/<battery>/<cycle>.json` | `GET /v1/demo/batteries/<battery>/cycles/<cycle>` |
| `data/benchmark.json` | `GET /v1/benchmark` |

Every file comes from the same payload function the endpoint uses, so the shapes cannot
drift. In static mode the browser evaluates `POST /v1/soh/estimate` itself from the bundle
(imputation, standardisation, linear model, clip, interval, domain check);
`estimate-from-cycle` and `batch` need the server.
