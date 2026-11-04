# Deployment

The service ships as one container image that holds the API, the web UI and every artifact it
reads. It is stateless, read-only and makes no outbound calls, so it runs the same on a
laptop, behind a reverse proxy, on Kubernetes or on a platform-as-a-service. A backend-free
static build of the web UI can be published on GitHub Pages.

## Contents

- [Container image](#container-image)
- [Docker Compose](#docker-compose)
- [Configuration](#configuration)
- [Updating models and results](#updating-models-and-results)
- [GitHub Pages static demo](#github-pages-static-demo)
- [Releases to GitHub Container Registry](#releases-to-github-container-registry)
- [Behind a reverse proxy](#behind-a-reverse-proxy)
- [Kubernetes](#kubernetes)
- [Hugging Face Spaces and Render](#hugging-face-spaces-and-render)
- [Operations](#operations)

## Container image

Build and run it hardened (BuildKit is required; it is the default builder since Docker 23):

```bash
docker build -t battery-soh .
docker run --rm -p 127.0.0.1:8000:8000 \
  --read-only --tmpfs /tmp \
  --cap-drop ALL --security-opt no-new-privileges \
  battery-soh
# http://127.0.0.1:8000, OpenAPI docs at /docs
```

What the [Dockerfile](../Dockerfile) produces:

- A multi-stage build on `python:3.11-slim`, pinned by the digest of its multi-arch index
  (Dependabot proposes digest refreshes): runtime dependencies (the core package plus the
  `[api]` extra) installed from [`requirements/runtime.lock`](../requirements/runtime.lock)
  with `pip install --require-hashes` (exact versions, every file checked against its
  SHA-256, binary wheels only) into a virtualenv, the `battery-soh` wheel built separately
  and installed with `--no-deps`. No compiler, no pip cache, no source tree, no tests and no
  PyTorch in the final image, and no package installer either: pip, setuptools and wheel
  are uninstalled from the interpreter after the install.
  [`requirements/README.md`](../requirements/README.md) explains how the lock is refreshed;
  CI regenerates it and fails if the committed file is stale.
- Application data under `/app` (`SOH_PROJECT_ROOT=/app`): `artifacts/models/` (required
  for readiness), `artifacts/demo/`, `reports/benchmark/`, `reports/audit/`, `frontend/` and
  `configs/`. Every file is owned by root and read-only to the process.
- A fixed non-root identity (uid and gid 10001), so orchestrators can enforce
  `runAsNonRoot`.
- A `HEALTHCHECK` on `/readyz` (standard library only; there is no curl in the image), which
  turns healthy once every model bundle has been verified against `MANIFEST.json`.
- `uvicorn` as PID 1 (`exec`), so SIGTERM drains in-flight requests (20 s graceful timeout);
  uvicorn's own access log is off because the application writes a structured one.
  `--limit-concurrency 32` makes each worker answer `503` immediately beyond 32 concurrent
  connections or tasks, instead of queueing without bound.
- BLAS and OpenMP thread pools capped at one thread: requests are small, and concurrency
  comes from workers and replicas.

The process writes nothing to disk; `--tmpfs /tmp` is only a safety net for libraries that
create temporary files. CI (job `docker` in [`ci.yml`](../.github/workflows/ci.yml)) builds
the image, runs it with a read-only root filesystem, no capabilities, 512 MB of memory and one
CPU, waits for `/readyz`, posts the sample estimate request, checks the security headers and
the user id, waits for the health check and records the image size.

## Docker Compose

```bash
docker compose up --build -d      # build and start
docker compose ps                 # STATUS shows (healthy) once models are verified
docker compose logs -f api        # structured JSON logs
docker compose down
```

[`docker-compose.yml`](../docker-compose.yml) applies the same hardening (read-only root
filesystem, `/tmp` as a small `noexec` tmpfs, all capabilities dropped,
`no-new-privileges`, an init process), limits the container to one CPU, 512 MB and 128
processes, rotates logs, and binds the port to `127.0.0.1` because the API has no
authentication by design. No `.env` file is needed; Compose reads one only for `${VAR}`
interpolation, so unrelated variables in a local `.env` never reach the container.

## Configuration

Application settings use the `SOH_` prefix and are read from the environment only (never
from a `.env` file). The full list is in [API.md](API.md#settings); the ones that matter for
deployment:

| Variable | Default | Notes |
|---|---|---|
| `SOH_LOG_JSON` | `false` (`true` in Compose) | One JSON object per line for log shippers |
| `SOH_LOG_LEVEL` | `INFO` | |
| `SOH_CORS_ORIGINS` | empty | Only needed when another origin calls the API from a browser; the bundled UI is same-origin |
| `SOH_MAX_BODY_BYTES` | `1000000` | 413 above this, before parsing |
| `SOH_MAX_BATCH_ITEMS` | `100` | Items per `/v1/soh/batch` request (at most 1,000); 422 above this, before any estimate is computed. Each item is a full estimate, so this bounds the CPU one request can buy |
| `SOH_ENABLE_METRICS` | `true` | `/metrics`; restrict it at the proxy if it is public |
| `SOH_ENABLE_DOCS` | `true` | `/docs` and `/redoc` load Swagger UI and ReDoc from jsDelivr; disable for a fully offline deployment |
| `PORT` | `8000` | Port inside the container (PaaS platforms set it) |
| `WEB_CONCURRENCY` | `1` | uvicorn workers; prefer one per container and scale with replicas |
| `LOG_LEVEL` | `info` | uvicorn's log level (lower case) |
| `FORWARDED_ALLOW_IPS` | `127.0.0.1` (uvicorn) | Addresses whose `X-Forwarded-*` headers are trusted; set it to your proxy |

Each worker loads its own NumPy and pandas stack and its own copy of the models, and keeps
its own Prometheus registry, so with several workers `/metrics` describes one worker per
scrape. Raise the memory limit together with `WEB_CONCURRENCY`.

## Updating models and results

All served data is committed and baked into the image. Regenerate it in an environment
installed with `-c requirements/constraints.txt` (the versions that produced the committed
results; see [`requirements/README.md`](../requirements/README.md)):

```bash
soh benchmark      # reports/benchmark/   (the training step copies its metrics)
soh train          # artifacts/models/: <model_id>.json, model_cards/, MANIFEST.json
soh report         # figures and docs/RESULTS.md
docker compose up --build -d
```

`soh train` refuses to write bundles whose leave-one-battery-out refits do not reproduce the
benchmark, and rewrites `MANIFEST.json`. Editing a bundle by hand without updating the
manifest makes the registry refuse every model, and `/readyz` stays 503: that is the intended
fail-closed behaviour.

## GitHub Pages static demo

[`.github/workflows/pages.yml`](../.github/workflows/pages.yml) runs when the CI workflow
completes for a push to `main` and deploys only if CI succeeded, building exactly the commit
CI tested (a manual run deploys the selected ref as it is). It installs the package with the
`[api]` extra under `requirements/constraints.txt`, runs
`soh export-static --out-dir site`, checks that `site/index.html` exists and deploys the
folder with the official Pages actions.

One-time setup after pushing the repository:

1. **Settings -> Pages -> Build and deployment -> Source: GitHub Actions.**
2. Run the *Pages* workflow (push to `main`, or *Actions -> Pages -> Run workflow*).
3. The site is served at `https://OWNER.github.io/REPO/`. Add the live-demo badge to the
   README (the commented placeholder at the top shows where).

The static build switches the UI to static mode (`assets/config.js`) and serves precomputed
JSON for models, demo cycles and the benchmark. Estimates from indicator values are computed
in the browser from the exported bundles; raw-array extraction needs the API. Preview locally:

```bash
soh export-static --out-dir site
python -m http.server 8080 --directory site    # ES modules need http://, not file://
```

The exporter only replaces an output folder that it created itself (marker file), never an
arbitrary directory.

## Releases to GitHub Container Registry

[`.github/workflows/release.yml`](../.github/workflows/release.yml) runs when a tag `vX.Y.Z`
is pushed:

```bash
# bump the version in pyproject.toml (and CHANGELOG.md) first; the workflow fails if the
# tag does not equal the pyproject.toml version
git tag v2.0.0
git push origin v2.0.0
```

It first runs the complete CI workflow on the tagged commit (`needs: ci`; lint, types,
tests including the reproduction checks, notebook, Docker smoke test); nothing is published
unless all of it passes. The publishing job then verifies that the tag equals the version in
`pyproject.toml`, builds for `linux/amd64` and `linux/arm64` from scratch (`no-cache`, so no
layer comes from a cache a pull-request build could have written), and pushes
`ghcr.io/owner/repo` (lower-cased) tagged `X.Y.Z`, `X.Y` and `sha-<short>` (and `latest`
for stable versions), with an SBOM and maximum-mode build provenance attached. Only this job
gets `packages: write`.

The publishing job runs in the GitHub environment `release`. Configure that environment to
require approval before anything is pushed, once:

1. **Settings -> Environments -> New environment**, name it `release`.
2. Enable **Required reviewers** and add the maintainers who may approve a release
   (optionally **Prevent self-review**).
3. Under **Deployment branches and tags**, allow only tags matching `v*`.

Without these rules the environment exists but does not pause the job. After the first
release, check the package's visibility in the repository's *Packages* settings, then:

```bash
docker pull ghcr.io/owner/repo:2.0.0
```

## Behind a reverse proxy

The application deliberately has no authentication and no rate limiting
([SECURITY.md](../SECURITY.md)). Before exposing it beyond localhost, put it behind a proxy
that terminates TLS, authenticates if needed, limits request rates and restricts `/metrics`.
An nginx example:

```nginx
limit_req_zone $binary_remote_addr zone=soh:10m rate=10r/s;
# One batch request buys up to SOH_MAX_BATCH_ITEMS (default 100) estimates: rate-limit it
# separately and much more strictly.
limit_req_zone $binary_remote_addr zone=soh_batch:10m rate=1r/s;

server {
    listen 443 ssl;
    server_name soh.example.org;
    ssl_certificate     /etc/ssl/certs/soh.crt;
    ssl_certificate_key /etc/ssl/private/soh.key;

    client_max_body_size 1m;            # the app enforces 1,000,000 bytes as well

    location = /metrics {               # scrape from inside the network only
        allow 10.0.0.0/8;
        deny all;
        auth_basic off;
        proxy_pass http://127.0.0.1:8000;
    }

    # Inherited by the locations below (they set no directives of these kinds themselves).
    auth_basic           "battery-soh";          # or SSO / mTLS
    auth_basic_user_file /etc/nginx/soh.htpasswd;
    proxy_set_header Host              $host;
    proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_set_header X-Request-ID      $request_id;   # correlates proxy and app logs
    proxy_read_timeout 30s;

    location = /v1/soh/batch {
        limit_req zone=soh_batch burst=2 nodelay;
        proxy_pass http://127.0.0.1:8000;
    }

    location / {
        limit_req zone=soh burst=20 nodelay;
        proxy_pass http://127.0.0.1:8000;
    }
}
```

Notes:

- The app keeps a well-formed incoming `X-Request-ID` and echoes it, so the proxy's id and
  the application's logs line up.
- Set `FORWARDED_ALLOW_IPS` to the proxy's address as seen by the container (for a proxy on
  the Docker host this is usually the bridge gateway), otherwise uvicorn ignores the
  forwarded headers.
- The app already sends a strict Content-Security-Policy and the other security headers; the
  proxy should not override them. Add `Strict-Transport-Security` at the proxy, where TLS is
  terminated.
- With authentication in front, browsers still load the UI and call the API from the same
  origin, so no CORS configuration is needed.

## Kubernetes

The image needs no special treatment. A minimal container spec:

```yaml
securityContext:
  runAsNonRoot: true
  runAsUser: 10001
  readOnlyRootFilesystem: true
  allowPrivilegeEscalation: false
  capabilities: {drop: ["ALL"]}
livenessProbe:  {httpGet: {path: /livez,  port: 8000}, periodSeconds: 10}
readinessProbe: {httpGet: {path: /readyz, port: 8000}, periodSeconds: 10}
resources:
  requests: {cpu: 100m, memory: 256Mi}
  limits:   {cpu: "1",  memory: 512Mi}
volumeMounts: [{name: tmp, mountPath: /tmp}]   # emptyDir
```

Scale with replicas rather than workers. Readiness stays false while bundles fail
verification, so a bad artifact never receives traffic.

## Hugging Face Spaces and Render

Both platforms build directly from the repository's Dockerfile. Neither adds authentication
unless you configure it; treat a deployment there as a public demo.

**Hugging Face Spaces (Docker SDK).** Create a Space with the Docker SDK and push this
repository to it. The Space's README front matter must declare the SDK and the port the app
listens on:

```yaml
---
title: Battery SOH
sdk: docker
app_port: 8000
---
```

The image already runs as a non-root user and needs no writable directory. The read-only and
capability flags are not configurable on Spaces; the image does not depend on them for
correctness.

**Render.** Create a *Web Service* from the repository with the Docker runtime. Render sets
`PORT`, which the image honours. Set the health check path to `/readyz` and
`SOH_LOG_JSON=true`. Free instances are suspended when idle, so the first request after a
pause waits for a cold start.

The static GitHub Pages build is the cheapest public demo: no server, no cold start.

## Operations

- **Probes.** `/livez` for liveness, `/readyz` for readiness; neither is logged at INFO.
- **Logs.** One structured access line per request with method, path, status, duration and
  request id; application records carry the same id. Errors return the id to the client
  (`request_id` in the problem body), so a user report can be traced to the log line.
- **Metrics.** RED metrics per route template, inference latency per model, prediction
  counts split by `in_domain` (a rising out-of-domain share is an early drift signal) and the
  number of loaded models. A dashboard needs three panels: request rate and error ratio per
  route, p50 and p99 latency, and the out-of-domain share.
- **Capacity.** An estimate is a dot product over at most eight inputs plus a domain check;
  `soh_inference_duration_seconds` separates model time from HTTP and JSON overhead, which is
  the number to watch when sizing replicas. One CPU per replica is the provisioned limit in
  Compose and CI.
