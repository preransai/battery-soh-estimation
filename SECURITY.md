# Security policy

## Supported versions

| Version | Supported |
|---|---|
| 2.x (this code base, `src/battery_soh`) | yes |
| 1.x (frozen under `legacy/`) | no: kept for reference only, never deployed |

## Reporting a vulnerability

Please report vulnerabilities **privately** through GitHub's
[private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability):
open the repository's **Security** tab and choose **Report a vulnerability**. Do not open
a public issue, pull request or discussion for a suspected vulnerability.

Please include the affected version or image tag (`soh info`, or the
`org.opencontainers.image.version` label), the endpoint or command involved, a minimal
reproduction, and the impact you expect. You can expect an acknowledgement within
7 days and an assessment with a remediation plan within 30 days. Fixes are released as
a new patch version and image tag, and credited in the advisory unless you prefer
otherwise.

## Threat model

battery-soh is a **public demonstration service**: a stateless, read-only API and web UI
that estimates battery state of health from numbers the caller sends. It stores no user
data, has no accounts and holds no secrets. The design goal is that an anonymous,
untrusted client can neither take the service over nor make it misbehave, and that
everything the service executes or loads is reviewable in this repository.

### Assets

- Integrity of the served estimates (model bundles, conformal quantiles, demo data).
- Availability of a small instance (CPU and memory).
- The host running the container.

### Controls

**No authentication, by design.** Every endpoint is read-only and computes over
request data plus public, versioned model files, so there is nothing to protect behind
a login. Do not add private data or write endpoints without first putting
authentication in front (for example a reverse proxy with SSO or mTLS). Docker Compose
binds the port to `127.0.0.1` for that reason.

**Strict input validation.** Request bodies are Pydantic models with `extra="forbid"`
(unknown fields are rejected), finite floats only (NaN and infinity are rejected), every
feature value bounded by the physically plausible range declared in
`battery_soh/features/registry.py`, raw time series capped at 20,000 samples per
array and batches capped at 100 items by default (`SOH_MAX_BATCH_ITEMS`, never more
than 1,000). Invalid input gets `422` with an RFC 7807 problem body; internal errors
return a generic `500` with a request id and never a stack trace.

**Request size and work limits.** Bodies larger than `SOH_MAX_BODY_BYTES` (default 1 MB)
are rejected with `413` before parsing. An oversized batch is refused before any
estimate is computed, and the container's uvicorn answers `503` beyond 32 concurrent
connections (`--limit-concurrency 32`) instead of queueing without bound.

**Browser hardening.** Every response carries security headers
(`X-Content-Type-Options`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`,
`Permissions-Policy`, `Cross-Origin-Opener-Policy` and `Cross-Origin-Resource-Policy:
same-origin`) and, on every path except the two documentation pages below, this strict
Content-Security-Policy:
`default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'`.
The web UI has no inline scripts, styles or event handlers, loads no third-party
resources at runtime (Chart.js is vendored), and writes API data into the page with
`textContent`, never `innerHTML`. Its `index.html` also declares the same policy in a
`<meta http-equiv="Content-Security-Policy">` tag (without `frame-ancestors`, which a meta
policy cannot carry), so the backend-free GitHub Pages build, where no response header
can be set, runs under it too. CORS is closed unless `SOH_CORS_ORIGINS` lists origins.

**The documentation pages are the one CSP exception.** `/docs` (Swagger UI) and
`/redoc` (ReDoc) load version-pinned bundles from jsDelivr, so they get their own policy
([`api/docs.py`](src/battery_soh/api/docs.py)): scripts and styles only from that one
package directory (`https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.17.14/` or
`https://cdn.jsdelivr.net/npm/redoc@2.1.5/`, not the whole CDN); a Subresource Integrity
hash (`sha384`) with `crossorigin="anonymous"` on every CDN tag, so a tampered file is
refused by the browser; Swagger's single inline bootstrap script allowed by its SHA-256
hash (no `'unsafe-inline'` for scripts); `'unsafe-inline'` for styles only, because both
UIs set inline styles at runtime; `worker-src 'self' blob:` for ReDoc. Everything else,
including `frame-ancestors 'none'`, is unchanged. Set `SOH_ENABLE_DOCS=false` to remove
both pages (and `/openapi.json`) for a deployment that must not reference a CDN at all.

**No pickle anywhere.** Models are exported as plain JSON bundles (coefficients, scaler
statistics, conformal quantiles, domain statistics) and evaluated with NumPy. Loading a
model never executes code, unlike pickle/joblib files.

**SHA-256-verified model artifacts.** At start-up the registry checks every bundle in
`artifacts/models/` against `MANIFEST.json` and refuses to serve a bundle whose hash
does not match; `/readyz` stays `503` until verification succeeds. The feature table
ships with a `.meta.json` sidecar holding its SHA-256, and `soh data download` verifies
the NASA archive against a pinned SHA-256 before extracting it.

**No outbound network calls.** The service makes no network requests at runtime: no
telemetry, no model downloads, no third-party APIs. The only network access in the
code base is the `soh data download` command, which a developer runs explicitly to
fetch the NASA dataset from its official source.

**Least-privilege container.** The image runs as uid 10001, owns none of its own files
(code, models and web assets belong to root and are read-only to the process), contains
no compiler, no package installer (pip is removed) and no source tree, and is built to run
with a read-only root filesystem,
all Linux capabilities dropped and `no-new-privileges` (see `docker-compose.yml`). CI
starts it that way on every change.

**Supply chain.** Every GitHub Action is pinned to a full commit SHA (with a `# vX.Y.Z`
comment), the Docker base image to the digest of its multi-arch index, and the image's
Python dependencies to exact versions with SHA-256 hashes
([`requirements/runtime.lock`](requirements/runtime.lock), installed with
`--require-hashes`; the runtime image keeps no pip). CI installs exact versions from
[`requirements/constraints.txt`](requirements/constraints.txt). Dependabot proposes weekly
updates for Python packages, the action SHAs and the base-image digest. A pre-commit hook
runs gitleaks on staged changes. CI workflows run with read-only `GITHUB_TOKEN`
permissions; only the release and Pages jobs receive the write scopes they need, and both
publish only after CI passed on the same commit: Pages deploys after a successful CI run on
`main`, and a release tag first runs the whole CI workflow, then waits in the `release`
environment (which should require a reviewer's approval) and builds the published image
without any build cache. Release images are published to GHCR with an SBOM and build
provenance attestations.

### Out of scope / known limitations

- **Volumetric denial of service.** There is no rate limiting in the application. Per-
  request work is bounded by the limits above; deploy behind a proxy or platform that
  rate-limits if the instance is publicly reachable.
- **Metrics exposure.** `/metrics` (Prometheus) is public and reveals request counts and
  latencies only. Restrict it at the proxy if that matters for your deployment.
- **Model misuse.** Estimates come with conformal intervals (whose coverage is measured on
  held-out cells, not guaranteed) and an out-of-domain flag,
  but the models are trained on four 18650 cells from one laboratory protocol. They are
  a research demonstration and must not drive safety-critical decisions.
- Vulnerabilities in the frozen v1 code under `legacy/` (never packaged or deployed).
