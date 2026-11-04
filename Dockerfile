# syntax=docker/dockerfile:1
#
# battery-soh: FastAPI service + static web UI for leakage-free SOH estimation.
#
#   docker build -t battery-soh .
#   docker run --rm -p 127.0.0.1:8000:8000 --read-only --tmpfs /tmp \
#       --cap-drop ALL --security-opt no-new-privileges battery-soh
#   open http://127.0.0.1:8000        (OpenAPI docs at /docs)
#
# Requires BuildKit (the default builder since Docker 23; `docker buildx build`).
#
# Stages
#   base     python:3.11-slim pinned by the digest of its multi-arch index (Dependabot
#            proposes digest refreshes, i.e. Debian and CPython patches), shared by every
#            stage so the virtualenv built in `deps` points at the exact interpreter that
#            later runs it.
#   deps     third-party runtime dependencies (core + [api] extra) in /opt/venv from
#            requirements/runtime.lock: exact versions with SHA-256 hashes
#            (`--require-hashes`), binary wheels only (no compiler). Rebuilt only when the
#            lock changes; requirements/README.md explains how to refresh it.
#   builder  the battery-soh wheel (PEP 517, hatchling). Rebuilt on source edits.
#   runtime  venv + wheel + read-only application data. Non-root, no compiler, no pip
#            (removed from the interpreter after the install), no source tree, no tests,
#            no torch.
#
# The wheel is installed in its own small layer on top of the dependency layer, so a
# code change ships as a layer of a few hundred KB instead of a new virtualenv.
#
# Application data copied into /app (SOH_PROJECT_ROOT, so battery_soh.config.get_paths()
# resolves every path below without further configuration):
#   artifacts/models/   REQUIRED for readiness: <model_id>.json bundles + MANIFEST.json
#                       (`soh train`). Without them the container starts but /readyz
#                       answers 503 and the HEALTHCHECK reports "unhealthy".
#   artifacts/demo/     optional: demo_cycles.json.gz (`soh demo-assets`) for the demo tab.
#   reports/benchmark/  optional: summary.json + predictions.csv (`soh benchmark`) for
#                       GET /v1/benchmark and the benchmark tab.
#   reports/audit/      optional: audit.json (`soh audit`), merged into GET /v1/benchmark
#                       for the leakage-audit tab.
#   frontend/           optional: static web UI mounted at / (API-only without it).
#   configs/            benchmark/training configs, shipped for provenance.
# All of them are committed to git. Each optional COPY uses a one-character bracket
# wildcard (e.g. "fronten[d]"): BuildKit treats a wildcard that matches nothing as an
# empty copy instead of an error, so the image still builds from a checkout in which a
# directory is missing (git does not track empty directories).
#
# CI (.github/workflows/ci.yml, job `docker`) builds this file, runs it with a read-only
# root filesystem, waits for /readyz, POSTs a sample request and reports the image size.

# python:3.11-slim (3.11.17-slim-trixie), multi-arch index digest resolved 2026-10-14.
FROM python:3.11-slim@sha256:e88e9763f943ec1834f992a4b51e0f24500486803e8bc534e5767af9ea65f6ce AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore

# ------------------------------------------------------------------------------ deps
FROM base AS deps
WORKDIR /build
# The venv has no pip of its own: the base interpreter's pip manages it through
# `pip --python`, so neither pip nor setuptools are added to /opt/venv.
RUN python -m venv --without-pip /opt/venv
COPY requirements/runtime.lock ./
# Install exactly the locked runtime requirements ([project].dependencies + the [api]
# extra; not dev/report, hence no matplotlib and no test tooling): every file is checked
# against its recorded SHA-256 (hash-checking mode also refuses any dependency the lock
# does not list), binary wheels only. Then delete the test suites that numpy, scipy,
# pandas and scikit-learn ship inside their wheels (never imported at runtime).
# pip byte-compiles everything it installs, so the read-only container never needs to
# write .pyc files.
RUN python -m pip --python /opt/venv/bin/python install --require-hashes --only-binary=:all: --requirement runtime.lock \
 && python -m pip --python /opt/venv/bin/python check \
 && for pkg in numpy scipy pandas sklearn; do \
        find /opt/venv/lib/python3.11/site-packages/"$pkg" -depth -type d -name tests -exec rm -rf {} +; \
    done

# --------------------------------------------------------------------------- builder
FROM base AS builder
WORKDIR /build
# README.md and LICENSE are part of the wheel metadata (readme / license-files).
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m pip wheel --no-deps --wheel-dir /dist .

# --------------------------------------------------------------------------- runtime
FROM base AS runtime

# Set by `make docker-build`, CI and the release workflow (version from pyproject.toml).
ARG VERSION=dev
ARG VCS_REF=unknown
ARG SOURCE_URL=""

LABEL org.opencontainers.image.title="battery-soh" \
      org.opencontainers.image.description="Leakage-free Li-ion battery state-of-health estimation from partial charge/discharge windows, with battery-level conformal intervals." \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.authors="Preran Sai Yarramasu" \
      org.opencontainers.image.source="${SOURCE_URL}" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}"

# SOH_PROJECT_ROOT pins every path (models, demo data, benchmark results, web UI) to
# /app regardless of the working directory. BLAS/OpenMP pools are capped at one thread:
# requests are tiny, concurrency comes from uvicorn workers, and unbounded pools
# oversubscribe CPU-limited containers. PORT / WEB_CONCURRENCY / LOG_LEVEL follow the
# usual PaaS conventions and are read by CMD below.
ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    SOH_PROJECT_ROOT=/app \
    PYTHONFAULTHANDLER=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PORT=8000 \
    WEB_CONCURRENCY=1 \
    LOG_LEVEL=info

# Unprivileged runtime identity with a fixed numeric UID/GID, so orchestrators can
# verify non-root (Kubernetes `runAsNonRoot` needs a numeric USER).
RUN groupadd --gid 10001 app \
 && useradd --uid 10001 --gid 10001 --no-create-home --home-dir /nonexistent \
        --shell /usr/sbin/nologin app

# Everything below is owned by root and only readable by uid 10001: the server cannot
# modify its own code, model bundles or web assets even when the container is started
# without --read-only. Ownership is set at COPY time (--chown), so there is no
# `chown -R` layer duplicating the files.
COPY --from=deps --chown=root:root /opt/venv /opt/venv
# The wheel is bind-mounted from the builder stage, so it never lands in a layer. pip is
# called through the base interpreter by absolute path: `python` on PATH is now the venv,
# which deliberately has no pip. `--no-deps`: every dependency is already installed from
# the hashed lock. Afterwards pip, setuptools and wheel are removed from the interpreter
# itself, so the running image has no package installer at all.
RUN --mount=type=bind,from=builder,source=/dist,target=/tmp/dist \
    /usr/local/bin/python -m pip --python /opt/venv/bin/python install --no-deps --no-index /tmp/dist/*.whl \
 && /usr/local/bin/python -m pip --python /opt/venv/bin/python check \
 && /usr/local/bin/python -m pip uninstall --yes pip setuptools wheel \
 && rm -f /usr/local/bin/pip /usr/local/bin/pip3 /usr/local/bin/pip3.11

WORKDIR /app
COPY --chown=root:root LICENSE ./
COPY --chown=root:root config[s] ./configs/
COPY --chown=root:root fronten[d] ./frontend/
COPY --chown=root:root artifact[s] ./artifacts/
COPY --chown=root:root reports/benchmar[k] ./reports/benchmark/
COPY --chown=root:root reports/audi[t] ./reports/audit/

USER 10001:10001
EXPOSE 8000

# Readiness, not liveness: /readyz is 200 only once the model registry has loaded and
# verified every bundle against MANIFEST.json. Uses the stdlib only (no curl in the
# image); a non-2xx status or a connection error raises -> exit 1 -> unhealthy.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request as r; r.urlopen('http://127.0.0.1:%s/readyz' % (os.environ.get('PORT') or '8000'), timeout=4)"]

# `exec` replaces the shell, so uvicorn is PID 1 and receives SIGTERM directly and drains
# in-flight requests. Access logging is done by the app's middleware (structured, with
# X-Request-ID), so uvicorn's own access log is disabled to avoid duplicates.
# --proxy-headers trusts X-Forwarded-* only from FORWARDED_ALLOW_IPS (uvicorn default
# 127.0.0.1); set that variable to your reverse proxy's address when deploying behind one.
# --limit-concurrency 32: beyond 32 concurrent connections or tasks per worker uvicorn
# answers 503 at once instead of queueing without bound (overload sheds load early).
# The process writes nothing to disk, so it runs with a read-only root filesystem
# (/tmp as tmpfs is only a safety net for libraries that create temporary files).
CMD ["sh", "-c", "exec uvicorn --factory battery_soh.api.app:create_app --host 0.0.0.0 --port \"${PORT:-8000}\" --workers \"${WEB_CONCURRENCY:-1}\" --log-level \"${LOG_LEVEL:-info}\" --proxy-headers --no-server-header --no-access-log --timeout-graceful-shutdown 20 --limit-concurrency 32"]
