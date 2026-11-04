# Developer and CI shortcuts for POSIX shells (Linux, macOS, WSL, Git Bash).
# Every target is a thin wrapper around the `soh` CLI or a standard tool; on Windows
# without make, run the same commands directly (see `make help` and CONTRIBUTING.md).

SHELL := /bin/sh
.DEFAULT_GOAL := help

VENV    ?= .venv
# Use the project virtualenv when it exists (POSIX or Windows layout), otherwise
# whatever python3 is on PATH.
PYTHON  ?= $(or $(wildcard $(VENV)/bin/python),$(wildcard $(VENV)/Scripts/python.exe),python3)
SOH     ?= $(PYTHON) -m battery_soh.cli
PYTEST  ?= $(PYTHON) -m pytest

SRC_DIRS     := src tests
HOST         ?= 127.0.0.1
PORT         ?= 8000
SITE_DIR     ?= site
BENCH_FLAGS  ?=
IMAGE        ?= battery-soh:local
VERSION      := $(shell sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml)
VCS_REF      := $(shell git rev-parse --short HEAD 2>/dev/null || echo unknown)

# scikit-learn OpenMP pools can oversubscribe (or even hang) small machines; one thread
# per process avoids that and keeps benchmark runtimes predictable.
export OMP_NUM_THREADS ?= 1
export PYTHONIOENCODING ?= utf-8

.PHONY: help install lint format typecheck test test-fast coverage data audit benchmark \
        train report demo-assets export-static reproduce serve docker-build docker-run clean

help: ## Show this help
	@echo "battery-soh $(VERSION): make targets (python: $(PYTHON))"
	@echo
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z_-]+:.*## / {printf "  %-14s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo
	@echo "Windows users: run the soh CLI directly, e.g. 'soh benchmark --quick' or"
	@echo "'python -m pytest -m \"not slow and not raw_data\"' (see CONTRIBUTING.md)."

install: ## Install the package in editable mode with dev extras and git hooks
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"
	$(PYTHON) -m pre_commit install

lint: ## Lint and check formatting (ruff), no changes made
	$(PYTHON) -m ruff check $(SRC_DIRS)
	$(PYTHON) -m ruff format --check $(SRC_DIRS)

format: ## Auto-fix lint findings and format the code (ruff)
	$(PYTHON) -m ruff check --fix $(SRC_DIRS)
	$(PYTHON) -m ruff format $(SRC_DIRS)

typecheck: ## Static type check (mypy --strict on src/battery_soh)
	$(PYTHON) -m mypy src/battery_soh

test: ## Run the full test suite (raw-data tests skip when data/raw/nasa is empty)
	$(PYTEST)

test-fast: ## Run the fast unit suite (no slow tests, no raw .mat files needed)
	$(PYTEST) -m "not slow and not raw_data"

coverage: ## Run tests with branch coverage (terminal report + coverage.xml + htmlcov/)
	$(PYTEST) -m "not raw_data" --cov=battery_soh --cov-report=term-missing \
		--cov-report=xml --cov-report=html

data: ## Download the NASA dataset (~200 MB, SHA-256 verified) and build the feature table
	$(SOH) data download
	$(SOH) data build

audit: ## Reproduce the v1 leakage audit -> reports/audit/
	$(SOH) audit

benchmark: ## Leave-one-battery-out benchmark -> reports/benchmark/ (BENCH_FLAGS=--quick for a smoke run)
	$(SOH) benchmark $(BENCH_FLAGS)

train: ## Fit the production JSON model bundles + model cards -> artifacts/models/
	$(SOH) train

report: ## Render figures and generated docs sections from the JSON results
	$(SOH) report

demo-assets: ## Build the downsampled demo cycles for the web UI -> artifacts/demo/
	$(SOH) demo-assets

export-static: ## Export the backend-free static site for GitHub Pages -> site/ (SITE_DIR=...)
	$(SOH) export-static --out-dir $(SITE_DIR)

reproduce: ## Rebuild everything from raw data: data, audit, benchmark, train, demo-assets, report
	$(MAKE) data
	$(MAKE) audit
	$(MAKE) benchmark
	$(MAKE) train
	$(MAKE) demo-assets
	$(MAKE) report

serve: ## Run the API + web UI locally with auto-reload on http://127.0.0.1:8000
	$(SOH) serve --host $(HOST) --port $(PORT) --reload

docker-build: ## Build the production image (IMAGE=battery-soh:local)
	docker build --tag $(IMAGE) --build-arg VERSION=$(VERSION) --build-arg VCS_REF=$(VCS_REF) .

docker-run: ## Run the image hardened: read-only root FS, no capabilities, loopback only
	docker run --rm --name battery-soh -p 127.0.0.1:$(PORT):8000 \
		--read-only --tmpfs /tmp:rw,noexec,nosuid,size=64m \
		--cap-drop ALL --security-opt no-new-privileges:true $(IMAGE)

clean: ## Remove caches, coverage output and build artefacts (never data, models or reports)
	rm -rf .pytest_cache .mypy_cache .ruff_cache .hypothesis htmlcov coverage.xml .coverage \
		.coverage.* build dist $(SITE_DIR)
	find src tests -type d -name __pycache__ -prune -exec rm -rf {} +
