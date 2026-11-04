"""Serving needs NumPy only: neither the bundle runtime nor the app ever imports scikit-learn.

One fresh interpreter checks both, in import order: first the bundle runtime on its own
(load, predict with an imputed input, attribute, interval, domain check), then the app
while it answers requests. Each stage records the scikit-learn modules loaded so far.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = r"""
import json, sys
from pathlib import Path

def sklearn_modules():
    return sorted(m for m in sys.modules if m.split(".")[0] == "sklearn")

models_dir, features = Path(sys.argv[1]), json.loads(sys.argv[2])
stages = {}

from battery_soh.models.linear_bundle import LinearBundle
bundle = LinearBundle.from_json(models_dir / "combined-absolute-ridge.json")
x = {name: features[name] for name in bundle.features[1:]}  # the first input is imputed
assert bundle.missing_features(x) == bundle.features[:1]
prediction = bundle.predict(x)
bundle.contributions(x)
bundle.interval(prediction, 0.9)
bundle.domain_check(x)
stages["bundle_runtime"] = sklearn_modules()

from fastapi.testclient import TestClient
from battery_soh.api.app import create_app
from battery_soh.api.settings import Settings
stages["app_import"] = sklearn_modules()

app = create_app(Settings(models_dir=models_dir, log_level="WARNING"))
with TestClient(app) as client:
    assert client.get("/readyz").status_code == 200
    assert client.post("/v1/soh/estimate", json={"features": features}).status_code == 200
    assert client.get("/v1/models/combined-absolute-ridge").status_code == 200
stages["serving"] = sklearn_modules()
print(json.dumps(stages))
"""


@pytest.fixture(scope="session")
def sklearn_modules_by_stage(
    models_dir: Path, sample_features: dict[str, float]
) -> dict[str, list[str]]:
    """``{stage: scikit-learn modules imported so far}`` from one subprocess run."""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "OMP_NUM_THREADS": "1"}
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT, str(models_dir), json.dumps(sample_features)],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    stages: dict[str, list[str]] = json.loads(result.stdout.strip().splitlines()[-1])
    return stages


def test_bundle_runtime_never_imports_sklearn(
    sklearn_modules_by_stage: dict[str, list[str]],
) -> None:
    assert sklearn_modules_by_stage["bundle_runtime"] == []


def test_serving_never_imports_sklearn(sklearn_modules_by_stage: dict[str, list[str]]) -> None:
    assert sklearn_modules_by_stage["app_import"] == [], "importing the app pulled in sklearn"
    assert sklearn_modules_by_stage["serving"] == []
