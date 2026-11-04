"""Benchmark internals that the integration runs cannot reach cheaply.

* the package's lazily loaded public API;
* the window-sweep consistency guard (raw data needed to reach it otherwise);
* ConvergenceWarnings are counted per fit, not once per code location;
* the configured confidence levels must include the reported 90 % level.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.exceptions import ConvergenceWarning

import battery_soh.evaluation.benchmark as benchmark
from battery_soh.config import get_paths
from battery_soh.evaluation.benchmark import fitting, runner
from battery_soh.evaluation.benchmark.config import (
    REPORTED_CONFIDENCE,
    load_config,
    percent_label,
)
from battery_soh.evaluation.benchmark.window_sweep import (
    _check_reproduces,
    window_sweep_section,
)

# --------------------------------------------------------------------------- package API


def test_every_public_name_resolves_to_its_submodule() -> None:
    for name in benchmark.__all__:
        assert getattr(benchmark, name) is not None, name
    assert benchmark.run_benchmark is runner.run_benchmark
    assert benchmark.evaluate_specs is runner.evaluate_specs
    assert set(benchmark.__all__) <= set(dir(benchmark))
    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        _ = benchmark.nope


def test_percent_labels() -> None:
    assert percent_label(REPORTED_CONFIDENCE) == "90"
    assert [percent_label(c) for c in (0.8, 0.95, 0.999)] == ["80", "95", "100"]


def test_confidence_levels_must_include_the_reported_level(tmp_path: Path) -> None:
    raw = yaml.safe_load((get_paths().configs / "benchmark.yaml").read_text(encoding="utf-8"))
    raw["confidence_levels"] = [0.8, 0.95]
    path = tmp_path / "bench.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match=r"must include 0\.9"):
        load_config(path)


# --------------------------------------------------------------------------- window sweep guard

NAN = np.nan


def test_window_sweep_guard_passes_a_faithful_re_extraction() -> None:
    swept = np.array([1.0, 2.0, NAN, 4.0])
    table = np.array([1.0, 2.0 * (1 + 1e-7), NAN, 4.0])
    assert _check_reproduces(swept, table, "x") == pytest.approx(1e-7, rel=1e-3)
    # Identical values count as no difference, zeros included.
    assert _check_reproduces(np.array([0.0, 1.0]), np.array([0.0, 1.0]), "x") == 0.0


@pytest.mark.parametrize(
    ("swept", "table", "match"),
    [
        ([NAN, NAN], [NAN, NAN], "no cycle where it is measured"),  # was np.max on empty
        ([1.0, NAN], [1.0, 2.0], "measurable on other cycles"),
        ([1.0, 1.0], [0.0, 1.0], "max relative difference inf"),  # was None -> passed
        ([1.0, 2.0], [1.0, 2.1], "max relative difference"),
    ],
    ids=["nothing-measured", "other-cycles", "zero-reference", "too-far"],
)
def test_window_sweep_guard_fails_when_it_cannot_confirm(
    swept: list[float], table: list[float], match: str
) -> None:
    with pytest.raises(RuntimeError, match=match):
        _check_reproduces(np.array(swept), np.array(table), "x")


def _sweep_config(tmp_path: Path, **window_sweep: Any) -> Any:
    raw = yaml.safe_load((get_paths().configs / "benchmark.yaml").read_text(encoding="utf-8"))
    raw["raw_dir"] = "data/raw/__absent_for_tests__"
    raw["window_sweep"].update(window_sweep)
    path = tmp_path / "bench.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return load_config(path)


def test_missing_raw_data_is_reported_with_a_repo_relative_path(tmp_path: Path) -> None:
    # Another window grid: neither a stored sweep in the output folder nor the committed one fits.
    cfg = _sweep_config(tmp_path, discharge={"v_high": [3.8], "v_low": [3.5]})
    out = tmp_path / "out"
    out.mkdir()
    stale = '{"status": "ok", "batteries": ["B0005"], "windows": []}'
    (out / "window_sweep.json").write_text(stale, encoding="utf-8")
    block = window_sweep_section(cfg, pd.DataFrame(), out)
    assert block == {
        "status": "skipped",
        "reason": "raw .mat files not found under data/raw/__absent_for_tests__",
    }
    assert (out / "window_sweep.json").read_text(encoding="utf-8") == stale  # left alone


def test_a_stored_sweep_of_the_same_grid_is_reused(tmp_path: Path) -> None:
    committed = get_paths().benchmark_dir / "window_sweep.json"
    if not committed.is_file():
        pytest.skip("no committed window sweep")
    cfg = _sweep_config(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / "window_sweep.json").write_bytes(committed.read_bytes())
    block = window_sweep_section(cfg, pd.DataFrame(), out)
    assert block["status"] == "ok"
    assert block["source"].startswith("cached (raw data unavailable): ")
    assert not Path(block["source"].split(": ", 1)[1]).is_absolute()


# --------------------------------------------------------------------------- convergence warnings


class _NonConverging:
    """Regressor whose every fit warns from the same code location, like sklearn's solvers."""

    def fit(self, X: Any, y: Any) -> _NonConverging:
        del X
        warnings.warn("did not converge", ConvergenceWarning, stacklevel=1)
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X: Any) -> np.ndarray[Any, np.dtype[np.float64]]:
        return np.full(len(X), self.mean_)


def _task(*, intervals: bool) -> Any:
    X = pd.DataFrame({"x": np.arange(30.0)})
    return fitting._Task(
        spec_index=0,
        fold="d",
        model="non_converging",
        kind="model",
        seed=0,
        nested=((1.0,), 1, 1e-9),
        X_train=X,
        y_train=np.arange(30.0),
        cal_groups=np.array(["a"] * 10 + ["b"] * 10 + ["c"] * 10, dtype=object),
        X_test=X.iloc[:5],
        test_positions=np.arange(5, dtype=np.intp),
        intervals=intervals,
        alphas=(0.1,),
        cost=1.0,
    )


@pytest.mark.parametrize(("intervals", "fits"), [(False, 1), (True, 4)])
def test_every_non_converged_fit_is_counted(
    monkeypatch: pytest.MonkeyPatch, intervals: bool, fits: int
) -> None:
    monkeypatch.setattr(fitting, "_factory", lambda *args: _NonConverging)
    # With intervals: three calibration folds plus the final model, all warning from one line.
    result = fitting._run_task(_task(intervals=intervals))
    assert result.convergence_warnings == fits
