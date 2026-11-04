"""Command-line interface: ``soh <command>``.

The CLI is a thin shell. Every command delegates to one public function in the
library, so everything it does is also callable (and tested) from Python.
Heavy imports happen inside the commands to keep ``soh --help`` instant.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from battery_soh import __version__
from battery_soh.config import get_paths
from battery_soh.logs import configure_logging

app = typer.Typer(
    name="soh",
    help="Leakage-free Li-ion battery state-of-health estimation.",
    no_args_is_help=True,
    add_completion=False,
)
data_app = typer.Typer(help="Download raw data and build the feature table.", no_args_is_help=True)
app.add_typer(data_app, name="data")

VerboseOpt = Annotated[bool, typer.Option("--verbose", "-v", help="Enable debug logging.")]


def _setup(verbose: bool) -> None:
    configure_logging("DEBUG" if verbose else "INFO")


@app.callback()
def _main() -> None:
    """Leakage-free Li-ion battery state-of-health estimation."""


@app.command()
def info() -> None:
    """Show the package version and where data, artifacts and reports live."""
    paths = get_paths()
    rows = {
        "version": __version__,
        "project_root": paths.root,
        "nasa_raw": paths.nasa_raw,
        "features_csv": paths.features_csv,
        "models_dir": paths.models_dir,
        "reports": paths.reports,
    }
    for key, value in rows.items():
        exists = (
            "" if key == "version" else ("  [ok]" if Path(str(value)).exists() else "  [missing]")
        )
        typer.echo(f"{key:14s} {value}{exists}")


@data_app.command("download")
def data_download(
    dest: Annotated[
        Path | None, typer.Option(help="Target directory (default: data/raw/nasa).")
    ] = None,
    force: Annotated[bool, typer.Option(help="Re-download even if files exist.")] = False,
    verbose: VerboseOpt = False,
) -> None:
    """Download the NASA PCoE battery dataset from its official source and verify it."""
    _setup(verbose)
    from battery_soh.data.download import download_nasa

    out = download_nasa(dest or get_paths().nasa_raw, force=force)
    typer.echo(f"NASA data ready in {out}")


@data_app.command("build")
def data_build(
    raw_dir: Annotated[Path | None, typer.Option(help="Directory with NASA .mat files.")] = None,
    out: Annotated[
        Path | None, typer.Option(help="Output CSV (default: data/processed/nasa_features.csv).")
    ] = None,
    verbose: VerboseOpt = False,
) -> None:
    """Extract one row of partial-window health indicators per discharge cycle."""
    _setup(verbose)
    from battery_soh.features.build import build_feature_table, write_feature_table

    paths = get_paths()
    table = build_feature_table(raw_dir or paths.nasa_raw)
    target = write_feature_table(table, out or paths.features_csv)
    typer.echo(f"Wrote {len(table)} cycles x {table.shape[1]} columns to {target}")


@app.command()
def audit(
    features: Annotated[Path | None, typer.Option(help="Feature table CSV.")] = None,
    out_dir: Annotated[
        Path | None, typer.Option(help="Output directory (default: reports/audit).")
    ] = None,
    verbose: VerboseOpt = False,
) -> None:
    """Reproduce the v1 leakage findings (before/after numbers for docs/LEAKAGE_AUDIT.md)."""
    _setup(verbose)
    from battery_soh.evaluation.audit import run_audit

    paths = get_paths()
    target = out_dir or paths.reports / "audit"
    summary = run_audit(features or paths.features_csv, target)
    for row in summary.get("rows", []):
        mae = row.get("mae")
        number = isinstance(mae, int | float) and not isinstance(mae, bool)
        mae_text = f"{mae:7.3f}" if number else f"{'n/a':>7s}"  # null in audit.json
        typer.echo(f"{mae_text}  {row.get('category') or '':12s}  {row.get('label') or ''}")
    typer.echo(f"Audit written to {target}")


@app.command()
def benchmark(
    config: Annotated[Path | None, typer.Option(help="Benchmark YAML config.")] = None,
    out_dir: Annotated[
        Path | None, typer.Option(help="Output directory (default: reports/benchmark).")
    ] = None,
    quick: Annotated[bool, typer.Option(help="Small smoke-test subset (used in CI).")] = False,
    verbose: VerboseOpt = False,
) -> None:
    """Run the leave-one-battery-out benchmark and write JSON/CSV results."""
    _setup(verbose)
    from battery_soh.evaluation.benchmark import load_config, run_benchmark

    paths = get_paths()
    config_path = config or paths.configs / "benchmark.yaml"
    target = out_dir or paths.benchmark_dir
    summary = run_benchmark(config_path, target, quick=quick)
    headline = summary.get("headline", {})
    keys = ("feature_set", "mode", "model", "n_batteries", "mae", "rmse", "r2", "coverage_90")
    typer.echo(json.dumps({k: headline.get(k) for k in keys}, indent=2, default=str))
    # Quick mode writes to a subfolder so it never overwrites the committed full results.
    written = target / load_config(config_path).quick_subdir if quick else target
    typer.echo(f"{'Quick' if quick else 'Full'} results written to {written}")


@app.command()
def train(
    config: Annotated[Path | None, typer.Option(help="Benchmark/training YAML config.")] = None,
    out_dir: Annotated[
        Path | None, typer.Option(help="Output directory (default: artifacts/models).")
    ] = None,
    verbose: VerboseOpt = False,
) -> None:
    """Fit the production model bundles (auditable JSON, no pickles) with model cards."""
    _setup(verbose)
    from battery_soh.training import train_bundles

    paths = get_paths()
    written = train_bundles(config or paths.configs / "benchmark.yaml", out_dir or paths.models_dir)
    for path in written:
        typer.echo(f"wrote {path}")


@app.command()
def report(verbose: VerboseOpt = False) -> None:
    """Render figures and the generated results sections of the docs from the JSON results."""
    _setup(verbose)
    from battery_soh.reporting.render import render_all

    render_all(get_paths())
    typer.echo("Reports rendered.")


@app.command("demo-assets")
def demo_assets(
    raw_dir: Annotated[Path | None, typer.Option(help="Directory with NASA .mat files.")] = None,
    verbose: VerboseOpt = False,
) -> None:
    """Build the small bundled demo dataset (downsampled raw cycles) used by the web UI."""
    _setup(verbose)
    from battery_soh.api.demo import build_demo_assets

    paths = get_paths()
    out = build_demo_assets(raw_dir or paths.nasa_raw, paths.demo_dir)
    typer.echo(f"Demo assets written to {out}")


@app.command("export-static")
def export_static(
    out_dir: Annotated[Path, typer.Option(help="Output directory for the static site.")] = Path(
        "site"
    ),
    verbose: VerboseOpt = False,
) -> None:
    """Export the web UI plus precomputed JSON as a backend-free static site (GitHub Pages)."""
    _setup(verbose)
    from battery_soh.api.static_export import export_static_site

    out = export_static_site(out_dir, get_paths())
    typer.echo(f"Static site written to {out}")


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Bind address.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port.")] = 8000,
    reload: Annotated[bool, typer.Option(help="Auto-reload on code changes (dev only).")] = False,
) -> None:
    """Run the API and web UI with uvicorn."""
    import uvicorn

    # Match the container: no `server:` banner, and the app's own structured access log
    # (request id, latency) replaces uvicorn's.
    uvicorn.run(
        "battery_soh.api.app:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
        server_header=False,
        access_log=False,
    )


if __name__ == "__main__":  # pragma: no cover
    app()
