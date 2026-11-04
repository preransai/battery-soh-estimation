"""One visual style for every report figure: palette, typography, axes and deterministic PNGs.

Figures are built on :class:`matplotlib.figure.Figure` directly (no ``pyplot``
state, no GUI backend). Every figure function runs inside :func:`context`
(matplotlib reads many style settings lazily, e.g. when tick labels are
created, so the whole build must happen inside the style) and is written by
:func:`save`, which strips the volatile PNG metadata so re-rendering the same
inputs gives byte-identical files.

Palette: the categorical slots are a validated colour-blind-safe order (blue,
orange, aqua, violet; every pair stays separable under simulated deutan,
protan and tritan vision, checked in OKLab). Identity is never carried by
colour alone: every categorical series also gets its own marker shape and a
legend entry. Ink, grid and surface colours are neutral and recessive so the
data is the loudest thing in the figure. :data:`LEAKY` (a status red) marks
results that use label information and always comes with a text label.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Final, Literal

import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.layout_engine import ConstrainedLayoutEngine
from matplotlib.text import Text

__all__ = [
    "AUDIT_LABELS",
    "AXIS",
    "BATTERY_COLORS",
    "BATTERY_MARKERS",
    "CONDITION_LABELS",
    "DPI",
    "FEATURE_LABELS",
    "FEATURE_SET_LABELS",
    "GRID",
    "INK",
    "INK_MUTED",
    "INK_SECONDARY",
    "LEAKY",
    "MODEL_LABELS",
    "NEGATIVE",
    "POSITIVE",
    "SERIES",
    "SERIES_MARKERS",
    "SURFACE",
    "WIDTH_IN",
    "battery_style",
    "context",
    "data_extent",
    "feature_label",
    "feature_set_label",
    "figure",
    "footnote",
    "model_label",
    "panel_title",
    "save",
    "series_style",
    "value_grid",
]

#: Output resolution; with :data:`WIDTH_IN` this gives 900 px wide figures (README width).
DPI: Final = 150
#: Default figure width in inches.
WIDTH_IN: Final = 6.0

SURFACE: Final = "#fcfcfb"
INK: Final = "#0b0b0b"
INK_SECONDARY: Final = "#52514e"
INK_MUTED: Final = "#898781"
GRID: Final = "#e1e0d9"
AXIS: Final = "#c3c2b7"

#: Categorical slots in fixed order (never cycled past the end).
SERIES: Final[tuple[str, ...]] = ("#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7")
#: Marker shape paired with each categorical slot (secondary encoding).
SERIES_MARKERS: Final[tuple[str, ...]] = ("o", "s", "^", "D")
#: Status red for leaky (label-side) results; always paired with a text label.
LEAKY: Final = "#d03b3b"
#: Diverging pair for signed quantities (coefficients).
POSITIVE: Final = "#2a78d6"
NEGATIVE: Final = "#e34948"

_CORE_ORDER: Final[tuple[str, ...]] = ("B0005", "B0006", "B0007", "B0018")
#: Fixed colour per core cell, so a cell looks the same in every figure.
BATTERY_COLORS: Final[Mapping[str, str]] = dict(zip(_CORE_ORDER, SERIES, strict=True))
BATTERY_MARKERS: Final[Mapping[str, str]] = dict(zip(_CORE_ORDER, SERIES_MARKERS, strict=True))

FEATURE_LABELS: Final[Mapping[str, str]] = {
    "chg_cc_time_390_419_s": "CC charge time 3.90-4.19 V",
    "chg_cc_time_400_419_s": "CC charge time 4.00-4.19 V",
    "chg_cv_time_s": "CV phase duration",
    "chg_ica_peak_height_ah_per_v": "ICA peak height",
    "chg_ica_peak_voltage_v": "ICA peak voltage",
    "chg_temp_rise_cc_c": "Temperature rise, CC charge",
    "dis_q_380_350_ah": "Charge delivered 3.80-3.50 V",
    "dis_time_380_350_s": "Discharge time 3.80-3.50 V",
    "dis_r0_ohm": "Ohmic resistance at load-on",
    "dis_temp_rise_win_c": "Temperature rise, discharge window",
    "eis_re_ohm": "EIS electrolyte resistance",
    "eis_rct_ohm": "EIS charge-transfer resistance",
    "ctx_cycle_index": "Cycle count",
    "ctx_ambient_c": "Ambient temperature",
    "ctx_discharge_current_a": "Discharge current",
}

MODEL_LABELS: Final[Mapping[str, str]] = {
    "ridge": "Ridge",
    "ridge_nfs": "Ridge + nested selection",
    "pls": "PLS",
    "elasticnet": "Elastic net",
    "gpr": "Gaussian process",
    "random_forest": "Random forest",
    "hgb": "Gradient boosting",
    "mlp": "MLP",
    "knn": "k-nearest neighbours",
    "mean": "Training mean",
    "cycle_only": "Cycle count only",
    "capacity_oracle": "Capacity oracle (leaky)",
}

FEATURE_SET_LABELS: Final[Mapping[str, str]] = {
    "charge": "Charge window",
    "discharge": "Discharge window",
    "combined": "Combined",
    "combined_eis": "Combined + EIS",
    "charge_no_cv": "Charge without CV time",
    "combined_no_cv": "Combined without CV time",
    "charge_ext": "Charge window (ext.)",
    "combined_ext": "Combined (ext.)",
    "baseline": "Baseline",
}

CONDITION_LABELS: Final[Mapping[str, str]] = {
    "2A_24C": "2 A, 24 \N{DEGREE SIGN}C",
    "4A_43C": "4 A, 43 \N{DEGREE SIGN}C",
    "1A_4C": "1 A, 4 \N{DEGREE SIGN}C",
}

#: Short labels of the leakage-audit rows (``audit.json`` row ids), shared by the
#: audit figure and the README table; ``docs/RESULTS.md`` keeps the full labels.
#: ``v2_relative_mode`` is the config's first relative-mode prediction spec, plain
#: Ridge (not the headline's nested selection); the audit tests pin that.
AUDIT_LABELS: Final[Mapping[str, str]] = {
    "v1_contaminated_split": "v1 as reported (test rows overlap training)",
    "v1_within_battery_holdout": "v1, honest within-battery hold-out",
    "v1_inputs_lobo": "v1 inputs, leave-one-battery-out",
    "v1_inputs_without_capacity_lobo": "v1 inputs minus capacity, LOBO",
    "capacity_only_lobo": "Capacity alone (oracle), LOBO",
    "v2_headline": "v2 pre-specified headline, LOBO",
    "v2_same_set_plain_ridge": "v2 same inputs, plain Ridge, LOBO",
    "v2_discharge_ridge": "v2 discharge window only, LOBO",
    "v2_relative_mode": "v2 relative mode, plain Ridge, LOBO",
}

# Keys typed as Any: matplotlib >= 3.11 annotates rc_context() with a Literal of known rc keys.
_RC: Final[dict[Any, Any]] = {
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.titlesize": 8.5,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK,
    "axes.titlelocation": "left",
    "axes.titlepad": 5.0,
    "axes.labelsize": 8,
    "axes.labelcolor": INK_SECONDARY,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "grid.linestyle": "-",
    "xtick.color": AXIS,
    "ytick.color": AXIS,
    "xtick.labelcolor": INK_SECONDARY,
    "ytick.labelcolor": INK_SECONDARY,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "legend.frameon": False,
    "legend.fontsize": 7.5,
    "legend.handlelength": 1.8,
    "legend.borderaxespad": 0.2,
    "legend.columnspacing": 1.4,
    "lines.linewidth": 1.5,
    "lines.markersize": 4.5,
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
    "patch.linewidth": 0.0,
    "svg.hashsalt": "battery-soh",
}


@contextmanager
def context() -> Iterator[None]:
    """Apply the report style; build and save every figure inside it."""
    with mpl.rc_context(_RC):
        yield


def figure(
    nrows: int = 1,
    ncols: int = 1,
    *,
    height_in: float,
    width_in: float = WIDTH_IN,
    sharex: bool = False,
    sharey: bool = False,
    height_ratios: Sequence[float] | None = None,
    width_ratios: Sequence[float] | None = None,
) -> tuple[Figure, list[list[Axes]]]:
    """A figure (constrained layout) with an ``nrows x ncols`` grid of axes.

    Call inside :func:`context`. Axes are returned as a nested list
    ``axes[row][col]``.
    """
    engine = ConstrainedLayoutEngine(h_pad=0.05, w_pad=0.05, hspace=0.06, wspace=0.04)
    fig = Figure(figsize=(width_in, height_in), dpi=DPI, layout=engine)
    FigureCanvasAgg(fig)  # explicit raster canvas: text can be measured, output is Agg-rendered
    gridspec_kw: dict[str, Any] = {}
    if height_ratios is not None:
        gridspec_kw["height_ratios"] = list(height_ratios)
    if width_ratios is not None:
        gridspec_kw["width_ratios"] = list(width_ratios)
    grid = fig.subplots(
        nrows, ncols, squeeze=False, sharex=sharex, sharey=sharey, gridspec_kw=gridspec_kw
    )
    return fig, [[grid[r, c] for c in range(ncols)] for r in range(nrows)]


def series_style(index: int) -> dict[str, Any]:
    """Colour and marker of categorical slot ``index`` (raises past the last slot)."""
    if not 0 <= index < len(SERIES):
        raise ValueError(f"only {len(SERIES)} categorical slots exist; facet or fold the rest")
    return {"color": SERIES[index], "marker": SERIES_MARKERS[index]}


def battery_style(battery_id: str) -> dict[str, Any]:
    """Fixed colour and marker of a core cell.

    Raises:
        ValueError: For a cell without a fixed style (facet instead of cycling colours).
    """
    if battery_id not in BATTERY_COLORS:
        raise ValueError(f"no fixed style for {battery_id}; only core cells are colour-coded")
    return {"color": BATTERY_COLORS[battery_id], "marker": BATTERY_MARKERS[battery_id]}


def feature_label(name: str) -> str:
    """Short human-readable label of a registry feature."""
    return FEATURE_LABELS.get(name, name)


def feature_set_label(name: str) -> str:
    """Human-readable label of a feature set."""
    return FEATURE_SET_LABELS.get(name, name)


def model_label(name: str) -> str:
    """Human-readable model name."""
    return MODEL_LABELS.get(name, name)


def panel_title(ax: Axes, text: str, *, subtitle: str | None = None) -> None:
    """Left-aligned bold panel title, with an optional lighter note on the right."""
    ax.set_title(text, loc="left", fontsize=8.5, fontweight="bold", color=INK)
    if subtitle:
        ax.set_title(subtitle, loc="right", fontsize=7.5, fontweight="normal", color=INK_SECONDARY)


def data_extent(ax: Axes, text: Text) -> tuple[float, float]:
    """Horizontal extent of a drawn text in ``ax``'s data coordinates (after layout)."""
    fig = ax.get_figure()
    if fig is None or not isinstance(fig.canvas, FigureCanvasAgg):
        raise ValueError("the axes must belong to a figure created by style.figure()")
    renderer = fig.canvas.get_renderer()  # type: ignore[no-untyped-call]
    box = text.get_window_extent(renderer)
    (x0, _), (x1, _) = ax.transData.inverted().transform([[box.x0, box.y0], [box.x1, box.y1]])
    return float(min(x0, x1)), float(max(x0, x1))


def value_grid(ax: Axes, axis: Literal["x", "y", "both"] = "y") -> None:
    """Hairline grid on the value axis only."""
    ax.grid(True, axis=axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def footnote(fig: Figure, text: str) -> None:
    """Small secondary-ink note below the axes (placed by the constrained layout)."""
    fig.supxlabel(text, fontsize=7, color=INK_SECONDARY, ha="left", x=0.01, wrap=True)


def save(fig: Figure, path: Path) -> Path:
    """Write ``fig`` as a deterministic PNG (no software or date metadata)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="png", dpi=DPI, facecolor=SURFACE, metadata={"Software": None})
    return path
