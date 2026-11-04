"""Report figures, each rendered from the benchmark/audit JSON (or the model bundles).

Every function takes already-loaded inputs, writes one PNG and returns its
path. Nothing here fits a model or reads files, so the figures are pure
functions of the committed results and can be regenerated (and tested) from
fixture JSON alone. Conventions shared by all figures (see :mod:`.style`):
SOH errors are in percentage points of the cell's own initial capacity,
"LOBO" is leave-one-battery-out, every axis carries its unit, and leaky or
post-hoc results are labelled as such wherever they appear.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MultipleLocator

from battery_soh.features.registry import FEATURES
from battery_soh.reporting import style

__all__ = [
    "AUDIT_ROWS",
    "coverage",
    "extended_lobo",
    "feature_contributions",
    "indicators_vs_soh",
    "leaderboard",
    "leakage_audit",
    "lobo_trajectories",
    "strongest_indicators",
    "window_sweep",
]

_PCT: Final = FuncFormatter(lambda v, _pos: f"{100 * v:.0f}%")
_MAE_LABEL: Final = "LOBO MAE (SOH percentage points)"
#: Leakage-audit rows shown, top to bottom, with their short labels.
AUDIT_ROWS: Final[tuple[tuple[str, str], ...]] = tuple(
    (row_id, style.AUDIT_LABELS[row_id])
    for row_id in (
        "v1_contaminated_split",
        "v1_within_battery_holdout",
        "v1_inputs_lobo",
        "v1_inputs_without_capacity_lobo",
        "capacity_only_lobo",
        "v2_headline",
        "v2_relative_mode",
    )
)


# =========================================================================== helpers


def _spec_text(spec: Mapping[str, Any]) -> str:
    """``feature set / mode / model``; plain ``ridge`` is named so, next to ``ridge_nfs``."""
    model = str(spec.get("model", "?"))
    name = "plain Ridge" if model == "ridge" else style.model_label(model)
    return f"{spec.get('feature_set', '?')} / {spec.get('mode', '?')} / {name}"


def _spec_rows(frame: pd.DataFrame, spec: Mapping[str, Any]) -> pd.DataFrame:
    mask = np.ones(len(frame), dtype=bool)
    for key in ("experiment", "feature_set", "mode", "model"):
        if key in frame.columns and spec.get(key) is not None:
            mask &= (frame[key] == spec[key]).to_numpy(dtype=bool)
    return frame.loc[mask]


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _text(ax: Axes, x: float, y: float, s: str, **kwargs: Any) -> None:
    params: dict[str, Any] = {"fontsize": 7, "color": style.INK_SECONDARY}
    params.update(kwargs)
    ax.text(x, y, s, **params)


def _place_reference_labels(
    ax: Axes, lines: Sequence[tuple[float, str, str]], rows: Sequence[float]
) -> int:
    """Label vertical reference lines in the headroom above the data, never overlapping.

    Call after the figure layout is final (``fig.draw_without_rendering()``).
    Labels are tried, in order of their line's position, right of the line,
    then left of it, on the first text row and then the next; the first
    placement that stays inside the axes and clears every placed label wins.

    Args:
        ax: Axes holding the reference lines.
        lines: ``(x, label, colour)`` per line.
        rows: Candidate y positions (data coordinates) of the text rows.

    Returns:
        Number of text rows used.
    """
    left, right = sorted(ax.get_xlim())
    gap = 0.012 * (right - left)
    placed: list[tuple[float, float, int]] = []
    used = 0
    for x, label, color in sorted(lines):
        accepted = False
        for r, y in enumerate(rows):
            for ha in ("left", "right"):
                anchor = x + gap if ha == "left" else x - gap
                text = ax.text(anchor, y, label, ha=ha, va="center", fontsize=6.6, color=color)
                x0, x1 = style.data_extent(ax, text)
                clear = all(x1 + gap <= a or x0 - gap >= b for a, b, rr in placed if rr == r)
                if x0 >= left and x1 <= right and clear:
                    placed.append((x0, x1, r))
                    used = max(used, r + 1)
                    accepted = True
                    break
                text.remove()
            if accepted:
                break
        if not accepted:  # nothing fits: keep it readable at the top, inside the axes
            ax.text(x + gap, rows[0], label, ha="left", va="center", fontsize=6.6, color=color)
            used = max(used, 1)
    return used


# =========================================================================== 1. trajectories


def lobo_trajectories(predictions: pd.DataFrame, summary: Mapping[str, Any], out: Path) -> Path:
    """True vs predicted SOH of every held-out core cell, headline spec with its 90% band.

    Args:
        predictions: ``predictions.csv`` (per-cycle out-of-fold predictions).
        summary: ``summary.json`` (headline spec, per-battery metrics and the
            optional relative-mode spec drawn for comparison).
        out: Output PNG path.
    """
    head = summary["headline"]
    rows = _spec_rows(predictions, head)
    if "scored" in rows.columns:
        rows = rows[rows["scored"].astype(bool)]
    if rows.empty:
        raise ValueError("predictions.csv has no rows for the headline spec")
    rel_spec = summary.get("relative_headline") or None
    rel_rows = _spec_rows(predictions, rel_spec) if rel_spec else rows.iloc[0:0]
    if "scored" in rel_rows.columns:
        rel_rows = rel_rows[rel_rows["scored"].astype(bool)]
    per = {str(r["battery_id"]): r for r in summary.get("per_battery", [])}
    rel_per = {str(r["battery_id"]): r for r in (rel_spec or {}).get("per_battery", [])}
    batteries = sorted(str(b) for b in rows["battery_id"].unique())
    ncols = 2
    nrows = math.ceil(len(batteries) / ncols)
    lo_all = np.nanmin(np.r_[rows["y"], rows["yhat"], rows["lo_90"]])
    hi_all = np.nanmax(np.r_[rows["y"], rows["yhat"], rows["hi_90"]])
    with style.context():
        fig, axes = style.figure(
            nrows, ncols, height_in=2.05 * nrows + 0.75, sharex=True, sharey=True
        )
        for i, battery in enumerate(batteries):
            ax = axes[i // ncols][i % ncols]
            d = rows[rows["battery_id"] == battery].sort_values("cycle_index")
            x = d["cycle_index"].to_numpy(dtype=float)
            ax.fill_between(
                x,
                d["lo_90"].to_numpy(float),
                d["hi_90"].to_numpy(float),
                color=style.SERIES[0],
                alpha=0.14,
                linewidth=0,
                label="90% conformal interval (headline)",
            )
            ax.plot(x, d["y"].to_numpy(float), color=style.INK, linewidth=1.2, label="True SOH")
            ax.plot(
                x,
                d["yhat"].to_numpy(float),
                color=style.SERIES[0],
                linewidth=1.5,
                label=f"Headline: {_spec_text(head)}",
            )
            note = []
            m = per.get(battery, {})
            if _finite(m.get("mae")) is not None:
                cov = _finite(m.get("coverage_90"))
                note.append(
                    f"headline MAE {float(m['mae']):.2f}"
                    + (f", 90% cov. {100 * cov:.0f}%" if cov is not None else "")
                )
            if rel_spec:
                r = rel_rows[rel_rows["battery_id"] == battery].sort_values("cycle_index")
                ax.plot(
                    r["cycle_index"].to_numpy(float),
                    r["yhat"].to_numpy(float),
                    color=style.SERIES[1],
                    linewidth=1.3,
                    label=f"Relative mode: {_spec_text(rel_spec)}",
                )
                rm = rel_per.get(battery, {})
                if _finite(rm.get("mae")) is not None:
                    note.append(f"relative MAE {float(rm['mae']):.2f}")
            _text(ax, 0.98, 0.96, "\n".join(note), transform=ax.transAxes, ha="right", va="top")
            style.panel_title(ax, f"{battery} held out")
            style.value_grid(ax)
            if i % ncols == 0:
                ax.set_ylabel("SOH (%)")
            if i // ncols == nrows - 1:
                ax.set_xlabel("Discharge cycle")
        for j in range(len(batteries), nrows * ncols):
            axes[j // ncols][j % ncols].set_visible(False)
        axes[0][0].set_ylim(5 * math.floor(lo_all / 5) - 1, 5 * math.ceil(hi_all / 5) + 1)
        handles, labels = axes[0][0].get_legend_handles_labels()
        order = [labels.index(lbl) for lbl in labels if lbl.startswith("True")]
        order += [k for k in range(len(labels)) if k not in order]
        fig.legend(
            [handles[k] for k in order],
            [labels[k] for k in order],
            loc="outside upper left",
            ncol=2,
        )
        style.footnote(
            fig,
            "Each panel is a cell the model never saw (fitting, feature selection, penalty and "
            f"calibration used the other {len(batteries) - 1})."
            + (" Relative mode: reference cycles not scored." if rel_spec else ""),
        )
        return style.save(fig, out)


# =========================================================================== 2. leakage audit


def leakage_audit(audit: Mapping[str, Any], out: Path) -> Path:
    """MAE (with R-squared) of the v1 evaluations next to the leakage-free v2 results.

    Args:
        audit: ``reports/audit/audit.json``.
        out: Output PNG path.
    """
    by_id = {str(r.get("id")): r for r in audit.get("rows", [])}
    shown = [(rid, label, by_id[rid]) for rid, label in AUDIT_ROWS if rid in by_id]
    if not shown:
        raise ValueError("audit.json has none of the expected rows")
    overlap = audit.get("contaminated_test_overlap") or {}
    with style.context():
        fig, axes = style.figure(height_in=0.36 * len(shown) + 1.3)
        ax = axes[0][0]
        y = np.arange(len(shown))
        maes = np.array([_finite(r.get("mae")) or 0.0 for _, _, r in shown])
        colors = [style.LEAKY if r.get("leaky") else style.SERIES[0] for _, _, r in shown]
        ax.barh(y, maes, height=0.5, color=colors)
        xmax = float(maes.max()) * 1.42
        for k, (rid, _, row) in enumerate(shown):
            r2 = _finite(row.get("r2"))
            label = f"{maes[k]:.2f}" + (
                f"   R\N{SUPERSCRIPT TWO} {r2:.2f}" if r2 is not None else ""
            )
            _text(ax, maes[k] + 0.02 * xmax, k, label, va="center", fontsize=7.2, color=style.INK)
            if rid == "v1_contaminated_split" and overlap.get("test_rows"):
                _text(
                    ax,
                    maes[k] + 0.02 * xmax,
                    k + 0.36,
                    f"{overlap.get('overlapping_rows')} of {overlap.get('test_rows')} "
                    "'test' rows were training rows",
                    va="center",
                    fontsize=6.5,
                    color=style.LEAKY,
                )
        ax.set_yticks(y, [label for _, label, _ in shown])
        ax.invert_yaxis()
        ax.set_xlim(0, xmax)
        ax.set_xlabel("MAE (SOH percentage points), lower is better")
        ax.tick_params(axis="y", length=0)
        style.value_grid(ax, "x")
        fig.legend(
            handles=[
                Patch(
                    color=style.LEAKY,
                    label="Leaky: inputs contain capacity or full-discharge proxies of it",
                ),
                Patch(color=style.SERIES[0], label="Leakage-free: partial-window indicators only"),
            ],
            loc="outside upper left",
            ncol=1,
        )
        style.footnote(
            fig,
            "Same Ridge for every v1 row; LOBO = leave-one-battery-out over the core cells "
            "(mean of per-cell MAE; R\N{SUPERSCRIPT TWO} pooled).",
        )
        return style.save(fig, out)


# =========================================================================== 3. leaderboard


def leaderboard(summary: Mapping[str, Any], out: Path) -> Path:
    """LOBO MAE of every model x pre-specified feature set, one panel per mode.

    Post-hoc ablation sets are left out (they are listed, labelled, in the
    results document). Baselines are vertical reference lines.
    """
    board = pd.DataFrame(summary.get("leaderboard", []))
    if board.empty:
        raise ValueError("summary.json has no leaderboard")
    head = summary["headline"]
    board = board[board["experiment"] == head["experiment"]]
    is_base = board.get("baseline", pd.Series(False, index=board.index)).astype(bool)
    post_hoc = board.get("post_hoc", pd.Series(False, index=board.index)).astype(bool)
    models_df = board[~is_base & ~post_hoc]
    preferred = ["combined", "discharge", "charge", "combined_eis"]
    present = list(dict.fromkeys(models_df["feature_set"]))
    sets = [s for s in preferred if s in present] + [s for s in present if s not in preferred]
    sets = sets[: len(style.SERIES)]
    ref = models_df[
        (models_df["feature_set"] == head["feature_set"]) & (models_df["mode"] == head["mode"])
    ].sort_values("mae")
    models = list(dict.fromkeys([*ref["model"], *sorted(models_df["model"].unique())]))
    modes = [m for m in ("absolute", "relative") if m in set(board["mode"])]
    titles = {
        "absolute": "Absolute indicators",
        "relative": "Relative to the commissioning reference",
    }
    finite_mae = pd.to_numeric(board["mae"], errors="coerce")
    xmax = float(math.ceil(float(finite_mae[finite_mae.notna()].max()) * 1.04 + 0.3))
    dodge = {fs: (i - (len(sets) - 1) / 2) * 0.17 for i, fs in enumerate(sets)}
    with style.context():
        fig, axes = style.figure(
            len(modes), 1, height_in=0.23 * len(models) * len(modes) + 1.45, sharex=True
        )
        refs_by_panel: list[list[tuple[float, str, str]]] = []
        for p, mode in enumerate(modes):
            ax = axes[p][0]
            part = models_df[models_df["mode"] == mode]
            pos = {m: i for i, m in enumerate(models)}
            for model, grp in part.groupby("model"):
                vals = grp["mae"].to_numpy(float)
                ax.hlines(
                    pos[str(model)],
                    vals.min(),
                    vals.max(),
                    color=style.GRID,
                    linewidth=1.6,
                    zorder=1,
                )
            for i, fs in enumerate(sets):
                d = part[part["feature_set"] == fs]
                st = style.series_style(i)
                ax.plot(
                    d["mae"].to_numpy(float),
                    [pos[str(m)] + dodge[fs] for m in d["model"]],
                    linestyle="none",
                    marker=st["marker"],
                    color=st["color"],
                    markersize=5,
                    markeredgecolor=style.SURFACE,
                    markeredgewidth=0.6,
                    zorder=3,
                    label=style.feature_set_label(fs),
                )
            if mode == head["mode"]:
                hit = part[
                    (part["feature_set"] == head["feature_set"]) & (part["model"] == head["model"])
                ]
                if not hit.empty:
                    ax.plot(
                        float(hit["mae"].iloc[0]),
                        pos[str(head["model"])] + dodge.get(str(head["feature_set"]), 0.0),
                        marker="o",
                        markersize=10.5,
                        markerfacecolor="none",
                        markeredgecolor=style.INK,
                        markeredgewidth=1.0,
                        linestyle="none",
                        zorder=4,
                        label="Pre-specified headline",
                    )
            refs = []
            for _, row in board[is_base & (board["mode"] == mode)].iterrows():
                mae = _finite(row["mae"])
                if mae is None or mae > xmax:
                    continue
                color = style.LEAKY if bool(row.get("leaky", False)) else style.INK_MUTED
                ax.axvline(mae, color=color, linewidth=0.9, zorder=0.5)
                refs.append((mae, f"{style.model_label(str(row['model']))} {mae:.2f}", color))
            refs_by_panel.append(refs)
            ax.set_yticks(range(len(models)), [style.model_label(m) for m in models])
            ax.set_ylim(len(models) - 0.5, -2.1)
            ax.set_xlim(0, xmax)
            ax.tick_params(axis="y", length=0)
            style.value_grid(ax, "x")
            style.panel_title(ax, titles.get(mode, mode))
        axes[-1][0].set_xlabel(_MAE_LABEL)
        axes[-1][0].xaxis.set_major_locator(MultipleLocator(1.0))
        handles, labels = axes[0][0].get_legend_handles_labels()
        handles.append(Line2D([], [], color=style.INK_MUTED, linewidth=0.9))
        labels.append("Baseline")
        handles.append(Line2D([], [], color=style.LEAKY, linewidth=0.9))
        labels.append("Leaky sanity check")
        fig.legend(handles, labels, loc="outside upper left", ncol=4)
        style.footnote(
            fig,
            f"{head.get('n_batteries', 'All')} core cells, each held out once; markers dodged "
            "vertically, the grey bar spans a model's feature sets. Post-hoc ablations are listed "
            "in docs/RESULTS.md.",
        )
        fig.draw_without_rendering()  # final layout, so label widths can be measured
        for p, refs in enumerate(refs_by_panel):
            _place_reference_labels(axes[p][0], refs, rows=(-1.0, -1.6))
        return style.save(fig, out)


# =========================================================================== 4. coverage


def coverage(summary: Mapping[str, Any], out: Path) -> Path:
    """Empirical vs nominal interval coverage on every held-out cell (cv and cv+).

    Rows: the headline spec and, when present, the relative-mode spec of the
    summary. Columns: the ``cv`` construction shipped in the bundles and group
    CV+.
    """
    blocks: list[tuple[str, Mapping[str, Any], list[Mapping[str, Any]]]] = []
    head = summary["headline"]
    if summary.get("coverage"):
        blocks.append(("Pre-specified headline", head, list(summary["coverage"])))
    rel = summary.get("relative_headline") or {}
    if rel.get("coverage"):
        blocks.append(("Relative mode", rel, list(rel["coverage"])))
    if not blocks:
        raise ValueError("summary.json has no coverage table")
    methods = (("", "cv (shipped)"), ("_cv_plus", "cv+"))
    with style.context():
        fig, axes = style.figure(
            len(blocks), 2, height_in=2.15 * len(blocks) + 0.95, sharex=True, sharey=True
        )
        for r, (name, _, table) in enumerate(blocks):
            rows = sorted(table, key=lambda t: float(t["nominal"]))
            nominal = [float(t["nominal"]) for t in rows]
            x = np.arange(len(rows), dtype=float)
            batteries = sorted({b for t in rows for b in (t.get("per_battery") or {})})
            for c, (suffix, method) in enumerate(methods):
                ax = axes[r][c]
                ax.plot(
                    x,
                    nominal,
                    color=style.INK_MUTED,
                    linewidth=1.0,
                    marker="_",
                    markersize=9,
                    label="Nominal (target)",
                    zorder=2,
                )
                for k, battery in enumerate(batteries):
                    st = style.battery_style(battery)
                    vals = [
                        _finite((t["per_battery"].get(battery) or {}).get(f"coverage{suffix}"))
                        for t in rows
                    ]
                    dodge = (k - (len(batteries) - 1) / 2) * 0.07
                    ax.plot(
                        x + dodge,
                        [np.nan if v is None else v for v in vals],
                        color=st["color"],
                        marker=st["marker"],
                        linewidth=1.0,
                        markersize=4.5,
                        markeredgecolor=style.SURFACE,
                        markeredgewidth=0.5,
                        label=battery,
                        zorder=3,
                    )
                mean_key = "empirical" if not suffix else "empirical_cv_plus"
                means = [_finite(t.get(mean_key)) for t in rows]
                ax.plot(
                    x,
                    [np.nan if v is None else v for v in means],
                    color=style.INK,
                    linewidth=2.0,
                    marker="o",
                    markersize=5,
                    label="Mean over cells",
                    zorder=4,
                )
                ax.set_xticks(x, [f"{100 * v:.0f}%" for v in nominal])
                ax.set_xlim(-0.4, len(rows) - 0.6)
                ax.set_ylim(-0.04, 1.06)
                ax.yaxis.set_major_formatter(_PCT)
                ax.yaxis.set_major_locator(MultipleLocator(0.2))
                style.value_grid(ax)
                style.panel_title(ax, f"{name}: {method}")
                if c == 0:
                    ax.set_ylabel("Empirical coverage")
                if r == len(blocks) - 1:
                    ax.set_xlabel("Nominal coverage")
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper left", ncol=6)
        notes = "; ".join(f"{name.lower()} = {_spec_text(spec)}" for name, spec, _ in blocks)
        style.footnote(
            fig,
            f"Each cell held out once; markers dodged horizontally. Specs: {notes}. "
            "cv = the fixed half-width shipped in the model bundles.",
        )
        return style.save(fig, out)


# =========================================================================== 5. window sweep


def window_sweep(sweep: Mapping[str, Any], out: Path) -> Path:
    """LOBO MAE of single-window indicators vs the share of capacity the window covers."""
    windows = list(sweep.get("windows", []))
    if not windows:
        raise ValueError("window sweep has no windows")
    modes = [m for m in ("absolute", "relative") if any(m in w for w in windows)]
    registry = {("discharge", 3.8, 3.5), ("charge", 3.9, 4.19)}

    def _series(w: Mapping[str, Any]) -> str:
        if w["kind"] == "discharge":
            return f"Discharge from {float(w['v_high']):.1f} V (labels: end voltage)"
        return f"Charge to {float(w.get('v_end', 4.19)):.2f} V (labels: start voltage)"

    def _key_tag(w: Mapping[str, Any]) -> tuple[tuple[str, float, float], str]:
        if w["kind"] == "discharge":
            high, low = float(w["v_high"]), float(w["v_low"])
            return ("discharge", round(high, 2), round(low, 2)), f"{low:.1f} V"
        start, end = float(w.get("v_start", math.nan)), float(w.get("v_end", 4.19))
        return ("charge", round(start, 2), round(end, 2)), f"{start:.1f} V"

    names = list(dict.fromkeys(_series(w) for w in windows))[: len(style.SERIES)]
    titles = {"absolute": "Absolute indicators", "relative": "Relative to the reference"}
    with style.context():
        fig, axes = style.figure(1, len(modes), height_in=3.4, sharey=True)
        for p, mode in enumerate(modes):
            ax = axes[0][p]
            curves: dict[str, tuple[list[Mapping[str, Any]], list[float], list[float]]] = {}
            for name in names:
                pts = sorted(
                    (
                        w
                        for w in windows
                        if _series(w) == name
                        and _finite((w.get(mode) or {}).get("mae")) is not None
                    ),
                    key=lambda w: float(w["capacity_fraction_mean"]),
                )
                xs = [100 * float(w["capacity_fraction_mean"]) for w in pts]
                ys = [float(w[mode]["mae"]) for w in pts]
                curves[name] = (pts, xs, ys)
            for i, name in enumerate(names):
                pts, xs, ys = curves[name]
                st = style.series_style(i)
                ax.plot(xs, ys, color=st["color"], linewidth=1.0, zorder=2)
                others = [(ox, oy) for n, (_, ox, oy) in curves.items() if n != name and ox]
                for w, xv, yv in zip(pts, xs, ys, strict=True):
                    full = float(w.get("availability", 1.0)) >= 0.95
                    ax.plot(
                        xv,
                        yv,
                        marker=st["marker"],
                        color=st["color"],
                        markersize=5.5,
                        markerfacecolor=st["color"] if full else style.SURFACE,
                        markeredgewidth=1.0,
                        linestyle="none",
                        zorder=3,
                    )
                    key, tag = _key_tag(w)
                    ring = key in registry
                    if ring:
                        ax.plot(
                            xv,
                            yv,
                            marker="o",
                            markersize=11,
                            markerfacecolor="none",
                            markeredgecolor=style.INK,
                            markeredgewidth=0.9,
                            zorder=4,
                        )
                    # Label on the side away from the nearest other curve at this x.
                    near = [
                        float(np.interp(xv, ox, oy))
                        for ox, oy in others
                        if min(ox) <= xv <= max(ox)
                    ]
                    below = bool(near) and yv < min(near, key=lambda v: abs(v - yv))
                    step = 9.0 if ring else 6.0
                    dy, va = (-step, "top") if below else (step, "bottom")
                    ax.annotate(
                        tag,
                        (xv, yv),
                        xytext=(0.0, dy),
                        textcoords="offset points",
                        ha="center",
                        va=va,
                        fontsize=6.3,
                        color=style.INK_SECONDARY,
                    )
            ax.set_xlim(0, 80)
            ax.set_ylim(bottom=0, top=ax.get_ylim()[1] * 1.08)
            ax.set_xlabel("Share of capacity covered by the window (%)")
            if p == 0:
                ax.set_ylabel(_MAE_LABEL)
            style.value_grid(ax)
            style.panel_title(ax, titles.get(mode, mode))
        handles: list[Any] = [
            Line2D([], [], color=style.SERIES[i], marker=style.SERIES_MARKERS[i], linewidth=1.0)
            for i in range(len(names))
        ]
        labels = list(names)
        handles.append(
            Line2D(
                [],
                [],
                marker="o",
                markersize=9,
                markerfacecolor="none",
                markeredgecolor=style.INK,
                linestyle="none",
            )
        )
        labels.append("Registry window (fixed before the sweep)")
        handles.append(
            Line2D(
                [],
                [],
                marker="o",
                markerfacecolor=style.SURFACE,
                markeredgecolor=style.INK_SECONDARY,
                linestyle="none",
            )
        )
        labels.append("Hollow: measurable on < 95% of cycles")
        fig.legend(handles, labels, loc="outside upper left", ncol=2)
        style.footnote(
            fig,
            f"Exploratory: Ridge, cells {', '.join(sweep.get('batteries', []))}; each window is "
            "scored on the cycles where it "
            "is measurable, so rows differ between windows.",
        )
        return style.save(fig, out)


# =========================================================================== 6. indicators


def strongest_indicators(
    table: pd.DataFrame,
    candidates: Sequence[str],
    *,
    batteries: Sequence[str],
    label: str = "soh_pct",
    n: int = 4,
) -> list[tuple[str, float]]:
    """The ``n`` candidates with the highest median within-cell |Pearson r| with the label.

    Returns:
        ``[(feature, median |r|)]``, strongest first (ties by name).
    """
    cells = table[table["battery_id"].isin(list(batteries)) & table["qc_valid"].astype(bool)]
    scored: list[tuple[str, float]] = []
    for name in candidates:
        rs = []
        for _, grp in cells.groupby("battery_id"):
            pair = grp[[name, label]].dropna()
            if len(pair) > 2 and pair[name].std() > 0 and pair[label].std() > 0:
                rs.append(abs(float(np.corrcoef(pair[name], pair[label])[0, 1])))
        if rs:
            scored.append((name, float(np.median(rs))))
    scored.sort(key=lambda t: (-t[1], t[0]))
    return scored[:n]


def indicators_vs_soh(
    table: pd.DataFrame,
    candidates: Sequence[str],
    out: Path,
    *,
    batteries: Sequence[str],
    label: str = "soh_pct",
) -> Path:
    """Scatter of the four strongest health indicators against SOH, one colour per cell."""
    chosen = strongest_indicators(table, candidates, batteries=batteries, label=label, n=4)
    if not chosen:
        raise ValueError("no indicator has enough data for a correlation")
    cells = table[table["battery_id"].isin(list(batteries)) & table["qc_valid"].astype(bool)]
    ncols = 2
    nrows = math.ceil(len(chosen) / ncols)
    with style.context():
        fig, axes = style.figure(nrows, ncols, height_in=2.05 * nrows + 0.75, sharex=True)
        for i, (name, r) in enumerate(chosen):
            ax = axes[i // ncols][i % ncols]
            for battery in batteries:
                d = cells[cells["battery_id"] == battery][[label, name]].dropna()
                st = style.battery_style(battery)
                ax.plot(
                    d[label].to_numpy(float),
                    d[name].to_numpy(float),
                    linestyle="none",
                    marker=st["marker"],
                    color=st["color"],
                    markersize=3.6,
                    markeredgecolor=style.SURFACE,
                    markeredgewidth=0.35,
                    alpha=0.9,
                    label=battery,
                )
            spec = FEATURES.get(name)
            ax.set_ylabel(spec.unit if spec else "")
            style.panel_title(ax, style.feature_label(name))
            # The data run along a diagonal; write the statistic in the empty corner.
            both = cells[[label, name]].dropna()
            rising = len(both) > 2 and float(np.corrcoef(both[label], both[name])[0, 1]) >= 0
            _text(
                ax,
                0.97 if rising else 0.03,
                0.05,
                f"median within-cell |r| = {r:.3f}",
                transform=ax.transAxes,
                ha="right" if rising else "left",
                va="bottom",
            )
            style.value_grid(ax)
            if i // ncols == nrows - 1:
                ax.set_xlabel("SOH (%)")
        for j in range(len(chosen), nrows * ncols):
            axes[j // ncols][j % ncols].set_visible(False)
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper left", ncol=len(batteries), markerscale=1.6)
        style.footnote(
            fig,
            "QC-valid cycles of the core cells. Strongest = highest median within-cell |Pearson r| "
            "with SOH among the combined-set candidates.",
        )
        return style.save(fig, out)


# =========================================================================== 7. extended


def extended_lobo(
    extended: Mapping[str, Any],
    loco: Mapping[str, Any] | None,
    out: Path,
    *,
    feature_set: str = "combined_ext",
    mode: str = "relative",
    models: Sequence[str] = ("ridge", "hgb"),
) -> Path:
    """Per-cell LOBO MAE on the extended multi-condition set (+ leave-one-condition-out)."""
    results = [
        r
        for r in extended.get("results", [])
        if r.get("feature_set") == feature_set and r.get("mode") == mode
    ]
    by_model = {str(r["model"]): r for r in results if str(r["model"]) in models}
    models = [m for m in models if m in by_model][: len(style.SERIES)]
    if not models:
        raise ValueError(f"extended results lack {feature_set}/{mode} for {list(models)}")
    conditions: Mapping[str, str] = extended.get("conditions", {})
    cond_order = list(dict.fromkeys(conditions.get(b, "") for b in sorted(conditions)))
    cond_order.sort(key=lambda c: (c != "2A_24C", c))
    cells = sorted(conditions, key=lambda b: (cond_order.index(conditions[b]), b))
    loco_rows = {}
    if loco and isinstance(loco.get("results"), list):
        loco_rows = {
            str(r["model"]): r
            for r in loco["results"]
            if r.get("feature_set") == feature_set
            and r.get("mode") == mode
            and str(r.get("model")) in models
        }
    has_loco = bool(loco_rows)
    kept = ""
    if mode == "relative":
        absolute = [style.feature_label(f) for f in extended.get("relative_keep_absolute", [])]
        kept = " (reference cycles not scored"
        kept += f"; kept absolute: {'; '.join(absolute)})" if absolute else ")"
    nrows = 2 if has_loco else 1
    ratios = [len(cells), len(cond_order) + 0.6] if has_loco else None
    with style.context():
        fig, axes = style.figure(
            nrows,
            1,
            height_in=0.2 * (len(cells) + (len(cond_order) + 2 if has_loco else 0)) + 1.6,
            sharex=True,
            height_ratios=ratios,
        )
        ax = axes[0][0]
        ypos = {b: i for i, b in enumerate(cells)}
        top = 0.0
        for i, model in enumerate(models):
            st = style.series_style(i)
            per = by_model[model].get("per_battery", {})
            xs = [_finite((per.get(b) or {}).get("mae")) for b in cells]
            ci = by_model[model].get("mae_ci")
            mean = _finite(by_model[model].get("mae"))
            label = style.model_label(model)
            if mean is not None:
                label += f": mean {mean:.2f}"
                if isinstance(ci, list) and len(ci) == 2:
                    label += f" [{float(ci[0]):.2f}, {float(ci[1]):.2f}]"
            ax.plot(
                [np.nan if v is None else v for v in xs],
                [ypos[b] for b in cells],
                linestyle="none",
                marker=st["marker"],
                color=st["color"],
                markersize=5,
                markeredgecolor=style.SURFACE,
                markeredgewidth=0.6,
                label=label,
                zorder=3,
            )
            top = max([top, *(v for v in xs if v is not None)])
        bounds = [
            k for k in range(1, len(cells)) if conditions[cells[k]] != conditions[cells[k - 1]]
        ]
        for k in bounds:
            ax.axhline(k - 0.5, color=style.AXIS, linewidth=0.7)
        ax.set_yticks(range(len(cells)), cells)
        ax.set_ylim(len(cells) - 0.5, -0.5)
        ax.tick_params(axis="y", length=0)
        groups = ax.secondary_yaxis("right")
        centers = []
        for cond in cond_order:
            members = [ypos[b] for b in cells if conditions[b] == cond]
            centers.append((min(members) + max(members)) / 2)
        groups.set_yticks(centers, [style.CONDITION_LABELS.get(c, c) for c in cond_order])
        groups.tick_params(axis="y", length=0, labelsize=7.5)
        groups.spines["right"].set_visible(False)
        style.value_grid(ax, "x")
        style.panel_title(
            ax, f"Leave-one-battery-out, {len(cells)} cells, {len(cond_order)} operating conditions"
        )
        if has_loco:
            ax2 = axes[1][0]
            for i, model in enumerate(models):
                st = style.series_style(i)
                per_cond = (loco_rows.get(model) or {}).get("per_condition", {})
                xs = [_finite((per_cond.get(c) or {}).get("mae")) for c in cond_order]
                ax2.plot(
                    [np.nan if v is None else v for v in xs],
                    range(len(cond_order)),
                    linestyle="none",
                    marker=st["marker"],
                    color=st["color"],
                    markersize=5,
                    markeredgecolor=style.SURFACE,
                    markeredgewidth=0.6,
                    zorder=3,
                )
                top = max([top, *(v for v in xs if v is not None)])
            ax2.set_yticks(
                range(len(cond_order)), [style.CONDITION_LABELS.get(c, c) for c in cond_order]
            )
            ax2.set_ylim(len(cond_order) - 0.5, -0.5)
            ax2.tick_params(axis="y", length=0)
            style.value_grid(ax2, "x")
            style.panel_title(ax2, "Leave-one-condition-out (whole condition held out)")
            ax2.set_xlabel("MAE (SOH percentage points), mean over the held-out cells")
        else:
            ax.set_xlabel(_MAE_LABEL)
        axes[0][0].set_xlim(0, max(1.0, top * 1.12))
        fig.legend(*axes[0][0].get_legend_handles_labels(), loc="outside upper left", ncol=2)
        style.footnote(
            fig,
            f"{style.feature_set_label(feature_set)} indicators, {mode} mode{kept}. Brackets: 95% "
            "bootstrap interval of the mean MAE, resampling cells.",
        )
        return style.save(fig, out)


# =========================================================================== 8. coefficients


def feature_contributions(bundles: Sequence[Mapping[str, Any]], out: Path) -> Path:
    """Standardized ridge coefficients of the production bundles, one panel per bundle.

    Candidates the nested selection dropped are shown as "not selected" so the
    reader sees the whole candidate set.
    """
    if not bundles:
        raise ValueError("no bundle to plot")
    candidates: list[str] = []
    for b in bundles:
        sel = b.get("selection") or {}
        for name in [*sel.get("candidates", []), *b["features"]]:
            if name not in candidates:
                candidates.append(str(name))
    values = [float(c) for b in bundles for c in b["estimator"]["coef"]]
    neg, pos = min(0.0, *values), max(0.0, *values)
    room = 0.3 * (pos - neg) or 1.0  # space for the value labels and "not selected"
    lo, span = neg - room, pos + room
    with style.context():
        fig, axes = style.figure(
            1, len(bundles), height_in=0.26 * len(candidates) + 1.35, sharey=True
        )
        for p, bundle in enumerate(bundles):
            ax = axes[0][p]
            coef = dict(
                zip(bundle["features"], map(float, bundle["estimator"]["coef"]), strict=True)
            )
            for k, name in enumerate(candidates):
                if name in coef:
                    value = coef[name]
                    ax.barh(
                        k,
                        value,
                        height=0.56,
                        color=style.POSITIVE if value >= 0 else style.NEGATIVE,
                    )
                    pad = 0.02 * (span - lo)
                    _text(
                        ax,
                        value + (pad if value >= 0 else -pad),
                        k,
                        f"{value:+.2f}",
                        ha="left" if value >= 0 else "right",
                        va="center",
                        fontsize=7,
                        color=style.INK,
                    )
                else:
                    _text(
                        ax,
                        0.02 * (span - lo),
                        k,
                        "not selected",
                        va="center",
                        fontsize=6.8,
                        color=style.INK_MUTED,
                    )
            ax.axvline(0, color=style.AXIS, linewidth=0.8)
            ax.set_yticks(range(len(candidates)), [style.feature_label(c) for c in candidates])
            ax.set_ylim(len(candidates) - 0.5, -0.5)
            ax.set_xlim(lo, span)
            ax.tick_params(axis="y", length=0)
            style.value_grid(ax, "x")
            alpha = _finite(bundle["estimator"].get("alpha"))
            style.panel_title(ax, str(bundle["model_id"]))
            ax.set_xlabel(
                "SOH points per 1 SD of the input"
                + (f"\n(ridge alpha = {alpha:g})" if alpha is not None else "")
            )
        style.footnote(
            fig,
            "Standardized ridge coefficients of the shipped bundles (relative mode: inputs are "
            "ratios to the reference). Inputs are correlated: signs are not physical effects.",
        )
        return style.save(fig, out)
