// Benchmark tab: headline metric cards, per-battery predicted-vs-true trajectories with the 90 %
// band, leaderboard (feature set x model), per-battery and coverage tables. Every field of the
// summary is optional; sections without data render an explanatory empty state.

import { api } from "./api.js";
import { chartsAvailable, makeChart, trajectoryConfig } from "./charts.js";
import { dataTable, emptyState, errorState, h, mount, skeleton, tag } from "./dom.js";
import { experimentName, featureSetName, fmt, fmtPct, fmtUtc, modelName, shortHash } from "./format.js";
import { isNum } from "./guards.js";
import { normalizeBenchmark } from "./normalize.js";

const COLLAPSED_ROWS = 10;

const SORTERS = {
  featureSet: (r) => r.featureSet,
  mode: (r) => r.mode,
  model: (r) => r.model,
  mae: (r) => r.mae,
  rmse: (r) => r.rmse,
  r2: (r) => r.r2,
  coverage90: (r) => r.coverage90,
};

function metricCard({ label, value, unit, sub, emphasis = false }) {
  return h("div", { class: ["metric", emphasis && "metric--emphasis"] },
    h("p", { class: "metric-label", text: label }),
    h("p", { class: "metric-value" }, h("span", { class: "tnum", text: value }), unit ? h("span", { class: "metric-unit", text: unit }) : null),
    sub ? h("p", { class: "metric-sub", text: sub }) : null);
}

function headlineSection(b) {
  const hd = b.headline;
  if (!hd) {
    return h("article", { class: "card" }, emptyState("No headline metrics", "The benchmark summary does not contain headline numbers or a leaderboard."));
  }
  const base = Object.fromEntries(b.baselines.map((x) => [x.name, x.mae]));
  const baseText = [
    isNum(base.mean) ? `mean predictor ${fmt(base.mean, 2)}` : null,
    isNum(base.cycle_only) ? `cycle count ${fmt(base.cycle_only, 2)}` : null,
  ].filter(Boolean).join(", ");
  const describe = [featureSetName(hd.featureSet), modelName(hd.model), hd.mode ? `${hd.mode} mode` : null,
    isNum(hd.nBatteries) ? `${hd.nBatteries} held-out batteries` : null, isNum(hd.n) ? `${fmt(hd.n, 0)} cycles` : null]
    .filter(Boolean).join(" \u00b7 ");
  return h("section", { class: "headline", attrs: { "aria-labelledby": "bench-headline-title" } },
    h("div", { class: "section-head" },
      h("h3", { id: "bench-headline-title", class: "section-title section-title--flush", text: "Headline: leave-one-battery-out" }),
      h("p", { class: "section-sub", text: `${describe}${b.headlineDerived ? " (derived from the leaderboard)" : ""}` })),
    h("div", { class: "metrics" },
      metricCard({ label: "Mean absolute error", value: fmt(hd.mae, 2), unit: "SOH pts", sub: baseText ? `Baselines: ${baseText}` : "Lower is better", emphasis: true }),
      metricCard({ label: "RMSE", value: fmt(hd.rmse, 2), unit: "SOH pts", sub: "Penalises large misses" }),
      metricCard({ label: "R\u00b2", value: fmt(hd.r2, 3), sub: "Across all held-out cycles" }),
      metricCard({ label: "90% interval coverage", value: fmtPct(hd.coverage90, 1), sub: isNum(hd.width90) ? `Mean width ${fmt(hd.width90, 1)} pts (target 90%)` : "Target 90%" })),
    relativeNote(b.relativeHeadline));
}

/** One line on the relative-mode result (indicators divided by the cell's commissioning reference). */
function relativeNote(rel) {
  if (!rel || !isNum(rel.mae)) return null;
  const parts = [`MAE ${fmt(rel.mae, 2)} SOH pts`];
  if (isNum(rel.r2)) parts.push(`R\u00b2 ${fmt(rel.r2, 3)}`);
  if (isNum(rel.coverage90)) parts.push(`90% coverage ${fmtPct(rel.coverage90, 1)}`);
  // Label the model explicitly: this row is plain Ridge, not the nested-selection headline model.
  const model = rel.model === "ridge" ? "plain Ridge" : modelName(rel.model);
  const spec = [featureSetName(rel.featureSet), model].filter((x) => x && x !== "\u2014").join(", ");
  return h("p", { class: "note" },
    h("strong", { text: "With a commissioning reference: " }),
    `dividing each indicator by the cell's own mean over its first valid cycles (relative mode${spec ? `, ${spec}` : ""}) gives ${parts.join(", ")}. `
    + "It needs those early cycles of the same cell, so the absolute model above is the headline.");
}

function trajectoriesSection(b) {
  const card = h("article", { class: "card", attrs: { "aria-labelledby": "bench-traj-title" } },
    h("header", { class: "card-head" },
      h("h3", { id: "bench-traj-title", class: "card-title", text: "Predicted vs true SOH, per held-out battery" }),
      h("p", { class: "card-sub", text: "Each panel is a battery the model never saw in training or calibration. The band is the 90% conformal interval." })));
  if (!b.predictions.size) {
    card.append(emptyState("No per-cycle predictions", "The summary has no out-of-fold predictions to plot."));
    return { node: card, draw: () => {} };
  }
  card.append(h("ul", { class: "legend", attrs: { "aria-label": "Legend" } },
    h("li", null, h("span", { class: "key key--line-true" }), "True SOH"),
    h("li", null, h("span", { class: "key key--line-pred" }), "Predicted"),
    h("li", null, h("span", { class: "key key--band" }), "90% interval")));
  const perBattery = new Map(b.perBattery.map((r) => [r.battery, r]));
  const all = [...b.predictions.values()].flat();
  const values = all.flatMap((p) => [p.y, p.yhat, p.lo, p.hi]).filter(isNum);
  const yMin = Math.max(0, Math.floor(Math.min(...values) / 10) * 10);
  const yMax = Math.min(120, Math.ceil(Math.max(...values) / 10) * 10);
  const yStep = yMax - yMin > 50 ? 20 : 10;
  const grid = h("div", { class: "small-multiples" });
  const pending = [];
  for (const [battery, points] of b.predictions) {
    const m = perBattery.get(battery);
    const errs = points.filter((p) => isNum(p.y) && isNum(p.yhat)).map((p) => Math.abs(p.y - p.yhat));
    const mae = m?.mae ?? (errs.length ? errs.reduce((a, c) => a + c, 0) / errs.length : null);
    const canvas = h("canvas", {
      role: "img",
      attrs: { "aria-label": `${battery}: predicted and true SOH over ${points.length} cycles, mean absolute error ${fmt(mae, 2)} points.` },
    });
    grid.append(h("figure", { class: "chart-figure" },
      h("figcaption", { class: "chart-caption" }, h("strong", { text: battery }), h("span", { class: "muted tnum", text: ` \u00b7 MAE ${fmt(mae, 2)}` })),
      h("div", { class: "chart-box chart-box--traj" }, canvas)));
    pending.push({ canvas, points });
  }
  card.append(grid);
  const draw = () => {
    if (!chartsAvailable()) return;
    for (const { canvas, points } of pending) makeChart(canvas, (pal) => trajectoryConfig(pal, points, { yMin, yMax, yStep }));
  };
  return { node: card, draw };
}

function leaderboardSection(b) {
  const card = h("article", { class: "card", attrs: { "aria-labelledby": "bench-board-title" } },
    h("header", { class: "card-head" },
      h("h3", { id: "bench-board-title", class: "card-title", text: "Leaderboard" }),
      h("p", { class: "card-sub", text: "Feature set \u00d7 model, leave-one-battery-out. Baselines and leaky sanity checks are labelled." })));
  if (!b.leaderboard.length) {
    card.append(emptyState("No leaderboard", "The summary has no per-model results."));
    return card;
  }
  const experiments = [...new Set(b.leaderboard.map((r) => r.experiment).filter(Boolean))];
  const state = { sortKey: "mae", sortDir: "asc", expanded: false, experiment: experiments.includes("core_lobo") ? "core_lobo" : experiments[0] || "" };
  const hd = b.headline;
  const isHeadline = (r) => Boolean(hd) && r.featureSet === hd.featureSet && r.model === hd.model
    && (!hd.mode || !r.mode || r.mode === hd.mode) && (!hd.experiment || !r.experiment || r.experiment === hd.experiment);
  const holder = h("div");
  const hasMode = b.leaderboard.some((r) => r.mode);
  const hasCoverage = b.leaderboard.some((r) => isNum(r.coverage90));

  function render() {
    let rows = b.leaderboard.filter((r) => !state.experiment || r.experiment === state.experiment || !r.experiment);
    const key = SORTERS[state.sortKey];
    const dir = state.sortDir === "asc" ? 1 : -1;
    rows = [...rows].sort((a, c) => {
      const x = key(a);
      const y = key(c);
      const missing = (v) => v === null || v === undefined || v === "";
      if (missing(x) || missing(y)) return Number(missing(x)) - Number(missing(y)); // missing values last

      return (typeof x === "string" ? x.localeCompare(y) : x - y) * dir;
    });
    const total = rows.length;
    if (!state.expanded && total > COLLAPSED_ROWS + 3) {
      // Collapsed: the best rows in the current order, plus the headline and every baseline.
      const top = rows.slice(0, COLLAPSED_ROWS);
      rows = [...top, ...rows.slice(COLLAPSED_ROWS).filter((r) => r.isBaseline || isHeadline(r))];
    }
    const columns = [
      {
        key: "featureSet", label: "Feature set",
        format: (v, r) => h("span", { class: "cell-stack" }, featureSetName(v),
          isHeadline(r) ? tag("headline", "accent") : null, r.isBaseline && r.featureSet !== "baseline" ? tag("baseline", "neutral") : null,
          r.postHoc ? tag("post hoc", "neutral", "Feature set defined after seeing results: a diagnostic ablation, not a valid model selection") : null,
          r.leaky ? tag("leaky", "warn", "Uses label-side information: sanity check only") : null),
      },
      ...(hasMode ? [{ key: "mode", label: "Mode", format: (v) => v || "\u2014" }] : []),
      { key: "model", label: "Model", format: (v) => modelName(v) },
      { key: "mae", label: "MAE", num: true, title: "Mean absolute error, SOH points", format: (v) => fmt(v, 2) },
      { key: "rmse", label: "RMSE", num: true, format: (v) => fmt(v, 2) },
      { key: "r2", label: "R\u00b2", num: true, format: (v) => fmt(v, 3) },
      ...(hasCoverage ? [{ key: "coverage90", label: "90% cov.", num: true, title: "Empirical coverage of the 90% interval", format: (v) => fmtPct(v, 1) }] : []),
    ];
    mount(holder, dataTable(columns, rows, {
      caption: "Leaderboard: feature set by model, leave-one-battery-out",
      captionHidden: true,
      wide: true,
      sortKey: state.sortKey,
      sortDir: state.sortDir,
      onSort: (k) => {
        state.sortDir = state.sortKey === k && state.sortDir === "asc" ? "desc" : "asc";
        state.sortKey = k;
        render();
        holder.querySelector(`th[aria-sort="${state.sortDir === "asc" ? "ascending" : "descending"}"] button`)?.focus();
      },
      rowClass: (r) => [isHeadline(r) && "row-highlight", r.leaky && "row-leaky"].filter(Boolean).join(" ") || null,
    }));
    if (total > COLLAPSED_ROWS + 3) {
      const toggle = h("button", {
        type: "button",
        class: "btn btn-small",
        attrs: { "aria-expanded": String(state.expanded) },
        text: state.expanded ? "Show fewer rows" : `Show all ${total} rows`,
        on: {
          click: () => {
            state.expanded = !state.expanded;
            render();
            holder.querySelector(".btn-small")?.focus();
          },
        },
      });
      holder.append(h("div", { class: "table-footer" },
        h("p", { class: "hint", text: state.expanded ? `${total} rows` : `Showing the first ${COLLAPSED_ROWS} rows in the current order, plus the headline and the baselines.` }), toggle));
    }
  }

  if (experiments.length > 1) {
    const select = h("select", { id: "bench-experiment" },
      experiments.map((e) => h("option", { value: e, text: experimentName(e), props: { selected: e === state.experiment } })));
    select.addEventListener("change", () => {
      state.experiment = select.value;
      state.expanded = false;
      render();
    });
    card.append(h("div", { class: "field field--inline" }, h("label", { attrs: { for: "bench-experiment" }, text: "Experiment" }), select));
  }
  card.append(holder);
  render();
  return card;
}

function perBatteryCard(b) {
  const card = h("article", { class: "card", attrs: { "aria-labelledby": "bench-pb-title" } },
    h("header", { class: "card-head" }, h("h3", { id: "bench-pb-title", class: "card-title", text: "Per held-out battery" })));
  if (!b.perBattery.length) {
    card.append(emptyState("No per-battery table", "The summary does not break metrics down by battery."));
    return card;
  }
  const hasCov = b.perBattery.some((r) => isNum(r.coverage90));
  card.append(dataTable([
    { key: "battery", label: "Battery" },
    { key: "n", label: "Cycles", num: true, format: (v) => fmt(v, 0) },
    { key: "mae", label: "MAE", num: true, format: (v) => fmt(v, 2) },
    { key: "rmse", label: "RMSE", num: true, format: (v) => fmt(v, 2) },
    { key: "r2", label: "R\u00b2", num: true, format: (v) => fmt(v, 3) },
    ...(hasCov ? [{ key: "coverage90", label: "90% cov.", num: true, format: (v) => fmtPct(v, 1) }] : []),
  ], b.perBattery, { caption: "Metrics per held-out battery", captionHidden: true }));
  return card;
}

function coverageCard(b) {
  const card = h("article", { class: "card", attrs: { "aria-labelledby": "bench-cov-title" } },
    h("header", { class: "card-head" },
      h("h3", { id: "bench-cov-title", class: "card-title", text: "Interval calibration" }),
      h("p", { class: "card-sub", text: "Share of held-out cycles whose true SOH falls inside the interval, at each nominal level." })));
  if (!b.coverage.length) {
    card.append(emptyState("No coverage table", "The summary does not report interval coverage."));
    return card;
  }
  card.append(dataTable([
    { key: "nominal", label: "Nominal", format: (v) => fmtPct(v, 0) },
    { key: "coverage", label: "Empirical", num: true, format: (v) => fmtPct(v, 1) },
    {
      key: "gap", label: "Gap", num: true, title: "Empirical minus nominal, percentage points",
      format: (_v, r) => (isNum(r.coverage) ? `${r.coverage - r.nominal >= 0 ? "+" : "\u2212"}${fmt(Math.abs(r.coverage - r.nominal) * 100, 1)} pp` : "\u2014"),
    },
    { key: "width", label: "Mean width", num: true, format: (v) => (isNum(v) ? `${fmt(v, 2)} pts` : "\u2014") },
  ], b.coverage, { caption: "Nominal versus empirical interval coverage", captionHidden: true }));
  return card;
}

function provenance(b) {
  const m = b.meta;
  const parts = [
    m.protocol ? `Protocol: ${m.protocol}` : null,
    m.generated ? `Generated ${fmtUtc(m.generated)}` : null,
    m.packageVersion ? `package ${m.packageVersion}` : null,
    m.configSha ? `config ${shortHash(m.configSha)}` : null,
    m.dataSha ? `data ${shortHash(m.dataSha)}` : null,
    isNum(m.runtime) ? `runtime ${fmt(m.runtime, 0)} s` : null,
  ].filter(Boolean);
  return parts.length ? h("p", { class: "provenance tnum", text: parts.join(" \u00b7 ") }) : null;
}

export function initBenchmark() {
  const root = document.getElementById("bench-root");
  async function load() {
    root.setAttribute("aria-busy", "true");
    mount(root, h("div", { class: "metrics" }, [0, 1, 2, 3].map(() => h("div", { class: "metric" }, skeleton({ lines: 2, label: "Loading benchmark" })))),
      h("article", { class: "card" }, skeleton({ chart: true, lines: 1, label: "Loading trajectories" })));
    let summary;
    try {
      summary = normalizeBenchmark(await api.benchmark());
    } catch (err) {
      root.setAttribute("aria-busy", "false");
      mount(root, errorState(err, { onRetry: load }));
      return;
    }
    const traj = trajectoriesSection(summary);
    mount(root,
      headlineSection(summary),
      traj.node,
      leaderboardSection(summary),
      h("div", { class: "grid grid--two" }, perBatteryCard(summary), coverageCard(summary)),
      provenance(summary));
    traj.draw();
    root.setAttribute("aria-busy", "false");
  }
  load();
}
