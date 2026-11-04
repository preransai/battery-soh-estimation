// Chart.js integration: palette from CSS custom properties (re-read on theme change), shared
// defaults, two small inline plugins (measurement-window shading, bar value labels) and the
// chart configurations used by the views. Chart.js 4.4 is vendored (assets/vendor) and loaded as
// a classic deferred script before this module runs.

import { h } from "./dom.js";
import { fmt, fmtSigned } from "./format.js";
import { isNum } from "./guards.js";

const handles = new Set();

/** True when the vendored Chart.js loaded. */
export const chartsAvailable = () => typeof window.Chart === "function";

const reducedMotion = () => window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/** Read the chart palette from CSS custom properties of the current theme. */
export function palette() {
  const css = getComputedStyle(document.documentElement);
  const v = (name) => css.getPropertyValue(name).trim();
  return {
    text: v("--chart-text"),
    muted: v("--chart-muted"),
    grid: v("--chart-grid"),
    axis: v("--chart-axis"),
    line: v("--chart-line"),
    current: v("--chart-current"),
    accent: v("--chart-accent"),
    band: v("--chart-band"),
    window: v("--chart-window"),
    windowLine: v("--chart-window-line"),
    pos: v("--chart-pos"),
    neg: v("--chart-neg"),
    truth: v("--chart-true"),
    crit: v("--chart-crit"),
    neutral: v("--chart-neutral"),
    tooltipBg: v("--chart-tooltip-bg"),
    tooltipText: v("--chart-tooltip-text"),
    surface: v("--surface"),
    font: getComputedStyle(document.body).fontFamily,
  };
}

function applyDefaults(pal) {
  const C = window.Chart;
  C.defaults.font.family = pal.font;
  C.defaults.font.size = 12;
  C.defaults.color = pal.text;
  C.defaults.borderColor = pal.grid;
  if (reducedMotion()) C.defaults.animation = false;
  else if (C.defaults.animation) C.defaults.animation.duration = 180;
  C.defaults.maintainAspectRatio = false;
  C.defaults.responsive = true;
  C.defaults.plugins.tooltip.backgroundColor = pal.tooltipBg;
  C.defaults.plugins.tooltip.titleColor = pal.tooltipText;
  C.defaults.plugins.tooltip.bodyColor = pal.tooltipText;
  C.defaults.plugins.tooltip.borderColor = pal.axis;
  C.defaults.plugins.tooltip.borderWidth = 1;
  C.defaults.plugins.tooltip.padding = 8;
  C.defaults.plugins.tooltip.cornerRadius = 6;
  C.defaults.plugins.tooltip.boxPadding = 4;
  C.defaults.plugins.legend.labels.boxWidth = 12;
  C.defaults.plugins.legend.labels.boxHeight = 12;
}

function render(handle) {
  if (handle.chart) handle.chart.destroy();
  handle.chart = null;
  if (!handle.build) return;
  const pal = palette();
  applyDefaults(pal);
  handle.chart = new window.Chart(handle.canvas, handle.build(pal));
}

/**
 * Create a theme-aware chart. `build(palette)` returns a Chart.js config; it is called again
 * whenever the theme changes or `set()` supplies a new builder.
 * @param {HTMLCanvasElement} canvas
 * @param {((pal: ReturnType<typeof palette>) => object)|null} build
 */
export function makeChart(canvas, build = null) {
  const handle = {
    canvas,
    build,
    chart: null,
    set(nextBuild) {
      this.build = nextBuild;
      if (chartsAvailable()) render(this);
      return this;
    },
    destroy() {
      if (this.chart) this.chart.destroy();
      this.chart = null;
      handles.delete(this);
    },
  };
  handles.add(handle);
  if (build && chartsAvailable()) render(handle);
  return handle;
}

/**
 * Show `message` in a chart box instead of its canvas, or the canvas again when `message` is
 * empty (no data, a failed request, or Chart.js unavailable).
 * @param {HTMLElement} box Element holding one canvas.
 * @param {string} message
 */
export function chartMessage(box, message) {
  const canvas = box.querySelector("canvas");
  canvas.hidden = Boolean(message);
  let msg = box.querySelector(".chart-empty");
  if (!message) {
    if (msg) msg.remove();
    return;
  }
  if (!msg) {
    msg = h("p", { class: "chart-empty" });
    box.append(msg);
  }
  msg.textContent = message;
}

document.addEventListener("soh:themechange", () => {
  if (!chartsAvailable()) return;
  for (const handle of handles) {
    if (handle.canvas.isConnected) render(handle);
    else handles.delete(handle);
  }
});

// ------------------------------------------------------------------ plugins
/** Shades x-ranges (measurement windows) and draws dashed horizontal threshold lines. */
const windowBands = {
  id: "windowBands",
  beforeDatasetsDraw(chart, _args, opts) {
    const { ctx, chartArea: a, scales } = chart;
    const x = scales.x;
    const y = scales.y;
    if (!x || !y || !opts) return;
    ctx.save();
    for (const band of opts.bands || []) {
      const x0 = Math.max(a.left, x.getPixelForValue(band.from));
      const x1 = Math.min(a.right, x.getPixelForValue(band.to));
      if (x1 > x0) {
        ctx.fillStyle = opts.fill;
        ctx.fillRect(x0, a.top, x1 - x0, a.bottom - a.top);
      }
    }
    ctx.setLineDash([4, 4]);
    ctx.lineWidth = 1;
    ctx.strokeStyle = opts.line;
    for (const level of opts.levels || []) {
      const py = y.getPixelForValue(level);
      if (py >= a.top && py <= a.bottom) {
        ctx.beginPath();
        ctx.moveTo(a.left, py);
        ctx.lineTo(a.right, py);
        ctx.stroke();
      }
    }
    ctx.restore();
  },
  afterDatasetsDraw(chart, _args, opts) {
    const { ctx, chartArea: a, scales } = chart;
    if (!opts || !opts.label || !(opts.bands || []).length) return;
    const band = opts.bands[0];
    const x0 = Math.max(a.left, scales.x.getPixelForValue(band.from));
    ctx.save();
    ctx.font = `600 11px ${opts.font}`;
    ctx.fillStyle = opts.text;
    ctx.textBaseline = "top";
    const w = ctx.measureText(opts.label).width;
    const tx = Math.min(x0 + 6, a.right - w - 4);
    ctx.fillText(opts.label, Math.max(a.left + 4, tx), a.top + 6);
    ctx.restore();
  },
};

/** Draws precomputed value labels at the end of horizontal bars. */
const barValueLabels = {
  id: "barValueLabels",
  afterDatasetsDraw(chart, _args, opts) {
    if (!opts || !Array.isArray(opts.labels)) return;
    const meta = chart.getDatasetMeta(0);
    const { ctx, chartArea: a } = chart;
    ctx.save();
    ctx.font = `600 11px ${opts.font}`;
    ctx.fillStyle = opts.color;
    ctx.textBaseline = "middle";
    meta.data.forEach((bar, i) => {
      const label = opts.labels[i];
      if (!label) return;
      const props = bar.getProps(["x", "y", "base"], true);
      const w = ctx.measureText(label).width;
      const positive = props.x >= props.base;
      let tx = positive ? props.x + 6 : props.x - 6 - w;
      let inside = false;
      // No room outside the bar end: draw inside it, in the contrasting colour.
      if (tx + w > a.right) [tx, inside] = [props.x - 6 - w, true];
      if (tx < a.left) [tx, inside] = [props.x + 6, true];
      ctx.fillStyle = inside && opts.insideColor ? opts.insideColor : opts.color;
      ctx.fillText(label, tx, props.y);
    });
    ctx.restore();
  },
};

// ------------------------------------------------------------------ configs
const isNarrow = () => Boolean(window.matchMedia && window.matchMedia("(max-width: 640px)").matches);

/** Split a category label into lines of at most `maxChars` characters (Chart.js multi-line label). */
function wrapWords(text, maxChars) {
  const lines = [];
  let line = "";
  for (const word of String(text).split(" ")) {
    if (line && `${line} ${word}`.length > maxChars) {
      lines.push(line);
      line = word;
    } else line = line ? `${line} ${word}` : word;
  }
  if (line) lines.push(line);
  return lines.length > 1 ? lines : String(text);
}
const axisTitle = (text, pal) => ({ display: true, text, color: pal.muted, font: { size: 11, weight: "600" } });
const thousands = (value) => fmt(value, 0);

/**
 * Voltage (left axis) and current (right axis) against time, with the measurement window shaded.
 * @param {object} pal
 * @param {{seg: {time_s: number[], voltage_v: number[], current_a: number[]}, kind: "charge"|"discharge",
 *   window: {from: number, to: number}|null, levels: number[], label: string}} o
 */
export function voltageConfig(pal, o) {
  const t = o.seg.time_s;
  const volts = [];
  const amps = [];
  for (let i = 0; i < t.length; i += 1) {
    if (isNum(t[i]) && isNum(o.seg.voltage_v[i])) volts.push({ x: t[i], y: o.seg.voltage_v[i] });
    if (isNum(t[i]) && isNum(o.seg.current_a[i])) amps.push({ x: t[i], y: o.seg.current_a[i] });
  }
  const charge = o.kind === "charge";
  return {
    type: "line",
    data: {
      datasets: [
        { label: "Voltage (V)", data: volts, yAxisID: "y", borderColor: pal.line, backgroundColor: pal.line, borderWidth: 1.75, pointRadius: 0, pointHoverRadius: 3, tension: 0 },
        { label: "Current (A)", data: amps, yAxisID: "y1", borderColor: pal.current, backgroundColor: pal.current, borderWidth: 1, pointRadius: 0, pointHoverRadius: 3, tension: 0 },
      ],
    },
    options: {
      parsing: false,
      normalized: true,
      interaction: { mode: "index", intersect: false, axis: "x" },
      scales: {
        x: { type: "linear", title: axisTitle("Time (s)", pal), grid: { color: pal.grid }, border: { color: pal.axis }, ticks: { maxTicksLimit: 6, callback: thousands } },
        y: {
          position: "left", title: axisTitle("Voltage (V)", pal), grid: { color: pal.grid }, border: { color: pal.axis },
          suggestedMin: charge ? 3.4 : 2.6, suggestedMax: 4.3, ticks: { maxTicksLimit: 6, callback: (v) => fmt(v, 1) },
        },
        y1: {
          position: "right", title: axisTitle("Current (A)", pal), grid: { drawOnChartArea: false }, border: { color: pal.axis },
          min: charge ? 0 : undefined, suggestedMin: charge ? 0 : -3, suggestedMax: charge ? 3 : 0.5, ticks: { maxTicksLimit: 5, callback: (v) => fmt(v, 1) },
        },
      },
      plugins: {
        legend: { position: "top", align: "end", labels: { usePointStyle: false, boxHeight: 3, boxWidth: 16, color: pal.text } },
        tooltip: {
          callbacks: {
            title: (items) => `t = ${fmt(items[0].parsed.x, 0)} s`,
            label: (item) => ` ${item.dataset.label.replace(/ \(.*\)$/, "")}: ${fmt(item.parsed.y, item.datasetIndex === 0 ? 3 : 3)} ${item.datasetIndex === 0 ? "V" : "A"}`,
          },
        },
        windowBands: {
          bands: o.window ? [o.window] : [],
          levels: o.levels,
          label: o.window ? o.label : "",
          fill: pal.window,
          line: pal.windowLine,
          text: pal.accent,
          font: pal.font,
        },
      },
    },
    plugins: [windowBands],
  };
}

/**
 * Signed contributions, sorted by magnitude (horizontal bars).
 * @param {object} pal
 * @param {Array<{label: string, contribution: number, detail: string}>} items
 */
export function contributionsConfig(pal, items) {
  const values = items.map((d) => d.contribution);
  const maxAbs = Math.max(0.5, ...values.map((v) => Math.abs(v)));
  // Symmetric axis on a round step (1, 2, 2.5 or 5 x 10^k) with room for the value labels.
  const rough = (maxAbs * 1.35) / 3;
  const magnitude = 10 ** Math.floor(Math.log10(rough));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * magnitude).find((s) => s >= rough);
  const bound = Math.ceil((maxAbs * 1.35) / step) * step;
  let digits = Math.max(0, Math.ceil(-Math.log10(step) - 1e-9));
  if (Math.abs(step * 10 ** digits - Math.round(step * 10 ** digits)) > 1e-9) digits += 1;
  return {
    type: "bar",
    data: {
      labels: items.map((d) => (isNarrow() ? wrapWords(d.label, 14) : d.label)),
      datasets: [{
        label: "Contribution (SOH points)",
        data: values,
        backgroundColor: values.map((v) => (v >= 0 ? pal.pos : pal.neg)),
        borderRadius: 3,
        borderSkipped: false,
        maxBarThickness: 18,
        categoryPercentage: 0.8,
        barPercentage: 0.9,
      }],
    },
    options: {
      indexAxis: "y",
      layout: { padding: { right: 8 } },
      scales: {
        x: {
          min: -bound, max: bound,
          title: axisTitle("Contribution to SOH (points)", pal),
          grid: { color: (ctx) => (ctx.tick && ctx.tick.value === 0 ? pal.axis : pal.grid), lineWidth: (ctx) => (ctx.tick && ctx.tick.value === 0 ? 1.5 : 1) },
          border: { display: false },
          ticks: { stepSize: step, maxRotation: 0, callback: (v) => fmtSigned(v, digits) },
        },
        y: { grid: { display: false }, border: { color: pal.axis }, ticks: { color: pal.text, autoSkip: false } },
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (item) => ` ${fmtSigned(item.parsed.x, 2)} SOH points`,
            afterLabel: (item) => items[item.dataIndex].detail,
          },
        },
        barValueLabels: { labels: values.map((v) => fmtSigned(v, 2)), color: pal.text, insideColor: pal.surface, font: pal.font },
      },
    },
    plugins: [barValueLabels],
  };
}

/**
 * Predicted vs true SOH along cycles with the interval band.
 * @param {object} pal
 * @param {Array<{cycle: number, y: number|null, yhat: number|null, lo: number|null, hi: number|null}>} points
 * @param {{yMin?: number, yMax?: number, yStep?: number, level?: string}} [o] `yStep` labels only
 *   multiples of the step, so an axis bound such as 50 next to 60 does not crowd the labels.
 */
export function trajectoryConfig(pal, points, o = {}) {
  const pts = (key) => points.filter((p) => isNum(p[key])).map((p) => ({ x: p.cycle, y: p[key] }));
  const level = o.level || "90%";
  return {
    type: "line",
    data: {
      datasets: [
        { label: `${level} interval (lower)`, data: pts("lo"), borderWidth: 0, pointRadius: 0, pointHoverRadius: 0, fill: false, backgroundColor: pal.band },
        { label: `${level} interval`, data: pts("hi"), borderWidth: 0, pointRadius: 0, pointHoverRadius: 0, fill: "-1", backgroundColor: pal.band },
        { label: "Predicted", data: pts("yhat"), borderColor: pal.accent, backgroundColor: pal.accent, borderWidth: 1.5, pointRadius: 1.6, pointHoverRadius: 3.5, tension: 0 },
        { label: "True SOH", data: pts("y"), borderColor: pal.truth, backgroundColor: pal.truth, borderWidth: 1.5, pointRadius: 0, pointHoverRadius: 3, tension: 0 },
      ],
    },
    options: {
      parsing: false,
      normalized: true,
      interaction: { mode: "index", intersect: false, axis: "x" },
      scales: {
        x: { type: "linear", title: axisTitle("Cycle", pal), grid: { color: pal.grid }, border: { color: pal.axis }, ticks: { maxTicksLimit: 6, precision: 0 } },
        y: {
          title: axisTitle("SOH (%)", pal), grid: { color: pal.grid }, border: { color: pal.axis },
          min: o.yMin, max: o.yMax,
          ticks: { maxTicksLimit: 8, stepSize: o.yStep, callback: (v) => (o.yStep && v % o.yStep !== 0 ? "" : fmt(v, 0)) },
        },
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          filter: (item) => item.datasetIndex !== 0,
          callbacks: {
            title: (items) => `Cycle ${fmt(items[0].parsed.x, 0)}`,
            label: (item) => {
              if (item.datasetIndex === 1) {
                const lo = item.chart.data.datasets[0].data[item.dataIndex];
                return ` ${level} interval: ${fmt(lo ? lo.y : NaN, 1)}\u2013${fmt(item.parsed.y, 1)}%`;
              }
              return ` ${item.dataset.label}: ${fmt(item.parsed.y, 2)}%`;
            },
          },
        },
      },
    },
  };
}

/**
 * Leakage audit: MAE of each scenario as horizontal bars coloured by category.
 * @param {object} pal
 * @param {Array<{label: string, mae: number, category: string}>} rows
 */
export function auditConfig(pal, rows) {
  const colour = (c) => (c === "v1_reported" ? pal.crit : c === "v2" ? pal.accent : pal.neutral);
  const narrow = isNarrow();
  const wrap = (text) => wrapWords(text, narrow ? 20 : 30);
  const maxMae = Math.max(1, ...rows.map((r) => (isNum(r.mae) ? r.mae : 0)));
  return {
    type: "bar",
    data: {
      labels: rows.map((r) => wrap(r.label)),
      datasets: [{
        label: "MAE (SOH points)",
        data: rows.map((r) => (isNum(r.mae) ? r.mae : null)),
        backgroundColor: rows.map((r) => colour(r.category)),
        borderRadius: 3,
        borderSkipped: false,
        maxBarThickness: 26,
      }],
    },
    options: {
      indexAxis: "y",
      layout: { padding: { right: 12 } },
      scales: {
        x: { min: 0, max: Math.ceil(maxMae * 1.2), title: axisTitle("Mean absolute error (SOH points, lower is better)", pal), grid: { color: pal.grid }, border: { display: false }, ticks: { maxTicksLimit: 6, includeBounds: false } },
        y: { grid: { display: false }, border: { color: pal.axis }, ticks: { color: pal.text, autoSkip: false, font: { size: narrow ? 11 : 12 } } },
      },
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { title: (items) => rows[items[0].dataIndex].label, label: (item) => ` MAE ${fmt(item.parsed.x, 2)} SOH points` } },
        barValueLabels: { labels: rows.map((r) => (isNum(r.mae) ? fmt(r.mae, 2) : "")), color: pal.text, font: pal.font },
      },
    },
    plugins: [barValueLabels],
  };
}
