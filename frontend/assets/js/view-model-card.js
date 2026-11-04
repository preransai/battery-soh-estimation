// Model card tab: renders GET /v1/models/{id} (the bundle without its precision matrix) as a
// readable card: performance, uncertainty calibration, inputs and coefficients, domain of
// applicability, training provenance and the raw JSON.

import { api } from "./api.js";
import { dataTable, emptyState, errorState, h, mount, skeleton, tag } from "./dom.js";
import { featureMeta } from "./features.js";
import { featureSetName, fmt, fmtFeature, fmtPct, fmtSigned, fmtUtc, modelName, shortHash } from "./format.js";
import { isNum, isObj } from "./guards.js";

const PREFERRED = "combined-absolute-ridge";
const at = (arr, j) => (Array.isArray(arr) && isNum(arr[j]) ? arr[j] : null);

/** Value stored under a confidence-level key ("0.9", "0.90", ...) of an object, or null. */
function levelValue(obj, level) {
  for (const [k, v] of Object.entries(obj)) if (Math.abs(Number(k) - level) < 1e-9 && isNum(v)) return v;
  return null;
}

function definitionList(pairs) {
  return h("dl", { class: "kv" }, pairs.filter(([, v]) => v !== null && v !== undefined && v !== "").map(([k, v]) =>
    h("div", { class: "kv-row" }, h("dt", { text: k }), h("dd", null, v instanceof Node ? v : String(v)))));
}

function card(titleId, title, sub, ...children) {
  return h("article", { class: "card", attrs: { "aria-labelledby": titleId } },
    h("header", { class: "card-head" }, h("h3", { id: titleId, class: "card-title", text: title }), sub ? h("p", { class: "card-sub", text: sub }) : null),
    ...children);
}

function headerCard(m) {
  const est = isObj(m.estimator) ? m.estimator : {};
  const tags = [
    m.feature_set ? tag(featureSetName(m.feature_set), "accent") : null,
    m.mode ? tag(`${m.mode} mode`, "neutral") : null,
    est.type ? tag(`${modelName(est.type)}${isNum(est.alpha) ? ` (\u03b1 = ${fmt(est.alpha, est.alpha < 1 ? 3 : 1)})` : ""}`, "neutral") : null,
  ].filter(Boolean);
  return h("article", { class: "card card--header", attrs: { "aria-labelledby": "mc-title" } },
    h("div", { class: "mc-head" },
      h("h3", { id: "mc-title", class: "mc-title" }, h("code", { text: m.model_id || "unknown model" })),
      h("div", { class: "tag-row" }, tags)),
    definitionList([
      ["Label", isObj(m.label) ? `${m.label.name || "soh_pct"}: ${m.label.definition || ""}` : null],
      ["Created", m.created_utc ? fmtUtc(m.created_utc) : null],
      ["Package version", m.package_version || null],
      ["Bundle schema", m.schema_version || null],
      ["Output range", Array.isArray(m.output_clip) ? `${fmt(m.output_clip[0], 0)}\u2013${fmt(m.output_clip[1], 0)}% (predictions are clipped)` : null],
      ["Relative mode", m.mode === "relative" ? `Each indicator is divided by the cell's mean over its first ${m.n_reference_cycles ?? 5} cycles` : null],
    ]));
}

function performanceCard(m) {
  const metrics = isObj(m.metrics) ? m.metrics : {};
  const mean = isObj(metrics.mean) ? metrics.mean : {};
  const coverage = isObj(metrics.coverage) ? metrics.coverage : {};
  const per = isObj(metrics.per_battery)
    ? Object.entries(metrics.per_battery).map(([battery, v]) => {
      const row = isObj(v) ? v : {};
      return { ...row, battery, cov90: isObj(row.coverage) ? levelValue(row.coverage, 0.9) : null };
    })
    : [];
  const children = [];
  const mini = (label, value) => h("div", { class: "mini-metric" }, h("p", { class: "mini-label", text: label }), h("p", { class: "mini-value tnum", text: value }));
  if ([mean.mae, mean.rmse, mean.r2].some(isNum)) {
    children.push(h("div", { class: "mini-metrics" }, mini("MAE", fmt(mean.mae, 2)), mini("RMSE", fmt(mean.rmse, 2)), mini("R\u00b2", fmt(mean.r2, 3))));
  }
  const levels = Object.entries(coverage).filter(([level, v]) => isNum(v) && Number.isFinite(Number(level))).sort(([a], [b]) => Number(a) - Number(b));
  if (levels.length) {
    children.push(h("div", { class: "mini-metrics" }, levels.map(([level, v]) => mini(`Coverage at ${fmtPct(Number(level), 0)}`, fmtPct(v, 1)))));
  }
  if (per.length) {
    const hasN = per.some((r) => isNum(r.n));
    const hasCov = per.some((r) => isNum(r.cov90));
    children.push(dataTable([
      { key: "battery", label: "Held-out battery" },
      ...(hasN ? [{ key: "n", label: "Cycles", num: true, format: (v) => fmt(v, 0) }] : []),
      { key: "mae", label: "MAE", num: true, format: (v) => fmt(v, 2) },
      { key: "rmse", label: "RMSE", num: true, format: (v) => fmt(v, 2) },
      { key: "r2", label: "R\u00b2", num: true, format: (v) => fmt(v, 3) },
      ...(hasCov ? [{ key: "cov90", label: "90% cov.", num: true, title: "Empirical coverage of the 90% interval on this held-out battery", format: (v) => fmtPct(v, 1) }] : []),
    ], per, { caption: "Per-battery leave-one-battery-out metrics", captionHidden: true }));
  }
  if (typeof metrics.evaluates === "string" && metrics.evaluates) {
    children.push(h("p", { class: "hint", text: `These metrics evaluate ${metrics.evaluates}.` }));
  }
  if (!children.length) children.push(emptyState("No metrics recorded", "This bundle carries no evaluation metrics."));
  return card("mc-perf", "Performance", `${metrics.protocol || "leave-one-battery-out"}; SOH points`, ...children);
}

function uncertaintyCard(m) {
  const conformal = isObj(m.conformal) ? m.conformal : {};
  const q = isObj(conformal.quantiles) ? conformal.quantiles : {};
  const rows = Object.entries(q).map(([level, v]) => ({ level: Number(level), half: v })).filter((r) => isNum(r.level)).sort((a, b) => a.level - b.level);
  return card("mc-unc", "Uncertainty", conformal.method || "Conformal prediction interval",
    rows.length
      ? dataTable([
        { key: "level", label: "Confidence", format: (v) => fmtPct(v, 0) },
        { key: "half", label: "Half-width", num: true, format: (v) => (isNum(v) ? `\u00b1${fmt(v, 2)} pts` : "unbounded") },
      ], rows, { caption: "Conformal half-widths by confidence level", captionHidden: true })
      : emptyState("No conformal quantiles", ""),
    definitionList([["Calibration residuals", isNum(conformal.n_scores) ? fmt(conformal.n_scores, 0) : null]]));
}

function inputsCard(m) {
  const features = Array.isArray(m.features) ? m.features : [];
  const pre = isObj(m.preprocessing) ? m.preprocessing : {};
  const est = isObj(m.estimator) ? m.estimator : {};
  const dom = isObj(m.domain) ? m.domain : {};
  const rel = m.mode === "relative";
  const rows = features.map((name, j) => ({
    name,
    unit: rel ? "ratio" : featureMeta(name).unit,
    coef: at(est.coef, j),
    mean: at(pre.scaler_mean, j),
    scale: at(pre.scaler_scale, j),
    median: at(pre.impute_median, j),
    min: at(dom.feature_min, j),
    max: at(dom.feature_max, j),
  }));
  const value = (v, row) => (rel ? fmt(v, 4) : fmtFeature(row.name, v));
  const table = rows.length
    ? dataTable([
      { key: "name", label: "Indicator", format: (name) => h("span", { class: "hi-name" }, h("span", { text: featureMeta(name).label }), h("code", { class: "hi-code", text: name })) },
      { key: "unit", label: "Unit", format: (v) => v || "\u2014" },
      { key: "coef", label: "Coefficient", num: true, title: "SOH points per standard deviation of the input", format: (v) => fmtSigned(v, 3) },
      { key: "mean", label: "Mean", num: true, format: value },
      { key: "scale", label: "Std. dev.", num: true, format: value },
      { key: "median", label: "Median (imputation)", num: true, format: value },
      { key: "min", label: "Train min", num: true, format: value },
      { key: "max", label: "Train max", num: true, format: value },
    ], rows, { caption: "Model inputs, standardisation and coefficients", captionHidden: true, wide: true })
    : emptyState("No inputs listed", "");
  return card("mc-inputs", "Inputs", `Prediction = ${fmt(est.intercept, 2)} + \u03a3 coefficient \u00d7 (input \u2212 mean) \u00f7 std. dev.; missing inputs take the median.`, table);
}

function domainCard(m) {
  const dom = isObj(m.domain) ? m.domain : {};
  const explanation = isObj(m.explanation) ? m.explanation : {};
  return card("mc-domain", "Domain of applicability", "An input is flagged when an indicator leaves its training range (widened by the margin) or is far from the training cloud.",
    definitionList([
      ["Range margin", isNum(dom.margin) ? `${fmt(dom.margin * 100, 0)}% of each training span` : null],
      ["Mahalanobis threshold", isNum(dom.mahalanobis_threshold) ? `${fmt(dom.mahalanobis_threshold, 2)}${isNum(dom.mahalanobis_quantile) ? ` (${fmtPct(dom.mahalanobis_quantile, 0)} of training distances)` : ""}` : null],
      ["Covariance shrinkage", isNum(dom.covariance_shrinkage) ? fmt(dom.covariance_shrinkage, 3) : null],
      ["Attribution", explanation.method || null],
      ["Base value", isNum(explanation.base_value) ? `${fmt(explanation.base_value, 2)}%` : null],
    ]));
}

function trainingCard(m) {
  const t = isObj(m.training) ? m.training : {};
  const protocol = isObj(m.training_protocol) ? m.training_protocol : {};
  const reference = isObj(protocol.relative_reference) ? protocol.relative_reference : {};
  const relative = m.mode === "relative";
  const hash = typeof t.features_sha256 === "string" ? h("code", { class: "hash", text: shortHash(t.features_sha256, 16), attrs: { title: t.features_sha256 } }) : null;
  return card("mc-train", "Training data", null,
    definitionList([
      ["Batteries", Array.isArray(t.batteries) ? t.batteries.join(", ") : null],
      ["Cycles", isNum(t.n_cycles) ? fmt(t.n_cycles, 0) : null],
      ["Reference cycles excluded", relative && isNum(protocol.n_reference_cycles_excluded) ? fmt(protocol.n_reference_cycles_excluded, 0) : null],
      ["Reference rule", relative && typeof reference.rule === "string" ? reference.rule : null],
      ["Feature table SHA-256", hash],
    ]));
}

/** Nested feature selection: shipped inputs, per-fold selection frequency and the elimination path. */
function selectionCard(m) {
  const sel = isObj(m.selection) ? m.selection : null;
  const final = isObj(sel?.final) ? sel.final : null;
  if (!sel || !final) return null;
  const candidates = Array.isArray(sel.candidates) ? sel.candidates.filter((x) => typeof x === "string") : [];
  const selected = new Set(Array.isArray(final.selected) ? final.selected : []);
  const freq = isObj(sel.lobo_frequency) ? sel.lobo_frequency : {};
  const folds = Array.isArray(sel.lobo_folds) ? sel.lobo_folds.length : null;
  const rows = candidates.map((name) => ({ name, shipped: selected.has(name), freq: isNum(freq[name]) ? freq[name] : null }));
  const path = (Array.isArray(final.path) ? final.path : []).filter(isObj).map((step, i) => ({
    step: i,
    removed: typeof step.removed === "string" ? step.removed : null,
    n: step.n_features,
    alpha: step.alpha,
    mae: step.inner_mae,
  }));
  const children = [];
  if (rows.length) {
    children.push(dataTable([
      { key: "name", label: "Candidate indicator", format: (name) => h("span", { class: "hi-name" }, h("span", { text: featureMeta(name).label }), h("code", { class: "hi-code", text: name })) },
      { key: "shipped", label: "Shipped model", rowHeader: false, format: (v) => (v ? tag("selected", "good") : h("span", { class: "muted", text: "dropped" })) },
      {
        key: "freq", label: folds ? `Kept in LOBO folds (of ${folds})` : "Kept in LOBO folds", num: true,
        title: "Share of the leave-one-battery-out folds whose nested selection kept this indicator", format: (v) => fmtPct(v, 0),
      },
    ], rows, { caption: "Candidate indicators and how often nested selection kept them", captionHidden: true }));
  }
  if (path.length) {
    children.push(h("details", { class: "details" }, h("summary", { text: "Backward-elimination path on all training cells" }),
      dataTable([
        { key: "step", label: "Step", format: (v) => String(v) },
        { key: "removed", label: "Removed", rowHeader: false, format: (v) => (v ? featureMeta(v).short : "none (all candidates)") },
        { key: "n", label: "Inputs", num: true, format: (v) => fmt(v, 0) },
        { key: "alpha", label: "\u03b1", num: true, format: (v) => (isNum(v) ? String(v) : "\u2014") },
        { key: "mae", label: "Inner MAE", num: true, format: (v) => fmt(v, 3) },
      ], path, { caption: "Backward-elimination path on all training cells", captionHidden: true })));
  }
  const sub = [
    "Inputs and ridge penalty chosen by inner leave-one-battery-out MAE",
    isNum(final.alpha) ? `final \u03b1 = ${final.alpha}` : null,
    isNum(final.inner_mae) ? `inner MAE ${fmt(final.inner_mae, 2)}` : null,
  ].filter(Boolean).join("; ");
  return card("mc-select", "Feature selection", `${sub}.`, ...children);
}

function rawJson(m) {
  const copy = JSON.parse(JSON.stringify(m));
  if (isObj(copy.domain) && Array.isArray(copy.domain.precision_matrix)) {
    const p = copy.domain.precision_matrix.length;
    copy.domain.precision_matrix = `[${p} x ${p} matrix omitted]`;
  }
  return h("details", { class: "details" }, h("summary", { text: "Raw model card JSON" }),
    h("pre", { class: "code-block", attrs: { tabindex: "0" } }, h("code", { text: JSON.stringify(copy, null, 2) })));
}

export function initModelCard() {
  const select = document.getElementById("card-model-select");
  const root = document.getElementById("card-root");
  let seq = 0;

  async function show(id) {
    const my = ++seq;
    root.setAttribute("aria-busy", "true");
    mount(root, h("article", { class: "card" }, skeleton({ lines: 6, label: "Loading model card" })));
    try {
      const m = await api.model(id);
      if (my !== seq) return;
      mount(root,
        headerCard(m),
        h("div", { class: "grid grid--two" }, performanceCard(m), uncertaintyCard(m)),
        inputsCard(m),
        selectionCard(m),
        h("div", { class: "grid grid--two" }, domainCard(m), trainingCard(m)),
        rawJson(m));
    } catch (err) {
      if (my !== seq) return;
      mount(root, errorState(err, { onRetry: () => show(id) }));
    } finally {
      if (my === seq) root.setAttribute("aria-busy", "false");
    }
  }

  async function boot() {
    root.setAttribute("aria-busy", "true");
    mount(root, h("article", { class: "card" }, skeleton({ lines: 6, label: "Loading models" })));
    let models;
    try {
      const raw = await api.models();
      models = (Array.isArray(raw) ? raw : []).filter((x) => x && typeof x.model_id === "string");
    } catch (err) {
      select.replaceChildren(h("option", { text: "Unavailable" }));
      root.setAttribute("aria-busy", "false");
      mount(root, errorState(err, { onRetry: boot }));
      return;
    }
    if (!models.length) {
      select.replaceChildren(h("option", { text: "No models" }));
      mount(root, emptyState("No models published", "The API registry has no model bundles."));
      root.setAttribute("aria-busy", "false");
      return;
    }
    select.replaceChildren(...models.map((x) => h("option", { value: x.model_id, text: x.model_id })));
    select.disabled = false;
    select.value = models.some((x) => x.model_id === PREFERRED) ? PREFERRED : models[0].model_id;
    select.addEventListener("change", () => show(select.value));
    show(select.value);
  }
  boot();
}
