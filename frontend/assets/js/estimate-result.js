// Estimate tab, result panel: the estimate card of the loaded demo cycle (with the cycle's
// leave-one-battery-out estimate when the served model is the benchmark's headline
// specification), the contributions chart and the table of extracted indicators.

import { api } from "./api.js";
import { chartMessage, chartsAvailable, contributionsConfig } from "./charts.js";
import { contributionItems, estimateSummary, heldOutBlock, indicatorTable, inSampleNote } from "./components.js";
import { emptyState, errorState, h, mount, setVar } from "./dom.js";
import { fmt, fmtSigned } from "./format.js";
import { isNum } from "./guards.js";

/**
 * @param {{el: object, state: object, charts: object}} view Elements, shared state (`models`,
 *   `cycle`, `est`, `estCycle`) and chart handles of the Estimate tab.
 */
export function createResultPanel({ el, state, charts }) {
  // Leave-one-battery-out predictions of the headline spec: {modelIds, byCycle} or null.
  let heldOut = null;

  const isRelativeModel = (id) => state.models.find((m) => m.id === id)?.mode === "relative";

  /**
   * Leave-one-battery-out predictions of the benchmark's headline specification, and the served
   * bundles that ARE that specification (same feature set and mode, trained by the same
   * procedure: bundle `training_protocol.procedure` equals the headline `model`). Only for those
   * bundles is a stored held-out estimate of the same model available. Non-blocking: the
   * result card is re-rendered when the data arrives; on any failure nothing is shown.
   */
  async function loadHeldOut() {
    try {
      const bench = await api.benchmark();
      const hd = bench?.headline;
      const rows = Array.isArray(bench?.predictions) ? bench.predictions : [];
      if (!hd || typeof hd.model !== "string" || !rows.length) return;
      const candidates = state.models.filter((m) => m.featureSet === hd.feature_set && m.mode === (hd.mode || "absolute"));
      const cards = await Promise.allSettled(candidates.map((m) => api.model(m.id)));
      const modelIds = new Set();
      cards.forEach((r, i) => {
        const proto = r.status === "fulfilled" ? r.value?.training_protocol : null;
        const sameExperiment = !proto?.experiment || !hd.experiment || proto.experiment === hd.experiment;
        if (proto && proto.procedure === hd.model && sameExperiment) modelIds.add(candidates[i].id);
      });
      if (!modelIds.size) return;
      const byCycle = new Map();
      for (const r of rows) {
        if (r && typeof r.battery_id === "string" && Number.isInteger(r.cycle_index)) byCycle.set(`${r.battery_id}#${r.cycle_index}`, r);
      }
      heldOut = { modelIds, byCycle };
      if (state.est && state.estCycle) renderDemoResult();
    } catch {
      heldOut = null; // optional enrichment: the estimate itself is unaffected
    }
  }

  /** The held-out estimate for the cycle an estimate was computed for, at the estimate's level. */
  function heldOutFor(est, cycle) {
    const ho = heldOut;
    if (!ho || !est || !cycle || !ho.modelIds.has(est.model_id)) return null;
    const row = ho.byCycle.get(`${cycle.battery?.id}#${cycle.cycle}`);
    if (!row || !isNum(row.yhat)) return null;
    let level = isNum(est.interval?.confidence) ? est.interval.confidence : 0.9;
    let lower = row[`lo_${Math.round(level * 100)}`];
    let upper = row[`hi_${Math.round(level * 100)}`];
    if (!isNum(lower) || !isNum(upper)) {
      level = 0.9;
      lower = row.lo_90;
      upper = row.hi_90;
    }
    return {
      yhat: row.yhat,
      lower: isNum(lower) ? lower : null,
      upper: isNum(upper) ? upper : null,
      level,
      truth: isNum(row.y) ? row.y : cycle.soh,
    };
  }

  function renderIndicators() {
    const feats = state.cycle?.features || {};
    if (!Object.keys(feats).length) {
      mount(el.hiTable, emptyState("No indicators", "This demo cycle has no extracted indicators."));
      return;
    }
    const used = state.est && Array.isArray(state.est.contributions) ? new Set(state.est.contributions.map((c) => c.feature)) : null;
    mount(el.hiTable, h("div", { class: "hi-split" },
      indicatorTable(feats, { used, families: ["charge"], caption: "Charge indicators" }),
      indicatorTable(feats, { used, families: ["discharge", "impedance", "context"], caption: "Discharge, impedance and context indicators" })));
  }

  function renderDemoResult() {
    const est = state.est;
    const cycle = state.estCycle;
    if (!est || !cycle) return;
    const held = heldOutFor(est, cycle);
    const notes = [
      inSampleNote(() => document.getElementById("tab-benchmark")?.click()),
      held ? heldOutBlock(held) : null,
    ];
    mount(el.result, estimateSummary(est, {
      truth: cycle.soh,
      source: `Demo cell ${cycle.battery.id} \u00b7 cycle ${cycle.cycle}`,
      afterTruth: notes,
    }));
  }

  function renderContributions() {
    el.contribBox.classList.remove("is-loading");
    const items = contributionItems(state.est, { relative: isRelativeModel(state.est?.model_id) });
    setVar(el.contribBox, "--rows", Math.max(items.length, 3));
    const canvas = el.contribBox.querySelector("canvas");
    if (!items.length) {
      charts.contrib.set(null);
      chartMessage(el.contribBox, "No contributions returned for this model.");
      el.contribNote.textContent = "";
      return;
    }
    if (!chartsAvailable()) {
      chartMessage(el.contribBox, items.map((d) => `${d.label}: ${fmtSigned(d.contribution, 2)}`).join("; "));
    } else {
      chartMessage(el.contribBox, "");
      charts.contrib.set((pal) => contributionsConfig(pal, items));
    }
    const sum = items.reduce((acc, d) => acc + d.contribution, 0);
    const base = state.est.base_value;
    const raw = isNum(base) ? base + sum : NaN;
    const clipped = isNum(raw) && Math.abs(raw - state.est.soh_pct) > 1e-6;
    el.contribNote.textContent = isNum(base)
      ? `Base value ${fmt(base, 1)}% ${sum >= 0 ? "+" : "\u2212"} ${fmt(Math.abs(sum), 1)} points from the indicators = ${fmt(raw, 1)}%${clipped ? `, clipped to ${fmt(state.est.soh_pct, 1)}%` : ""}.`
      : "";
    canvas.setAttribute("aria-label", `Contributions in SOH points: ${items.map((d) => `${d.label} ${fmtSigned(d.contribution, 2)}`).join(", ")}.`);
  }

  /** Replace the card with an error (and a retry) and drop the explanation of the last estimate. */
  function showError(err, onRetry) {
    mount(el.result, errorState(err, { onRetry }));
    el.contribBox.classList.remove("is-loading");
    charts.contrib.set(null);
    chartMessage(el.contribBox, "No estimate to explain.");
    el.contribNote.textContent = "";
    state.est = null;
    state.estCycle = null;
    if (state.cycle) renderIndicators();
  }

  return { loadHeldOut, heldOutFor, renderIndicators, renderDemoResult, renderContributions, showError };
}
