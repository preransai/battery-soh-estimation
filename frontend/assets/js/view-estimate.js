// Estimate tab: demo cell picker (battery + cycle), raw V(t) with measurement windows, extracted
// indicators, SOH estimate with conformal interval vs true SOH, contributions, plus the manual
// indicators form (POST /v1/soh/estimate) and CSV upload (POST /v1/soh/estimate-from-cycle).
//
// Honesty note: the shipped bundles were trained on all four demo cells, so the error of a demo
// estimate is in-sample. The result card says so, and when the estimate comes from the bundle of
// the benchmark's headline specification it also shows that cycle's leave-one-battery-out
// estimate (GET /v1/benchmark `predictions`), whose model never saw the cell.
//
// This module owns the demo picker and the order of its requests; the parts of the tab live in
// estimate-curves.js (raw curves), estimate-result.js (result card, contributions, indicators),
// estimate-manual.js (manual form), estimate-csv.js (CSV upload) and estimate-data.js (payload
// readers and signal helpers).

import { api, isAbort } from "./api.js";
import { chartMessage, makeChart } from "./charts.js";
import { announce, debounce, emptyState, errorState, h, mount, skeleton } from "./dom.js";
import { createCsvUpload } from "./estimate-csv.js";
import { renderCurves } from "./estimate-curves.js";
import { normalizeBatteries, normalizeCycle, normalizeModels, referenceFor } from "./estimate-data.js";
import { createManualForm } from "./estimate-manual.js";
import { createResultPanel } from "./estimate-result.js";
import { isLeaky } from "./features.js";
import { fmt, fmtLevel } from "./format.js";
import { isNum } from "./guards.js";

function demoFeatures(features) {
  const out = {};
  for (const [k, v] of Object.entries(features)) if (isNum(v) && !isLeaky(k)) out[k] = v;
  return out;
}

export function initEstimate() {
  const $ = (id) => document.getElementById(id);
  const el = {
    battery: $("battery-select"), slider: $("cycle-slider"), output: $("cycle-output"), model: $("model-select"),
    confidence: $("confidence-select"), result: $("demo-result"), hiTable: $("hi-table"),
    chargeBox: $("charge-box"), dischargeBox: $("discharge-box"), chargeNote: $("charge-note"), dischargeNote: $("discharge-note"),
    contribBox: $("contrib-box"), contribNote: $("contrib-note"),
    manualForm: $("manual-form"), manualFields: $("manual-fields"), manualError: $("manual-form-error"),
    manualResult: $("manual-result"), manualSub: $("manual-sub"), prefill: $("manual-prefill"),
    csvForm: $("csv-form"), csvSegment: $("csv-segment"), csvFile: $("csv-file"), csvSummary: $("csv-summary"),
    csvNote: $("csv-note"), csvError: $("csv-form-error"), csvSubmit: $("csv-submit"), csvSample: $("csv-sample"),
    csvResult: $("csv-result"),
  };
  // `cycle` is the loaded demo cycle (with the battery it belongs to); `est` is the latest
  // rendered demo estimate and `estCycle` the cycle it was computed for (they can differ while
  // a newer cycle is loading).
  const state = { batteries: [], models: [], battery: null, idx: 0, cycle: null, est: null, estCycle: null };
  const charts = {
    charge: makeChart($("chart-charge")),
    discharge: makeChart($("chart-discharge")),
    contrib: makeChart($("chart-contrib")),
  };
  // Request tokens: a response is used only if no newer request of the same kind started.
  let cycleSeq = 0; // demo cycle fetches (loadCycle)
  let cycleSettled = 0; // the newest cycle fetch that has finished (== cycleSeq: none pending)
  let estimateSeq = 0; // demo estimates (refreshEstimate, also run after every cycle load)
  let estimateController = null;

  const confidence = () => Number(el.confidence.value) || 0.9;
  const selectedModel = () => state.models.find((m) => m.id === el.model.value) || null;
  const currentEntry = () => state.battery?.cycles[state.idx] || null;

  const view = { el, state, charts, confidence, selectedModel };
  const result = createResultPanel(view);
  const manual = createManualForm(view);
  const csv = createCsvUpload(view);

  function setResultBusy(busy) {
    el.result.setAttribute("aria-busy", String(busy));
    el.result.classList.toggle("is-updating", busy);
  }

  // ---------------------------------------------------------------- boot
  async function boot() {
    mount(el.result, skeleton({ lines: 4, label: "Loading estimate" }));
    mount(el.hiTable, skeleton({ lines: 5, label: "Loading indicators" }));
    for (const box of [el.chargeBox, el.dischargeBox, el.contribBox]) box.classList.add("is-loading");
    try {
      const [batteries, models] = await Promise.all([api.demoBatteries(), api.models()]);
      state.batteries = normalizeBatteries(batteries);
      state.models = normalizeModels(models);
    } catch (err) {
      for (const box of [el.chargeBox, el.dischargeBox, el.contribBox]) box.classList.remove("is-loading");
      el.battery.replaceChildren(h("option", { text: "Unavailable" }));
      for (const box of [el.chargeBox, el.dischargeBox, el.contribBox]) chartMessage(box, "No data: the demo cells could not be loaded.");
      mount(el.result, errorState(err, { onRetry: retryBoot }));
      mount(el.hiTable, emptyState("No demo data", "The demo cells could not be loaded."));
      return;
    }
    for (const box of [el.chargeBox, el.dischargeBox, el.contribBox]) chartMessage(box, "");
    populateModels();
    manual.renderManualFields();
    csv.updateCsvAvailability();
    if (!state.batteries.length) {
      for (const box of [el.chargeBox, el.dischargeBox, el.contribBox]) box.classList.remove("is-loading");
      mount(el.result, emptyState("No demo cells available", "The server has no demo assets. Use the manual form below."));
      mount(el.hiTable, emptyState("No demo cycle selected", ""));
      return;
    }
    el.battery.replaceChildren(...state.batteries.map((b) => h("option", { value: b.id, text: b.nCycles ? `${b.id} (${b.nCycles} cycles)` : b.id })));
    el.battery.disabled = false;
    selectBattery(state.batteries[0].id, 0.6);
    result.loadHeldOut(); // after the first cycle request: the estimate must not wait for it
  }

  function retryBoot() {
    el.battery.replaceChildren(h("option", { text: "Loading\u2026" }));
    boot();
  }

  function populateModels() {
    const options = [h("option", { value: "", text: "Auto (best for the available indicators)" })];
    for (const m of state.models) {
      const mae = isNum(m.mae) ? ` \u00b7 MAE ${fmt(m.mae, 2)}` : "";
      options.push(h("option", { value: m.id, text: `${m.id}${mae}` }));
    }
    el.model.replaceChildren(...options);
    el.model.disabled = state.models.length === 0;
  }

  function selectBattery(id, fraction = null) {
    const prevFraction = state.battery && state.battery.cycles.length > 1 ? state.idx / (state.battery.cycles.length - 1) : 0.6;
    state.battery = state.batteries.find((b) => b.id === id) || state.batteries[0];
    el.battery.value = state.battery.id;
    const n = state.battery.cycles.length;
    state.idx = Math.round((fraction ?? prevFraction) * (n - 1));
    el.slider.min = "0";
    el.slider.max = String(n - 1);
    el.slider.value = String(state.idx);
    el.slider.disabled = n < 2;
    updateCycleOutput();
    loadCycle();
  }

  function updateCycleOutput() {
    const entry = currentEntry();
    if (!entry) return;
    const total = state.battery.nCycles ? ` of ${state.battery.nCycles}` : "";
    const soh = isNum(entry.soh) ? `, true SOH ${fmt(entry.soh, 1)}%` : "";
    el.output.textContent = `Cycle ${entry.cycle}${soh ? ` \u00b7 SOH ${fmt(entry.soh, 1)}%` : ""}`;
    el.slider.setAttribute("aria-valuetext", `Cycle ${entry.cycle}${total}${soh}`);
  }

  // ---------------------------------------------------------------- demo cycle
  // A cycle load and an estimate refresh have separate tokens: changing the model or the
  // interval while a cycle is loading must not discard that cycle (it is estimated, with the
  // current settings, as soon as it arrives).
  async function loadCycle() {
    const seq = ++cycleSeq;
    const entry = currentEntry();
    if (!entry) return;
    const battery = state.battery;
    setResultBusy(true);
    let raw;
    try {
      raw = await api.demoCycle(battery.id, entry.cycle);
    } catch (err) {
      if (seq !== cycleSeq) return;
      cycleSettled = seq;
      if (isAbort(err)) return;
      estimateSeq += 1; // an estimate of the previous cycle still in flight must not replace this error
      if (estimateController) estimateController.abort();
      showDemoError(err, loadCycle);
      return;
    }
    if (seq !== cycleSeq) return;
    cycleSettled = seq;
    state.cycle = { ...normalizeCycle(raw), battery };
    if (state.cycle.soh === null && isNum(entry.soh)) state.cycle.soh = entry.soh;
    renderCurves(state.cycle, el, charts);
    result.renderIndicators();
    refreshEstimate();
  }

  function refreshEstimate() {
    if (!state.cycle) return;
    const seq = ++estimateSeq;
    setResultBusy(true);
    runDemoEstimate(seq).catch((err) => {
      if (seq === estimateSeq && !isAbort(err)) showDemoError(err, refreshEstimate);
    });
  }

  async function runDemoEstimate(seq) {
    const cycle = state.cycle; // the battery of the cycle, not the (possibly newer) selection
    const model = selectedModel();
    const body = { features: demoFeatures(cycle.features), confidence: confidence() };
    if (model) body.model_id = model.id;
    if (model?.mode === "relative") body.reference = referenceFor(cycle.battery, model);
    if (estimateController) estimateController.abort();
    estimateController = new AbortController();
    const est = await api.estimate(body, { signal: estimateController.signal });
    if (seq !== estimateSeq) return;
    state.est = est;
    state.estCycle = cycle;
    if (cycleSettled === cycleSeq) setResultBusy(false); // else a newer cycle is still loading
    result.renderDemoResult();
    result.renderContributions();
    result.renderIndicators();
    const level = fmtLevel(est?.interval?.confidence ?? confidence());
    const truth = isNum(cycle.soh) ? ` True SOH ${fmt(cycle.soh, 1)} percent.` : "";
    const held = result.heldOutFor(est, cycle);
    const heldText = held ? ` Held-out estimate ${fmt(held.yhat, 1)} percent.` : "";
    announce(`${cycle.battery.id}, cycle ${cycle.cycle}: estimated SOH ${fmt(est.soh_pct, 1)} percent, ${level} interval ${fmt(est.interval?.lower, 1)} to ${fmt(est.interval?.upper, 1)}.${truth}${heldText}`);
  }

  function showDemoError(err, retry = loadCycle) {
    setResultBusy(false);
    result.showError(err, retry);
  }

  // ---------------------------------------------------------------- events
  const debouncedLoad = debounce(loadCycle, 140);
  el.battery.addEventListener("change", () => selectBattery(el.battery.value));
  el.slider.addEventListener("input", () => {
    state.idx = Number(el.slider.value);
    updateCycleOutput();
    debouncedLoad();
  });
  el.slider.addEventListener("change", () => {
    debouncedLoad.cancel();
    if (state.cycle?.cycle !== currentEntry()?.cycle) loadCycle();
  });
  el.model.addEventListener("change", () => {
    manual.renderManualFields();
    csv.updateCsvAvailability();
    refreshEstimate();
  });
  el.confidence.addEventListener("change", refreshEstimate);

  boot();
}
