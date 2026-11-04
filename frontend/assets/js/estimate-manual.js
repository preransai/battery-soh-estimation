// Estimate tab, manual form (POST /v1/soh/estimate): one field per input of the selected model
// (or of every absolute model for automatic selection), a reference field per input for a
// relative model, training-range hints from the model cards, client-side validation, and a
// prefill from the loaded demo cycle.

import { api } from "./api.js";
import { estimateSummary, topDrivers } from "./components.js";
import { announce, errorState, h, mount, skeleton } from "./dom.js";
import { referenceFor } from "./estimate-data.js";
import { featureMeta, sortFeatures } from "./features.js";
import { fmt, fmtLevel } from "./format.js";
import { isNum } from "./guards.js";

function numberInput(id, name, describedBy) {
  return h("input", {
    id, name, type: "number", step: "any", inputmode: "decimal", autocomplete: "off",
    attrs: { "aria-describedby": describedBy },
  });
}

function setFieldError(input, message) {
  const container = input.closest("[data-feature]");
  const err = container?.querySelector(".field-error");
  const hintId = `${container?.dataset.feature ? `mf-${container.dataset.feature}-hint` : ""}`;
  if (message) {
    input.setAttribute("aria-invalid", "true");
    if (err) {
      err.textContent = message;
      err.hidden = false;
      input.setAttribute("aria-describedby", `${hintId} ${err.id}`.trim());
    }
  } else {
    input.removeAttribute("aria-invalid");
    input.setAttribute("aria-describedby", hintId);
  }
}

function readInput(input, name) {
  const raw = input.value.trim();
  if (input.validity.badInput) return { error: "Enter a number." };
  if (!raw) return { value: null };
  const value = Number(raw);
  if (!Number.isFinite(value)) return { error: "Enter a finite number." };
  const [lo, hi] = featureMeta(name).range;
  if (value < lo || value > hi) return { error: `Must be between ${fmt(lo, lo % 1 ? 1 : 0)} and ${fmt(hi, hi % 1 ? 1 : 0)}.` };
  return { value };
}

/**
 * Wire the manual form; returns `renderManualFields`, to call whenever the model list or the
 * selected model changes.
 * @param {{el: object, state: object, confidence: () => number, selectedModel: () => object|null}} view
 */
export function createManualForm({ el, state, confidence, selectedModel }) {
  let hintSeq = 0; // training-range hints: a response is used only if no newer request started

  /** Absolute models (the ones automatic selection may use without a reference). */
  const absoluteModels = () => state.models.filter((m) => m.mode === "absolute");

  /**
   * Inputs offered by the manual form: the selected model's, or for automatic selection the
   * union of the absolute models' inputs (the server then picks the best complete model).
   */
  function manualFeatureList() {
    const model = selectedModel();
    if (model && model.features.length) return model.features;
    const union = new Set(absoluteModels().flatMap((m) => m.features));
    return union.size
      ? sortFeatures(union)
      : sortFeatures(["chg_cc_time_390_419_s", "chg_cv_time_s", "chg_ica_peak_height_ah_per_v", "chg_ica_peak_voltage_v", "chg_temp_rise_cc_c", "dis_q_380_350_ah", "dis_r0_ohm", "dis_temp_rise_win_c"]);
  }

  /** Model cards that describe the manual inputs (for the training-range hints). */
  function hintModelIds() {
    const model = selectedModel();
    return model ? [model.id] : absoluteModels().map((m) => m.id);
  }

  function renderManualFields() {
    const previous = new Map([...el.manualFields.querySelectorAll("input")].map((i) => [i.id, i.value]));
    const feats = manualFeatureList();
    const relative = selectedModel()?.mode === "relative";
    el.manualSub.textContent = relative
      ? "Relative model: enter each indicator and the cell's reference value (its mean over the first five valid cycles). Blank fields are imputed and reported."
      : "Enter indicators measured on your cell. Blank fields are imputed with the training median (and reported).";
    const nodes = feats.map((name) => {
      const meta = featureMeta(name);
      const unit = meta.unit ? ` (${meta.unit})` : "";
      const hintId = `mf-${name}-hint`;
      const errId = `mf-${name}-err`;
      const hint = h("p", { class: "hint tnum", id: hintId, text: `Plausible ${fmt(meta.range[0], 0)}\u2013${fmt(meta.range[1], meta.range[1] < 10 ? 1 : 0)}${unit}` });
      const err = h("p", { class: "field-error", id: errId, hidden: true });
      if (!relative) {
        const input = numberInput(`mf-${name}`, name, hintId);
        return h("div", { class: "field", dataset: { feature: name } },
          h("label", { attrs: { for: `mf-${name}` } }, meta.label, h("span", { class: "unit", text: unit })), input, hint, err);
      }
      return h("fieldset", { class: "field field--pair", dataset: { feature: name } },
        h("legend", null, meta.label, h("span", { class: "unit", text: unit })),
        h("div", { class: "pair" },
          h("div", null, h("label", { class: "sublabel", attrs: { for: `mf-${name}` }, text: "This cycle" }), numberInput(`mf-${name}`, name, hintId)),
          h("div", null, h("label", { class: "sublabel", attrs: { for: `mr-${name}` }, text: "Reference" }), numberInput(`mr-${name}`, `ref-${name}`, hintId))),
        hint, err);
    });
    el.manualFields.replaceChildren(...nodes);
    for (const input of el.manualFields.querySelectorAll("input")) {
      if (previous.has(input.id)) input.value = previous.get(input.id);
    }
    const ids = hintModelIds();
    const token = ++hintSeq;
    if (!ids.length) return;
    Promise.allSettled(ids.map((id) => api.model(id))).then((results) => {
      if (token !== hintSeq) return; // the fields were re-rendered for another model meanwhile
      const done = new Set();
      for (const r of results) {
        const card = r.status === "fulfilled" ? r.value : null;
        const fmin = card?.domain?.feature_min;
        const fmax = card?.domain?.feature_max;
        const names = Array.isArray(card?.features) ? card.features : [];
        if (!Array.isArray(fmin) || !Array.isArray(fmax)) continue;
        names.forEach((name, j) => {
          const hint = document.getElementById(`mf-${name}-hint`);
          if (done.has(name) || !hint || !isNum(fmin[j]) || !isNum(fmax[j])) return;
          done.add(name);
          const meta = featureMeta(name);
          const relativeCard = card.mode === "relative";
          const digits = relativeCard ? 3 : meta.digits;
          const unit = !relativeCard && meta.unit ? ` ${meta.unit}` : "";
          hint.textContent = `Training range ${fmt(fmin[j], digits)}\u2013${fmt(fmax[j], digits)}${unit}${relativeCard ? " (ratio to reference)" : ""}`;
        });
      }
    });
  }

  function clearManualErrors() {
    for (const input of el.manualFields.querySelectorAll("input")) setFieldError(input, "");
    for (const err of el.manualFields.querySelectorAll(".field-error")) {
      err.hidden = true;
      err.textContent = "";
    }
    el.manualError.hidden = true;
    el.manualError.textContent = "";
  }

  async function submitManual(event) {
    event.preventDefault();
    clearManualErrors();
    const relative = selectedModel()?.mode === "relative";
    const features = {};
    const reference = {};
    let firstInvalid = null;
    for (const container of el.manualFields.querySelectorAll("[data-feature]")) {
      const name = container.dataset.feature;
      const input = document.getElementById(`mf-${name}`);
      const r = readInput(input, name);
      if (r.error) {
        setFieldError(input, r.error);
        firstInvalid = firstInvalid || input;
        continue;
      }
      if (r.value !== null) features[name] = r.value;
      if (relative) {
        const refInput = document.getElementById(`mr-${name}`);
        const ref = readInput(refInput, name);
        if (ref.error) {
          setFieldError(refInput, `Reference: ${ref.error}`);
          firstInvalid = firstInvalid || refInput;
        } else if (r.value !== null && !name.startsWith("ctx_") && (ref.value === null || ref.value === 0)) {
          setFieldError(refInput, "Enter a non-zero reference value (relative models divide by it).");
          firstInvalid = firstInvalid || refInput;
        } else if (ref.value !== null) reference[name] = ref.value;
      }
    }
    if (firstInvalid) {
      el.manualError.textContent = "Some values need attention.";
      el.manualError.hidden = false;
      firstInvalid.focus();
      return;
    }
    if (!Object.keys(features).length) {
      el.manualError.textContent = "Enter at least one indicator.";
      el.manualError.hidden = false;
      el.manualFields.querySelector("input")?.focus();
      return;
    }
    const model = selectedModel();
    const body = { features, confidence: confidence() };
    if (model) body.model_id = model.id;
    if (relative) body.reference = reference;
    const button = el.manualForm.querySelector('button[type="submit"]');
    button.disabled = true;
    el.manualResult.setAttribute("aria-busy", "true");
    mount(el.manualResult, skeleton({ lines: 3, label: "Estimating" }));
    try {
      const est = await api.estimate(body);
      mount(el.manualResult, estimateSummary(est, { compact: true, source: "Result for your indicators" }), topDrivers(est));
      announce(`Estimated SOH ${fmt(est.soh_pct, 1)} percent, ${fmtLevel(est.interval?.confidence)} interval ${fmt(est.interval?.lower, 1)} to ${fmt(est.interval?.upper, 1)}.`);
    } catch (err) {
      for (const fe of err?.fieldErrors || []) {
        const prefix = fe.path[0] === "reference" ? "mr" : "mf";
        const input = document.getElementById(`${prefix}-${fe.field}`);
        if (input) setFieldError(input, fe.message);
      }
      mount(el.manualResult, errorState(err, { compact: true }));
    } finally {
      button.disabled = false;
      el.manualResult.setAttribute("aria-busy", "false");
    }
  }

  function prefillManual() {
    if (!state.cycle) {
      el.manualError.textContent = "Load a demo cycle first.";
      el.manualError.hidden = false;
      return;
    }
    clearManualErrors();
    const model = selectedModel();
    let reference = null;
    if (model?.mode === "relative") {
      try {
        reference = referenceFor(state.cycle.battery, model);
      } catch (err) {
        mount(el.manualResult, errorState(err, { compact: true }));
      }
    }
    let filled = 0;
    for (const container of el.manualFields.querySelectorAll("[data-feature]")) {
      const name = container.dataset.feature;
      const v = state.cycle.features[name];
      const input = document.getElementById(`mf-${name}`);
      input.value = isNum(v) ? String(Number(v.toPrecision(6))) : "";
      if (isNum(v)) filled += 1;
      const refInput = document.getElementById(`mr-${name}`);
      if (refInput) refInput.value = reference && isNum(reference[name]) ? String(Number(reference[name].toPrecision(6))) : "";
    }
    announce(`Filled ${filled} indicators from ${state.cycle.battery.id} cycle ${state.cycle.cycle}.`);
  }

  el.manualForm.addEventListener("submit", submitManual);
  el.manualForm.addEventListener("reset", () => {
    clearManualErrors();
    mount(el.manualResult);
  });
  el.prefill.addEventListener("click", prefillManual);

  return { renderManualFields };
}
