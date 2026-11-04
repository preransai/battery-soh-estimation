// Estimate tab, CSV upload (POST /v1/soh/estimate-from-cycle): a raw charge or discharge record
// is parsed and summarised in the browser, then uploaded for indicator extraction on the server;
// the loaded demo cycle can be downloaded as a sample file. The upload needs the Python API, so
// it is disabled in the static demo (the sample download still works).

import { api, IS_STATIC, isAbort } from "./api.js";
import { estimateSummary, indicatorTable, topDrivers } from "./components.js";
import { CsvError, MAX_FILE_BYTES, describeSegment, parseCycleCsv, segmentToCsv } from "./csv.js";
import { announce, errorState, h, mount, skeleton } from "./dom.js";
import { median } from "./estimate-data.js";
import { featureMeta, isKnownFeature } from "./features.js";
import { fmt } from "./format.js";
import { isNum } from "./guards.js";

const MAX_BODY_BYTES = 1_000_000;

/**
 * Wire the CSV form; returns `updateCsvAvailability`, to call whenever the selected model changes.
 * @param {{el: object, state: object, confidence: () => number, selectedModel: () => object|null}} view
 */
export function createCsvUpload({ el, state, confidence, selectedModel }) {
  let parsed = null; // the parsed segment of the chosen file
  let csvSeq = 0; // file parsing and upload: a newer file or upload supersedes older ones
  let csvController = null;

  function updateCsvAvailability() {
    const relative = selectedModel()?.mode === "relative";
    let note = "";
    if (IS_STATIC) {
      note = "Uploading a raw cycle needs the Python API, because indicator extraction runs on the server. In this static demo you can still download a sample CSV; run the Docker image to try the upload.";
    } else if (relative) {
      note = "Relative models need the cell's commissioning reference, which a single upload cannot provide. Choose Auto or an absolute model to upload a cycle.";
    }
    el.csvNote.textContent = note;
    el.csvNote.hidden = !note;
    el.csvFile.disabled = IS_STATIC;
    el.csvSubmit.disabled = IS_STATIC || relative;
  }

  function csvFail(message) {
    el.csvError.textContent = message;
    el.csvError.hidden = !message;
  }

  /** A new file (or segment type) supersedes any parse or upload still in flight. */
  function supersedeCsv() {
    const token = ++csvSeq;
    if (csvController) csvController.abort();
    csvController = null;
    if (el.csvResult.getAttribute("aria-busy") === "true") {
      el.csvResult.setAttribute("aria-busy", "false");
      mount(el.csvResult);
    }
    updateCsvAvailability();
    return token;
  }

  async function onCsvFile() {
    const token = supersedeCsv();
    csvFail("");
    parsed = null;
    el.csvSummary.textContent = "";
    const file = el.csvFile.files && el.csvFile.files[0];
    if (!file) return;
    if (file.size > MAX_FILE_BYTES) {
      csvFail(`The file is ${fmt(file.size / 1048576, 1)} MB; the limit is ${fmt(MAX_FILE_BYTES / 1048576, 0)} MB.`);
      return;
    }
    try {
      const text = await file.text();
      if (token !== csvSeq) return; // another file was chosen meanwhile
      const seg = parseCycleCsv(text);
      const d = describeSegment(seg);
      parsed = seg;
      const kind = el.csvSegment.value;
      const medI = median(seg.current_a);
      const signHint = kind === "charge" && medI < 0 ? " Charge current looks negative; it should be positive." : "";
      el.csvSummary.textContent = `${fmt(d.rows, 0)} samples \u00b7 ${fmt(d.duration, 0)} s \u00b7 ${fmt(d.vStart, 3)} V \u2192 ${fmt(d.vEnd, 3)} V (range ${fmt(d.vMin, 2)}\u2013${fmt(d.vMax, 2)} V).${signHint}`;
    } catch (err) {
      if (token !== csvSeq) return;
      csvFail(err instanceof CsvError ? err.message : "The file could not be read.");
    }
  }

  async function submitCsv(event) {
    event.preventDefault();
    csvFail("");
    if (IS_STATIC) return;
    if (!parsed) {
      csvFail("Choose a CSV file first.");
      el.csvFile.focus();
      return;
    }
    const kind = el.csvSegment.value;
    const { rows, ...segment } = parsed;
    const body = { [kind]: segment, confidence: confidence() };
    const model = selectedModel();
    if (model) body.model_id = model.id;
    if (JSON.stringify(body).length > MAX_BODY_BYTES) {
      csvFail(`The request would exceed 1 MB (${fmt(rows, 0)} samples). Downsample the record and try again.`);
      return;
    }
    const token = ++csvSeq;
    if (csvController) csvController.abort();
    const controller = new AbortController();
    csvController = controller;
    el.csvSubmit.disabled = true;
    el.csvResult.setAttribute("aria-busy", "true");
    mount(el.csvResult, skeleton({ lines: 3, label: "Extracting indicators" }));
    try {
      const res = await api.estimateFromCycle(body, { signal: controller.signal });
      if (token !== csvSeq) return; // superseded by a newer file or upload
      const feats = {};
      for (const [k, v] of Object.entries(res?.features || {})) if (isKnownFeature(k) && featureMeta(k).family !== "context") feats[k] = isNum(v) ? v : null;
      const est = res?.estimate;
      mount(el.csvResult,
        estimateSummary(est, { compact: true, source: `Result for the uploaded ${kind}` }),
        topDrivers(est),
        h("details", { class: "details" }, h("summary", { text: "Extracted indicators" }),
          indicatorTable(feats, { used: new Set((est?.contributions || []).map((c) => c.feature)), caption: "Indicators extracted from the upload", families: ["charge", "discharge", "impedance"] })));
      announce(`Uploaded ${kind}: estimated SOH ${fmt(est?.soh_pct, 1)} percent.`);
    } catch (err) {
      if (token !== csvSeq || isAbort(err)) return;
      mount(el.csvResult, errorState(err, { compact: true }));
    } finally {
      if (token === csvSeq) {
        csvController = null;
        el.csvResult.setAttribute("aria-busy", "false");
        updateCsvAvailability();
      }
    }
  }

  function downloadSample() {
    csvFail("");
    const kind = el.csvSegment.value;
    const seg = state.cycle?.[kind];
    if (!seg) {
      csvFail(state.cycle ? `Demo cycle ${state.cycle.cycle} has no ${kind} record; pick another cycle.` : "Load a demo cycle first.");
      return;
    }
    const blob = new Blob([segmentToCsv(seg)], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = h("a", { href: url, download: `${state.cycle.battery.id}_cycle${String(state.cycle.cycle).padStart(3, "0")}_${kind}.csv`, hidden: true });
    document.body.append(a);
    a.click();
    a.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    announce(`Downloaded the ${kind} of ${state.cycle.battery.id} cycle ${state.cycle.cycle} as CSV.`);
  }

  el.csvFile.addEventListener("change", onCsvFile);
  el.csvSegment.addEventListener("change", () => {
    if (parsed) onCsvFile();
  });
  el.csvForm.addEventListener("submit", submitCsv);
  el.csvSample.addEventListener("click", downloadSample);

  return { updateCsvAvailability };
}
