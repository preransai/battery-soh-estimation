// Estimate tab, raw curves: the charge and discharge voltage of the loaded demo cycle with the
// measurement window shaded, a note with the window's indicators below each chart, and an
// accessible text summary of each chart.

import { chartMessage, chartsAvailable, voltageConfig } from "./charts.js";
import { measurementWindow } from "./estimate-data.js";
import { fmt, fmtFeatureWithUnit } from "./format.js";
import { isNum } from "./guards.js";

const WINDOWS = {
  charge: { levels: [3.9, 4.19], label: "Window 3.90\u20134.19 V" },
  discharge: { levels: [3.8, 3.5], label: "Window 3.80\u20133.50 V" },
};

/**
 * Draw both curves of a demo cycle (from `normalizeCycle`).
 * @param {object} cycle
 * @param {{chargeBox: HTMLElement, dischargeBox: HTMLElement, chargeNote: HTMLElement, dischargeNote: HTMLElement}} el
 * @param {{charge: object, discharge: object}} charts Chart handles from `makeChart`.
 */
export function renderCurves(cycle, el, charts) {
  for (const kind of ["charge", "discharge"]) {
    const box = kind === "charge" ? el.chargeBox : el.dischargeBox;
    const note = kind === "charge" ? el.chargeNote : el.dischargeNote;
    box.classList.remove("is-loading");
    const seg = cycle[kind];
    const f = cycle.features;
    if (!seg) {
      charts[kind].set(null);
      chartMessage(box, kind === "charge" ? "No charge record precedes this discharge." : "No discharge record for this cycle.");
      note.textContent = "";
      continue;
    }
    if (!chartsAvailable()) {
      chartMessage(box, "Charts are unavailable (the charting library did not load).");
    } else {
      chartMessage(box, "");
    }
    const window = measurementWindow(seg, kind);
    const { levels, label } = WINDOWS[kind];
    charts[kind].set((pal) => voltageConfig(pal, { seg, kind, window, levels, label }));
    const t = seg.time_s;
    const v = seg.voltage_v;
    if (kind === "charge") {
      note.textContent = isNum(f.chg_cc_time_390_419_s)
        ? `CC time 3.90\u20134.19 V: ${fmtFeatureWithUnit("chg_cc_time_390_419_s", f.chg_cc_time_390_419_s)} \u00b7 CV phase: ${fmtFeatureWithUnit("chg_cv_time_s", f.chg_cv_time_s)}`
        : "Window not reached: the charge must start below 3.85 V to measure 3.90\u20134.19 V.";
    } else {
      note.textContent = isNum(f.dis_q_380_350_ah)
        ? `Q 3.80\u20133.50 V: ${fmtFeatureWithUnit("dis_q_380_350_ah", f.dis_q_380_350_ah)} in ${fmtFeatureWithUnit("dis_time_380_350_s", f.dis_time_380_350_s)} \u00b7 R\u2080: ${fmtFeatureWithUnit("dis_r0_ohm", f.dis_r0_ohm)}`
        : "Window not reached: the loaded voltage did not cross 3.80\u20133.50 V.";
    }
    const span = window ? ` The ${label.toLowerCase()} spans ${fmt(window.to - window.from, 0)} seconds.` : " The measurement window is not reached.";
    box.querySelector("canvas").setAttribute("aria-label",
      `${kind === "charge" ? "Charge" : "Discharge"} voltage from ${fmt(v[0], 2)} V to ${fmt(v[v.length - 1], 2)} V over ${fmt(t[t.length - 1] - t[0], 0)} seconds.${span}`);
  }
}
