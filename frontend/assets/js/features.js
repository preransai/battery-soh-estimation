// Display metadata for model inputs. Mirrors src/battery_soh/features/registry.py (names, units,
// plausible ranges); the API remains the source of truth for validation. Pure module (no DOM),
// importable from Node for the static-backend self-check.

/** @typedef {{label: string, short: string, unit: string, family: string, digits: number,
 *   range: [number, number], window?: string, description: string, leaky?: boolean}} FeatureMeta */

/** @type {Readonly<Record<string, FeatureMeta>>} */
export const FEATURES = Object.freeze({
  chg_cc_time_390_419_s: {
    label: "CC charge time 3.90\u20134.19 V", short: "CC time 3.90\u20134.19 V", unit: "s",
    family: "charge", digits: 0, range: [0, 20000], window: "3.90\u20134.19 V",
    description: "Time for the voltage to rise from 3.90 V to 4.19 V during constant-current charge.",
  },
  chg_cc_time_400_419_s: {
    label: "CC charge time 4.00\u20134.19 V", short: "CC time 4.00\u20134.19 V", unit: "s",
    family: "charge", digits: 0, range: [0, 20000], window: "4.00\u20134.19 V",
    description: "Time for the voltage to rise from 4.00 V to 4.19 V during constant-current charge.",
  },
  chg_cv_time_s: {
    label: "CV phase duration", short: "CV duration", unit: "s",
    family: "charge", digits: 0, range: [0, 50000], window: "4.19 V until I < 0.1 A",
    description: "Duration of the constant-voltage phase (V \u2265 4.19 V until current < 0.1 A).",
  },
  chg_ica_peak_height_ah_per_v: {
    label: "ICA peak height", short: "ICA peak height", unit: "Ah/V",
    family: "charge", digits: 2, range: [0, 100], window: "3.90\u20134.18 V",
    description: "Height of the incremental-capacity (dQ/dV) peak on the CC charge segment.",
  },
  chg_ica_peak_voltage_v: {
    label: "ICA peak voltage", short: "ICA peak voltage", unit: "V",
    family: "charge", digits: 3, range: [3.0, 4.5], window: "3.90\u20134.18 V",
    description: "Voltage at which the incremental-capacity peak occurs.",
  },
  chg_temp_rise_cc_c: {
    label: "Temperature rise, CC charge window", short: "Temp. rise (charge)", unit: "\u00b0C",
    family: "charge", digits: 2, range: [-20, 40], window: "3.90\u20134.19 V",
    description: "Cell temperature rise between 3.90 V and 4.19 V in the CC charge phase.",
  },
  dis_q_380_350_ah: {
    label: "Charge delivered 3.80\u20133.50 V", short: "Q 3.80\u20133.50 V", unit: "Ah",
    family: "discharge", digits: 3, range: [0, 5], window: "3.80\u20133.50 V",
    description: "Charge delivered while the loaded voltage falls from 3.80 V to 3.50 V.",
  },
  dis_time_380_350_s: {
    label: "Discharge window duration", short: "Window duration", unit: "s",
    family: "discharge", digits: 0, range: [0, 20000], window: "3.80\u20133.50 V",
    description: "Duration of the 3.80 V to 3.50 V discharge window.",
  },
  dis_r0_ohm: {
    label: "Ohmic resistance at load-on", short: "R\u2080 at load-on", unit: "\u03a9",
    family: "discharge", digits: 4, range: [0, 2], window: "load-on step",
    description: "Ohmic resistance from the voltage step when the load is applied (\u0394V/\u0394I).",
  },
  dis_temp_rise_win_c: {
    label: "Temperature rise, discharge window", short: "Temp. rise (discharge)", unit: "\u00b0C",
    family: "discharge", digits: 2, range: [-20, 40], window: "3.80\u20133.50 V",
    description: "Cell temperature rise inside the 3.80 V to 3.50 V discharge window.",
  },
  eis_re_ohm: {
    label: "Electrolyte resistance (EIS)", short: "Re (EIS)", unit: "\u03a9",
    family: "impedance", digits: 4, range: [0, 2], window: "latest EIS before the cycle",
    description: "Electrolyte resistance from the most recent impedance measurement before the cycle.",
  },
  eis_rct_ohm: {
    label: "Charge-transfer resistance (EIS)", short: "Rct (EIS)", unit: "\u03a9",
    family: "impedance", digits: 4, range: [0, 2], window: "latest EIS before the cycle",
    description: "Charge-transfer resistance from the most recent impedance measurement before the cycle.",
  },
  ctx_cycle_index: {
    label: "Cycle index", short: "Cycle index", unit: "cycles",
    family: "context", digits: 0, range: [0, 10000],
    description: "Zero-based count of discharge cycles completed by the cell.",
  },
  ctx_ambient_c: {
    label: "Ambient temperature", short: "Ambient", unit: "\u00b0C",
    family: "context", digits: 1, range: [-40, 80],
    description: "Ambient (chamber) temperature of the test protocol.",
  },
  ctx_discharge_current_a: {
    label: "Discharge current", short: "Discharge current", unit: "A",
    family: "context", digits: 3, range: [0, 20],
    description: "Median absolute current under load during discharge.",
  },
  leg_capacity_ah: {
    label: "Full-discharge capacity (v1)", short: "Capacity (v1)", unit: "Ah",
    family: "legacy", digits: 3, range: [0, 5], leaky: true,
    description: "v1 input: the numerator of the SOH label (target leakage).",
  },
  leg_record_time_s: {
    label: "Record length (v1)", short: "Record length (v1)", unit: "s",
    family: "legacy", digits: 0, range: [0, 20000], leaky: true,
    description: "v1 'discharge_time': record length including the rest tail (a capacity proxy).",
  },
  leg_current_mean_a: {
    label: "Whole-record mean current (v1)", short: "Mean current (v1)", unit: "A",
    family: "legacy", digits: 3, range: [0, 20], leaky: true,
    description: "v1 whole-record mean |I| (a capacity proxy).",
  },
  leg_voltage_mean_v: {
    label: "Whole-record mean voltage (v1)", short: "Mean voltage (v1)", unit: "V",
    family: "legacy", digits: 3, range: [0, 5], leaky: true,
    description: "v1 whole-record mean voltage (needs the full discharge).",
  },
  leg_temp_max_c: {
    label: "Whole-record max temperature (v1)", short: "Max temp. (v1)", unit: "\u00b0C",
    family: "legacy", digits: 2, range: [-40, 100], leaky: true,
    description: "v1 whole-record maximum temperature (needs the full discharge).",
  },
});

/** Registry order, used to sort feature lists consistently. */
export const FEATURE_ORDER = Object.freeze(Object.keys(FEATURES));

export const FAMILY_LABELS = Object.freeze({
  charge: "Partial CC charge",
  discharge: "Partial discharge window",
  impedance: "Impedance (EIS)",
  context: "Context",
  legacy: "v1 inputs (leaky, audit only)",
  other: "Other",
});

/**
 * Metadata for a feature, with a readable fallback for names this build does not know.
 * @param {string} name
 * @returns {FeatureMeta}
 */
export function featureMeta(name) {
  const known = Object.prototype.hasOwnProperty.call(FEATURES, name) ? FEATURES[name] : null;
  if (known) return known;
  return {
    label: name, short: name, unit: "", family: "other", digits: 3,
    range: [-Infinity, Infinity], description: "",
  };
}

/** @param {string} name */
export function isKnownFeature(name) {
  return Object.prototype.hasOwnProperty.call(FEATURES, name);
}

/** @param {string} name */
export function isLeaky(name) {
  return Boolean(featureMeta(name).leaky);
}

/**
 * Sort feature names into registry order (unknown names last, alphabetically).
 * @param {Iterable<string>} names
 */
export function sortFeatures(names) {
  const rank = (n) => {
    const i = FEATURE_ORDER.indexOf(n);
    return i === -1 ? FEATURE_ORDER.length : i;
  };
  return [...names].sort((a, b) => rank(a) - rank(b) || a.localeCompare(b));
}
