// Reusable presentational pieces shared by the views: estimate summary (big number, interval
// bar, truth/error, domain badge, warnings), indicator tables and contribution helpers.

import { FAMILY_LABELS, featureMeta, isLeaky, sortFeatures } from "./features.js";
import { fmt, fmtFeature, fmtLevel, fmtSigned } from "./format.js";
import { isNum } from "./guards.js";
import { dataTable, h, setVar, tag } from "./dom.js";

const EOL_SOH = 80;

/** Clamp a value into [0, 100] percent of a domain. */
function position(value, lo, hi) {
  return `${Math.min(100, Math.max(0, ((value - lo) / (hi - lo)) * 100)).toFixed(3)}%`;
}

/** Axis domain for the interval bar: covers the interval, estimate and truth, in steps of 5. */
function barDomain(values) {
  const finite = values.filter(isNum);
  let lo = Math.floor((Math.min(...finite) - 4) / 5) * 5;
  let hi = Math.ceil((Math.max(...finite) + 4) / 5) * 5;
  lo = Math.max(0, lo);
  hi = Math.min(120, hi);
  if (hi - lo < 20) {
    const pad = (20 - (hi - lo)) / 2;
    lo = Math.max(0, Math.floor((lo - pad) / 5) * 5);
    hi = Math.min(120, Math.ceil((hi + pad) / 5) * 5);
  }
  return [lo, hi];
}

/**
 * Horizontal interval bar with estimate tick, optional true-SOH marker and the 80 % EOL line.
 * Positions are CSS custom properties set through the CSSOM (CSP-safe).
 */
export function intervalBar({ estimate, lower, upper, truth = null, level }) {
  const [lo, hi] = barDomain([estimate, lower, upper, truth]);
  const track = h("div", { class: "ibar-track" });
  const range = h("div", { class: "ibar-range" });
  setVar(range, "--from", position(lower, lo, hi));
  setVar(range, "--to", position(upper, lo, hi));
  track.append(range);
  if (EOL_SOH > lo && EOL_SOH < hi) {
    const eol = h("div", { class: "ibar-eol" });
    setVar(eol, "--at", position(EOL_SOH, lo, hi));
    track.append(eol);
  }
  const point = h("div", { class: "ibar-point" });
  setVar(point, "--at", position(estimate, lo, hi));
  track.append(point);
  if (isNum(truth)) {
    const t = h("div", { class: "ibar-true" });
    setVar(t, "--at", position(truth, lo, hi));
    track.append(t);
  }
  const ticks = h("div", { class: "ibar-axis", attrs: { "aria-hidden": "true" } });
  const step = hi - lo > 40 ? 20 : hi - lo > 20 ? 10 : 5;
  for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) {
    const tick = h("span", { class: "ibar-tick", text: String(v) });
    setVar(tick, "--at", position(v, lo, hi));
    ticks.append(tick);
  }
  const label = `${level} interval ${fmt(lower, 1)} to ${fmt(upper, 1)} percent, estimate ${fmt(estimate, 1)}`
    + (isNum(truth) ? `, true SOH ${fmt(truth, 1)}` : "");
  const legend = h("ul", { class: "ibar-legend", attrs: { "aria-hidden": "true" } },
    h("li", null, h("span", { class: "key key--range" }), `${level} interval`),
    h("li", null, h("span", { class: "key key--point" }), "Estimate"),
    isNum(truth) ? h("li", null, h("span", { class: "key key--true" }), "True SOH") : null,
    EOL_SOH > lo && EOL_SOH < hi ? h("li", null, h("span", { class: "key key--eol" }), "80% end of life") : null);
  return h("div", { class: "ibar", attrs: { role: "img", "aria-label": label } }, track, ticks, legend);
}

/** Domain-of-applicability badge plus a one-line reason when outside. */
export function domainBadge(domain) {
  if (!domain || typeof domain.in_domain !== "boolean") return tag("Domain check unavailable", "neutral");
  if (domain.in_domain) {
    return h("span", { class: "badge badge--good" }, h("span", { class: "badge-dot", attrs: { "aria-hidden": "true" } }), "In training domain");
  }
  return h("span", { class: "badge badge--warn" }, h("span", { class: "badge-dot", attrs: { "aria-hidden": "true" } }), "Outside training domain");
}

function domainDetail(domain) {
  if (!domain) return null;
  const parts = [];
  if (Array.isArray(domain.out_of_range) && domain.out_of_range.length) {
    parts.push(`Outside the training range: ${domain.out_of_range.map((f) => featureMeta(f).short).join(", ")}.`);
  }
  if (isNum(domain.mahalanobis) && isNum(domain.threshold)) {
    const far = domain.mahalanobis > domain.threshold;
    parts.push(`Mahalanobis distance ${fmt(domain.mahalanobis, 2)} ${far ? ">" : "\u2264"} ${fmt(domain.threshold, 2)} (threshold from the training cycles).`);
  }
  return parts.length ? h("p", { class: "domain-detail", text: parts.join(" ") }) : null;
}

/** Bullet list of server warnings. */
export function warningsList(warnings) {
  if (!Array.isArray(warnings) || !warnings.length) return null;
  return h("ul", { class: "warnings", attrs: { "aria-label": "Warnings" } }, warnings.map((w) => h("li", { text: String(w) })));
}

/**
 * Estimate summary block.
 * @param {object} est Response of POST /v1/soh/estimate.
 * @param {{truth?: number|null, compact?: boolean, source?: string, afterTruth?: Array<Node|null>}} [opts]
 *   afterTruth: nodes placed right after the true-SOH row (notes that qualify that error).
 */
export function estimateSummary(est, { truth = null, compact = false, source = "", afterTruth = [] } = {}) {
  const level = fmtLevel(est?.interval?.confidence ?? 0.9);
  const soh = est?.soh_pct;
  const lower = est?.interval?.lower;
  const upper = est?.interval?.upper;
  const halfWidth = isNum(lower) && isNum(upper) ? (upper - lower) / 2 : NaN;
  const head = h("div", { class: "result-head" },
    h("p", { class: compact ? "soh-number soh-number--compact" : "soh-number" },
      h("span", { class: "soh-value", text: fmt(soh, 1) }), h("span", { class: "soh-unit", text: "%" })),
    domainBadge(est?.domain));
  const intervalLine = h("p", { class: "interval-line" },
    h("span", { class: "muted", text: `${level} interval ` }),
    h("strong", { class: "tnum", text: `${fmt(lower, 1)}\u2013${fmt(upper, 1)}%` }),
    isNum(halfWidth) ? h("span", { class: "muted tnum", text: ` (\u00b1${fmt(halfWidth, 1)} pts)` }) : null);
  const out = h("div", { class: "estimate-summary" }, source ? h("p", { class: "result-source", text: source }) : null, head, intervalLine);
  if (isNum(soh) && isNum(lower) && isNum(upper)) out.append(intervalBar({ estimate: soh, lower, upper, truth, level }));
  if (isNum(truth)) {
    const err = soh - truth;
    const inside = isNum(lower) && truth >= lower && truth <= upper;
    out.append(h("dl", { class: "truth-row" },
      h("div", null, h("dt", { text: "True SOH" }), h("dd", { class: "tnum", text: `${fmt(truth, 1)}%` })),
      h("div", null, h("dt", { text: "Error" }), h("dd", { class: "tnum", text: `${fmtSigned(err, 2)} pts` })),
      h("div", null, h("dt", { text: "Covered" }), h("dd", null, inside ? tag("Inside interval", "good") : tag("Outside interval", "warn")))));
    for (const node of afterTruth) if (node) out.append(node);
  }
  const detail = est?.domain && !est.domain.in_domain ? domainDetail(est.domain) : null;
  if (detail) out.append(detail);
  const warnings = warningsList(est?.warnings);
  if (warnings) out.append(warnings);
  out.append(h("p", { class: "model-line" },
    h("span", { class: "muted", text: "Model " }), h("code", { text: est?.model_id ?? "\u2014" }),
    isNum(est?.base_value) ? h("span", { class: "muted", text: ` \u00b7 base value ${fmt(est.base_value, 1)}%` }) : null));
  return out;
}

/**
 * "In-sample" caveat for an estimate on a demo cycle: the shipped models were trained on every
 * demo cell, so the error shown above it is not a test result.
 * @param {() => void} [onShowBenchmark] Opens the Benchmark tab.
 */
export function inSampleNote(onShowBenchmark) {
  const target = onShowBenchmark
    ? h("button", { type: "button", class: "link-btn", text: "Benchmark tab", on: { click: onShowBenchmark } })
    : "Benchmark tab";
  return h("p", { class: "insample-note" },
    h("strong", { text: "In-sample: " }),
    "the shipped models were trained on all four demo cells. Held-out accuracy: ", target, ".");
}

/**
 * Leave-one-battery-out estimate of the headline specification for one demo cycle (the model
 * of that benchmark fold never saw this cell).
 * @param {{yhat: number, lower: number|null, upper: number|null, level: number, truth: number|null}} p
 */
export function heldOutBlock({ yhat, lower, upper, level, truth }) {
  const levelText = fmtLevel(level);
  const hasInterval = isNum(lower) && isNum(upper);
  const rows = [
    h("div", null, h("dt", { text: "Estimate" }), h("dd", { class: "tnum", text: `${fmt(yhat, 1)}%` })),
    hasInterval ? h("div", null, h("dt", { text: `${levelText} interval` }), h("dd", { class: "tnum", text: `${fmt(lower, 1)}\u2013${fmt(upper, 1)}%` })) : null,
  ];
  if (isNum(truth)) {
    rows.push(h("div", null, h("dt", { text: "Error" }), h("dd", { class: "tnum", text: `${fmtSigned(yhat - truth, 2)} pts` })));
    if (hasInterval) {
      const inside = truth >= lower && truth <= upper;
      rows.push(h("div", null, h("dt", { text: "Covered" }), h("dd", null, inside ? tag("Inside interval", "good") : tag("Outside interval", "warn"))));
    }
  }
  return h("section", { class: "heldout", attrs: { "aria-label": "Held-out estimate" } },
    h("p", { class: "heldout-title", text: "Held-out estimate (model trained without this cell)" }),
    h("dl", { class: "truth-row heldout-row" }, rows),
    h("p", { class: "heldout-note", text: "Leave-one-battery-out prediction of the headline specification for this cycle, from the benchmark." }));
}

/**
 * Contributions sorted by |contribution| with display labels and tooltips.
 * @param {object} est Response of POST /v1/soh/estimate.
 * @param {{relative?: boolean}} [opts] relative: the estimate comes from a relative-mode model,
 *   whose inputs are ratios to the cell's commissioning reference (context features excepted).
 */
export function contributionItems(est, { relative = false } = {}) {
  const list = Array.isArray(est?.contributions) ? est.contributions : [];
  return list
    .filter((c) => c && typeof c.feature === "string" && isNum(c.contribution))
    .map((c) => {
      const meta = featureMeta(c.feature);
      const ratio = relative && meta.family !== "context";
      let valueText = "\u2014";
      if (isNum(c.value)) {
        valueText = ratio
          ? `${fmt(c.value, 3)} \u00d7 reference`
          : `${fmtFeature(c.feature, c.value)}${meta.unit ? ` ${meta.unit}` : ""}`;
      }
      const imputed = Array.isArray(est?.domain?.missing) && est.domain.missing.includes(c.feature);
      return {
        feature: c.feature,
        label: meta.short,
        contribution: c.contribution,
        detail: ` Input ${valueText}${imputed ? " (imputed training median)" : ""}`,
      };
    })
    .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));
}

/** Compact "top drivers" list (text) for inline results. */
export function topDrivers(est, n = 3) {
  const items = contributionItems(est).slice(0, n);
  if (!items.length) return null;
  return h("div", { class: "drivers" },
    h("p", { class: "drivers-title", text: "Largest contributions" }),
    h("ul", { class: "drivers-list" }, items.map((d) => h("li", null,
      h("span", { class: "drivers-name", text: d.label }),
      h("span", { class: ["drivers-value", "tnum", d.contribution >= 0 ? "pos" : "neg"], text: `${fmtSigned(d.contribution, 2)} pts` })))));
}

const MISSING_TEXT = {
  charge: ["not reached", "The charge window was not reached in this record"],
  discharge: ["not reached", "The discharge window was not reached in this record"],
  impedance: ["not measured", "No impedance (EIS) measurement precedes this cycle"],
};

/**
 * Table of extracted health indicators grouped by family.
 * @param {Record<string, number|null>} features
 * @param {{used?: Set<string>|null, caption?: string, families?: string[]|null}} [opts]
 */
export function indicatorTable(features, { used = null, caption = "Extracted health indicators", families = null } = {}) {
  const names = sortFeatures(Object.keys(features || {})).filter((n) => !isLeaky(n) && (!families || families.includes(featureMeta(n).family)));
  const rows = [];
  let family = null;
  for (const name of names) {
    const meta = featureMeta(name);
    if (meta.family !== family) {
      family = meta.family;
      rows.push({ __group: FAMILY_LABELS[family] || family });
    }
    rows.push({ name, value: features[name], unit: meta.unit, window: meta.window || "\u2014" });
  }
  const columns = [
    {
      key: "name", label: "Indicator",
      format: (name) => h("span", { class: "hi-name" }, h("span", { text: featureMeta(name).label }), h("code", { class: "hi-code", text: name })),
    },
    {
      key: "value", label: "Value", num: true,
      format: (v, row) => {
        if (isNum(v)) return h("span", null, fmtFeature(row.name, v), row.unit ? h("span", { class: "cell-unit", text: ` ${row.unit}` }) : null);
        const [text, title] = MISSING_TEXT[featureMeta(row.name).family] || ["not available", "Not available for this record"];
        return h("span", { class: "muted", text, attrs: { title } });
      },
    },
  ];
  if (used) {
    columns.push({
      key: "name", label: "In model", rowHeader: false, title: "Input of the model that produced the estimate",
      format: (name) => (used.has(name) ? h("span", { class: "check" }, h("span", { attrs: { "aria-hidden": "true" }, text: "\u2713" }), h("span", { class: "visually-hidden", text: "yes" })) : h("span", { class: "muted" }, h("span", { attrs: { "aria-hidden": "true" }, text: "\u2014" }), h("span", { class: "visually-hidden", text: "no" }))),
    });
  }
  return dataTable(columns, rows, { caption, captionHidden: true, label: caption });
}
