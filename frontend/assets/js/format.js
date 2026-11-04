// Number formatting helpers. Every formatter returns an em dash for missing/non-finite values.

import { featureMeta } from "./features.js";
import { isNum } from "./guards.js";

const DASH = "\u2014";
const MINUS = "\u2212";
const cache = new Map();

function formatter(min, max) {
  const key = `${min}:${max}`;
  if (!cache.has(key)) {
    cache.set(key, new Intl.NumberFormat("en-US", { minimumFractionDigits: min, maximumFractionDigits: max }));
  }
  return cache.get(key);
}

/** Fixed-decimals number with thousands separators and a typographic minus. */
export function fmt(value, digits = 2) {
  if (!isNum(value)) return DASH;
  const out = formatter(digits, digits).format(value);
  return out.replace("-", MINUS);
}

/** Signed number: "+1.23" / "\u22120.45". */
export function fmtSigned(value, digits = 2) {
  if (!isNum(value)) return DASH;
  const rounded = Number(value.toFixed(digits));
  const body = formatter(digits, digits).format(Math.abs(value));
  if (rounded === 0) return body;
  return (value > 0 ? "+" : MINUS) + body;
}

/** Fraction (0.903) as a percentage string ("90.3%"). Values > 1.5 are treated as percents. */
export function fmtPct(fraction, digits = 1) {
  if (!isNum(fraction)) return DASH;
  const pct = Math.abs(fraction) > 1.5 ? fraction : fraction * 100;
  return `${fmt(pct, digits)}%`;
}

/** Confidence level (0.9) as "90%". */
export function fmtLevel(level) {
  if (!isNum(level)) return DASH;
  const pct = level * 100;
  return `${Number.isInteger(Math.round(pct * 10) / 10) ? Math.round(pct) : pct.toFixed(1)}%`;
}

/** A feature value with the registry's display precision (no unit). */
export function fmtFeature(name, value) {
  return fmt(value, featureMeta(name).digits);
}

/** A feature value with its unit, e.g. "2,987 s". */
export function fmtFeatureWithUnit(name, value) {
  if (!isNum(value)) return DASH;
  const { unit } = featureMeta(name);
  return unit ? `${fmtFeature(name, value)} ${unit}` : fmtFeature(name, value);
}

/** Short hex digest for display ("3b8e0f6a\u2026"). */
export function shortHash(value, n = 10) {
  return typeof value === "string" && value.length > n ? `${value.slice(0, n)}\u2026` : value || DASH;
}

/** ISO timestamp as "2026-10-10 09:38 UTC" (no locale surprises). */
export function fmtUtc(iso) {
  if (typeof iso !== "string") return DASH;
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return `${d.toISOString().slice(0, 16).replace("T", " ")} UTC`;
}

/** Pretty model names for leaderboard rows. */
const MODEL_NAMES = {
  ridge: "Ridge", ridge_nfs: "Ridge, nested selection", pls: "PLS", elasticnet: "Elastic net", gpr: "Gaussian process", random_forest: "Random forest",
  hgb: "Gradient boosting", mlp: "MLP", knn: "k-NN", mean: "Mean predictor", cycle_only: "Cycle count only",
  capacity_oracle: "Capacity oracle",
};
export function modelName(id) {
  return (typeof id === "string" && MODEL_NAMES[id]) || id || DASH;
}

const SET_NAMES = {
  charge: "Charge HIs", discharge: "Discharge HIs", combined: "Charge + discharge", combined_eis: "Combined + EIS",
  charge_no_cv: "Charge HIs without CV", combined_no_cv: "Combined without CV",
  charge_ext: "Charge (extended)", combined_ext: "Combined (extended)", cycle_only: "Cycle index", baseline: "Baseline",
  legacy_v1: "v1 inputs (leaky)", legacy_v1_no_capacity: "v1 inputs without capacity (leaky)", capacity_oracle: "Capacity (leaky)",
};

const EXPERIMENT_NAMES = {
  core_lobo: "4 core cells, leave one battery out",
  extended_lobo: "12 cells, leave one battery out",
  leave_one_condition_out: "12 cells, leave one condition out",
};
/** Readable experiment names for the leaderboard selector. */
export function experimentName(id) {
  return (typeof id === "string" && EXPERIMENT_NAMES[id]) || (typeof id === "string" ? id.replace(/_/g, " ") : DASH);
}
export function featureSetName(id) {
  return (typeof id === "string" && SET_NAMES[id]) || id || DASH;
}
