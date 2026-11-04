// Estimate tab, data helpers (no DOM): tolerant readers of the demo-cell and model payloads, the
// commissioning reference a relative model needs, and the signal helpers that locate the
// measurement windows on a downsampled curve.

import { ApiError } from "./api.js";
import { featureMeta, isKnownFeature, isLeaky, sortFeatures } from "./features.js";
import { isNum, isObj } from "./guards.js";

// ------------------------------------------------------------------ payload normalisation
const pickArray = (obj, ...keys) => keys.map((k) => obj?.[k]).find(Array.isArray) || null;

function normalizeSegment(raw) {
  if (!raw || typeof raw !== "object") return null;
  const seg = {
    time_s: pickArray(raw, "time_s", "time", "t"),
    voltage_v: pickArray(raw, "voltage_v", "voltage", "v"),
    current_a: pickArray(raw, "current_a", "current", "i"),
    temperature_c: pickArray(raw, "temperature_c", "temperature", "temp") || [],
  };
  if (!seg.time_s || !seg.voltage_v || !seg.current_a) return null;
  const n = seg.time_s.length;
  if (n < 2 || seg.voltage_v.length !== n || seg.current_a.length !== n) return null;
  if (seg.temperature_c.length !== n) seg.temperature_c = new Array(n).fill(Number.NaN);
  return seg;
}

/** A demo cycle (GET /v1/demo/batteries/{id}/cycles/{cycle}): raw segments, indicators, true SOH. */
export function normalizeCycle(raw) {
  const segs = raw?.segments && typeof raw.segments === "object" ? raw.segments : raw?.raw && typeof raw.raw === "object" ? raw.raw : raw;
  const features = {};
  for (const [k, v] of Object.entries(raw?.features || {})) {
    if (isKnownFeature(k)) features[k] = isNum(v) ? v : null;
  }
  const soh = [raw?.soh_pct, raw?.true_soh_pct, raw?.soh].find(isNum);
  return {
    batteryId: raw?.battery_id,
    cycle: raw?.cycle_index,
    soh: isNum(soh) ? soh : null,
    charge: normalizeSegment(segs?.charge),
    discharge: normalizeSegment(segs?.discharge),
    features,
  };
}

/** Finite, non-leaky health-indicator values of a commissioning reference (context features need none). */
function normalizeReference(raw) {
  const ref = {};
  if (!isObj(raw)) return ref;
  for (const [k, v] of Object.entries(raw)) {
    if (isKnownFeature(k) && !isLeaky(k) && !k.startsWith("ctx_") && isNum(v)) ref[k] = v;
  }
  return ref;
}

/** The demo cells (GET /v1/demo/batteries) with their cycles in order and their reference. */
export function normalizeBatteries(raw) {
  const list = Array.isArray(raw) ? raw : Array.isArray(raw?.batteries) ? raw.batteries : [];
  return list
    .filter((b) => b && typeof b.battery_id === "string")
    .map((b) => ({
      id: b.battery_id,
      nCycles: isNum(b.n_cycles) ? b.n_cycles : null,
      // Commissioning reference computed by the server exactly like training (mean of each
      // indicator over the cell's first QC-valid cycles); relative models divide by it.
      reference: normalizeReference(b.reference),
      cycles: (Array.isArray(b.cycles) ? b.cycles : [])
        .map((c) => (isNum(c) ? { cycle: c, soh: null } : { cycle: c?.cycle_index, soh: isNum(c?.soh_pct) ? c.soh_pct : null }))
        .filter((c) => Number.isInteger(c.cycle))
        .sort((a, b) => a.cycle - b.cycle),
    }))
    .filter((b) => b.cycles.length);
}

/** The served models (GET /v1/models): id, feature set, mode, inputs and LOBO MAE. */
export function normalizeModels(raw) {
  const list = Array.isArray(raw) ? raw : Array.isArray(raw?.models) ? raw.models : [];
  return list
    .filter((m) => m && typeof m.model_id === "string")
    .map((m) => ({
      id: m.model_id,
      featureSet: m.feature_set || "",
      mode: m.mode === "relative" ? "relative" : "absolute",
      features: sortFeatures((Array.isArray(m.features) ? m.features : []).map((f) => (typeof f === "string" ? f : f?.name)).filter(Boolean)),
      mae: isNum(m.metrics?.lobo_mae) ? m.metrics.lobo_mae : null,
    }));
}

/**
 * Commissioning reference of a demo cell for a relative model: the per-indicator reference
 * published with GET /v1/demo/batteries (computed by the server exactly as in training),
 * restricted to the model's inputs. Throws when an input has no usable reference.
 */
export function referenceFor(battery, model) {
  const ref = battery?.reference || {};
  const needed = (model?.features || []).filter((f) => !f.startsWith("ctx_"));
  const missing = needed.filter((f) => !isNum(ref[f]) || ref[f] === 0);
  if (missing.length) {
    throw new ApiError({
      status: 0,
      title: "No commissioning reference",
      detail: `Demo cell ${battery?.id ?? ""} has no reference value for ${missing.map((f) => featureMeta(f).short).join(", ")}, which the relative model ${model.id} needs.`,
    });
  }
  return Object.fromEntries(needed.map((f) => [f, ref[f]]));
}

// ------------------------------------------------------------------ signal helpers
function crossing(t, v, level, rising, from) {
  for (let i = Math.max(from, 0); i < v.length; i += 1) {
    if (rising ? v[i] >= level : v[i] <= level) {
      if (i === from || i === 0) return t[i];
      const v0 = v[i - 1];
      const v1 = v[i];
      return v1 === v0 ? t[i] : t[i - 1] + ((level - v0) * (t[i] - t[i - 1])) / (v1 - v0);
    }
  }
  return null;
}

export const median = (arr) => {
  const s = [...arr].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

/** Approximate time span of the measurement window on the (downsampled) curve, for shading only. */
export function measurementWindow(seg, kind) {
  const { time_s: t, voltage_v: v, current_a: c } = seg;
  if (kind === "charge") {
    const k0 = c.findIndex((x) => x > 0.5);
    if (k0 < 0 || v[k0] >= 3.85) return null;
    const a = crossing(t, v, 3.9, true, k0);
    const b = crossing(t, v, 4.19, true, k0);
    return a !== null && b !== null && b > a ? { from: a, to: b } : null;
  }
  const loaded = c.map(Math.abs).filter((x) => x > 0.1);
  if (loaded.length < 5) return null;
  const iLoad = median(loaded);
  const k0 = c.findIndex((x) => Math.abs(x) > 0.5 * iLoad);
  if (k0 < 0 || v[k0] < 3.8) return null;
  const a = crossing(t, v, 3.8, false, k0);
  const b = crossing(t, v, 3.5, false, k0);
  return a !== null && b !== null && b > a ? { from: a, to: b } : null;
}
