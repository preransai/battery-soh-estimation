// Backend-free implementation of POST /v1/soh/estimate for the static (GitHub Pages) demo.
//
// It evaluates an exported model bundle (docs/MODEL_CARD.md#bundle-format) exactly like
// the Python runtime battery_soh.models.linear_bundle.LinearBundle:
//   impute missing inputs with the training median -> standardise -> linear model -> clip;
//   contributions coef_j * z_j (they sum with the base value to the unclipped prediction);
//   interval = prediction +/- conformal quantile, clipped to the output range;
//   domain check = per-feature [min, max] widened by margin * span (missing inputs skipped)
//   plus the Mahalanobis distance of the standardised input from domain.center under
//   domain.precision_matrix, compared with domain.mahalanobis_threshold.
// Errors are thrown as StaticBackendError(status, body) whose body is the server's RFC 7807
// problem document, including the 422 validation problem (same title, `detail` text and
// `errors` list as battery_soh.api.errors for the same request), so the API client
// normalises both modes identically. tests/e2e/test_static_backend_parity.py checks this
// against the API's schema and service code. Pure module: no DOM, no fetch; runs in Node.

import { FEATURE_ORDER, featureMeta, isKnownFeature, isLeaky } from "./features.js";
import { isNum, isObj } from "./guards.js";

export const SUPPORTED_CONFIDENCES = Object.freeze([0.8, 0.9, 0.95]);
/** Request fields in the API schema's order (battery_soh.api.schemas.EstimateRequest). */
const BODY_KEYS = Object.freeze(["features", "reference", "model_id", "confidence"]);
/** battery_soh.api.schemas.MODEL_ID_PATTERN */
const MODEL_ID_PATTERN = "^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$";
const MODEL_ID_RE = new RegExp(MODEL_ID_PATTERN);
/** Features a client may send, in registry order (battery_soh.api.schemas.INPUT_FEATURES). */
const INPUT_FEATURES = Object.freeze(FEATURE_ORDER.filter((name) => !isLeaky(name)));
const INPUT_SET = new Set(INPUT_FEATURES);
const LEAKY_SET = new Set(FEATURE_ORDER.filter((name) => isLeaky(name)));
// battery_soh.api.errors: error-list cap and truncation limits.
const MAX_REPORTED_ERRORS = 50;
const MAX_TEXT = 300;
const MAX_LOC_PART = 80;

export class StaticBackendError extends Error {
  /**
   * @param {number} status HTTP-equivalent status code.
   * @param {object} body Server-shaped error body.
   */
  constructor(status, body) {
    super(typeof body?.detail === "string" ? body.detail : `HTTP ${status}`);
    this.name = "StaticBackendError";
    this.status = status;
    this.body = body;
  }
}

function problem(status, title, detail, type = "about:blank") {
  return new StaticBackendError(status, { type, title, status, detail });
}

/** Context features are never divided by a reference (battery_soh.features.relative.is_context_feature). */
function isContextFeature(name) {
  return isKnownFeature(name) ? featureMeta(name).family === "context" : name.startsWith("ctx_");
}

/**
 * Canonical key of a confidence level, matching Python's f"{round(c, 6):g}" ("0.9", "0.95").
 * @param {number|string} confidence
 */
export function confidenceKey(confidence) {
  const value = Number(confidence);
  if (!(value > 0 && value < 1)) throw new RangeError(`confidence must be in (0, 1), got ${confidence}`);
  return String(Math.round(value * 1e6) / 1e6);
}

const clip = (text, limit) => (text.length <= limit ? text : `${text.slice(0, limit - 3)}...`);

/**
 * The API's 422 problem for a list of Pydantic-style errors ({type, loc, msg}), built exactly
 * like battery_soh.api.errors (sanitize_errors + summarize_errors): at most 50 errors, long
 * strings truncated, `detail` = the first three as "loc: msg" joined by "; " plus "(+N more)".
 * @param {Array<{type: string, loc: Array<string|number>, msg: string}>} raw
 */
export function validationProblem(raw) {
  const errors = raw.slice(0, MAX_REPORTED_ERRORS).map((e) => ({
    type: clip(String(e.type), 60),
    loc: e.loc.map((part) => (typeof part === "number" ? part : clip(String(part), MAX_LOC_PART))),
    msg: clip(String(e.msg), MAX_TEXT),
  }));
  const parts = errors.slice(0, 3).map((e) => {
    let loc = e.loc.map(String);
    if (loc.length && ["body", "query", "path", "header"].includes(loc[0])) loc = loc.slice(1);
    const where = loc.join(".");
    return where ? `${where}: ${e.msg}` : e.msg;
  });
  let detail = parts.join("; ");
  if (raw.length > 3) detail += ` (+${raw.length - 3} more)`;
  return new StaticBackendError(422, {
    type: "/problems/validation-error",
    title: "Invalid input",
    status: 422,
    detail: detail || "The request is invalid.",
    errors,
  });
}

/**
 * Validate a FeatureValues object like the API's model (battery_soh.api.schemas.FeatureValues):
 * label-side/proxy features (leg_*) are refused as a whole; then every registry input in
 * registry order (strict finite number within the plausible range, or null); then unknown keys.
 * Returns the accepted values, or null when the object itself is invalid.
 */
function checkFeatureValues(value, where, err) {
  if (!isObj(value)) {
    err([where], "model_attributes_type", "Input should be a valid dictionary or object to extract fields from");
    return null;
  }
  const present = Object.keys(value).filter((k) => value[k] !== undefined); // as JSON would send it
  const leaky = present.filter((k) => LEAKY_SET.has(k)).sort();
  if (leaky.length) {
    err([where], "value_error",
      `Value error, label-side/proxy feature(s) ${leaky.join(", ")} are not accepted (they need the full discharge and leak the SOH label)`);
    return null;
  }
  const out = {};
  for (const name of INPUT_FEATURES) {
    if (!present.includes(name)) continue;
    const v = value[name];
    if (v === null) {
      out[name] = null;
      continue;
    }
    if (typeof v !== "number") {
      err([where, name], "float_type", "Input should be a valid number");
      continue;
    }
    if (!Number.isFinite(v)) {
      err([where, name], "finite_number", "Input should be a finite number");
      continue;
    }
    const [lo, hi] = featureMeta(name).range;
    if (v < lo) {
      err([where, name], "greater_than_equal", `Input should be greater than or equal to ${lo}`);
      continue;
    }
    if (v > hi) {
      err([where, name], "less_than_equal", `Input should be less than or equal to ${hi}`);
      continue;
    }
    out[name] = v;
  }
  for (const name of present) {
    if (!INPUT_SET.has(name)) err([where, name], "extra_forbidden", "Extra inputs are not permitted");
  }
  return out;
}

/**
 * Validate an estimate request body exactly like POST /v1/soh/estimate (the API's Pydantic
 * model as FastAPI applies it: extra="forbid", strict finite floats, registry plausible
 * ranges, leaky features refused, model-id pattern, confidence literal). Throws the API's
 * 422 validation problem (see validationProblem).
 * @param {unknown} body
 * @returns {{features: Record<string, number|null>, reference: Record<string, number|null>|null,
 *   modelId: string|null, confidence: number}}
 */
export function validateEstimateBody(body) {
  const errors = [];
  const err = (loc, type, msg) => errors.push({ type, loc: ["body", ...loc], msg });
  if (body === null || body === undefined) {
    throw validationProblem([{ type: "missing", loc: ["body"], msg: "Field required" }]);
  }
  if (!isObj(body)) {
    throw validationProblem([{
      type: "model_attributes_type", loc: ["body"],
      msg: "Input should be a valid dictionary or object to extract fields from",
    }]);
  }
  const given = (key) => Object.prototype.hasOwnProperty.call(body, key) && body[key] !== undefined;
  let features = {};
  if (!given("features")) err(["features"], "missing", "Field required");
  else features = checkFeatureValues(body.features, "features", err) || {};
  let reference = null;
  if (given("reference") && body.reference !== null) {
    reference = checkFeatureValues(body.reference, "reference", err) || {};
  }
  let modelId = null;
  if (given("model_id") && body.model_id !== null) {
    if (typeof body.model_id !== "string") err(["model_id"], "string_type", "Input should be a valid string");
    else if (!MODEL_ID_RE.test(body.model_id)) {
      err(["model_id"], "string_pattern_mismatch", `String should match pattern '${MODEL_ID_PATTERN}'`);
    } else modelId = body.model_id;
  }
  let confidence = 0.9;
  if (given("confidence")) {
    if (!SUPPORTED_CONFIDENCES.includes(body.confidence)) {
      err(["confidence"], "literal_error", "Input should be 0.8, 0.9 or 0.95");
    } else {
      confidence = body.confidence;
    }
  }
  for (const key of Object.keys(body)) {
    if (!BODY_KEYS.includes(key) && body[key] !== undefined) err([key], "extra_forbidden", "Extra inputs are not permitted");
  }
  if (errors.length) throw validationProblem(errors);
  return { features, reference, modelId, confidence };
}

/**
 * Pick the bundle exactly like the API (battery_soh.api.inference.choose_model):
 * an explicit id must exist; otherwise, among the absolute models whose inputs are all present
 * and (only when a reference is given) the relative models whose inputs all have a usable
 * reference, the lowest recorded LOBO MAE wins (ties: model id). If none is complete, the
 * absolute model covering the most provided inputs is used (the rest are median-imputed).
 * @param {Array<{model_id: string, mode?: string, features?: string[], metrics?: object}>} models
 * @param {Set<string>} present Names of provided (non-null) features.
 * @param {string|null} modelId
 * @param {Set<string>|null} [referenced] Provided features usable in relative mode (context
 *   features, or a finite non-zero reference); null when the request has no reference.
 */
export function selectModel(models, present, modelId, referenced = null) {
  if (!Array.isArray(models) || models.length === 0) {
    throw problem(503, "No model available", "The static demo has no model bundles.", "/problems/no-models");
  }
  if (modelId) {
    const found = models.find((m) => m.model_id === modelId);
    if (!found) {
      throw problem(404, "Model not found", `No model bundle with id '${modelId}'. See GET /v1/models.`, "/problems/model-not-found");
    }
    return found;
  }
  const mae = (m) => (isNum(m.metrics?.lobo_mae) ? m.metrics.lobo_mae : Infinity);
  const rank = (a, b) => mae(a) - mae(b) || (a.model_id < b.model_id ? -1 : a.model_id > b.model_id ? 1 : 0);
  const best = (list) => (list.length ? list.reduce((a, b) => (rank(b, a) < 0 ? b : a)) : null);
  const modeOf = (m) => (m.mode === "relative" ? "relative" : "absolute");
  const complete = (mode, available) => best(models.filter((m) => modeOf(m) === mode && (m.features ?? []).every((f) => available.has(f))));
  const found = [complete("absolute", present)];
  if (referenced) found.push(complete("relative", referenced));
  const choice = best(found.filter(Boolean));
  if (choice) return choice;
  // Partial inputs: absolute models only, at least one provided input; the rest are imputed.
  const overlap = (m) => (m.features ?? []).filter((f) => present.has(f)).length;
  const partial = models.filter((m) => modeOf(m) === "absolute" && overlap(m) > 0);
  if (partial.length) return partial.reduce((a, b) => (overlap(b) > overlap(a) || (overlap(b) === overlap(a) && rank(b, a) < 0) ? b : a));
  const hint = models.some((m) => modeOf(m) === "absolute")
    ? ""
    : " The loaded models are relative: provide every input together with its commissioning 'reference'.";
  const needed = [...new Set(models.flatMap((m) => m.features ?? []))].sort();
  throw problem(422, "No usable features",
    `No loaded model uses any of the provided features; models need some of: ${needed.join(", ")}.${hint}`,
    "/problems/no-usable-features");
}

function vector(value, name, n, { allowNull = false } = {}) {
  if (!Array.isArray(value) || value.length !== n) {
    throw problem(500, "Invalid model bundle", `${name} must be a list of length ${n}.`);
  }
  return value.map((v) => {
    if (v === null && allowNull) return Number.NaN;
    if (!isNum(v)) throw problem(500, "Invalid model bundle", `${name} contains a non-finite value.`);
    return v;
  });
}

/**
 * Parse and validate a bundle document once (mirrors LinearBundle.__init__ checks).
 * @param {object} doc Bundle JSON (docs/MODEL_CARD.md#bundle-format).
 */
export function compileBundle(doc) {
  if (!isObj(doc)) throw problem(500, "Invalid model bundle", "Bundle must be a JSON object.");
  if (String(doc.schema_version ?? "").split(".")[0] !== "1") {
    throw problem(500, "Invalid model bundle", `Unsupported bundle schema_version '${doc.schema_version}'.`);
  }
  const features = doc.features;
  if (!Array.isArray(features) || !features.length || !features.every((f) => typeof f === "string")) {
    throw problem(500, "Invalid model bundle", "Bundle features must be a non-empty list of names.");
  }
  const p = features.length;
  const pre = doc.preprocessing ?? {};
  const est = doc.estimator ?? {};
  const dom = doc.domain ?? {};
  const scale = vector(pre.scaler_scale, "scaler_scale", p);
  if (scale.some((s) => s <= 0)) throw problem(500, "Invalid model bundle", "scaler_scale must be positive.");
  if (!isNum(est.intercept)) throw problem(500, "Invalid model bundle", "estimator.intercept must be finite.");
  const clip = vector(doc.output_clip ?? [0, 120], "output_clip", 2);
  if (!(clip[0] < clip[1])) throw problem(500, "Invalid model bundle", "output_clip must be [low, high].");
  const precision = dom.precision_matrix;
  if (!Array.isArray(precision) || precision.length !== p) {
    throw problem(500, "Invalid model bundle", `domain.precision_matrix must be a ${p}x${p} matrix (use the full bundle, not the model card).`);
  }
  const quantiles = new Map();
  for (const [key, value] of Object.entries(doc.conformal?.quantiles ?? {})) {
    const q = value === null ? Infinity : value;
    if (typeof q !== "number" || Number.isNaN(q) || q < 0) {
      throw problem(500, "Invalid model bundle", `Conformal quantile '${key}' must be a number >= 0 or null.`);
    }
    quantiles.set(confidenceKey(key), q);
  }
  const threshold = dom.mahalanobis_threshold;
  if (!isNum(threshold)) throw problem(500, "Invalid model bundle", "domain.mahalanobis_threshold must be finite.");
  const margin = dom.margin ?? 0.1;
  if (!isNum(margin) || margin < 0) throw problem(500, "Invalid model bundle", "domain.margin must be >= 0.");
  return {
    modelId: String(doc.model_id),
    mode: doc.mode === "relative" ? "relative" : "absolute",
    features: features.slice(),
    median: vector(pre.impute_median, "impute_median", p),
    mean: vector(pre.scaler_mean, "scaler_mean", p),
    scale,
    coef: vector(est.coef, "coef", p),
    intercept: est.intercept,
    clip,
    quantiles,
    method: String(doc.conformal?.method ?? ""),
    fmin: vector(dom.feature_min, "feature_min", p, { allowNull: true }),
    fmax: vector(dom.feature_max, "feature_max", p, { allowNull: true }),
    margin,
    center: dom.center == null ? new Array(p).fill(0) : vector(dom.center, "domain.center", p),
    precision: precision.map((row, i) => vector(row, `precision_matrix[${i}]`, p)),
    threshold,
  };
}

/**
 * Evaluate a compiled bundle on one input (already in the model's mode, i.e. relativised for
 * relative models). Mirrors LinearBundle.explain / interval / domain_check.
 * @param {ReturnType<typeof compileBundle>} b
 * @param {Record<string, number|null|undefined>} inputs
 * @param {number} confidence
 */
export function evaluate(b, inputs, confidence) {
  const p = b.features.length;
  const x = b.features.map((n) => (isNum(inputs[n]) ? inputs[n] : Number.NaN));
  const missing = b.features.filter((_, j) => Number.isNaN(x[j]));
  const used = x.map((v, j) => (Number.isNaN(v) ? b.median[j] : v));
  const z = used.map((v, j) => (v - b.mean[j]) / b.scale[j]);
  const parts = z.map((zj, j) => b.coef[j] * zj);
  const raw = b.intercept + parts.reduce((acc, c) => acc + c, 0);
  const [lo, hi] = b.clip;
  const prediction = Math.min(Math.max(raw, lo), hi);

  const key = confidenceKey(confidence);
  if (!b.quantiles.has(key)) {
    const available = [...b.quantiles.keys()].sort().join(", ");
    throw problem(422, "Unsupported confidence", `No conformal quantile for confidence ${key}; available: ${available}.`);
  }
  const q = b.quantiles.get(key);
  const lower = Math.max(prediction - q, lo);
  const upper = Math.min(prediction + q, hi);

  const outOfRange = [];
  for (let j = 0; j < p; j += 1) {
    let span = b.fmax[j] - b.fmin[j];
    if (!(span > 0)) span = Math.max(Math.abs(b.fmax[j]), 1); // NaN stays NaN (Math.max propagates)
    const low = b.fmin[j] - b.margin * span;
    const high = b.fmax[j] + b.margin * span;
    if (!Number.isNaN(x[j]) && !Number.isNaN(low) && (x[j] < low || x[j] > high)) outOfRange.push(b.features[j]);
  }
  const diff = z.map((zj, j) => zj - b.center[j]);
  let d2 = 0;
  for (let i = 0; i < p; i += 1) {
    let row = 0;
    for (let j = 0; j < p; j += 1) row += b.precision[i][j] * diff[j];
    d2 += diff[i] * row;
  }
  const mahalanobis = Math.sqrt(Math.max(d2, 0));
  return {
    rawPrediction: raw,
    prediction,
    lower,
    upper,
    contributions: b.features.map((feature, j) => ({ feature, value: used[j], contribution: parts[j] })),
    domain: {
      in_domain: outOfRange.length === 0 && mahalanobis <= b.threshold,
      out_of_range: outOfRange,
      mahalanobis,
      threshold: b.threshold,
      missing,
    },
  };
}

/**
 * Divide health indicators by the cell's commissioning reference (relative mode); context
 * features (ctx_*) are left untouched. Throws a 422 problem when a needed reference is absent,
 * or when a ratio is not finite (e.g. a subnormal reference), like battery_soh.api.inference.
 */
export function applyReference(features, reference, needed, modelId) {
  const out = { ...features };
  const bad = [];
  const nonFinite = [];
  for (const name of needed) {
    const v = features[name];
    if (!isNum(v) || isContextFeature(name)) continue;
    const r = reference?.[name];
    if (!isNum(r) || r === 0) {
      bad.push(name);
      continue;
    }
    out[name] = v / r;
    if (!Number.isFinite(out[name])) nonFinite.push(name);
  }
  if (bad.length) {
    throw problem(422, "Reference required",
      `Model '${modelId}' is relative: provide non-zero 'reference' values for ${bad.join(", ")}.`,
      "/problems/reference-required");
  }
  if (nonFinite.length) {
    throw problem(422, "Reference required",
      `The reference values for ${nonFinite.sort().join(", ")} give non-finite ratios.`,
      "/problems/reference-required");
  }
  return out;
}

/**
 * Full POST /v1/soh/estimate in the browser.
 * @param {unknown} body Request body (same JSON as the API).
 * @param {{listModels: () => Promise<any[]>, loadBundle: (id: string) => Promise<object>}} deps
 */
export async function estimate(body, { listModels, loadBundle }) {
  const req = validateEstimateBody(body);
  const provided = Object.entries(req.features).filter(([, v]) => v !== null).map(([k]) => k);
  const present = new Set(provided);
  const refValues = req.reference ? Object.entries(req.reference).filter(([, v]) => v !== null) : [];
  // Like the API: a reference with no provided value counts as no reference at all.
  const referenced = refValues.length
    ? new Set(provided.filter((f) => isContextFeature(f) || (isNum(req.reference[f]) && req.reference[f] !== 0)))
    : null;
  const summary = selectModel(await listModels(), present, req.modelId, referenced);
  const bundle = compileBundle(await loadBundle(summary.model_id));
  if (!bundle.features.some((f) => present.has(f))) {
    throw problem(422, "No usable features",
      `Model '${bundle.modelId}' needs at least one of: ${bundle.features.join(", ")}.`, "/problems/no-usable-features");
  }
  let inputs = req.features;
  if (bundle.mode === "relative") inputs = applyReference(req.features, req.reference, bundle.features, bundle.modelId);
  const r = evaluate(bundle, inputs, req.confidence);
  if (![r.rawPrediction, r.domain.mahalanobis, ...r.contributions.map((c) => c.contribution)].every(Number.isFinite)) {
    throw problem(422, "Input out of numeric range",
      "The inputs produce a non-finite model output; check their units and scale.", "/problems/non-finite-result");
  }
  const warnings = [];
  if (r.domain.missing.length) {
    warnings.push(`Imputed the training median for ${r.domain.missing.length} missing feature(s): ${r.domain.missing.join(", ")}.`);
  }
  if (r.prediction !== r.rawPrediction) {
    warnings.push(`Prediction clipped to the physical range [${bundle.clip[0]}, ${bundle.clip[1]}].`);
  }
  if (bundle.quantiles.get(confidenceKey(req.confidence)) === Infinity) {
    warnings.push(`No finite conformal interval at confidence ${req.confidence} (too few calibration scores); the interval spans the whole output range.`);
  }
  if (!r.domain.in_domain) warnings.push("Input is outside the training domain; the interval may not hold.");
  return {
    model_id: bundle.modelId,
    soh_pct: r.prediction,
    interval: { lower: r.lower, upper: r.upper, confidence: req.confidence, method: bundle.method },
    base_value: bundle.intercept,
    contributions: r.contributions,
    domain: r.domain,
    warnings,
  };
}
