// API client for the /v1 contract (docs/API.md).
//
// * Same-origin, relative URLs ("v1/..."), so the app also works behind a path prefix.
// * Every failure becomes an ApiError with {status, title, detail, fieldErrors, requestId}:
//   RFC 7807 problem details, FastAPI/Pydantic 422 bodies ({detail: [{loc, msg, type}]}),
//   plain {detail: "..."} bodies, network failures and timeouts are all normalised here.
// * Static mode (window.SOH_CONFIG = {mode: "static", dataBase: "data/"}, written by the static
//   export): GET endpoints read precomputed JSON files and POST /v1/soh/estimate runs in the
//   browser (static-backend.js). Expected layout under dataBase:
//     models.json                      <- GET /v1/models
//     models/<model_id>.json           <- full bundle incl. domain.precision_matrix
//     demo/batteries.json              <- GET /v1/demo/batteries
//     demo/<battery_id>/<cycle>.json   <- GET /v1/demo/batteries/<id>/cycles/<cycle>
//     benchmark.json                   <- GET /v1/benchmark

import { isObj } from "./guards.js";
import * as staticBackend from "./static-backend.js";

const DEFAULT_TIMEOUT_MS = 20000;
const SAFE_SEGMENT = /^[A-Za-z0-9_.-]{1,64}$/;

function readConfig() {
  const raw = (typeof window !== "undefined" && window.SOH_CONFIG) || {};
  const mode = raw.mode === "static" ? "static" : "api";
  let dataBase = typeof raw.dataBase === "string" && raw.dataBase ? raw.dataBase : "data/";
  if (!dataBase.endsWith("/")) dataBase += "/";
  return Object.freeze({ mode, dataBase });
}

export const CONFIG = readConfig();
export const IS_STATIC = CONFIG.mode === "static";

export class ApiError extends Error {
  /**
   * @param {{status: number, title: string, detail?: string, type?: string, requestId?: string|null,
   *   fieldErrors?: Array<{field: string, path: string[], message: string}>}} info
   */
  constructor({ status, title, detail = "", type = "about:blank", requestId = null, fieldErrors = [] }) {
    super(detail ? `${title}: ${detail}` : title);
    this.name = "ApiError";
    this.status = status;
    this.title = title;
    this.detail = detail;
    this.type = type;
    this.requestId = requestId;
    this.fieldErrors = fieldErrors;
  }
}

const STATUS_TITLES = {
  400: "Bad request", 401: "Not authorised", 403: "Forbidden", 404: "Not found",
  405: "Method not allowed", 408: "Request timed out", 413: "Request too large",
  415: "Unsupported media type", 422: "Invalid input", 429: "Too many requests",
  500: "Server error", 501: "Not available", 502: "Bad gateway", 503: "Service unavailable",
  504: "Gateway timeout",
};

/** Turn a Pydantic `loc` (["body", "features", "dis_r0_ohm"]) into a field error. */
function fieldError(item) {
  const loc = Array.isArray(item?.loc) ? item.loc.map(String) : [];
  const path = loc[0] === "body" || loc[0] === "query" || loc[0] === "path" ? loc.slice(1) : loc;
  const raw = typeof item?.msg === "string" ? item.msg : typeof item?.message === "string" ? item.message : "Invalid value";
  return {
    field: path.length ? path[path.length - 1] : "",
    path,
    message: raw.replace(/^Value error, /, ""),
  };
}

/**
 * Normalise any error response body into an ApiError.
 * @param {number} status
 * @param {unknown} body Parsed JSON (or text) of the response.
 * @param {string|null} [headerRequestId] Value of the X-Request-ID header.
 */
export function normalizeError(status, body, headerRequestId = null) {
  const obj = isObj(body) ? body : {};
  const requestId = (typeof obj.request_id === "string" && obj.request_id) || headerRequestId || null;
  const fallbackTitle = STATUS_TITLES[status] || (status ? `HTTP ${status}` : "Request failed");
  const listSource = Array.isArray(obj.detail) ? obj.detail : Array.isArray(obj.errors) ? obj.errors : null;
  const fieldErrors = listSource ? listSource.map(fieldError) : [];
  let title = typeof obj.title === "string" && obj.title ? obj.title : fallbackTitle;
  let detail = "";
  if (typeof obj.detail === "string") detail = obj.detail;
  else if (fieldErrors.length) {
    if (!obj.title) title = "Invalid input";
    detail = fieldErrors
      .slice(0, 4)
      .map((e) => (e.path.length ? `${e.path.join(".")}: ${e.message}` : e.message))
      .join("; ");
    if (fieldErrors.length > 4) detail += ` (+${fieldErrors.length - 4} more)`;
  } else if (typeof body === "string" && body.trim() && body.length < 300 && !/^\s*</.test(body)) {
    detail = body.trim();
  }
  return new ApiError({
    status,
    title,
    detail,
    type: typeof obj.type === "string" ? obj.type : "about:blank",
    requestId,
    fieldErrors,
  });
}

function resolveUrl(path) {
  return new URL(path, document.baseURI).toString();
}

/** fetch() with a timeout, caller cancellation and uniform error normalisation. */
async function fetchJson(url, { method = "GET", body, signal, timeoutMs = DEFAULT_TIMEOUT_MS } = {}) {
  const controller = new AbortController();
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);
  const onAbort = () => controller.abort();
  if (signal) {
    if (signal.aborted) controller.abort();
    else signal.addEventListener("abort", onAbort, { once: true });
  }
  let response;
  try {
    response = await fetch(resolveUrl(url), {
      method,
      headers: body === undefined ? { Accept: "application/json" } : { Accept: "application/json", "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
      credentials: "same-origin",
      cache: "no-cache",
      signal: controller.signal,
    });
  } catch (err) {
    if (signal?.aborted) throw new DOMException("Request cancelled", "AbortError");
    if (timedOut) {
      throw new ApiError({ status: 0, title: "Request timed out", detail: `No response within ${Math.round(timeoutMs / 1000)} s.` });
    }
    throw new ApiError({ status: 0, title: "Network error", detail: "Could not reach the server. Check that the API is running." });
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener("abort", onAbort);
  }
  const contentType = response.headers.get("content-type") || "";
  const isJson = contentType.includes("json");
  let payload = null;
  let unreadable = false;
  try {
    payload = isJson ? await response.json() : await response.text();
  } catch (err) {
    if (signal?.aborted) throw new DOMException("Request cancelled", "AbortError");
    unreadable = true;
    payload = null;
  }
  if (!response.ok) throw normalizeError(response.status, payload, response.headers.get("x-request-id"));
  // A 2xx whose body is not valid JSON is an error, never a `null` result (which the GET
  // cache would otherwise keep for the rest of the session).
  if (isJson && !unreadable) return payload;
  throw new ApiError({
    status: response.status,
    title: "Unexpected response",
    detail: unreadable ? "The server's JSON response could not be read." : "The server did not return JSON.",
    requestId: response.headers.get("x-request-id"),
  });
}

/**
 * Wait for a shared promise on behalf of one caller: the caller's AbortSignal rejects only its
 * own wait (with an AbortError), never the shared request other callers are waiting on.
 */
function forCaller(promise, signal) {
  if (!signal) return promise;
  if (signal.aborted) return Promise.reject(new DOMException("Request cancelled", "AbortError"));
  return new Promise((resolve, reject) => {
    const onAbort = () => reject(new DOMException("Request cancelled", "AbortError"));
    signal.addEventListener("abort", onAbort, { once: true });
    promise.then(
      (value) => {
        signal.removeEventListener("abort", onAbort);
        resolve(value);
      },
      (err) => {
        signal.removeEventListener("abort", onAbort);
        reject(err);
      },
    );
  });
}

// ---------------------------------------------------------------- static mode
function staticPath(apiPath) {
  const parts = apiPath.split("/");
  const ok = (s) => SAFE_SEGMENT.test(s);
  if (apiPath === "v1/models") return "models.json";
  if (parts.length === 3 && parts[1] === "models" && ok(parts[2])) return `models/${parts[2]}.json`;
  if (apiPath === "v1/demo/batteries") return "demo/batteries.json";
  if (parts.length === 6 && parts[2] === "batteries" && parts[4] === "cycles" && ok(parts[3]) && /^\d{1,6}$/.test(parts[5])) {
    return `demo/${parts[3]}/${parts[5]}.json`;
  }
  if (apiPath === "v1/benchmark") return "benchmark.json";
  return null;
}

async function staticGet(apiPath, opts) {
  const file = staticPath(apiPath);
  if (!file) {
    throw new ApiError({ status: 501, title: "Not available in the static demo", detail: `${apiPath} has no precomputed data.` });
  }
  try {
    return await fetchJson(CONFIG.dataBase + file, opts);
  } catch (err) {
    if (err instanceof ApiError && err.status === 404) {
      throw new ApiError({ status: 404, title: "Not found", detail: `The static demo has no data for ${apiPath}.` });
    }
    throw err;
  }
}

function toApiError(err) {
  if (err instanceof staticBackend.StaticBackendError) return normalizeError(err.status, err.body);
  return err;
}

// ---------------------------------------------------------------- public API
const cache = new Map();

/**
 * GET with an in-memory cache of successful responses (failed requests are evicted). The
 * shared request runs without any caller's AbortSignal (only its timeout applies); each
 * caller's signal cancels just that caller's wait, so one superseded view cannot abort a
 * response that other views are waiting for.
 */
function cachedGet(apiPath, opts = {}) {
  const { signal, ...shared } = opts;
  let promise = cache.get(apiPath);
  if (!promise) {
    promise = (IS_STATIC ? staticGet(apiPath, shared) : fetchJson(apiPath, shared)).catch((err) => {
      if (cache.get(apiPath) === promise) cache.delete(apiPath);
      throw err;
    });
    cache.set(apiPath, promise);
  }
  return forCaller(promise, signal);
}

const enc = encodeURIComponent;

export const api = {
  /** GET /readyz (static mode: synthetic readiness from models.json). */
  async ready(opts) {
    if (IS_STATIC) {
      const models = await cachedGet("v1/models", opts);
      return { status: "ready", models: models.map((m) => m.model_id) };
    }
    return fetchJson("readyz", { ...opts, timeoutMs: 8000 });
  },
  /** GET /v1/models */
  models: (opts) => cachedGet("v1/models", opts),
  /** GET /v1/models/{id} */
  model: (id, opts) => cachedGet(`v1/models/${enc(id)}`, opts),
  /** GET /v1/demo/batteries */
  demoBatteries: (opts) => cachedGet("v1/demo/batteries", opts),
  /** GET /v1/demo/batteries/{id}/cycles/{k} */
  demoCycle: (batteryId, cycleIndex, opts) => cachedGet(`v1/demo/batteries/${enc(batteryId)}/cycles/${enc(String(cycleIndex))}`, opts),
  /** GET /v1/benchmark */
  benchmark: (opts) => cachedGet("v1/benchmark", opts),
  /** POST /v1/soh/estimate */
  async estimate(body, opts = {}) {
    if (!IS_STATIC) return fetchJson("v1/soh/estimate", { ...opts, method: "POST", body });
    try {
      return await staticBackend.estimate(body, {
        listModels: () => cachedGet("v1/models", opts),
        loadBundle: (id) => cachedGet(`v1/models/${enc(id)}`, opts),
      });
    } catch (err) {
      throw toApiError(err);
    }
  },
  /** POST /v1/soh/estimate-from-cycle (needs the Python feature extractor: API mode only). */
  async estimateFromCycle(body, opts = {}) {
    if (IS_STATIC) {
      throw new ApiError({
        status: 501,
        title: "Not available in the static demo",
        detail: "Feature extraction from raw cycles runs in the Python API.",
      });
    }
    return fetchJson("v1/soh/estimate-from-cycle", { ...opts, method: "POST", body, timeoutMs: 45000 });
  },
};

/** True for errors caused by our own cancellation (superseded requests). */
export function isAbort(err) {
  return err?.name === "AbortError";
}
