// Type guards for untrusted JSON (API responses, model bundles, form input), shared by every
// module. No imports, so the static backend can use them in the browser and in Node.js alike.

/** True for a finite number (not NaN or +/-Infinity, and not a numeric string). */
export const isNum = (v) => typeof v === "number" && Number.isFinite(v);

/** True for a plain JSON object: not null and not an array. */
export const isObj = (v) => v !== null && typeof v === "object" && !Array.isArray(v);
