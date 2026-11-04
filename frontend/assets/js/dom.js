// Tiny DOM helpers. Everything that may carry API or user data goes through textContent or
// attribute setters; nothing in this app assigns innerHTML.

/**
 * Create an element.
 * @param {string} tag
 * @param {object|null} [props] class, text, attrs, dataset, on, hidden, props (DOM properties)
 * @param {...(Node|string|number|null|undefined|false|Array)} children
 */
export function h(tag, props, ...children) {
  const el = document.createElement(tag);
  applyProps(el, props);
  append(el, children);
  return el;
}

function applyProps(el, props) {
  if (!props) return;
  for (const [key, value] of Object.entries(props)) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "class") el.className = Array.isArray(value) ? value.filter(Boolean).join(" ") : value;
    else if (key === "text") el.textContent = String(value);
    else if (key === "attrs") {
      for (const [a, v] of Object.entries(value)) {
        if (v !== undefined && v !== null && v !== false) el.setAttribute(a, v === true ? "" : String(v));
      }
    } else if (key === "dataset") Object.assign(el.dataset, value);
    else if (key === "on") {
      for (const [evt, fn] of Object.entries(value)) el.addEventListener(evt, fn);
    } else if (key === "hidden") el.hidden = Boolean(value);
    else if (key === "props") Object.assign(el, value);
    else el.setAttribute(key, value === true ? "" : String(value));
  }
}

function append(el, children) {
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    el.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
}

/** Remove all children. */
function clear(el) {
  while (el.firstChild) el.removeChild(el.firstChild);
  return el;
}

/** Replace the children of `el`. */
export function mount(el, ...children) {
  clear(el);
  append(el, children);
  return el;
}

let announceTimer = 0;
/**
 * Politely announce a short message to assistive technology (single shared live region).
 * @param {string} message
 */
export function announce(message) {
  const region = document.getElementById("announcer");
  if (!region) return;
  clearTimeout(announceTimer);
  region.textContent = "";
  // A fresh text node after a tick makes screen readers re-announce identical messages.
  announceTimer = window.setTimeout(() => {
    region.textContent = message;
  }, 60);
}

/** Skeleton placeholder block(s) while content loads. */
export function skeleton({ lines = 3, chart = false, label = "Loading" } = {}) {
  const wrap = h("div", { class: "skeleton", attrs: { role: "status" } }, h("span", { class: "visually-hidden", text: `${label}\u2026` }));
  if (chart) wrap.append(h("div", { class: "skeleton-block skeleton-chart", attrs: { "aria-hidden": "true" } }));
  for (let i = 0; i < lines; i += 1) {
    wrap.append(h("div", { class: ["skeleton-block", "skeleton-line", i === lines - 1 && "skeleton-line--short"], attrs: { "aria-hidden": "true" } }));
  }
  return wrap;
}

/** Neutral empty state. */
export function emptyState(title, body, extra = null) {
  return h("div", { class: "state state--empty" },
    h("p", { class: "state-title", text: title }),
    body ? h("p", { class: "state-body", text: body }) : null,
    extra);
}

/**
 * Error state with optional retry; rendered with role="alert" so it is announced.
 * @param {any} err ApiError or Error
 * @param {{onRetry?: () => void, compact?: boolean}} [opts] The retry button is only offered
 *   when retrying can help (network errors, timeouts, rate limits and server errors).
 */
export function errorState(err, { onRetry, compact = false } = {}) {
  const title = err?.title || err?.name || "Something went wrong";
  const detail = err?.detail ?? (err?.title ? "" : err?.message) ?? "";
  const meta = [];
  if (err?.status) meta.push(`HTTP ${err.status}`);
  if (err?.requestId) meta.push(`request ${err.requestId}`);
  const box = h("div", { class: ["state", "state--error", compact && "state--compact"], attrs: { role: "alert" } },
    h("p", { class: "state-title", text: title }),
    detail ? h("p", { class: "state-body", text: detail }) : null,
    meta.length ? h("p", { class: "state-meta", text: meta.join(" \u00b7 ") }) : null);
  // Retrying the same request cannot fix a client error (other than a timeout or rate limit).
  const status = Number(err?.status) || 0;
  const retryable = !(status >= 400 && status < 500 && status !== 408 && status !== 429);
  if (onRetry && retryable) {
    box.append(h("button", { type: "button", class: "btn btn-small", text: "Try again", on: { click: onRetry } }));
  }
  return box;
}

/** Small inline tag/pill. */
export function tag(text, variant = "neutral", title = null) {
  return h("span", { class: `tag tag--${variant}`, text, attrs: { title } });
}

/**
 * Accessible data table. Columns: {key, label, num?, format?(value,row) -> string|Node, sortable?,
 * title?}. Options: caption, sortKey, sortDir, onSort(key), rowClass(row), scroll label.
 */
export function dataTable(columns, rows, opts = {}) {
  const { caption, sortKey, sortDir = "asc", onSort, rowClass, label, captionHidden = false, wide = false } = opts;
  const table = h("table", { class: ["data-table", wide && "data-table--wide"] });
  if (caption) table.append(h("caption", { class: captionHidden ? "visually-hidden" : null, text: caption }));
  const headRow = h("tr");
  for (const col of columns) {
    const th = h("th", { scope: "col", class: col.num ? "num" : null, attrs: { title: col.title || null } });
    if (onSort && col.sortable !== false) {
      const active = col.key === sortKey;
      th.setAttribute("aria-sort", active ? (sortDir === "asc" ? "ascending" : "descending") : "none");
      th.append(h("button", {
        type: "button",
        class: "sort-btn",
        on: { click: () => onSort(col.key) },
      }, col.label, h("span", { class: "sort-icon", attrs: { "aria-hidden": "true" }, text: active ? (sortDir === "asc" ? "\u25b2" : "\u25bc") : "\u2195" })));
    } else {
      th.textContent = col.label;
    }
    headRow.append(th);
  }
  table.append(h("thead", null, headRow));
  // A group heading spans the columns but heads the rows below it: each group gets its own
  // <tbody>, so scope="rowgroup" covers exactly that group's rows.
  let body = h("tbody");
  for (const row of rows) {
    if (row && row.__group) {
      if (body.rows.length) table.append(body);
      body = h("tbody");
      body.append(h("tr", { class: "group-row" }, h("th", { scope: "rowgroup", attrs: { colspan: String(columns.length) }, text: row.__group })));
      continue;
    }
    const tr = h("tr", { class: rowClass ? rowClass(row) : null });
    columns.forEach((col, i) => {
      const raw = row[col.key];
      const content = col.format ? col.format(raw, row) : raw ?? "\u2014";
      const cell = h(i === 0 && col.rowHeader !== false ? "th" : "td", {
        class: col.num ? "num" : null,
        scope: i === 0 && col.rowHeader !== false ? "row" : null,
      });
      cell.append(content instanceof Node ? content : document.createTextNode(String(content)));
      tr.append(cell);
    });
    body.append(tr);
  }
  table.append(body);
  return h("div", { class: "table-wrap", attrs: { tabindex: "0", role: "region", "aria-label": label || caption || "Data table" } }, table);
}

/** Debounce helper. */
export function debounce(fn, ms) {
  let t = 0;
  const wrapped = (...args) => {
    clearTimeout(t);
    t = window.setTimeout(() => fn(...args), ms);
  };
  wrapped.cancel = () => clearTimeout(t);
  return wrapped;
}

/** Set a CSS custom property through the CSSOM (allowed by the strict CSP). */
export function setVar(el, name, value) {
  el.style.setProperty(name, String(value));
}
