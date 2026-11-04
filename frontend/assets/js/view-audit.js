// Leakage audit tab: "v1 as reported vs honest" chart (left card), fact tiles and the full results
// table (below the cards), from the optional `audit` field of GET /v1/benchmark. The plain-language
// explanation is static HTML.

import { api } from "./api.js";
import { auditConfig, chartsAvailable, makeChart } from "./charts.js";
import { dataTable, emptyState, errorState, h, mount, setVar, skeleton, tag } from "./dom.js";
import { fmt } from "./format.js";
import { isNum } from "./guards.js";
import { normalizeAudit } from "./normalize.js";

const CATEGORY_LABELS = {
  v1_reported: "v1 as reported",
  honest: "Honest re-evaluation",
  v2: "v2 (this project)",
};

function factTiles(facts) {
  const tiles = [];
  if (isNum(facts.sohVsCapacityR2)) {
    tiles.push(["Within-battery R\u00b2 of SOH vs capacity", fmt(facts.sohVsCapacityR2, 3), "SOH is capacity divided by a constant, so an input explained it exactly."]);
  }
  if (facts.overlap) {
    tiles.push(["v1 test rows that were training rows", `${fmt(facts.overlap.rows, 0)} / ${fmt(facts.overlap.total, 0)}`, "The reported test score was mostly measured on seen data."]);
  }
  if (isNum(facts.capacityReconstructionR2)) {
    tiles.push(["Capacity rebuilt from v1 proxies", `R\u00b2 ${fmt(facts.capacityReconstructionR2, 3)}`, "Record length and whole-record means encode the label numerator."]);
  }
  if (!tiles.length) return null;
  return h("div", { class: "facts" }, tiles.map(([label, value, sub]) => h("div", { class: "fact" },
    h("p", { class: "fact-value tnum", text: value }),
    h("p", { class: "fact-label", text: label }),
    h("p", { class: "fact-sub", text: sub }))));
}

function auditTable(rows) {
  return dataTable([
    {
      key: "label", label: "Evaluation",
      format: (v, r) => h("span", { class: "cell-block" },
        h("span", { class: "cell-stack" }, v, r.leaky ? tag("leaky input", "warn") : null),
        r.note ? h("span", { class: "cell-note", text: r.note }) : null),
    },
    { key: "category", label: "Kind", rowHeader: false, format: (v) => CATEGORY_LABELS[v] || v },
    { key: "mae", label: "MAE", num: true, format: (v) => fmt(v, 2) },
    { key: "r2", label: "R\u00b2", num: true, format: (v) => fmt(v, 3) },
  ], rows, {
    caption: "Audit results: v1 as reported versus honest evaluations",
    captionHidden: true,
    rowClass: (r) => (r.category === "v1_reported" ? "row-leaky" : r.category === "v2" ? "row-highlight" : null),
  });
}

export function initAudit() {
  const root = document.getElementById("audit-root");
  const factsRoot = document.getElementById("audit-facts");
  const tableCard = document.getElementById("audit-table-card");
  const tableRoot = document.getElementById("audit-table");
  function clearExtras() {
    mount(factsRoot);
    mount(tableRoot);
    tableCard.hidden = true;
  }
  async function load() {
    root.setAttribute("aria-busy", "true");
    clearExtras();
    mount(root, skeleton({ chart: true, lines: 2, label: "Loading audit" }));
    let audit;
    try {
      const summary = await api.benchmark();
      audit = normalizeAudit(summary && typeof summary === "object" ? summary.audit : null);
    } catch (err) {
      root.setAttribute("aria-busy", "false");
      mount(root, errorState(err, { onRetry: load }));
      return;
    }
    root.setAttribute("aria-busy", "false");
    if (!audit) {
      mount(root, emptyState(
        "No audit results in this benchmark",
        "The benchmark summary served here has no audit section. Running \u201csoh audit\u201d reproduces the v1 numbers next to an honest evaluation; the explanation alongside still applies."));
      return;
    }
    const rows = audit.rows;
    const children = [];
    if (rows.some((r) => isNum(r.mae))) {
      const plotted = rows.filter((r) => isNum(r.mae));
      const canvas = h("canvas", {
        role: "img",
        attrs: { "aria-label": `Mean absolute error by evaluation: ${plotted.map((r) => `${r.label} ${fmt(r.mae, 2)}`).join("; ")}.` },
      });
      const box = h("div", { class: "chart-box chart-box--audit" }, canvas);
      setVar(box, "--rows", plotted.length);
      const present = [...new Set(plotted.map((r) => r.category))];
      children.push(
        h("ul", { class: "legend", attrs: { "aria-label": "Legend" } },
          ["v1_reported", "honest", "v2"].filter((c) => present.includes(c)).map((c) => h("li", null, h("span", { class: `key key--${c}` }), CATEGORY_LABELS[c]))),
        box);
      if (chartsAvailable()) queueMicrotask(() => makeChart(canvas, (pal) => auditConfig(pal, plotted)));
    }
    if (children.length) {
      // Chart in the card; the detailed table gets its own full-width card below.
      mount(tableRoot, auditTable(rows));
      tableCard.hidden = false;
    } else if (rows.length) {
      children.push(auditTable(rows));
    } else {
      children.push(emptyState("No audit rows", "The audit section has summary facts only."));
    }
    mount(factsRoot, factTiles(audit.facts));
    mount(root, ...children);
  }
  load();
}
