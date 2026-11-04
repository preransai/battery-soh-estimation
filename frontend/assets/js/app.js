// Entry point: theme toggle, service status, tabs with lazily initialised views.

import { api, IS_STATIC } from "./api.js";
import { initThemeToggle } from "./theme.js";
import { initTabs } from "./tabs.js";
import { initEstimate } from "./view-estimate.js";
import { initBenchmark } from "./view-benchmark.js";
import { initAudit } from "./view-audit.js";
import { initModelCard } from "./view-model-card.js";

const VIEWS = { estimate: initEstimate, benchmark: initBenchmark, audit: initAudit, model: initModelCard };
const started = new Set();

function startView(name) {
  if (started.has(name) || !VIEWS[name]) return;
  started.add(name);
  try {
    VIEWS[name]();
  } catch (err) {
    // A programming error in one view must not take the others down; surface it in the console.
    console.error(`view "${name}" failed to start`, err);
  }
}

async function showServiceStatus() {
  const pill = document.getElementById("service-status");
  const text = pill.querySelector(".status-text");
  if (IS_STATIC) {
    pill.dataset.state = "static";
    text.textContent = "Static demo";
    return;
  }
  try {
    const ready = await api.ready();
    const n = Array.isArray(ready?.models) ? ready.models.length : null;
    pill.dataset.state = "ok";
    text.textContent = n === null ? "API ready" : `API ready \u00b7 ${n} model${n === 1 ? "" : "s"}`;
  } catch (err) {
    pill.dataset.state = "down";
    text.textContent = err?.status === 503 ? "API not ready" : "API unreachable";
  }
}

function main() {
  initThemeToggle(document.getElementById("theme-toggle"));
  if (IS_STATIC) {
    document.getElementById("static-banner").hidden = false;
    document.getElementById("footer-links").hidden = true;
    document.documentElement.dataset.mode = "static";
  }
  showServiceStatus();
  startView("estimate");
  initTabs(document.getElementById("tablist"), { onSelect: (name) => startView(name) });
}

main();
