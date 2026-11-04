// Runtime configuration. The repository ships API mode (same-origin /v1 endpoints).
// The static export (GitHub Pages demo) overwrites this file with
//   window.SOH_CONFIG = Object.freeze({ mode: "static", dataBase: "data/" });
// and writes precomputed JSON under data/ (see assets/js/api.js for the layout).
window.SOH_CONFIG = Object.freeze({ mode: "api" });
