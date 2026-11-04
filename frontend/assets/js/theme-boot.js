// Classic (non-module) script loaded in <head>: applies a persisted theme choice before first
// paint so a saved preference never flashes the other theme. No choice = follow the OS setting.
(function () {
  try {
    var saved = window.localStorage.getItem("soh-theme");
    if (saved === "light" || saved === "dark") document.documentElement.setAttribute("data-theme", saved);
  } catch (e) {
    /* storage unavailable (privacy mode, sandbox): fall back to prefers-color-scheme */
  }
})();
