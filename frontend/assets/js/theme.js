// Light/dark theme: follows prefers-color-scheme until the user toggles; the explicit choice is
// persisted in localStorage (guarded: storage may be unavailable). Emits "soh:themechange" on
// document so charts can re-read the palette.

const KEY = "soh-theme";
const media = window.matchMedia ? window.matchMedia("(prefers-color-scheme: dark)") : null;

function stored() {
  try {
    const v = window.localStorage.getItem(KEY);
    return v === "light" || v === "dark" ? v : null;
  } catch {
    return null;
  }
}

function persist(value) {
  try {
    window.localStorage.setItem(KEY, value);
  } catch {
    /* ignore: the choice then lasts for this page view only */
  }
}

/** @returns {"light"|"dark"} */
export function currentTheme() {
  const explicit = document.documentElement.getAttribute("data-theme");
  if (explicit === "light" || explicit === "dark") return explicit;
  return media && media.matches ? "dark" : "light";
}

function notify() {
  document.dispatchEvent(new CustomEvent("soh:themechange", { detail: { theme: currentTheme() } }));
}

/**
 * Wire the toggle button (a toggle button: constant name "Dark theme", aria-pressed state).
 * @param {HTMLButtonElement} button
 */
export function initThemeToggle(button) {
  const sync = () => {
    const dark = currentTheme() === "dark";
    button.setAttribute("aria-pressed", String(dark));
    button.title = dark ? "Switch to light theme" : "Switch to dark theme";
  };
  const saved = stored();
  if (saved) document.documentElement.setAttribute("data-theme", saved);
  sync();
  button.addEventListener("click", () => {
    const next = currentTheme() === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    persist(next);
    sync();
    notify();
  });
  if (media) {
    const onChange = () => {
      if (!stored()) {
        sync();
        notify();
      }
    };
    if (media.addEventListener) media.addEventListener("change", onChange);
    else if (media.addListener) media.addListener(onChange);
  }
}
