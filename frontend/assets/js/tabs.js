// WAI-ARIA tabs (automatic activation): roving tabindex, Left/Right/Home/End keys, URL hash sync
// (#estimate, #benchmark, #audit, #model) so sections can be linked and survive reloads.

/**
 * @param {HTMLElement} tablist Element with role="tablist" containing role="tab" buttons.
 * @param {{onSelect?: (name: string, panel: HTMLElement) => void}} [opts]
 * @returns {{select: (name: string) => void}}
 */
export function initTabs(tablist, { onSelect } = {}) {
  const tabs = [...tablist.querySelectorAll('[role="tab"]')];

  // On narrow screens the tab strip scrolls sideways; keep the selected tab visible without
  // scrolling the page vertically (the tablist is position: relative, so offsetLeft is local).
  function revealHorizontally(tab) {
    const left = tab.offsetLeft;
    const right = left + tab.offsetWidth;
    if (left < tablist.scrollLeft) tablist.scrollLeft = Math.max(0, left - 16);
    else if (right > tablist.scrollLeft + tablist.clientWidth) tablist.scrollLeft = right - tablist.clientWidth + 16;
  }
  // Mark which edges hide tabs (CSS fades them) so a cut-off label reads as "scroll for more".
  const bar = tablist.parentElement;
  function updateOverflow() {
    const max = tablist.scrollWidth - tablist.clientWidth;
    const edges = [];
    if (tablist.scrollLeft > 1) edges.push("start");
    if (tablist.scrollLeft < max - 1) edges.push("end");
    if (edges.length) bar.dataset.overflow = edges.join(" ");
    else delete bar.dataset.overflow;
  }
  tablist.addEventListener("scroll", updateOverflow, { passive: true });
  window.addEventListener("resize", updateOverflow);
  const nameOf = (tab) => tab.dataset.tab;
  const panelOf = (tab) => document.getElementById(tab.getAttribute("aria-controls"));

  function activate(tab, { focus = false, updateHash = true } = {}) {
    for (const t of tabs) {
      const selected = t === tab;
      t.setAttribute("aria-selected", String(selected));
      t.tabIndex = selected ? 0 : -1;
      panelOf(t).hidden = !selected;
    }
    if (focus) tab.focus();
    revealHorizontally(tab);
    updateOverflow();
    if (updateHash && window.location.hash !== `#${nameOf(tab)}`) {
      window.history.replaceState(null, "", `#${nameOf(tab)}`);
    }
    if (onSelect) onSelect(nameOf(tab), panelOf(tab));
  }

  tablist.addEventListener("click", (event) => {
    const tab = event.target.closest('[role="tab"]');
    if (tab && tabs.includes(tab)) activate(tab, { focus: true });
  });

  tablist.addEventListener("keydown", (event) => {
    const index = tabs.indexOf(document.activeElement);
    if (index === -1) return;
    let next = null;
    switch (event.key) {
      case "ArrowRight":
        next = tabs[(index + 1) % tabs.length];
        break;
      case "ArrowLeft":
        next = tabs[(index - 1 + tabs.length) % tabs.length];
        break;
      case "Home":
        next = tabs[0];
        break;
      case "End":
        next = tabs[tabs.length - 1];
        break;
      default:
        return;
    }
    event.preventDefault();
    activate(next, { focus: true });
  });

  const fromHash = () => tabs.find((t) => `#${nameOf(t)}` === window.location.hash);
  window.addEventListener("hashchange", () => {
    const tab = fromHash();
    if (tab && tab.getAttribute("aria-selected") !== "true") activate(tab, { updateHash: false });
  });

  activate(fromHash() || tabs[0], { updateHash: false });
  return {
    select(name) {
      const tab = tabs.find((t) => nameOf(t) === name);
      if (tab) activate(tab, { focus: true });
    },
  };
}
