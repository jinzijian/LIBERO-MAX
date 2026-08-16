(() => {
  const root = document.documentElement;
  const toggle = document.querySelector(".theme-toggle");
  if (!toggle) return;
  toggle.addEventListener("click", () => {
    const current = root.dataset.theme;
    const systemDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    const next = current === "dark" || (!current && systemDark) ? "light" : "dark";
    root.dataset.theme = next;
    try { localStorage.setItem("libero-max-theme", next); } catch (error) { /* Optional. */ }
  });
})();
