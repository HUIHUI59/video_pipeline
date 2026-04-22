// pod_control M1 baseline — tab switcher + health ping.
// Feature tabs (Prepare/Pods/Run/Monitor) land in M3-M6.

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

function switchTab(name) {
  $$("nav button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  $$(".tab").forEach((s) => s.classList.toggle("active", s.id === `tab-${name}`));
}

$$("nav button").forEach((btn) => {
  btn.addEventListener("click", () => switchTab(btn.dataset.tab));
});

async function pingHealth() {
  const el = $("#health-status");
  const version = $("#version");
  try {
    const r = await fetch("/api/health");
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const data = await r.json();
    el.textContent = `ok · data_root=${data.data_root}`;
    el.classList.add("ok");
    el.classList.remove("err");
    version.textContent = `v${data.version}`;
  } catch (err) {
    el.textContent = `health check failed: ${err.message}`;
    el.classList.add("err");
    el.classList.remove("ok");
  }
}

pingHealth();
