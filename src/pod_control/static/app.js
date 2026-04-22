// pod_control M1 baseline + M3 Prepare tab.
// Pods/Run/Monitor land in M4-M6.

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

function switchTab(name) {
  $$("nav button").forEach((b) =>
    b.classList.toggle("active", b.dataset.tab === name)
  );
  $$(".tab").forEach((s) =>
    s.classList.toggle("active", s.id === `tab-${name}`)
  );
}

$$("nav button").forEach((btn) => {
  btn.addEventListener("click", () => switchTab(btn.dataset.tab));
});

async function jsonOrThrow(r) {
  if (!r.ok) {
    let detail = "";
    try {
      const body = await r.json();
      detail = body?.detail?.error?.message || JSON.stringify(body);
    } catch {
      detail = await r.text();
    }
    throw new Error(`${r.status} ${detail}`);
  }
  return r.json();
}

async function pingHealth() {
  const el = $("#health-status");
  const version = $("#version");
  try {
    const data = await jsonOrThrow(await fetch("/api/health"));
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

// ── Prepare tab state ───────────────────────────────────────────────

const CATEGORY_DEFAULTS = ["single", "dominant", "multi"];
const CATEGORY_CHOICES = ["single", "dominant", "multi", "wide", "landscape"];

const state = {
  currentMovie: null,
  page: 1,
  pageSize: 20,
};

function renderCategoryChecks() {
  const host = $("#category-checks");
  host.innerHTML = "";
  for (const cat of CATEGORY_CHOICES) {
    const id = `cat-${cat}`;
    const wrap = document.createElement("label");
    wrap.innerHTML = `<input type="checkbox" id="${id}" value="${cat}" ${
      CATEGORY_DEFAULTS.includes(cat) ? "checked" : ""
    } /> ${cat}`;
    host.appendChild(wrap);
  }
}

function currentFilterQuery() {
  const cats = $$("#category-checks input:checked").map((el) => el.value);
  const params = new URLSearchParams();
  params.set("categories", cats.join(","));
  params.set("skip_bad_quality", $("#skip-bad-quality").checked);
  params.set("skip_landscape", $("#skip-landscape").checked);
  const max = $("#max-shots").value.trim();
  if (max) params.set("max_shots", max);
  params.set("page", state.page);
  params.set("page_size", state.pageSize);
  return params;
}

function currentFilterParams() {
  return {
    categories: $$("#category-checks input:checked").map((el) => el.value),
    skip_bad_quality: $("#skip-bad-quality").checked,
    skip_landscape: $("#skip-landscape").checked,
    max_shots: $("#max-shots").value.trim()
      ? parseInt($("#max-shots").value, 10)
      : null,
  };
}

async function loadMovies() {
  const ul = $("#movie-list");
  ul.innerHTML = `<li class="muted">loading…</li>`;
  try {
    const { movies } = await jsonOrThrow(await fetch("/api/movies"));
    ul.innerHTML = "";
    if (!movies.length) {
      ul.innerHTML = `<li class="muted">no manifests found</li>`;
      return;
    }
    for (const m of movies) {
      const li = document.createElement("li");
      li.dataset.movie = m.movie;
      li.innerHTML = `<span>${m.movie}</span><span class="count">${m.total_shots} shots · ok=${m.quality_ok_count}</span>`;
      li.addEventListener("click", () => selectMovie(m.movie));
      ul.appendChild(li);
    }
  } catch (err) {
    ul.innerHTML = `<li class="muted">${err.message}</li>`;
  }
}

async function loadBatches() {
  const ul = $("#batch-list");
  ul.innerHTML = `<li class="muted">loading…</li>`;
  try {
    const { batches } = await jsonOrThrow(await fetch("/api/batches"));
    ul.innerHTML = "";
    if (!batches.length) {
      ul.innerHTML = `<li class="muted">no batches yet</li>`;
      return;
    }
    for (const b of batches) {
      const li = document.createElement("li");
      li.innerHTML = `<span>${b.name} <small class="muted">(${b.movie} · ${b.shot_count})</small></span>`;
      const del = document.createElement("button");
      del.className = "delete-btn";
      del.textContent = "delete";
      del.addEventListener("click", async (e) => {
        e.stopPropagation();
        if (!confirm(`delete batch ${b.name}?`)) return;
        try {
          const r = await fetch(`/api/batches/${b.name}`, { method: "DELETE" });
          if (!r.ok) throw new Error(`HTTP ${r.status}`);
          await loadBatches();
        } catch (err) {
          alert(err.message);
        }
      });
      li.appendChild(del);
      ul.appendChild(li);
    }
  } catch (err) {
    ul.innerHTML = `<li class="muted">${err.message}</li>`;
  }
}

function selectMovie(movie) {
  state.currentMovie = movie;
  state.page = 1;
  $$("#movie-list li").forEach((li) =>
    li.classList.toggle("active", li.dataset.movie === movie)
  );
  $("#empty-hint").hidden = true;
  $("#filter-panel").hidden = false;
  $("#current-movie").textContent = movie;
  applyFilter();
}

async function fetchPreview(extraParams = {}) {
  const params = currentFilterQuery();
  for (const [k, v] of Object.entries(extraParams)) params.set(k, v);
  return jsonOrThrow(
    await fetch(`/api/movies/${state.currentMovie}/preview?${params}`)
  );
}

function renderPreview(result) {
  const grid = $("#preview-grid");
  grid.innerHTML = "";
  if (!result.shots.length) {
    grid.innerHTML = `<p class="muted">no matching shots</p>`;
  }
  for (const shot of result.shots) {
    const card = document.createElement("div");
    card.className = "preview-card";
    const stem = shot.path.split("/").pop().replace(/\.mp4$/, "");
    const movie = shot.source_movie;
    card.innerHTML = `
      <video src="/clips/${movie}/${stem}.mp4" controls preload="metadata" muted></video>
      <div class="shot-meta">
        <span>${stem}</span>
        <span>${shot.shot_category} · ${shot.duration_sec.toFixed(1)}s</span>
      </div>`;
    grid.appendChild(card);
  }
  $("#match-count").textContent = `${result.total} matched`;
  $("#page-info").textContent = `page ${result.page}`;
  $("#page-prev").disabled = result.page <= 1 || result.sampled;
  $("#page-next").disabled =
    result.sampled || result.page * result.page_size >= result.total;
}

async function applyFilter() {
  if (!state.currentMovie) return;
  try {
    renderPreview(await fetchPreview());
  } catch (err) {
    $("#preview-grid").innerHTML = `<p class="muted">${err.message}</p>`;
  }
}

async function randomSample() {
  if (!state.currentMovie) return;
  try {
    const seed = Math.floor(Math.random() * 2 ** 31);
    renderPreview(await fetchPreview({ sample_seed: seed, page_size: 10 }));
  } catch (err) {
    $("#preview-grid").innerHTML = `<p class="muted">${err.message}</p>`;
  }
}

$("#apply-filter").addEventListener("click", () => {
  state.page = 1;
  applyFilter();
});
$("#random-sample").addEventListener("click", randomSample);
$("#page-prev").addEventListener("click", () => {
  state.page = Math.max(1, state.page - 1);
  applyFilter();
});
$("#page-next").addEventListener("click", () => {
  state.page += 1;
  applyFilter();
});

$("#save-batch-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const msg = $("#save-batch-msg");
  msg.textContent = "saving…";
  const name = $("#batch-name").value.trim();
  try {
    const r = await fetch("/api/batches", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name,
        movie: state.currentMovie,
        filter_params: currentFilterParams(),
      }),
    });
    const data = await jsonOrThrow(r);
    msg.textContent = `saved · ${data.shot_count} shots`;
    $("#batch-name").value = "";
    await loadBatches();
  } catch (err) {
    msg.textContent = err.message;
  }
});

// ── Pods tab state ───────────────────────────────────────────────

const podsState = {
  currentPod: null,   // null = creating new; object = editing existing
};

function dotClass(ok) {
  if (ok === true) return "test-dot ok";
  if (ok === false) return "test-dot err";
  return "test-dot";
}

async function loadPods() {
  const ul = $("#pods-list");
  ul.innerHTML = `<li class="muted">loading…</li>`;
  try {
    const { pods } = await jsonOrThrow(await fetch("/api/pods"));
    ul.innerHTML = "";
    if (!pods.length) {
      ul.innerHTML = `<li class="muted">no profiles yet</li>`;
      return;
    }
    for (const p of pods) {
      const li = document.createElement("li");
      li.dataset.pod = p.name;
      li.innerHTML = `
        <span><span class="${dotClass(p.last_test_ok)}"></span>${p.name}</span>
        <span class="count">${p.user}@${p.host}</span>`;
      li.addEventListener("click", () => selectPod(p));
      ul.appendChild(li);
    }
  } catch (err) {
    ul.innerHTML = `<li class="muted">${err.message}</li>`;
  }
}

function populatePodForm(pod) {
  const f = $("#pod-form");
  for (const key of ["name", "host", "user", "port", "ssh_key", "workspace"]) {
    f.elements[key].value = pod ? pod[key] ?? "" : "";
  }
  if (pod) {
    f.elements.name.setAttribute("readonly", "readonly");
  } else {
    f.elements.name.removeAttribute("readonly");
    f.elements.port.value = 22;
  }
}

function selectPod(pod) {
  podsState.currentPod = pod;
  $$("#pods-list li").forEach((li) =>
    li.classList.toggle("active", li.dataset.pod === pod.name)
  );
  $("#pod-form-title").textContent = `Edit · ${pod.name}`;
  populatePodForm(pod);
  $("#pod-test-btn").disabled = false;
  $("#pod-delete-btn").disabled = false;
  $("#pod-form-msg").textContent = "";
}

function resetPodForm() {
  podsState.currentPod = null;
  $$("#pods-list li").forEach((li) => li.classList.remove("active"));
  $("#pod-form-title").textContent = "New pod profile";
  populatePodForm(null);
  $("#pod-test-btn").disabled = true;
  $("#pod-delete-btn").disabled = true;
  $("#pod-form-msg").textContent = "";
}

$("#new-pod-btn").addEventListener("click", resetPodForm);

$("#pod-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const msg = $("#pod-form-msg");
  msg.textContent = "saving…";
  const f = e.currentTarget;
  const payload = {
    name: f.elements.name.value.trim(),
    host: f.elements.host.value.trim(),
    user: f.elements.user.value.trim(),
    port: parseInt(f.elements.port.value, 10) || 22,
    ssh_key: f.elements.ssh_key.value.trim(),
    workspace: f.elements.workspace.value.trim(),
  };
  const editing = podsState.currentPod != null;
  const url = editing
    ? `/api/pods/${encodeURIComponent(payload.name)}`
    : "/api/pods";
  const method = editing ? "PUT" : "POST";
  try {
    const r = await fetch(url, {
      method,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await jsonOrThrow(r);
    msg.textContent = editing ? "updated" : "saved";
    await loadPods();
    selectPod(data);
  } catch (err) {
    msg.textContent = err.message;
  }
});

$("#pod-test-btn").addEventListener("click", async () => {
  if (!podsState.currentPod) return;
  const msg = $("#pod-form-msg");
  msg.textContent = "testing…";
  try {
    const r = await fetch(
      `/api/pods/${encodeURIComponent(podsState.currentPod.name)}/test`,
      { method: "POST" }
    );
    const data = await jsonOrThrow(r);
    msg.textContent = data.ok
      ? `ok · ${data.latency_ms}ms`
      : `fail · ${data.message}`;
    await loadPods();
  } catch (err) {
    msg.textContent = err.message;
  }
});

$("#pod-delete-btn").addEventListener("click", async () => {
  if (!podsState.currentPod) return;
  if (!confirm(`delete ${podsState.currentPod.name}?`)) return;
  try {
    const r = await fetch(
      `/api/pods/${encodeURIComponent(podsState.currentPod.name)}`,
      { method: "DELETE" }
    );
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    await loadPods();
    resetPodForm();
  } catch (err) {
    $("#pod-form-msg").textContent = err.message;
  }
});

// ── Run tab state ────────────────────────────────────────────────

let runPollTimer = null;

function fmtEpoch(t) {
  if (!t) return "—";
  const d = new Date(t * 1000);
  return d.toLocaleString();
}

async function populateRunSelectors() {
  const [batchesR, podsR] = await Promise.all([
    fetch("/api/batches"), fetch("/api/pods"),
  ]);
  const batches = (await batchesR.json()).batches || [];
  const pods = (await podsR.json()).pods || [];
  const batchSel = $("#run-batch-sel");
  batchSel.innerHTML = batches
    .filter((b) => b.status === "ready")
    .map((b) => `<option value="${b.name}">${b.name} (${b.movie}, ${b.shot_count})</option>`)
    .join("");
  const podSel = $("#run-pod-sel");
  podSel.innerHTML = pods
    .map((p) => `<option value="${p.name}">${p.name} (${p.user}@${p.host})</option>`)
    .join("");
}

function renderActive(active) {
  const panel = $("#active-run-panel");
  const killBtn = $("#run-kill-btn");
  if (!active) {
    panel.innerHTML = `<span class="muted">none</span>`;
    killBtn.disabled = true;
    killBtn.dataset.runId = "";
    return;
  }
  panel.innerHTML = `
    <dl class="run-meta">
      <div><dt>id</dt><dd>${active.id}</dd></div>
      <div><dt>batch</dt><dd>${active.batch_name}</dd></div>
      <div><dt>pod</dt><dd>${active.pod_name}</dd></div>
      <div><dt>status</dt><dd class="run-status ${active.status}">${active.status}</dd></div>
      <div><dt>pid</dt><dd>${active.pid ?? "—"}</dd></div>
      <div><dt>started</dt><dd>${fmtEpoch(active.started_at)}</dd></div>
    </dl>`;
  killBtn.disabled = false;
  killBtn.dataset.runId = active.id;
}

function renderHistory(history) {
  const ul = $("#run-history");
  if (!history.length) {
    ul.innerHTML = `<li class="muted">no runs yet</li>`;
    return;
  }
  ul.innerHTML = history
    .map((h) => `
      <li>
        <span>${h.id}</span>
        <span class="count">
          ${h.batch_name} · <span class="run-status ${h.status}">${h.status}</span>
          · exit=${h.exit_code ?? "—"}
        </span>
      </li>`)
    .join("");
}

async function refreshRuns() {
  try {
    const data = await jsonOrThrow(await fetch("/api/runs"));
    renderActive(data.active[0] || null);
    renderHistory(data.history || []);
  } catch (err) {
    $("#active-run-panel").textContent = err.message;
  }
}

function startRunPolling() {
  if (runPollTimer) return;
  runPollTimer = setInterval(refreshRuns, 3000);
}
function stopRunPolling() {
  if (runPollTimer) clearInterval(runPollTimer);
  runPollTimer = null;
}

$("#run-launch-btn").addEventListener("click", async () => {
  const msg = $("#run-form-msg");
  msg.textContent = "launching…";
  try {
    const r = await fetch("/api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        batch_name: $("#run-batch-sel").value,
        pod_name: $("#run-pod-sel").value,
        preset_path: $("#run-preset").value.trim() || null,
      }),
    });
    const data = await jsonOrThrow(r);
    msg.textContent = `launched run ${data.id}`;
    await refreshRuns();
    await populateRunSelectors();
    startRunPolling();
  } catch (err) {
    msg.textContent = err.message;
  }
});

$("#run-kill-btn").addEventListener("click", async () => {
  const runId = $("#run-kill-btn").dataset.runId;
  if (!runId) return;
  if (!confirm(`kill run ${runId}?`)) return;
  try {
    const r = await fetch(`/api/runs/${encodeURIComponent(runId)}/kill`,
                          { method: "POST" });
    await jsonOrThrow(r);
    await refreshRuns();
    await populateRunSelectors();
  } catch (err) {
    $("#run-form-msg").textContent = err.message;
  }
});

// Refresh the selectors + poll whenever user switches to the Run tab.
document
  .querySelector('nav button[data-tab="run"]')
  .addEventListener("click", async () => {
    await populateRunSelectors();
    await refreshRuns();
    startRunPolling();
  });

// ── Bootstrap --------------------------------------------------------

renderCategoryChecks();
resetPodForm();
pingHealth();
loadMovies();
loadBatches();
loadPods();
populateRunSelectors();
refreshRuns();
