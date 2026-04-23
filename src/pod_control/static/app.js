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
  selectedMovies: new Set(),
  page: 1,
  pageSize: 20,
};

function renderCategoryChecks() {
  const host = $("#category-checks");
  host.innerHTML = "";
  for (const cat of CATEGORY_CHOICES) {
    const id = `cat-${cat}`;
    const label = document.createElement("label");
    label.className = "chip";
    const checked = CATEGORY_DEFAULTS.includes(cat);
    if (checked) label.classList.add("on");
    label.innerHTML =
      `<input type="checkbox" id="${id}" value="${cat}" ${checked ? "checked" : ""} /> ${cat}`;
    label.querySelector("input").addEventListener("change", (e) => {
      label.classList.toggle("on", e.target.checked);
    });
    host.appendChild(label);
  }
}

function _floatOrNull(id) {
  const v = $(id).value.trim();
  return v ? parseFloat(v) : null;
}

function currentFilterQuery() {
  const cats = $$("#category-checks input:checked").map((el) => el.value);
  const params = new URLSearchParams();
  params.set("categories", cats.join(","));
  params.set("skip_bad_quality", $("#skip-bad-quality").checked);
  params.set("skip_landscape", $("#skip-landscape").checked);
  const max = $("#max-shots").value.trim();
  if (max) params.set("max_shots", max);
  const minDur = _floatOrNull("#min-duration");
  if (minDur !== null) params.set("min_duration_sec", minDur);
  const maxDur = _floatOrNull("#max-duration");
  if (maxDur !== null) params.set("max_duration_sec", maxDur);
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
    min_duration_sec: _floatOrNull("#min-duration"),
    max_duration_sec: _floatOrNull("#max-duration"),
  };
}

// Full movie list (cached) so filter/search + select-all work without refetch.
let _moviesCache = [];

function renderMovieList() {
  const ul = $("#movie-list");
  const q = ($("#movie-search").value || "").trim().toLowerCase();
  ul.innerHTML = "";
  const visible = _moviesCache.filter(
    (m) => !q || m.movie.toLowerCase().includes(q)
  );
  if (!visible.length) {
    ul.innerHTML = `<li class="muted">${
      _moviesCache.length ? "no match" : "no manifests found"
    }</li>`;
  }
  for (const m of visible) {
    const li = document.createElement("li");
    li.dataset.movie = m.movie;
    const isChecked = state.selectedMovies.has(m.movie);
    if (isChecked) li.classList.add("active");
    li.innerHTML = `
      <label>
        <input type="checkbox" class="movie-cb" value="${m.movie}"${isChecked ? " checked" : ""} />
        <span class="movie-name">${m.movie}</span>
      </label>
      <span class="count">${m.total_shots} · ok=${m.quality_ok_count}</span>`;
    li.querySelector(".movie-cb").addEventListener("change", (e) => {
      e.stopPropagation();
      toggleMovie(m.movie, e.target.checked);
    });
    li.addEventListener("click", (e) => {
      // If the click already landed on the checkbox/label, browser will
      // toggle it and fire change — don't double-handle.
      if (e.target.tagName === "INPUT"
          || e.target.tagName === "LABEL"
          || e.target.classList.contains("movie-name")) return;
      const cb = li.querySelector(".movie-cb");
      cb.checked = !cb.checked;
      toggleMovie(m.movie, cb.checked);
    });
    ul.appendChild(li);
  }
  updateMoviesCount();
}

function updateMoviesCount() {
  const selected = state.selectedMovies.size;
  const total = _moviesCache.length;
  $("#movies-count").textContent =
    selected > 0 ? `${selected}/${total} selected` : `${total} total`;
}

async function loadMovies() {
  const ul = $("#movie-list");
  ul.innerHTML = `<li class="muted">loading…</li>`;
  try {
    const { movies } = await jsonOrThrow(await fetch("/api/movies"));
    _moviesCache = movies;
    // Drop any selected movies that no longer exist (e.g. output root changed).
    const known = new Set(movies.map((m) => m.movie));
    for (const m of [...state.selectedMovies]) {
      if (!known.has(m)) state.selectedMovies.delete(m);
    }
    renderMovieList();
  } catch (err) {
    ul.innerHTML = `<li class="muted">${err.message}</li>`;
  }
}

$("#movie-search").addEventListener("input", renderMovieList);

$("#movies-select-all").addEventListener("click", () => {
  // Respect the current search filter — only select what's visible.
  const q = ($("#movie-search").value || "").trim().toLowerCase();
  for (const m of _moviesCache) {
    if (!q || m.movie.toLowerCase().includes(q)) {
      state.selectedMovies.add(m.movie);
    }
  }
  _afterBulkSelectionChange();
});

$("#movies-clear").addEventListener("click", () => {
  state.selectedMovies.clear();
  _afterBulkSelectionChange();
});

function _afterBulkSelectionChange() {
  state.page = 1;
  renderMovieList();
  const count = state.selectedMovies.size;
  if (count === 0) {
    $("#empty-hint").hidden = false;
    $("#filter-panel").hidden = true;
    return;
  }
  $("#empty-hint").hidden = true;
  $("#filter-panel").hidden = false;
  $("#current-movie").textContent = count === 1
    ? [...state.selectedMovies][0]
    : `${count} movies`;
  applyFilter();
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
      li.className = "batch-row";
      li.dataset.name = b.name;
      const moviesLabel = (b.movies && b.movies.length > 1)
        ? `${b.movies.length} movies`
        : (b.movies?.[0] ?? b.movie ?? "?");
      li.innerHTML = `
        <span class="batch-main">
          <span class="batch-name">${b.name}</span>
          <small class="muted">(${moviesLabel} · ${b.shot_count})</small>
          <span class="status-badge status-${b.status}">${b.status}</span>
        </span>`;
      li.addEventListener("click", (e) => {
        if (e.target.closest(".delete-btn")) return;
        loadBatchIntoFilter(b);
      });
      const del = document.createElement("button");
      del.className = "delete-btn";
      del.textContent = "delete";
      del.addEventListener("click", async (e) => {
        e.stopPropagation();
        if (!confirm(`delete batch ${b.name}?`)) return;
        try {
          const r = await fetch(`/api/batches/${b.name}`, { method: "DELETE" });
          if (!r.ok) {
            const body = await r.json().catch(() => ({}));
            throw new Error(body.detail || `HTTP ${r.status}`);
          }
          await loadBatches();
        } catch (err) {
          alert(err.message);
        }
      });
      li.appendChild(del);
      ul.appendChild(li);
    }
    _applyBatchSearch?.();
  } catch (err) {
    ul.innerHTML = `<li class="muted">${err.message}</li>`;
  }
}

// Load a saved batch's movies + filter params back into the filter panel.
function loadBatchIntoFilter(batch) {
  // 1. Highlight the selected row.
  $$("#batch-list li").forEach((li) => {
    li.classList.toggle("batch-active", li.dataset.name === batch.name);
  });

  // 2. Reset + set movie selection from batch.movies.
  state.selectedMovies = new Set(batch.movies || []);
  $$("#movie-list input[type=checkbox]").forEach((cb) => {
    cb.checked = state.selectedMovies.has(cb.value);
    cb.closest("li")?.classList.toggle("active", cb.checked);
  });
  updateMoviesCount?.();

  // 3. Populate filter inputs from batch.filter_params.
  const fp = batch.filter_params || {};
  const cats = new Set(fp.categories || []);
  $$("#category-checks input[type=checkbox]").forEach((cb) => {
    cb.checked = cats.has(cb.value);
    cb.closest("label.chip")?.classList.toggle("on", cb.checked);
  });
  $("#skip-bad-quality").checked = fp.skip_bad_quality !== false;
  $("#skip-landscape").checked = fp.skip_landscape !== false;
  $("#max-shots").value = fp.max_shots ?? "";
  $("#min-duration").value = fp.min_duration_sec ?? "";
  $("#max-duration").value = fp.max_duration_sec ?? "";

  // 4. Show filter panel, hide empty hint.
  const count = state.selectedMovies.size;
  $("#empty-hint").hidden = count > 0;
  $("#filter-panel").hidden = count === 0;
  $("#current-movie").textContent = count === 1
    ? [...state.selectedMovies][0]
    : `${count} movies · ${batch.name}`;

  state.page = 1;
  if (count > 0) applyFilter();
}

function toggleMovie(movie, selected) {
  if (selected) state.selectedMovies.add(movie);
  else state.selectedMovies.delete(movie);
  state.page = 1;
  $$("#movie-list li").forEach((li) =>
    li.classList.toggle("active", state.selectedMovies.has(li.dataset.movie))
  );
  updateMoviesCount();
  const count = state.selectedMovies.size;
  if (count === 0) {
    $("#empty-hint").hidden = false;
    $("#filter-panel").hidden = true;
    return;
  }
  $("#empty-hint").hidden = true;
  $("#filter-panel").hidden = false;
  $("#current-movie").textContent = count === 1
    ? [...state.selectedMovies][0]
    : `${count} movies`;
  applyFilter();
}

async function fetchPreview(extraParams = {}) {
  const selected = [...state.selectedMovies];
  if (selected.length === 0) {
    return { shots: [], total: 0, page: 1, page_size: state.pageSize };
  }
  const [first, ...rest] = selected;
  const params = currentFilterQuery();
  if (rest.length) params.set("movies", rest.join(","));
  for (const [k, v] of Object.entries(extraParams)) params.set(k, v);
  return jsonOrThrow(
    await fetch(`/api/movies/${encodeURIComponent(first)}/preview?${params}`)
  );
}

function _hasSelection() {
  return state.selectedMovies.size > 0;
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
  if (!_hasSelection()) return;
  try {
    renderPreview(await fetchPreview());
  } catch (err) {
    $("#preview-grid").innerHTML = `<p class="muted">${err.message}</p>`;
  }
}

async function randomSample() {
  if (!_hasSelection()) return;
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
        movies: [...state.selectedMovies],
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

// ── Direct launch (Prepare) ─────────────────────────────────────

async function populateDirectPodSel() {
  try {
    const { pods } = await jsonOrThrow(await fetch("/api/pods"));
    const sel = $("#direct-pod-sel");
    if (!pods.length) {
      sel.innerHTML = `<option value="">no pods — add one in Pods tab</option>`;
      sel.disabled = true;
      $("#direct-launch-btn").disabled = true;
      return;
    }
    sel.innerHTML = pods
      .map((p) => `<option value="${p.name}">${p.name} (${p.user}@${p.host})</option>`)
      .join("");
    sel.disabled = false;
    $("#direct-launch-btn").disabled = false;
  } catch (err) {
    $("#direct-launch-msg").textContent = err.message;
  }
}

$("#direct-launch-btn").addEventListener("click", async () => {
  const msg = $("#direct-launch-msg");
  const movies = [...state.selectedMovies];
  if (!movies.length) {
    msg.textContent = "pick at least one movie first";
    return;
  }
  const podName = $("#direct-pod-sel").value;
  if (!podName) {
    msg.textContent = "pick a pod first";
    return;
  }
  const label = movies.length === 1 ? `movie "${movies[0]}"`
                                    : `${movies.length} movies`;
  if (!confirm(`launch inference on ${label} using pod "${podName}"?`)) {
    return;
  }
  msg.textContent = "launching…";
  try {
    const r = await fetch("/api/runs/quick", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        movies,
        pod_name: podName,
        preset_path: $("#direct-preset").value.trim() || null,
        filter_params: currentFilterParams(),
      }),
    });
    const data = await jsonOrThrow(r);
    msg.textContent = `launched run ${data.id} — switching to Monitor…`;
    await loadBatches();
    await populateRunSelectors();
    await refreshRuns();
    // Auto-hop to Monitor tab so the user sees the log immediately.
    switchTab("monitor");
    await monitorInit();
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
  if (!batches.length) {
    batchSel.innerHTML = `<option value="">no batches saved — go to Prepare</option>`;
  } else {
    batchSel.innerHTML = batches.map((b) => {
      const moviesLabel = b.movies?.length > 1
        ? `${b.movies.length} movies`
        : (b.movies?.[0] ?? b.movie ?? "?");
      const disabled = b.status !== "ready" ? " disabled" : "";
      return `<option value="${b.name}" data-status="${b.status}"${disabled}>${b.name} · ${moviesLabel} · ${b.shot_count} shots [${b.status}]</option>`;
    }).join("");
  }
  // Pick the first ready option (first non-disabled).
  for (const opt of batchSel.options) {
    if (!opt.disabled && opt.value) {
      opt.selected = true;
      break;
    }
  }
  const podSel = $("#run-pod-sel");
  podSel.innerHTML = pods.length
    ? pods.map((p) => `<option value="${p.name}">${p.name} (${p.user}@${p.host})</option>`).join("")
    : `<option value="">no pods — go to Pods</option>`;

  // If a non-ready batch is currently selected (or none ready exists),
  // surface a reset shortcut next to the selector.
  _renderResetHint(batches);
}

function _renderResetHint(batches) {
  const hint = $("#run-reset-hint");
  if (!hint) return;
  const stuck = batches.filter((b) => b.status === "failed" || b.status === "done");
  if (!stuck.length) {
    hint.innerHTML = "";
    return;
  }
  hint.innerHTML = `
    <span class="muted">Stuck batches:</span>
    ${stuck.map((b) => `
      <button class="inline-btn" data-reset="${b.name}" type="button">
        reset ${b.name} (${b.status})
      </button>`).join("")}`;
  hint.querySelectorAll("button[data-reset]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const name = btn.dataset.reset;
      try {
        const r = await fetch(`/api/batches/${encodeURIComponent(name)}/reset`,
                              { method: "POST" });
        await jsonOrThrow(r);
        await populateRunSelectors();
        await loadBatches();
      } catch (err) {
        alert(err.message);
      }
    });
  });
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
    _refreshBulkDeleteState();
    return;
  }
  ul.innerHTML = history.map((h) => `
    <li class="hist-row" data-id="${h.id}" data-search="${h.id} ${h.batch_name} ${h.pod_name}">
      <input type="checkbox" class="hist-check" data-id="${h.id}" />
      <span class="hist-meta">
        <span class="hist-id">${h.id}</span>
        <span class="count">
          ${h.batch_name} · ${h.pod_name} ·
          <span class="run-status ${h.status}">${h.status}</span>
          · exit=${h.exit_code ?? "—"}
        </span>
      </span>
      <button class="delete-btn hist-del-one" data-id="${h.id}" type="button">delete</button>
    </li>`).join("");

  ul.querySelectorAll(".hist-check").forEach((cb) => {
    cb.addEventListener("change", _refreshBulkDeleteState);
  });
  ul.querySelectorAll(".hist-del-one").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const id = btn.dataset.id;
      if (!confirm(`delete history entry ${id}?`)) return;
      try {
        const r = await fetch(`/api/runs/${encodeURIComponent(id)}`,
                              { method: "DELETE" });
        if (!r.ok && r.status !== 204) throw new Error(`HTTP ${r.status}`);
        await refreshRuns();
      } catch (err) {
        alert(err.message);
      }
    });
  });
  _applyHistorySearch();
  _refreshBulkDeleteState();
}

function _refreshBulkDeleteState() {
  const btn = $("#hist-bulk-delete");
  if (!btn) return;
  const checked = document.querySelectorAll(".hist-check:checked").length;
  btn.disabled = checked === 0;
  btn.textContent = checked > 0 ? `Delete selected (${checked})` : "Delete selected";
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

// ── Clear history ─────────────────────────────────────────────────────
$("#clear-history-btn").addEventListener("click", async () => {
  if (!confirm("Clear all run history? This cannot be undone.")) return;
  try {
    await fetch("/api/runs", { method: "DELETE" });
    await refreshRuns();
  } catch (err) {
    alert(err.message);
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

// Refresh the direct-launch pod dropdown whenever user re-enters Prepare.
document
  .querySelector('nav button[data-tab="prepare"]')
  .addEventListener("click", () => {
    populateDirectPodSel();
  });

// ── Monitor tab state ───────────────────────────────────────────────

const monitorState = {
  runId: null,
  localOffset: 0,
  accLog: "",        // accumulated stdout.log content for stage parsing
  startedAt: null,
  timer: null,
  podName: null,      // for pod-direct log fallback
  podLogOffset: 0,    // separate offset for remote pod_runner.log
};

function monitorStop() {
  if (monitorState.timer) clearInterval(monitorState.timer);
  monitorState.timer = null;
}

function monitorLoop() {
  monitorStop();
  monitorState.timer = setInterval(monitorPoll, 2000);
}

// Derive current stage from accumulated log text.
function detectStage(log) {
  let stage = null;
  for (const line of log.split("\n")) {
    const m = line.match(/\[pod_control:stage=(\w+)\]/);
    if (m) stage = m[1];
  }
  return stage;
}

function renderStageStepper(stage, checkpoint) {
  const steps = ["push", "run", "pull", "done"];
  const stageOrder = { push: 0, run: 1, pull: 2, done: 3 };
  const currentIdx = stageOrder[stage] ?? -1;

  // Decide label for "run" step based on checkpoint
  const ck = checkpoint || { done: 0, pending: 0, failed: 0 };
  const inInference = stage === "run" && ck.done > 0;

  for (const step of steps) {
    const el = document.querySelector(`.stage-step[data-stage="${step}"]`);
    if (!el) continue;
    const idx = stageOrder[step];
    el.classList.remove("step-done", "step-active", "step-pending");
    const icon = el.querySelector(".stage-icon");
    if (idx < currentIdx) {
      el.classList.add("step-done");
      icon.textContent = "✓";
    } else if (idx === currentIdx) {
      el.classList.add("step-active");
      icon.textContent = "→";
    } else {
      el.classList.add("step-pending");
      icon.textContent = "·";
    }
    if (step === "run") {
      el.querySelector(".stage-label").textContent =
        inInference ? "Inference" : (stage === "run" ? "Setup" : "Run");
    }
  }

  // Inference progress bar
  const progressSection = $("#monitor-inference-progress");
  if (stage === "run" && inInference) {
    progressSection.hidden = false;
    const total = ck.done + ck.failed + ck.pending;
    const processed = ck.done + ck.failed;
    const pct = total > 0 ? Math.round((processed / total) * 100) : 0;
    $("#monitor-progress-fill").style.width = `${pct}%`;

    let etaStr = "";
    if (monitorState.startedAt && processed > 0 && ck.pending > 0) {
      const elapsedSec = (Date.now() / 1000) - monitorState.startedAt;
      const rate = processed / elapsedSec;
      const etaSec = Math.round(ck.pending / rate);
      etaStr = etaSec < 60
        ? ` · ETA ~${etaSec}s`
        : ` · ETA ~${Math.round(etaSec / 60)}m`;
    }
    const elapsedMin = monitorState.startedAt
      ? Math.round((Date.now() / 1000 - monitorState.startedAt) / 60)
      : 0;
    $("#monitor-progress-stats").textContent =
      `done ${ck.done} · failed ${ck.failed} · pending ${ck.pending}` +
      ` · ${pct}% · elapsed ${elapsedMin}m${etaStr}`;
  } else {
    progressSection.hidden = true;
  }
}

async function monitorInit() {
  monitorStop();
  try {
    // Get both active and history in one call.
    const runs = await jsonOrThrow(await fetch("/api/runs"));
    const active_run = runs.active?.[0] || null;
    const run = active_run || runs.history?.[0] || null;

    const pullBtn = $("#monitor-pull-btn");
    if (!run) {
      $("#monitor-summary").innerHTML = `<span class="muted">no active run</span>`;
      $("#monitor-status").textContent = "";
      $("#monitor-log").textContent = "";
      $("#monitor-stages-panel").hidden = true;
      monitorState.runId = null;
      monitorState.localOffset = 0;
      monitorState.accLog = "";
      monitorState.podName = null;
      monitorState.podLogOffset = 0;
      pullBtn.disabled = true;
      return;
    }

    monitorState.runId = run.id;
    monitorState.localOffset = 0;
    monitorState.accLog = "";
    monitorState.startedAt = run.started_at || null;
    monitorState.podName = run.pod_name;
    monitorState.podLogOffset = 0;
    pullBtn.disabled = false;
    pullBtn.dataset.runId = run.id;

    $("#monitor-log").textContent = "";
    $("#monitor-stages-panel").hidden = false;
    const isActive = active_run && active_run.id === run.id;
    const statusLabel = isActive ? run.status : `${run.status} (detached, pod may still be running)`;
    $("#monitor-summary").innerHTML = `
      <dl class="run-meta">
        <div><dt>id</dt><dd>${run.id}</dd></div>
        <div><dt>batch</dt><dd>${run.batch_name}</dd></div>
        <div><dt>pod</dt><dd>${run.pod_name}</dd></div>
        <div><dt>status</dt><dd class="run-status ${run.status}">${statusLabel}</dd></div>
        <div><dt>pid</dt><dd>${run.pid ?? "—"}</dd></div>
        <div><dt>started</dt><dd>${fmtEpoch(run.started_at)}</dd></div>
      </dl>`;

    await monitorPoll();
    // Always keep polling: even detached runs can be tracked via pod-direct.
    monitorLoop();
  } catch (err) {
    $("#monitor-summary").textContent = err.message;
  }
}

async function monitorPoll() {
  if (!monitorState.runId) return;
  const id = encodeURIComponent(monitorState.runId);
  const podName = monitorState.podName;
  try {
    // 1. Incremental LOCAL log (stdout.log — has bash stage markers + SSH tail).
    const localR = await fetch(`/api/runs/${id}/local-tail?offset=${monitorState.localOffset}`);
    const localBody = await jsonOrThrow(localR);
    if (localBody.text) {
      monitorState.accLog += localBody.text;
      const pre = $("#monitor-log");
      pre.innerHTML += highlightLogChunk(localBody.text);
      if ($("#monitor-autoscroll").checked) pre.scrollTop = pre.scrollHeight;
    }
    monitorState.localOffset = localBody.next_offset;
    $("#monitor-offset").textContent = `offset ${localBody.next_offset}`;

    // 2. POD-DIRECT log tail (independent of active_run) — picks up new
    //    pod_runner.log output even when local bash already died.
    if (podName && localBody.finished) {
      try {
        const podR = await fetch(
          `/api/pods/${encodeURIComponent(podName)}/log-tail` +
          `?offset=${monitorState.podLogOffset}`
        );
        const podBody = await jsonOrThrow(podR);
        if (podBody.text) {
          monitorState.accLog += podBody.text;
          const pre = $("#monitor-log");
          pre.innerHTML += `\n<span class="log-stage-marker">[pod-direct]</span> ` +
                           highlightLogChunk(podBody.text);
          if ($("#monitor-autoscroll").checked) pre.scrollTop = pre.scrollHeight;
        }
        monitorState.podLogOffset = podBody.next_offset;
      } catch (_) { /* pod unreachable; keep polling */ }
    }

    // 3. Checkpoint: prefer pod-direct (works for detached runs too).
    let checkpoint = { done: 0, failed: 0, pending: 0 };
    if (podName) {
      try {
        const ckR = await fetch(`/api/pods/${encodeURIComponent(podName)}/checkpoint`);
        checkpoint = await jsonOrThrow(ckR);
      } catch (_) { /* ssh may be unavailable */ }
    }

    // 4. Stage detection + stepper render.
    const stage = detectStage(monitorState.accLog);
    if (stage) renderStageStepper(stage, checkpoint);
    $("#monitor-status").textContent = stage ? `stage: ${stage}` : "";
  } catch (err) {
    $("#monitor-status").textContent = err.message;
  }
}

// Pull results button
$("#monitor-pull-btn").addEventListener("click", async () => {
  const btn = $("#monitor-pull-btn");
  const runId = btn.dataset.runId || monitorState.runId;
  if (!runId) return;
  btn.disabled = true;
  btn.textContent = "Pulling…";
  try {
    await fetch(`/api/runs/${encodeURIComponent(runId)}/pull`, { method: "POST" });
  } catch (err) {
    alert(err.message);
  }
  setTimeout(() => {
    btn.disabled = false;
    btn.textContent = "Pull results";
  }, 2000);
});

document
  .querySelector('nav button[data-tab="monitor"]')
  .addEventListener("click", monitorInit);

// ── Output-root setting ─────────────────────────────────────────────

async function loadOutputRoot() {
  try {
    const data = await jsonOrThrow(
      await fetch("/api/settings/output-root")
    );
    const sel = $("#output-root-sel");
    const opts = [...new Set([data.current, ...data.candidates].filter(Boolean))];
    sel.innerHTML = opts
      .map((p) => `<option value="${p}" ${p === data.current ? "selected" : ""}>${p}</option>`)
      .join("");
    if (!opts.length) {
      sel.innerHTML = `<option value="">(no candidates)</option>`;
    }
    $("#output-root-input").value = "";
    $("#output-root-msg").textContent = data.current
      ? `current: ${data.current}`
      : "not configured";
  } catch (err) {
    $("#output-root-msg").textContent = err.message;
  }
}

$("#output-root-apply").addEventListener("click", async () => {
  const typed = $("#output-root-input").value.trim();
  const picked = $("#output-root-sel").value;
  const path = typed || picked;
  if (!path) {
    $("#output-root-msg").textContent = "pick from list or type a path";
    return;
  }
  $("#output-root-msg").textContent = "applying…";
  try {
    const r = await fetch("/api/settings/output-root", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    });
    const data = await jsonOrThrow(r);
    $("#output-root-msg").textContent = `switched to ${data.current}`;
    // Reload dependent panels so the new root takes effect visibly.
    await loadOutputRoot();
    await loadMovies();
    await loadBatches();
    pingHealth();
  } catch (err) {
    $("#output-root-msg").textContent = err.message;
  }
});

// ── Bootstrap --------------------------------------------------------

// ── Preset selector + YAML editor ───────────────────────────────────

let _presets = [];

async function loadPresets() {
  const sel = $("#run-preset-sel");
  if (!sel) return;
  try {
    const { configs } = await jsonOrThrow(await fetch("/api/configs"));
    _presets = configs || [];
    if (!_presets.length) {
      sel.innerHTML = `<option value="">(no presets in configs/)</option>`;
      $("#run-preset").value = "";
      $("#run-preset-meta").textContent = "";
      return;
    }
    sel.innerHTML = _presets.map((p) => {
      const label = p.model ? `${p.name} — ${p.model}` : p.name;
      return `<option value="${p.name}">${label}</option>`;
    }).join("");
    _onPresetChange();
  } catch (err) {
    sel.innerHTML = `<option value="">${err.message}</option>`;
  }
}

function _onPresetChange() {
  const sel = $("#run-preset-sel");
  const name = sel.value;
  $("#run-preset").value = name ? `configs/${name}` : "";
  const p = _presets.find((x) => x.name === name);
  if (!p) {
    $("#run-preset-meta").textContent = "";
    return;
  }
  const parts = [];
  if (p.model) parts.push(`Model: ${p.model}`);
  if (p.max_model_len) parts.push(`Context: ${p.max_model_len}`);
  if (p.rounds) parts.push(`Rounds: ${p.rounds}`);
  if (p.error) parts.push(`⚠ ${p.error}`);
  $("#run-preset-meta").textContent = parts.join(" · ");
}

function _initYamlModal() {
  const sel = $("#run-preset-sel");
  if (sel) sel.addEventListener("change", _onPresetChange);

  const editBtn = $("#run-preset-edit-btn");
  const modal = $("#yaml-modal");
  const ta = $("#yaml-modal-text");
  const msg = $("#yaml-modal-msg");
  const title = $("#yaml-modal-title");

  if (!editBtn || !modal) return;

  function open() {
    modal.hidden = false;
    document.body.classList.add("modal-open");
    document.addEventListener("keydown", _escClose);
  }
  function close() {
    modal.hidden = true;
    document.body.classList.remove("modal-open");
    msg.textContent = "";
    document.removeEventListener("keydown", _escClose);
  }
  function _escClose(e) { if (e.key === "Escape") close(); }

  editBtn.addEventListener("click", async () => {
    const name = $("#run-preset-sel").value;
    if (!name) { alert("no preset selected"); return; }
    title.textContent = `Edit ${name}`;
    ta.value = "loading…";
    msg.textContent = "";
    open();
    try {
      const data = await jsonOrThrow(
        await fetch(`/api/configs/${encodeURIComponent(name)}`)
      );
      ta.value = data.raw_yaml;
      ta.dataset.name = name;
    } catch (err) {
      ta.value = "";
      msg.textContent = err.message;
    }
  });
  $("#yaml-modal-close").addEventListener("click", close);
  $("#yaml-modal-cancel").addEventListener("click", close);
  $("#yaml-modal-save").addEventListener("click", async () => {
    const name = ta.dataset.name;
    msg.textContent = "saving…";
    try {
      const r = await fetch(`/api/configs/${encodeURIComponent(name)}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ raw_yaml: ta.value }),
      });
      const data = await jsonOrThrow(r);
      msg.textContent = `saved · model=${data.meta?.model ?? "?"}`;
      await loadPresets();
      // restore selection
      const s = $("#run-preset-sel");
      if ([...s.options].some((o) => o.value === name)) s.value = name;
      _onPresetChange();
      setTimeout(close, 600);
    } catch (err) {
      msg.textContent = err.message;
    }
  });
  modal.addEventListener("click", (e) => {
    if (e.target === modal) close();
  });
}

// ── Search filters (batch list + history) ───────────────────────────

function _initSearchInputs() {
  const bs = $("#batch-search");
  if (bs) bs.addEventListener("input", () => _applyBatchSearch());
  const hs = $("#history-search");
  if (hs) hs.addEventListener("input", () => _applyHistorySearch());
}

function _applyBatchSearch() {
  const q = ($("#batch-search")?.value || "").toLowerCase().trim();
  document.querySelectorAll("#batch-list li").forEach((li) => {
    const txt = li.textContent.toLowerCase();
    li.hidden = q && !txt.includes(q);
  });
}

function _applyHistorySearch() {
  const q = ($("#history-search")?.value || "").toLowerCase().trim();
  document.querySelectorAll("#run-history .hist-row").forEach((li) => {
    const txt = (li.dataset.search || "").toLowerCase();
    li.hidden = q && !txt.includes(q);
  });
}

// ── Bulk delete history ─────────────────────────────────────────────

function _initBulkDelete() {
  const btn = $("#hist-bulk-delete");
  if (!btn) return;
  btn.addEventListener("click", async () => {
    const ids = [...document.querySelectorAll(".hist-check:checked")]
      .map((cb) => cb.dataset.id);
    if (!ids.length) return;
    if (!confirm(`delete ${ids.length} run(s) from history?`)) return;
    let failed = 0;
    for (const id of ids) {
      try {
        const r = await fetch(`/api/runs/${encodeURIComponent(id)}`,
                              { method: "DELETE" });
        if (!r.ok && r.status !== 204) failed++;
      } catch (_) { failed++; }
    }
    if (failed) alert(`${failed} delete(s) failed`);
    await refreshRuns();
  });
}

// ── Browser notifications ───────────────────────────────────────────

function _initNotifications() {
  const btn = $("#enable-notifs-btn");
  if (!btn || !("Notification" in window)) {
    if (btn) btn.hidden = true;
    return;
  }
  function refreshLabel() {
    const p = Notification.permission;
    btn.textContent = p === "granted" ? "🔔 On" : (p === "denied" ? "🔕 Blocked" : "🔔 Notify");
    btn.disabled = p === "denied";
  }
  refreshLabel();
  btn.addEventListener("click", async () => {
    if (Notification.permission === "default") {
      await Notification.requestPermission();
    }
    refreshLabel();
  });
}

function _fireRunNotification(record) {
  if (!("Notification" in window) || Notification.permission !== "granted") return;
  const title = `Run ${record.status}`;
  const body = `${record.batch_name} on ${record.pod_name} · exit=${record.exit_code ?? "—"}`;
  try { new Notification(title, { body, icon: "/favicon.ico" }); } catch (_) {}
}

let _lastSeenActiveId = sessionStorage.getItem("podc_lastActiveId") || null;

function _initMonitorRunWatcher() {
  // Piggyback on the existing 3s refreshRuns poll in startRunPolling.
  // Wrap refreshRuns once to detect transition from active → finished.
  const orig = refreshRuns;
  refreshRuns = async function watchedRefresh() {
    await orig();
    try {
      const data = await jsonOrThrow(await fetch("/api/runs"));
      const cur = data.active?.[0]?.id || null;
      if (_lastSeenActiveId && !cur) {
        // Active just disappeared → check if history[0] is the same id.
        const finished = data.history?.find((h) => h.id === _lastSeenActiveId);
        if (finished) _fireRunNotification(finished);
      }
      _lastSeenActiveId = cur;
      sessionStorage.setItem("podc_lastActiveId", cur || "");
    } catch (_) { /* polling errors handled in orig */ }
  };
}

// ── Log highlight (Monitor) ─────────────────────────────────────────

function _escapeHtml(s) {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function highlightLogChunk(text) {
  const lines = text.split("\n");
  const out = lines.map((line) => {
    const esc = _escapeHtml(line);
    if (/\[pod_control:stage=/.test(line)) {
      return `<span class="log-stage-marker">${esc}</span>`;
    }
    if (/(ERROR|Traceback|\[ERR\]|FAIL)/i.test(line)) {
      return `<span class="log-error">${esc}</span>`;
    }
    if (/(WARN|\[WARN\])/i.test(line)) {
      return `<span class="log-warn">${esc}</span>`;
    }
    return esc;
  });
  return out.join("\n");
}

renderCategoryChecks();
resetPodForm();
pingHealth();
loadOutputRoot();
loadMovies();
loadBatches();
loadPods();
populateDirectPodSel();
populateRunSelectors();
loadPresets();
refreshRuns();
_initSearchInputs();
_initBulkDelete();
_initYamlModal();
_initNotifications();
_initMonitorRunWatcher();
