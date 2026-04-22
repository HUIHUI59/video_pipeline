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

// ── Bootstrap --------------------------------------------------------

renderCategoryChecks();
pingHealth();
loadMovies();
loadBatches();
