"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

// ------------------------------------------------------------- BYOK key ---
//
// The public demo has no server-side OpenAI key (see README): only read-only
// endpoints work for free. Live actions (ask, build index, run evaluation)
// need a key. `EventSource` can't send custom headers, and a key in the URL
// query string would end up in server/CDN access logs — so BYOK is only sent
// as a request header on a fetch(), never in a URL. Session-only storage: it
// disappears when the tab closes, never touches localStorage or a cookie.
const BYOK_STORAGE_KEY = "rag_demo_openai_key";

function getByokKey() {
  return sessionStorage.getItem(BYOK_STORAGE_KEY) || "";
}

function setByokKey(key) {
  if (key) sessionStorage.setItem(BYOK_STORAGE_KEY, key);
  else sessionStorage.removeItem(BYOK_STORAGE_KEY);
}

function authHeaders() {
  const key = getByokKey();
  return key ? { "X-OpenAI-Key": key } : {};
}

// ------------------------------------------------------- fetch-based SSE ---
//
// Drop-in replacement for `new EventSource(url)` with the same interface
// (`addEventListener(name, handler)`, `.onerror`, `.close()`) the rest of
// this file already uses — the only reason it exists is that native
// EventSource has no way to attach the X-OpenAI-Key header above, since it
// only ever issues plain GET requests with no custom headers.
class FetchEventSource extends EventTarget {
  constructor(url, { headers = {} } = {}) {
    super();
    this._closed = false;
    this.onerror = null;
    this._run(url, headers);
  }

  async _run(url, headers) {
    let response;
    try {
      response = await fetch(url, { headers });
    } catch (err) {
      if (!this._closed && this.onerror) this.onerror(err);
      return;
    }
    if (!response.ok || !response.body) {
      if (!this._closed && this.onerror) this.onerror(new Error(`HTTP ${response.status}`));
      return;
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    try {
      while (!this._closed) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let idx;
        while ((idx = buffer.indexOf("\n\n")) !== -1) {
          const raw = buffer.slice(0, idx);
          buffer = buffer.slice(idx + 2);
          if (this._closed) break;
          let name = "message";
          let data = "";
          for (const line of raw.split("\n")) {
            if (line.startsWith("event:")) name = line.slice(6).trim();
            else if (line.startsWith("data:")) data += line.slice(5).trim();
          }
          if (data) this.dispatchEvent(new MessageEvent(name, { data }));
        }
      }
    } catch (err) {
      if (!this._closed && this.onerror) this.onerror(err);
    }
  }

  close() {
    this._closed = true;
  }
}

// ---------------------------------------------------------------- tabs ----

let exploreCodeInitialized = false;

function switchToTab(tabName) {
  const btn = $(`.tab-btn[data-tab="${tabName}"]`);
  if (!btn) return;
  $$(".tab-btn").forEach((b) => b.classList.remove("active"));
  $$(".tab-panel").forEach((p) => p.classList.remove("active"));
  btn.classList.add("active");
  $(`#tab-${tabName}`).classList.add("active");
  if (tabName === "explore") {
    // Refresh every time (not just the first visit): if the index got
    // rebuilt in the other tab in the meantime, a stale "not built yet"
    // or stale chunk list would be actively misleading here.
    renderDbSummary($("#db-summary"));
    loadDocPicker();
    $("#explore-doc-detail").hidden = true;
    $("#explore-chunks-card").hidden = true;
    if (!exploreCodeInitialized) {
      exploreCodeInitialized = true;
      loadCodeSnippet("chunking"); // code never changes at runtime — fetch once, cache
    }
  }
}

$$(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => switchToTab(btn.dataset.tab));
});

$$("[data-tab-link]").forEach((link) => {
  link.addEventListener("click", (e) => {
    e.preventDefault();
    switchToTab(link.dataset.tabLink);
  });
});

// --------------------------------------------------------------- status ---

async function checkStatus() {
  const banner = $("#status-banner");
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    updateStatusBanner(data);
    updateAskAvailability(data);
  } catch (err) {
    banner.textContent = "Could not reach the server.";
    banner.className = "status-banner status-warn";
  }
}

function updateStatusBanner(data) {
  const banner = $("#status-banner");
  const usable = data.has_api_key || !!getByokKey();
  if (!data.index_exists) {
    banner.textContent = "Index not built yet";
    banner.className = "status-banner status-warn";
  } else if (!usable) {
    banner.textContent = "Index ready — paste an OpenAI key above to ask live questions";
    banner.className = "status-banner status-warn";
  } else {
    banner.textContent = `Index ready: ${data.n_vectors} vectors · ${data.dim} dimensions`;
    banner.className = "status-banner status-ok";
  }
}

function updateByokBanner(data) {
  const keylessDeployment = !data.has_api_key;
  $("#byok-banner").hidden = !keylessDeployment;
  const hasKey = !!getByokKey();
  $("#byok-clear").hidden = !hasKey;
  $("#byok-status").textContent = hasKey ? "Key set for this tab." : "";
  $("#byok-key-input").placeholder = hasKey ? "sk-… (already set)" : "sk-…";
}

function updateAskAvailability(data) {
  updateByokBanner(data);
  const keyless = !data.has_api_key;
  const hasUsableKey = data.has_api_key || !!getByokKey();
  const missingKeyHint = keyless
    ? "Paste your OpenAI API key above first."
    : "Set OPENAI_API_KEY in .env first.";

  const btn = $("#btn-ask");
  const hint = $("#ask-disabled-hint");
  const disabled = !data.index_exists || !hasUsableKey;
  btn.disabled = disabled;
  hint.hidden = !disabled;
  hint.textContent = !hasUsableKey ? missingKeyHint : "Build the index first, in the \"Build the Index\" tab.";

  const evalBtn = $("#btn-eval");
  const evalHint = $("#eval-disabled-hint");
  evalBtn.disabled = disabled;
  evalHint.hidden = !disabled;
  evalHint.textContent = hint.textContent;

  const ingestBtn = $("#btn-ingest");
  const ingestHint = $("#ingest-disabled-hint");
  if (ingestBtn) {
    ingestBtn.disabled = !hasUsableKey;
    if (ingestHint) {
      ingestHint.hidden = hasUsableKey;
      ingestHint.textContent = missingKeyHint;
    }
  }
}

$("#byok-save").addEventListener("click", () => {
  const input = $("#byok-key-input");
  const key = input.value.trim();
  if (!key) return;
  setByokKey(key);
  input.value = "";
  checkStatus();
});
$("#byok-key-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter") $("#byok-save").click();
});
$("#byok-clear").addEventListener("click", () => {
  setByokKey("");
  checkStatus();
});

// --------------------------------------------------------- pipeline UI ----

function resetPipeline(containerId) {
  $$(`#${containerId} .pbox`).forEach((box) => {
    box.classList.remove("active", "done", "error");
  });
}

function setBoxState(containerId, boxName, state) {
  const box = $(`#${containerId} .pbox[data-box="${boxName}"]`);
  if (!box) return;
  box.classList.remove("active", "done", "error");
  if (state) box.classList.add(state);
}

function setBoxDetail(boxName, text) {
  const el = $(`.pbox-detail[data-detail="${boxName}"]`);
  if (el) el.textContent = text;
}

function setAskBoxDetail(boxName, text) {
  const el = $(`.pbox-detail[data-detail-ask="${boxName}"]`);
  if (el) el.textContent = text;
}

function flashCard(card) {
  card.classList.remove("flash");
  // Force a reflow so re-adding the class restarts the animation even if
  // the same card was just flashed a moment ago.
  void card.offsetWidth;
  card.classList.add("flash");
}

function fmtTs(ts) {
  const d = new Date(ts * 1000);
  return d.toLocaleTimeString("en-GB", { hour12: false }) + "." + String(d.getMilliseconds()).padStart(3, "0");
}

function appendLog(logId, name, payload, isError = false) {
  const log = $(`#${logId}`);
  const entry = document.createElement("div");
  entry.className = "log-entry" + (isError ? " error" : "");
  const summaryLine = document.createElement("span");
  summaryLine.innerHTML = `<span class="log-ts">${fmtTs(payload.ts || Date.now() / 1000)}</span><span class="log-name">${name}</span>`;
  entry.appendChild(summaryLine);
  const details = document.createElement("details");
  const s = document.createElement("summary");
  s.textContent = "payload";
  const pre = document.createElement("pre");
  pre.textContent = JSON.stringify(payload, null, 2);
  details.appendChild(s);
  details.appendChild(pre);
  entry.appendChild(details);
  log.appendChild(entry);
  log.scrollTop = log.scrollHeight;
}

function renderEmbeddingStripInto(container, preview) {
  container.innerHTML = "";
  const max = Math.max(...preview.map((v) => Math.abs(v)), 1e-6);
  preview.forEach((v) => {
    const cell = document.createElement("div");
    cell.className = "embedding-cell";
    const intensity = Math.abs(v) / max; // 0..1
    const hue = v >= 0 ? 210 : 340; // blue for positive, pink for negative
    cell.style.background = `hsl(${hue}, 70%, ${18 + intensity * 40}%)`;
    cell.title = v.toFixed(4);
    container.appendChild(cell);
  });
}

// ------------------------------------------------- click-a-box-to-inspect -
//
// Every pbox in both pipelines is clickable. Boxes that already have a rich
// visualization further down the page (embedding strip, score bars, prompt,
// answer, doc/batch lists) scroll to and flash that existing card, instead
// of duplicating the same data in a second place. Boxes with no dedicated
// card of their own (Index saved, Question, LLM) show their raw output in
// a small shared panel below the diagram.

const askBoxPayloads = new Map(); // boxName -> [{event, payload}, ...]

function recordBoxPayload(map, boxName, eventName, payload) {
  if (!map.has(boxName)) map.set(boxName, []);
  map.get(boxName).push({ event: eventName, payload });
}

function hideBoxOutput(prefix) {
  const panel = $(`#${prefix}-box-output`);
  panel.hidden = true;
  $$(`#pipeline-${prefix} .pbox.selected`).forEach((b) => b.classList.remove("selected"));
}

function selectBox(prefix, boxName) {
  $$(`#pipeline-${prefix} .pbox`).forEach((b) => b.classList.remove("selected"));
  const box = $(`#pipeline-${prefix} .pbox[data-box="${boxName}"]`);
  if (box) box.classList.add("selected");
}

function showRawBoxOutput(prefix, boxName, title, entries) {
  selectBox(prefix, boxName);
  const panel = $(`#${prefix}-box-output`);
  const content = $(".box-output-content", panel);
  $(".box-output-title", panel).textContent = title;
  content.innerHTML = "";
  if (!entries || entries.length === 0) {
    const msg = document.createElement("p");
    msg.className = "muted";
    msg.textContent = prefix === "init"
      ? "Run the ingestion first — click \"Build index\" above."
      : "Ask a question first — click \"Ask\" above.";
    content.appendChild(msg);
  } else {
    entries.forEach(({ event, payload }) => {
      const name = document.createElement("div");
      name.className = "snippet-name";
      name.textContent = event;
      const pre = document.createElement("pre");
      pre.className = "code-block";
      pre.textContent = JSON.stringify(payload, null, 2);
      content.appendChild(name);
      content.appendChild(pre);
    });
  }
  panel.hidden = false;
  panel.scrollIntoView({ behavior: "smooth", block: "center" });
}

function scrollToBoxCard(prefix, boxName, cardEl) {
  selectBox(prefix, boxName);
  $(`#${prefix}-box-output`).hidden = true;
  if (cardEl.tagName === "DETAILS" || $("details", cardEl)) {
    const details = cardEl.tagName === "DETAILS" ? cardEl : $("details", cardEl);
    if (details) details.open = true;
  }
  cardEl.scrollIntoView({ behavior: "smooth", block: "center" });
  flashCard(cardEl);
}

async function renderDbSummary(container) {
  container.innerHTML = "";
  const loading = document.createElement("p");
  loading.className = "muted";
  loading.textContent = "Loading…";
  container.appendChild(loading);

  let data;
  try {
    const res = await fetch("/api/kb/index");
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      container.innerHTML = "";
      const msg = document.createElement("p");
      msg.className = "muted";
      msg.textContent = err.detail || "Index not built yet — build it in the \"Build the Index\" tab first.";
      container.appendChild(msg);
      return;
    }
    data = await res.json();
  } catch (err) {
    container.innerHTML = "";
    const msg = document.createElement("p");
    msg.className = "muted";
    msg.textContent = "Could not load the index summary.";
    container.appendChild(msg);
    return;
  }

  container.innerHTML = "";

  const summary = document.createElement("p");
  summary.className = "db-summary-line";
  const nVec = document.createElement("strong");
  nVec.textContent = data.n_vectors;
  const dim = document.createElement("strong");
  dim.textContent = data.dim;
  const dtype = document.createElement("strong");
  dtype.textContent = data.dtype;
  const path1 = document.createElement("span");
  path1.className = "db-path";
  path1.textContent = data.vectors_path;
  const path2 = document.createElement("span");
  path2.className = "db-path";
  path2.textContent = data.meta_path;
  summary.append(
    nVec, " vectors × ", dim, " dimensions (dtype ", dtype, "), stored as two plain files: ",
    path1, " (the vectors, a NumPy array) and ", path2,
    " (a JSON list with one entry per chunk: source, chunk_index, and its text — n_words below is computed, not a stored field)."
  );
  container.appendChild(summary);

  const wrap = document.createElement("div");
  wrap.className = "db-table-wrap";
  const table = document.createElement("table");
  table.className = "db-table";
  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");
  ["#", "source", "chunk_index", "n_words"].forEach((h) => {
    const th = document.createElement("th");
    th.textContent = h;
    headRow.appendChild(th);
  });
  thead.appendChild(headRow);
  table.appendChild(thead);
  const tbody = document.createElement("tbody");
  data.chunks.forEach((c, i) => {
    const tr = document.createElement("tr");
    const tdRow = document.createElement("td");
    tdRow.textContent = i;
    const tdSource = document.createElement("td");
    tdSource.className = "db-source";
    tdSource.textContent = c.source;
    const tdIdx = document.createElement("td");
    tdIdx.textContent = c.chunk_index;
    const tdWords = document.createElement("td");
    tdWords.textContent = c.n_words;
    tr.append(tdRow, tdSource, tdIdx, tdWords);
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  wrap.appendChild(table);
  container.appendChild(wrap);
}

function handleInitBoxClick(boxName) {
  if (boxName === "documents" || boxName === "chunking") {
    scrollToBoxCard("init", boxName, $("#doc-list").closest(".card"));
  } else if (boxName === "embeddings") {
    scrollToBoxCard("init", boxName, $("#batch-list").closest(".card"));
  } else if (boxName === "index") {
    selectBox("init", boxName);
    const panel = $("#init-box-output");
    $(".box-output-title", panel).textContent = "Index saved — what's in the database";
    panel.hidden = false;
    renderDbSummary($(".box-output-content", panel)); // asks the server directly, so it's accurate even before this page ever ran an ingestion
    panel.scrollIntoView({ behavior: "smooth", block: "center" });
  }
}

function handleAskBoxClick(boxName) {
  const cardTargets = {
    embedding: () => $("#embedding-strip").closest(".card"),
    search: () => $("#score-bars").closest(".card"),
    topk: () => $("#score-bars").closest(".card"),
    prompt: () => $("#prompt-details"),
    answer: () => $("#answer-text").closest(".card"),
  };
  if (cardTargets[boxName]) {
    scrollToBoxCard("ask", boxName, cardTargets[boxName]());
    return;
  }
  // "question" and "llm" have no dedicated card further down — show their
  // raw event payload in the shared panel instead.
  const title = boxName === "question" ? "Question received" : "LLM call";
  showRawBoxOutput("ask", boxName, title, askBoxPayloads.get(boxName));
}

$$("#pipeline-init .pbox").forEach((box) => {
  box.addEventListener("click", () => handleInitBoxClick(box.dataset.box));
  box.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); handleInitBoxClick(box.dataset.box); }
  });
});
$$("#pipeline-ask .pbox").forEach((box) => {
  box.addEventListener("click", () => handleAskBoxClick(box.dataset.box));
  box.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); handleAskBoxClick(box.dataset.box); }
  });
});

// ============================================================ INGEST =====

const docRows = new Map();
const batchRows = new Map();

function resetIngestUI() {
  resetPipeline("pipeline-init");
  ["documents", "chunking", "embeddings", "index"].forEach((b) => setBoxDetail(b, "—"));
  $("#doc-list").innerHTML = "";
  $("#batch-list").innerHTML = "";
  $("#log-init").innerHTML = "";
  docRows.clear();
  batchRows.clear();
  $("#ingest-progress").textContent = "";
  hideBoxOutput("init");
}

function startIngest() {
  const btn = $("#btn-ingest");
  btn.disabled = true;
  resetIngestUI();

  const es = new FetchEventSource("/api/ingest/stream", { headers: authHeaders() });
  const close = () => {
    es.close();
    btn.disabled = false;
    checkStatus();
  };

  es.addEventListener("ingest_start", (e) => {
    const payload = JSON.parse(e.data);
    appendLog("log-init", "ingest_start", payload);
    setBoxState("pipeline-init", "documents", "active");
    $("#ingest-progress").textContent = "Reading documents…";
  });

  es.addEventListener("docs_loaded", (e) => {
    const payload = JSON.parse(e.data);
    appendLog("log-init", "docs_loaded", payload);
    setBoxState("pipeline-init", "documents", "done");
    setBoxDetail("documents", `${payload.count} docs`);
    setBoxState("pipeline-init", "chunking", "active");
    const list = $("#doc-list");
    list.innerHTML = "";
    payload.documents.forEach((doc) => {
      const row = document.createElement("div");
      row.className = "doc-row";
      row.innerHTML = `<span>${doc.source}</span><span class="badge">${doc.n_words} words · <em>chunking…</em></span>`;
      list.appendChild(row);
      docRows.set(doc.source, row);
    });
    $("#ingest-progress").textContent = "Splitting documents into chunks…";
  });

  es.addEventListener("doc_chunked", (e) => {
    const payload = JSON.parse(e.data);
    appendLog("log-init", "doc_chunked", payload);
    const row = docRows.get(payload.source);
    if (row) {
      const badge = row.querySelector(".badge");
      badge.innerHTML = badge.innerHTML.replace(/·.*$/, `· <strong>${payload.n_chunks} chunks</strong>`);
    }
  });

  es.addEventListener("chunking_done", (e) => {
    const payload = JSON.parse(e.data);
    appendLog("log-init", "chunking_done", payload);
    setBoxState("pipeline-init", "chunking", "done");
    setBoxDetail("chunking", `${payload.total_chunks} chunks`);
    setBoxState("pipeline-init", "embeddings", "active");
    $("#ingest-progress").textContent = "Generating embeddings with OpenAI…";
  });

  es.addEventListener("embedding_batch_start", (e) => {
    const payload = JSON.parse(e.data);
    appendLog("log-init", "embedding_batch_start", payload);
    const row = document.createElement("div");
    row.className = "batch-row";
    row.innerHTML = `<span>Batch ${payload.batch_index + 1}/${payload.total_batches}</span><span class="badge"><em>processing ${payload.batch_size} chunks…</em></span>`;
    $("#batch-list").appendChild(row);
    batchRows.set(payload.batch_index, row);
    setBoxDetail("embeddings", `batch ${payload.batch_index + 1}/${payload.total_batches}`);
  });

  es.addEventListener("embedding_batch_done", (e) => {
    const payload = JSON.parse(e.data);
    appendLog("log-init", "embedding_batch_done", payload);
    const row = batchRows.get(payload.batch_index);
    if (row) {
      row.querySelector(".badge").innerHTML = `${payload.n_vectors} vectors · ${payload.elapsed_ms}ms`;
    }
  });

  let indexPersisted = true;

  es.addEventListener("index_saved", (e) => {
    const payload = JSON.parse(e.data);
    appendLog("log-init", "index_saved", payload);
    indexPersisted = payload.persisted;
    setBoxState("pipeline-init", "embeddings", "done");
    setBoxState("pipeline-init", "index", "active");
    setBoxDetail("index", `${payload.n_vectors} × ${payload.dim}${payload.persisted ? "" : " (in memory only)"}`);
    $("#ingest-progress").textContent = payload.persisted
      ? "Saving index to disk…"
      : "Disk is read-only here — activating the new index in memory instead…";
  });

  es.addEventListener("ingest_done", (e) => {
    const payload = JSON.parse(e.data);
    appendLog("log-init", "ingest_done", payload);
    setBoxState("pipeline-init", "index", "done");
    const persistNote = indexPersisted
      ? ""
      : " Not written to disk (read-only on this deployment) — active in memory for this session; your next question may or may not land on this same instance.";
    $("#ingest-progress").textContent =
      `Done in ${payload.duration_ms}ms — ${payload.n_docs} docs, ${payload.n_chunks} chunks.${persistNote}`;
  });

  es.addEventListener("ingest_error", (e) => {
    const payload = JSON.parse(e.data);
    appendLog("log-init", "ingest_error", payload, true);
  });

  es.addEventListener("pipeline_done", (e) => {
    appendLog("log-init", "pipeline_done", JSON.parse(e.data));
    close();
  });

  es.addEventListener("pipeline_error", (e) => {
    const payload = JSON.parse(e.data);
    appendLog("log-init", "pipeline_error", payload, true);
    $("#ingest-progress").textContent = `Error: ${payload.message}`;
    $$("#pipeline-init .pbox.active").forEach((b) => b.classList.replace("active", "error"));
    close();
  });

  es.onerror = () => {
    // Connection dropped unexpectedly (not via a terminal event) — close it
    // anyway so the button doesn't stay disabled forever.
    close();
  };
}

$("#btn-ingest").addEventListener("click", startIngest);

// ============================================================== ASK =======

function resetAskUI() {
  resetPipeline("pipeline-ask");
  $("#embedding-strip").innerHTML = "";
  $("#score-bars").innerHTML = "";
  $("#vector-map").innerHTML = "";
  $("#vector-map-ref-label").textContent = "";
  $("#vector-map-reset-ref").hidden = true;
  lastQueryVectorMapData = null;
  $("#prompt-text").textContent = "";
  $("#answer-text").textContent = "";
  $("#sources-chips").innerHTML = "";
  $("#log-ask").innerHTML = "";
  ["question", "embedding", "search", "topk", "prompt", "llm", "answer"].forEach((b) => setAskBoxDetail(b, "—"));
  askBoxPayloads.clear();
  hideBoxOutput("ask");
}

function renderScoreBars(allScores, topKSources) {
  const container = $("#score-bars");
  container.innerHTML = "";
  const maxScore = Math.max(...allScores.map((s) => s.score), 1e-6);
  allScores.forEach((s) => {
    const row = document.createElement("div");
    row.className = "score-row";
    const pct = Math.max(0, (s.score / maxScore) * 100);

    const label = document.createElement("span");
    label.className = "score-label";
    label.title = s.text_preview; // set via property, not innerHTML: real KB text can contain quotes
    label.textContent = `${s.source} #${s.chunk_index}`;

    const track = document.createElement("span");
    track.className = "score-track";
    const fill = document.createElement("span");
    fill.className = "score-fill";
    fill.style.width = `${pct}%`;
    track.appendChild(fill);

    const value = document.createElement("span");
    value.className = "score-value";
    value.textContent = s.score.toFixed(3);

    row.appendChild(label);
    row.appendChild(track);
    row.appendChild(value);
    container.appendChild(row);
  });
  // Highlight exactly the real top-K: the first N rows whose source is in
  // topKSources, in the order they arrived (already sorted by score by
  // the backend).
  const rows = $$(".score-row", container);
  let remaining = new Map();
  topKSources.forEach((src) => remaining.set(src, (remaining.get(src) || 0) + 1));
  rows.forEach((row, i) => {
    const src = allScores[i].source;
    if (remaining.get(src) > 0) {
      row.classList.add("topk");
      remaining.set(src, remaining.get(src) - 1);
    }
  });
}

// The last real search's map data, cached so "← Back to your question" can
// restore it locally without re-running the question.
let lastQueryVectorMapData = null;

async function rerootVectorMap(source, chunkIndex) {
  const label = `${source} #${chunkIndex}`;
  const refLabelEl = $("#vector-map-ref-label");
  refLabelEl.textContent = `Loading neighbors of ${label}…`;
  try {
    const res = await fetch(`/api/kb/similarity?source=${encodeURIComponent(source)}&chunk_index=${chunkIndex}`);
    if (!res.ok) {
      refLabelEl.textContent = "Could not load that chunk's neighbors.";
      return;
    }
    const data = await res.json();
    const self = data.all_scores.find((s) => s.source === source && s.chunk_index === chunkIndex);
    // No topKSources here on purpose: "top-K used as context" isn't a
    // meaningful concept for an arbitrary chunk-to-chunk similarity view,
    // so renderVectorMap skips the lines/highlighting entirely.
    renderVectorMap($("#vector-map"), data.all_scores, [], { x: self.x, y: self.y }, label);
    refLabelEl.textContent = `Showing neighbors of: ${label}`;
    $("#vector-map-reset-ref").hidden = false;
  } catch (err) {
    refLabelEl.textContent = "Could not load that chunk's neighbors.";
  }
}

$("#vector-map-reset-ref").addEventListener("click", () => {
  if (!lastQueryVectorMapData) return;
  const { allScores, topKSources, referencePoint } = lastQueryVectorMapData;
  renderVectorMap($("#vector-map"), allScores, topKSources, referencePoint, "Your question");
  $("#vector-map-ref-label").textContent = "";
  $("#vector-map-reset-ref").hidden = true;
});

// Pan/zoom window into the map's fixed 560×340 coordinate space. Persists
// across re-renders that keep the same question (reroot / back-to-question)
// so panning/zooming isn't lost when the reference point changes; only a
// fresh question or the explicit "Reset view" button snaps it back.
const VECTOR_MAP_W = 560, VECTOR_MAP_H = 340;
let vectorMapView = { x: 0, y: 0, w: VECTOR_MAP_W, h: VECTOR_MAP_H };

function resetVectorMapView() {
  vectorMapView = { x: 0, y: 0, w: VECTOR_MAP_W, h: VECTOR_MAP_H };
  const svg = document.querySelector("#vector-map .vector-map-svg");
  if (svg) svg.setAttribute("viewBox", `${vectorMapView.x} ${vectorMapView.y} ${vectorMapView.w} ${vectorMapView.h}`);
}

$("#vector-map-reset-view").addEventListener("click", resetVectorMapView);

// Drag-to-pan and wheel/pinch-to-zoom over the map's viewBox. Dot/query
// clicks are excluded from starting a drag (checked by the caller via
// e.target's class) so re-centering still works with a single click.
function attachVectorMapPanZoom(svg) {
  let dragging = false;
  let lastClientX = 0, lastClientY = 0;

  svg.addEventListener("pointerdown", (e) => {
    if (e.target.closest(".vector-dot, .vector-query")) return;
    dragging = true;
    lastClientX = e.clientX;
    lastClientY = e.clientY;
    svg.classList.add("grabbing");
    svg.setPointerCapture(e.pointerId);
  });

  svg.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    const rect = svg.getBoundingClientRect();
    vectorMapView.x -= (e.clientX - lastClientX) * (vectorMapView.w / rect.width);
    vectorMapView.y -= (e.clientY - lastClientY) * (vectorMapView.h / rect.height);
    lastClientX = e.clientX;
    lastClientY = e.clientY;
    svg.setAttribute("viewBox", `${vectorMapView.x} ${vectorMapView.y} ${vectorMapView.w} ${vectorMapView.h}`);
  });

  const endDrag = (e) => {
    dragging = false;
    svg.classList.remove("grabbing");
    if (e && svg.hasPointerCapture(e.pointerId)) svg.releasePointerCapture(e.pointerId);
  };
  svg.addEventListener("pointerup", endDrag);
  svg.addEventListener("pointercancel", endDrag);

  svg.addEventListener("wheel", (e) => {
    e.preventDefault();
    const rect = svg.getBoundingClientRect();
    const nx = (e.clientX - rect.left) / rect.width;
    const ny = (e.clientY - rect.top) / rect.height;
    const worldX = vectorMapView.x + nx * vectorMapView.w;
    const worldY = vectorMapView.y + ny * vectorMapView.h;

    const factor = e.deltaY > 0 ? 1.1 : 0.9;
    const minW = VECTOR_MAP_W * 0.1, maxW = VECTOR_MAP_W * 1.5;
    const newW = Math.min(maxW, Math.max(minW, vectorMapView.w * factor));
    const newH = newW * (VECTOR_MAP_H / VECTOR_MAP_W);

    vectorMapView.x = worldX - nx * newW;
    vectorMapView.y = worldY - ny * newH;
    vectorMapView.w = newW;
    vectorMapView.h = newH;
    svg.setAttribute("viewBox", `${vectorMapView.x} ${vectorMapView.y} ${vectorMapView.w} ${vectorMapView.h}`);
  }, { passive: false });
}

// 2D scatter of every chunk's PCA-projected vector plus a reference point
// (your question, or — after clicking a dot — an indexed chunk's own
// vector), in the same projected space computed server-side in
// retriever.py, so the frontend never touches raw 1536-dim numbers. Reuses
// the same "which rows are really top-K" trick as renderScoreBars: allScores
// is sorted by score, so the first N whose source appears in topKSources are
// the real top-K. Pass an empty topKSources to skip the top-K
// lines/highlighting entirely — that's what chunk-to-chunk reroot mode does,
// since "top-K actually used as context" isn't a meaningful concept there.
function renderVectorMap(container, allScores, topKSources, referencePoint, referenceLabel) {
  container.innerHTML = "";
  if (!allScores.length) {
    const msg = document.createElement("p");
    msg.className = "muted";
    msg.textContent = "No chunks to show yet.";
    container.appendChild(msg);
    return;
  }

  const xs = allScores.map((s) => s.x).concat([referencePoint.x]);
  const ys = allScores.map((s) => s.y).concat([referencePoint.y]);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const padX = (maxX - minX || 1) * 0.15;
  const padY = (maxY - minY || 1) * 0.15;
  const rangeX = (maxX - minX) + padX * 2 || 1;
  const rangeY = (maxY - minY) + padY * 2 || 1;

  const W = VECTOR_MAP_W, H = VECTOR_MAP_H;
  const toPx = (x, y) => [
    ((x - minX + padX) / rangeX) * W,
    H - ((y - minY + padY) / rangeY) * H, // flip Y so it doesn't read upside down
  ];

  const topKSet = new Set();
  const remaining = new Map();
  topKSources.forEach((src) => remaining.set(src, (remaining.get(src) || 0) + 1));
  allScores.forEach((s) => {
    const left = remaining.get(s.source) || 0;
    if (left > 0) {
      topKSet.add(`${s.source}#${s.chunk_index}`);
      remaining.set(s.source, left - 1);
    }
  });

  const svgNS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(svgNS, "svg");
  svg.setAttribute("viewBox", `${vectorMapView.x} ${vectorMapView.y} ${vectorMapView.w} ${vectorMapView.h}`);
  svg.setAttribute("class", "vector-map-svg");

  const [qx, qy] = toPx(referencePoint.x, referencePoint.y);

  // Lines first, so the dots render on top of them.
  allScores.forEach((s) => {
    if (!topKSet.has(`${s.source}#${s.chunk_index}`)) return;
    const [px, py] = toPx(s.x, s.y);
    const line = document.createElementNS(svgNS, "line");
    line.setAttribute("x1", qx);
    line.setAttribute("y1", qy);
    line.setAttribute("x2", px);
    line.setAttribute("y2", py);
    line.setAttribute("class", "vector-link");
    svg.appendChild(line);
  });

  allScores.forEach((s) => {
    const isTopK = topKSet.has(`${s.source}#${s.chunk_index}`);
    const [px, py] = toPx(s.x, s.y);
    const circle = document.createElementNS(svgNS, "circle");
    circle.setAttribute("cx", px);
    circle.setAttribute("cy", py);
    circle.setAttribute("r", isTopK ? 6 : 3.5);
    circle.setAttribute("class", "vector-dot" + (isTopK ? " topk" : ""));
    const title = document.createElementNS(svgNS, "title");
    title.textContent = `${s.source} #${s.chunk_index} — score ${s.score.toFixed(3)} (click to re-center here)`;
    circle.appendChild(title);
    circle.addEventListener("click", () => rerootVectorMap(s.source, s.chunk_index));
    svg.appendChild(circle);
  });

  const query = document.createElementNS(svgNS, "circle");
  query.setAttribute("cx", qx);
  query.setAttribute("cy", qy);
  query.setAttribute("r", 7);
  query.setAttribute("class", "vector-query");
  const qTitle = document.createElementNS(svgNS, "title");
  qTitle.textContent = referenceLabel;
  query.appendChild(qTitle);
  svg.appendChild(query);

  container.appendChild(svg);
  attachVectorMapPanZoom(svg);

  const legend = document.createElement("div");
  legend.className = "vector-map-legend";
  legend.innerHTML = `
    <span><span class="legend-swatch legend-query"></span>reference point</span>
    <span><span class="legend-swatch legend-topk"></span>top-K (used as context)</span>
    <span><span class="legend-swatch legend-dot"></span>other chunks</span>
  `;
  container.appendChild(legend);
}

function askQuestion() {
  const input = $("#question-input");
  const question = input.value.trim();
  if (!question) return;

  const btn = $("#btn-ask");
  btn.disabled = true;
  resetAskUI();
  setBoxState("pipeline-ask", "question", "active");
  currentTiming = null;

  const url = `/api/ask/stream?question=${encodeURIComponent(question)}`;
  const es = new FetchEventSource(url, { headers: authHeaders() });
  const close = () => {
    es.close();
    btn.disabled = false;
    checkStatus();
  };

  es.addEventListener("question_received", (e) => {
    const payload = JSON.parse(e.data);
    appendLog("log-ask", "question_received", payload);
    recordBoxPayload(askBoxPayloads, "question", "question_received", payload);
    setBoxState("pipeline-ask", "question", "done");
    setAskBoxDetail("question", payload.question.length > 28 ? payload.question.slice(0, 28) + "…" : payload.question);
    currentTiming = { t_question: payload.ts };
  });

  es.addEventListener("embedding_query_start", (e) => {
    appendLog("log-ask", "embedding_query_start", JSON.parse(e.data));
    setBoxState("pipeline-ask", "embedding", "active");
  });

  es.addEventListener("embedding_query_done", (e) => {
    const payload = JSON.parse(e.data);
    appendLog("log-ask", "embedding_query_done", payload);
    setBoxState("pipeline-ask", "embedding", "done");
    setAskBoxDetail("embedding", `${payload.dim} dims`);
    $("#embedding-dim").textContent = payload.dim;
    renderEmbeddingStripInto($("#embedding-strip"), payload.preview);
    if (currentTiming) currentTiming.embedding_ms = payload.elapsed_ms;
  });

  es.addEventListener("search_start", (e) => {
    const payload = JSON.parse(e.data);
    appendLog("log-ask", "search_start", payload);
    setBoxState("pipeline-ask", "search", "active");
    if (currentTiming) currentTiming.t_search_start = payload.ts;
  });

  es.addEventListener("search_done", (e) => {
    const payload = JSON.parse(e.data);
    appendLog("log-ask", "search_done", payload);
    setBoxState("pipeline-ask", "search", "done");
    if (currentTiming && currentTiming.t_search_start) {
      currentTiming.search_ms = Math.max(0, Math.round((payload.ts - currentTiming.t_search_start) * 1000));
    }
    setAskBoxDetail("search", `${payload.all_scores.length} chunks scored`);
    setBoxState("pipeline-ask", "topk", "active");
    renderScoreBars(payload.all_scores, payload.top_k_sources);
    lastQueryVectorMapData = {
      allScores: payload.all_scores,
      topKSources: payload.top_k_sources,
      referencePoint: payload.query_projection,
    };
    vectorMapView = { x: 0, y: 0, w: VECTOR_MAP_W, h: VECTOR_MAP_H }; // fresh question: forget any old pan/zoom
    renderVectorMap($("#vector-map"), payload.all_scores, payload.top_k_sources, payload.query_projection, "Your question");
    $("#vector-map-ref-label").textContent = "";
    $("#vector-map-reset-ref").hidden = true;
    setBoxState("pipeline-ask", "topk", "done");
    setAskBoxDetail("topk", `${payload.top_k_sources.length} selected`);
  });

  es.addEventListener("no_context", (e) => {
    appendLog("log-ask", "no_context", JSON.parse(e.data));
    $("#answer-text").textContent = "No relevant information found in the knowledge base.";
  });

  es.addEventListener("prompt_built", (e) => {
    const payload = JSON.parse(e.data);
    appendLog("log-ask", "prompt_built", payload);
    setBoxState("pipeline-ask", "prompt", "done");
    setAskBoxDetail("prompt", `${payload.prompt.length} chars`);
    $("#prompt-text").textContent = `[SYSTEM]\n${payload.system_prompt}\n\n[USER]\n${payload.prompt}`;
  });

  es.addEventListener("llm_start", (e) => {
    const payload = JSON.parse(e.data);
    appendLog("log-ask", "llm_start", payload);
    recordBoxPayload(askBoxPayloads, "llm", "llm_start", payload);
    setBoxState("pipeline-ask", "llm", "active");
    setBoxState("pipeline-ask", "answer", "active");
    setAskBoxDetail("llm", payload.model);
    setAskBoxDetail("answer", "streaming…");
    $("#answer-text").textContent = "";
  });

  es.addEventListener("llm_token", (e) => {
    const payload = JSON.parse(e.data);
    $("#answer-text").textContent += payload.delta;
  });

  es.addEventListener("llm_done", (e) => {
    const payload = JSON.parse(e.data);
    appendLog("log-ask", "llm_done", payload);
    recordBoxPayload(askBoxPayloads, "llm", "llm_done", payload);
    setBoxState("pipeline-ask", "llm", "done");
    if (currentTiming && currentTiming.embedding_ms != null && currentTiming.search_ms != null) {
      currentTiming.llm_ms = payload.elapsed_ms;
      currentTiming.total_ms = Math.max(0, Math.round((payload.ts - currentTiming.t_question) * 1000));
      renderLatencyBreakdown(currentTiming);
      sessionLatencies.push(currentTiming.total_ms);
      renderLatencyDistribution();
    }
  });

  es.addEventListener("answer_done", (e) => {
    const payload = JSON.parse(e.data);
    appendLog("log-ask", "answer_done", payload);
    setBoxState("pipeline-ask", "answer", "done");
    setAskBoxDetail("answer", `${payload.answer.split(/\s+/).length} words`);
    const chips = $("#sources-chips");
    chips.innerHTML = "";
    payload.sources.forEach((src) => {
      const chip = document.createElement("span");
      chip.className = "chip";
      chip.textContent = src;
      chips.appendChild(chip);
    });
  });

  es.addEventListener("answer_error", (e) => {
    appendLog("log-ask", "answer_error", JSON.parse(e.data), true);
  });

  es.addEventListener("pipeline_done", (e) => {
    appendLog("log-ask", "pipeline_done", JSON.parse(e.data));
    close();
  });

  es.addEventListener("pipeline_error", (e) => {
    const payload = JSON.parse(e.data);
    appendLog("log-ask", "pipeline_error", payload, true);
    $("#answer-text").textContent = `Error: ${payload.message}`;
    $$("#pipeline-ask .pbox.active").forEach((b) => b.classList.replace("active", "error"));
    close();
  });

  es.onerror = () => close();
}

$("#btn-ask").addEventListener("click", askQuestion);
$("#question-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter") askQuestion();
});

// ============================================================ EXPLORE =====
//
// Lets you browse the raw documents, see exactly how each one was cut into
// chunks (with real word offsets and overlap), and — once the index exists
// — inspect the actual stored vector for any chunk. The code panels fetch
// real source code from the server via inspect.getsource(), so what you
// read here can never drift from what actually ran.

const codeSnippetCache = new Map();

async function loadDocPicker() {
  const picker = $("#explore-doc-picker");
  picker.textContent = "Loading…";
  try {
    const res = await fetch("/api/documents");
    const docs = await res.json();
    picker.innerHTML = "";
    docs.forEach((doc) => {
      const btn = document.createElement("button");
      btn.className = "doc-pick-btn";
      btn.textContent = `${doc.source} (${doc.n_words}w)`;
      btn.addEventListener("click", () => selectExploreDoc(doc.source, btn));
      picker.appendChild(btn);
    });
  } catch (err) {
    picker.textContent = "Could not load the document list.";
  }
}

async function selectExploreDoc(source, btnEl) {
  $$(".doc-pick-btn", $("#explore-doc-picker")).forEach((b) => b.classList.remove("active"));
  btnEl.classList.add("active");

  const detailCard = $("#explore-doc-detail");
  const chunksCard = $("#explore-chunks-card");
  detailCard.hidden = false;
  chunksCard.hidden = false;
  $("#explore-doc-title").textContent = source;
  $("#explore-doc-text").textContent = "Loading…";
  $("#explore-chunks-list").innerHTML = '<p class="muted">Loading chunks…</p>';

  try {
    const [docRes, chunksRes] = await Promise.all([
      fetch(`/api/kb/documents/${encodeURIComponent(source)}`),
      fetch(`/api/kb/chunks?source=${encodeURIComponent(source)}`),
    ]);
    const doc = await docRes.json();
    const chunks = await chunksRes.json();
    $("#explore-doc-text").textContent = doc.text;
    renderChunkList(chunks);
  } catch (err) {
    $("#explore-doc-text").textContent = "Could not load this document.";
    $("#explore-chunks-list").innerHTML = "";
  }
}

function renderChunkList(chunks) {
  const list = $("#explore-chunks-list");
  list.innerHTML = "";
  chunks.forEach((chunk) => {
    const card = document.createElement("div");
    card.className = "chunk-card";

    const head = document.createElement("div");
    head.className = "chunk-card-head";

    const idxBadge = document.createElement("span");
    idxBadge.className = "chunk-badge";
    idxBadge.textContent = `chunk #${chunk.chunk_index}`;
    head.appendChild(idxBadge);

    const rangeBadge = document.createElement("span");
    rangeBadge.className = "chunk-badge";
    rangeBadge.textContent = `words ${chunk.start_word}–${chunk.end_word} (${chunk.n_words}w)`;
    head.appendChild(rangeBadge);

    if (chunk.overlap_words > 0) {
      const overlapBadge = document.createElement("span");
      overlapBadge.className = "chunk-badge overlap";
      overlapBadge.textContent = `overlaps ${chunk.overlap_words}w with the previous chunk`;
      head.appendChild(overlapBadge);
    }

    const vectorBadge = document.createElement("span");
    vectorBadge.className = "chunk-badge " + (chunk.has_vector ? "vector-yes" : "vector-no");
    vectorBadge.textContent = chunk.has_vector ? "vector stored ✓" : "not indexed yet";
    head.appendChild(vectorBadge);

    card.appendChild(head);

    const text = document.createElement("div");
    text.className = "chunk-text";
    text.textContent = chunk.text;
    card.appendChild(text);

    if (chunk.has_vector) {
      card.appendChild(buildVectorToggle(chunk));
    }

    list.appendChild(card);
  });
}

function buildVectorToggle(chunk) {
  const wrapper = document.createElement("div");

  const btn = document.createElement("button");
  btn.className = "chunk-vector-btn";
  btn.textContent = "Show the real embedding vector →";

  const panel = document.createElement("div");
  panel.className = "chunk-vector-panel";
  panel.hidden = true;

  btn.addEventListener("click", async () => {
    panel.hidden = !panel.hidden;
    btn.textContent = panel.hidden
      ? "Show the real embedding vector →"
      : "Hide the embedding vector ↑";
    if (!panel.hidden && !panel.dataset.loaded) {
      panel.dataset.loaded = "1";
      panel.textContent = "Loading…";
      try {
        const res = await fetch(
          `/api/kb/vector?source=${encodeURIComponent(chunk.source)}&chunk_index=${chunk.chunk_index}`
        );
        const data = await res.json();
        panel.innerHTML = "";

        const meta = document.createElement("div");
        meta.className = "chunk-vector-meta";
        meta.textContent =
          `${data.dim} dimensions · L2 norm = ${data.norm.toFixed(4)} ` +
          `(normalized to 1, so cosine similarity is a plain dot product)`;
        panel.appendChild(meta);

        const strip = document.createElement("div");
        strip.className = "embedding-strip";
        panel.appendChild(strip);
        renderEmbeddingStripInto(strip, data.vector.slice(0, 32));

        const details = document.createElement("details");
        const summary = document.createElement("summary");
        summary.textContent = `View all ${data.dim} raw numbers`;
        const pre = document.createElement("pre");
        pre.className = "code-block";
        pre.textContent = JSON.stringify(data.vector);
        details.appendChild(summary);
        details.appendChild(pre);
        panel.appendChild(details);
      } catch (err) {
        panel.textContent = "Could not load this vector.";
      }
    }
  });

  wrapper.appendChild(btn);
  wrapper.appendChild(panel);
  return wrapper;
}

async function loadCodeSnippet(key) {
  const viewer = $("#code-viewer");
  if (codeSnippetCache.has(key)) {
    renderCodeSnippets(codeSnippetCache.get(key));
    return;
  }
  viewer.textContent = "Loading…";
  try {
    const res = await fetch(`/api/code?key=${encodeURIComponent(key)}`);
    const data = await res.json();
    codeSnippetCache.set(key, data.snippets);
    renderCodeSnippets(data.snippets);
  } catch (err) {
    viewer.textContent = "Could not load the source code.";
  }
}

function renderCodeSnippets(snippets) {
  const viewer = $("#code-viewer");
  viewer.innerHTML = "";
  snippets.forEach((snip) => {
    const name = document.createElement("div");
    name.className = "snippet-name";
    name.textContent = `${snip.name}()`;
    const pre = document.createElement("pre");
    pre.className = "code-block";
    pre.textContent = snip.source;
    viewer.appendChild(name);
    viewer.appendChild(pre);
  });
}

$$("#code-picker .chip-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    $$("#code-picker .chip-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    loadCodeSnippet(btn.dataset.codeKey);
  });
});

// ============================================================== EVAL ======
//
// Runs eval/evaluate.py's Recall@K and Faithfulness (LLM-as-judge) — the
// same functions the CLI (`python -m eval.evaluate`) calls — streamed live
// via SSE instead of printed to a terminal.

function createResultsTable(container, headers) {
  container.innerHTML = "";
  const wrap = document.createElement("div");
  wrap.className = "db-table-wrap";
  const table = document.createElement("table");
  table.className = "db-table";
  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");
  headers.forEach((h) => {
    const th = document.createElement("th");
    th.textContent = h;
    headRow.appendChild(th);
  });
  thead.appendChild(headRow);
  table.appendChild(thead);
  const tbody = document.createElement("tbody");
  table.appendChild(tbody);
  wrap.appendChild(table);
  container.appendChild(wrap);
  return tbody;
}

function appendResultRow(tbody, cells, okCell) {
  const tr = document.createElement("tr");
  cells.forEach((text, i) => {
    const td = document.createElement("td");
    td.textContent = text;
    if (okCell && i === okCell.index) td.className = okCell.ok ? "eval-ok" : "eval-bad";
    tr.appendChild(td);
  });
  tbody.appendChild(tr);
}

function resetEvalUI() {
  $("#eval-progress").textContent = "";
  $("#metric-recall").textContent = "—";
  $("#metric-faithfulness").textContent = "—";
  $("#metric-hallucination").textContent = "";
  $("#eval-recall-section").hidden = true;
  $("#eval-faithfulness-section").hidden = true;
  $("#eval-recall-table").innerHTML = "";
  $("#eval-faithfulness-table").innerHTML = "";
  $("#log-eval").innerHTML = "";
}

function fmtSnapshotDate(iso) {
  try {
    return new Date(iso).toLocaleString("en-GB", { dateStyle: "medium", timeStyle: "short" });
  } catch {
    return iso;
  }
}

async function loadEvalSnapshot() {
  try {
    const res = await fetch("/api/eval/snapshot");
    if (!res.ok) {
      $("#eval-source-label").textContent = "No snapshot committed yet — click \"Run evaluation live\" below.";
      return;
    }
    const snap = await res.json();
    $("#metric-recall").textContent = `${Math.round(snap.recall * 100)}%`;
    $("#metric-faithfulness").textContent = `${Math.round(snap.faithfulness * 100)}%`;
    $("#metric-hallucination").textContent =
      `Hallucination rate: ${Math.round((1 - snap.faithfulness) * 100)}% (= 1 − faithfulness, derived, not a separate measurement)`;
    $("#eval-source-label").textContent =
      `Snapshot computed ${fmtSnapshotDate(snap.generated_at)} against this exact index (${snap.embedding_model} / ${snap.chat_model}) — committed to the repo, not live.`;

    const recallTbody = createResultsTable($("#eval-recall-table"), ["Question", "Expected", "Retrieved", "Result"]);
    $("#eval-recall-section").hidden = false;
    snap.recall_items.forEach((item) => {
      appendResultRow(
        recallTbody,
        [item.question, item.expected_sources.join(", "), item.retrieved_sources.join(", "), item.hit ? "hit" : "miss"],
        { index: 3, ok: item.hit }
      );
    });

    const faithTbody = createResultsTable($("#eval-faithfulness-table"), ["Question", "Verdict"]);
    $("#eval-faithfulness-section").hidden = false;
    snap.faithfulness_items.forEach((item) => {
      appendResultRow(
        faithTbody,
        [item.question, item.is_faithful ? "faithful" : "not faithful"],
        { index: 1, ok: item.is_faithful }
      );
    });
  } catch (err) {
    $("#eval-source-label").textContent = "Could not load the eval snapshot.";
  }
}

function runEvaluation() {
  const btn = $("#btn-eval");
  btn.disabled = true;
  resetEvalUI();
  $("#eval-source-label").textContent = "Running live against your own key…";
  $("#eval-progress").textContent = "Running Recall@K…";

  const recallTbody = createResultsTable($("#eval-recall-table"), ["Question", "Expected", "Retrieved", "Result"]);
  $("#eval-recall-section").hidden = false;
  let faithTbody = null;

  const es = new FetchEventSource("/api/eval/stream", { headers: authHeaders() });
  const close = () => {
    es.close();
    btn.disabled = false;
  };

  es.addEventListener("recall_item", (e) => {
    const payload = JSON.parse(e.data);
    appendLog("log-eval", "recall_item", payload);
    appendResultRow(
      recallTbody,
      [payload.question, payload.expected_sources.join(", "), payload.retrieved_sources.join(", "), payload.hit ? "hit" : "miss"],
      { index: 3, ok: payload.hit }
    );
  });

  es.addEventListener("recall_done", (e) => {
    const payload = JSON.parse(e.data);
    appendLog("log-eval", "recall_done", payload);
    $("#metric-recall").textContent = `${Math.round(payload.recall * 100)}%`;
    $("#eval-scores").hidden = false;
    $("#eval-progress").textContent = "Running Faithfulness (LLM-as-judge)…";
    faithTbody = createResultsTable($("#eval-faithfulness-table"), ["Question", "Verdict"]);
    $("#eval-faithfulness-section").hidden = false;
  });

  es.addEventListener("faithfulness_item", (e) => {
    const payload = JSON.parse(e.data);
    appendLog("log-eval", "faithfulness_item", payload);
    appendResultRow(
      faithTbody,
      [payload.question, payload.is_faithful ? "faithful" : "not faithful"],
      { index: 1, ok: payload.is_faithful }
    );
  });

  es.addEventListener("faithfulness_done", (e) => {
    const payload = JSON.parse(e.data);
    appendLog("log-eval", "faithfulness_done", payload);
    const faithfulness = payload.score;
    $("#metric-faithfulness").textContent = `${Math.round(faithfulness * 100)}%`;
    // Deliberately not an independent tile: hallucination rate here is just
    // the algebraic complement of faithfulness, not a second measurement.
    $("#metric-hallucination").textContent =
      `Hallucination rate: ${Math.round((1 - faithfulness) * 100)}% (= 1 − faithfulness, derived, not a separate measurement)`;
  });

  es.addEventListener("pipeline_done", (e) => {
    appendLog("log-eval", "pipeline_done", JSON.parse(e.data));
    $("#eval-progress").textContent = "Done.";
    $("#eval-source-label").textContent = "Live run — just now, against this exact index, with your key.";
    close();
  });

  es.addEventListener("pipeline_error", (e) => {
    const payload = JSON.parse(e.data);
    appendLog("log-eval", "pipeline_error", payload, true);
    $("#eval-progress").textContent = `Error: ${payload.message}`;
    $("#eval-source-label").textContent = "Live run failed — showing whatever loaded before the error.";
    close();
  });

  es.onerror = () => close();
}

$("#btn-eval").addEventListener("click", runEvaluation);
loadEvalSnapshot();

// ----------------------------------------------------------- latency ------
//
// Real per-stage timings from the SSE events of the question you actually
// just asked (not simulated), plus an honestly-labeled running distribution
// across every question asked this session — no percentiles pretending to
// be statistically meaningful from a handful of samples.

const sessionLatencies = []; // total ms per question asked this session
let currentTiming = null;

function renderLatencyBreakdown(timing) {
  const container = $("#latency-breakdown");
  container.innerHTML = "";
  const total = Math.max(1, timing.embedding_ms + timing.search_ms + timing.llm_ms);

  const bar = document.createElement("div");
  bar.className = "latency-bar";
  [
    ["embedding", timing.embedding_ms],
    ["search", timing.search_ms],
    ["llm", timing.llm_ms],
  ].forEach(([kind, ms]) => {
    const seg = document.createElement("div");
    seg.className = `latency-segment ${kind}`;
    seg.style.width = `${Math.max(1, (ms / total) * 100)}%`;
    seg.textContent = ms / total >= 0.12 ? `${ms}ms` : "";
    bar.appendChild(seg);
  });
  container.appendChild(bar);

  const legend = document.createElement("p");
  legend.className = "latency-legend";
  const parts = [
    ["Embedding", timing.embedding_ms],
    ["Search", timing.search_ms],
    ["LLM", timing.llm_ms],
    ["Total", timing.total_ms],
  ];
  parts.forEach(([label, ms], i) => {
    if (i > 0) legend.append(" · ");
    legend.append(`${label}: `);
    const strong = document.createElement("strong");
    strong.textContent = `${ms}ms`;
    legend.appendChild(strong);
  });
  container.appendChild(legend);
}

function percentile(sortedArr, p) {
  const idx = Math.min(sortedArr.length - 1, Math.floor(p * sortedArr.length));
  return sortedArr[idx];
}

function renderLatencyDistribution() {
  const container = $("#latency-distribution");
  container.innerHTML = "";
  const n = sessionLatencies.length;
  const line = document.createElement("p");
  line.className = "latency-distribution-line";
  if (n < 2) {
    line.textContent = `Based on ${n} question${n === 1 ? "" : "s"} asked this session — ask a ` +
      `few more to see a real distribution (percentiles of 1-2 samples aren't meaningful).`;
  } else {
    const sorted = [...sessionLatencies].sort((a, b) => a - b);
    let text = `Based on ${n} questions asked this session: min ${sorted[0]}ms · ` +
      `P50 ${percentile(sorted, 0.5)}ms · max ${sorted[sorted.length - 1]}ms`;
    text += n >= 5
      ? ` · P95 ${percentile(sorted, 0.95)}ms`
      : ` (P95/P99 need at least 5 samples — currently ${n})`;
    line.textContent = text;
  }
  container.appendChild(line);
}

// -------------------------------------------------------------- init ------

checkStatus();
