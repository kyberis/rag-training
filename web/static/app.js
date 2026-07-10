"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

// ---------------------------------------------------------------- tabs ----

let exploreInitialized = false;

$$(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    $$(".tab-btn").forEach((b) => b.classList.remove("active"));
    $$(".tab-panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    $(`#tab-${btn.dataset.tab}`).classList.add("active");
    if (btn.dataset.tab === "explore" && !exploreInitialized) {
      exploreInitialized = true;
      loadDocPicker();
      loadCodeSnippet("chunking");
    }
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
  if (!data.has_api_key) {
    banner.textContent = "Missing OPENAI_API_KEY in .env";
    banner.className = "status-banner status-warn";
  } else if (!data.index_exists) {
    banner.textContent = "Index not built yet";
    banner.className = "status-banner status-warn";
  } else {
    banner.textContent = `Index ready: ${data.n_vectors} vectors · ${data.dim} dimensions`;
    banner.className = "status-banner status-ok";
  }
}

function updateAskAvailability(data) {
  const btn = $("#btn-ask");
  const hint = $("#ask-disabled-hint");
  const disabled = !data.index_exists || !data.has_api_key;
  btn.disabled = disabled;
  hint.hidden = !disabled;
  hint.textContent = !data.has_api_key
    ? "Set OPENAI_API_KEY in .env first."
    : "Build the index first, in the \"Build the Index\" tab.";
}

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
}

function startIngest() {
  const btn = $("#btn-ingest");
  btn.disabled = true;
  resetIngestUI();

  const es = new EventSource("/api/ingest/stream");
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

  es.addEventListener("index_saved", (e) => {
    const payload = JSON.parse(e.data);
    appendLog("log-init", "index_saved", payload);
    setBoxState("pipeline-init", "embeddings", "done");
    setBoxState("pipeline-init", "index", "active");
    setBoxDetail("index", `${payload.n_vectors} × ${payload.dim}`);
    $("#ingest-progress").textContent = "Saving index to disk…";
  });

  es.addEventListener("ingest_done", (e) => {
    const payload = JSON.parse(e.data);
    appendLog("log-init", "ingest_done", payload);
    setBoxState("pipeline-init", "index", "done");
    $("#ingest-progress").textContent = `Done in ${payload.duration_ms}ms — ${payload.n_docs} docs, ${payload.n_chunks} chunks.`;
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
  $("#prompt-text").textContent = "";
  $("#answer-text").textContent = "";
  $("#sources-chips").innerHTML = "";
  $("#log-ask").innerHTML = "";
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

function askQuestion() {
  const input = $("#question-input");
  const question = input.value.trim();
  if (!question) return;

  const btn = $("#btn-ask");
  btn.disabled = true;
  resetAskUI();
  setBoxState("pipeline-ask", "question", "active");

  const url = `/api/ask/stream?question=${encodeURIComponent(question)}`;
  const es = new EventSource(url);
  const close = () => {
    es.close();
    btn.disabled = false;
    checkStatus();
  };

  es.addEventListener("question_received", (e) => {
    appendLog("log-ask", "question_received", JSON.parse(e.data));
    setBoxState("pipeline-ask", "question", "done");
  });

  es.addEventListener("embedding_query_start", (e) => {
    appendLog("log-ask", "embedding_query_start", JSON.parse(e.data));
    setBoxState("pipeline-ask", "embedding", "active");
  });

  es.addEventListener("embedding_query_done", (e) => {
    const payload = JSON.parse(e.data);
    appendLog("log-ask", "embedding_query_done", payload);
    setBoxState("pipeline-ask", "embedding", "done");
    $("#embedding-dim").textContent = payload.dim;
    renderEmbeddingStripInto($("#embedding-strip"), payload.preview);
  });

  es.addEventListener("search_start", (e) => {
    appendLog("log-ask", "search_start", JSON.parse(e.data));
    setBoxState("pipeline-ask", "search", "active");
  });

  es.addEventListener("search_done", (e) => {
    const payload = JSON.parse(e.data);
    appendLog("log-ask", "search_done", payload);
    setBoxState("pipeline-ask", "search", "done");
    setBoxState("pipeline-ask", "topk", "active");
    renderScoreBars(payload.all_scores, payload.top_k_sources);
    setBoxState("pipeline-ask", "topk", "done");
  });

  es.addEventListener("no_context", (e) => {
    appendLog("log-ask", "no_context", JSON.parse(e.data));
    $("#answer-text").textContent = "No relevant information found in the knowledge base.";
  });

  es.addEventListener("prompt_built", (e) => {
    const payload = JSON.parse(e.data);
    appendLog("log-ask", "prompt_built", payload);
    setBoxState("pipeline-ask", "prompt", "done");
    $("#prompt-text").textContent = `[SYSTEM]\n${payload.system_prompt}\n\n[USER]\n${payload.prompt}`;
  });

  es.addEventListener("llm_start", (e) => {
    appendLog("log-ask", "llm_start", JSON.parse(e.data));
    setBoxState("pipeline-ask", "llm", "active");
    setBoxState("pipeline-ask", "answer", "active");
    $("#answer-text").textContent = "";
  });

  es.addEventListener("llm_token", (e) => {
    const payload = JSON.parse(e.data);
    $("#answer-text").textContent += payload.delta;
  });

  es.addEventListener("llm_done", (e) => {
    appendLog("log-ask", "llm_done", JSON.parse(e.data));
    setBoxState("pipeline-ask", "llm", "done");
  });

  es.addEventListener("answer_done", (e) => {
    const payload = JSON.parse(e.data);
    appendLog("log-ask", "answer_done", payload);
    setBoxState("pipeline-ask", "answer", "done");
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

// -------------------------------------------------------------- init ------

checkStatus();
