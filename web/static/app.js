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

// ---------------------------------------------------------- demo session ---
//
// The landing screen's "Start the demo" button mints a UUID client-side and
// sends it as a header on every request from then on — same mechanism as
// the BYOK key above (sessionStorage, header-only, never a cookie or URL
// param), so it isolates this visitor's index rebuilds/questions from every
// other visitor's, with the server auto-expiring it after 24h (see
// src/session_store.py). Skipping cookies keeps this consistent with BYOK
// and means no new server-side session infrastructure beyond a header read.
const SESSION_STORAGE_KEY = "rag_demo_session_id";

function getSessionId() {
  return sessionStorage.getItem(SESSION_STORAGE_KEY) || "";
}

function setSessionId(id) {
  if (id) sessionStorage.setItem(SESSION_STORAGE_KEY, id);
  else sessionStorage.removeItem(SESSION_STORAGE_KEY);
}

function authHeaders() {
  const key = getByokKey();
  const sessionId = getSessionId();
  const headers = {};
  if (key) headers["X-OpenAI-Key"] = key;
  if (sessionId) headers["X-Session-Id"] = sessionId;
  return headers;
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
let trainingInitialized = false;

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
      runTokenizer(); // seed with the textarea's default example text
    }
  }
  if (tabName === "training" && !trainingInitialized) {
    trainingInitialized = true;
    loadFinetuneExamples(); // zero-cost, no key needed — fetch once
  }
}

$$(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => switchToTab(btn.dataset.tab));
});

$$("[data-tab-link]").forEach((link) => {
  link.addEventListener("click", (e) => {
    e.preventDefault();
    switchToTab(link.dataset.tabLink);
    if (link.dataset.mode) setAskMode(link.dataset.mode); // e.g. deep-link straight into Agentic RAG
  });
});

// ------------------------------------------------------- ask mode toggle --
//
// "Classic RAG" / "Agentic RAG" inside the Ask a Question tab — same
// active/hidden show-and-hide pattern as switchToTab's top-level tabs
// above, just scoped to the two panels inside #tab-ask instead of the
// whole page.

let askMode = "classic"; // "classic" | "agentic"

function setAskMode(mode) {
  askMode = mode;
  $$(".mode-btn", $("#tab-ask")).forEach((b) => b.classList.toggle("active", b.dataset.mode === mode));
  $("#ask-mode-classic").classList.toggle("active", mode === "classic");
  $("#ask-mode-classic").hidden = mode !== "classic";
  $("#ask-mode-agentic").classList.toggle("active", mode === "agentic");
  $("#ask-mode-agentic").hidden = mode !== "agentic";
}

$$(".mode-btn", $("#tab-ask")).forEach((btn) => {
  btn.addEventListener("click", () => setAskMode(btn.dataset.mode));
});

// ---------------------------------------------------- prompting lab tabs --
//
// Same active/hidden idiom as setAskMode above, scoped to #tab-prompting
// instead — see that function's comment for why the selector must be
// scoped rather than global.

function setPromptMode(mode) {
  $$(".mode-btn", $("#tab-prompting")).forEach((b) => b.classList.toggle("active", b.dataset.promptMode === mode));
  ["variants", "temperature", "structured"].forEach((m) => {
    const panel = $(`#prompting-mode-${m}`);
    panel.classList.toggle("active", m === mode);
    panel.hidden = m !== mode;
  });
}

$$(".mode-btn", $("#tab-prompting")).forEach((btn) => {
  btn.addEventListener("click", () => setPromptMode(btn.dataset.promptMode));
});

function runPromptVariants() {
  const question = $("#prompting-variants-input").value.trim();
  if (!question) return;
  const btn = $("#btn-prompting-variants");
  btn.disabled = true;
  const t0 = performance.now();
  ["zero_shot", "few_shot", "cot"].forEach((v) => {
    $(`#prompting-variant-${v}`).textContent = "";
  });

  const url = `/api/prompting/variants/stream?question=${encodeURIComponent(question)}`;
  const es = new FetchEventSource(url, { headers: authHeaders() });
  const close = () => { es.close(); btn.disabled = false; checkStatus(); };

  es.addEventListener("variant_token", (e) => {
    const payload = JSON.parse(e.data);
    $(`#prompting-variant-${payload.variant}`).textContent += payload.delta;
  });
  es.addEventListener("no_context", () => {
    ["zero_shot", "few_shot", "cot"].forEach((v) => {
      $(`#prompting-variant-${v}`).textContent = "No relevant information found in the knowledge base.";
    });
  });
  es.addEventListener("pipeline_error", (e) => {
    const payload = JSON.parse(e.data);
    ["zero_shot", "few_shot", "cot"].forEach((v) => {
      if (!$(`#prompting-variant-${v}`).textContent) $(`#prompting-variant-${v}`).textContent = `Error: ${payload.message}`;
    });
    close();
  });
  es.addEventListener("pipeline_done", () => {
    recordTechniqueLatency("prompting_variants", Math.round(performance.now() - t0));
    close();
  });
  es.onerror = () => close();
}

$("#btn-prompting-variants").addEventListener("click", runPromptVariants);

// Temperature is a property of a *distribution*, not of a single draw —
// one sample per value can't show it (a lucky/unlucky single draw looks
// the same regardless of temperature). Each temperature card shows every
// sample plus how much they agree with each other (word-overlap Jaccard,
// same idiom as the Consistency card in Metrics): near 100% at
// temperature=0 (deterministic), visibly lower at temperature=2.
function runTemperaturePlayground() {
  const question = $("#prompting-temperature-input").value.trim();
  if (!question) return;
  const btn = $("#btn-prompting-temperature");
  btn.disabled = true;
  const t0 = performance.now();
  const grid = $("#prompting-temperature-grid");
  grid.innerHTML = "";

  const url = `/api/prompting/temperature/stream?question=${encodeURIComponent(question)}`;
  const es = new FetchEventSource(url, { headers: authHeaders() });
  const close = () => { es.close(); btn.disabled = false; checkStatus(); };

  function cardFor(temperature, nSamples) {
    const id = `prompting-temp-${String(temperature).replace(".", "_")}`;
    let card = $(`#${id}`);
    if (!card) {
      card = document.createElement("div");
      card.className = "card";
      card.id = id;
      let samplesHtml = "";
      for (let i = 0; i < nSamples; i++) {
        samplesHtml += `<div class="temp-sample muted small" id="${id}-sample-${i}">Waiting…</div>`;
      }
      card.innerHTML =
        `<h3>temperature = ${temperature}</h3>` +
        `<div class="temp-samples">${samplesHtml}</div>` +
        `<p class="temp-agreement muted small" id="${id}-agreement"></p>`;
      grid.appendChild(card);
    }
    return card;
  }

  es.addEventListener("temp_start", (e) => {
    const payload = JSON.parse(e.data);
    cardFor(payload.temperature, payload.n_samples);
  });
  es.addEventListener("temp_sample_done", (e) => {
    const payload = JSON.parse(e.data);
    const id = `prompting-temp-${String(payload.temperature).replace(".", "_")}`;
    const el = $(`#${id}-sample-${payload.sample_index}`);
    if (el) {
      el.textContent = `Sample ${payload.sample_index + 1}: ${payload.answer}`;
      el.classList.remove("muted");
    }
  });
  es.addEventListener("temp_summary", (e) => {
    const payload = JSON.parse(e.data);
    const id = `prompting-temp-${String(payload.temperature).replace(".", "_")}`;
    const el = $(`#${id}-agreement`);
    if (el) {
      const pct = Math.round(payload.avg_jaccard_similarity * 100);
      el.textContent = `Agreement across ${payload.n_samples} samples: ${pct}% word-overlap ` +
        (pct >= 90 ? "(near-identical — low temperature stays consistent)" : "(visibly diverging)");
    }
  });
  es.addEventListener("no_context", () => {
    grid.innerHTML = '<p class="muted">No relevant information found in the knowledge base.</p>';
  });
  es.addEventListener("pipeline_error", (e) => {
    const payload = JSON.parse(e.data);
    grid.innerHTML = `<p class="muted">Error: ${payload.message}</p>`;
    close();
  });
  es.addEventListener("pipeline_done", () => {
    recordTechniqueLatency("prompting_temperature", Math.round(performance.now() - t0));
    close();
  });
  es.onerror = () => close();
}

$("#btn-prompting-temperature").addEventListener("click", runTemperaturePlayground);

async function runStructuredOutput() {
  const question = $("#prompting-structured-input").value.trim();
  if (!question) return;
  const btn = $("#btn-prompting-structured");
  const structuredEl = $("#prompting-structured-result");
  const freetextEl = $("#prompting-freetext-result");
  btn.disabled = true;
  structuredEl.textContent = "Running…";
  freetextEl.textContent = "Running…";
  const t0 = performance.now();

  try {
    const res = await fetch(`/api/prompting/structured?question=${encodeURIComponent(question)}`, {
      headers: authHeaders(),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
      structuredEl.textContent = `Error: ${err.detail}`;
      freetextEl.textContent = "";
      return;
    }
    const data = await res.json();
    if (!data.structured) {
      structuredEl.textContent = "No relevant information found in the knowledge base.";
      freetextEl.textContent = "";
      return;
    }
    const problems = data.structured.problems;
    const validationLine = problems.length
      ? `⚠ Schema validation failed: ${problems.join(" ")}`
      : "✓ Schema validation passed.";
    structuredEl.innerHTML =
      `<pre class="code-block">${JSON.stringify(data.structured.payload, null, 2)}</pre>` +
      `<p class="muted small">${validationLine}</p>`;
    freetextEl.textContent = data.freetext.answer;
    recordTechniqueLatency("prompting_structured", Math.round(performance.now() - t0));
  } catch (err) {
    structuredEl.textContent = "Could not run this comparison.";
    freetextEl.textContent = "";
  } finally {
    btn.disabled = false;
    checkStatus();
  }
}

$("#btn-prompting-structured").addEventListener("click", runStructuredOutput);

// -------------------------------------------------------------- training --

function setTrainingMode(mode) {
  $$(".mode-btn", $("#tab-training")).forEach((b) => b.classList.toggle("active", b.dataset.trainingMode === mode));
  ["finetune", "classifier"].forEach((m) => {
    const panel = $(`#training-mode-${m}`);
    panel.classList.toggle("active", m === mode);
    panel.hidden = m !== mode;
  });
}

$$(".mode-btn", $("#tab-training")).forEach((btn) => {
  btn.addEventListener("click", () => setTrainingMode(btn.dataset.trainingMode));
});

async function loadFinetuneExamples() {
  const container = $("#finetune-examples");
  try {
    const res = await fetch("/api/training/finetune-example");
    const examples = await res.json();
    container.innerHTML = "";
    examples.forEach((ex) => {
      const details = document.createElement("details");
      const summary = document.createElement("summary");
      summary.textContent = `${ex.record.messages[1].content} (source: ${ex.source_document})`;
      const pre = document.createElement("pre");
      pre.className = "code-block";
      pre.textContent = JSON.stringify(ex.record, null, 2);
      details.appendChild(summary);
      details.appendChild(pre);
      container.appendChild(details);
    });
  } catch (err) {
    container.innerHTML = '<p class="muted">Could not load the fine-tuning examples.</p>';
  }
}

async function trainClassifier() {
  const btn = $("#btn-classifier-train");
  const result = $("#classifier-train-result");
  btn.disabled = true;
  result.innerHTML = "<p class=\"muted\">Training…</p>";
  try {
    const res = await fetch("/api/training/classifier/train", { headers: authHeaders() });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
      result.innerHTML = `<p class="muted">${err.detail}</p>`;
      return;
    }
    const data = await res.json();
    const counts = data.labels.map((l) => `${l}: ${data.n_examples_per_label[l]} chunks`).join(", ");
    const avgSeparation =
      data.centroid_separation.reduce((sum, p) => sum + p.cosine_similarity, 0) / data.centroid_separation.length;
    result.innerHTML =
      `<p class="small">Trained ${data.labels.length} centroids (one per document): ${counts}</p>` +
      `<p class="muted small">Average pairwise cosine similarity between centroids: ${avgSeparation.toFixed(3)} ` +
      `— the closer to 1.0, the harder the classes are to tell apart with this little data per class.</p>`;
  } catch (err) {
    result.innerHTML = '<p class="muted">Could not train the classifier.</p>';
  } finally {
    btn.disabled = false;
  }
}

$("#btn-classifier-train").addEventListener("click", trainClassifier);

const CLASSIFIER_COMPARE_TABLE_HEADERS = ["Question", "Expected", "Classifier prediction", "RAG top-1"];

function classifierCompareRowCells(item) {
  return [item.question, item.expected_sources.join(", "), item.classifier_prediction, item.rag_top1];
}

function resetClassifierCompareUI() {
  $("#classifier-compare-progress").textContent = "";
  $("#metric-classifier-accuracy").textContent = "—";
  $("#metric-rag-top1-accuracy").textContent = "—";
  $("#classifier-compare-table-section").hidden = true;
  $("#classifier-compare-table").innerHTML = "";
}

async function loadClassifierCompareSnapshot() {
  try {
    const res = await fetch("/api/eval/snapshot");
    if (!res.ok) return;
    const snap = await res.json();
    if (snap.classifier_accuracy == null || snap.rag_top1_accuracy == null) {
      $("#classifier-compare-source-label").textContent = "No snapshot committed yet — click \"Run comparison live\" below.";
      return;
    }
    $("#metric-classifier-accuracy").textContent = `${Math.round(snap.classifier_accuracy * 100)}%`;
    $("#metric-rag-top1-accuracy").textContent = `${Math.round(snap.rag_top1_accuracy * 100)}%`;
    $("#classifier-compare-source-label").textContent =
      `Snapshot computed ${fmtSnapshotDate(snap.generated_at)} against this exact index — committed to the repo, not live.`;

    const tbody = createResultsTable($("#classifier-compare-table"), CLASSIFIER_COMPARE_TABLE_HEADERS);
    $("#classifier-compare-table-section").hidden = false;
    (snap.classifier_items || []).forEach((item) => {
      appendResultRow(tbody, classifierCompareRowCells(item), [
        { index: 2, ok: item.classifier_hit },
        { index: 3, ok: item.rag_hit },
      ]);
    });
  } catch (err) {
    $("#classifier-compare-source-label").textContent = "Could not load the comparison snapshot.";
  }
}

function runClassifierComparison() {
  const btn = $("#btn-classifier-compare");
  btn.disabled = true;
  const t0 = performance.now();
  resetClassifierCompareUI();
  $("#classifier-compare-source-label").textContent = "Running live against your own key…";
  $("#classifier-compare-progress").textContent = "Embedding each question and predicting both ways…";

  const tbody = createResultsTable($("#classifier-compare-table"), CLASSIFIER_COMPARE_TABLE_HEADERS);
  $("#classifier-compare-table-section").hidden = false;

  const es = new FetchEventSource("/api/training/classifier/compare/stream", { headers: authHeaders() });
  const close = () => {
    es.close();
    btn.disabled = false;
  };

  es.addEventListener("classifier_item", (e) => {
    const payload = JSON.parse(e.data);
    appendLog("log-classifier-compare", "classifier_item", payload);
    appendResultRow(tbody, classifierCompareRowCells(payload), [
      { index: 2, ok: payload.classifier_hit },
      { index: 3, ok: payload.rag_hit },
    ]);
  });

  es.addEventListener("classifier_done", (e) => {
    const payload = JSON.parse(e.data);
    appendLog("log-classifier-compare", "classifier_done", payload);
    $("#metric-classifier-accuracy").textContent = `${Math.round(payload.classifier_accuracy * 100)}%`;
    $("#metric-rag-top1-accuracy").textContent = `${Math.round(payload.rag_top1_accuracy * 100)}%`;
  });

  es.addEventListener("pipeline_done", (e) => {
    appendLog("log-classifier-compare", "pipeline_done", JSON.parse(e.data));
    $("#classifier-compare-progress").textContent = "Done.";
    $("#classifier-compare-source-label").textContent = "Live run — just now, against this exact index, with your key.";
    recordTechniqueLatency("classifier_compare", Math.round(performance.now() - t0));
    close();
  });

  es.addEventListener("pipeline_error", (e) => {
    const payload = JSON.parse(e.data);
    appendLog("log-classifier-compare", "pipeline_error", payload, true);
    $("#classifier-compare-progress").textContent = `Error: ${payload.message}`;
    $("#classifier-compare-source-label").textContent = "Live run failed — showing whatever loaded before the error.";
    close();
  });

  es.onerror = () => close();
}

$("#btn-classifier-compare").addEventListener("click", runClassifierComparison);
loadClassifierCompareSnapshot();

// --------------------------------------------------------------- status ---

async function checkStatus() {
  const banner = $("#status-banner");
  try {
    const res = await fetch("/api/status", { headers: authHeaders() });
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

  // Prompting Lab: same gating as Ask (needs a built index + a usable key).
  [
    ["btn-prompting-variants", "prompting-variants-disabled-hint"],
    ["btn-prompting-temperature", "prompting-temperature-disabled-hint"],
    ["btn-prompting-structured", "prompting-structured-disabled-hint"],
  ].forEach(([btnId, hintId]) => {
    const promptBtn = $(`#${btnId}`);
    const promptHint = $(`#${hintId}`);
    if (!promptBtn) return;
    promptBtn.disabled = disabled;
    if (promptHint) {
      promptHint.hidden = !disabled;
      promptHint.textContent = !hasUsableKey ? missingKeyHint : "Build the index first, in the \"Build the Index\" tab.";
    }
  });

  // Agent Frameworks: same gating as Ask/Prompting Lab.
  const langgraphBtn = $("#btn-langgraph-run");
  const langgraphHint = $("#langgraph-disabled-hint");
  if (langgraphBtn) {
    langgraphBtn.disabled = disabled;
    if (langgraphHint) {
      langgraphHint.hidden = !disabled;
      langgraphHint.textContent = !hasUsableKey ? missingKeyHint : "Build the index first, in the \"Build the Index\" tab.";
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
  $("#rerank-info").hidden = true;
  $("#rerank-info-content").innerHTML = "";
  ["question", "embedding", "search", "topk", "prompt", "llm", "answer"].forEach((b) => setAskBoxDetail(b, "—"));
  askBoxPayloads.clear();
  hideBoxOutput("ask");
}

function renderRerankInfo(before, after) {
  const container = $("#rerank-info-content");
  container.innerHTML = "";
  const beforeList = before.map((c) => `${c.source} #${c.chunk_index}`).join(", ");
  const afterList = after
    .map((c) => `${c.source} #${c.chunk_index} (${c.rerank_score.toFixed(1)})`)
    .join(", ");
  const p1 = document.createElement("p");
  p1.className = "muted small";
  p1.textContent = `Top by cosine similarity (before): ${beforeList}`;
  const p2 = document.createElement("p");
  p2.className = "small";
  p2.textContent = `Top by LLM relevance score (after): ${afterList}`;
  container.appendChild(p1);
  container.appendChild(p2);
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
    const res = await fetch(`/api/kb/similarity?source=${encodeURIComponent(source)}&chunk_index=${chunkIndex}`, { headers: authHeaders() });
    if (!res.ok) {
      refLabelEl.textContent = "Could not load that chunk's neighbors.";
      return;
    }
    const data = await res.json();
    const self = data.all_scores.find((s) => s.source === source && s.chunk_index === chunkIndex);
    // No topKSources here on purpose: "top-K used as context" isn't a
    // meaningful concept for an arbitrary chunk-to-chunk similarity view,
    // so renderVectorMap skips the lines/highlighting entirely.
    renderVectorMap($("#vector-map"), data.all_scores, [], { x: self.x, y: self.y, z: self.z }, label);
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

$("#vector-map-reset-view").addEventListener("click", () => {
  if (window.VectorMap3D) window.VectorMap3D.resetView();
});

// Thin adapter over the 3D renderer in vector-map-3d.js (Three.js — this
// project's one frontend dependency, vendored locally, loaded as a
// separate <script type="module">). Kept under the same name/signature
// the rest of this file already calls (reroot, "back to your question",
// the live search_done handler), so none of those call sites needed to
// change — only this function's body did, when the map moved from a
// hand-rolled SVG scatter to a real 3D scene.
function renderVectorMap(container, allScores, topKSources, referencePoint, referenceLabel) {
  if (!window.VectorMap3D) {
    container.innerHTML = '<p class="muted">3D map unavailable.</p>';
    return;
  }
  window.VectorMap3D.render(container, allScores, topKSources, referencePoint, referenceLabel, rerootVectorMap);
}

function askQuestionClassic(question) {
  const btn = $("#btn-ask");
  btn.disabled = true;
  resetAskUI();
  setBoxState("pipeline-ask", "question", "active");
  currentTiming = null;

  const rerank = $("#rerank-toggle")?.checked;
  const url = `/api/ask/stream?question=${encodeURIComponent(question)}${rerank ? "&rerank=true" : ""}`;
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

  es.addEventListener("rerank_start", (e) => {
    appendLog("log-ask", "rerank_start", JSON.parse(e.data));
  });

  es.addEventListener("rerank_done", (e) => {
    const payload = JSON.parse(e.data);
    appendLog("log-ask", "rerank_done", payload);
    $("#rerank-info").hidden = false;
    renderRerankInfo(payload.before, payload.after);
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
      recordTechniqueLatency(rerank ? "classic_rerank" : "classic", currentTiming.total_ms);
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
    lastClassicResult = {
      question,
      answer: payload.answer,
      sources: payload.sources,
      chunks: payload.chunks || [],
      iterations: 1,
      elapsed_ms: currentTiming ? currentTiming.total_ms : null,
    };
    maybeRenderComparison();
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

// ------------------------------------------------------------- agentic ----
//
// Same question, different pipeline: the model decides via tool calling
// whether to call retrieve_context zero, one, or several times (see
// src/agentic_rag.py). Reuses embedding_query_*/search_* events emitted by
// the same retriever.retrieve() classic mode uses, each tagged with which
// loop iteration they belong to — so unlike the classic handlers above,
// these are additive (append to a growing trace) instead of overwrite.

let agenticTraceSteps = {}; // iteration -> DOM elements for that step
let agenticStepOrder = [];  // iteration numbers in the order their steps were created

function resetAgenticUI() {
  $("#agentic-trace").innerHTML = '<p class="muted">The step-by-step trace will appear here once you ask a question in this mode.</p>';
  $("#agentic-answer-text").textContent = "—";
  $("#agentic-sources-chips").innerHTML = "";
  $("#agentic-summary").textContent = "Ask a question in this mode to see how many searches it took.";
  $("#log-agentic").innerHTML = "";
  agenticTraceSteps = {};
  agenticStepOrder = [];
}

function startTraceStep(iteration) {
  const trace = $("#agentic-trace");
  if (agenticStepOrder.length === 0) trace.innerHTML = ""; // clear the placeholder on first real step

  // The previous step, if any, gets one more line noting it decided to
  // search again — the only signal of that decision is a new iteration
  // starting at all, there's no separate "search again" event.
  if (agenticStepOrder.length > 0) {
    const prevEl = agenticTraceSteps[agenticStepOrder[agenticStepOrder.length - 1]];
    if (prevEl && !prevEl.dataset.resolved) {
      const decision = document.createElement("div");
      decision.className = "trace-decision";
      decision.textContent = "→ decided to search again";
      prevEl.appendChild(decision);
      prevEl.dataset.resolved = "1";
    }
  }

  const step = document.createElement("div");
  step.className = "trace-step";
  const head = document.createElement("div");
  head.className = "trace-step-head";
  const badge = document.createElement("span");
  badge.className = "trace-step-badge";
  badge.textContent = `Iteration ${iteration}`;
  head.appendChild(badge);
  step.appendChild(head);
  trace.appendChild(step);

  agenticTraceSteps[iteration] = step;
  agenticStepOrder.push(iteration);
  return step;
}

function appendTraceQuery(iteration, query) {
  const step = agenticTraceSteps[iteration];
  if (!step) return;
  const q = document.createElement("div");
  q.className = "trace-query";
  q.append("→ searched: ");
  const queryText = document.createElement("span");
  queryText.className = "trace-query-text";
  queryText.textContent = `"${query}"`;
  q.appendChild(queryText);
  step.appendChild(q);
}

function appendTraceChunks(iteration, allScores, topKSources) {
  const step = agenticTraceSteps[iteration];
  if (!step) return;
  // Same "which rows are really the returned set" trick as renderScoreBars:
  // allScores is sorted by score, the first N whose source is in
  // topKSources are the chunks retrieve_context actually returned.
  const remaining = new Map();
  topKSources.forEach((src) => remaining.set(src, (remaining.get(src) || 0) + 1));
  const list = document.createElement("ul");
  list.className = "trace-chunks";
  allScores.forEach((s) => {
    const left = remaining.get(s.source) || 0;
    if (left <= 0) return;
    remaining.set(s.source, left - 1);
    const li = document.createElement("li");
    li.textContent = `${s.source} #${s.chunk_index} (score ${s.score.toFixed(3)})`;
    list.appendChild(li);
  });
  step.appendChild(list);
}

function resolveTraceStep(iteration, text) {
  const step = agenticTraceSteps[iteration];
  if (!step || step.dataset.resolved) return;
  const decision = document.createElement("div");
  decision.className = "trace-decision decided";
  decision.textContent = text;
  step.appendChild(decision);
  step.dataset.resolved = "1";
}

function askQuestionAgentic(question) {
  const btn = $("#btn-ask");
  btn.disabled = true;
  resetAgenticUI();

  const url = `/api/ask/agentic/stream?question=${encodeURIComponent(question)}`;
  const es = new FetchEventSource(url, { headers: authHeaders() });
  const close = () => {
    es.close();
    btn.disabled = false;
    checkStatus();
  };

  es.addEventListener("question_received", (e) => {
    appendLog("log-agentic", "question_received", JSON.parse(e.data));
  });

  es.addEventListener("agent_iteration_start", (e) => {
    const payload = JSON.parse(e.data);
    appendLog("log-agentic", "agent_iteration_start", payload);
    startTraceStep(payload.iteration);
  });

  es.addEventListener("agent_tool_call", (e) => {
    const payload = JSON.parse(e.data);
    appendLog("log-agentic", "agent_tool_call", payload);
    appendTraceQuery(payload.iteration, payload.query);
  });

  es.addEventListener("embedding_query_start", (e) => {
    appendLog("log-agentic", "embedding_query_start", JSON.parse(e.data));
  });
  es.addEventListener("embedding_query_done", (e) => {
    appendLog("log-agentic", "embedding_query_done", JSON.parse(e.data));
  });
  es.addEventListener("search_start", (e) => {
    appendLog("log-agentic", "search_start", JSON.parse(e.data));
  });
  es.addEventListener("search_done", (e) => {
    const payload = JSON.parse(e.data);
    appendLog("log-agentic", "search_done", payload);
    appendTraceChunks(payload.iteration, payload.all_scores, payload.top_k_sources);
  });

  es.addEventListener("agent_no_tool_call", (e) => {
    const payload = JSON.parse(e.data);
    appendLog("log-agentic", "agent_no_tool_call", payload);
    resolveTraceStep(payload.iteration, "→ decided it has enough context — answering now");
  });

  es.addEventListener("agent_answer", (e) => {
    appendLog("log-agentic", "agent_answer", JSON.parse(e.data));
  });

  es.addEventListener("agent_done", (e) => {
    const payload = JSON.parse(e.data);
    appendLog("log-agentic", "agent_done", payload);
    $("#agentic-answer-text").textContent = payload.answer;
    const chips = $("#agentic-sources-chips");
    chips.innerHTML = "";
    payload.sources.forEach((src) => {
      const chip = document.createElement("span");
      chip.className = "chip";
      chip.textContent = src;
      chips.appendChild(chip);
    });
    $("#agentic-summary").innerHTML = "";
    const summary = $("#agentic-summary");
    [
      ["Iterations", payload.iterations],
      ["Chunks retrieved (total)", payload.chunks.length],
      ["Elapsed", `${payload.elapsed_ms}ms`],
      ["Sources cited", payload.sources.length],
    ].forEach(([label, value], i) => {
      if (i > 0) summary.append(" · ");
      summary.append(`${label}: `);
      const strong = document.createElement("strong");
      strong.textContent = value;
      summary.appendChild(strong);
    });
    lastAgenticResult = {
      question,
      answer: payload.answer,
      sources: payload.sources,
      chunks: payload.chunks,
      iterations: payload.iterations,
      elapsed_ms: payload.elapsed_ms,
    };
    maybeRenderComparison();
  });

  es.addEventListener("agent_error", (e) => {
    const payload = JSON.parse(e.data);
    appendLog("log-agentic", "agent_error", payload, true);
    $("#agentic-answer-text").textContent = `Error: ${payload.message}`;
  });

  es.addEventListener("pipeline_done", (e) => {
    appendLog("log-agentic", "pipeline_done", JSON.parse(e.data));
    close();
  });

  es.addEventListener("pipeline_error", (e) => {
    const payload = JSON.parse(e.data);
    appendLog("log-agentic", "pipeline_error", payload, true);
    $("#agentic-answer-text").textContent = `Error: ${payload.message}`;
    close();
  });

  es.onerror = () => close();
}

// --------------------------------------------------- classic vs agentic ---

let lastClassicResult = null; // { question, answer, sources, chunks, iterations, elapsed_ms }
let lastAgenticResult = null;

function maybeRenderComparison() {
  if (!lastClassicResult || !lastAgenticResult) return;
  if (lastClassicResult.question !== lastAgenticResult.question) return;
  renderComparisonCard(lastClassicResult, lastAgenticResult);
}

function renderComparisonCard(classicResult, agenticResult) {
  const card = $("#ask-comparison");
  const content = $("#ask-comparison-content");
  content.innerHTML = "";

  const grid = document.createElement("div");
  grid.className = "ask-comparison-grid";

  [["Classic", classicResult], ["Agentic", agenticResult]].forEach(([label, r]) => {
    const col = document.createElement("div");
    col.className = "ask-comparison-col";
    const h4 = document.createElement("h4");
    h4.textContent = label;
    col.appendChild(h4);

    const stats = [
      ["Searches", r.iterations],
      ["Chunks retrieved", r.chunks.length],
      ["Sources cited", r.sources.length],
      ["Answer length", `${r.answer.split(/\s+/).length} words`],
      ["Elapsed", r.elapsed_ms != null ? `${r.elapsed_ms}ms` : "—"],
    ];
    stats.forEach(([statLabel, value]) => {
      const row = document.createElement("div");
      row.className = "ask-comparison-stat";
      const span = document.createElement("span");
      span.textContent = statLabel;
      const strong = document.createElement("strong");
      strong.textContent = value;
      row.appendChild(span);
      row.appendChild(strong);
      col.appendChild(row);
    });
    grid.appendChild(col);
  });

  content.appendChild(grid);
  card.hidden = false;
}

// --------------------------------------------------- example question -----

const AGENTIC_EXAMPLE_QUESTION = "¿Cómo cancelo un turno y puedo pedir reembolso?";

$("#btn-agentic-example").addEventListener("click", () => {
  const input = $("#question-input");
  input.value = AGENTIC_EXAMPLE_QUESTION;
  input.focus();
});

// --------------------------------------------------------- dispatcher -----

function askQuestion() {
  const question = $("#question-input").value.trim();
  if (!question) return;
  if (askMode === "agentic") askQuestionAgentic(question);
  else askQuestionClassic(question);
}

$("#btn-ask").addEventListener("click", askQuestion);
$("#question-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter") askQuestion();
});

// =================================================== AGENT FRAMEWORKS =====
//
// The same bounded ReAct agent as the Ask tab's "Agentic RAG" mode, run
// twice for one question — once hand-rolled (src/agentic_rag.py, reused
// as-is), once via LangGraph's StateGraph (src/langgraph_agent.py) — in a
// single request (/api/ask/agentic-compare/stream) so it's one rate-limit
// hit, not two. Every event from that stream carries a `run:
// "handrolled"|"langgraph"` tag; only the langgraph run drives the graph
// diagram, since the hand-rolled run has no graph to animate.

let graphLoopCount = 0;
let langgraphResult = null;
let handrolledResult = null;

function setGraphNodeState(node, state) {
  const el = $(`.graph-node[data-node="${node}"]`);
  if (!el) return;
  el.classList.remove("active", "done");
  if (state) el.classList.add(state);
}

function setGraphEdgeState(edge, state) {
  // Backend sends "agent->tools" etc.; data-edge attributes use hyphens
  // (arrows are awkward in HTML attribute values).
  const key = edge.replace("->", "-");
  const el = $(`.graph-edge[data-edge="${key}"]`);
  if (!el) return;
  el.classList.remove("taken", "done");
  if (state) el.classList.add(state);
  if (key === "tools-agent" && state === "taken") {
    graphLoopCount += 1;
    const badge = $("#graph-loop-count");
    if (graphLoopCount > 1) {
      badge.hidden = false;
      badge.textContent = `looped ×${graphLoopCount}`;
    }
  }
}

function resetAgentFrameworksUI() {
  $("#langgraph-progress").textContent = "";
  graphLoopCount = 0;
  langgraphResult = null;
  handrolledResult = null;
  $$(".graph-node").forEach((n) => n.classList.remove("active", "done"));
  $$(".graph-edge").forEach((e) => e.classList.remove("taken", "done"));
  $("#graph-loop-count").hidden = true;
  $("#graph-node-detail-agent").textContent = "—";
  $("#graph-node-detail-tools").textContent = "—";
  $("#langgraph-answer-text").textContent = "—";
  $("#langgraph-sources-chips").innerHTML = "";
  $("#handrolled-answer-text").textContent = "—";
  $("#handrolled-sources-chips").innerHTML = "";
  $("#agentic-frameworks-comparison-content").innerHTML = "";
  $("#log-langgraph").innerHTML = "";
}

function renderSourceChips(containerId, sources) {
  const chips = $(`#${containerId}`);
  chips.innerHTML = "";
  sources.forEach((src) => {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.textContent = src;
    chips.appendChild(chip);
  });
}

function maybeRenderAgentFrameworksComparison() {
  if (!langgraphResult || !handrolledResult) return;
  const content = $("#agentic-frameworks-comparison-content");
  content.innerHTML = "";
  const grid = document.createElement("div");
  grid.className = "ask-comparison-grid";

  [["Hand-rolled", handrolledResult], ["LangGraph", langgraphResult]].forEach(([label, r]) => {
    const col = document.createElement("div");
    col.className = "ask-comparison-col";
    const h4 = document.createElement("h4");
    h4.textContent = label;
    col.appendChild(h4);

    const stats = [
      ["Searches", r.iterations],
      ["Chunks retrieved", r.chunks.length],
      ["Sources cited", r.sources.length],
      ["Elapsed", r.elapsed_ms != null ? `${r.elapsed_ms}ms` : "—"],
      ["Dependencies added", label === "LangGraph" ? "~25.5MB (langgraph + langchain-core)" : "0"],
    ];
    stats.forEach(([statLabel, value]) => {
      const row = document.createElement("div");
      row.className = "ask-comparison-stat";
      const span = document.createElement("span");
      span.textContent = statLabel;
      const strong = document.createElement("strong");
      strong.textContent = value;
      row.appendChild(span);
      row.appendChild(strong);
      col.appendChild(row);
    });
    grid.appendChild(col);
  });

  content.appendChild(grid);
}

function runAgentFrameworksCompare() {
  const question = $("#langgraph-question-input").value.trim();
  if (!question) return;
  const btn = $("#btn-langgraph-run");
  btn.disabled = true;
  resetAgentFrameworksUI();
  $("#langgraph-progress").textContent = "Running the hand-rolled agent first…";
  const t0 = performance.now();

  const url = `/api/ask/agentic-compare/stream?question=${encodeURIComponent(question)}`;
  const es = new FetchEventSource(url, { headers: authHeaders() });
  const close = () => {
    es.close();
    btn.disabled = false;
    checkStatus();
  };

  // ---- LangGraph-only: drive the live graph diagram ----
  es.addEventListener("graph_node_start", (e) => {
    const payload = JSON.parse(e.data);
    appendLog("log-langgraph", "graph_node_start", payload);
    if (payload.run !== "langgraph") return;
    setGraphNodeState(payload.node, "active");
  });
  es.addEventListener("graph_node_done", (e) => {
    const payload = JSON.parse(e.data);
    appendLog("log-langgraph", "graph_node_done", payload);
    if (payload.run !== "langgraph") return;
    setGraphNodeState(payload.node, "done");
  });
  es.addEventListener("graph_edge_taken", (e) => {
    const payload = JSON.parse(e.data);
    appendLog("log-langgraph", "graph_edge_taken", payload);
    if (payload.run !== "langgraph") return;
    setGraphEdgeState(payload.edge, "taken");
    if (payload.edge === "agent->end") setGraphNodeState("end", "done");
  });
  es.addEventListener("agent_tool_call", (e) => {
    const payload = JSON.parse(e.data);
    appendLog("log-langgraph", "agent_tool_call", payload);
    if (payload.run !== "langgraph") return;
    $("#graph-node-detail-tools").textContent = `"${payload.query}"`;
  });

  // ---- both runs: log everything, capture each run's final result ----
  ["agent_iteration_start", "embedding_query_start", "embedding_query_done",
    "search_start", "search_done", "agent_no_tool_call", "agent_answer"].forEach((name) => {
    es.addEventListener(name, (e) => appendLog("log-langgraph", name, JSON.parse(e.data)));
  });
  es.addEventListener("question_received", (e) => {
    const payload = JSON.parse(e.data);
    appendLog("log-langgraph", "question_received", payload);
    if (payload.run === "langgraph") $("#langgraph-progress").textContent = "Hand-rolled done — now running the LangGraph agent…";
  });

  es.addEventListener("agent_done", (e) => {
    const payload = JSON.parse(e.data);
    appendLog("log-langgraph", "agent_done", payload);
    const result = { answer: payload.answer, sources: payload.sources, chunks: payload.chunks, iterations: payload.iterations, elapsed_ms: payload.elapsed_ms };
    if (payload.run === "langgraph") {
      langgraphResult = result;
      $("#langgraph-answer-text").textContent = payload.answer;
      renderSourceChips("langgraph-sources-chips", payload.sources);
      // If it never looped, agent->end still needs marking done for a
      // clean final frame (agent_no_tool_call fires but that's a log
      // event, not a node-state one).
      setGraphNodeState("agent", "done");
    } else {
      handrolledResult = result;
      $("#handrolled-answer-text").textContent = payload.answer;
      renderSourceChips("handrolled-sources-chips", payload.sources);
    }
    maybeRenderAgentFrameworksComparison();
  });

  es.addEventListener("agent_error", (e) => {
    const payload = JSON.parse(e.data);
    appendLog("log-langgraph", "agent_error", payload, true);
    const target = payload.run === "langgraph" ? "#langgraph-answer-text" : "#handrolled-answer-text";
    $(target).textContent = `Error: ${payload.message}`;
  });

  es.addEventListener("pipeline_done", (e) => {
    appendLog("log-langgraph", "pipeline_done", JSON.parse(e.data));
    $("#langgraph-progress").textContent = "Done.";
    recordTechniqueLatency("agentic_compare", Math.round(performance.now() - t0));
    close();
  });
  es.addEventListener("pipeline_error", (e) => {
    const payload = JSON.parse(e.data);
    appendLog("log-langgraph", "pipeline_error", payload, true);
    $("#langgraph-progress").textContent = `Error: ${payload.message}`;
    $("#langgraph-answer-text").textContent = `Error: ${payload.message}`;
    close();
  });

  es.onerror = () => close();
}

$("#btn-langgraph-run").addEventListener("click", runAgentFrameworksCompare);

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

// ------------------------------------------------- Explore tab: tokenizer --
//
// Zero-cost, zero-key: tiktoken runs locally on the server (no OpenAI
// call), so this can run freely and often, unlike everything in the Ask
// tab. Shows the real gpt-4o-mini tokenizer against the word-based
// chunking shown one card above, making "word count is a proxy for token
// count" (see README) a measured ratio instead of a claim.

async function runTokenizer() {
  const text = $("#tokenizer-input").value;
  const statsEl = $("#tokenizer-stats");
  const vizEl = $("#tokenizer-viz");
  if (!text.trim()) {
    statsEl.textContent = "";
    vizEl.innerHTML = "";
    return;
  }
  statsEl.textContent = "Tokenizing…";
  try {
    const res = await fetch(`/api/tokenize?text=${encodeURIComponent(text)}`);
    const data = await res.json();
    statsEl.textContent =
      `${data.n_tokens} tokens (${data.encoding}) from ${data.n_words} words ` +
      `→ ${data.words_per_token} words/token`;
    vizEl.innerHTML = "";
    data.tokens.forEach((tok, i) => {
      const span = document.createElement("span");
      span.className = `token-span t${i % 6}`;
      span.textContent = tok.text;
      span.title = `token id ${tok.id}`;
      vizEl.appendChild(span);
    });
  } catch (err) {
    statsEl.textContent = "Could not tokenize this text.";
    vizEl.innerHTML = "";
  }
}

$("#tokenizer-run-btn")?.addEventListener("click", runTokenizer);

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
  const okCells = Array.isArray(okCell) ? okCell : okCell ? [okCell] : [];
  const tr = document.createElement("tr");
  cells.forEach((text, i) => {
    const td = document.createElement("td");
    td.textContent = text;
    const match = okCells.find((oc) => oc.index === i);
    if (match) td.className = match.ok ? "eval-ok" : "eval-bad";
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

// ======================================================== COMPARE ======
//
// Runs eval/evaluate.py's evaluate_retrieval_comparison() — the same
// Recall@K question set, run once through the real embeddings retriever
// and once through a naive keyword-overlap baseline (no OpenAI calls),
// so the gap between them is a measured number, not a claim.

const COMPARE_TABLE_HEADERS = ["Question", "Expected", "Embeddings", "Keyword search"];

function compareRowCells(item) {
  return [
    item.question,
    item.expected_sources.join(", "),
    item.rag_hit ? "hit" : "miss",
    item.keyword_hit ? "hit" : "miss",
  ];
}

function resetCompareUI() {
  $("#compare-progress").textContent = "";
  $("#metric-rag-recall").textContent = "—";
  $("#metric-keyword-recall").textContent = "—";
  $("#compare-table-section").hidden = true;
  $("#compare-table").innerHTML = "";
}

async function loadCompareSnapshot() {
  try {
    const res = await fetch("/api/eval/snapshot");
    if (!res.ok) return;
    const snap = await res.json();
    if (snap.rag_recall == null || snap.keyword_recall == null) {
      $("#compare-source-label").textContent = "No snapshot committed yet — click \"Run comparison live\" below.";
      return;
    }
    $("#metric-rag-recall").textContent = `${Math.round(snap.rag_recall * 100)}%`;
    $("#metric-keyword-recall").textContent = `${Math.round(snap.keyword_recall * 100)}%`;
    $("#compare-source-label").textContent =
      `Snapshot computed ${fmtSnapshotDate(snap.generated_at)} against this exact index — committed to the repo, not live.`;

    const tbody = createResultsTable($("#compare-table"), COMPARE_TABLE_HEADERS);
    $("#compare-table-section").hidden = false;
    (snap.compare_items || []).forEach((item) => {
      appendResultRow(tbody, compareRowCells(item), [
        { index: 2, ok: item.rag_hit },
        { index: 3, ok: item.keyword_hit },
      ]);
    });
  } catch (err) {
    $("#compare-source-label").textContent = "Could not load the comparison snapshot.";
  }
}

function runComparison() {
  const btn = $("#btn-compare");
  btn.disabled = true;
  resetCompareUI();
  $("#compare-source-label").textContent = "Running live against your own key…";
  $("#compare-progress").textContent = "Running Recall@K for both retrievers…";

  const tbody = createResultsTable($("#compare-table"), COMPARE_TABLE_HEADERS);
  $("#compare-table-section").hidden = false;

  const es = new FetchEventSource("/api/eval/compare/stream", { headers: authHeaders() });
  const close = () => {
    es.close();
    btn.disabled = false;
  };

  es.addEventListener("compare_item", (e) => {
    const payload = JSON.parse(e.data);
    appendLog("log-compare", "compare_item", payload);
    appendResultRow(tbody, compareRowCells(payload), [
      { index: 2, ok: payload.rag_hit },
      { index: 3, ok: payload.keyword_hit },
    ]);
  });

  es.addEventListener("compare_done", (e) => {
    const payload = JSON.parse(e.data);
    appendLog("log-compare", "compare_done", payload);
    $("#metric-rag-recall").textContent = `${Math.round(payload.rag_recall * 100)}%`;
    $("#metric-keyword-recall").textContent = `${Math.round(payload.keyword_recall * 100)}%`;
  });

  es.addEventListener("pipeline_done", (e) => {
    appendLog("log-compare", "pipeline_done", JSON.parse(e.data));
    $("#compare-progress").textContent = "Done.";
    $("#compare-source-label").textContent = "Live run — just now, against this exact index, with your key.";
    close();
  });

  es.addEventListener("pipeline_error", (e) => {
    const payload = JSON.parse(e.data);
    appendLog("log-compare", "pipeline_error", payload, true);
    $("#compare-progress").textContent = `Error: ${payload.message}`;
    $("#compare-source-label").textContent = "Live run failed — showing whatever loaded before the error.";
    close();
  });

  es.onerror = () => close();
}

$("#btn-compare").addEventListener("click", runComparison);
loadCompareSnapshot();

// ================================================== RERANK COMPARISON ======
//
// Runs eval/evaluate.py's evaluate_rerank_comparison() — Recall@K on the
// same 8-candidate pool, with vs. without the LLM-based reranker
// (src/reranker.py) re-sorting it — structurally identical to the
// embeddings-vs-keyword card above, different columns.

const RERANK_COMPARE_TABLE_HEADERS = ["Question", "Expected", "Without reranking", "With reranking"];

function rerankCompareRowCells(item) {
  return [
    item.question,
    item.expected_sources.join(", "),
    item.no_rerank_hit ? "hit" : "miss",
    item.rerank_hit ? "hit" : "miss",
  ];
}

function resetRerankCompareUI() {
  $("#rerank-compare-progress").textContent = "";
  $("#metric-no-rerank-recall").textContent = "—";
  $("#metric-rerank-recall").textContent = "—";
  $("#rerank-compare-table-section").hidden = true;
  $("#rerank-compare-table").innerHTML = "";
}

async function loadRerankCompareSnapshot() {
  try {
    const res = await fetch("/api/eval/snapshot");
    if (!res.ok) return;
    const snap = await res.json();
    if (snap.no_rerank_recall == null || snap.rerank_recall == null) {
      $("#rerank-compare-source-label").textContent = "No snapshot committed yet — click \"Run comparison live\" below.";
      return;
    }
    $("#metric-no-rerank-recall").textContent = `${Math.round(snap.no_rerank_recall * 100)}%`;
    $("#metric-rerank-recall").textContent = `${Math.round(snap.rerank_recall * 100)}%`;
    $("#rerank-compare-source-label").textContent =
      `Snapshot computed ${fmtSnapshotDate(snap.generated_at)} against this exact index — committed to the repo, not live.`;

    const tbody = createResultsTable($("#rerank-compare-table"), RERANK_COMPARE_TABLE_HEADERS);
    $("#rerank-compare-table-section").hidden = false;
    (snap.rerank_compare_items || []).forEach((item) => {
      appendResultRow(tbody, rerankCompareRowCells(item), [
        { index: 2, ok: item.no_rerank_hit },
        { index: 3, ok: item.rerank_hit },
      ]);
    });
  } catch (err) {
    $("#rerank-compare-source-label").textContent = "Could not load the comparison snapshot.";
  }
}

function runRerankComparison() {
  const btn = $("#btn-rerank-compare");
  btn.disabled = true;
  resetRerankCompareUI();
  $("#rerank-compare-source-label").textContent = "Running live against your own key…";
  $("#rerank-compare-progress").textContent = "Retrieving 8 candidates and reranking each question…";

  const tbody = createResultsTable($("#rerank-compare-table"), RERANK_COMPARE_TABLE_HEADERS);
  $("#rerank-compare-table-section").hidden = false;

  const es = new FetchEventSource("/api/eval/rerank/stream", { headers: authHeaders() });
  const close = () => {
    es.close();
    btn.disabled = false;
  };

  es.addEventListener("rerank_compare_item", (e) => {
    const payload = JSON.parse(e.data);
    appendLog("log-rerank-compare", "rerank_compare_item", payload);
    appendResultRow(tbody, rerankCompareRowCells(payload), [
      { index: 2, ok: payload.no_rerank_hit },
      { index: 3, ok: payload.rerank_hit },
    ]);
  });

  es.addEventListener("rerank_compare_done", (e) => {
    const payload = JSON.parse(e.data);
    appendLog("log-rerank-compare", "rerank_compare_done", payload);
    $("#metric-no-rerank-recall").textContent = `${Math.round(payload.no_rerank_recall * 100)}%`;
    $("#metric-rerank-recall").textContent = `${Math.round(payload.rerank_recall * 100)}%`;
  });

  es.addEventListener("pipeline_done", (e) => {
    appendLog("log-rerank-compare", "pipeline_done", JSON.parse(e.data));
    $("#rerank-compare-progress").textContent = "Done.";
    $("#rerank-compare-source-label").textContent = "Live run — just now, against this exact index, with your key.";
    close();
  });

  es.addEventListener("pipeline_error", (e) => {
    const payload = JSON.parse(e.data);
    appendLog("log-rerank-compare", "pipeline_error", payload, true);
    $("#rerank-compare-progress").textContent = `Error: ${payload.message}`;
    $("#rerank-compare-source-label").textContent = "Live run failed — showing whatever loaded before the error.";
    close();
  });

  es.onerror = () => close();
}

$("#btn-rerank-compare").addEventListener("click", runRerankComparison);
loadRerankCompareSnapshot();

// ================================================== CONSISTENCY ============
//
// Runs eval/evaluate.py's evaluate_consistency() — the same question
// through answer() 5 times at temperature=0, to measure how repeatable
// the answers actually are.

function renderConsistencyResult(container, data) {
  container.innerHTML = "";
  const p1 = document.createElement("p");
  p1.className = "small";
  p1.textContent =
    `${data.n_unique_answers} unique answer${data.n_unique_answers === 1 ? "" : "s"} out of ` +
    `${data.n_runs} runs · exact-match rate: ${Math.round(data.exact_match_rate * 100)}% · ` +
    `cited sources agree: ${data.sources_agree ? "yes" : "no"}`;
  const p2 = document.createElement("p");
  p2.className = "muted small";
  p2.textContent = `Average word-overlap (Jaccard) between every pair of answers: ${data.avg_jaccard_similarity.toFixed(2)}`;
  container.appendChild(p1);
  container.appendChild(p2);
}

async function loadConsistencySnapshot() {
  try {
    const res = await fetch("/api/eval/snapshot");
    if (!res.ok) return;
    const snap = await res.json();
    if (!snap.consistency) {
      $("#consistency-source-label").textContent = "No snapshot committed yet — click \"Run comparison live\" below.";
      return;
    }
    $("#consistency-source-label").textContent =
      `Snapshot computed ${fmtSnapshotDate(snap.generated_at)} against this exact index — committed to the repo, not live.`;
    renderConsistencyResult($("#consistency-result"), snap.consistency);
  } catch (err) {
    $("#consistency-source-label").textContent = "Could not load the consistency snapshot.";
  }
}

function runConsistency() {
  const btn = $("#btn-consistency");
  btn.disabled = true;
  $("#consistency-source-label").textContent = "Running live against your own key…";
  $("#consistency-progress").textContent = "Asking the same question 5 times…";
  $("#consistency-result").innerHTML = "";
  $("#log-consistency").innerHTML = "";

  const es = new FetchEventSource("/api/eval/consistency/stream", { headers: authHeaders() });
  const close = () => {
    es.close();
    btn.disabled = false;
  };

  es.addEventListener("consistency_run", (e) => {
    const payload = JSON.parse(e.data);
    appendLog("log-consistency", "consistency_run", payload);
    $("#consistency-progress").textContent = `Run ${payload.run + 1} of 5…`;
  });

  es.addEventListener("consistency_done", (e) => {
    const payload = JSON.parse(e.data);
    appendLog("log-consistency", "consistency_done", payload);
    renderConsistencyResult($("#consistency-result"), payload);
  });

  es.addEventListener("pipeline_done", (e) => {
    appendLog("log-consistency", "pipeline_done", JSON.parse(e.data));
    $("#consistency-progress").textContent = "Done.";
    $("#consistency-source-label").textContent = "Live run — just now, against this exact index, with your key.";
    close();
  });

  es.addEventListener("pipeline_error", (e) => {
    const payload = JSON.parse(e.data);
    appendLog("log-consistency", "pipeline_error", payload, true);
    $("#consistency-progress").textContent = `Error: ${payload.message}`;
    $("#consistency-source-label").textContent = "Live run failed — showing whatever loaded before the error.";
    close();
  });

  es.onerror = () => close();
}

$("#btn-consistency").addEventListener("click", runConsistency);
loadConsistencySnapshot();

// ----------------------------------------------------------- latency ------
//
// Real per-stage timings from the SSE events of the question you actually
// just asked (not simulated), plus an honestly-labeled running distribution
// across every question asked this session — no percentiles pretending to
// be statistically meaningful from a handful of samples.

const sessionLatencies = []; // total ms per question asked this session (classic Ask only)
let currentTiming = null;

// Per-technique latency, covering every new technique added beyond classic
// Ask — same "real numbers from this session, not simulated" idiom as
// sessionLatencies above, just broken out by which pipeline produced them.
const techniqueLatencies = {};
const TECHNIQUE_LABELS = {
  classic: "Classic RAG",
  classic_rerank: "Classic RAG + reranking",
  prompting_variants: "Prompting Lab: zero/few-shot/CoT",
  prompting_temperature: "Prompting Lab: temperature",
  prompting_structured: "Prompting Lab: structured output",
  classifier_compare: "Classifier vs. RAG comparison",
  agentic_compare: "Agent Frameworks: hand-rolled vs. LangGraph",
};

function recordTechniqueLatency(name, ms) {
  (techniqueLatencies[name] = techniqueLatencies[name] || []).push(ms);
  renderTechniqueLatencies();
}

function renderTechniqueLatencies() {
  const container = $("#technique-latency");
  if (!container) return;
  const names = Object.keys(techniqueLatencies).filter((n) => techniqueLatencies[n].length > 0);
  if (names.length === 0) {
    container.innerHTML = '<p class="muted small">No technique run yet this session.</p>';
    return;
  }
  const tbody = createResultsTable(container, ["Technique", "Runs", "P50", "Min", "Max"]);
  names.forEach((name) => {
    const sorted = [...techniqueLatencies[name]].sort((a, b) => a - b);
    appendResultRow(tbody, [
      TECHNIQUE_LABELS[name] || name,
      String(sorted.length),
      `${percentile(sorted, 0.5)}ms`,
      `${sorted[0]}ms`,
      `${sorted[sorted.length - 1]}ms`,
    ]);
  });
}

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

// ---------------------------------------------------------- landing CTA ---

function showAppShell() {
  $("#landing").hidden = true;
  $("#app-shell").hidden = false;
}

// ------------------------------------------------------- hash deep links --
//
// Lets an external link (e.g. a card on the home page) jump straight into
// a specific tab/mode instead of landing on the generic "Build the Index"
// tab — e.g. /RAG/#prompting:temperature or /RAG/#training:classifier.
// Falls back to the normal first-tab landing when there's no hash, or it
// doesn't match a known tab.

const VALID_HASH_TABS = ["init", "ask", "prompting", "training", "explore", "metrics", "langgraph"];

function parseHashTarget() {
  const hash = location.hash.replace(/^#/, "");
  if (!hash) return null;
  const [tab, mode] = hash.split(":");
  if (!VALID_HASH_TABS.includes(tab)) return null;
  return { tab, mode: mode || null };
}

function applyHashTarget(target) {
  switchToTab(target.tab);
  if (!target.mode) return;
  if (target.tab === "ask") setAskMode(target.mode);
  if (target.tab === "prompting") setPromptMode(target.mode);
  if (target.tab === "training") setTrainingMode(target.mode);
}

async function startDemoSession() {
  const buttons = $$("#btn-start-demo, [data-start-demo-link]");
  const statusEl = $("#landing-start-status");
  buttons.forEach((b) => (b.disabled = true));
  statusEl.hidden = false;
  statusEl.textContent = "Setting up your demo instance…";

  const sessionId = crypto.randomUUID();
  try {
    const res = await fetch("/api/session/start", {
      method: "POST",
      headers: { "X-Session-Id": sessionId },
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      statusEl.textContent = err.detail || "Could not start the demo. Try again.";
      buttons.forEach((b) => (b.disabled = false));
      return;
    }
    setSessionId(sessionId);
    showAppShell();
    // Start at step 1 by default, same as a fresh page load — the index is
    // already seeded, so there's nothing to build first, but the tour
    // still starts from the top — unless a hash deep-link asked for a
    // specific tab (e.g. a card on the home page).
    const target = parseHashTarget();
    if (target) applyHashTarget(target);
    else switchToTab("init");
    checkStatus();
  } catch (err) {
    statusEl.textContent = "Could not reach the server. Try again.";
    buttons.forEach((b) => (b.disabled = false));
  }
}

$$("#btn-start-demo, [data-start-demo-link]").forEach((btn) => {
  btn.addEventListener("click", startDemoSession);
});

// -------------------------------------------------------------- init ------

// A tab reload within the same browser tab keeps its sessionStorage, so a
// visitor who already started a demo goes straight back into it instead of
// seeing the landing screen again.
if (getSessionId()) {
  showAppShell();
  const target = parseHashTarget();
  if (target) applyHashTarget(target);
} else if (parseHashTarget()) {
  // A deep link from outside (e.g. the home page) with no session yet:
  // skip the extra landing-screen click and start the demo immediately.
  startDemoSession();
}

checkStatus();
