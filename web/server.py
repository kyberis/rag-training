"""
Web server for the UI that shows the RAG pipeline in real time.

This module is the only part of the project that knows about
FastAPI/SSE — all it does is call the functions in src/ (which already
emit progress events via the on_event parameter) and translate those
events into Server-Sent Events for the static frontend to consume.

Run with:
    python -m web.server
    (or) uvicorn web.server:app --reload

Then open http://127.0.0.1:8000 in your browser.
"""
from __future__ import annotations

import inspect
import json
import queue
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from eval.evaluate import (
    evaluate_classifier_vs_rag,
    evaluate_consistency,
    evaluate_faithfulness,
    evaluate_recall_at_k,
    evaluate_rerank_comparison,
    evaluate_retrieval_comparison,
    load_golden_dataset,
    load_rerank_hard_examples,
)
from src import (
    agentic_rag,
    chunking,
    classifier,
    config,
    embeddings,
    finetune_illustration,
    langgraph_agent,
    prompting_lab,
    rag,
    reranker,
    tokenizer_demo,
    vector_store,
)
from src.agentic_rag import answer_agentic
from src.ingest import build_index, load_documents
from src.langgraph_agent import answer_agentic_langgraph
from src.rag import answer
from src.retriever import get_default_store, get_store, set_store, similarity_from_indexed_chunk
from src.session_store import SESSION_TTL_SECONDS, get_session_store

app = FastAPI(title="RAG demo — pipeline en vivo")

STATIC_DIR = Path(__file__).resolve().parent / "static"

# Whitelist of functions that can be shown in the Explore tab's code
# viewer. An arbitrary path sent by the client is never evaluated — only
# one of these fixed keys can be requested, and the real source is read
# with inspect.getsource() to guarantee what's shown is exactly what ran.
CODE_REGISTRY: dict[str, list] = {
    "chunking": [chunking.chunk_spans, chunking.chunk_text],
    "embeddings": [embeddings.embed_texts, embeddings.embed_query],
    "vector_store_add": [vector_store.SimpleVectorStore.add],
    "vector_store_search": [vector_store.SimpleVectorStore.search],
    "prompt": [rag._build_prompt],
    "agentic_loop": [agentic_rag.answer_agentic],
    "langgraph_agent": [langgraph_agent.answer_agentic_langgraph, langgraph_agent._build_graph],
    "tokenizer": [tokenizer_demo.tokenize_text],
    "prompting_variants": [prompting_lab.run_prompt_variants, prompting_lab._variant_prompt],
    "structured_output": [prompting_lab.run_structured_output, prompting_lab._validate_structured],
    "reranker": [reranker.rerank_chunks],
    "finetune_illustration": [finetune_illustration.build_finetune_examples],
    "classifier": [classifier.fit_nearest_centroid, classifier.predict_nearest_centroid],
}


def _load_index_meta() -> list[dict]:
    if not config.INDEX_META_PATH.exists():
        return []
    with open(config.INDEX_META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def sse_event(name: str, payload: dict) -> str:
    return f"event: {name}\ndata: {json.dumps(payload)}\n\n"


def _immediate_error(message: str) -> StreamingResponse:
    return StreamingResponse(
        iter([sse_event("pipeline_error", {"message": message})]),
        media_type="text/event-stream",
    )


NO_KEY_MESSAGE = (
    "No hay ninguna OpenAI API key disponible. Si corrés esto localmente, "
    "copiá .env.example a .env y completá tu clave (ver README.md). Si es la "
    "demo pública, pegá tu propia clave en el campo de arriba — se usa solo "
    "para este pedido y nunca se guarda en el servidor."
)


def _resolve_api_key(x_openai_key: str | None) -> str | None:
    """BYOK key from the browser, if the visitor supplied one — never logged,
    never stored, only forwarded to the OpenAI SDK call it's needed for.
    Falls back to the server's own key (config.OPENAI_API_KEY) when set,
    which is what keeps local dev / self-hosting frictionless: no key field
    shows up in the UI at all when the server already has one (see
    /api/status's has_api_key). On the public deployment this fallback is
    real (see README, section "Demo pública en Vercel") — that's exactly why
    every caller of this function also calls _check_rate_limit() first when
    no BYOK key was supplied, so the server's own key can't be drained by
    anonymous traffic.
    """
    return x_openai_key or config.OPENAI_API_KEY


# --- Rate limiting: only applies when a request falls back to the server's
# own key (config.OPENAI_API_KEY). A visitor who brings their own key spends
# their own money, so they're never limited here.
#
# This is in-memory, per-process state — best-effort, not a hard guarantee.
# Vercel's serverless instances don't share memory, and this resets on every
# cold start, so a determined abuser spread across many instances could
# still exceed these numbers. It bounds the *typical* worst case cheaply,
# without adding an external database/Redis dependency for a small
# educational demo. The real backstop is a hard spending limit set directly
# on the OpenAI account (platform.openai.com/settings/organization/limits).
_rate_limit_lock = threading.Lock()
_ip_hits: dict[str, list[float]] = {}
_eval_ip_hits: dict[str, list[float]] = {}
_session_start_ip_hits: dict[str, list[float]] = {}
_global_units: list[tuple[float, int]] = []

_IP_DAILY_LIMIT = 20           # free ask/ingest requests per IP per day
_EVAL_IP_DAILY_LIMIT = 1       # free live evaluation runs per IP per day
_SESSION_START_IP_HOURLY_LIMIT = 20  # generous — costs no OpenAI money, just bounds Redis/memory session storage
_GLOBAL_DAILY_UNIT_BUDGET = 300  # shared "OpenAI call" budget per day, all visitors combined
_ASK_UNITS = 2                  # 1 embedding call + 1 chat completion call
_AGENTIC_ASK_UNITS = 9          # worst case: up to 4 chat completions + up to 3+ embedding calls (bounded ReAct loop)
_AGENTIC_COMPARE_UNITS = _AGENTIC_ASK_UNITS * 2  # hand-rolled + LangGraph runs, same worst case each
_INGEST_UNITS = 1               # chunks are embedded in a single batch call
_EVAL_UNITS = 30                # 10 embeddings + 10 answers + 10 judge calls
_COMPARE_UNITS = 10             # 10 embeddings only — keyword search makes no OpenAI calls
_PROMPT_VARIANTS_UNITS = 4      # 1 embedding + 3 chat completions (zero-shot / few-shot / CoT)
_PROMPT_TEMPERATURE_UNITS = 1 + len(config.TEMPERATURE_PLAYGROUND_VALUES) * config.TEMPERATURE_SAMPLES  # 1 embedding + n_samples chat calls per temperature
_PROMPT_STRUCTURED_UNITS = 3    # 1 embedding + 1 structured call + 1 free-text call
_RERANK_UNITS = 1               # 1 extra batched chat call on top of _ASK_UNITS
_RERANK_COMPARE_UNITS = 16      # 8 embeddings + 8 batched rerank calls (rerank_hard_examples.json)
_CLASSIFIER_COMPARE_UNITS = 10  # 10 embeddings only — classifier prediction makes no OpenAI call
_CONSISTENCY_RUNS = 5
_CONSISTENCY_UNITS = _CONSISTENCY_RUNS * _ASK_UNITS  # 5 runs x (1 embedding + 1 chat call) each


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _check_rate_limit(request: Request, units: int, is_eval: bool) -> str | None:
    """Returns an error message if this request should be blocked, else None.
    Only call this when there's no BYOK key — see _resolve_api_key above.
    """
    now = time.time()
    ip = _client_ip(request)
    with _rate_limit_lock:
        if is_eval:
            hits = _eval_ip_hits.setdefault(ip, [])
            hits[:] = [t for t in hits if now - t < 86400]
            if len(hits) >= _EVAL_IP_DAILY_LIMIT:
                return (
                    "Ya corriste la evaluación en vivo gratis hoy desde tu IP — los resultados "
                    "de la última corrida real ya están arriba (snapshot committeado). Pegá tu "
                    "propia OpenAI API key arriba para volver a correrla ahora mismo."
                )
        else:
            hits = _ip_hits.setdefault(ip, [])
            hits[:] = [t for t in hits if now - t < 86400]
            if len(hits) >= _IP_DAILY_LIMIT:
                return (
                    f"Ya usaste tus {_IP_DAILY_LIMIT} acciones gratis de hoy en esta demo "
                    "pública. Pegá tu propia OpenAI API key arriba para seguir sin esperar, o "
                    "volvé mañana."
                )

        _global_units[:] = [(t, u) for t, u in _global_units if now - t < 86400]
        used = sum(u for _, u in _global_units)
        if used + units > _GLOBAL_DAILY_UNIT_BUDGET:
            return (
                "Esta demo pública ya usó su presupuesto gratis compartido de hoy (entre todos "
                "los visitantes). Pegá tu propia OpenAI API key arriba para seguir probando, o "
                "volvé después de medianoche UTC."
            )

        hits.append(now)
        _global_units.append((now, units))
    return None


def _check_session_start_limit(request: Request) -> str | None:
    """Separate from _check_rate_limit above: starting a demo session costs
    no OpenAI money (it's a copy of an already-computed store), so it's
    outside the shared unit budget — but it does write to session storage
    (Redis or in-memory), so an unlimited endpoint would let anyone script
    thousands of sessions and inflate that storage for free.
    """
    now = time.time()
    ip = _client_ip(request)
    with _rate_limit_lock:
        hits = _session_start_ip_hits.setdefault(ip, [])
        hits[:] = [t for t in hits if now - t < 3600]
        if len(hits) >= _SESSION_START_IP_HOURLY_LIMIT:
            return "Demasiadas sesiones de demo iniciadas desde tu IP en la última hora. Probá de nuevo más tarde."
        hits.append(now)
    return None


def _has_usable_index(session_id: str | None) -> bool:
    """Whether an /api/ask/stream, /api/eval/stream or /api/kb/similarity
    call can proceed: either this session already has its own store, or the
    shared global index exists (in which case _get_store(session_id) will
    lazily seed the session from it on first use).
    """
    if session_id:
        if get_session_store().get(session_id) is not None:
            return True
    return config.INDEX_VECTORS_PATH.exists()


def run_pipeline_as_sse(target_fn: Callable[[Callable[[str, dict], None]], dict]) -> StreamingResponse:
    """Runs target_fn(on_event) in a separate thread and streams every event
    it emits as SSE, always ending with pipeline_done or pipeline_error.

    target_fn makes blocking calls (to the OpenAI API), which is why it
    runs in its own thread instead of directly in the async handler — that
    way it doesn't block uvicorn's event loop for the duration of the
    ingestion or the answer generation.
    """
    q: queue.Queue = queue.Queue()
    sentinel = object()

    def on_event(name: str, payload: dict) -> None:
        q.put((name, payload))

    def worker() -> None:
        try:
            result = target_fn(on_event)
            q.put(("pipeline_done", result))
        except Exception as e:
            q.put(("pipeline_error", {"message": str(e)}))
        finally:
            q.put(sentinel)

    threading.Thread(target=worker, daemon=True).start()

    def generator():
        while True:
            item = q.get()
            if item is sentinel:
                break
            name, payload = item
            yield sse_event(name, payload)

    return StreamingResponse(generator(), media_type="text/event-stream")


@app.get("/api/status")
def status(x_session_id: str | None = Header(default=None, alias="X-Session-Id")):
    has_api_key = bool(config.OPENAI_API_KEY)
    if x_session_id:
        session_store = get_session_store().get(x_session_id)
        if session_store is not None:
            return {
                "index_exists": True,
                "has_api_key": has_api_key,
                "n_vectors": int(session_store.vectors.shape[0]),
                "dim": int(session_store.vectors.shape[1]),
            }
        return {"index_exists": False, "has_api_key": has_api_key, "n_vectors": None, "dim": None}

    index_exists = config.INDEX_VECTORS_PATH.exists()
    n_vectors = dim = None
    if index_exists:
        vectors = np.load(config.INDEX_VECTORS_PATH)
        n_vectors, dim = int(vectors.shape[0]), int(vectors.shape[1])
    return {
        "index_exists": index_exists,
        "has_api_key": has_api_key,
        "n_vectors": n_vectors,
        "dim": dim,
    }


@app.get("/api/documents")
def documents():
    docs = load_documents()
    return [{"source": d["source"], "n_words": len(d["text"].split())} for d in docs]


@app.get("/api/kb/documents/{source}")
def kb_document(source: str):
    """Full text of a document, for the Explore tab.

    `source` is only ever used as a dict key built from the real files on
    disk — it's never concatenated into a path, so there's no way to
    request a file outside of data/docplanner_kb/.
    """
    docs_by_source = {d["source"]: d for d in load_documents()}
    doc = docs_by_source.get(source)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Documento desconocido: {source}")
    return {"source": doc["source"], "text": doc["text"], "n_words": len(doc["text"].split())}


@app.get("/api/kb/chunks")
def kb_chunks(source: str | None = None):
    """Chunks for one or all documents, with their word range and how much
    overlap they have with the previous chunk — recomputed live with
    chunking.chunk_spans(), available even before the index is built.
    """
    docs = load_documents()
    if source is not None:
        docs = [d for d in docs if d["source"] == source]
        if not docs:
            raise HTTPException(status_code=404, detail=f"Documento desconocido: {source}")

    indexed_keys = {(m["source"], m["chunk_index"]) for m in _load_index_meta()}

    result = []
    for doc in docs:
        words = doc["text"].split()
        spans = chunking.chunk_spans(doc["text"])
        for i, (start, end) in enumerate(spans):
            overlap_words = max(0, spans[i - 1][1] - start) if i > 0 else 0
            result.append({
                "source": doc["source"],
                "chunk_index": i,
                "text": " ".join(words[start:end]),
                "start_word": start,
                "end_word": end,
                "n_words": end - start,
                "overlap_words": overlap_words,
                "has_vector": (doc["source"], i) in indexed_keys,
            })
    return result


@app.get("/api/tokenize")
def tokenize(text: str):
    """Splits arbitrary text into real gpt-4o-mini tokens (o200k_base),
    no OpenAI call needed — tiktoken runs entirely locally. Same free
    tier as the other /api/kb/* read-only endpoints.
    """
    return tokenizer_demo.tokenize_text(text)


@app.get("/api/training/finetune-example")
def training_finetune_example():
    """Illustrative fine-tuning examples built from this project's own
    real data — see src/finetune_illustration.py. No training actually
    happens, no OpenAI call is made: this is purely "here's what a
    training example looks like," free tier same as /api/kb/*.
    """
    return finetune_illustration.build_finetune_examples()


@app.get("/api/training/classifier/train")
def training_classifier_train(
    x_session_id: str | None = Header(default=None, alias="X-Session-Id"),
):
    """Fits the nearest-centroid classifier (src/classifier.py) on the
    already-indexed chunk vectors — a real, if tiny, training step, but
    zero OpenAI calls (it reuses vectors already computed at ingestion
    time), so this is always free, no key needed.
    """
    if not _has_usable_index(x_session_id):
        raise HTTPException(status_code=409, detail="No hay índice construido todavía.")
    store = get_store(x_session_id)
    basis = classifier.fit_nearest_centroid(store)
    return {
        "labels": basis["labels"],
        "n_examples_per_label": basis["n_examples_per_label"],
        "centroid_separation": classifier.centroid_separation(basis),
    }


@app.get("/api/kb/index")
def kb_index():
    """Summary of what's literally persisted on disk: the array in
    index/vectors.npy and the list in index/meta.json. This is the
    project's "database" — there's nothing else behind it.
    """
    if not config.INDEX_VECTORS_PATH.exists():
        raise HTTPException(status_code=404, detail="Todavía no se construyó el índice.")

    vectors = np.load(config.INDEX_VECTORS_PATH)
    meta = _load_index_meta()
    return {
        "vectors_path": str(config.INDEX_VECTORS_PATH),
        "meta_path": str(config.INDEX_META_PATH),
        "n_vectors": int(vectors.shape[0]),
        "dim": int(vectors.shape[1]),
        "dtype": str(vectors.dtype),
        "chunks": [
            {"source": m["source"], "chunk_index": m["chunk_index"], "n_words": len(m["text"].split())}
            for m in meta
        ],
    }


@app.get("/api/kb/vector")
def kb_vector(source: str, chunk_index: int):
    """The real stored vector for one specific chunk, read directly from
    index/vectors.npy — not an approximation or a recomputation.
    """
    if not config.INDEX_VECTORS_PATH.exists():
        raise HTTPException(status_code=404, detail="Todavía no se construyó el índice.")

    meta = _load_index_meta()
    row = next(
        (i for i, m in enumerate(meta) if m["source"] == source and m["chunk_index"] == chunk_index),
        None,
    )
    if row is None:
        raise HTTPException(status_code=404, detail="Ese chunk no está en el índice construido.")

    vectors = np.load(config.INDEX_VECTORS_PATH)
    vec = vectors[row]
    return {
        "source": source,
        "chunk_index": chunk_index,
        "dim": int(vec.shape[0]),
        "norm": float(np.linalg.norm(vec)),
        "vector": vec.tolist(),
    }


@app.get("/api/kb/similarity")
def kb_similarity(
    source: str,
    chunk_index: int,
    x_session_id: str | None = Header(default=None, alias="X-Session-Id"),
):
    """Cosine similarity from one already-indexed chunk to every other chunk
    in the index, in the same shape as search_done's all_scores. Lets the
    vector map show real distances from any chunk the user clicks on, using
    that chunk's own stored vector — no extra embedding call needed.
    """
    if not _has_usable_index(x_session_id):
        raise HTTPException(status_code=404, detail="Todavía no se construyó el índice.")
    try:
        return similarity_from_indexed_chunk(source, chunk_index, session_id=x_session_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/code")
def get_code(key: str):
    """Real source code for one of the pipeline's functions, via
    inspect.getsource(). `key` can only be one of CODE_REGISTRY's fixed
    keys — an arbitrary function name is never evaluated.
    """
    funcs = CODE_REGISTRY.get(key)
    if not funcs:
        raise HTTPException(status_code=404, detail=f"Clave de código desconocida: {key}")
    return {
        "key": key,
        "snippets": [
            {"name": f.__qualname__, "source": inspect.getsource(f)}
            for f in funcs
        ],
    }


@app.get("/api/eval/golden")
def eval_golden():
    return load_golden_dataset()


@app.get("/api/eval/snapshot")
def eval_snapshot():
    """Real Recall@K + Faithfulness results, computed once by the project
    maintainer and committed to the repo — so anyone opening the public
    demo sees real numbers (not made up) without spending a single OpenAI
    call. 'Run evaluation' below repeats the same thing live, for whoever
    brings their own API key.
    """
    if not config.RESULTS_SNAPSHOT_PATH.exists():
        raise HTTPException(status_code=404, detail="Todavía no se generó el snapshot de evaluación.")
    with open(config.RESULTS_SNAPSHOT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/eval/stream")
def eval_stream(
    request: Request,
    x_openai_key: str | None = Header(default=None, alias="X-OpenAI-Key"),
    x_session_id: str | None = Header(default=None, alias="X-Session-Id"),
):
    """Runs Recall@K + Faithfulness (LLM-as-judge) against the golden dataset.

    ~30 real calls to the OpenAI API (10 question embeddings + 10 answers +
    10 judgments), takes close to a minute — the frontend warns about the
    cost/time before this button. The already-computed results are in
    /api/eval/snapshot if you don't want to spend your own key.

    If there's an active demo session (X-Session-Id), evaluates against
    that session's own index instead of the shared one — so the numbers
    reflect what that visitor actually built.
    """
    if not _has_usable_index(x_session_id):
        return _immediate_error(
            "No hay índice construido todavía. Andá a la pestaña 'Build the Index' y construilo primero."
        )
    if not x_openai_key:
        limit_error = _check_rate_limit(request, units=_EVAL_UNITS, is_eval=True)
        if limit_error:
            return _immediate_error(limit_error)
    api_key = _resolve_api_key(x_openai_key)
    if not api_key:
        return _immediate_error(NO_KEY_MESSAGE)

    def target(on_event):
        golden = load_golden_dataset()
        recall = evaluate_recall_at_k(golden, on_event=on_event, api_key=api_key, session_id=x_session_id)
        faithfulness = evaluate_faithfulness(golden, on_event=on_event, api_key=api_key, session_id=x_session_id)
        return {"recall": recall, "faithfulness": faithfulness}

    return run_pipeline_as_sse(target)


@app.get("/api/eval/compare/stream")
def eval_compare_stream(
    request: Request,
    x_openai_key: str | None = Header(default=None, alias="X-OpenAI-Key"),
    x_session_id: str | None = Header(default=None, alias="X-Session-Id"),
):
    """Runs Recall@K against the golden dataset twice: once through the real
    embeddings retriever, once through a naive keyword-overlap baseline
    (src/keyword_retriever.py) — same questions, same index, so the gap
    between them is a measured number. ~10 OpenAI calls (question
    embeddings only, the keyword side is free), shares the eval daily
    free-run budget with /api/eval/stream. Already-computed results are in
    /api/eval/snapshot if you don't want to spend your own key.
    """
    if not _has_usable_index(x_session_id):
        return _immediate_error(
            "No hay índice construido todavía. Andá a la pestaña 'Build the Index' y construilo primero."
        )
    if not x_openai_key:
        limit_error = _check_rate_limit(request, units=_COMPARE_UNITS, is_eval=True)
        if limit_error:
            return _immediate_error(limit_error)
    api_key = _resolve_api_key(x_openai_key)
    if not api_key:
        return _immediate_error(NO_KEY_MESSAGE)

    def target(on_event):
        golden = load_golden_dataset()
        return evaluate_retrieval_comparison(golden, on_event=on_event, api_key=api_key, session_id=x_session_id)

    return run_pipeline_as_sse(target)


@app.get("/api/eval/rerank/stream")
def eval_rerank_stream(
    request: Request,
    x_openai_key: str | None = Header(default=None, alias="X-OpenAI-Key"),
    x_session_id: str | None = Header(default=None, alias="X-Session-Id"),
):
    """Recall@1 with vs. without the LLM-based reranker (src/reranker.py),
    on a small set of questions curated because plain cosine similarity
    gets the top-ranked source wrong on all of them (see
    eval.evaluate.load_rerank_hard_examples) — the main 10-question golden
    dataset is already correct at top-1 for every question, so it can't
    show the reranker's effect at all. ~16 OpenAI calls (8 embeddings + 8
    batched rerank calls), shares the eval daily free-run budget with the
    other /api/eval/*/stream routes.
    """
    if not _has_usable_index(x_session_id):
        return _immediate_error(
            "No hay índice construido todavía. Andá a la pestaña 'Build the Index' y construilo primero."
        )
    if not x_openai_key:
        limit_error = _check_rate_limit(request, units=_RERANK_COMPARE_UNITS, is_eval=True)
        if limit_error:
            return _immediate_error(limit_error)
    api_key = _resolve_api_key(x_openai_key)
    if not api_key:
        return _immediate_error(NO_KEY_MESSAGE)

    def target(on_event):
        hard_examples = load_rerank_hard_examples()
        return evaluate_rerank_comparison(hard_examples, top_k=1, on_event=on_event, api_key=api_key, session_id=x_session_id)

    return run_pipeline_as_sse(target)


@app.get("/api/training/classifier/compare/stream")
def training_classifier_compare_stream(
    request: Request,
    x_openai_key: str | None = Header(default=None, alias="X-OpenAI-Key"),
    x_session_id: str | None = Header(default=None, alias="X-Session-Id"),
):
    """Nearest-centroid classifier (trained on indexed chunks) vs. RAG's
    top-1 retrieval, on the 10 golden-dataset questions — see
    eval.evaluate.evaluate_classifier_vs_rag. 10 OpenAI calls (question
    embeddings only; both predictions reuse the same embedding).
    """
    if not _has_usable_index(x_session_id):
        return _immediate_error(
            "No hay índice construido todavía. Andá a la pestaña 'Build the Index' y construilo primero."
        )
    if not x_openai_key:
        limit_error = _check_rate_limit(request, units=_CLASSIFIER_COMPARE_UNITS, is_eval=True)
        if limit_error:
            return _immediate_error(limit_error)
    api_key = _resolve_api_key(x_openai_key)
    if not api_key:
        return _immediate_error(NO_KEY_MESSAGE)

    def target(on_event):
        golden = load_golden_dataset()
        return evaluate_classifier_vs_rag(golden, on_event=on_event, api_key=api_key, session_id=x_session_id)

    return run_pipeline_as_sse(target)


@app.get("/api/eval/consistency/stream")
def eval_consistency_stream(
    request: Request,
    question: str | None = None,
    x_openai_key: str | None = Header(default=None, alias="X-OpenAI-Key"),
    x_session_id: str | None = Header(default=None, alias="X-Session-Id"),
):
    """Runs the same question through answer() 5 times (always
    temperature=0) and measures how consistent the results really are —
    see eval.evaluate.evaluate_consistency. Defaults to the first golden
    question, same one used for the committed snapshot's sample, so a
    live re-run is directly comparable to it.
    """
    if not _has_usable_index(x_session_id):
        return _immediate_error(
            "No hay índice construido todavía. Andá a la pestaña 'Build the Index' y construilo primero."
        )
    if not x_openai_key:
        limit_error = _check_rate_limit(request, units=_CONSISTENCY_UNITS, is_eval=True)
        if limit_error:
            return _immediate_error(limit_error)
    api_key = _resolve_api_key(x_openai_key)
    if not api_key:
        return _immediate_error(NO_KEY_MESSAGE)

    def target(on_event):
        q = question or load_golden_dataset()[0]["question"]
        return evaluate_consistency(q, n_runs=_CONSISTENCY_RUNS, on_event=on_event, api_key=api_key, session_id=x_session_id)

    return run_pipeline_as_sse(target)


@app.get("/api/ingest/stream")
def ingest_stream(
    request: Request,
    x_openai_key: str | None = Header(default=None, alias="X-OpenAI-Key"),
    x_session_id: str | None = Header(default=None, alias="X-Session-Id"),
):
    if not x_openai_key:
        limit_error = _check_rate_limit(request, units=_INGEST_UNITS, is_eval=False)
        if limit_error:
            return _immediate_error(limit_error)
    api_key = _resolve_api_key(x_openai_key)
    if not api_key:
        return _immediate_error(NO_KEY_MESSAGE)

    def target(on_event):
        # With an active session, the rebuild is never saved to the shared
        # index on disk (persist=False) — it stays isolated to that session.
        store = build_index(on_event=on_event, api_key=api_key, persist=(x_session_id is None))
        # Activates the freshly built store directly in memory (global) or
        # in the session (Redis/memory) — works the same whether the disk
        # is read-only (serverless deploy) or writable (local): it doesn't
        # depend on rereading it from disk.
        set_store(store, session_id=x_session_id)
        return {"n_vectors": int(store.vectors.shape[0]), "dim": int(store.vectors.shape[1])}

    return run_pipeline_as_sse(target)


@app.post("/api/session/start")
def session_start(
    request: Request,
    x_session_id: str | None = Header(default=None, alias="X-Session-Id"),
):
    """Seeds a new demo session with a copy of the shared index, so the
    landing screen's CTA delivers an "already built" experience without
    forcing a rebuild first. The session stays isolated from the global
    index and from any other session, and expires on its own after 24h
    (see session_store.py).
    """
    if not x_session_id:
        raise HTTPException(status_code=400, detail="Falta el header X-Session-Id.")
    limit_error = _check_session_start_limit(request)
    if limit_error:
        raise HTTPException(status_code=429, detail=limit_error)
    if not config.INDEX_VECTORS_PATH.exists():
        raise HTTPException(
            status_code=409,
            detail=(
                "No hay índice construido todavía en este servidor. Si corrés esto "
                "localmente, construilo primero: python -m src.ingest"
            ),
        )
    store = get_default_store()
    set_store(store, session_id=x_session_id)
    return {
        "n_vectors": int(store.vectors.shape[0]),
        "dim": int(store.vectors.shape[1]),
        "expires_at": int(time.time()) + SESSION_TTL_SECONDS,
    }


@app.get("/api/ask/stream")
def ask_stream(
    request: Request,
    question: str,
    top_k: int | None = None,
    rerank: bool = False,
    x_openai_key: str | None = Header(default=None, alias="X-OpenAI-Key"),
    x_session_id: str | None = Header(default=None, alias="X-Session-Id"),
):
    if not _has_usable_index(x_session_id):
        return _immediate_error(
            "No hay índice construido todavía. Andá a la pestaña 'Inicialización' y construilo primero."
        )
    if not x_openai_key:
        units = _ASK_UNITS + _RERANK_UNITS if rerank else _ASK_UNITS
        limit_error = _check_rate_limit(request, units=units, is_eval=False)
        if limit_error:
            return _immediate_error(limit_error)
    api_key = _resolve_api_key(x_openai_key)
    if not api_key:
        return _immediate_error(NO_KEY_MESSAGE)

    def target(on_event):
        return answer(question, top_k=top_k, on_event=on_event, api_key=api_key, session_id=x_session_id, rerank=rerank)

    return run_pipeline_as_sse(target)


@app.get("/api/ask/agentic/stream")
def ask_agentic_stream(
    request: Request,
    question: str,
    top_k: int | None = None,
    x_openai_key: str | None = Header(default=None, alias="X-OpenAI-Key"),
    x_session_id: str | None = Header(default=None, alias="X-Session-Id"),
):
    """Same question as /api/ask/stream, but with the model deciding via
    tool calling whether to call retrieve_context zero, one, or several
    times before answering (ReAct) — see src/agentic_rag.py. Same guards
    as classic mode, but with its own rate-limit weight: up to 4 loop
    turns, each with its own model call and (if it requests the tool) its
    own embedding call.
    """
    if not _has_usable_index(x_session_id):
        return _immediate_error(
            "No hay índice construido todavía. Andá a la pestaña 'Inicialización' y construilo primero."
        )
    if not x_openai_key:
        limit_error = _check_rate_limit(request, units=_AGENTIC_ASK_UNITS, is_eval=False)
        if limit_error:
            return _immediate_error(limit_error)
    api_key = _resolve_api_key(x_openai_key)
    if not api_key:
        return _immediate_error(NO_KEY_MESSAGE)

    def target(on_event):
        return answer_agentic(question, top_k=top_k, on_event=on_event, api_key=api_key, session_id=x_session_id)

    return run_pipeline_as_sse(target)


@app.get("/api/ask/agentic-compare/stream")
def ask_agentic_compare_stream(
    request: Request,
    question: str,
    top_k: int | None = None,
    x_openai_key: str | None = Header(default=None, alias="X-OpenAI-Key"),
    x_session_id: str | None = Header(default=None, alias="X-Session-Id"),
):
    """Runs the same question through both the hand-rolled ReAct agent
    (src/agentic_rag.py) and the LangGraph rebuild of it
    (src/langgraph_agent.py) — see the "Agent Frameworks" tab. Both runs
    happen inside this one request/rate-limit hit (not two separate
    fetches), same shape as the other combined comparison routes
    (/api/eval/compare/stream, /api/eval/rerank/stream): each event is
    tagged with which run it belongs to (`run: "handrolled"|"langgraph"`)
    so the frontend can route it to the right side of the UI.
    """
    if not _has_usable_index(x_session_id):
        return _immediate_error(
            "No hay índice construido todavía. Andá a la pestaña 'Build the Index' y construilo primero."
        )
    if not x_openai_key:
        limit_error = _check_rate_limit(request, units=_AGENTIC_COMPARE_UNITS, is_eval=False)
        if limit_error:
            return _immediate_error(limit_error)
    api_key = _resolve_api_key(x_openai_key)
    if not api_key:
        return _immediate_error(NO_KEY_MESSAGE)

    def target(on_event):
        def tag(run_name):
            return lambda name, payload: on_event(name, {**payload, "run": run_name})

        handrolled = answer_agentic(question, top_k=top_k, on_event=tag("handrolled"), api_key=api_key, session_id=x_session_id)
        langgraph_result = answer_agentic_langgraph(question, top_k=top_k, on_event=tag("langgraph"), api_key=api_key, session_id=x_session_id)
        return {"handrolled": handrolled, "langgraph": langgraph_result}

    return run_pipeline_as_sse(target)


@app.get("/api/prompting/variants/stream")
def prompting_variants_stream(
    request: Request,
    question: str,
    x_openai_key: str | None = Header(default=None, alias="X-OpenAI-Key"),
    x_session_id: str | None = Header(default=None, alias="X-Session-Id"),
):
    """Zero-shot vs. few-shot vs. Chain-of-Thought, same question, same
    retrieved context — see src/prompting_lab.py. No free/snapshot tier:
    this is a live interactive demo, same as /api/ask/stream.
    """
    if not _has_usable_index(x_session_id):
        return _immediate_error(
            "No hay índice construido todavía. Andá a la pestaña 'Build the Index' y construilo primero."
        )
    if not x_openai_key:
        limit_error = _check_rate_limit(request, units=_PROMPT_VARIANTS_UNITS, is_eval=False)
        if limit_error:
            return _immediate_error(limit_error)
    api_key = _resolve_api_key(x_openai_key)
    if not api_key:
        return _immediate_error(NO_KEY_MESSAGE)

    def target(on_event):
        return prompting_lab.run_prompt_variants(question, on_event=on_event, api_key=api_key, session_id=x_session_id)

    return run_pipeline_as_sse(target)


@app.get("/api/prompting/temperature/stream")
def prompting_temperature_stream(
    request: Request,
    question: str,
    x_openai_key: str | None = Header(default=None, alias="X-OpenAI-Key"),
    x_session_id: str | None = Header(default=None, alias="X-Session-Id"),
):
    """Same question/prompt, several temperature values — see
    src/prompting_lab.py's TEMPERATURE_PLAYGROUND_VALUES.
    """
    if not _has_usable_index(x_session_id):
        return _immediate_error(
            "No hay índice construido todavía. Andá a la pestaña 'Build the Index' y construilo primero."
        )
    if not x_openai_key:
        limit_error = _check_rate_limit(request, units=_PROMPT_TEMPERATURE_UNITS, is_eval=False)
        if limit_error:
            return _immediate_error(limit_error)
    api_key = _resolve_api_key(x_openai_key)
    if not api_key:
        return _immediate_error(NO_KEY_MESSAGE)

    def target(on_event):
        return prompting_lab.run_temperature_playground(question, on_event=on_event, api_key=api_key, session_id=x_session_id)

    return run_pipeline_as_sse(target)


@app.get("/api/prompting/structured")
def prompting_structured(
    request: Request,
    question: str,
    x_openai_key: str | None = Header(default=None, alias="X-OpenAI-Key"),
    x_session_id: str | None = Header(default=None, alias="X-Session-Id"),
):
    """Structured-output (json_schema mode) vs. free text on the same
    prompt — see src/prompting_lab.py. Plain JSON, not SSE: only 3 short
    calls total (1 embedding + 2 chat), nothing worth token-streaming.
    First non-SSE OpenAI-calling route in the project, so it uses
    HTTPException for errors instead of the SSE-only _immediate_error.
    """
    if not _has_usable_index(x_session_id):
        raise HTTPException(status_code=409, detail="No hay índice construido todavía.")
    if not x_openai_key:
        limit_error = _check_rate_limit(request, units=_PROMPT_STRUCTURED_UNITS, is_eval=False)
        if limit_error:
            raise HTTPException(status_code=429, detail=limit_error)
    api_key = _resolve_api_key(x_openai_key)
    if not api_key:
        raise HTTPException(status_code=400, detail=NO_KEY_MESSAGE)

    try:
        return prompting_lab.run_structured_output(question, api_key=api_key, session_id=x_session_id)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


# Static files last: Starlette resolves routes in registration order, so
# if the "/" mount came earlier it would swallow /api/*.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("web.server:app", host="127.0.0.1", port=8000, reload=True)
