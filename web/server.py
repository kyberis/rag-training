"""
Server web para la UI que muestra el pipeline RAG en tiempo real.

Este módulo es la única parte del proyecto que sabe de FastAPI/SSE — todo
lo que hace es llamar a las funciones de src/ (que ya emiten eventos de
progreso vía el parámetro on_event) y traducir esos eventos a
Server-Sent Events para que el frontend estático los consuma.

Correr con:
    python -m web.server
    (o) uvicorn web.server:app --reload

Después abrir http://127.0.0.1:8000 en el navegador.
"""
from __future__ import annotations

import inspect
import json
import queue
import threading
from pathlib import Path
from typing import Callable

import numpy as np
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from eval.evaluate import evaluate_faithfulness, evaluate_recall_at_k, load_golden_dataset
from src import chunking, config, embeddings, rag, vector_store
from src.ingest import build_index, load_documents
from src.rag import answer
from src.retriever import reset_store, similarity_from_indexed_chunk

app = FastAPI(title="RAG demo — pipeline en vivo")

STATIC_DIR = Path(__file__).resolve().parent / "static"

# Whitelist de funciones que se pueden mostrar en el visor de código de la
# pestaña Explore. Nunca se evalúa un path arbitrario mandado por el
# cliente — solo se permite pedir una de estas claves fijas, y se lee el
# código fuente real con inspect.getsource() para garantizar que lo que se
# muestra es exactamente lo que corrió.
CODE_REGISTRY: dict[str, list] = {
    "chunking": [chunking.chunk_spans, chunking.chunk_text],
    "embeddings": [embeddings.embed_texts, embeddings.embed_query],
    "vector_store_add": [vector_store.SimpleVectorStore.add],
    "vector_store_search": [vector_store.SimpleVectorStore.search],
    "prompt": [rag._build_prompt],
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
    /api/status's has_api_key).
    """
    return x_openai_key or config.OPENAI_API_KEY


def run_pipeline_as_sse(target_fn: Callable[[Callable[[str, dict], None]], dict]) -> StreamingResponse:
    """Corre target_fn(on_event) en un thread aparte y transmite cada evento
    que emita como SSE, terminando siempre con pipeline_done o pipeline_error.

    target_fn hace llamadas bloqueantes (a la API de OpenAI), por eso corre
    en un thread propio en vez de directamente en el handler async — así no
    bloquea el event loop de uvicorn mientras dura la ingestión o la
    generación de la respuesta.
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
def status():
    has_api_key = bool(config.OPENAI_API_KEY)
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
    """Texto completo de un documento, para la pestaña Explore.

    `source` solo se usa como clave de un diccionario armado a partir de
    los archivos reales en disco — nunca se concatena a un path, así que
    no hay forma de pedir un archivo fuera de data/docplanner_kb/.
    """
    docs_by_source = {d["source"]: d for d in load_documents()}
    doc = docs_by_source.get(source)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Documento desconocido: {source}")
    return {"source": doc["source"], "text": doc["text"], "n_words": len(doc["text"].split())}


@app.get("/api/kb/chunks")
def kb_chunks(source: str | None = None):
    """Chunks de uno o todos los documentos, con su rango de palabras y
    cuánto overlap tienen con el chunk anterior — recalculado en vivo con
    chunking.chunk_spans(), disponible incluso antes de construir el índice.
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


@app.get("/api/kb/index")
def kb_index():
    """Resumen de lo que hay literalmente persistido en disco: el array de
    index/vectors.npy y la lista de index/meta.json. Esto es "la base de
    datos" del proyecto — no hay nada más detrás.
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
    """El vector real guardado para un chunk puntual, leído directamente de
    index/vectors.npy — no una aproximación ni un recálculo.
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
def kb_similarity(source: str, chunk_index: int):
    """Cosine similarity from one already-indexed chunk to every other chunk
    in the index, in the same shape as search_done's all_scores. Lets the
    vector map show real distances from any chunk the user clicks on, using
    that chunk's own stored vector — no extra embedding call needed.
    """
    if not config.INDEX_VECTORS_PATH.exists():
        raise HTTPException(status_code=404, detail="Todavía no se construyó el índice.")
    try:
        return similarity_from_indexed_chunk(source, chunk_index)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/code")
def get_code(key: str):
    """Código fuente real de una de las funciones del pipeline, vía
    inspect.getsource(). `key` solo puede ser una de las claves fijas de
    CODE_REGISTRY — nunca se evalúa un nombre de función arbitrario.
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
    """Resultados reales de Recall@K + Faithfulness, calculados una vez por
    el mantenedor del proyecto y committeados al repo — así cualquiera que
    abra la demo pública ve números reales (no inventados) sin gastar un
    solo llamado a OpenAI. 'Run evaluation' abajo repite lo mismo en vivo,
    para quien traiga su propia API key.
    """
    if not config.RESULTS_SNAPSHOT_PATH.exists():
        raise HTTPException(status_code=404, detail="Todavía no se generó el snapshot de evaluación.")
    with open(config.RESULTS_SNAPSHOT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/eval/stream")
def eval_stream(x_openai_key: str | None = Header(default=None, alias="X-OpenAI-Key")):
    """Corre Recall@K + Faithfulness (LLM-as-judge) contra el golden dataset.

    ~30 llamadas reales a la API de OpenAI (10 embeddings de pregunta + 10
    respuestas + 10 juicios), toma cerca de un minuto — el frontend avisa
    el costo/tiempo antes de este botón. Los resultados ya calculados están
    en /api/eval/snapshot si no querés gastar tu propia clave.
    """
    if not config.INDEX_VECTORS_PATH.exists():
        return _immediate_error(
            "No hay índice construido todavía. Andá a la pestaña 'Build the Index' y construilo primero."
        )
    api_key = _resolve_api_key(x_openai_key)
    if not api_key:
        return _immediate_error(NO_KEY_MESSAGE)

    def target(on_event):
        golden = load_golden_dataset()
        recall = evaluate_recall_at_k(golden, on_event=on_event, api_key=api_key)
        faithfulness = evaluate_faithfulness(golden, on_event=on_event, api_key=api_key)
        return {"recall": recall, "faithfulness": faithfulness}

    return run_pipeline_as_sse(target)


@app.get("/api/ingest/stream")
def ingest_stream(x_openai_key: str | None = Header(default=None, alias="X-OpenAI-Key")):
    api_key = _resolve_api_key(x_openai_key)
    if not api_key:
        return _immediate_error(NO_KEY_MESSAGE)

    def target(on_event):
        store = build_index(on_event=on_event, api_key=api_key)
        # El server vive como proceso largo: si no invalidamos el store
        # cacheado, las próximas preguntas seguirían usando el índice viejo.
        reset_store()
        return {"n_vectors": int(store.vectors.shape[0]), "dim": int(store.vectors.shape[1])}

    return run_pipeline_as_sse(target)


@app.get("/api/ask/stream")
def ask_stream(
    question: str,
    top_k: int | None = None,
    x_openai_key: str | None = Header(default=None, alias="X-OpenAI-Key"),
):
    if not config.INDEX_VECTORS_PATH.exists():
        return _immediate_error(
            "No hay índice construido todavía. Andá a la pestaña 'Inicialización' y construilo primero."
        )
    api_key = _resolve_api_key(x_openai_key)
    if not api_key:
        return _immediate_error(NO_KEY_MESSAGE)

    def target(on_event):
        return answer(question, top_k=top_k, on_event=on_event, api_key=api_key)

    return run_pipeline_as_sse(target)


# Estáticos al final: Starlette resuelve rutas en el orden en que se
# registran, así que si el mount de "/" fuera antes se comería /api/*.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("web.server:app", host="127.0.0.1", port=8000, reload=True)
