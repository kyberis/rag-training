"""
Capa de retrieval: dado un texto de consulta, devuelve los top-K chunks
más relevantes del índice ya construido (ver ingest.py). Esta es la pieza
que se mide con Recall@K (punto 1 del framework).
"""
from __future__ import annotations

import time

from . import config
from .embeddings import embed_query
from .events import EventCallback, emit
from .vector_store import SimpleVectorStore

_store: SimpleVectorStore | None = None


def _get_store() -> SimpleVectorStore:
    global _store
    if _store is None:
        if not config.INDEX_VECTORS_PATH.exists():
            raise RuntimeError(
                "No hay índice construido todavía. Corré primero: python -m src.ingest"
            )
        _store = SimpleVectorStore.load(config.INDEX_VECTORS_PATH, config.INDEX_META_PATH)
    return _store


def reset_store() -> None:
    """Invalida el índice cacheado en memoria.

    Necesario cuando el proceso vive más que una sola consulta (por ejemplo
    el server web): si se reconstruye el índice en disco, hay que forzar a
    que la próxima llamada a retrieve() lo vuelva a cargar en vez de seguir
    usando el store viejo cacheado en _store.
    """
    global _store
    _store = None


def retrieve(query: str, top_k: int | None = None, on_event: EventCallback | None = None) -> list[dict]:
    top_k = top_k or config.TOP_K
    store = _get_store()

    emit(on_event, "embedding_query_start", query=query)
    t0 = time.time()
    query_vector = embed_query(query)
    emit(
        on_event,
        "embedding_query_done",
        dim=len(query_vector),
        elapsed_ms=int((time.time() - t0) * 1000),
        preview=query_vector[:32],
    )

    emit(on_event, "search_start", top_k=top_k)
    all_results = store.search(query_vector, top_k=len(store.metadata))
    top_results = all_results[:top_k]
    emit(
        on_event,
        "search_done",
        all_scores=[
            {
                "source": r["source"],
                "chunk_index": r["chunk_index"],
                "score": r["score"],
                "text_preview": r["text"][:160],
            }
            for r in all_results
        ],
        top_k_sources=[r["source"] for r in top_results],
    )
    return top_results
