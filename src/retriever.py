"""
Capa de retrieval: dado un texto de consulta, devuelve los top-K chunks
más relevantes del índice ya construido (ver ingest.py). Esta es la pieza
que se mide con Recall@K (punto 1 del framework).
"""
from __future__ import annotations

from . import config
from .embeddings import embed_query
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


def retrieve(query: str, top_k: int | None = None) -> list[dict]:
    top_k = top_k or config.TOP_K
    store = _get_store()
    query_vector = embed_query(query)
    return store.search(query_vector, top_k=top_k)
