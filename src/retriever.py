"""
Capa de retrieval: dado un texto de consulta, devuelve los top-K chunks
más relevantes del índice ya construido (ver ingest.py). Esta es la pieza
que se mide con Recall@K (punto 1 del framework).
"""
from __future__ import annotations

import time

import numpy as np

from . import config
from .embeddings import embed_query
from .events import EventCallback, emit
from .vector_store import SimpleVectorStore

_store: SimpleVectorStore | None = None

# Base PCA (mean + top-2 components) fit on the currently loaded store's
# vectors, used to project the 1536-dim vectors down to 2D for the "vector
# map" visualization. Cached per store so repeated questions land in the
# same 2D space instead of jittering around on every search.
_pca_basis: dict | None = None


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
    global _store, _pca_basis
    _store = None
    _pca_basis = None


def _get_pca_basis(store: SimpleVectorStore) -> dict:
    """Fits (and caches) a 2-component PCA basis on the store's vectors.

    Plain numpy SVD, no sklearn dependency: centering the vectors and taking
    the top-2 right singular vectors gives the same 2 principal components,
    which is all the "vector map" visualization needs.
    """
    global _pca_basis
    if _pca_basis is None:
        vectors = store.vectors.astype(np.float64)
        mean = vectors.mean(axis=0)
        _, _, vt = np.linalg.svd(vectors - mean, full_matrices=False)
        _pca_basis = {"mean": mean, "components": vt[:2]}
    return _pca_basis


def _project(store: SimpleVectorStore, vector) -> tuple[float, float]:
    basis = _get_pca_basis(store)
    xy = (np.array(vector, dtype=np.float64) - basis["mean"]) @ basis["components"].T
    return float(xy[0]), float(xy[1])


def _xy_by_key(store: SimpleVectorStore) -> dict[tuple[str, int], tuple[float, float]]:
    basis = _get_pca_basis(store)
    projected = (store.vectors.astype(np.float64) - basis["mean"]) @ basis["components"].T
    return {
        (m["source"], m["chunk_index"]): (float(projected[i, 0]), float(projected[i, 1]))
        for i, m in enumerate(store.metadata)
    }


def _annotate_with_projection(results: list[dict], xy_by_key: dict) -> list[dict]:
    """Attaches each result's 2D PCA position — the shape the vector-map
    frontend needs, whether `results` came from a fresh query embedding or
    from an already-indexed chunk's own stored vector.
    """
    return [
        {
            "source": r["source"],
            "chunk_index": r["chunk_index"],
            "score": r["score"],
            "text_preview": r["text"][:160],
            "x": xy_by_key[(r["source"], r["chunk_index"])][0],
            "y": xy_by_key[(r["source"], r["chunk_index"])][1],
        }
        for r in results
    ]


def retrieve(
    query: str,
    top_k: int | None = None,
    on_event: EventCallback | None = None,
    api_key: str | None = None,
) -> list[dict]:
    top_k = top_k or config.TOP_K
    store = _get_store()

    emit(on_event, "embedding_query_start", query=query)
    t0 = time.time()
    query_vector = embed_query(query, api_key=api_key)
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
    query_x, query_y = _project(store, query_vector)

    emit(
        on_event,
        "search_done",
        all_scores=_annotate_with_projection(all_results, _xy_by_key(store)),
        top_k_sources=[r["source"] for r in top_results],
        query_projection={"x": query_x, "y": query_y},
    )
    return top_results


def similarity_from_indexed_chunk(source: str, chunk_index: int) -> dict:
    """Same all_scores shape as retrieve()'s search_done event, but using an
    already-indexed chunk's own stored vector as the reference point instead
    of embedding a new query — lets the vector map show real distances from
    any chunk the user clicks on, with no extra OpenAI call.
    """
    store = _get_store()
    row = next(
        (i for i, m in enumerate(store.metadata) if m["source"] == source and m["chunk_index"] == chunk_index),
        None,
    )
    if row is None:
        raise ValueError(f"Ese chunk no está en el índice construido: {source}#{chunk_index}")

    reference_vector = store.vectors[row].tolist()
    all_results = store.search(reference_vector, top_k=len(store.metadata))
    return {
        "source": source,
        "chunk_index": chunk_index,
        "all_scores": _annotate_with_projection(all_results, _xy_by_key(store)),
    }
