"""
Capa de retrieval: dado un texto de consulta, devuelve los top-K chunks
más relevantes del índice ya construido (ver ingest.py). Esta es la pieza
que se mide con Recall@K (punto 1 del framework).
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict

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

# In-process hot cache for session-scoped stores: session_id -> {"store",
# "pca_basis"}. This is purely a cache in front of the durable per-session
# storage (Redis or in-memory — see session_store.py), which stays the
# source of truth; losing an entry here just costs one extra round-trip to
# re-fetch it, never data loss. Bounded so a warm process serving many
# sessions doesn't grow unbounded.
_SESSION_CACHE_MAX = 64
_session_cache: "OrderedDict[str, dict]" = OrderedDict()
_session_cache_lock = threading.Lock()


def _cache_get(session_id: str) -> dict | None:
    with _session_cache_lock:
        entry = _session_cache.get(session_id)
        if entry is not None:
            _session_cache.move_to_end(session_id)
        return entry


def _cache_put(session_id: str, entry: dict) -> None:
    with _session_cache_lock:
        _session_cache[session_id] = entry
        _session_cache.move_to_end(session_id)
        while len(_session_cache) > _SESSION_CACHE_MAX:
            _session_cache.popitem(last=False)


def get_default_store() -> SimpleVectorStore:
    """El índice global compartido (el que vive en index/vectors.npy), sin
    pasar por ninguna sesión. Único punto de acceso público a ese store para
    quien necesite sembrar una sesión nueva a partir de él (ver
    web/server.py's /api/session/start).
    """
    return _get_store(session_id=None)


def _get_store(session_id: str | None = None) -> SimpleVectorStore:
    if session_id is None:
        global _store
        if _store is None:
            if not config.INDEX_VECTORS_PATH.exists():
                raise RuntimeError(
                    "No hay índice construido todavía. Corré primero: python -m src.ingest"
                )
            _store = SimpleVectorStore.load(config.INDEX_VECTORS_PATH, config.INDEX_META_PATH)
        return _store

    entry = _cache_get(session_id)
    if entry is not None:
        return entry["store"]

    from .session_store import get_session_store

    store = get_session_store().get(session_id)
    if store is None:
        # Sesión nueva (o vencida): la sembramos con una copia del índice
        # compartido, así una sesión inventada o expirada se recupera a un
        # estado funcional en vez de fallar — esto no permite esquivar el
        # rate limiting, que sigue siendo por IP, no por sesión (ver
        # web/server.py).
        get_session_store().set(session_id, _get_store(session_id=None))
        # Se relee en vez de cachear la variable de arriba: así lo que queda
        # en _session_cache es siempre un objeto independiente del store
        # global (session_store.set()/.get() copian), nunca una referencia
        # aliaseada al singleton compartido.
        store = get_session_store().get(session_id)

    _cache_put(session_id, {"store": store, "pca_basis": None})
    return store


def reset_store() -> None:
    """Invalida el índice cacheado en memoria (solo el global, no las sesiones).

    Necesario cuando el proceso vive más que una sola consulta (por ejemplo
    el server web): si se reconstruye el índice en disco, hay que forzar a
    que la próxima llamada a retrieve() lo vuelva a cargar en vez de seguir
    usando el store viejo cacheado en _store.
    """
    global _store, _pca_basis
    _store = None
    _pca_basis = None


def set_store(store: SimpleVectorStore, session_id: str | None = None) -> None:
    """Pone en memoria un store ya construido, sin pasar por disco.

    Sin session_id: activa el store global para esta instancia del servidor
    (en un filesystem de solo lectura — la demo pública en Vercel — guardar
    el índice en disco puede fallar, pero el objeto igual existe en memoria).

    Con session_id: el store queda aislado a esa sesión (ver
    session_store.py), nunca pisa el índice global ni el de otra sesión.
    """
    if session_id is None:
        global _store, _pca_basis
        _store = store
        _pca_basis = None
        return

    from .session_store import get_session_store

    get_session_store().set(session_id, store)
    _cache_put(session_id, {"store": store, "pca_basis": None})


def _fit_pca(store: SimpleVectorStore) -> dict:
    """Plain numpy SVD, no sklearn dependency: centering the vectors and
    taking the top-2 right singular vectors gives the same 2 principal
    components, which is all the "vector map" visualization needs.
    """
    vectors = store.vectors.astype(np.float64)
    mean = vectors.mean(axis=0)
    _, _, vt = np.linalg.svd(vectors - mean, full_matrices=False)
    return {"mean": mean, "components": vt[:2]}


def _get_pca_basis(store: SimpleVectorStore, session_id: str | None = None) -> dict:
    """Fits (and caches) a 2-component PCA basis on the store's vectors.

    Session-scoped when session_id is given — otherwise one visitor's
    projection could leak into another's "vector map".
    """
    if session_id is None:
        global _pca_basis
        if _pca_basis is None:
            _pca_basis = _fit_pca(store)
        return _pca_basis

    entry = _cache_get(session_id)
    if entry is None:
        # _get_store(session_id) siempre puebla el cache antes de que se
        # pueda llegar acá; este es un fallback defensivo, sin cachear.
        return _fit_pca(store)
    if entry["pca_basis"] is None:
        entry["pca_basis"] = _fit_pca(store)
    return entry["pca_basis"]


def _project(basis: dict, vector) -> tuple[float, float]:
    xy = (np.array(vector, dtype=np.float64) - basis["mean"]) @ basis["components"].T
    return float(xy[0]), float(xy[1])


def _xy_by_key(basis: dict, store: SimpleVectorStore) -> dict[tuple[str, int], tuple[float, float]]:
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
    session_id: str | None = None,
) -> list[dict]:
    top_k = top_k or config.TOP_K
    store = _get_store(session_id)

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
    basis = _get_pca_basis(store, session_id)
    query_x, query_y = _project(basis, query_vector)

    emit(
        on_event,
        "search_done",
        all_scores=_annotate_with_projection(all_results, _xy_by_key(basis, store)),
        top_k_sources=[r["source"] for r in top_results],
        query_projection={"x": query_x, "y": query_y},
    )
    return top_results


def similarity_from_indexed_chunk(source: str, chunk_index: int, session_id: str | None = None) -> dict:
    """Same all_scores shape as retrieve()'s search_done event, but using an
    already-indexed chunk's own stored vector as the reference point instead
    of embedding a new query — lets the vector map show real distances from
    any chunk the user clicks on, with no extra OpenAI call.
    """
    store = _get_store(session_id)
    row = next(
        (i for i, m in enumerate(store.metadata) if m["source"] == source and m["chunk_index"] == chunk_index),
        None,
    )
    if row is None:
        raise ValueError(f"Ese chunk no está en el índice construido: {source}#{chunk_index}")

    reference_vector = store.vectors[row].tolist()
    all_results = store.search(reference_vector, top_k=len(store.metadata))
    basis = _get_pca_basis(store, session_id)
    return {
        "source": source,
        "chunk_index": chunk_index,
        "all_scores": _annotate_with_projection(all_results, _xy_by_key(basis, store)),
    }
