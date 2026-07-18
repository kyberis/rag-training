"""
Retrieval layer: given a query text, returns the top-K most relevant
chunks from the already-built index (see ingest.py). This is the piece
measured by Recall@K (framework point 1).
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

# Base PCA (mean + top-3 components) fit on the currently loaded store's
# vectors, used to project the 1536-dim vectors down to 3D for the "vector
# map" visualization. Cached per store so repeated questions land in the
# same 3D space instead of jittering around on every search.
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


def get_chunks(session_id: str | None = None) -> list[dict]:
    """All indexed chunks' metadata (source, chunk_index, text), with no
    vector search involved — the read-only access point used by retrieval
    strategies that don't use embeddings (see keyword_retriever.py).
    """
    return _get_store(session_id).metadata


def get_store(session_id: str | None = None) -> SimpleVectorStore:
    """Public access to the active store (session-scoped or the shared
    global one) — for callers that need direct access to vectors/metadata
    instead of going through retrieve()'s embed-then-search flow (see
    classifier.py, trained directly on the store's own chunk vectors).
    """
    return _get_store(session_id)


def get_default_store() -> SimpleVectorStore:
    """The shared global index (the one living in index/vectors.npy), with
    no session involved. The only public access point to that store for
    anyone who needs to seed a new session from it (see
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
        # New (or expired) session: seed it with a copy of the shared
        # index, so a made-up or expired session recovers to a working
        # state instead of failing — this doesn't allow dodging rate
        # limiting, which stays IP-keyed, not session-keyed (see
        # web/server.py).
        get_session_store().set(session_id, _get_store(session_id=None))
        # Re-read instead of caching the variable above: this way what
        # stays in _session_cache is always an object independent from the
        # global store (session_store.set()/.get() copy), never a
        # reference aliased to the shared singleton.
        store = get_session_store().get(session_id)

    _cache_put(session_id, {"store": store, "pca_basis": None})
    return store


def reset_store() -> None:
    """Invalidates the in-memory cached index (only the global one, not sessions).

    Needed when the process outlives a single query (e.g. the web server):
    if the index gets rebuilt on disk, the next call to retrieve() needs to
    be forced to reload it instead of continuing to use the old store
    cached in _store.
    """
    global _store, _pca_basis
    _store = None
    _pca_basis = None


def set_store(store: SimpleVectorStore, session_id: str | None = None) -> None:
    """Activates an already-built store in memory, without going through disk.

    Without session_id: activates the global store for this server
    instance (on a read-only filesystem — the public Vercel demo — saving
    the index to disk can fail, but the object still exists in memory).

    With session_id: the store stays isolated to that session (see
    session_store.py), never overwriting the global index or another
    session's.
    """
    if session_id is None:
        global _store, _pca_basis
        _store = store
        _pca_basis = None
        return

    from .session_store import get_session_store

    get_session_store().set(session_id, store)
    _cache_put(session_id, {"store": store, "pca_basis": None})


_PCA_COMPONENTS = 3  # 3D vector map — see web/static/vector-map-3d.js


def _fit_pca(store: SimpleVectorStore) -> dict:
    """Plain numpy SVD, no sklearn dependency: centering the vectors and
    taking the top-3 right singular vectors gives the same 3 principal
    components, which is all the "vector map" visualization needs.
    """
    vectors = store.vectors.astype(np.float64)
    mean = vectors.mean(axis=0)
    _, _, vt = np.linalg.svd(vectors - mean, full_matrices=False)
    components = vt[:_PCA_COMPONENTS]
    if components.shape[0] < _PCA_COMPONENTS:
        # A store with fewer chunks than requested components (SVD can't
        # return more rows than min(n_chunks, dim)) — pad with zeros so the
        # rest of the code can always assume 3 components, no special cases.
        pad = np.zeros((_PCA_COMPONENTS - components.shape[0], components.shape[1]))
        components = np.vstack([components, pad])
    return {"mean": mean, "components": components}


def _get_pca_basis(store: SimpleVectorStore, session_id: str | None = None) -> dict:
    """Fits (and caches) a 3-component PCA basis on the store's vectors.

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
        # _get_store(session_id) always populates the cache before this
        # point can be reached; this is a defensive fallback, uncached.
        return _fit_pca(store)
    if entry["pca_basis"] is None:
        entry["pca_basis"] = _fit_pca(store)
    return entry["pca_basis"]


def _project(basis: dict, vector) -> tuple[float, float, float]:
    xyz = (np.array(vector, dtype=np.float64) - basis["mean"]) @ basis["components"].T
    return float(xyz[0]), float(xyz[1]), float(xyz[2])


def _xy_by_key(basis: dict, store: SimpleVectorStore) -> dict[tuple[str, int], tuple[float, float, float]]:
    projected = (store.vectors.astype(np.float64) - basis["mean"]) @ basis["components"].T
    return {
        (m["source"], m["chunk_index"]): (float(projected[i, 0]), float(projected[i, 1]), float(projected[i, 2]))
        for i, m in enumerate(store.metadata)
    }


def _annotate_with_projection(results: list[dict], xyz_by_key: dict) -> list[dict]:
    """Attaches each result's 3D PCA position — the shape the vector-map
    frontend needs, whether `results` came from a fresh query embedding or
    from an already-indexed chunk's own stored vector.
    """
    return [
        {
            "source": r["source"],
            "chunk_index": r["chunk_index"],
            "score": r["score"],
            "text_preview": r["text"][:160],
            "x": xyz_by_key[(r["source"], r["chunk_index"])][0],
            "y": xyz_by_key[(r["source"], r["chunk_index"])][1],
            "z": xyz_by_key[(r["source"], r["chunk_index"])][2],
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
    query_x, query_y, query_z = _project(basis, query_vector)

    emit(
        on_event,
        "search_done",
        all_scores=_annotate_with_projection(all_results, _xy_by_key(basis, store)),
        top_k_sources=[r["source"] for r in top_results],
        query_projection={"x": query_x, "y": query_y, "z": query_z},
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
