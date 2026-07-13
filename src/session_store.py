"""
Per-session storage for a SimpleVectorStore, expiring after 24h.

This is what isolates each visitor's demo instance (see the landing
screen / "Start the demo" button in the UI) from the shared global index
the rest of the code uses (retriever.py with no session_id) — each
session has its own copy of the vectors, which never overwrites the
committed index in index/.

Two backends behind the same minimal interface:

- RedisSessionStore: durable across cold starts, correct on Vercel (a
  Redis key's native TTL is literally "gets deleted after a day" — no
  cleanup cron/thread needed).
- InMemorySessionStore: fallback for local development with no Redis
  running. Good enough because a local process (`python -m web.server`)
  lives a long time.

Neither one is the source of truth for the shared index — that's still
config.INDEX_VECTORS_PATH/INDEX_META_PATH on disk, unchanged.
"""
from __future__ import annotations

import io
import pickle
import threading
import time
from typing import Protocol

import numpy as np

from . import config
from .vector_store import SimpleVectorStore

SESSION_TTL_SECONDS = 86400  # 24h, fixed from creation time (not renewed on use)


class SessionStore(Protocol):
    def get(self, session_id: str) -> SimpleVectorStore | None: ...
    def set(self, session_id: str, store: SimpleVectorStore) -> None: ...
    def delete(self, session_id: str) -> None: ...


class InMemorySessionStore:
    def __init__(self) -> None:
        self._data: dict[str, dict] = {}  # session_id -> {"vectors", "metadata", "created_at"}
        self._lock = threading.Lock()

    def get(self, session_id: str) -> SimpleVectorStore | None:
        with self._lock:
            entry = self._data.get(session_id)
            if entry is None:
                return None
            if time.time() - entry["created_at"] > SESSION_TTL_SECONDS:
                del self._data[session_id]
                return None
            store = SimpleVectorStore()
            store.vectors = entry["vectors"]
            store.metadata = entry["metadata"]
            return store

    def set(self, session_id: str, store: SimpleVectorStore) -> None:
        with self._lock:
            # .copy() the array and shallow-copy the list: without this, two
            # sessions seeded from the same shared default store would end
            # up holding references to the exact same underlying objects —
            # harmless today (nothing mutates them in place), but it breaks
            # the "each session owns independent data" guarantee this store
            # exists to provide.
            self._data[session_id] = {
                "vectors": store.vectors.copy(),
                "metadata": list(store.metadata),
                "created_at": time.time(),
            }

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._data.pop(session_id, None)


class RedisSessionStore:
    def __init__(self, url: str) -> None:
        # Imported here, not at the top of the module: this way the
        # in-memory-only path never requires the package to be installed.
        import redis

        self._client = redis.Redis.from_url(url)

    @staticmethod
    def _key(session_id: str) -> str:
        return f"session:{session_id}:store"

    def get(self, session_id: str) -> SimpleVectorStore | None:
        raw = self._client.get(self._key(session_id))
        if raw is None:
            return None
        # Safe: the only data deserialized here is what this same server
        # wrote in set() below — never anything sent by a visitor.
        envelope = pickle.loads(raw)
        store = SimpleVectorStore()
        store.vectors = np.load(io.BytesIO(envelope["vectors_npy"]))
        store.metadata = envelope["metadata"]
        return store

    def set(self, session_id: str, store: SimpleVectorStore) -> None:
        buf = io.BytesIO()
        np.save(buf, store.vectors)  # real .npy bytes: preserves dtype/shape exactly
        envelope = {"vectors_npy": buf.getvalue(), "metadata": store.metadata}
        self._client.set(self._key(session_id), pickle.dumps(envelope), ex=SESSION_TTL_SECONDS)

    def delete(self, session_id: str) -> None:
        self._client.delete(self._key(session_id))


_session_store_singleton: SessionStore | None = None


def get_session_store() -> SessionStore:
    global _session_store_singleton
    if _session_store_singleton is None:
        if config.REDIS_URL:
            _session_store_singleton = RedisSessionStore(config.REDIS_URL)
        else:
            _session_store_singleton = InMemorySessionStore()
    return _session_store_singleton
