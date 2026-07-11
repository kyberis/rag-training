"""
Almacenamiento por-sesión de un SimpleVectorStore, con expiración a las 24h.

Esto es lo que aísla la instancia de demo de cada visitante (ver la pantalla
de aterrizaje / botón "Start the demo" en la UI) del índice global compartido
que usa el resto del código (retriever.py sin session_id) — cada sesión tiene
su propia copia de los vectores, que nunca pisa el índice committeado en
index/.

Dos backends detrás de la misma interfaz mínima:

- RedisSessionStore: durable entre cold starts, correcto en Vercel (el TTL
  nativo de una key de Redis es literalmente "se borra después de un día" —
  no hace falta ningún cron/thread de limpieza).
- InMemorySessionStore: fallback para desarrollo local sin Redis corriendo.
  Alcanza porque un proceso local (`python -m web.server`) vive mucho tiempo.

Ninguno de los dos es la fuente de verdad del índice compartido — ese sigue
siendo config.INDEX_VECTORS_PATH/INDEX_META_PATH en disco, sin cambios.
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

SESSION_TTL_SECONDS = 86400  # 24h, fijo desde la creación (no se renueva con el uso)


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
        # Importado acá, no en el tope del módulo: así el camino
        # solo-en-memoria nunca requiere tener el paquete instalado.
        import redis

        self._client = redis.Redis.from_url(url)

    @staticmethod
    def _key(session_id: str) -> str:
        return f"session:{session_id}:store"

    def get(self, session_id: str) -> SimpleVectorStore | None:
        raw = self._client.get(self._key(session_id))
        if raw is None:
            return None
        # Seguro: la única data que se deserializa acá es la que este mismo
        # servidor escribió en set() más abajo — nunca algo mandado por un
        # visitante.
        envelope = pickle.loads(raw)
        store = SimpleVectorStore()
        store.vectors = np.load(io.BytesIO(envelope["vectors_npy"]))
        store.metadata = envelope["metadata"]
        return store

    def set(self, session_id: str, store: SimpleVectorStore) -> None:
        buf = io.BytesIO()
        np.save(buf, store.vectors)  # bytes .npy reales: preserva dtype/shape exacto
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
