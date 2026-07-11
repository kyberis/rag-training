"""
Pipeline de ingestión (offline / batch).

Corré esto cada vez que agregás o cambiás documentos en data/docplanner_kb/:

    python -m src.ingest

Qué hace, paso a paso (ver README.md para el detalle completo):
1. Lee todos los .md de data/docplanner_kb/
2. Corta cada documento en chunks con overlap (chunking.py)
3. Embebe cada chunk con la API de OpenAI (embeddings.py)
4. Guarda los vectores + su metadata en index/ (vector_store.py)
"""
from __future__ import annotations

import time

from . import config
from .chunking import chunk_text
from .embeddings import embed_texts
from .events import EventCallback, emit
from .vector_store import SimpleVectorStore


def load_documents() -> list[dict]:
    docs = []
    for path in sorted(config.DATA_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        docs.append({"source": path.name, "text": text})
    return docs


def build_index(
    on_event: EventCallback | None = None,
    api_key: str | None = None,
    persist: bool = True,
) -> SimpleVectorStore:
    start = time.time()
    emit(on_event, "ingest_start")
    try:
        docs = load_documents()
        if not docs:
            raise RuntimeError(f"No se encontraron documentos .md en {config.DATA_DIR}")

        emit(
            on_event,
            "docs_loaded",
            documents=[{"source": d["source"], "n_words": len(d["text"].split())} for d in docs],
            count=len(docs),
        )

        all_chunks: list[str] = []
        all_metadata: list[dict] = []
        for doc in docs:
            chunks = chunk_text(doc["text"])
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadata.append({
                    "source": doc["source"],
                    "chunk_index": i,
                    "text": chunk,
                })
            emit(on_event, "doc_chunked", source=doc["source"], n_chunks=len(chunks))

        emit(on_event, "chunking_done", total_chunks=len(all_chunks))

        print(f"Documentos leídos: {len(docs)}")
        print(f"Chunks generados: {len(all_chunks)}")
        print("Generando embeddings con OpenAI... (puede tardar unos segundos)")

        store = SimpleVectorStore()
        batch_size = 100  # evita mandar requests gigantes si el KB crece mucho
        total_batches = (len(all_chunks) + batch_size - 1) // batch_size
        for batch_index, i in enumerate(range(0, len(all_chunks), batch_size)):
            batch_texts = all_chunks[i:i + batch_size]
            batch_meta = all_metadata[i:i + batch_size]
            emit(
                on_event,
                "embedding_batch_start",
                batch_index=batch_index,
                batch_size=len(batch_texts),
                total_batches=total_batches,
            )
            t0 = time.time()
            vectors = embed_texts(batch_texts, api_key=api_key)
            store.add(vectors, batch_meta)
            emit(
                on_event,
                "embedding_batch_done",
                batch_index=batch_index,
                n_vectors=len(vectors),
                elapsed_ms=int((time.time() - t0) * 1000),
            )

        if not persist:
            # Reconstrucción con alcance de sesión (ver retriever.set_store()
            # con session_id): nunca debe pisar el índice compartido en
            # disco, ni siquiera en local donde sí sería escribible — queda
            # aislado en la sesión de ese visitante (Redis o memoria, ver
            # session_store.py).
            print("Índice de esta sesión — no se guarda en el índice compartido de disco.")
            emit(
                on_event,
                "index_saved",
                vectors_path=None,
                meta_path=None,
                n_vectors=store.vectors.shape[0],
                dim=store.vectors.shape[1],
                persisted=False,
                note="Índice de tu sesión de demo — vive aislado ahí (24h), no se guarda en el índice compartido.",
            )
        else:
            try:
                store.save(config.INDEX_VECTORS_PATH, config.INDEX_META_PATH)
                print(f"Índice guardado en {config.INDEX_DIR}/")
                emit(
                    on_event,
                    "index_saved",
                    vectors_path=str(config.INDEX_VECTORS_PATH),
                    meta_path=str(config.INDEX_META_PATH),
                    n_vectors=store.vectors.shape[0],
                    dim=store.vectors.shape[1],
                    persisted=True,
                )
            except OSError as e:
                # Read-only filesystem (the public Vercel deployment): the index
                # still exists in memory and works for this pipeline run and this
                # server instance — it just can't be written to disk here. See
                # retriever.set_store(), which is what the web server uses to
                # activate it without going through disk at all.
                print(f"No se pudo guardar el índice en disco ({e}); sigue disponible en memoria.")
                emit(
                    on_event,
                    "index_saved",
                    vectors_path=None,
                    meta_path=None,
                    n_vectors=store.vectors.shape[0],
                    dim=store.vectors.shape[1],
                    persisted=False,
                    note="Filesystem de solo lectura (deploy serverless) — el índice quedó activo en memoria para esta instancia, pero no en disco.",
                )
        emit(
            on_event,
            "ingest_done",
            n_docs=len(docs),
            n_chunks=len(all_chunks),
            duration_ms=int((time.time() - start) * 1000),
        )
        return store
    except Exception as e:
        emit(on_event, "ingest_error", message=str(e))
        raise


if __name__ == "__main__":
    build_index()
