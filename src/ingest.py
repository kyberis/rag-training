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

from . import config
from .chunking import chunk_text
from .embeddings import embed_texts
from .vector_store import SimpleVectorStore


def load_documents() -> list[dict]:
    docs = []
    for path in sorted(config.DATA_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        docs.append({"source": path.name, "text": text})
    return docs


def build_index() -> SimpleVectorStore:
    docs = load_documents()
    if not docs:
        raise RuntimeError(f"No se encontraron documentos .md en {config.DATA_DIR}")

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

    print(f"Documentos leídos: {len(docs)}")
    print(f"Chunks generados: {len(all_chunks)}")
    print("Generando embeddings con OpenAI... (puede tardar unos segundos)")

    store = SimpleVectorStore()
    batch_size = 100  # evita mandar requests gigantes si el KB crece mucho
    for i in range(0, len(all_chunks), batch_size):
        batch_texts = all_chunks[i:i + batch_size]
        batch_meta = all_metadata[i:i + batch_size]
        vectors = embed_texts(batch_texts)
        store.add(vectors, batch_meta)

    store.save(config.INDEX_VECTORS_PATH, config.INDEX_META_PATH)
    print(f"Índice guardado en {config.INDEX_DIR}/")
    return store


if __name__ == "__main__":
    build_index()
