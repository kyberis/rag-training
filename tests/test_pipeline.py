"""
Pipeline smoke test with NO real API key needed.

Replaces OpenAI embeddings with a fake but deterministic version (word
hashing) so we can prove chunking + vector store + retrieval work end to
end, with no spending on API calls.

This does NOT replace eval/evaluate.py (which does measure real quality
with real embeddings and LLM) — it's only here to verify the code has no
"plumbing" bugs before spending on API calls.

Usage:
    python -m tests.test_pipeline
"""
from __future__ import annotations

import numpy as np

from src.chunking import chunk_text
from src.vector_store import SimpleVectorStore


def fake_embed(text: str, dim: int = 64) -> list[float]:
    """Fake but deterministic embedding: each word adds to a fixed
    dimension of the vector, based on the word's hash. It has no real
    semantic meaning — it's only good for testing the plumbing without
    needing an API key."""
    vec = np.zeros(dim)
    for word in text.lower().split():
        idx = hash(word) % dim
        vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    return (vec / norm if norm > 0 else vec).tolist()


def test_chunking():
    text = " ".join([f"palabra{i}" for i in range(500)])
    chunks = chunk_text(text, chunk_size=180, overlap=40)
    assert len(chunks) >= 3, "Se esperaban al menos 3 chunks para 500 palabras"
    print(f"OK chunking: {len(chunks)} chunks generados a partir de 500 palabras")


def test_vector_store_roundtrip():
    store = SimpleVectorStore()
    docs = [
        "el paciente puede cancelar la cita hasta 24 horas antes sin costo",
        "la teleconsulta permite hablar con el medico por videollamada",
        "el pago se puede hacer con tarjeta de credito o debito",
    ]
    vectors = [fake_embed(d) for d in docs]
    metas = [{"source": f"doc_{i}.md", "text": d} for i, d in enumerate(docs)]
    store.add(vectors, metas)

    query_vector = fake_embed("quiero cancelar mi cita sin pagar nada")
    results = store.search(query_vector, top_k=1)
    assert len(results) == 1
    print(f"OK retrieval: top resultado -> {results[0]['source']} "
          f"(score={results[0]['score']:.3f})")


if __name__ == "__main__":
    test_chunking()
    test_vector_store_roundtrip()
    print("\nTodos los smoke tests pasaron sin necesidad de API key.")
