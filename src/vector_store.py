"""
Vector store mínimo, casero, basado en numpy.

En producción esto sería Pinecone, Weaviate o Chroma (bases de datos
vectoriales de verdad, con índices aproximados tipo HNSW para buscar rápido
sobre millones de vectores). Acá lo hacemos a mano con una matriz numpy y
similitud coseno por fuerza bruta, porque con un puñado de documentos
alcanza y de paso se ve exactamente qué hace un vector store por dentro:
guardar vectores + su metadata, y devolver los más parecidos a un vector
de consulta.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class SimpleVectorStore:
    def __init__(self):
        self.vectors: np.ndarray | None = None   # matriz (n_chunks, dim)
        self.metadata: list[dict] = []            # una entrada de metadata por chunk

    def add(self, vectors: list[list[float]], metadatas: list[dict]) -> None:
        arr = np.array(vectors, dtype=np.float32)
        # Normalizamos cada vector para que el producto punto sea
        # directamente equivalente a similitud coseno.
        arr = arr / np.linalg.norm(arr, axis=1, keepdims=True)
        if self.vectors is None:
            self.vectors = arr
        else:
            self.vectors = np.vstack([self.vectors, arr])
        self.metadata.extend(metadatas)

    def search(self, query_vector: list[float], top_k: int = 4) -> list[dict]:
        if self.vectors is None or len(self.metadata) == 0:
            return []
        q = np.array(query_vector, dtype=np.float32)
        q = q / np.linalg.norm(q)
        scores = self.vectors @ q  # similitud coseno de la query contra cada chunk
        top_idx = np.argsort(-scores)[:top_k]
        return [
            {**self.metadata[i], "score": float(scores[i])}
            for i in top_idx
        ]

    def save(self, vectors_path: Path, meta_path: Path) -> None:
        vectors_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(vectors_path, self.vectors)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, vectors_path: Path, meta_path: Path) -> "SimpleVectorStore":
        store = cls()
        store.vectors = np.load(vectors_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            store.metadata = json.load(f)
        return store
