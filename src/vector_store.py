"""
Minimal, homemade vector store, built on numpy.

In production this would be Pinecone, Weaviate or Chroma (real vector
databases, with approximate indexes like HNSW to search fast over millions
of vectors). Here we do it by hand with a numpy matrix and brute-force
cosine similarity, because a handful of documents is plenty, and it shows
exactly what a vector store does under the hood: store vectors + their
metadata, and return the ones most similar to a query vector.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class SimpleVectorStore:
    def __init__(self):
        self.vectors: np.ndarray | None = None   # matrix (n_chunks, dim)
        self.metadata: list[dict] = []            # one metadata entry per chunk

    def add(self, vectors: list[list[float]], metadatas: list[dict]) -> None:
        arr = np.array(vectors, dtype=np.float32)
        # Normalize each vector so the dot product is directly equivalent
        # to cosine similarity.
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
        scores = self.vectors @ q  # cosine similarity of the query against every chunk
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
