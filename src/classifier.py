"""
A genuinely small, real, numpy-only training loop: a nearest-centroid
classifier over the already-indexed chunk vectors, one centroid per
source document (9 classes). No gradient descent, no new dependency —
same "hand-roll it with numpy instead of pulling in sklearn" choice as
retriever.py's PCA (see _fit_pca there).

Training data: the ~20 chunk vectors already in index/vectors.npy,
labeled by their own source document. Test data: the 10 golden-dataset
questions (eval/golden_dataset.json) — different text than any chunk, so
this is a genuine held-out split, not circular.

With only 2-3 examples per class, this is deliberately a toy — and that
smallness IS the pedagogical point, not a flaw to hide: a parametric
classifier this size will not generalize richly, which is exactly why
RAG (which needs no training data at all, just an index) is the more
practical choice for a knowledge base this size. See
eval.evaluate.evaluate_classifier_vs_rag for the measured comparison.
"""
from __future__ import annotations

import numpy as np

from .vector_store import SimpleVectorStore


def fit_nearest_centroid(store: SimpleVectorStore) -> dict:
    """One centroid per source document: the mean of that document's chunk
    vectors, re-normalized to unit length (the mean of several unit
    vectors isn't itself unit length). Prediction is then just "which
    centroid has the highest cosine similarity" — mirrors this project's
    own SimpleVectorStore.search cosine convention, just with one vector
    per class instead of one per chunk.
    """
    labels = sorted({m["source"] for m in store.metadata})
    centroids = []
    counts = []
    for label in labels:
        idx = [i for i, m in enumerate(store.metadata) if m["source"] == label]
        vectors = store.vectors[idx].astype(np.float64)
        centroid = vectors.mean(axis=0)
        norm = np.linalg.norm(centroid)
        centroids.append(centroid / norm if norm > 0 else centroid)
        counts.append(len(idx))
    return {
        "labels": labels,
        "centroids": np.vstack(centroids),
        "n_examples_per_label": dict(zip(labels, counts)),
    }


def predict_nearest_centroid(basis: dict, query_vector) -> tuple[str, dict]:
    """Returns (predicted_label, {label: cosine_score}) for one query vector."""
    q = np.array(query_vector, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm > 0:
        q = q / norm
    scores = basis["centroids"] @ q
    best = int(np.argmax(scores))
    return basis["labels"][best], dict(zip(basis["labels"], scores.tolist()))


def centroid_separation(basis: dict) -> list[dict]:
    """Pairwise cosine similarity between every pair of centroids — a cheap
    way to show how distinguishable the 9 classes are from each other
    (closer to 1 = harder to tell apart), without needing any test data.
    """
    labels = basis["labels"]
    centroids = basis["centroids"]
    pairs = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            pairs.append({
                "a": labels[i],
                "b": labels[j],
                "cosine_similarity": float(centroids[i] @ centroids[j]),
            })
    return pairs
