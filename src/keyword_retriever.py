"""
Naive keyword-search retriever: no embeddings, no OpenAI calls, no vector
math — just lexical overlap between the query and each chunk's text. This
is the "before RAG" baseline the Metrics tab compares the real embeddings
retriever against, to make the value of semantic search visible instead
of assumed (see retriever.py for the embeddings-based version this
mirrors).
"""
from __future__ import annotations

import re

from . import config
from .retriever import get_chunks

# Common Spanish words that appear in nearly every question/chunk in this
# domain and would otherwise dominate the overlap score without carrying
# any topical meaning.
_STOPWORDS = {
    "a", "al", "cómo", "como", "con", "cuál", "cual", "cuándo", "cuando",
    "de", "del", "donde", "dónde", "e", "el", "en", "es", "esta", "está",
    "hay", "la", "las", "lo", "los", "mi", "mis", "no", "o", "para", "por",
    "puedo", "que", "qué", "se", "si", "sin", "son", "su", "sus", "tiene",
    "tienen", "un", "una", "y",
}

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS}


def retrieve_keyword(query: str, top_k: int | None = None, session_id: str | None = None) -> list[dict]:
    """Scores every indexed chunk by the number of unique query tokens it
    shares (plain lexical overlap, no ranking model), and returns the
    top-K. Same output shape as retriever.retrieve() so callers (eval,
    the /api/eval/compare/stream route) can treat both retrievers
    interchangeably.
    """
    top_k = top_k or config.TOP_K
    query_tokens = _tokenize(query)
    chunks = get_chunks(session_id)

    scored = [
        {**chunk, "score": len(query_tokens & _tokenize(chunk["text"]))}
        for chunk in chunks
    ]
    scored.sort(key=lambda c: c["score"], reverse=True)
    return scored[:top_k]
