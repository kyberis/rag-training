"""
Simple word-based chunking, with overlap.

We split on words (not characters) so we never cut a word in half, and we
add overlap so context that happens to sit right on the border between two
chunks isn't lost. This is the classic "bad chunking beats a great
embedding model" problem: if the answer's text gets split awkwardly across
chunks with no overlap, retrieval can miss it even though nothing else in
the pipeline is wrong.

In production you'd usually count real tokens with a tokenizer (e.g.
tiktoken) instead of words, to respect the embedding model's exact token
limit. We count words here to avoid an extra dependency and keep the
example easy to read.
"""
from __future__ import annotations

from . import config


def chunk_spans(text: str, chunk_size: int | None = None, overlap: int | None = None) -> list[tuple[int, int]]:
    """Word-index ranges (start, end) for each chunk, without slicing the text yet.

    Exposed separately from chunk_text() so callers that want to *display*
    chunking (e.g. the web UI's Explore tab) can show exactly which word
    range each chunk covers, and how much it overlaps with its neighbor.
    """
    chunk_size = chunk_size or config.CHUNK_SIZE_WORDS
    overlap = overlap or config.CHUNK_OVERLAP_WORDS

    words = text.split()
    if not words:
        return []

    spans: list[tuple[int, int]] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        spans.append((start, end))
        if end >= len(words):
            break
        start = end - overlap  # step back "overlap" words for the next chunk
    return spans


def chunk_text(text: str, chunk_size: int | None = None, overlap: int | None = None) -> list[str]:
    words = text.split()
    return [" ".join(words[start:end]) for start, end in chunk_spans(text, chunk_size, overlap)]
