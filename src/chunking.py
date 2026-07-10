"""
Chunking simple basado en palabras, con overlap.

Partimos por palabras (no por caracteres) para no cortar palabras a la
mitad, y agregamos overlap para no perder contexto que quede justo en el
borde entre dos chunks. Esto es exactamente el problema descripto en el
punto 7 del framework: un chunking mal hecho hace que el RAG no encuentre
la respuesta aunque el embedding model sea excelente.

En producción, en vez de contar palabras se suele usar un tokenizer real
(tiktoken, por ejemplo) para respetar el límite de tokens del embedding
model. Acá usamos palabras para no agregar una dependencia extra y
mantener el ejemplo legible.
"""
from __future__ import annotations

from . import config


def chunk_text(text: str, chunk_size: int | None = None, overlap: int | None = None) -> list[str]:
    chunk_size = chunk_size or config.CHUNK_SIZE_WORDS
    overlap = overlap or config.CHUNK_OVERLAP_WORDS

    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
        start = end - overlap  # retrocedemos "overlap" palabras para el siguiente chunk
    return chunks
