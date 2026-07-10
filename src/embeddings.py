"""
Thin wrapper around the OpenAI embeddings API.

Isolating this in its own module (instead of calling the API from
everywhere) is what lets you swap embedding providers (Cohere, a local
model, Voyage) by touching a single file, without having to go modify
ingest.py, retriever.py, etc.
"""
from __future__ import annotations

from openai import OpenAI

from . import config

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if not config.OPENAI_API_KEY:
            raise RuntimeError(
                "Falta OPENAI_API_KEY. Copiá .env.example a .env y completá tu clave "
                "(ver README.md, sección 'Cómo correrlo')."
            )
        _client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _client


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts in a single request (batch).

    Sending several texts together is cheaper and faster than one request
    per chunk — important once the knowledge base has thousands of chunks.
    """
    client = _get_client()
    response = client.embeddings.create(model=config.EMBEDDING_MODEL, input=texts)
    # The API returns results in the same order as the input list.
    return [item.embedding for item in response.data]


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
