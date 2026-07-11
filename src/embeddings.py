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


def _get_client(api_key: str | None = None) -> OpenAI:
    """Build a client for a caller-supplied key (BYOK, e.g. the public demo),
    or fall back to the cached client for the server's own key (local dev /
    self-hosted, where OPENAI_API_KEY lives in .env). BYOK clients are never
    cached — the key only lives for the duration of this one call.
    """
    if api_key:
        return OpenAI(api_key=api_key)

    global _client
    if _client is None:
        if not config.OPENAI_API_KEY:
            raise RuntimeError(
                "Falta una OpenAI API key. Si corrés esto localmente, copiá .env.example "
                "a .env y completá tu clave (ver README.md). Si es la demo pública, pegá "
                "tu propia clave en el campo de la parte superior de la página."
            )
        _client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _client


def embed_texts(texts: list[str], api_key: str | None = None) -> list[list[float]]:
    """Embed a list of texts in a single request (batch).

    Sending several texts together is cheaper and faster than one request
    per chunk — important once the knowledge base has thousands of chunks.
    """
    client = _get_client(api_key)
    response = client.embeddings.create(model=config.EMBEDDING_MODEL, input=texts)
    # The API returns results in the same order as the input list.
    return [item.embedding for item in response.data]


def embed_query(text: str, api_key: str | None = None) -> list[float]:
    return embed_texts([text], api_key=api_key)[0]
