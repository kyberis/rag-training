"""
Wrapper delgado sobre la API de embeddings de OpenAI.

Aislar esto en su propio módulo (en vez de llamar a la API desde todos
lados) es lo que permite cambiar de proveedor de embeddings (Cohere, un
modelo local, Voyage) tocando un solo archivo, sin tener que salir a
modificar ingest.py, retriever.py, etc.
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
    """Embebe una lista de textos en un solo request (batch).

    Mandar varios textos juntos es más barato y más rápido que un request
    por chunk — importante cuando el knowledge base tiene miles de chunks.
    """
    client = _get_client()
    response = client.embeddings.create(model=config.EMBEDDING_MODEL, input=texts)
    # La API devuelve los resultados en el mismo orden que la lista de entrada.
    return [item.embedding for item in response.data]


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
