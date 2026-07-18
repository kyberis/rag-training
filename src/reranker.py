"""
LLM-based pointwise reranker: an extra step between retrieval and
generation that re-scores the retrieved candidates for relevance and
re-sorts them, instead of trusting cosine similarity alone.

Design decision (see README "How to extend it"): the obvious choice here
would be a dedicated reranking API like Cohere Rerank, but that would add
a second LLM provider to a project that's deliberately OpenAI-only. This
reranker asks the existing chat model to score all candidates in a single
batched Structured Outputs call instead — cheaper and lower-latency than
one call per chunk, at the cost of a slightly less independent judgment
per chunk (the model sees all candidates at once, so its scores aren't
fully decorrelated). That tradeoff is stated here, not hidden.
"""
from __future__ import annotations

import json

from openai import OpenAI

from . import config
from .events import EventCallback, emit

RERANK_SCHEMA = {
    "name": "rerank_scores",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "scores": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer", "description": "Posición del fragmento en la lista (0-based)."},
                        "score": {"type": "number", "description": "Relevancia de 0 (nada relevante) a 10 (responde la pregunta directamente)."},
                    },
                    "required": ["index", "score"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["scores"],
        "additionalProperties": False,
    },
}

_RERANK_SYSTEM_PROMPT = (
    "Sos un evaluador de relevancia. Te doy una PREGUNTA y una lista numerada de "
    "FRAGMENTOS. Para cada fragmento, asigná un score de 0 a 10 según qué tan "
    "directamente ayuda a responder la pregunta (10 = responde la pregunta "
    "directamente, 0 = no tiene nada que ver). Devolvé un score para cada "
    "fragmento, en el mismo orden en que fueron dados."
)


def rerank_chunks(
    question: str,
    chunks: list[dict],
    top_k: int | None = None,
    on_event: EventCallback | None = None,
    api_key: str | None = None,
) -> list[dict]:
    """Scores every candidate in `chunks` for relevance to `question` in one
    batched call, re-sorts by that score (descending), and returns the top
    `top_k` (config.TOP_K by default) — same dict shape as retriever.retrieve()
    plus a "rerank_score" key, so callers can treat the result the same way.
    """
    top_k = top_k or config.TOP_K
    if not chunks:
        return []

    fragments = "\n\n".join(f"[{i}] {c['text']}" for i, c in enumerate(chunks))
    user_prompt = f"PREGUNTA:\n{question}\n\nFRAGMENTOS:\n{fragments}"

    client = OpenAI(api_key=api_key or config.OPENAI_API_KEY)
    emit(on_event, "rerank_start", n_candidates=len(chunks))
    response = client.chat.completions.create(
        model=config.CHAT_MODEL,
        messages=[
            {"role": "system", "content": _RERANK_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        response_format={"type": "json_schema", "json_schema": RERANK_SCHEMA},
    )
    payload = json.loads(response.choices[0].message.content)
    scores_by_index = {s["index"]: s["score"] for s in payload["scores"]}

    scored = [
        {**c, "rerank_score": scores_by_index.get(i, 0.0)}
        for i, c in enumerate(chunks)
    ]
    scored.sort(key=lambda c: c["rerank_score"], reverse=True)
    reranked = scored[:top_k]

    emit(
        on_event,
        "rerank_done",
        before=[{"source": c["source"], "chunk_index": c["chunk_index"], "score": c.get("score")} for c in chunks[:top_k]],
        after=[{"source": c["source"], "chunk_index": c["chunk_index"], "rerank_score": c["rerank_score"]} for c in reranked],
    )
    return reranked
