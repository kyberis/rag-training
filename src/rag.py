"""
Orquestación RAG: retrieval + generación.

answer(pregunta) hace exactamente lo que describimos en la teoría:
1. Recupera los chunks más relevantes (retriever.py)
2. Arma un prompt que incluye esos chunks como contexto, citando su fuente
3. Le pide al LLM que responda SOLO en base a ese contexto
4. Devuelve la respuesta + las fuentes usadas (necesario para medir
   faithfulness después, ver eval/evaluate.py)
"""
from __future__ import annotations

import time

from openai import OpenAI

from . import config
from .events import EventCallback, emit
from .retriever import retrieve

SYSTEM_PROMPT = """Sos el asistente de soporte de DocPlanner.
Respondé UNICAMENTE en base al CONTEXTO provisto abajo. Si el contexto no
tiene la informacion necesaria para responder, decilo explicitamente en
vez de inventar algo.
Cuando respondas, cita entre corchetes el documento fuente que usaste,
por ejemplo: [cancellation_policy.md].
"""


def _build_prompt(question: str, chunks: list[dict]) -> str:
    context_blocks = []
    for c in chunks:
        context_blocks.append(f"[Fuente: {c['source']}]\n{c['text']}")
    context = "\n\n---\n\n".join(context_blocks)
    return f"CONTEXTO:\n{context}\n\nPREGUNTA DEL USUARIO:\n{question}"


def answer(
    question: str,
    top_k: int | None = None,
    on_event: EventCallback | None = None,
    api_key: str | None = None,
    session_id: str | None = None,
) -> dict:
    emit(on_event, "question_received", question=question, top_k=top_k or config.TOP_K)

    try:
        chunks = retrieve(question, top_k=top_k, on_event=on_event, api_key=api_key, session_id=session_id)

        if not chunks:
            emit(on_event, "no_context")
            return {
                "answer": "No encontré información relevante en la base de conocimiento.",
                "sources": [],
                "chunks": [],
            }

        client = OpenAI(api_key=api_key or config.OPENAI_API_KEY)
        prompt = _build_prompt(question, chunks)
        sources = sorted({c["source"] for c in chunks})
        emit(on_event, "prompt_built", system_prompt=SYSTEM_PROMPT, prompt=prompt, sources=sources)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        emit(on_event, "llm_start", model=config.CHAT_MODEL)
        t_llm = time.time()

        if on_event is None:
            # Camino sin instrumentar (chat.py, eval/evaluate.py): sin streaming,
            # igual que siempre.
            response = client.chat.completions.create(
                model=config.CHAT_MODEL,
                messages=messages,
                temperature=0,
            )
            full_answer = response.choices[0].message.content
        else:
            # La UI pidió eventos: pedimos la respuesta en streaming para poder
            # emitir cada token a medida que llega (efecto "tiempo real" real,
            # no simulado).
            stream = client.chat.completions.create(
                model=config.CHAT_MODEL,
                messages=messages,
                temperature=0,
                stream=True,
            )
            parts: list[str] = []
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    parts.append(delta)
                    emit(on_event, "llm_token", delta=delta)
            full_answer = "".join(parts)

        emit(on_event, "llm_done", answer=full_answer, elapsed_ms=int((time.time() - t_llm) * 1000))
        emit(on_event, "answer_done", answer=full_answer, sources=sources, chunks=chunks)
        return {
            "answer": full_answer,
            "sources": sources,
            "chunks": chunks,
        }
    except Exception as e:
        emit(on_event, "answer_error", message=str(e))
        raise
