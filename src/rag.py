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

from openai import OpenAI

from . import config
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


def answer(question: str, top_k: int | None = None) -> dict:
    chunks = retrieve(question, top_k=top_k)

    if not chunks:
        return {
            "answer": "No encontré información relevante en la base de conocimiento.",
            "sources": [],
            "chunks": [],
        }

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    prompt = _build_prompt(question, chunks)

    response = client.chat.completions.create(
        model=config.CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    sources = sorted({c["source"] for c in chunks})
    return {
        "answer": response.choices[0].message.content,
        "sources": sources,
        "chunks": chunks,
    }
