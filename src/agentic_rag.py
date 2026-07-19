"""
Agentic RAG orchestration: retrieval-as-tool + a bounded ReAct loop.

Deliberately the same input/output contract as src/rag.py::answer() —
that way the server and frontend can treat both modes symmetrically. The
real difference is on the inside: here the model decides, via native
OpenAI tool calling (no LangChain or LangGraph, same as the rest of the
RAG pipeline — see src/langgraph_agent.py and the "Agent Frameworks" tab
for the one deliberate, labeled exception, which rebuilds this exact
agent with LangGraph specifically to make the tradeoff measurable),
whether to call retrieve() zero, one, or several times before
answering — the ReAct pattern (Reason + Act) the Metrics tab's glossary
mentions.

This file is deliberately independent from rag.py (beyond the cited-context
format, which reuses the same "[Fuente: ...]" style): the idea is that each
one is readable end to end on its own in the Explore tab's code viewer —
"here's exactly the classic one, here's exactly the agentic one" — not a
shared abstraction.
"""
from __future__ import annotations

import json
import time

from openai import OpenAI

from . import config
from .events import EventCallback, emit
from .retriever import retrieve

AGENT_SYSTEM_PROMPT = """Sos el asistente de soporte de DocPlanner.
Tenés una herramienta, retrieve_context, que busca fragmentos en la base de
conocimiento. Llamala una o más veces hasta tener información suficiente
para responder — si la pregunta tiene varias partes, o el primer resultado
no alcanza, reformulá la consulta y llamala de nuevo antes de responder.
Respondé UNICAMENTE en base a lo que hayas recuperado con la herramienta,
citando entre corchetes el documento fuente, por ejemplo:
[cancellation_policy.md]. Si después de buscar no encontrás la información
necesaria, decilo explícitamente en vez de inventar algo.
"""

RETRIEVE_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "retrieve_context",
        "description": (
            "Busca los fragmentos más relevantes de la base de conocimiento de "
            "DocPlanner para una consulta dada. Devuelve como máximo unos pocos "
            "fragmentos — si no alcanzan, podés llamar la herramienta de nuevo "
            "con una consulta reformulada o más específica."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "La consulta de búsqueda — la pregunta original o una reformulación más específica.",
                },
            },
            "required": ["query"],
        },
    },
}


def _tag_iteration(on_event: EventCallback | None, iteration: int) -> EventCallback | None:
    """Wraps on_event to inject {"iteration": N} into every event a nested
    call to retrieve() emits — without touching retriever.py: since
    events.emit() already does on_event(name, {**payload, "ts": ...}), it's
    enough to intercept here which callback we pass to retrieve().
    """
    if on_event is None:
        return None
    return lambda name, payload: on_event(name, {**payload, "iteration": iteration})


def _format_tool_result(chunks: list[dict]) -> str:
    if not chunks:
        return "No se encontraron fragmentos relevantes para esa consulta."
    return "\n\n---\n\n".join(f"[Fuente: {c['source']}]\n{c['text']}" for c in chunks)


def answer_agentic(
    question: str,
    top_k: int | None = None,
    on_event: EventCallback | None = None,
    api_key: str | None = None,
    session_id: str | None = None,
) -> dict:
    per_call_top_k = top_k or config.AGENTIC_TOP_K
    emit(on_event, "question_received", question=question, top_k=per_call_top_k)

    try:
        client = OpenAI(api_key=api_key or config.OPENAI_API_KEY)
        messages: list = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        all_chunks: list[dict] = []
        seen_sources: set[str] = set()
        t_start = time.time()

        for iteration in range(1, config.AGENTIC_MAX_ITERATIONS + 1):
            emit(on_event, "agent_iteration_start", iteration=iteration)
            force_final = iteration == config.AGENTIC_MAX_ITERATIONS
            t_turn = time.time()

            response = client.chat.completions.create(
                model=config.CHAT_MODEL,
                messages=messages,
                tools=[RETRIEVE_TOOL_SCHEMA],
                tool_choice="none" if force_final else "auto",
                temperature=0,
            )
            msg = response.choices[0].message

            if not msg.tool_calls:
                # The model decided it already has enough context (or this
                # was the last allowed turn, forced with tool_choice=
                # "none" — that guarantees the loop always terminates here,
                # never in an endless loop).
                emit(on_event, "agent_no_tool_call", iteration=iteration, elapsed_ms=int((time.time() - t_turn) * 1000))
                sources = sorted(seen_sources)
                final_text = msg.content or "No pude generar una respuesta."
                emit(on_event, "agent_answer", answer=final_text)
                emit(
                    on_event,
                    "agent_done",
                    answer=final_text,
                    sources=sources,
                    chunks=all_chunks,
                    iterations=iteration,
                    elapsed_ms=int((time.time() - t_start) * 1000),
                )
                return {"answer": final_text, "sources": sources, "chunks": all_chunks, "iterations": iteration}

            # Required order: the assistant message carrying the
            # tool_calls has to go BEFORE its role="tool" replies in the
            # messages list — otherwise the next API call returns a 400.
            # The SDK's ChatCompletionMessage object can be appended
            # straight into a list of dicts with no conversion needed.
            messages.append(msg)

            for call_index, tool_call in enumerate(msg.tool_calls):
                args = json.loads(tool_call.function.arguments)
                query = args.get("query", question)
                emit(on_event, "agent_tool_call", iteration=iteration, call_index=call_index, query=query)

                chunks = retrieve(
                    query,
                    top_k=per_call_top_k,
                    on_event=_tag_iteration(on_event, iteration),
                    api_key=api_key,
                    session_id=session_id,
                )
                all_chunks.extend(chunks)
                seen_sources.update(c["source"] for c in chunks)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": _format_tool_result(chunks),
                })

        # Unreachable in practice: the last iteration always runs with
        # tool_choice="none", which guarantees a text-only response and the
        # return inside the if above. Kept as a defensive safety net.
        raise RuntimeError("El loop agéntico agotó las iteraciones sin devolver una respuesta final.")
    except Exception as e:
        emit(on_event, "agent_error", message=str(e))
        raise
