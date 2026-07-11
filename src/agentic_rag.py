"""
Orquestación RAG agéntica: retrieval-as-tool + loop ReAct acotado.

Mismo contrato de entrada/salida que src/rag.py::answer() a propósito — así
el server y el frontend tratan ambos modos simétricamente. La diferencia
real está adentro: acá el modelo decide, vía tool calling nativo de OpenAI
(sin LangChain ni LangGraph, igual que el resto del proyecto), si llama a
retrieve() cero, una, o varias veces antes de responder — el patrón ReAct
(Reason + Act) que la pestaña de Metrics menciona en su glosario.

Este archivo es deliberadamente independiente de rag.py (más allá del
formato de contexto citado, que reutiliza el mismo estilo "[Fuente: ...]"):
la idea es que cada uno sea legible de punta a punta por separado en el
visor de código de la pestaña Explore — "acá está exactamente el clásico,
acá está exactamente el agéntico" — no una abstracción compartida.
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
    """Envuelve on_event para inyectar {"iteration": N} en cada evento que
    emita una llamada anidada a retrieve() — sin tocar retriever.py: como
    events.emit() ya hace on_event(name, {**payload, "ts": ...}), alcanza
    con interceptar acá qué callback le pasamos a retrieve().
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
                # El modelo decidió que ya tiene suficiente contexto (o esta
                # fue la última vuelta permitida, forzada con tool_choice=
                # "none" — eso garantiza que el loop siempre termina acá,
                # nunca en un bucle sin fin).
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

            # Orden obligatorio: el mensaje del asistente que trae los
            # tool_calls tiene que ir ANTES que sus respuestas role="tool" en
            # la lista de mensajes — si no, la próxima llamada a la API
            # devuelve un 400. El objeto ChatCompletionMessage del SDK se
            # puede appendear directo a una lista de dicts sin conversión.
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

        # Inalcanzable en la práctica: la última iteración siempre corre con
        # tool_choice="none", lo que garantiza una respuesta de solo texto y
        # el return dentro del if de arriba. Queda como resguardo defensivo.
        raise RuntimeError("El loop agéntico agotó las iteraciones sin devolver una respuesta final.")
    except Exception as e:
        emit(on_event, "agent_error", message=str(e))
        raise
