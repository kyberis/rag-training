"""
The same bounded ReAct agent as src/agentic_rag.py, orchestrated with
LangGraph's StateGraph instead of a hand-rolled loop — a deliberate,
single, labeled exception to this project's "no agent framework" rule
(see agentic_rag.py's docstring), built to make the tradeoff measurable
rather than asserted. See the "Agent Frameworks" tab.

Same system prompt, same tool schema, same tool-result formatting, same
retrieval call, same model, same iteration cap as agentic_rag.py — all
imported from there, not reimplemented, so the two are actually
comparable and can't silently drift apart. The only real difference is
the orchestration: a two-node cyclic graph (agent -> tools -> agent,
with a conditional agent -> end) instead of a Python for-loop.

Node functions call the raw OpenAI SDK directly, exactly like the rest
of this project — LangGraph supplies only the graph/state machinery,
never langchain-openai or any other model wrapper.
"""
from __future__ import annotations

import json
import time
from typing import TypedDict

from langgraph.graph import END, StateGraph
from openai import OpenAI

from . import config
from .agentic_rag import AGENT_SYSTEM_PROMPT, RETRIEVE_TOOL_SCHEMA, _format_tool_result
from .events import EventCallback, emit
from .retriever import retrieve


class AgentState(TypedDict):
    messages: list
    iteration: int
    all_chunks: list[dict]
    seen_sources: list[str]


def _tag_iteration(on_event: EventCallback | None, iteration: int) -> EventCallback | None:
    if on_event is None:
        return None
    return lambda name, payload: on_event(name, {**payload, "iteration": iteration})


def _build_graph(
    client: OpenAI,
    on_event: EventCallback | None,
    api_key: str | None,
    session_id: str | None,
    per_call_top_k: int,
):
    """Built fresh per-request (compiling does no I/O — see the plan's
    diligence notes) so the closures below can carry on_event/api_key/
    session_id exactly like answer_agentic() does with its own
    parameters, keeping the two functions maximally comparable side by
    side in the Explore tab's code viewer.
    """

    def agent_node(state: AgentState) -> dict:
        emit(on_event, "graph_node_start", node="agent")
        iteration = state["iteration"] + 1
        emit(on_event, "agent_iteration_start", iteration=iteration)
        force_final = iteration == config.AGENTIC_MAX_ITERATIONS
        t_turn = time.time()

        response = client.chat.completions.create(
            model=config.CHAT_MODEL,
            messages=state["messages"],
            tools=[RETRIEVE_TOOL_SCHEMA],
            tool_choice="none" if force_final else "auto",
            temperature=0,
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            emit(on_event, "agent_no_tool_call", iteration=iteration, elapsed_ms=int((time.time() - t_turn) * 1000))
        emit(on_event, "graph_node_done", node="agent")
        return {"messages": state["messages"] + [msg], "iteration": iteration}

    def tools_node(state: AgentState) -> dict:
        emit(on_event, "graph_node_start", node="tools")
        last_msg = state["messages"][-1]
        new_messages: list = []
        all_chunks = list(state["all_chunks"])
        seen_sources = list(state["seen_sources"])

        for call_index, tool_call in enumerate(last_msg.tool_calls):
            args = json.loads(tool_call.function.arguments)
            query = args.get("query", "")
            emit(on_event, "agent_tool_call", iteration=state["iteration"], call_index=call_index, query=query)

            chunks = retrieve(
                query,
                top_k=per_call_top_k,
                on_event=_tag_iteration(on_event, state["iteration"]),
                api_key=api_key,
                session_id=session_id,
            )
            all_chunks.extend(chunks)
            for c in chunks:
                if c["source"] not in seen_sources:
                    seen_sources.append(c["source"])

            new_messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": _format_tool_result(chunks),
            })

        emit(on_event, "graph_edge_taken", edge="tools->agent")
        emit(on_event, "graph_node_done", node="tools")
        return {
            "messages": state["messages"] + new_messages,
            "all_chunks": all_chunks,
            "seen_sources": seen_sources,
        }

    def route_after_agent(state: AgentState) -> str:
        last_msg = state["messages"][-1]
        if last_msg.tool_calls:
            emit(on_event, "graph_edge_taken", edge="agent->tools")
            return "tools"
        emit(on_event, "graph_edge_taken", edge="agent->end")
        return "end"

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", route_after_agent, {"tools": "tools", "end": END})
    graph.add_edge("tools", "agent")
    return graph.compile()


def answer_agentic_langgraph(
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
        app = _build_graph(client, on_event, api_key, session_id, per_call_top_k)

        initial_state: AgentState = {
            "messages": [
                {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            "iteration": 0,
            "all_chunks": [],
            "seen_sources": [],
        }
        t_start = time.time()
        final_state = app.invoke(initial_state, config={"recursion_limit": 25})

        final_msg = final_state["messages"][-1]
        final_text = final_msg.content or "No pude generar una respuesta."
        sources = sorted(final_state["seen_sources"])
        iterations = final_state["iteration"]

        emit(on_event, "agent_answer", answer=final_text)
        emit(
            on_event,
            "agent_done",
            answer=final_text,
            sources=sources,
            chunks=final_state["all_chunks"],
            iterations=iterations,
            elapsed_ms=int((time.time() - t_start) * 1000),
        )
        return {"answer": final_text, "sources": sources, "chunks": final_state["all_chunks"], "iterations": iterations}
    except Exception as e:
        emit(on_event, "agent_error", message=str(e))
        raise
