"""
Prompting Lab: same retrieved context, different prompting/sampling
techniques, run side by side so the effect of each is something you watch
happen, not a claim you take on faith.

Kept deliberately separate from rag.py (same philosophy as
agentic_rag.py's docstring: each pipeline should be readable end to end
on its own in the Explore/code-viewer tabs, not hidden behind a shared
abstraction) — this module only reuses retriever.retrieve() for the
context-gathering step, since re-embedding the same question 3+ times
per run would be pure waste, not a meaningful part of the demo.
"""
from __future__ import annotations

import json
import time

from openai import OpenAI

from . import config
from .events import EventCallback, emit
from .rag import SYSTEM_PROMPT, _build_prompt
from .retriever import retrieve

FEWSHOT_EXAMPLES: list[dict] = [
    {
        "question": "¿Cuánto tiempo antes puedo cancelar una cita sin costo?",
        "answer": (
            "Podés cancelar sin costo si lo hacés con al menos 24 horas de anticipación "
            "al turno [02_cancellation_policy.md]."
        ),
    },
    {
        "question": "¿Puedo hacer una consulta por videollamada con mi médico?",
        "answer": (
            "Sí, DocPlanner permite agendar teleconsultas por videollamada con los médicos "
            "que las tengan habilitadas en su perfil [03_teleconsultation.md]."
        ),
    },
]


def _variant_prompt(variant: str, question: str, chunks: list[dict]) -> tuple[str, str]:
    """Returns (system_prompt, user_prompt) for one of the 3 variants, all
    built over the exact same retrieved chunks so the only thing that
    changes between them is the prompting technique itself.
    """
    base_prompt = _build_prompt(question, chunks)

    if variant == "zero_shot":
        return SYSTEM_PROMPT, base_prompt

    if variant == "few_shot":
        examples = "\n\n".join(
            f"Pregunta: {ex['question']}\nRespuesta: {ex['answer']}"
            for ex in FEWSHOT_EXAMPLES[: config.PROMPT_FEWSHOT_EXAMPLES]
        )
        prompt = (
            f"Ejemplos de cómo responder (mismo estilo, mismo formato de cita):\n\n{examples}"
            f"\n\n---\n\n{base_prompt}"
        )
        return SYSTEM_PROMPT, prompt

    if variant == "cot":
        prompt = (
            f"{base_prompt}\n\n"
            "Antes de responder, pensá paso a paso en voz alta bajo el encabezado "
            "'Razonamiento:' (qué dice cada fuente relevante y cómo se conecta con la "
            "pregunta), y después escribí la respuesta final bajo el encabezado "
            "'Respuesta:'. Solo la línea de 'Respuesta:' debe llevar la cita entre corchetes."
        )
        return SYSTEM_PROMPT, prompt

    raise ValueError(f"Variante de prompting desconocida: {variant}")


VARIANT_LABELS = {
    "zero_shot": "Zero-shot",
    "few_shot": "Few-shot",
    "cot": "Chain-of-Thought",
}


def run_prompt_variants(
    question: str,
    top_k: int | None = None,
    on_event: EventCallback | None = None,
    api_key: str | None = None,
    session_id: str | None = None,
) -> dict:
    """Retrieves context once, then runs the same question through 3
    prompting techniques (zero-shot, few-shot, chain-of-thought) over that
    same context — 1 embedding call + 3 chat calls total.
    """
    chunks = retrieve(question, top_k=top_k, on_event=on_event, api_key=api_key, session_id=session_id)
    if not chunks:
        emit(on_event, "no_context")
        return {"variants": {}}

    client = OpenAI(api_key=api_key or config.OPENAI_API_KEY)
    results = {}
    for variant in ("zero_shot", "few_shot", "cot"):
        system_prompt, prompt = _variant_prompt(variant, question, chunks)
        emit(on_event, "variant_start", variant=variant, label=VARIANT_LABELS[variant], prompt=prompt)
        t0 = time.time()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        if on_event is None:
            response = client.chat.completions.create(model=config.CHAT_MODEL, messages=messages, temperature=0)
            full_answer = response.choices[0].message.content
        else:
            stream = client.chat.completions.create(
                model=config.CHAT_MODEL, messages=messages, temperature=0, stream=True
            )
            parts: list[str] = []
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    parts.append(delta)
                    emit(on_event, "variant_token", variant=variant, delta=delta)
            full_answer = "".join(parts)
        emit(on_event, "variant_done", variant=variant, answer=full_answer, elapsed_ms=int((time.time() - t0) * 1000))
        results[variant] = {"answer": full_answer}

    return {"variants": results, "sources": sorted({c["source"] for c in chunks})}


def run_temperature_playground(
    question: str,
    temperatures: list[float] | None = None,
    top_k: int | None = None,
    on_event: EventCallback | None = None,
    api_key: str | None = None,
    session_id: str | None = None,
) -> dict:
    """Same context, same zero-shot prompt, run once per temperature value
    — shows determinism-vs-creativity live instead of asserting it.
    """
    temperatures = temperatures if temperatures is not None else config.TEMPERATURE_PLAYGROUND_VALUES
    chunks = retrieve(question, top_k=top_k, on_event=on_event, api_key=api_key, session_id=session_id)
    if not chunks:
        emit(on_event, "no_context")
        return {"runs": []}

    client = OpenAI(api_key=api_key or config.OPENAI_API_KEY)
    prompt = _build_prompt(question, chunks)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    runs = []
    for temperature in temperatures:
        key = str(temperature)
        emit(on_event, "temp_start", temperature=temperature)
        t0 = time.time()
        if on_event is None:
            response = client.chat.completions.create(
                model=config.CHAT_MODEL, messages=messages, temperature=temperature
            )
            full_answer = response.choices[0].message.content
        else:
            stream = client.chat.completions.create(
                model=config.CHAT_MODEL, messages=messages, temperature=temperature, stream=True
            )
            parts: list[str] = []
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    parts.append(delta)
                    emit(on_event, "temp_token", temperature=temperature, delta=delta)
            full_answer = "".join(parts)
        emit(on_event, "temp_done", temperature=temperature, answer=full_answer, elapsed_ms=int((time.time() - t0) * 1000))
        runs.append({"temperature": temperature, "answer": full_answer})

    return {"runs": runs, "sources": sorted({c["source"] for c in chunks})}


# --------------------------------------------------------- structured output --
#
# Complements Agentic RAG's tool-calling (the model decides whether to call a
# tool) with the other half of "structured" LLM output: the *final answer
# itself* constrained to a schema (json_schema mode), not a decision about
# what to call next.

STRUCTURED_OUTPUT_SCHEMA = {
    "name": config.STRUCTURED_OUTPUT_SCHEMA_NAME,
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "answer": {"type": "string", "description": "La respuesta a la pregunta del usuario."},
            "sources": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Nombres de archivo de las fuentes citadas, ej. '02_cancellation_policy.md'.",
            },
            "confidence": {
                "type": "number",
                "description": "Confianza propia del modelo en la respuesta, entre 0.0 y 1.0.",
            },
        },
        "required": ["answer", "sources", "confidence"],
        "additionalProperties": False,
    },
}


def _validate_structured(payload: dict, schema: dict) -> list[str]:
    """Small hand-rolled schema check (types + required keys only) instead
    of pulling in a jsonschema library — mirrors this project's existing
    "avoid an extra dependency for something this size" calls (see
    chunking.py's word-count-not-tiktoken choice). Returns a list of
    human-readable problems; empty means it passed.
    """
    problems: list[str] = []
    props = schema["schema"]["properties"]
    for key in schema["schema"]["required"]:
        if key not in payload:
            problems.append(f"Falta el campo requerido '{key}'.")
            continue
        expected = props[key]["type"]
        value = payload[key]
        type_ok = {
            "string": isinstance(value, str),
            "number": isinstance(value, (int, float)) and not isinstance(value, bool),
            "array": isinstance(value, list),
        }.get(expected, True)
        if not type_ok:
            problems.append(f"'{key}' debería ser {expected}, llegó {type(value).__name__}.")
    if "confidence" in payload and isinstance(payload["confidence"], (int, float)):
        if not (0.0 <= payload["confidence"] <= 1.0):
            problems.append("'confidence' debería estar entre 0.0 y 1.0.")
    return problems


def run_structured_output(
    question: str,
    top_k: int | None = None,
    on_event: EventCallback | None = None,
    api_key: str | None = None,
    session_id: str | None = None,
) -> dict:
    """One structured-output call (json_schema mode, validated against
    STRUCTURED_OUTPUT_SCHEMA) plus one free-text call on the same prompt,
    so the difference is visible side by side. Not streamed: both calls are
    short and the point is the final shape, not watching tokens arrive.
    """
    chunks = retrieve(question, top_k=top_k, on_event=on_event, api_key=api_key, session_id=session_id)
    if not chunks:
        emit(on_event, "no_context")
        return {}

    client = OpenAI(api_key=api_key or config.OPENAI_API_KEY)
    prompt = _build_prompt(question, chunks)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    emit(on_event, "structured_start")
    structured_response = client.chat.completions.create(
        model=config.CHAT_MODEL,
        messages=messages,
        temperature=0,
        response_format={"type": "json_schema", "json_schema": STRUCTURED_OUTPUT_SCHEMA},
    )
    raw_structured = structured_response.choices[0].message.content
    structured_payload = json.loads(raw_structured)
    problems = _validate_structured(structured_payload, STRUCTURED_OUTPUT_SCHEMA)
    emit(on_event, "structured_done", raw=raw_structured, payload=structured_payload, problems=problems)

    emit(on_event, "freetext_start")
    freetext_response = client.chat.completions.create(model=config.CHAT_MODEL, messages=messages, temperature=0)
    freetext_answer = freetext_response.choices[0].message.content
    emit(on_event, "freetext_done", answer=freetext_answer)

    return {
        "structured": {"raw": raw_structured, "payload": structured_payload, "problems": problems},
        "freetext": {"answer": freetext_answer},
        "sources": sorted({c["source"] for c in chunks}),
    }
