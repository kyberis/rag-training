"""
Evaluates the RAG system against a golden dataset (see framework point 5).

Run:
    python -m eval.evaluate

Computes two metrics, the same ones already covered in the theory:

1. Recall@K (point 1): does the expected source document show up among
   the top-K chunks the retriever returned for that question?

2. Faithfulness (point 2), measured with LLM-as-a-judge (point 4): is the
   answer the RAG generated backed by the context it retrieved, or did it
   invent something that wasn't there?

Requires OPENAI_API_KEY configured (see README.md) and the index already
built (python -m src.ingest).
"""
from __future__ import annotations

import json
from pathlib import Path

from openai import OpenAI

from src import config
from src.events import EventCallback, emit
from src.rag import answer
from src.retriever import retrieve

GOLDEN_PATH = Path(__file__).resolve().parent / "golden_dataset.json"

JUDGE_PROMPT = """Sos un evaluador estricto. Te doy un CONTEXTO y una RESPUESTA.
Respondé UNICAMENTE "SI" si cada afirmacion de la RESPUESTA esta respaldada
por el CONTEXTO, o "NO" si la respuesta inventa o afirma algo que no esta
en el contexto. No expliques nada, respondé solo con SI o NO.

CONTEXTO:
{context}

RESPUESTA:
{answer}
"""


def load_golden_dataset() -> list[dict]:
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_recall_at_k(
    golden: list[dict],
    top_k: int | None = None,
    on_event: EventCallback | None = None,
    api_key: str | None = None,
    session_id: str | None = None,
) -> float:
    top_k = top_k or config.TOP_K
    hits = 0
    for item in golden:
        retrieved = retrieve(item["question"], top_k=top_k, api_key=api_key, session_id=session_id)
        retrieved_sources = {c["source"] for c in retrieved}
        expected = set(item["expected_sources"])
        hit = bool(expected & retrieved_sources)
        hits += int(hit)
        emit(
            on_event,
            "recall_item",
            question=item["question"],
            expected_sources=sorted(expected),
            retrieved_sources=sorted(retrieved_sources),
            hit=hit,
        )
        if not hit:
            print(f"  [MISS] '{item['question']}' -> esperaba {expected}, "
                  f"encontró {retrieved_sources}")
    recall = hits / len(golden)
    print(f"\nRecall@{top_k}: {recall:.0%} ({hits}/{len(golden)})")
    emit(on_event, "recall_done", recall=recall, hits=hits, total=len(golden), top_k=top_k)
    return recall


def evaluate_faithfulness(
    golden: list[dict],
    on_event: EventCallback | None = None,
    api_key: str | None = None,
    session_id: str | None = None,
) -> float:
    client = OpenAI(api_key=api_key or config.OPENAI_API_KEY)
    faithful_count = 0
    for item in golden:
        result = answer(item["question"], api_key=api_key, session_id=session_id)
        context = "\n\n".join(c["text"] for c in result["chunks"])
        judge_prompt = JUDGE_PROMPT.format(context=context, answer=result["answer"])
        judge_response = client.chat.completions.create(
            model=config.CHAT_MODEL,
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0,
        )
        verdict = judge_response.choices[0].message.content.strip().upper()
        is_faithful = verdict.startswith("SI")
        faithful_count += int(is_faithful)
        emit(
            on_event,
            "faithfulness_item",
            question=item["question"],
            answer=result["answer"],
            is_faithful=is_faithful,
        )
        print(f"  [{'OK' if is_faithful else 'FALLA'}] {item['question']}")

    score = faithful_count / len(golden)
    print(f"\nFaithfulness score: {score:.0%} ({faithful_count}/{len(golden)})")
    emit(on_event, "faithfulness_done", score=score, faithful_count=faithful_count, total=len(golden))
    return score


if __name__ == "__main__":
    golden = load_golden_dataset()
    print("=== Evaluando Recall@K ===")
    evaluate_recall_at_k(golden)
    print("\n=== Evaluando Faithfulness (LLM-as-judge) ===")
    evaluate_faithfulness(golden)
