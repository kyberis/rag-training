"""
Evaluación del sistema RAG contra un golden dataset (ver punto 5 del framework).

Corré:
    python -m eval.evaluate

Calcula dos métricas, las mismas que ya vimos en la teoría:

1. Recall@K (punto 1): ¿el documento fuente esperado aparece entre los
   top-K chunks que devolvió el retriever para esa pregunta?

2. Faithfulness (punto 2), medido con LLM-as-a-judge (punto 4): ¿la
   respuesta generada por el RAG está respaldada por el contexto que
   recuperó, o inventó algo que no estaba ahí?

Requiere OPENAI_API_KEY configurada (ver README.md) y el índice ya
construido (python -m src.ingest).
"""
from __future__ import annotations

import json
from pathlib import Path

from openai import OpenAI

from src import config
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


def evaluate_recall_at_k(golden: list[dict], top_k: int | None = None) -> float:
    top_k = top_k or config.TOP_K
    hits = 0
    for item in golden:
        retrieved = retrieve(item["question"], top_k=top_k)
        retrieved_sources = {c["source"] for c in retrieved}
        expected = set(item["expected_sources"])
        if expected & retrieved_sources:
            hits += 1
        else:
            print(f"  [MISS] '{item['question']}' -> esperaba {expected}, "
                  f"encontró {retrieved_sources}")
    recall = hits / len(golden)
    print(f"\nRecall@{top_k}: {recall:.0%} ({hits}/{len(golden)})")
    return recall


def evaluate_faithfulness(golden: list[dict]) -> float:
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    faithful_count = 0
    for item in golden:
        result = answer(item["question"])
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
        print(f"  [{'OK' if is_faithful else 'FALLA'}] {item['question']}")

    score = faithful_count / len(golden)
    print(f"\nFaithfulness score: {score:.0%} ({faithful_count}/{len(golden)})")
    return score


if __name__ == "__main__":
    golden = load_golden_dataset()
    print("=== Evaluando Recall@K ===")
    evaluate_recall_at_k(golden)
    print("\n=== Evaluando Faithfulness (LLM-as-judge) ===")
    evaluate_faithfulness(golden)
