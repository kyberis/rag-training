"""
Generates eval/results_snapshot.json: a committed snapshot of a real run
of evaluate_recall_at_k() + evaluate_faithfulness() against the current
index.

Why it exists: the public demo (Vercel) doesn't have its own
OPENAI_API_KEY set aside for this (see README, BYOK section) — so it
can't run this evaluation live for free for every visitor. This snapshot
is the most recent real run, shown by default in the Metrics tab; the
"Run evaluation" button is still available for anyone who wants to
repeat it live with their own key.

Run after any change to data/docplanner_kb/, eval/golden_dataset.json,
or the index:

    python -m src.ingest
    python -m eval.generate_snapshot
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from src import config
from .evaluate import (
    evaluate_classifier_vs_rag,
    evaluate_consistency,
    evaluate_faithfulness,
    evaluate_recall_at_k,
    evaluate_rerank_comparison,
    evaluate_retrieval_comparison,
    load_golden_dataset,
    load_rerank_hard_examples,
)


def generate() -> dict:
    events: dict[str, list] = {
        "recall_item": [], "faithfulness_item": [], "compare_item": [], "rerank_compare_item": [],
        "classifier_item": [],
    }

    def on_event(name: str, payload: dict) -> None:
        if name in events:
            events[name].append(payload)

    golden = load_golden_dataset()
    print("=== Evaluando Recall@K ===")
    recall = evaluate_recall_at_k(golden, on_event=on_event)
    print("\n=== Evaluando Faithfulness (LLM-as-judge) ===")
    faithfulness = evaluate_faithfulness(golden, on_event=on_event)
    print("\n=== Comparando embeddings vs. keyword search ===")
    comparison = evaluate_retrieval_comparison(golden, on_event=on_event)
    print("\n=== Comparando con vs. sin reranking (preguntas difíciles a propósito) ===")
    rerank_comparison = evaluate_rerank_comparison(load_rerank_hard_examples(), top_k=1, on_event=on_event)
    print("\n=== Comparando clasificador entrenado vs. RAG top-1 ===")
    classifier_comparison = evaluate_classifier_vs_rag(golden, on_event=on_event)
    print("\n=== Midiendo consistencia (misma pregunta, 5 corridas) ===")
    consistency = evaluate_consistency(golden[0]["question"], n_runs=5)

    snapshot = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "embedding_model": config.EMBEDDING_MODEL,
        "chat_model": config.CHAT_MODEL,
        "recall": recall,
        "faithfulness": faithfulness,
        "recall_items": events["recall_item"],
        "faithfulness_items": events["faithfulness_item"],
        "rag_recall": comparison["rag_recall"],
        "keyword_recall": comparison["keyword_recall"],
        "compare_items": events["compare_item"],
        "no_rerank_recall": rerank_comparison["no_rerank_recall"],
        "rerank_recall": rerank_comparison["rerank_recall"],
        "rerank_compare_items": events["rerank_compare_item"],
        "classifier_accuracy": classifier_comparison["classifier_accuracy"],
        "rag_top1_accuracy": classifier_comparison["rag_top1_accuracy"],
        "classifier_items": events["classifier_item"],
        "consistency": consistency,
    }
    with open(config.RESULTS_SNAPSHOT_PATH, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)
    print(f"\nSnapshot guardado en {config.RESULTS_SNAPSHOT_PATH}")
    return snapshot


if __name__ == "__main__":
    generate()
