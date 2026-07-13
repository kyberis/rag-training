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
from .evaluate import evaluate_faithfulness, evaluate_recall_at_k, load_golden_dataset


def generate() -> dict:
    events: dict[str, list] = {"recall_item": [], "faithfulness_item": []}

    def on_event(name: str, payload: dict) -> None:
        if name in events:
            events[name].append(payload)

    golden = load_golden_dataset()
    print("=== Evaluando Recall@K ===")
    recall = evaluate_recall_at_k(golden, on_event=on_event)
    print("\n=== Evaluando Faithfulness (LLM-as-judge) ===")
    faithfulness = evaluate_faithfulness(golden, on_event=on_event)

    snapshot = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "embedding_model": config.EMBEDDING_MODEL,
        "chat_model": config.CHAT_MODEL,
        "recall": recall,
        "faithfulness": faithfulness,
        "recall_items": events["recall_item"],
        "faithfulness_items": events["faithfulness_item"],
    }
    with open(config.RESULTS_SNAPSHOT_PATH, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)
    print(f"\nSnapshot guardado en {config.RESULTS_SNAPSHOT_PATH}")
    return snapshot


if __name__ == "__main__":
    generate()
