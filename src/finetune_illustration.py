"""
Illustrative-only fine-tuning walkthrough: shows what a fine-tuning
training example actually looks like (the real JSONL shape the OpenAI
fine-tuning API expects), built from this project's own real data — never
a hand-written toy example — but with zero training actually happening
and zero OpenAI calls made. See src/classifier.py for the module that
does run a real (if small) training loop.
"""
from __future__ import annotations

import json

from . import config
from .ingest import load_documents

GOLDEN_PATH = config.BASE_DIR / "eval" / "golden_dataset.json"

FINETUNE_SYSTEM_PROMPT = "Sos el asistente de soporte de DocPlanner. Respondé de forma breve y precisa."

# How much of the source document to use as the illustrative "assistant"
# completion — the real fine-tuning example would be a hand-written ideal
# answer, not a raw document excerpt; this is clearly labeled as such.
_EXCERPT_WORDS = 60


def build_finetune_examples() -> list[dict]:
    """One example per golden-dataset question, in the exact JSONL record
    shape OpenAI's fine-tuning API expects: {"messages": [system, user,
    assistant]}. The "assistant" turn is a real excerpt from that
    question's expected source document (not an invented answer) — real
    KB content, clearly not a hand-curated golden answer.
    """
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        golden = json.load(f)

    docs_by_source = {d["source"]: d["text"] for d in load_documents()}

    examples = []
    for item in golden:
        source = item["expected_sources"][0]
        text = docs_by_source.get(source, "")
        excerpt = " ".join(text.split()[:_EXCERPT_WORDS])
        examples.append({
            "source_document": source,
            "record": {
                "messages": [
                    {"role": "system", "content": FINETUNE_SYSTEM_PROMPT},
                    {"role": "user", "content": item["question"]},
                    {"role": "assistant", "content": excerpt + "…"},
                ]
            },
        })
    return examples
