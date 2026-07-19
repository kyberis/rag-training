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
import re
from itertools import combinations
from pathlib import Path

from openai import OpenAI

from src import config
from src.classifier import fit_nearest_centroid, predict_nearest_centroid
from src.embeddings import embed_query
from src.events import EventCallback, emit
from src.keyword_retriever import retrieve_keyword
from src.rag import answer
from src.reranker import rerank_chunks
from src.retriever import get_store, retrieve

GOLDEN_PATH = Path(__file__).resolve().parent / "golden_dataset.json"
RERANK_HARD_EXAMPLES_PATH = Path(__file__).resolve().parent / "rerank_hard_examples.json"

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


def load_rerank_hard_examples() -> list[dict]:
    """A separate, small set of questions curated specifically because
    plain cosine similarity gets the top-ranked chunk's source wrong on
    all of them (verified by hand against this project's real index) —
    the main golden_dataset.json's 10 questions are all comfortably
    correct even at top-1, which means Recall@K there stays at 100%
    with or without reranking and can't show the reranker's effect at
    all. These questions exist to make that effect measurable instead of
    disclaimed away. See eval/rerank_hard_examples.json.
    """
    with open(RERANK_HARD_EXAMPLES_PATH, "r", encoding="utf-8") as f:
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


def evaluate_retrieval_comparison(
    golden: list[dict],
    top_k: int | None = None,
    on_event: EventCallback | None = None,
    api_key: str | None = None,
    session_id: str | None = None,
) -> dict:
    """Same Recall@K as evaluate_recall_at_k(), computed for both the real
    embeddings retriever and a naive keyword-overlap baseline
    (keyword_retriever.retrieve_keyword) on the exact same questions and
    index — so the gap between them (if any) is a real, measured number,
    not an illustrative claim. Powers the "Embeddings vs. keyword search"
    card in the Metrics tab.
    """
    top_k = top_k or config.TOP_K
    rag_hits = 0
    keyword_hits = 0
    for item in golden:
        expected = set(item["expected_sources"])

        rag_retrieved = retrieve(item["question"], top_k=top_k, api_key=api_key, session_id=session_id)
        rag_sources = {c["source"] for c in rag_retrieved}
        rag_hit = bool(expected & rag_sources)
        rag_hits += int(rag_hit)

        keyword_retrieved = retrieve_keyword(item["question"], top_k=top_k, session_id=session_id)
        keyword_sources = {c["source"] for c in keyword_retrieved}
        keyword_hit = bool(expected & keyword_sources)
        keyword_hits += int(keyword_hit)

        emit(
            on_event,
            "compare_item",
            question=item["question"],
            expected_sources=sorted(expected),
            rag_hit=rag_hit,
            rag_sources=sorted(rag_sources),
            keyword_hit=keyword_hit,
            keyword_sources=sorted(keyword_sources),
        )

    rag_recall = rag_hits / len(golden)
    keyword_recall = keyword_hits / len(golden)
    emit(
        on_event,
        "compare_done",
        rag_recall=rag_recall,
        keyword_recall=keyword_recall,
        total=len(golden),
        top_k=top_k,
    )
    return {"rag_recall": rag_recall, "keyword_recall": keyword_recall}


def evaluate_rerank_comparison(
    golden: list[dict],
    top_k: int | None = None,
    on_event: EventCallback | None = None,
    api_key: str | None = None,
    session_id: str | None = None,
) -> dict:
    """Recall@K with vs. without the LLM-based reranker (src/reranker.py),
    on the exact same over-retrieved candidate pool both times — the only
    difference is whether that pool gets re-scored and re-sorted before
    being cut down to top_k, so any measured lift is the reranker's actual
    effect, not an artifact of comparing different candidate sets.

    Called with top_k=1 against load_rerank_hard_examples() (not the main
    golden dataset) — the reranker's job is fixing *ranking* mistakes,
    which only shows up at a strict cutoff on questions plain cosine
    similarity actually gets wrong at rank 1. At the main dataset's usual
    top_k=4, both sides saturate at 100% and there's nothing to measure.
    """
    top_k = top_k or config.TOP_K
    no_rerank_hits = 0
    rerank_hits = 0
    for item in golden:
        expected = set(item["expected_sources"])

        candidates = retrieve(item["question"], top_k=config.RERANK_CANDIDATES, api_key=api_key, session_id=session_id)

        no_rerank_sources = {c["source"] for c in candidates[:top_k]}
        no_rerank_hit = bool(expected & no_rerank_sources)
        no_rerank_hits += int(no_rerank_hit)

        reranked = rerank_chunks(item["question"], candidates, top_k=top_k, api_key=api_key)
        rerank_sources = {c["source"] for c in reranked}
        rerank_hit = bool(expected & rerank_sources)
        rerank_hits += int(rerank_hit)

        emit(
            on_event,
            "rerank_compare_item",
            question=item["question"],
            expected_sources=sorted(expected),
            no_rerank_hit=no_rerank_hit,
            no_rerank_sources=sorted(no_rerank_sources),
            rerank_hit=rerank_hit,
            rerank_sources=sorted(rerank_sources),
        )

    no_rerank_recall = no_rerank_hits / len(golden)
    rerank_recall = rerank_hits / len(golden)
    emit(
        on_event,
        "rerank_compare_done",
        no_rerank_recall=no_rerank_recall,
        rerank_recall=rerank_recall,
        total=len(golden),
        top_k=top_k,
    )
    return {"no_rerank_recall": no_rerank_recall, "rerank_recall": rerank_recall}


def evaluate_classifier_vs_rag(
    golden: list[dict],
    on_event: EventCallback | None = None,
    api_key: str | None = None,
    session_id: str | None = None,
) -> dict:
    """Compares the tiny nearest-centroid classifier (src/classifier.py,
    trained on the ~20 already-indexed chunk vectors) against RAG's top-1
    retrieval, on the 10 golden-dataset questions — a genuine held-out
    test set the classifier never saw during training. Each question is
    embedded once and that same vector is used for both predictions, so
    this costs 10 OpenAI calls total, not 20.
    """
    store = get_store(session_id)
    basis = fit_nearest_centroid(store)

    classifier_hits = 0
    rag_hits = 0
    for item in golden:
        expected = set(item["expected_sources"])
        query_vector = embed_query(item["question"], api_key=api_key)

        predicted_label, scores = predict_nearest_centroid(basis, query_vector)
        classifier_hit = predicted_label in expected
        classifier_hits += int(classifier_hit)

        rag_top1 = store.search(query_vector, top_k=1)
        rag_source = rag_top1[0]["source"] if rag_top1 else None
        rag_hit = rag_source in expected
        rag_hits += int(rag_hit)

        emit(
            on_event,
            "classifier_item",
            question=item["question"],
            expected_sources=sorted(expected),
            classifier_prediction=predicted_label,
            classifier_hit=classifier_hit,
            rag_top1=rag_source,
            rag_hit=rag_hit,
        )

    classifier_accuracy = classifier_hits / len(golden)
    rag_top1_accuracy = rag_hits / len(golden)
    emit(
        on_event,
        "classifier_done",
        classifier_accuracy=classifier_accuracy,
        rag_top1_accuracy=rag_top1_accuracy,
        total=len(golden),
        n_examples_per_label=basis["n_examples_per_label"],
    )
    return {"classifier_accuracy": classifier_accuracy, "rag_top1_accuracy": rag_top1_accuracy}


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


_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _word_set(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def evaluate_consistency(
    question: str,
    n_runs: int = 5,
    on_event: EventCallback | None = None,
    api_key: str | None = None,
    session_id: str | None = None,
) -> dict:
    """Runs the exact same question through answer() n_runs times (always
    at temperature=0 — rag.py hardcodes it) and measures how consistent
    the results actually are: exact-answer agreement, whether the cited
    source set agrees, and the average pairwise Jaccard word-overlap
    between every pair of answers. temperature=0 usually means near
    deterministic, not guaranteed — this is a real, not simulated,
    measurement of how true that holds up over several real API calls.
    """
    runs = []
    for i in range(n_runs):
        result = answer(question, api_key=api_key, session_id=session_id)
        runs.append({"answer": result["answer"], "sources": sorted(result["sources"])})
        emit(on_event, "consistency_run", run=i, answer=result["answer"], sources=runs[-1]["sources"])

    unique_answers = {r["answer"] for r in runs}
    unique_source_sets = {tuple(r["sources"]) for r in runs}
    exact_match_rate = 1.0 if len(unique_answers) == 1 else (n_runs - len(unique_answers) + 1) / n_runs
    sources_agree = len(unique_source_sets) == 1

    pairs = list(combinations(range(n_runs), 2))
    jaccard_scores = []
    for i, j in pairs:
        a, b = _word_set(runs[i]["answer"]), _word_set(runs[j]["answer"])
        union = a | b
        jaccard_scores.append(len(a & b) / len(union) if union else 1.0)
    avg_jaccard = sum(jaccard_scores) / len(jaccard_scores) if jaccard_scores else 1.0

    result = {
        "question": question,
        "n_runs": n_runs,
        "n_unique_answers": len(unique_answers),
        "exact_match_rate": exact_match_rate,
        "sources_agree": sources_agree,
        "avg_jaccard_similarity": avg_jaccard,
    }
    emit(on_event, "consistency_done", **result)
    return result


if __name__ == "__main__":
    golden = load_golden_dataset()
    print("=== Evaluando Recall@K ===")
    evaluate_recall_at_k(golden)
    print("\n=== Evaluando Faithfulness (LLM-as-judge) ===")
    evaluate_faithfulness(golden)
    print("\n=== Comparando embeddings vs. keyword search ===")
    evaluate_retrieval_comparison(golden)
