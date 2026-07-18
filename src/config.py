"""
Central configuration for the RAG pipeline.

Every other module imports its constants from here. It's the only part
of the code you should need to touch to change the model, the chunk
size, or how many results the retriever returns.
"""
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Tokenizer cache (see src/tokenizer_demo.py) ---
# tiktoken downloads its BPE vocab from a remote blob on first use by
# default, which doesn't work on Vercel's read-only filesystem. The vocab
# for o200k_base is pre-fetched once and committed under vendor/, and this
# env var must be set before tiktoken is ever imported (same "set env
# before first use" idiom as load_dotenv() above).
TIKTOKEN_CACHE_DIR = Path(__file__).resolve().parent.parent / "vendor" / "tiktoken_cache"
os.environ["TIKTOKEN_CACHE_DIR"] = str(TIKTOKEN_CACHE_DIR)
TOKENIZER_ENCODING_NAME = os.getenv("TOKENIZER_ENCODING_NAME", "o200k_base")

# --- Keys and models ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# --- Chunking parameters (see framework point 7: chunking + embeddings) ---
CHUNK_SIZE_WORDS = 180      # size of each chunk, in words (a simple proxy for tokens)
CHUNK_OVERLAP_WORDS = 40    # overlap between consecutive chunks

# --- Retrieval parameters (see point 1: Recall@K) ---
TOP_K = 4                  # how many chunks are retrieved per question

# --- Agentic RAG parameters (see src/agentic_rag.py) ---
AGENTIC_TOP_K = 2           # smaller than TOP_K: gives the model a real reason to search again
AGENTIC_MAX_ITERATIONS = 4  # hard cap; the last turn forces tool_choice="none" so it always ends with an answer

# --- Prompting Lab parameters (see src/prompting_lab.py) ---
PROMPT_FEWSHOT_EXAMPLES = 2                   # how many Q&A examples to prepend in the few-shot variant
# 0.0 -> 2.0 is OpenAI's full range; a narrower spread (e.g. 0.0/0.7/1.2)
# made the demo's own point invisible — see TEMPERATURE_SAMPLES below for
# why a wide spread alone still isn't enough on its own.
TEMPERATURE_PLAYGROUND_VALUES = [0.0, 1.0, 2.0]
# Temperature is a property of a *distribution*, not of one draw — a single
# sample per value can't show it (a lucky/unlucky draw looks the same
# either way). Several samples per temperature, compared to each other,
# make the determinism-vs-diversity effect a measured number instead of a
# claim resting on one paragraph "looking" different.
TEMPERATURE_SAMPLES = 3
# High temperature has a known failure mode: without a cap, generation can
# run long (repetitive/rambling continuations) instead of just being
# differently-worded — a real latency/cost risk for a public demo, not a
# theoretical one. Capped short since these are single-paragraph answers.
TEMPERATURE_MAX_TOKENS = 300
STRUCTURED_OUTPUT_SCHEMA_NAME = "docplanner_answer"

# --- Reranker parameters (see src/reranker.py) ---
# Must be > TOP_K: both the "with reranking" and "without" comparison take
# the same 8-candidate pool and only differ in what happens to it, so the
# comparison isolates the reranker's actual effect instead of comparing
# against a smaller, different candidate set.
RERANK_CANDIDATES = 8

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "docplanner_kb"
INDEX_DIR = BASE_DIR / "index"
INDEX_VECTORS_PATH = INDEX_DIR / "vectors.npy"
INDEX_META_PATH = INDEX_DIR / "meta.json"
RESULTS_SNAPSHOT_PATH = BASE_DIR / "eval" / "results_snapshot.json"

# --- Demo sessions (see src/session_store.py) ---
# If set, per-visitor sessions (landing page -> "Start the demo") are
# stored in Redis with a native 24h TTL — correct even across different
# serverless instances on Vercel. If unset, an in-process dict is used
# instead (fine for local development, a single long-running process).
REDIS_URL = os.getenv("REDIS_URL") or os.getenv("KV_URL")
