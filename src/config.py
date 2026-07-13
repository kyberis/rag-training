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
