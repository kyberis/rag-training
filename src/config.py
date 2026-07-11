"""
Configuración central del pipeline RAG.

Todos los demás módulos importan sus constantes desde acá. Es la única
parte del código que deberías tocar para cambiar de modelo, de tamaño de
chunk, o de cuántos resultados trae el retriever.
"""
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Claves y modelos ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# --- Parámetros de chunking (ver punto 7 del framework: chunking + embeddings) ---
CHUNK_SIZE_WORDS = 180      # tamaño de cada chunk, en palabras (proxy simple de tokens)
CHUNK_OVERLAP_WORDS = 40    # solapamiento entre chunks consecutivos

# --- Parámetros de retrieval (ver punto 1: Recall@K) ---
TOP_K = 4                  # cuántos chunks se recuperan por pregunta

# --- Rutas ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "docplanner_kb"
INDEX_DIR = BASE_DIR / "index"
INDEX_VECTORS_PATH = INDEX_DIR / "vectors.npy"
INDEX_META_PATH = INDEX_DIR / "meta.json"
RESULTS_SNAPSHOT_PATH = BASE_DIR / "eval" / "results_snapshot.json"
