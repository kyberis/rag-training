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

# --- Parámetros de RAG agéntico (ver src/agentic_rag.py) ---
AGENTIC_TOP_K = 2           # más chico que TOP_K: le da al modelo un motivo real para volver a buscar
AGENTIC_MAX_ITERATIONS = 4  # tope duro; la última vuelta fuerza tool_choice="none" así siempre termina con una respuesta

# --- Rutas ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "docplanner_kb"
INDEX_DIR = BASE_DIR / "index"
INDEX_VECTORS_PATH = INDEX_DIR / "vectors.npy"
INDEX_META_PATH = INDEX_DIR / "meta.json"
RESULTS_SNAPSHOT_PATH = BASE_DIR / "eval" / "results_snapshot.json"

# --- Sesiones de demo (ver src/session_store.py) ---
# Si está seteada, las sesiones por-visitante (landing page -> "Start the
# demo") se guardan en Redis con TTL nativo de 24h — correcto incluso entre
# instancias serverless distintas en Vercel. Si no está seteada, se usa un
# dict en memoria del proceso (alcanza para desarrollo local, un solo
# proceso long-running).
REDIS_URL = os.getenv("REDIS_URL") or os.getenv("KV_URL")
