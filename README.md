# RAG practice project — DocPlanner Support Assistant

**Public demo, nothing to install: [learning.trefolio.com/RAG](https://learning.trefolio.com/RAG)** (also mirrored at [rag-training.vercel.app/RAG](https://rag-training.vercel.app/RAG)) — documents, chunks, the index, the source code, and the last real evaluation run are browsable for free; to ask a brand-new question live or rebuild the index you'll need to paste your own OpenAI API key (see section 2.9, "why" and how it works).

This project is a real, runnable RAG (Retrieval Augmented Generation) system, built as a technical prep exercise. The use case is hypothetical: a support assistant that answers patient questions based on a synthetic knowledge base inspired by DocPlanner's public business model (a marketplace connecting patients with doctors, operating as ZnanyLekarz in Poland and Doctoralia in Spain/Latam, among other brands).

**Important:** the documents in `data/docplanner_kb/` are synthetic content that I (the assistant) generated for this exercise. They are not real internal DocPlanner documentation — they're inspired by how that kind of platform publicly works (bookings, cancellations, teleconsultation, payments, reviews, privacy, clinic admin panel).

---

## 1. What was built, and why, step by step

### 1.1 The knowledge base (`data/docplanner_kb/`)

Nine markdown documents, each covering a distinct support topic:

| File | Topic |
|---|---|
| `01_booking_policy.md` | How to book an appointment |
| `02_cancellation_policy.md` | Cancellation and rescheduling |
| `03_teleconsultation.md` | Video-call consultations |
| `04_payments_insurance.md` | Payments, insurance, refunds |
| `05_doctor_profiles_faq.md` | Doctor verification, profiles |
| `06_reviews_ratings.md` | Reviews and moderation |
| `07_account_privacy.md` | Account, personal data, GDPR |
| `08_clinic_admin_tims.md` | Admin panel for clinics |
| `09_no_show_policy.md` | No-shows |

I deliberately chose overlapping topics (e.g. cancellation shows up in the cancellation doc, the teleconsultation doc, and the no-show doc) so the retriever actually has to discriminate which source is truly most relevant — if everything were perfectly unique, Recall@K would be trivial and wouldn't prove anything.

### 1.2 Chunking (`src/chunking.py`, `src/config.py`)

Each document is cut into fragments of **180 words with 40 words of overlap** (`CHUNK_SIZE_WORDS` / `CHUNK_OVERLAP_WORDS` in `config.py`). I cut by words, not characters, so words never get split in half, and the overlap exists so context sitting right at the boundary between two chunks isn't lost — the same problem covered in point 7 of the concepts framework (bad chunking = a RAG that can't find the answer even with a perfect embedding).

In production, instead of counting words you'd use a real tokenizer (e.g. `tiktoken`) to respect the embedding model's exact token limit. Here I count words to avoid adding an extra dependency.

### 1.3 Embeddings (`src/embeddings.py`)

Uses the OpenAI API (`text-embedding-3-small` by default, configurable via `.env`). It's isolated in its own module on purpose: if you ever want to switch to Cohere or a self-hosted model, you touch one file, not the whole pipeline.

### 1.4 Vector store (`src/vector_store.py`)

Instead of using Pinecone or Weaviate (which would require an external account and add complexity), I implemented a homemade vector store with `numpy`: it stores a matrix of normalized vectors plus their metadata (source document, chunk text), and searches by cosine similarity via dot product. With 9 documents this is more than enough performance-wise, and it shows exactly what a vector store does under the hood, instead of being a black box. The index is persisted to `index/vectors.npy` + `index/meta.json`.

### 1.5 Ingestion (`src/ingest.py`)

This script wires everything above together: reads the `.md` files, chunks them, embeds them in batches of 100, and saves the index. It runs once (or whenever the knowledge base changes) — it's the *offline* half of a RAG pipeline, separate from the *online* half that answers questions in real time.

### 1.6 Retriever (`src/retriever.py`)

Given a query, it embeds it with the same model used at ingestion time, and returns the top-K most similar chunks (K=4 by default). This is the piece measured by **Recall@K**.

### 1.7 RAG orchestration (`src/rag.py`)

The `answer(question)` function does exactly what the theory describes: retrieval → assembles a prompt with the retrieved chunks as context (citing each one's source) → passes it to the LLM with an explicit instruction to answer only from that context and say "I don't know" if the context isn't enough → returns the answer along with the sources used, so every claim can be traced back to where it came from.

### 1.8 CLI (`chat.py`)

A simple terminal loop for asking the assistant questions interactively.

### 1.9 Evaluation (`eval/golden_dataset.json`, `eval/evaluate.py`)

A golden dataset (point 5 of the framework) with 10 real questions a patient might ask the assistant, each paired with the source document that *should* show up among the results. `evaluate.py` computes:

- **Recall@K**: for each question, did the expected document show up among the top-K retrieved?
- **Faithfulness**, via **LLM-as-a-judge** (point 4): for each question, I generate the full answer and ask a second LLM call to judge, given the context and the answer, whether the answer is 100% supported by that context or invented something.

This directly connects four of the concepts already covered (Recall@K, Faithfulness, LLM-as-judge, golden dataset) in code that actually runs, not just theory.

### 1.10 Zero-cost smoke test (`tests/test_pipeline.py`)

Before spending on real API calls, this test replaces embeddings with a fake but deterministic version (based on hashing words) to prove chunking + vector store + retrieval work end to end with no "plumbing" bugs. I already ran it myself while building the project — it passes with no API key needed.

---

## 2. How to run it

### 2.1 Requirements

- Python 3.10+ (tested on 3.10)
- An OpenAI API key (https://platform.openai.com/api-keys) — the project makes real calls to the embeddings and chat APIs, so you'll need credit loaded on your account (costs are minimal: a few cents for the whole knowledge base and several questions)

### 2.2 Installation

```bash
cd /Users/mcsuarez/rag-training
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2.3 Set up your API key

```bash
cp .env.example .env
```

Edit `.env` and replace `sk-...` with your real OpenAI key.

### 2.4 Verify the code works, at zero cost (optional but recommended)

```bash
python -m tests.test_pipeline
```

You should see:
```
OK chunking: 4 chunks generados a partir de 500 palabras
OK retrieval: top resultado -> doc_0.md (score=0.378)

Todos los smoke tests pasaron sin necesidad de API key.
```

### 2.5 Build the index (ingestion)

```bash
python -m src.ingest
```

This reads the 9 documents, generates ~20 chunks, embeds them with OpenAI, and saves the index to `index/`. It runs once (or again if you edit the documents in `data/docplanner_kb/`).

### 2.6 Talk to the assistant

```bash
python chat.py
```

Example session:
```
Vos: ¿cuánto tiempo antes puedo cancelar sin que me cobren?
Asistente: Podés cancelar sin costo si lo hacés con al menos 24 horas de
anticipación al turno. Si cancelás con menos anticipación, algunas
clínicas pueden aplicar un cargo por cancelación tardía, según su propia
configuración [02_cancellation_policy.md].
Fuentes: 02_cancellation_policy.md
```

### 2.7 Run the full evaluation (Recall@K + Faithfulness)

```bash
python -m eval.evaluate
```

This runs the golden dataset's 10 questions against the retriever (Recall@K) and against the full pipeline with LLM-as-judge (Faithfulness), and prints the final score for each metric.

### 2.8 Web UI — watch the pipeline in real time

Besides the CLI, the project has a web UI (`web/`) built for the same educational goal: show, graphically, step by step, and in real time, what the system does both while building the index (ingestion) and while answering a question (retrieval + generation), including how text becomes vectors and how they're stored.

```bash
python -m web.server
# or, equivalently:
uvicorn web.server:app --reload
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser. It requires the same `.env` with `OPENAI_API_KEY` as the CLI (see 2.3). **The web UI's interface is in English** (built to be shared), while the knowledge base and the assistant's answers stay in Spanish — the UI clarifies this in a banner.

The first thing you see is a **landing screen**: it explains what RAG is, shows a non-interactive preview of both pipelines (offline: Documents → Chunking → Embeddings → Vector index; online: Question → Search → Top-K → Prompt → LLM → Answer), and a short glossary of the concepts you're about to see (chunking, embeddings, cosine similarity/top-K, prompt assembly, RAG vs fine-tuning, Recall@K, Faithfulness). The **"Start the demo"** button spins up an isolated session for that visitor there (see 2.9), and only then does the four-tab app appear:

- **Build the Index**: a button to (re)build the index, with an animated diagram (Documents → Chunking → Embeddings → Index saved) that lights up live, showing how many chunks came out of each document and each embedding batch's progress.
- **Ask a Question**: ask something and watch the diagram live (Question → Embedding → Search → Top-K → Prompt → LLM → Answer), your question's embedding as a color strip, the cosine similarity of **every** chunk against your question (with the top-K highlighted, so you can see why they won over the rest), a **3D map** of where your question lands relative to every chunk — rendered with Three.js, with real orbit controls (drag to rotate, scroll to zoom, pan) and an exact-similarity label on hover or tap, not just a rough visual guess — the final assembled prompt with its cited context, and the answer streaming in token by token, for real. A **Classic RAG vs. Agentic RAG toggle** lets you run the exact same question through two different pipelines and compare them side by side: Classic mode is the single fixed pass just described; Agentic mode gives the model a `retrieve_context` tool via native OpenAI tool calling (no LangChain/LangGraph) and lets *it* decide whether to call it zero, one, or several times — reformulating the query in between — before answering, showing a step-by-step trace of every search it chose to run.
- **Explore the Data**: browse the 9 documents, see exactly what word range each chunk was cut from (with the overlap highlighted), and — once the index is built — inspect the real stored vector for any chunk (dimension, norm, and the raw 1536 numbers). A code viewer shows the **real source code** for `chunking`, `embeddings`, `vector_store`, `_build_prompt`, and the agentic loop, fetched live via `inspect.getsource()` — never a copy that could drift out of sync with what actually ran.
- **Metrics & Concepts**: Recall@K and Faithfulness (LLM-as-a-judge) run against the 10-question golden dataset — the committed snapshot by default, or live if you have a key. Hallucination rate is shown as exactly what it is, `1 − faithfulness`, never a separate measurement. An **Embeddings vs. keyword search** card runs the same golden dataset through the real embeddings retriever and a naive, dependency-free lexical-overlap baseline (`src/keyword_retriever.py`), side by side — Recall@K for each plus a per-question hit/miss table, so the value semantic retrieval adds over "just match shared words" is a measured number, not a claim. Real latency (P50/P95/P99, with the minimum sample count honestly required before showing percentiles) for the questions you asked this session. And a filtered glossary of RAG/AI-engineering concepts, including which ones were deliberately left out and why (DORA metrics, CodeScene, EU AI Act — none of them apply to a single-author local demo).

Under the hood, `src/ingest.py`, `src/retriever.py`, `src/rag.py`, `src/agentic_rag.py`, and `eval/evaluate.py` accept a few optional parameters: `on_event` (default `None`, emits progress events), `api_key` (default `None`, uses `config.OPENAI_API_KEY`), and `session_id` (default `None`, uses the shared global index) — the CLI passes none of them and keeps working exactly as before. `src/chunking.py` also exposes `chunk_spans()` (word ranges per chunk, used by the "Explore" viewer). `eval/generate_snapshot.py` runs a real evaluation and saves it to `eval/results_snapshot.json` (committed) — run it again (`python -m eval.generate_snapshot`) if you change the documents or the golden dataset. The server (`web/server.py`) is the only part of the project that knows about FastAPI: it translates those events into Server-Sent Events (SSE) for the static frontend (`web/static/`, plain HTML/CSS/JS, no build step, with Three.js as its only vendored frontend dependency), and also exposes read-only endpoints for browsing documents/chunks/vectors/code, all whitelisted (no path is ever assembled or symbol evaluated from client input).

### 2.9 Public demo on Vercel — who pays for the OpenAI calls

The demo at [learning.trefolio.com/RAG](https://learning.trefolio.com/RAG) **does have its own OpenAI API key**, configured as an environment variable on Vercel (`OPENAI_API_KEY`), so anyone can try the live pipeline with zero setup friction. Since that key pays for every anonymous visitor's calls, it has two layers of protection:

- **Server-side rate limiting** (`_check_rate_limit()` in `web/server.py`): 5 free actions per IP per hour (ask / rebuild the index), 1 free live evaluation run per IP per day, and a shared budget of ~300 OpenAI calls per day across all visitors combined. This only applies when the request does *not* bring its own key — anyone who pastes theirs never runs into these limits, since they're spending their own money, not the demo's.
  - **Honest limit, not a cryptographic guarantee:** the counter lives in the process's memory, not a shared database. I confirmed in production that it does block sequential bursts from the same visitor (a 6th question in a row returns the limit error without spending anything), but a deliberate, parallel abuse spread across several serverless instances could exceed it — Vercel doesn't share memory between instances. For a small educational demo this is proportionate; it avoids adding an external dependency (Redis/Vercel KV) for a problem of this scale.
  - **The real backstop is the spending cap on the OpenAI account** (platform.openai.com/settings/organization/limits) — set a hard limit there, independent of any bug or gap in this code.
- **Always free, no key needed:** "Explore the Data" (documents, chunks, vectors, real source code) and the "Metrics & Concepts" snapshot (`eval/results_snapshot.json`, from a real run committed to the repo) never call OpenAI at all — they don't count against any limit.
- Pasting your own key (the field at the top of the page) skips the limits above and lets you use the demo without depending on the shared budget. It's stored only in that tab's `sessionStorage` (gone when you close it), travels only as a header (`X-OpenAI-Key`) on the request that needs it — never in the URL, never logged or stored on the server.
- Locally (`python -m web.server` with your own `.env`), the rate limiting still runs in code, but since you're effectively the only user in practice, you shouldn't notice it.

**If you rebuild the index from the public demo without going through a session** (calling the API directly, with no `X-Session-Id` header): Vercel's filesystem is read-only outside of `/tmp`, so the freshly built index **isn't saved to disk** — it stays active in memory (`retriever.set_store()`) only for the serverless instance that handled that request. Your next question may or may not land on that same instance (Vercel doesn't guarantee session affinity). That's not a bug, it's serverless routing — which is exactly why the pre-built, committed index remains the reliable fallback for any other instance.

**Demo sessions (landing page → "Start the demo"):** every visitor who starts the demo from the landing screen gets a UUID generated in the browser (`crypto.randomUUID()`, stored in `sessionStorage`, sent as an `X-Session-Id` header — the same mechanism as the BYOK key, never a cookie), which `/api/session/start` uses to seed them a copy of the shared index. From there, every rebuild or question in that session stays isolated — it never touches the shared index on disk or another session's — and it expires on its own after 24h. Unlike the paragraph above, **this is a real guarantee, not best-effort**, as long as the deployment has `REDIS_URL` configured (see `.env.example`): Redis's native TTL deletes the session on its own, with no cron job, and since Redis is shared across serverless instances, your session survives no matter which one answers you. Without `REDIS_URL` configured (or locally with no Redis running), sessions fall back to an in-process dict — works great for local development, but on Vercel that's back to best-effort like the rest of this paragraph. To provision it: Vercel Marketplace → a Redis integration (e.g. Upstash) → copy the URL it injects into the project's env vars.

**How it was deployed** (in case you want to reproduce or fork it):

```bash
npm i -g vercel   # if you don't have it
vercel link       # once, per project
vercel deploy --prod
```

- `pyproject.toml` tells Vercel where the app is (`web.server:app`) and defines the build step.
- `build_static.py` copies `web/static/` to `public/` on every deploy — Vercel serves `public/**` directly from its CDN, without going through the Python function, so `web/static/` stays the single source of truth.
- `vercel.json` raises the function timeout to 60s (the full evaluation makes ~30 real calls and can get close to the limit).
- `index/vectors.npy` + `index/meta.json` are committed (see `.gitignore`) so the read-only tabs work without depending on a prior ingestion on each instance.
- **`.vercelignore` is critical:** unlike what I assumed, the Vercel CLI does *not* automatically exclude files listed in `.gitignore` — so without an explicit `.vercelignore`, `.env` (with your real key) would end up uploaded in the function bundle. If you fork this, don't delete it.
- **Real statelessness (outside of a session):** each request can land on a different serverless instance, with no shared disk. If you rebuild the index without going through the session flow, your next question might not see it — not a bug, the "Build the Index" tab explains it. Inside a demo session (see 2.9) this doesn't apply: with `REDIS_URL` configured, your index survives 24h no matter which instance answers you.
- `REDIS_URL` (optional, see 2.9): unconfigured, demo sessions fall back to process memory — they work, but only within the same serverless instance. Provision it via the Vercel Marketplace (e.g. Upstash Redis) so they're durable across instances.

---

## 3. How to extend it (for the interview conversation)

This project already ships two RAG pipelines side by side, toggled live in the "Ask a Question" tab (see 2.8): **classic RAG**, where retrieval always runs before generation with no decision involved, and **agentic RAG** (`src/agentic_rag.py`), where `retrieve()` is exposed as a tool via native OpenAI tool calling that the model can call zero, one, or several times on its own — reformulating the query between calls — before deciding it has enough context to answer. That gives you an agent that can skip retrieval entirely if the question doesn't need it, or search again with a different query if the first attempt fell short — at the cost of more LLM calls and less predictable latency (the same trade-off already covered in the ReAct and tool-calling concepts).

Other natural extensions, in increasing order of effort:
1. Replace the homemade vector store with Chroma or Pinecone if the knowledge base grew to thousands of documents.
2. Add a re-ranker (Cohere Rerank) between retrieval and generation to boost precision without changing the embedding model.
3. Cache embeddings for frequently asked questions to cut cost and latency (P50/P95/P99).
4. Instrument `rag.py`/`agentic_rag.py` with logging of latency and which documents get cited most, to spot gaps in the knowledge base over time.
