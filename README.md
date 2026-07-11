# RAG de práctica — DocPlanner Support Assistant

**Demo pública, sin instalar nada: [rag-training.vercel.app](https://rag-training.vercel.app)** — documentos, chunks, índice, código fuente y la última evaluación real son navegables gratis; para hacer una pregunta nueva en vivo o reconstruir el índice necesitás pegar tu propia OpenAI API key (ver sección 2.9, "por qué" y cómo funciona).

Este proyecto es un RAG (Retrieval Augmented Generation) real y ejecutable, construido como ejercicio de preparación técnica. El caso de uso es hipotético: un asistente de soporte que responde preguntas de pacientes en base a una base de conocimiento sintética inspirada en el modelo de negocio público de DocPlanner (marketplace que conecta pacientes con médicos, opera como ZnanyLekarz en Polonia y Doctoralia en España/Latam, entre otras marcas).

**Importante:** los documentos en `data/docplanner_kb/` son contenido sintético que yo (el asistente) generé para este ejercicio. No son documentación interna real de DocPlanner — están inspirados en cómo funciona públicamente ese tipo de plataforma (reservas, cancelaciones, teleconsulta, pagos, reseñas, privacidad, panel de administración de clínicas).

---

## 1. Qué se construyó y por qué, paso a paso

### 1.1 La base de conocimiento (`data/docplanner_kb/`)

Nueve documentos markdown, cada uno cubriendo un tema de soporte distinto:

| Archivo | Tema |
|---|---|
| `01_booking_policy.md` | Cómo reservar un turno |
| `02_cancellation_policy.md` | Cancelación y reprogramación |
| `03_teleconsultation.md` | Consultas por videollamada |
| `04_payments_insurance.md` | Pagos, obras sociales, reembolsos |
| `05_doctor_profiles_faq.md` | Verificación de médicos, perfiles |
| `06_reviews_ratings.md` | Reseñas y moderación |
| `07_account_privacy.md` | Cuenta, datos personales, GDPR |
| `08_clinic_admin_tims.md` | Panel de administración para clínicas |
| `09_no_show_policy.md` | Ausencias (no-shows) |

Elegí temas que se solapan a propósito (por ejemplo, cancelación aparece tanto en el documento de cancelación como en el de teleconsulta y no-show) para que el retriever tenga que discriminar cuál es realmente la fuente más relevante — si todo fuera perfectamente único, Recall@K sería trivial y no probaría nada.

### 1.2 Chunking (`src/chunking.py`, `src/config.py`)

Cada documento se corta en fragmentos de **180 palabras con 40 de solapamiento** (`CHUNK_SIZE_WORDS` / `CHUNK_OVERLAP_WORDS` en `config.py`). Corto por palabras, no por caracteres, para no partir palabras a la mitad, y el overlap existe para no perder contexto que quede justo en el borde entre dos chunks — el mismo problema que explicamos en el punto 7 del framework de conceptos (chunking mal hecho = RAG que no encuentra la respuesta aunque el embedding sea perfecto).

En producción, en vez de contar palabras se usaría un tokenizer real (por ejemplo `tiktoken`) para respetar el límite exacto de tokens del embedding model. Acá cuento palabras para no sumar una dependencia extra.

### 1.3 Embeddings (`src/embeddings.py`)

Uso la API de OpenAI (`text-embedding-3-small` por defecto, configurable en `.env`). Está aislado en un solo módulo a propósito: si mañana querés cambiar a Cohere o a un modelo self-hosted, tocás un solo archivo, no todo el pipeline.

### 1.4 Vector store (`src/vector_store.py`)

En vez de usar Pinecone o Weaviate (que requerirían una cuenta externa y agregarían complejidad), implementé un vector store casero con `numpy`: guarda una matriz de vectores normalizados + su metadata (documento fuente, texto del chunk), y busca por similitud coseno con producto punto. Con 9 documentos esto es más que suficiente en performance, y de paso muestra exactamente qué hace un vector store por dentro, sin que sea una caja negra. El índice se persiste en `index/vectors.npy` + `index/meta.json`.

### 1.5 Ingestión (`src/ingest.py`)

Este script conecta todo lo anterior: lee los `.md`, los chunkea, los embebe en batches de a 100, y guarda el índice. Se corre una sola vez (o cada vez que cambia la base de conocimiento) — es el pipeline *offline* de un RAG, separado del pipeline *online* que responde preguntas en tiempo real.

### 1.6 Retriever (`src/retriever.py`)

Dado un texto de consulta, lo embebe con el mismo modelo usado en la ingestión, y devuelve los top-K chunks más parecidos (K=4 por defecto). Esta es la pieza que se mide con **Recall@K**.

### 1.7 Orquestación RAG (`src/rag.py`)

La función `answer(pregunta)` hace lo que describimos en la teoría: retrieval → arma un prompt con los chunks recuperados como contexto (citando la fuente de cada uno) → se lo pasa al LLM con una instrucción explícita de responder solo en base a ese contexto y de decir "no lo sé" si el contexto no alcanza → devuelve la respuesta junto con las fuentes usadas, para que se pueda auditar de dónde salió cada afirmación.

### 1.8 CLI (`chat.py`)

Un loop simple de terminal para hacerle preguntas al asistente de forma interactiva.

### 1.9 Evaluación (`eval/golden_dataset.json`, `eval/evaluate.py`)

Un golden dataset (punto 5 del framework) con 10 preguntas reales que un paciente le haría al asistente, cada una con el documento fuente que *debería* aparecer entre los resultados. `evaluate.py` calcula:

- **Recall@K**: para cada pregunta, ¿el documento esperado apareció entre los top-K recuperados?
- **Faithfulness**, vía **LLM-as-a-judge** (punto 4): para cada pregunta, genero la respuesta completa y le pido a otro llamado al LLM que juzgue, viendo el contexto y la respuesta, si la respuesta está 100% respaldada por ese contexto o si inventó algo.

Esto conecta directamente cuatro de los conceptos que ya vimos (Recall@K, Faithfulness, LLM-as-judge, Golden dataset) en código que corre de verdad, no solo en la teoría.

### 1.10 Smoke test sin costo (`tests/test_pipeline.py`)

Antes de gastar en llamadas reales a la API, este test reemplaza los embeddings por una versión falsa pero determinística (basada en hash de palabras) para probar que chunking + vector store + retrieval funcionan de punta a punta sin errores de "plomería". Ya lo corrí yo mismo al construir el proyecto — pasa sin necesitar ninguna API key.

---

## 2. Cómo correrlo

### 2.1 Requisitos

- Python 3.10+ (probado con 3.10)
- Una API key de OpenAI (https://platform.openai.com/api-keys) — el proyecto hace llamadas reales a la API de embeddings y de chat, así que vas a necesitar crédito cargado en tu cuenta (los costos son mínimos: unos centavos para todo el knowledge base y varias preguntas)

### 2.2 Instalación

```bash
cd /Users/mcsuarez/rag-training
python3 -m venv venv
source venv/bin/activate        # en Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2.3 Configurar tu API key

```bash
cp .env.example .env
```

Editá `.env` y reemplazá `sk-...` por tu clave real de OpenAI.

### 2.4 Verificar que el código funciona, sin gastar nada (opcional pero recomendado)

```bash
python -m tests.test_pipeline
```

Deberías ver:
```
OK chunking: 4 chunks generados a partir de 500 palabras
OK retrieval: top resultado -> doc_0.md (score=0.378)

Todos los smoke tests pasaron sin necesidad de API key.
```

### 2.5 Construir el índice (ingestión)

```bash
python -m src.ingest
```

Esto lee los 9 documentos, genera ~20 chunks, los embebe con OpenAI, y guarda el índice en `index/`. Se corre una sola vez (o de nuevo si editás los documentos de `data/docplanner_kb/`).

### 2.6 Hablar con el asistente

```bash
python chat.py
```

Ejemplo de sesión:
```
Vos: ¿cuánto tiempo antes puedo cancelar sin que me cobren?
Asistente: Podés cancelar sin costo si lo hacés con al menos 24 horas de
anticipación al turno. Si cancelás con menos anticipación, algunas
clínicas pueden aplicar un cargo por cancelación tardía, según su propia
configuración [02_cancellation_policy.md].
Fuentes: 02_cancellation_policy.md
```

### 2.7 Correr la evaluación completa (Recall@K + Faithfulness)

```bash
python -m eval.evaluate
```

Esto corre las 10 preguntas del golden dataset contra el retriever (Recall@K) y contra el pipeline completo con LLM-as-judge (Faithfulness), e imprime el score final de cada métrica.

### 2.8 UI web — ver el pipeline en tiempo real

Además del CLI, el proyecto tiene una UI web (`web/`) pensada para el mismo objetivo educativo: mostrar gráficamente, paso a paso y en tiempo real, qué hace el sistema tanto al construir el índice (ingestión) como al responder una pregunta (retrieval + generación), incluyendo cómo el texto se convierte en vectores y cómo se guardan.

```bash
python -m web.server
# o, equivalente:
uvicorn web.server:app --reload
```

Después abrí [http://127.0.0.1:8000](http://127.0.0.1:8000) en el navegador. Requiere el mismo `.env` con `OPENAI_API_KEY` que el CLI (ver 2.3). **La interfaz de la UI web está en inglés** (pensada para compartirse), aunque la base de conocimiento y las respuestas del asistente siguen en español — la UI lo aclara en un banner.

Lo primero que se ve es una **pantalla de aterrizaje**: explica qué es RAG, muestra un preview no interactivo de los dos pipelines (offline: Documents → Chunking → Embeddings → Vector index; online: Question → Search → Top-K → Prompt → LLM → Answer) y un glosario corto de los conceptos que se van a ver (chunking, embeddings, cosine similarity/top-K, prompt assembly, RAG vs fine-tuning, Recall@K, Faithfulness). El botón **"Start the demo"** arranca ahí una sesión aislada para ese visitante (ver 2.9) y recién ahí aparece la app de cuatro pestañas:

- **Build the Index**: botón para (re)construir el índice, con un diagrama animado (Documents → Chunking → Embeddings → Index saved) que se va iluminando en vivo, mostrando cuántos chunks salió de cada documento y el progreso de cada batch de embeddings.
- **Ask a Question**: hacé una pregunta y mirá en vivo el diagrama (Question → Embedding → Search → Top-K → Prompt → LLM → Answer), el embedding de tu pregunta como una tira de color, la similitud coseno de **todos** los chunks contra tu pregunta (con los top-K resaltados, para ver por qué ganaron sobre el resto), un mapa 2D (PCA) de dónde cae tu pregunta respecto a cada chunk, el prompt final armado con el contexto citado, y la respuesta apareciendo en streaming real token por token. Una nota expandible aclara que esto es RAG clásico de un solo paso — sin LangChain ni LangGraph, sin loop de decisión — para no leerse como agéntico sin serlo.
- **Explore the Data**: navegá los 9 documentos, mirá exactamente en qué rango de palabras se cortó cada chunk (con el overlap resaltado), y — una vez construido el índice — inspeccioná el vector real guardado para cualquier chunk (dimensión, norma, y los 1536 números crudos). Un visor de código muestra el **código fuente real** de `chunking`, `embeddings`, `vector_store` y `_build_prompt`, obtenido en vivo con `inspect.getsource()` — nunca una copia que se pueda desincronizar del código que corrió de verdad.
- **Metrics & Concepts**: Recall@K y Faithfulness (LLM-as-a-judge) corridos contra el golden dataset de 10 preguntas — el snapshot committeado por default, o en vivo si tenés una key. Hallucination rate se muestra como lo que es, `1 − faithfulness`, nunca una medición separada. Latencia real (P50/P95/P99, con el mínimo de muestras honestamente exigido antes de mostrar percentiles) de las preguntas que hiciste en la sesión. Y un glosario filtrado de conceptos de RAG/AI-engineering, incluyendo cuáles quedaron deliberadamente afuera y por qué (DORA metrics, CodeScene, EU AI Act — ninguno aplica a una demo local de un solo autor).

Internamente, `src/ingest.py`, `src/retriever.py`, `src/rag.py` y `eval/evaluate.py` aceptan dos parámetros opcionales: `on_event` (default `None`, emite eventos de progreso) y `api_key` (default `None`, usa `config.OPENAI_API_KEY`) — el CLI no pasa ninguno de los dos y sigue funcionando exactamente igual que antes. `src/chunking.py` expone además `chunk_spans()` (rangos de palabras por chunk, usado para el visor de "Explore"). `eval/generate_snapshot.py` corre una evaluación real y la guarda en `eval/results_snapshot.json` (committeado) — correlo de nuevo (`python -m eval.generate_snapshot`) si cambiás los documentos o el golden dataset. El server (`web/server.py`) es la única parte del proyecto que sabe de FastAPI: traduce esos eventos a Server-Sent Events (SSE) para el frontend estático (`web/static/`, HTML/CSS/JS plano, sin build step), y expone además endpoints de solo lectura para navegar documentos/chunks/vectores/código, todos con whitelist (nunca se arma un path ni se evalúa un símbolo a partir de input del cliente).

### 2.9 Demo pública en Vercel — quién paga las llamadas a OpenAI

La demo en [rag-training.vercel.app](https://rag-training.vercel.app) **sí tiene una OpenAI API key propia** configurada como variable de entorno en Vercel (`OPENAI_API_KEY`), para que cualquiera pueda probar el pipeline en vivo sin fricción de entrada. Como esa key paga las llamadas de cualquier visitante anónimo, tiene dos capas de protección:

- **Rate limiting server-side** (`_check_rate_limit()` en `web/server.py`): 5 acciones gratis por IP por hora (preguntar / reconstruir índice), 1 corrida de evaluación en vivo gratis por IP por día, y un presupuesto compartido de ~300 llamadas a OpenAI por día entre todos los visitantes. Solo aplica cuando el pedido *no* trae su propia key — quien pega la suya nunca choca con estos límites, porque gasta su propia plata, no la del demo.
  - **Límite honesto, no criptográfico:** el contador vive en memoria del proceso, no en una base de datos compartida. Confirmé en producción que sí bloquea ráfagas secuenciales del mismo visitante (una 6ª pregunta seguida devuelve el error de límite sin gastar), pero un abuso deliberado y paralelo repartido entre varias instancias serverless podría superarlo — Vercel no comparte memoria entre instancias. Para un demo educativo chico esto es proporcional; no agrega una dependencia externa (Redis/Vercel KV) para un problema de esta escala.
  - **El backstop real es el tope de gasto de la cuenta de OpenAI** (platform.openai.com/settings/organization/limits) — poné un límite duro ahí, independiente de cualquier bug o gap en este código.
- **Gratis y sin key, siempre:** "Explore the Data" (documentos, chunks, vectores, código fuente real) y el snapshot de "Metrics & Concepts" (`eval/results_snapshot.json`, de una corrida real committeada al repo) no llaman a OpenAI en absoluto — no cuentan contra ningún límite.
- Pegar tu propia key (campo arriba de la página) evita los límites de arriba y te deja usar la demo sin depender del presupuesto compartido. Se guarda solo en `sessionStorage` de esa pestaña (desaparece al cerrarla), viaja únicamente como header (`X-OpenAI-Key`) en el pedido que la necesita — nunca en la URL, nunca logueada ni guardada en el servidor.
- Localmente (`python -m web.server` con tu `.env`), el rate limiting igual corre por código, pero como sos el único usuario en la práctica no debería notarse.

**Si reconstruís el índice desde la demo pública sin pasar por una sesión** (llamando a la API directo, sin header `X-Session-Id`): el filesystem de Vercel es de solo lectura fuera de `/tmp`, así que el índice recién construido **no se guarda en disco** — queda activo en memoria (`retriever.set_store()`) solo para la instancia serverless que atendió ese pedido. Tu próxima pregunta puede o no caer en esa misma instancia (Vercel no garantiza afinidad de sesión). No es un bug, es routing serverless — por eso el índice pre-construido committeado sigue siendo el fallback confiable para cualquier otra instancia.

**Sesiones de demo (landing page → "Start the demo"):** cada visitante que arranca la demo desde la pantalla de aterrizaje recibe un UUID generado en el browser (`crypto.randomUUID()`, guardado en `sessionStorage`, mandado como header `X-Session-Id` — mismo mecanismo que la key BYOK, nunca una cookie), con el que `/api/session/start` le siembra una copia del índice compartido. Desde ahí, cada reconstrucción o pregunta de esa sesión queda aislada — nunca pisa el índice compartido en disco ni el de otra sesión — y expira sola a las 24h. A diferencia del párrafo anterior, **esto sí es una garantía real y no un best-effort**, siempre que el deploy tenga `REDIS_URL` configurada (ver `.env.example`): el TTL nativo de Redis borra la sesión solo, sin ningún cron, y como Redis es compartido entre instancias serverless, tu sesión sobrevive sin importar cuál te responda. Sin `REDIS_URL` configurada (o en local sin Redis corriendo), las sesiones caen a un dict en memoria del proceso — anduvo perfecto para desarrollo local, pero en Vercel eso vuelve a ser best-effort como el resto de este párrafo. Provisionarlo: Vercel Marketplace → una integración de Redis (p. ej. Upstash) → copiá la URL que te inyecte a las env vars del proyecto.

**Cómo se deployó** (por si querés reproducirlo o forkearlo):

```bash
npm i -g vercel   # si no la tenés
vercel link       # una vez, por proyecto
vercel deploy --prod
```

- `pyproject.toml` le dice a Vercel dónde está la app (`web.server:app`) y define el build step.
- `build_static.py` copia `web/static/` a `public/` en cada deploy — Vercel sirve `public/**` directo desde su CDN, sin pasar por la función Python, así que `web/static/` sigue siendo la única fuente de verdad.
- `vercel.json` sube el timeout de la función a 60s (la evaluación completa hace ~30 llamadas reales y puede acercarse al límite).
- `index/vectors.npy` + `index/meta.json` están committeados (ver `.gitignore`) para que las pestañas de solo lectura funcionen sin depender de una ingestión previa en cada instancia.
- **`.vercelignore` es crítico:** a diferencia de lo que yo asumía, el CLI de Vercel *no* excluye automáticamente los archivos listados en `.gitignore` — así que sin un `.vercelignore` explícito, `.env` (con tu key real) terminaría subido al bundle de la función. Si forkeás esto, no lo borres.
- **Statelessness real (fuera de una sesión):** cada request puede caer en una instancia serverless distinta, sin disco compartido. Si reconstruís el índice sin pasar por el flujo de sesión, tu próxima pregunta podría no verlo — no es un bug, la pestaña "Build the Index" lo explica. Dentro de una sesión de demo (ver 2.9) esto no aplica: con `REDIS_URL` configurada, tu índice sobrevive 24h sin importar qué instancia te responda.
- `REDIS_URL` (opcional, ver 2.9): sin configurar, las sesiones de demo caen a memoria del proceso — funcionan, pero solo dentro de la misma instancia serverless. Provisionala vía Vercel Marketplace (p. ej. Upstash Redis) para que sean durables entre instancias.

---

## 3. Cómo extenderlo (para la conversación de entrevista)

Este proyecto implementa RAG "clásico": el retrieval siempre corre antes de generar, sin que el modelo decida nada. Si quisiera evolucionarlo a **agentic RAG** (como discutimos: RAG + tool calling), el cambio sería exponer `retrieve()` como una *tool* que el modelo puede invocar cero, una o varias veces según lo necesite, en vez de llamarla siempre de entrada en `rag.py`. Eso te da un agente que puede decidir no buscar nada si la pregunta no lo requiere, o buscar de nuevo con otra query si la primera búsqueda no alcanzó — al costo de más llamadas al LLM y latencia menos predecible (el mismo trade-off que ya vimos en los puntos de ReAct y tool calling).

Otras extensiones naturales, en orden de esfuerzo creciente:
1. Reemplazar el vector store casero por Chroma o Pinecone si el knowledge base creciera a miles de documentos.
2. Agregar un re-ranker (Cohere Rerank) entre el retrieval y la generación para subir precisión sin cambiar el embedding model.
3. Cachear embeddings de preguntas frecuentes para bajar costo y latencia (P50/P95/P99).
4. Instrumentar `rag.py` con logging de latencia y de qué documentos se citan más, para detectar huecos en la base de conocimiento con el tiempo.
