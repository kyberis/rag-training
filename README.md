# RAG de práctica — DocPlanner Support Assistant

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

---

## 3. Cómo extenderlo (para la conversación de entrevista)

Este proyecto implementa RAG "clásico": el retrieval siempre corre antes de generar, sin que el modelo decida nada. Si quisiera evolucionarlo a **agentic RAG** (como discutimos: RAG + tool calling), el cambio sería exponer `retrieve()` como una *tool* que el modelo puede invocar cero, una o varias veces según lo necesite, en vez de llamarla siempre de entrada en `rag.py`. Eso te da un agente que puede decidir no buscar nada si la pregunta no lo requiere, o buscar de nuevo con otra query si la primera búsqueda no alcanzó — al costo de más llamadas al LLM y latencia menos predecible (el mismo trade-off que ya vimos en los puntos de ReAct y tool calling).

Otras extensiones naturales, en orden de esfuerzo creciente:
1. Reemplazar el vector store casero por Chroma o Pinecone si el knowledge base creciera a miles de documentos.
2. Agregar un re-ranker (Cohere Rerank) entre el retrieval y la generación para subir precisión sin cambiar el embedding model.
3. Cachear embeddings de preguntas frecuentes para bajar costo y latencia (P50/P95/P99).
4. Instrumentar `rag.py` con logging de latencia y de qué documentos se citan más, para detectar huecos en la base de conocimiento con el tiempo.
