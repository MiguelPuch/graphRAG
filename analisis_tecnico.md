# Análisis Técnico Detallado del Sistema RAG

> **Fecha de análisis:** Marzo 2026  
> **Versión del sistema:** Hybrid RAG (Milvus + Neo4j + GPT-4o-mini)

---

## Índice

1. [Visión general de la arquitectura](#1-visión-general-de-la-arquitectura)
2. [Componentes técnicos](#2-componentes-técnicos)
3. [Pipeline de ingesta y Chunking](#3-pipeline-de-ingesta-y-chunking)
4. [Embeddings y Vector Store](#4-embeddings-y-vector-store)
5. [Retrieval: cómo se recuperan los chunks](#5-retrieval-cómo-se-recuperan-los-chunks)
6. [Grafo legal: Neo4j GraphRAG](#6-grafo-legal-neo4j-graphrag)
7. [Router de preguntas](#7-router-de-preguntas)
8. [Orquestador Híbrido](#8-orquestador-híbrido)
9. [Generación: el LLM](#9-generación-el-llm)
10. [Ejemplo paso a paso: de pregunta a respuesta](#10-ejemplo-paso-a-paso-de-pregunta-a-respuesta)
11. [Ejemplo de generación de chunks](#11-ejemplo-de-generación-de-chunks)
12. [Ejemplo: cuándo se buscan vecinos y cuándo no](#12-ejemplo-cuándo-se-buscan-vecinos-y-cuándo-no)
13. [Problemas principales detectados](#13-problemas-principales-detectados)

---

## 1. Visión general de la arquitectura

El sistema es un **RAG híbrido** con dos motores completamente distintos que conviven bajo un orquestador único:

```
                          ┌─────────────────────────────────────────┐
                          │           API (FastAPI / api.py)         │
                          └───────────────────┬─────────────────────┘
                                              │
                          ┌───────────────────▼─────────────────────┐
                          │        HybridRAGEngine (hybrid_engine.py)│
                          │   ┌──────────┐    ┌──────────────────┐  │
                          │   │  Router  │    │  HybridMetrics   │  │
                          │   │(LLM+heur)│    │  (estadísticas)  │  │
                          │   └────┬─────┘    └──────────────────┘  │
                          │        │ route: legal / technical / both │
                          │  ┌─────▼──────┐  ┌───────────────────┐  │
                          │  │  RAGEngine │  │  GraphRAGEngine   │  │
                          │  │  (Milvus)  │  │     (Neo4j)       │  │
                          │  └─────┬──────┘  └────────┬──────────┘  │
                          └────────┼─────────────────┼─────────────-┘
                                   │                  │
                           ┌───────▼───────┐  ┌──────▼──────┐
                           │  Milvus       │  │  Neo4j      │
                           │  (vectores    │  │  (grafo de  │
                           │  técnicos)    │  │  normativa) │
                           └───────────────┘  └─────────────┘
                                          │
                                  ┌───────▼───────┐
                                  │  OpenAI LLM   │
                                  │ (gpt-4o-mini) │
                                  └───────────────┘
```

### Dos tipos de documentos, dos motores

| Motor | Base de datos | Tipo de documentos | Tipo de búsqueda |
|---|---|---|---|
| `RAGEngine` | **Milvus** (vectorial) | Técnicos: manuales XBRL, plantillas, formularios de estados financieros | Similitud semántica (coseno) + reranking léxico |
| `GraphRAGEngine` | **Neo4j** (grafo) | Legales: Leyes, Reglamentos, Decretos, Circulares CNMV, BOE | Keyword scoring + BM25 + grafo de artículos |

---

## 2. Componentes técnicos

### 2.1 `config.py` — Configuración centralizada

Todos los parámetros del sistema se leen de variables de entorno con valores por defecto:

| Parámetro | Variable de entorno | Valor por defecto | Descripción |
|---|---|---|---|
| Modelo LLM | `RAG_LLM_MODEL` | `gpt-4o-mini` | Modelo de generación |
| Temperatura LLM | `RAG_LLM_TEMPERATURE` | `0.1` | Creatividad del LLM |
| Modelo de embeddings | `RAG_EMBED_MODEL` | `intfloat/multilingual-e5-base` | Modelo de 768 dimensiones |
| Max tokens por chunk | `RAG_CHUNK_MAX_TOKENS` | `530` | Tamaño máximo de cada fragmento |
| Overlap de chunks | `RAG_CHUNK_OVERLAP` | `50` | Tokens de solapamiento entre chunks |
| URI Milvus | `MILVUS_URI` | `http://milvus:19530` | Base de datos vectorial |
| URI Neo4j | `NEO4J_URI` | `bolt://neo4j:7687` | Base de datos de grafo |
| Modelo router | `RAG_ROUTER_MODEL` | mismo que LLM | Modelo para clasificar preguntas |

### 2.2 `DocumentConverter` — PDF/DOCX → Markdown

- Usa **Docling** para convertir PDFs a Markdown.
- Si el fichero ya es `.md` o `.markdown`, lo lee directamente.
- Genera un **slug único** para cada documento: `nombre_limpio_<hash_md5_8chars>`.
- Los Markdowns convertidos se almacenan en disco en `DOCS_DIR`.

### 2.3 `MarkdownChunker` — División inteligente

Responsable de dividir el Markdown en `Chunk` objects. Descripto en detalle en la sección 3.

### 2.4 `EmbeddingService` — Vectorización

- Modelo: `intfloat/multilingual-e5-base` (768 dimensiones).
- Para pasajes usa el prefijo `"passage: "` y para queries `"query: "` (requisito del modelo E5).
- Inicialización **lazy**: solo carga el modelo cuando se necesita por primera vez.

### 2.5 `VectorStore` — Milvus

- Colección: `rag_documents` con índice **IVF_FLAT / COSINE**.
- Campos almacenados: `id`, `doc_slug`, `filename`, `text`, `embedding`.
- **Ingesta idempotente**: antes de insertar, borra los chunks del mismo `doc_slug`.
- Búsqueda con parámetro `nprobe=16` (número de particiones IVF a explorar).

### 2.6 `LLMGenerator` — OpenAI

- Modelo `gpt-4o-mini`, temperatura `0.1`.
- Recibe el contexto empaquetado en etiquetas `<fragmento id="N" fuente="fichero">`.
- **System prompt estricto**: el LLM no puede usar conocimiento externo, solo los fragmentos proporcionados.

### 2.7 `QueryRouter` — Clasificación de intención

Decide si la pregunta va al motor técnico, legal o a ambos. Descripto en sección 7.

### 2.8 `GraphRAGEngine` — Neo4j

Motor para documentos legales. No usa vectores: trabaja con **keyword scoring**, **BM25**, y estructuras de grafo de artículos en Neo4j. Descripto en sección 6.

---

## 3. Pipeline de ingesta y Chunking

### 3.1 Flujo de ingesta técnica

```
Fichero PDF/DOCX/MD
        │
        ▼
DocumentConverter.convert_to_markdown()
        │  (Docling convierte a Markdown)
        ▼
MarkdownChunker.chunk()
        │
        ├─── _attach_table_notes()       ← adhiere notas de tablas al bloque tabla
        ├─── _split_by_headers()         ← divide por cabeceras (# ## ###)
        │
        ├─── Para cada sección:
        │     ├─── Si tokens ≤ 530  →  chunk directo
        │     └─── Si tokens > 530  →  _split_long_section()
        │              ├─ divide por párrafos (\n\n)
        │              ├─ si párrafo > budget: divide por líneas (\n)
        │              └─ prefixa el header de sección a CADA sub-chunk
        ▼
Lista de Chunk objects (id, doc_slug, filename, text, metadata)
        │
        ▼
EmbeddingService.embed_texts()
        │  (prefija "passage: " a cada texto)
        ▼
VectorStore.insert()
        │  (inserta en Milvus)
        ▼
          ✓ Ingesta completada
```

### 3.2 Estrategia de chunking detallada

El chunker opera en 4 pasos:

**Paso 1 — Pre-procesado:** elimina artefactos de conversión PDF (`<!--pagebreak-->`) y ejecuta `_attach_table_notes` para que las notas aclaratorias debajo de tablas (como "Nota a): ..." o "Donde dice:") no se separen de la tabla en el chunk.

**Paso 2 — Split por Headers:** divide el Markdown usando el patrón `^#{1,6}\s+.+$`. Cada sección delimitada por un header se convierte en una unidad de trabajo. El header queda al inicio de la sección.

**Paso 3 — Split por longitud:** si una sección supera los 530 tokens:
1. Se extrae el header de la sección.
2. Se cuenta el presupuesto disponible: `budget = 530 - tokens(header)`.
3. Se divide por párrafos (`\n\n`); los párrafos que exceden solos el presupuesto se dividen a su vez por líneas (`\n`).
4. Se agrupan piezas hasta llenar el presupuesto.
5. El header se **prefixa a cada sub-chunk** para mantener el contexto semántico.

**Paso 4 — Deduplicación por fingerprint:** en el retrieval se usa un fingerprint de los primeros 650 caracteres normalizados para evitar que aparezcan chunks casi idénticos.

---

## 11. Ejemplo de generación de chunks

Supóngase el siguiente fragmento de un manual técnico en Markdown:

```markdown
## Artículo 5 — Cumplimentación del Estado LQ-B

El estado LQ-B recoge la posición de liquidez de la entidad gestora.
Debe cumplimentarse mensualmente antes del día 20 del mes siguiente.

### Celda A01 — Activos líquidos de nivel 1

Los activos líquidos de nivel 1 incluyen:
- Efectivo y reservas en bancos centrales
- Títulos de deuda emitida o garantizada por administraciones centrales
- Bonos con calificación mínima AA-

(a) Se excluyen los activos pignorados como garantía.
(b) Los importes se expresan en miles de euros.

| Código | Descripción                          | Valor mínimo |
|--------|--------------------------------------|--------------|
| A01.1  | Efectivo                             | 0            |
| A01.2  | Reservas banco central               | 0            |
Nota: Los importes de A01.1 y A01.2 deben coincidir con el estado BG-C.
```

**Resultado del chunking:**

Asumiendo que el total supera los 530 tokens, el chunker:

1. Detecta el header `## Artículo 5 — Cumplimentación del Estado LQ-B`.
2. `_attach_table_notes` adhiere la nota `"Nota: Los importes..."` al bloque de tabla antes de chunking.
3. Divide en sub-secciones. La sección `### Celda A01` con la tabla y nota queda en un sub-chunk.
4. Prefixa el header de nivel 2 a cada sub-chunk.

**Chunk 1** (≈ 80 tokens):
```
## Artículo 5 — Cumplimentación del Estado LQ-B

El estado LQ-B recoge la posición de liquidez de la entidad gestora.
Debe cumplimentarse mensualmente antes del día 20 del mes siguiente.
```

**Chunk 2** (≈ 180 tokens, con header propagado):
```
## Artículo 5 — Cumplimentación del Estado LQ-B

### Celda A01 — Activos líquidos de nivel 1

Los activos líquidos de nivel 1 incluyen:
- Efectivo y reservas en bancos centrales
- Títulos de deuda emitida o garantizada por administraciones centrales
- Bonos con calificación mínima AA-

(a) Se excluyen los activos pignorados como garantía.
(b) Los importes se expresan en miles de euros.

| Código | Descripción                          | Valor mínimo |
|--------|--------------------------------------|--------------|
| A01.1  | Efectivo                             | 0            |
| A01.2  | Reservas banco central               | 0            |
Notas asociadas a la tabla:
- Nota: Los importes de A01.1 y A01.2 deben coincidir con el estado BG-C.
```

Nótese cómo:
- El header `## Artículo 5` aparece en **ambos chunks** en lugar de solo el primero.
- La nota de tabla ha sido adherida al bloque de la tabla por `_attach_table_notes`.
- El chunk 2 tiene toda la información necesaria para que el LLM responda sobre la celda A01.

---

## 4. Embeddings y Vector Store

### 4.1 Modelo E5 multilingual

`intfloat/multilingual-e5-base` es un modelo de 768 dimensiones entrenado para recuperación asimétrica: los **pasajes** (chunks del corpus) se codifican con el prefijo `"passage: "` y las **queries** (preguntas del usuario) con `"query: "`. Esta asimetría permite que el modelo encuentre pasajes relevantes aunque la pregunta y el pasaje estén redactados de forma diferente.

### 4.2 Esquema de Milvus

```
Collection: rag_documents
├── id         VARCHAR(64)   PK
├── doc_slug   VARCHAR(256)
├── filename   VARCHAR(512)
├── text       VARCHAR(65535)
└── embedding  FLOAT_VECTOR(768)  ← índice IVF_FLAT, métrica COSINE
```

**Índice IVF_FLAT:** el corpus vectorial se divide en `nlist=128` clústeres. En cada búsqueda se exploran los `nprobe=16` clústeres más cercanos al vector de la query. Esto balancea velocidad y recall.

### 4.3 Overfetch y reranking

Al hacer retrieval, el sistema solicita a Milvus **5× más chunks** de los necesarios (`top_k×5`, con un mínimo de 20 y máximo de 80). Después aplica un reranker híbrido local:

```
score_final = 0.65 × score_vectorial_normalizado
            + 0.25 × cobertura_léxica   (tokens de la query en el chunk)
            + 0.10 × cobertura_anchor   (tokens largos ≥7 chars de la query)
```

El `score_vectorial_normalizado` transforma el coseno (rango -1..1) a 0..1 mediante `(coseno + 1) / 2`.

---

## 5. Retrieval: cómo se recuperan los chunks

### 5.1 Ruta técnica (Milvus)

```
Pregunta del usuario
      │
      ▼
EmbeddingService.embed_query(question)
   └─ añade prefijo "query: " al texto
   └─ genera vector de 768 dims
      │
      ▼
VectorStore.search(query_embedding, top_k=overfetch_k)
   └─ búsqueda ANN en Milvus (coseno, nprobe=16)
   └─ devuelve hasta 80 candidatos con score bruto
      │
      ▼
RAGEngine._rerank_with_lexical(question, chunks, top_k)
   └─ calcula score combinado (65% vector, 25% léxico, 10% anchor)
   └─ deduplica por fingerprint (primeros 650 chars normalizados)
   └─ selecciona los mejores top_k
      │
      ▼
Lista final de Chunk (top_k chunks con score combinado)
```

### 5.2 Ruta legal (Neo4j)

Usa el motor `GraphRAGEngine.search_units()`. No hay vectores. El proceso es:

```
Pregunta del usuario
      │
      ▼
_query_signals()
   ├─ normaliza y tokeniza la pregunta
   ├─ llama al LLM para clasificar intención:
   │    intent: generic | definition | article_lookup | article_list |
   │            comparison | yes_no | requirements | exclusion | effective_date
   ├─ extrae article_numbers, entities, topics, legal_refs, numeric_tokens
   └─ detecta flags: enumerative_need, asks_extreme, asks_modal, asks_comparison...
      │
      ▼
_candidate_rows() → Cypher query en Neo4j
   └─ calcula kw_hits, root_hits, head_kw_hits, ref_hits con REDUCE
   └─ ordena por kw_hits DESC, ref_hits DESC, position ASC
   └─ devuelve hasta 1200 candidatos
      │
      ├─── Si hay article_numbers explícitos:  _fetch_rows_by_articles()
      └─── Si no hay artículos:               _fetch_rows_by_heading_terms()
      │
      ▼
_rank_select_generic()
   ├─ _score_row(): score lexical ponderado (keywords, content_terms, roots, refs...)
   ├─ BM25 sobre los candidatos
   ├─ score_final = score_lexical + 0.45 × BM25_normalizado
   ├─ aplica bonus/penalizaciones por intent:
   │    - article_lookup: +1.20 si el artículo coincide, ×0.15 si no
   │    - exclusion: +0.42 por artículos con "excluye"
   │    - asks_extreme: +0.25 si hay números, +0.18 si hay "máximo/euros"
   │    - comparison: +0.38 si hay ≥2 entidades presentes
   ├─ deduplica por fingerprint y limita a per_article_limit chunks por artículo
   └─ devuelve top_k GraphChunks
```

---

## 6. Grafo legal: Neo4j GraphRAG

### 6.1 Estructura del grafo

```
(DocumentoNormativo)
        │
        ├── [:CONTIENE] ──► (UnidadNormativa)  ← un artículo o fragmento
        │                         │
        │                         └── [:MENTIONS_STATE] ──► (EstadoFinanciero)
        │                         └── [:IMPACTA_ESTADO]  ──► (EstadoFinanciero)
        └── título, hash (SHA-256), fecha_revision...
```

### 6.2 Ingesta legal

1. Se repara el Markdown (función `_repair_visible_text`).
2. Se calcula el SHA-256 del Markdown: si coincide con el hash almacenado → **skip** (idempotencia).
3. `_extract_units()` divide el Markdown por cabeceras de artículo (`## Artículo N` o `Artículo N` inline):
   - Ignora filas de tabla, índices de contenido (líneas con `......123`), cabeceras de dump.
   - Descarta fragmentos con menos de 80 caracteres.
   - Un artículo = una `UnidadNormativa`.
4. Para cada unidad, extrae códigos de estados financieros (`lqb`, `bg`, etc.) y crea nodos `EstadoFinanciero` vinculados.
5. Se crean índices fulltext en Neo4j para acelerar búsquedas.

### 6.3 Bridge States

Cuando el motor legal recupera unidades normativas, el orquestador ejecuta `find_bridge_states()`. Esta consulta Cypher busca si alguna de las unidades recuperadas tiene relación `MENTIONS_STATE` con nodos `EstadoFinanciero`. Si los hay, enriquece la pregunta técnica con los códigos de estado encontrados (ej: "estado lqb", "estado bg") y re-lanza la búsqueda técnica en Milvus. Esto crea un **puente** entre normativa legal y documentación técnica, útil para preguntas mixtas.

---

## 7. Router de preguntas

El router (`QueryRouter` en `routing.py`) decide la ruta con una llamada LLM:

```json
Pregunta → LLM → {
  "route": "legal" | "technical" | "both",
  "confidence": 0.0..1.0,
  "reason": "..."
}
```

**Reglas del prompt del router:**
- **`legal` por defecto** para preguntas sobre ley, artículos, CNMV, ECR, EICC, SGEIC, SGIIC, SCR, FCR, definiciones, autorizaciones, comercialización, inversiones, requisitos, sanciones o régimen jurídico.
- **`technical` solo** si la pregunta trata de cumplimentación operativa de estados/celdas/campos/códigos/XBRL/XML/manual.
- **`both`** solo si se pide expresamente mezcla legal y técnica.
- En caso de duda entre legal y technical → **elige `legal`**.

**Fallback de evidencia:** después de recuperar chunks, el orquestador evalúa la calidad de la evidencia (score mínimo, cobertura de keywords). Si la ruta inicial no tiene evidencia suficiente, lanza un **probe ligero** (`light_probe`) en el motor alternativo:
- Ruta `legal` sin evidencia → probe a Milvus técnico.
- Ruta `technical` sin evidencia → probe a Neo4j legal.

Si el probe tiene evidencia, la ruta final cambia o se amplía a `both`.

### 7.1 Clasificación de documentos para ingesta

La función `classify_document_name()` usa vocabularios de tokens para clasificar el nombre del fichero:
- **Señales legales:** `ley`, `reglamento`, `real`, `decreto`, `directiva`, `circular`, `boe`, `cnmv`...
- **Señales técnicas:** `manual`, `cumplimentacion`, `reporte`, `xbrl`, `xml`, `lqb`, `bg`...

Si un documento tiene señales de ambos tipos, se ingesta en **ambos motores** (función `is_bridge_legal_document`).

---

## 8. Orquestador Híbrido

El `HybridRAGEngine.query_debug()` es el flujo principal:

```
1. _expand_followup_question()
   └─ si la pregunta es anafórica ("¿Y eso?", "¿En ese caso?", "¿Cómo?")
      concatena la última pregunta de usuario del historial

2. QueryRouter.route_with_debug()
   └─ decide: legal | technical | both

3. Recuperación paralela según la ruta
   ├─ need_legal  → GraphRAGEngine.search_units()
   └─ need_technical → RAGEngine.retrieve()

4. Evaluación de evidencia
   ├─ _evaluate_technical_evidence(): top_score, chunk_count, keyword_coverage
   └─ _evaluate_legal_evidence(): idem para legal

5. Probe fallback (si evidencia débil)
   ├─ ruta=legal, evidencia insuficiente → light_probe() técnico
   └─ ruta=technical, evidencia insuficiente → light_probe() legal

6. Bridge States
   └─ si hay chunks legales, busca EstadoFinanciero vinculados
   └─ si encuentra bridge, enriquece la pregunta y re-busca en Milvus

7. _resolve_route()
   └─ puede promover la ruta de "legal" a "both" si ambos motores tienen evidencia

8. _build_answer()
   ├─ ruta=legal  → GraphRAGEngine.generate_from_chunks()
   ├─ ruta=technical → RAGEngine.generate_from_chunks()
   └─ ruta=both  → genera por separado y concatena bloques

9. Devuelve HybridQueryResult con answer, sources, chunks, route...
```

---

## 9. Generación: el LLM

La generación es independiente en cada motor:

### 9.1 Motor técnico (RAGEngine / LLMGenerator)

El contexto se construye enumerando los chunks:
```xml
<fragmento id="1" fuente="manual_lqb_v3.pdf" (relevancia: 87.4%)>
[texto del chunk]
</fragmento>
<fragmento id="2" fuente="manual_lqb_v3.pdf" (relevancia: 82.1%)>
...
```

El **system prompt** tiene 8 reglas estrictas:
1. Responder SOLO con información de los fragmentos (prohibido conocimiento externo).
2. Verificar que cada dato aparece explícitamente en al menos un fragmento.
3. Reproducir textos legales fielmente, nunca completar ni parafrasear.
4. Citar siempre la fuente y el número de fragmento.
5. Si la respuesta no está, responder exactamente `"NO ENCONTRADO EN EL DOCUMENTO"`.
6. Formato markdown ligero, sin inventar secciones.
7. Si hay contradicción entre fragmentos, señalarla.
8. Responder siempre en español.

### 9.2 Motor legal (GraphRAGEngine / `_compose_answer_llm`)

Similar pero con contexto más compacto (snippet de 1800 chars por chunk, máximo 8 chunks). El prompt añade:
- La **intención estimada** (`intent`) para orientar el formato de respuesta.
- Instrucciones de citación mediante `[n]` inline.
- Instrucción explícita de no repetir el contexto completo.

El historial de chat se inyecta en ambos motores (últimas 6 interacciones en el caso legal, N turnos configurables en el técnico).

---

## 10. Ejemplo paso a paso: de pregunta a respuesta

**Pregunta del usuario:**  
> "¿Cuándo debe presentarse el Estado LQ-B y cuál es la frecuencia de reporte?"

### Paso 1 — Expansión anafórica
La pregunta no es anafórica (no contiene "eso", "ese caso", "cómo"...), por lo que se usa tal cual.

### Paso 2 — Router
El LLM router analiza la pregunta. Detecta "Estado LQ-B" y palabras como "presentarse", "frecuencia", "reporte" → señales **técnicas** (cumplimentación operativa).  
Resultado: `route = "technical"`, `confidence = 0.85`.

### Paso 3 — Recuperación técnica
```
embed_query("query: ¿Cuándo debe presentarse el Estado LQ-B y cuál es la frecuencia de reporte?")
→ vector de 768 dims

Milvus.search(vector, top_k=40)   ← overfetch (8×5)
→ devuelve 40 candidatos con score coseno

_rerank_with_lexical():
  - "estado", "lqb", "presentarse", "frecuencia", "reporte" → tokens de búsqueda léxica
  - chunk con "LQ-B mensual antes del día 20" → score combinado alto
  - deduplica por fingerprint

→ devuelve top 8 chunks
```

### Paso 4 — Evaluación de evidencia técnica
```
chunk_count=8, top_score=0.88, keyword_coverage=0.72
→ evidence_ok = True (top_score ≥ threshold y coverage ≥ 0.45)
```

No se necesita probe fallback legal.

### Paso 5 — Bridge States
Se buscan `EstadoFinanciero` vinculados a los chunks legales (no hay chunks legales en esta consulta).  
Sin bridge states.

### Paso 6 — Resolución de ruta
Ruta inicial = `technical`, evidencia ok → ruta final = `technical`.

### Paso 7 — Generación
```
LLMGenerator.generate(question, top_8_chunks)
→ system_prompt (8 reglas estrictas)
→ user: <contexto>[top 8 fragmentos XML]</contexto>\nPregunta: ...
→ gpt-4o-mini responde en español citando fragmenos
```

**Respuesta generada:**  
> "Según el manual_estado_lqb_v2.pdf (fragmento 1), el Estado LQ-B debe presentarse mensualmente antes del día 20 del mes siguiente al período de referencia. La frecuencia de reporte es, por tanto, mensual."

### Paso 8 — Respuesta al usuario
```json
{
  "answer": "Según manual_estado_lqb_v2.pdf (fragmento 1), ...",
  "sources": ["manual_estado_lqb_v2.pdf"],
  "route": "technical",
  "route_reason": "operativa de cumplimentación de estado/reporte",
  "route_confidence": 0.85,
  "chunks": [...]
}
```

---

## 12. Ejemplo: cuándo se buscan vecinos y cuándo no

En el motor técnico (Milvus), **no hay búsqueda de vecinos en grafos** — solo similitud vectorial + reranking léxico.

En el motor legal (Neo4j), la "vecindad" es la relación `CONTIENE` del grafo, pero tampoco se hace una expansión de vecinos en grafo durante el retrieval. La adyacencia estructural se expresa porque **un artículo completo = una `UnidadNormativa`**, por lo que el artículo entero con sus párrafos adyacentes ya está en un solo nodo.

La funcionalidad más cercana a "búsqueda de vecinos" es el mecanismo de **Bridge States**:

### Caso en que SÍ se activa el bridge (búsqueda de vecinos entre motores)

**Pregunta:** "¿Qué normativa CNMV obliga a reportar el estado LQ-B y cómo se cumplimenta la celda A01?"

1. Router → `both` (o `legal` con probe técnico).
2. Law engine recupera unidades normativas de, por ejemplo, la Circular CNMV 4/2015 que mencionan el "Estado LQ-B".
3. `find_bridge_states()` detecta que esas unidades tienen relación `[:MENTIONS_STATE]` con el nodo `EstadoFinanciero {code: "lqb"}`.
4. Orquestador enriquece la pregunta: `"¿Qué normativa... + estado lqb"`.
5. Se re-lanza búsqueda en Milvus con la pregunta enriquecida → recupera chunks del manual técnico de LQ-B.
6. Respuesta final incluye bloques legal + técnico.

### Caso en que NO se activa el bridge

**Pregunta:** "¿Cuándo vence el plazo para presentar el Estado LQ-B?"

1. Router → `technical`.
2. Milvus recupera chunks del manual técnico con score alto.
3. `_evaluate_technical_evidence()` → `evidence_ok = True`.
4. No hay chunks legales, por tanto `find_bridge_states([])` retorna `[]` inmediatamente.
5. No se enriquece la pregunta, no hay probe.
6. Respuesta solo del motor técnico.

---

## 13. Problemas principales detectados

### 🔴 Problema 1 (crítico): Dependencia total en el LLM para el routing

**Descripción:** El `QueryRouter` está configurado con `ROUTER_USE_LLM=True` y **no tiene fallback heurístico funcional**. Aunque existen vocabularios de tokens legales/técnicos (`LEGAL_DOC_TOKENS`, `TECH_DOC_TOKENS`), el método `route_with_debug()` lanza un `RuntimeError` directamente si el LLM no está disponible:

```python
if llm is None:
    raise RuntimeError("LLM router unavailable: enable OPENAI_API_KEY and RAG_ROUTER_USE_LLM=true")
```

**Consecuencia:** si la API de OpenAI falla, está caída, o se supera el budget/rate limit, **el sistema completo deja de responder** — ni siquiera intenta un fallback léxico. La infraestructura heurística está definida (vocabularios, función `_score_tokens`) pero nunca se llama en el flujo actual.

**Impacto:** disponibilidad 0% cuando OpenAI no está accesible.

---

### 🔴 Problema 2 (crítico): El LLM de intención legal se llama dos veces por query

**Descripción:** `GraphRAGEngine._query_signals()` llama a `_intent_via_llm()` en cada invocación. Pero este método se llama en varios puntos del pipeline para la misma pregunta:
- Una vez en `search_units()` → `_query_signals()`.
- Otra vez en `_rank_select_generic()` → `_score_row()` (que llama internamente a `signals` ya calculadas → OK aquí gracias al caché `_intent_cache`).
- Otra vez en `_compose_answer_llm()` → `_query_signals()` nuevamente.
- Otra vez en `_extractive_answer_by_intent()` → `_query_signals()`.

El caché `_intent_cache` mitiga parcialmente el problema (clave = `normalized_question`), pero el caché vive en la instancia del objeto y **no persiste entre requests** en entornos multi-worker. En un servidor con múltiples workers de uvicorn/gunicorn, cada worker tiene su propio caché vacío. Resultado: **2-4 llamadas LLM extra por query** que podrían evitarse.

---

### 🟡 Problema 3 (importante): Chunking sin overlap real

**Descripción:** la configuración define `CHUNK_OVERLAP=50` tokens, pero la clase `MarkdownChunker` **no implementa el solapamiento entre chunks**. El parámetro `self.overlap` se lee pero nunca se aplica en `_split_long_section()`. Los chunks son contiguos sin solapamiento.

**Consecuencia:** cuando la información relevante está en el límite entre dos chunks consecutivos (por ejemplo, el inicio de un artículo en el chunk N describiendo condiciones que se aclaran con una excepción en el primer párrafo del chunk N+1), el retrieval puede recuperar solo uno de los dos chunks y la respuesta es incompleta.

---

### 🟡 Problema 4 (importante): Límite duro de 65.000 caracteres en Milvus

**Descripción:** en `VectorStore.insert()` el texto se trunca a 65.000 caracteres:
```python
[c.text[:65000] for c in chunks]
```

El campo `text` en Milvus está declarado como `VARCHAR(65535)`. Si un chunk (especialmente de un artículo legal muy largo) excede este límite, el texto almacenado estará truncado silenciosamente. El embedding sí se genera con el texto completo original, pero el LLM solo recibirá el texto truncado al recuperar el chunk.

---

### 🟡 Problema 5 (importante): `VectorStore.count()` usa `query()` sobre hasta 16.384 entidades

**Descripción:** el método `count()` hace:
```python
rows = collection.query(expr='id != ""', output_fields=["id"], limit=16384)
return len(rows)
```
Si la colección tiene más de 16.384 chunks, el conteo es incorrecto. La línea de fallback `int(collection.num_entities)` es más correcta pero solo se usa si falla la query.

---

### 🟠 Problema 6 (menor): Anaphora detection muy simplista

**Descripción:** `_is_anaphoric_question()` detecta preguntas de seguimiento solo con un regex muy limitado:
```python
re.search(r"^(y|entonces|en que condiciones|que condiciones|como|y eso|cual de)", q)
```

Preguntas habituales de seguimiento como "¿Y para las SCR?" o "¿Qué pasa si supera ese umbral?" o simplemente "¿Y si no?" no serían detectadas como anafóricas, por lo que no se expandirían con el contexto previo de la conversación.

---

### 🟠 Problema 7 (menor): Caché de intenciones no thread-safe en multi-worker

**Descripción:** `_intent_cache` es un `dict` en el objeto `GraphRAGEngine`. Con múltiples threads dentro del mismo worker, no hay ningún mecanismo de sincronización (`threading.Lock`) para acceder al caché. En carga alta, podrían producirse lecturas/escrituras concurrentes corruptas.

---

### Resumen de problemas

| # | Severidad | Problema | Impacto |
|---|---|---|---|
| 1 | 🔴 Crítico | Router sin fallback heurístico funcional | Caída total si OpenAI no responde |
| 2 | 🔴 Crítico | 2-4 llamadas LLM extra por query en contexto multi-worker | Latencia y coste extra |
| 3 | 🟡 Importante | Overlap de chunks no implementado | Respuestas incompletas en límites de chunk |
| 4 | 🟡 Importante | Truncado silencioso en texto > 65.000 chars en Milvus | Contexto incompleto enviado al LLM |
| 5 | 🟡 Importante | `count()` incorrecto con > 16.384 chunks | Estadísticas de monitoreo erróneas |
| 6 | 🟠 Menor | Detección de preguntas anafóricas incompleta | Seguimiento de conversación deficiente |
| 7 | 🟠 Menor | Intent cache sin lock en entorno multi-thread | Posible corrupción en alta concurrencia |

---

*Análisis generado el 12/03/2026 a partir del código fuente del workspace.*
