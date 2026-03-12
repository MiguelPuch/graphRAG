# Hybrid RAG Normativo (Milvus + Neo4j)

Servicio único `rag-api` con orquestación híbrida para chatbot normativo sobre regulación CNMV.

- `technical` → RAG clásico en Milvus (manuales, requisitos técnicos, estados, validaciones XBRL/XML).
- `legal` → GraphRAG en Neo4j (leyes, reales decretos, directivas, reglamentos, circulares, acuerdos).
- `both` → solo cuando hay evidencia mixta real en ambos motores.

---

## Índice

1. [Arquitectura del sistema](#arquitectura-del-sistema)
2. [Componentes técnicos](#componentes-técnicos)
3. [Pipeline de ingesta y chunking](#pipeline-de-ingesta-y-chunking)
4. [Embeddings y Vector Store](#embeddings-y-vector-store)
5. [Retrieval](#retrieval)
6. [Grafo legal: Neo4j GraphRAG](#grafo-legal-neo4j-graphrag)
7. [Router de preguntas](#router-de-preguntas)
8. [Orquestador híbrido](#orquestador-híbrido)
9. [Generación LLM](#generación-llm)
10. [Ejemplo paso a paso](#ejemplo-paso-a-paso)
11. [Endpoints y contratos de debug](#endpoints-y-contratos-de-debug)
12. [Arranque rápido](#arranque-rápido)
13. [Comandos de auditoría](#comandos-de-auditoría)
14. [Tests](#tests)

---

## Arquitectura del sistema

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
                          └────────┼─────────────────┼──────────────┘
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

### Dos motores, dos bases de datos

| Motor | Base de datos | Tipo de documentos | Tipo de búsqueda |
|---|---|---|---|
| `RAGEngine` | **Milvus** (vectorial) | Manuales XBRL, plantillas, formularios de estados financieros | Similitud semántica (coseno) + reranking léxico |
| `GraphRAGEngine` | **Neo4j** (grafo) | Leyes, Reglamentos, Decretos, Circulares CNMV, BOE | Keyword scoring + BM25 + grafo de artículos |

---

## Componentes técnicos

### Configuración (`config.py`)

Todos los parámetros se leen de variables de entorno:

| Parámetro | Variable | Por defecto | Descripción |
|---|---|---|---|
| Modelo LLM | `RAG_LLM_MODEL` | `gpt-4o-mini` | Modelo de generación |
| Temperatura LLM | `RAG_LLM_TEMPERATURE` | `0.1` | Creatividad del LLM |
| Modelo embeddings | `RAG_EMBED_MODEL` | `intfloat/multilingual-e5-base` | 768 dimensiones |
| Max tokens/chunk | `RAG_CHUNK_MAX_TOKENS` | `530` | Tamaño máximo por fragmento |
| Overlap chunks | `RAG_CHUNK_OVERLAP` | `50` | Tokens de solapamiento |
| URI Milvus | `MILVUS_URI` | `http://milvus:19530` | Base de datos vectorial |
| URI Neo4j | `NEO4J_URI` | `bolt://neo4j:7687` | Base de datos de grafo |
| Modelo router | `RAG_ROUTER_MODEL` | igual que LLM | Clasifica preguntas |

### `DocumentConverter` — PDF/DOCX → Markdown

- Usa **Docling** para convertir PDFs a Markdown estructurado.
- Si el fichero ya es `.md` o `.markdown`, lo lee directamente sin conversión.
- Genera un **slug único** por documento: `nombre_limpio_<hash_md5_8chars>`.
- Ingesta **idempotente**: antes de insertar en Milvus borra los chunks anteriores del mismo slug.

### `EmbeddingService` — Vectorización

- Modelo `intfloat/multilingual-e5-base` (768 dimensiones, multilingüe).
- Pasajes se codifican con prefijo `"passage: "` y queries con `"query: "` (asimetría del modelo E5).
- Inicialización **lazy**: el modelo solo se carga la primera vez que se necesita.

### `VectorStore` — Milvus

- Colección `rag_documents`, índice **IVF_FLAT / COSINE**, `nlist=128`, `nprobe=16`.
- Campos: `id (PK)`, `doc_slug`, `filename`, `text (VARCHAR 65535)`, `embedding (FLOAT_VECTOR 768)`.

### `LLMGenerator` — OpenAI

- Modelo `gpt-4o-mini`, temperatura `0.1`.
- Contexto empaquetado en etiquetas `<fragmento id="N" fuente="fichero" relevancia="XX%">`.
- System prompt con **8 reglas estrictas**: solo información de fragmentos, sin conocimiento externo, citar siempre la fuente, responder `"NO ENCONTRADO EN EL DOCUMENTO"` si no hay evidencia.

---

## Pipeline de ingesta y chunking

### Flujo técnico (Milvus)

```
Fichero PDF/DOCX/MD
        │
        ▼
DocumentConverter.convert_to_markdown()    ← Docling convierte a Markdown
        │
        ▼
MarkdownChunker.chunk()
        ├─── _attach_table_notes()          ← adhiere notas de tablas al bloque tabla
        ├─── _split_by_headers()            ← divide por cabeceras (# ## ###)
        │
        └─── Para cada sección:
              ├─── Si tokens ≤ 530  →  chunk directo
              └─── Si tokens > 530  →  _split_long_section()
                       ├─ divide por párrafos (\n\n)
                       ├─ si párrafo > budget: divide por líneas (\n)
                       └─ prefixa el header de sección a CADA sub-chunk
        │
        ▼
EmbeddingService.embed_texts()             ← prefija "passage: " a cada texto
        │
        ▼
VectorStore.insert()                       ← inserta en Milvus
```

### Estrategia de chunking en detalle

El chunker opera en 4 pasos:

**1 — Pre-procesado:** elimina artefactos de conversión PDF (`<!--pagebreak-->`). `_attach_table_notes` adhiere notas aclaratorias (como `"Nota a): ..."` o `"Donde dice:"`) al bloque de tabla para que no queden separadas.

**2 — Split por headers:** divide el Markdown por el patrón `^#{1,6}\s+.+$`. Cada sección delimitada por un header es una unidad de trabajo.

**3 — Split por longitud:** si una sección supera los 530 tokens:
1. Se extrae el header y se calcula el presupuesto: `budget = 530 − tokens(header)`.
2. Se divide por párrafos (`\n\n`); párrafos que solos excedan el presupuesto se dividen por líneas (`\n`).
3. Se agrupan piezas hasta llenar el presupuesto.
4. El header se **prefixa a CADA sub-chunk** para preservar el contexto semántico.

**4 — Deduplicación:** en el retrieval se usa un fingerprint de los primeros 650 chars normalizados para evitar chunks casi idénticos.

### Ejemplo de chunks generados

Entrada Markdown:
```markdown
## Artículo 5 — Cumplimentación del Estado LQ-B

El estado LQ-B recoge la posición de liquidez de la entidad gestora.
Debe cumplimentarse mensualmente antes del día 20 del mes siguiente.

### Celda A01 — Activos líquidos de nivel 1

Los activos líquidos de nivel 1 incluyen:
- Efectivo y reservas en bancos centrales
- Títulos de deuda emitida o garantizada por administraciones centrales

| Código | Descripción               | Valor mínimo |
|--------|---------------------------|--------------|
| A01.1  | Efectivo                  | 0            |
| A01.2  | Reservas banco central    | 0            |
Nota: Los importes de A01.1 y A01.2 deben coincidir con el estado BG-C.
```

**Chunk 1** (≈ 80 tokens):
```
## Artículo 5 — Cumplimentación del Estado LQ-B

El estado LQ-B recoge la posición de liquidez de la entidad gestora.
Debe cumplimentarse mensualmente antes del día 20 del mes siguiente.
```

**Chunk 2** (≈ 180 tokens — header propagado + tabla con nota adherida):
```
## Artículo 5 — Cumplimentación del Estado LQ-B

### Celda A01 — Activos líquidos de nivel 1

Los activos líquidos de nivel 1 incluyen:
- Efectivo y reservas en bancos centrales
- Títulos de deuda emitida o garantizada por administraciones centrales

| Código | Descripción               | Valor mínimo |
|--------|---------------------------|--------------|
| A01.1  | Efectivo                  | 0            |
| A01.2  | Reservas banco central    | 0            |
Notas asociadas a la tabla:
- Nota: Los importes de A01.1 y A01.2 deben coincidir con el estado BG-C.
```

### Flujo legal (Neo4j)

```
Fichero PDF/DOCX/MD
        │
        ▼
DocumentConverter.convert_to_markdown()
        │
        ▼
GraphRAGEngine._extract_units()
   ├─ divide por cabeceras "## Artículo N" o "Artículo N" inline
   ├─ descarta tablas, índices de contenido, fragmentos < 80 chars
   └─ un artículo = una UnidadNormativa
        │
        ▼
Neo4j: MERGE (DocumentoNormativo)-[:CONTIENE]->(UnidadNormativa)
        │
        ▼
Para cada unidad: extrae códigos de estados (lqb, bg...)
   └─ MERGE (UnidadNormativa)-[:MENTIONS_STATE]->(EstadoFinanciero)
```

Ingesta **idempotente por SHA-256**: si el hash del Markdown no cambió, se salta la re-indexación.

---

## Embeddings y Vector Store

### Modelo E5 multilingual

`intfloat/multilingual-e5-base` es un modelo asimétrico de 768 dimensiones. Los **pasajes** del corpus llevan el prefijo `"passage: "` y las **queries** llevan `"query: "`. Esta asimetría permite encontrar fragmentos relevantes aunque la pregunta y el pasaje usen vocabulario diferente.

### Overfetch y reranking

Milvus recupera **5× más chunks** de los necesarios (mín. 20, máx. 80) y luego se aplica un reranker híbrido local:

```
score_final = 0.65 × score_vectorial_normalizado
            + 0.25 × cobertura_léxica   (tokens de la query en el chunk)
            + 0.10 × cobertura_anchor   (tokens largos ≥7 chars)
```

`score_vectorial_normalizado = (coseno + 1) / 2` para transformar el rango -1..1 a 0..1.

---

## Retrieval

### Ruta técnica (Milvus)

```
Pregunta
   │
   ▼
embed_query("query: <pregunta>")         → vector 768 dims
   │
   ▼
Milvus.search(vector, top_k=overfetch)   → hasta 80 candidatos (coseno, nprobe=16)
   │
   ▼
_rerank_with_lexical()
   ├─ score combinado: 65% vector + 25% léxico + 10% anchor
   ├─ deduplica por fingerprint (primeros 650 chars normalizados)
   └─ devuelve top_k chunks con score final
```

### Ruta legal (Neo4j)

No hay vectores. El proceso es puramente léxico + grafo:

```
Pregunta
   │
   ▼
_query_signals()
   ├─ LLM clasifica intención:
   │    generic | definition | article_lookup | article_list |
   │    comparison | yes_no | requirements | exclusion | effective_date
   └─ extrae: article_numbers, entities, topics, legal_refs, numeric_tokens
   │
   ▼
_candidate_rows() — Cypher en Neo4j
   └─ kw_hits, root_hits, head_kw_hits, ref_hits con REDUCE
   └─ hasta 1200 candidatos ordenados por relevancia léxica
   │
   ├─── article_numbers explícitos → _fetch_rows_by_articles()
   └─── sin artículos              → _fetch_rows_by_heading_terms()
   │
   ▼
_rank_select_generic()
   ├─ _score_row(): 0.30×keywords + 0.24×content + 0.20×roots + 0.14×long_terms + ...
   ├─ BM25 sobre los candidatos
   ├─ score_final = score_lexical + 0.45 × BM25_normalizado
   ├─ bonus/penalizaciones por intent:
   │    article_lookup: +1.20 si artículo coincide, ×0.15 si no
   │    exclusion:      +0.42 si el artículo contiene "excluye"
   │    asks_extreme:   +0.25 si hay cifras, +0.18 si hay "máximo/euros"
   │    comparison:     +0.38 si hay ≥2 entidades presentes
   └─ top_k GraphChunks
```

---

## Grafo legal: Neo4j GraphRAG

### Estructura del grafo

```
(DocumentoNormativo)
        │
        ├── [:CONTIENE] ──► (UnidadNormativa)   ← artículo o fragmento
        │                         │
        │                         ├── [:MENTIONS_STATE] ──► (EstadoFinanciero)
        │                         └── [:IMPACTA_ESTADO]  ──► (EstadoFinanciero)
        └── titulo, hash SHA-256, fecha_revision
```

### Nodos puente (Bridge States)

Cuando el motor legal recupera unidades normativas, `find_bridge_states()` busca si alguna tiene relación `[:MENTIONS_STATE]` con nodos `EstadoFinanciero`. Si los hay, el orquestador **enriquece la pregunta** con los códigos de estado (p.ej. `"estado lqb"`) y relanza la búsqueda en Milvus. Esto crea un puente legal → técnico para preguntas mixtas.

Además, documentos legales puente (`circular`, `acuerdo`, `resolucion`, `orden`) se replican también en Milvus durante la ingesta para evitar huecos de recuperación.

---

## Router de preguntas

El `QueryRouter` decide la ruta en dos fases:

**Fase 1 — Normalización:** minúsculas, eliminación de tildes/diacríticos, `_` y `-` a espacio, colapso de espacios.

**Fase 2 — Arbitraje LLM:**

```json
Pregunta → LLM → {
  "route": "legal" | "technical" | "both",
  "confidence": 0.0..1.0,
  "reason": "..."
}
```

Reglas del prompt:
- **`legal` por defecto** para preguntas sobre ley, artículos, CNMV, ECR, EICC, SGEIC, SGIIC, SCR, FCR, definiciones, autorizaciones, comercialización, inversiones, requisitos, sanciones o régimen jurídico.
- **`technical` solo** si la pregunta trata de cumplimentación operativa de estados/celdas/campos/códigos/XBRL/XML/manual.
- **`both`** solo si se pide expresamente mezcla legal y técnica.
- En caso de duda entre legal y technical → **elige `legal`**.

### Clasificación documental para ingesta (`POST /ingest target=auto`)

La función `classify_document_name()` clasifica el fichero por su nombre:

- **`legal`**: `ley`, `real decreto`, `directiva`, `reglamento`, `orden`, `circular`, `resolucion`, `acuerdo`, `boe`, `doue`, `cnmv`...
- **`technical`**: `manual`, `cumplimentacion`, `estado M51/CR3/...`, `xml`, `xbrl`, `requisitos tecnicos`...
- **`mixed/unknown`**: se enruta a técnico por defecto.

Si un documento tiene señales de ambos tipos (`is_bridge_legal_document`), se ingesta en **ambos motores**.

---

## Orquestador híbrido

`HybridRAGEngine.query_debug()` es el flujo principal de cada consulta:

```
1. _expand_followup_question()
   └─ si la pregunta es anafórica ("¿Y eso?", "¿En ese caso?")
      concatena la última pregunta del historial de chat

2. QueryRouter.route_with_debug()
   └─ decide: legal | technical | both

3. Recuperación según la ruta
   ├─ need_legal     → GraphRAGEngine.search_units()
   └─ need_technical → RAGEngine.retrieve()

4. Evaluación de evidencia
   ├─ _evaluate_technical_evidence(): top_score, chunk_count, keyword_coverage
   └─ _evaluate_legal_evidence():     idem para legal

5. Probe fallback (si evidencia débil)
   ├─ ruta=legal,      evidencia insuficiente → light_probe() en Milvus
   └─ ruta=technical,  evidencia insuficiente → light_probe() en Neo4j

6. Bridge States
   └─ busca EstadoFinanciero vinculados a las unidades legales recuperadas
   └─ si encuentra bridge, enriquece la pregunta y re-busca en Milvus

7. _resolve_route()
   └─ puede promover la ruta de "legal" a "both" si ambos motores tienen evidencia

8. _build_answer()
   ├─ ruta=legal     → GraphRAGEngine.generate_from_chunks()
   ├─ ruta=technical → RAGEngine.generate_from_chunks()
   └─ ruta=both      → genera en ambos motores y concatena bloques

9. Devuelve HybridQueryResult: answer, sources, chunks, route, confidence
```

### Ejemplo: bridge states activo vs. inactivo

**Bridge SÍ activo** — Pregunta: *"¿Qué normativa CNMV obliga a reportar el estado LQ-B y cómo se cumplimenta la celda A01?"*

1. Router → `both`.
2. Neo4j recupera la Circular CNMV que menciona el "Estado LQ-B".
3. `find_bridge_states()` detecta relación `[:MENTIONS_STATE]` → nodo `EstadoFinanciero {code: "lqb"}`.
4. Orquestador enriquece la pregunta con `"estado lqb"` y relanza búsqueda en Milvus.
5. Respuesta final: bloque legal + bloque técnico.

**Bridge NO activo** — Pregunta: *"¿Cuándo vence el plazo para presentar el Estado LQ-B?"*

1. Router → `technical`.
2. Milvus devuelve chunks con score alto. `evidence_ok = True`.
3. No hay chunks legales → `find_bridge_states([])` retorna `[]` de inmediato.
4. Respuesta solo del motor técnico.

---

## Generación LLM

### Motor técnico

Contexto en etiquetas XML numeradas:
```xml
<fragmento id="1" fuente="manual_lqb_v3.pdf" (relevancia: 87.4%)>
  [texto del chunk]
</fragmento>
```

System prompt con 8 reglas: solo información de fragmentos · verificar datos antes de incluirlos · reproducir textos legales fielmente · citar siempre la fuente · responder `"NO ENCONTRADO EN EL DOCUMENTO"` si no hay evidencia · formato markdown ligero · señalar contradicciones · responder en español.

### Motor legal

Mismo esquema pero con snippets de hasta 1.800 chars por chunk (máximo 8 chunks). El prompt añade la **intención estimada** (`intent`) para orientar el formato de respuesta y usa citaciones inline `[n]`.

El historial de conversación se inyecta en ambos motores para soporte de preguntas de seguimiento.

---

## Ejemplo paso a paso

**Pregunta:** *"¿Cuándo debe presentarse el Estado LQ-B y cuál es la frecuencia de reporte?"*

| Paso | Acción | Resultado |
|---|---|---|
| 1 | Expansión anafórica | No aplica (pregunta directa) |
| 2 | Router LLM | `route=technical`, `confidence=0.85` |
| 3 | `embed_query` → Milvus overfetch | 40 candidatos recuperados |
| 4 | `_rerank_with_lexical` | top 8 chunks, top_score=0.88 |
| 5 | Evaluación evidencia | `evidence_ok=True` (score ≥ threshold, coverage=0.72) |
| 6 | Bridge States | No hay chunks legales → sin bridge |
| 7 | Resolución ruta | `route_final=technical` |
| 8 | `LLMGenerator.generate` | gpt-4o-mini responde con cita de fuente |

**Respuesta:**
> "Según manual_estado_lqb_v2.pdf (fragmento 1), el Estado LQ-B debe presentarse mensualmente antes del día 20 del mes siguiente al período de referencia."

---

## Endpoints y contratos de debug

### Endpoints disponibles

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/health` | Estado del servicio |
| `GET` | `/stats` | Estadísticas (chunks indexados, métricas) |
| `POST` | `/ingest` | Ingesta un documento (`target=auto\|legal\|technical`) |
| `POST` | `/ingest-corpus` | Ingesta una carpeta completa |
| `POST` | `/query` | Consulta estándar |
| `POST` | `/route-debug` | Debug del router |
| `POST` | `/query-debug` | Trazabilidad end-to-end |

### `POST /route-debug`

```json
{
  "normalized_question": "...",
  "heuristic_scores": {},
  "heuristic_decision": {},
  "llm_decision": { "route": "legal", "confidence": 0.9, "reason": "..." },
  "final_decision": { "route": "legal", "confidence": 0.9 }
}
```

### `POST /query-debug`

```json
{
  "answer": "...",
  "route_initial": "legal",
  "route_final": "both",
  "route_reason": "...",
  "route_confidence": 0.9,
  "engines_used": ["legal", "technical"],
  "bridge_fallback_used": true,
  "legal_trace": {
    "nodes": [...],
    "retrieval_strategies": {},
    "query_signals": {}
  },
  "technical_trace": {
    "chunks": [...],
    "evaluation": {}
  },
  "sources": [...]
}
```

Parámetro `debug_probe_mode`: `auto` | `force_legal` | `force_technical` | `force_both`.

---

## Arranque rápido

```bash
docker compose -f docker-compose.hybrid.yml up -d --build
```

Comprobación:

```bash
curl http://localhost:8001/health
curl http://localhost:8001/stats
```

---

## Comandos de auditoría

### PowerShell — Clasificación de ruta

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8001/route-debug `
  -ContentType 'application/json' `
  -Body '{"question":"Que norma regula la validacion del estado CR3?"}' |
  ConvertTo-Json -Depth 12
```

### PowerShell — Trazabilidad completa

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8001/query-debug `
  -ContentType 'application/json' `
  -Body '{"question":"Que norma regula la validacion del estado CR3?","top_k":8,"debug_probe_mode":"auto","include_raw_text":false}' |
  ConvertTo-Json -Depth 20
```

### PowerShell — Forzar motor para comparar

```powershell
# Forzar motor legal
Invoke-RestMethod -Method Post -Uri http://localhost:8001/query-debug `
  -ContentType 'application/json' `
  -Body '{"question":"Que dice el articulo 5 de la Ley 22/2014?","debug_probe_mode":"force_legal"}' |
  ConvertTo-Json -Depth 20

# Forzar motor técnico
Invoke-RestMethod -Method Post -Uri http://localhost:8001/query-debug `
  -ContentType 'application/json' `
  -Body '{"question":"Como se rellena el estado M51?","debug_probe_mode":"force_technical"}' |
  ConvertTo-Json -Depth 20
```

### Neo4j — Verificar nodos puente

```powershell
docker exec neo4j cypher-shell -u neo4j -p neo4jpassword `
  "MATCH ()-[r:IMPACTA_ESTADO]->() RETURN count(r) AS impacta_estado;"
```

### Batch de preguntas

```powershell
python ask_questions.py `
  --input questions.txt `
  --output data/output.json `
  --base-url http://localhost:8001 `
  --endpoint query-debug `
  --top-k 8 `
  --timeout 180 `
  --carry-history `
  --history-turns 4 `
  --history-mode anaphoric
```

---
