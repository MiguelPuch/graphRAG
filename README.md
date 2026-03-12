# Hybrid RAG Normativo (Milvus + Neo4j)

Servicio único `rag-api` con orquestación híbrida para chatbot normativo:

- `technical` -> RAG clásico en Milvus (manuales, requisitos técnicos, estados, validaciones).
- `legal` -> GraphRAG en Neo4j (leyes, reales decretos, directivas, reglamentos, circulares, acuerdos).
- `both` -> solo cuando hay evidencia mixta real.

## Arquitectura de decisión (single-first)

El enrutador de preguntas funciona en 3 fases:

1. Normalización de pregunta:
- minúsculas
- eliminación de tildes/diacríticos
- `_` y `-` convertidos a espacio
- colapso de espacios

2. Heurística estructurada (sin pesos opacos):
- señales legales (`ley`, `articulo`, `disposicion`, `obligacion`, `vigencia`, etc.)
- señales técnicas (`manual`, `estado`, `validacion`, `xml`, `columna`, etc.)
- regla mixta fuerte: `estado X` + ancla legal explícita -> candidata a `both`

3. Arbitraje LLM solo en empate:
- no fuerza `both` por defecto
- `both` se acepta cuando hay mezcla explícita

Después del routing inicial, el orquestador valida evidencia de recuperación:

- si el motor inicial devuelve evidencia fuerte, mantiene ruta single.
- si devuelve evidencia débil, hace `light probe` al otro motor.
- cambia de motor si el alternativo tiene mejor evidencia.
- usa `both` solo cuando ambos motores aportan evidencia útil.

## Clasificación documental por nombre (primera capa)

`POST /ingest target=auto` usa clasificación por filename:

- `legal`: `ley`, `real decreto`, `real decreto-ley`, `directiva`, `reglamento`, `orden`, `circular`, `resolucion`, `acuerdo`, `correccion`, `boe`, `doue`, `memorando`, `tratado`, `cifradoc`.
- `technical`: `manual`, `cumplimentacion`, `requisitos tecnicos`, `requerimientos tecnicos`, `guia de procedimientos`, `norma tecnica`, `estados financieros`, `estado M51/CR3/...`, `xml`, `xbrl`.
- `mixed/unknown`: se enruta a técnico por defecto (operativa más segura para ingestión).

## Nodos puente (respaldo legal -> técnico)

En Neo4j se crea `(:UnidadNormativa)-[:IMPACTA_ESTADO]->(:EstadoFinanciero)` solo por mención literal.

Además, documentos legales puente (p. ej. `circular`, `acuerdo`, `resolucion`, `orden`) se replican también en Milvus durante la ingesta legal para evitar huecos de recuperación en preguntas híbridas.

## Mejoras de recuperación aplicadas

### Técnico (Milvus)
- chunking con preservación de notas asociadas a tablas.
- búsqueda vectorial con `overfetch` (top_k ampliado) + reranking léxico.
- trazabilidad de score final por chunk en `metadata.retrieval_scores`.

### Legal (Neo4j)
- indexación de `texto_norm` (texto normalizado).
- búsqueda por keywords + referencias de `articulo`, `norma` y `estado`.
- scoring con desglose (`keyword/article/norma/state`) para auditoría.

## Endpoints

- `GET /health`
- `GET /stats`
- `POST /ingest` (`target=auto|legal|technical`)
- `POST /ingest-corpus`
- `POST /query`
- `POST /route-debug`
- `POST /query-debug`

## Contratos de debug

### `POST /route-debug`
Devuelve señales internas del router:
- `normalized_question`
- `heuristic_scores`
- `heuristic_decision`
- `llm_decision`
- `final_decision`

### `POST /query-debug`
Devuelve trazabilidad end-to-end:
- `route_initial`, `route_final`, `route_reason`, `route_confidence`
- `engines_used`, `bridge_fallback_used`
- `legal_trace.nodes` (nodos Neo4j recuperados)
- `technical_trace.chunks` (chunks Milvus recuperados)
- `answer` final

## Arranque rápido

```bash
docker compose -f docker-compose.hybrid.yml up -d --build
```

Comprobación:

```bash
curl http://localhost:8001/health
curl http://localhost:8001/stats
```

## Comandos de auditoría (PowerShell)

Clasificación de ruta:

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8001/route-debug -ContentType 'application/json' -Body '{"question":"Que norma regula la validacion del estado CR3?"}' | ConvertTo-Json -Depth 12
```

Trazabilidad completa (ruta, nodos, chunks, respuesta):

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8001/query-debug -ContentType 'application/json' -Body '{"question":"Que norma regula la validacion del estado CR3?","top_k":8,"debug_probe_mode":"auto","include_raw_text":false}' | ConvertTo-Json -Depth 20
```

Forzar motor para comparar:

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8001/query-debug -ContentType 'application/json' -Body '{"question":"Que dice el articulo 5 de la Ley 22/2014?","debug_probe_mode":"force_legal"}' | ConvertTo-Json -Depth 20
Invoke-RestMethod -Method Post -Uri http://localhost:8001/query-debug -ContentType 'application/json' -Body '{"question":"Como se rellena el estado M51?","debug_probe_mode":"force_technical"}' | ConvertTo-Json -Depth 20
```

Verificar nodos puente literales:

```powershell
docker exec neo4j cypher-shell -u neo4j -p neo4jpassword "MATCH ()-[r:IMPACTA_ESTADO]->() RETURN count(r) AS impacta_estado;"
```

## Notebook de operación

Notebook disponible en `notebooks/hybrid_orchestrator_debug.ipynb`:

- convierte PDFs de `data/documents/nodosPuentes`
- ingesta automática
- debug interactivo por pregunta
- batch desde `questions.txt`
- export opcional de reporte JSON

## Tests

```bash
pytest -q
```

Si quieres validar solo routing/orquestación:

```bash
pytest tests/test_routing.py tests/test_hybrid_orchestration.py -q
```
