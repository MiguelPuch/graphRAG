"""Compact GraphRAG engine for legal/regulatory corpus in Neo4j."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import (
    LEGAL_EVIDENCE_MAX_ITEMS,
    LEGAL_INTENT_MODEL,
    LEGAL_REQUIRE_NUMERIC_OVERLAP_FOR_NUMERIC_QUERY,
    LEGAL_SNIPPET_MAX_CHARS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    NEO4J_CONNECT_TIMEOUT_SECONDS,
    NEO4J_DATABASE,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
    OPENAI_API_KEY,
    STARTUP_MAX_RETRIES,
    STARTUP_RETRY_INTERVAL_SECONDS,
)
from graph_text_utils import (
    ARTICLE_REF_PATTERN,
    LEGAL_ENTITY_TOKENS,
    LEGAL_REF_PATTERN,
    LEGAL_STOPWORDS,
    _dedupe,
    _extract_state_codes,
    _normalize_article_number,
    _normalize_for_search,
    _repair_visible_text,
    _tokens,
    looks_like_heading_dump,
)

logger = logging.getLogger(__name__)

NUMERIC_TOKEN_PATTERN = re.compile(r"\b\d+(?:[.,]\d+)?%?\b")
ARTICLE_HEADING_MD_PATTERN = re.compile(
    r"^\s*##\s*articulo\s+(\d+(?:\s*(?:bis|ter|quater|quinquies|sexies|septies|octies|nonies|decies|undecies|duodecies))?)\b",
    flags=re.IGNORECASE,
)
ARTICLE_HEADING_INLINE_PATTERN = re.compile(
    r"^\s*articulo\s+(\d+(?:\s*(?:bis|ter|quater|quinquies|sexies|septies|octies|nonies|decies|undecies|duodecies))?)\b",
    flags=re.IGNORECASE,
)
GENERIC_QUERY_TERMS = {
    "que",
    "cual",
    "cuales",
    "como",
    "donde",
    "cuando",
    "segun",
    "ley",
    "norma",
    "articulo",
    "articulos",
    "regula",
    "regulan",
    "establece",
    "diferencia",
    "frente",
    "sobre",
    "entre",
    "puede",
    "pueden",
    "debe",
    "deben",
    "permite",
    "entidad",
    "entidades",
    "sociedad",
    "sociedades",
}
PROTECTED_CONTENT_TERMS = {
    "ecr",
    "eicc",
    "sgeic",
    "sgiic",
    "scr",
    "fcr",
    "capital",
    "riesgo",
    "gestora",
    "gestoras",
    "inversores",
    "obligaciones",
    "informacion",
    "autorizacion",
    "requisitos",
    "funciones",
    "comercializacion",
    "sancionador",
    "diversificacion",
    "inmobiliarias",
    "definicion",
}


@dataclass
class GraphChunk:
    id: str
    text: str
    source: str
    score: float | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class GraphRAGResult:
    answer: str
    chunks: list[GraphChunk]
    sources: list[str]


class GraphRAGEngine:
    """Generic, low-bias legal GraphRAG over Neo4j."""

    def __init__(self):
        from neo4j import GraphDatabase

        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
            connection_timeout=NEO4J_CONNECT_TIMEOUT_SECONDS,
        )
        self._llm = None
        self._intent_cache: dict[str, dict[str, Any]] = {}
        self._last_search_debug: dict[str, Any] = {}
        self._last_generation_debug: dict[str, Any] = {}
        self._wait_for_neo4j()
        self._ensure_schema()

    def close(self) -> None:
        self.driver.close()

    def _wait_for_neo4j(self) -> None:
        for attempt in range(1, STARTUP_MAX_RETRIES + 1):
            try:
                self.driver.verify_connectivity()
                return
            except Exception as exc:
                if attempt == STARTUP_MAX_RETRIES:
                    raise RuntimeError(f"Neo4j unavailable after {attempt} attempts: {exc}") from exc
                time.sleep(STARTUP_RETRY_INTERVAL_SECONDS)

    def _ensure_schema(self) -> None:
        statements = [
            "CREATE CONSTRAINT doc_norm_id IF NOT EXISTS FOR (d:DocumentoNormativo) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT unidad_norm_id IF NOT EXISTS FOR (u:UnidadNormativa) REQUIRE u.id IS UNIQUE",
            "CREATE CONSTRAINT estado_fin_id IF NOT EXISTS FOR (s:EstadoFinanciero) REQUIRE s.id IS UNIQUE",
            "CREATE INDEX unidad_text_norm IF NOT EXISTS FOR (u:UnidadNormativa) ON (u.text_norm)",
            "CREATE INDEX unidad_texto_norm IF NOT EXISTS FOR (u:UnidadNormativa) ON (u.texto_norm)",
            "CREATE INDEX unidad_article IF NOT EXISTS FOR (u:UnidadNormativa) ON (u.article)",
            "CREATE INDEX unidad_numero IF NOT EXISTS FOR (u:UnidadNormativa) ON (u.numero)",
            "CREATE INDEX doc_titulo_norm IF NOT EXISTS FOR (d:DocumentoNormativo) ON (d.titulo_norm)",
        ]
        with self.driver.session(database=NEO4J_DATABASE) as session:
            for stmt in statements:
                try:
                    session.run(stmt)
                except Exception as exc:
                    # Neo4j can raise EquivalentSchemaRuleAlreadyExists when schema exists with another name.
                    msg = str(exc)
                    if "EquivalentSchemaRuleAlreadyExists" not in msg and "already exists" not in msg.lower():
                        raise
            # Optional: fulltext index (best effort).
            for stmt in [
                "CREATE FULLTEXT INDEX unidad_text_ft IF NOT EXISTS FOR (u:UnidadNormativa) ON EACH [u.text, u.text_norm]",
                "CREATE FULLTEXT INDEX unidad_texto_ft IF NOT EXISTS FOR (u:UnidadNormativa) ON EACH [u.texto, u.texto_norm]",
            ]:
                try:
                    session.run(stmt)
                except Exception:
                    pass

    def _get_llm(self):
        if self._llm is not None:
            return self._llm
        if not OPENAI_API_KEY:
            return None
        from openai import OpenAI

        self._llm = OpenAI(api_key=OPENAI_API_KEY)
        return self._llm

    def _intent_via_llm(self, question: str, normalized_question: str, article_numbers: list[str]) -> dict[str, Any]:
        cache_key = normalized_question
        if cache_key in self._intent_cache:
            return dict(self._intent_cache[cache_key])

        client = self._get_llm()
        if client is None:
            raise RuntimeError("LLM intent classifier unavailable: OPENAI_API_KEY is required")

        system = (
            "Clasifica consultas juridicas para RAG. Responde SOLO JSON con claves: "
            "intent (generic|definition|article_lookup|article_list|comparison|yes_no|requirements|exclusion|effective_date), "
            "article_numbers (array de strings), entities (array), topics (array), confidence (0..1)."
        )
        user = (
            f"Pregunta: {question}\n"
            "No inventes articulos. Si no hay articulo explicito, deja article_numbers vacio. "
            "En comparison usa comparison; en preguntas 'que articulos regulan' usa article_list."
        )

        try:
            rsp = client.chat.completions.create(
                model=LEGAL_INTENT_MODEL,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
            )
            raw = rsp.choices[0].message.content or "{}"
            parsed = json.loads(raw)
            intent = str(parsed.get("intent") or "generic").strip().lower()
            if intent not in {
                "generic",
                "definition",
                "article_lookup",
                "article_list",
                "comparison",
                "yes_no",
                "requirements",
                "exclusion",
                "effective_date",
            }:
                intent = "generic"
            llm_articles = [_normalize_article_number(a) for a in (parsed.get("article_numbers") or []) if a]
            merged_articles = _dedupe([a for a in (article_numbers + llm_articles) if a])
            entities = [str(e).strip().lower() for e in (parsed.get("entities") or []) if str(e).strip()]
            topics = [str(t).strip().lower() for t in (parsed.get("topics") or []) if str(t).strip()]
            confidence = float(parsed.get("confidence") or 0.6)
            data = {
                "intent": intent,
                "article_numbers": merged_articles,
                "entities": entities[:12],
                "topics": topics[:12],
                "confidence": max(0.0, min(confidence, 1.0)),
                "source": "llm",
            }
        except Exception as exc:
            raise RuntimeError(f"LLM intent classifier failed: {exc}") from exc

        self._intent_cache[cache_key] = dict(data)
        return data

    def _slug(self, filename: str) -> str:
        base = Path(filename).stem.lower()
        base = re.sub(r"[^a-z0-9_\- ]", "", base)
        base = re.sub(r"[\s_]+", "_", base).strip("_")
        suffix = hashlib.md5(filename.encode("utf-8")).hexdigest()[:8]
        return f"{base}_{suffix}"

    def _repair_markdown_preserve_structure(self, markdown: str) -> str:
        if not markdown:
            return ""
        lines = [_repair_visible_text(line) for line in str(markdown).splitlines()]
        return "\n".join(lines)

    def _extract_units(self, markdown: str) -> list[dict[str, Any]]:
        raw_lines = [ln.rstrip() for ln in str(markdown or "").splitlines()]

        # Remove OCR/table of contents noise but keep article headers and body content.
        cleaned_lines: list[str] = []
        for line in raw_lines:
            stripped = line.strip()
            if not stripped:
                cleaned_lines.append("")
                continue
            if stripped.startswith("<!--"):
                continue
            if stripped.startswith("|"):
                continue
            if re.search(r"\.{8,}\s*\d+\s*$", _normalize_for_search(stripped)):
                continue
            cleaned_lines.append(stripped)

        # Build article sections first; fallback to paragraph blocks if no article heading exists.
        sections: list[dict[str, Any]] = []
        current_heading = ""
        current_article: str | None = None
        current_lines: list[str] = []

        def flush_current() -> None:
            if not current_lines:
                return
            body = "\n".join(current_lines).strip()
            if not body:
                return
            text = f"{current_heading}\n{body}".strip() if current_heading else body
            sections.append({"article": current_article, "text": text})

        for line in cleaned_lines:
            norm = _normalize_for_search(line)
            if not norm:
                if current_lines and current_lines[-1] != "":
                    current_lines.append("")
                continue
            if looks_like_heading_dump(line) and not norm.startswith("## articulo"):
                continue

            match = ARTICLE_HEADING_MD_PATTERN.match(norm) or ARTICLE_HEADING_INLINE_PATTERN.match(norm)
            if match and "|" not in line:
                flush_current()
                current_article = _normalize_article_number(match.group(1))
                current_heading = line
                current_lines = []
                continue

            if current_article is None:
                # Keep short preamble units only when they look meaningful.
                if len(norm) > 30 and not re.fullmatch(r"[\-\.\s0-9]+", norm):
                    current_lines.append(line)
                continue

            current_lines.append(line)

        flush_current()

        if not sections:
            blocks = [b.strip() for b in re.split(r"\n\s*\n", "\n".join(cleaned_lines)) if b.strip()]
            sections = [{"article": None, "text": b} for b in blocks]

        units: list[dict[str, Any]] = []
        idx = 0
        for section in sections:
            article = section.get("article")
            section_text = str(section.get("text") or "").strip()
            if not section_text:
                continue
            if section_text.count("|") >= 2:
                continue
            section_text = re.sub(r"[ \t]+", " ", section_text).strip()
            section_text = re.sub(r"\n{3,}", "\n\n", section_text)
            section_norm = _normalize_for_search(section_text)
            if not section_norm:
                continue
            if re.search(r"\.{8,}\s*\d+\s*$", section_norm):
                continue
            if len(section_text) < 80:
                continue

            # Keep legal context intact: one article/section == one retrieval unit.
            idx += 1
            units.append(
                {
                    "id": str(idx),
                    "text": section_text,
                    "text_norm": section_norm,
                    "article": article,
                    "position": idx,
                }
            )

        return units

    def _get_document_hash(self, doc_id: str) -> str | None:
        with self.driver.session(database=NEO4J_DATABASE) as session:
            row = session.run(
                "MATCH (d:DocumentoNormativo {id: $id}) RETURN d.hash AS h",
                {"id": doc_id},
            ).single()
            return row["h"] if row and row.get("h") else None

    def ingest_markdown(self, markdown: str, filename: str, source_url: str | None = None) -> dict[str, Any]:
        markdown = self._repair_markdown_preserve_structure(markdown)
        doc_id = self._slug(filename)
        doc_hash = hashlib.sha256(markdown.encode("utf-8")).hexdigest()
        units = self._extract_units(markdown)
        titulo_norm = _normalize_for_search(filename)

        if self._get_document_hash(doc_id) == doc_hash:
            return {
                "status": "skipped",
                "doc_id": doc_id,
                "filename": filename,
                "reason": "same_hash",
                "units": 0,
            }

        with self.driver.session(database=NEO4J_DATABASE) as session:
            session.run(
                """
                MERGE (d:DocumentoNormativo {id: $doc_id})
                SET d.filename = $filename,
                    d.titulo = $filename,
                    d.titulo_norm = $titulo_norm,
                    d.hash = $doc_hash,
                    d.url = $source_url,
                    d.source_url = $source_url,
                    d.fecha_revision = timestamp(),
                    d.updated_at = timestamp()
                """,
                {
                    "doc_id": doc_id,
                    "filename": filename,
                    "titulo_norm": titulo_norm,
                    "doc_hash": doc_hash,
                    "source_url": source_url,
                },
            )
            session.run(
                """
                MATCH (d:DocumentoNormativo {id: $doc_id})-[r:CONTIENE|HAS_UNIT]->(u:UnidadNormativa)
                DELETE r
                WITH u
                OPTIONAL MATCH (u)<-[:CONTIENE|HAS_UNIT]-(:DocumentoNormativo)
                WITH u, count(*) AS refs
                WHERE refs = 0
                DETACH DELETE u
                """,
                {"doc_id": doc_id},
            )

            for item in units:
                unit_id = f"{doc_id}:{item['id']}"
                session.run(
                    """
                    MATCH (d:DocumentoNormativo {id: $doc_id})
                    MERGE (u:UnidadNormativa {id: $unit_id})
                    SET u.text = $text,
                        u.texto = $text,
                        u.text_norm = $text_norm,
                        u.texto_norm = $text_norm,
                        u.article = $article,
                        u.numero = $article,
                        u.position = $position,
                        u.orden_secuencial = $position,
                        u.documento_id = $doc_id,
                        u.documento_titulo = $filename,
                        u.tipo_unidad = CASE WHEN $article IS NULL THEN 'fragmento' ELSE 'articulo' END,
                        u.fecha_revision = timestamp()
                    MERGE (d)-[:CONTIENE]->(u)
                    """,
                    {
                        "doc_id": doc_id,
                        "filename": filename,
                        "unit_id": unit_id,
                        "text": item["text"],
                        "text_norm": item["text_norm"],
                        "article": item["article"],
                        "position": item["position"],
                    },
                )

                for code in _extract_state_codes(item["text_norm"]):
                    state_id = f"estado_{code}"
                    session.run(
                        """
                        MATCH (u:UnidadNormativa {id: $unit_id})
                        MERGE (s:EstadoFinanciero {id: $state_id})
                        SET s.code = $code,
                            s.codigo = $code,
                            s.label = $code
                        MERGE (u)-[:MENTIONS_STATE]->(s)
                        MERGE (u)-[:IMPACTA_ESTADO]->(s)
                        """,
                        {"unit_id": unit_id, "state_id": state_id, "code": code},
                    )

        return {
            "status": "ok",
            "doc_id": doc_id,
            "filename": filename,
            "units": len(units),
        }

    def _query_signals(self, question: str) -> dict[str, Any]:
        normalized = _normalize_for_search(question)
        all_tokens = _tokens(normalized)
        base_keywords = [t for t in all_tokens if t not in LEGAL_STOPWORDS][:24]
        article_numbers = [_normalize_article_number(m.group(1)) for m in ARTICLE_REF_PATTERN.finditer(normalized)]
        intent_data = self._intent_via_llm(
            question=question,
            normalized_question=normalized,
            article_numbers=_dedupe(article_numbers),
        )
        intent = str(intent_data.get("intent") or "generic").strip().lower()
        topic_tokens = [str(t).strip().lower() for t in (intent_data.get("topics") or []) if str(t).strip()]
        entity_tokens = [str(t).strip().lower() for t in (intent_data.get("entities") or []) if str(t).strip()]
        keywords = _dedupe(base_keywords + topic_tokens + entity_tokens)[:28]
        keyword_roots: list[str] = []
        for token in keywords:
            if len(token) >= 6:
                keyword_roots.append(token[:5])
            if len(token) >= 8:
                keyword_roots.append(token[:6])
        keyword_roots = _dedupe(keyword_roots)
        article_numbers = _dedupe([_normalize_article_number(a) for a in (intent_data.get("article_numbers") or []) if a])
        legal_refs: list[str] = []
        for m in LEGAL_REF_PATTERN.finditer(normalized):
            raw = (m.group(2) or "").lower().strip()
            if not raw:
                continue
            legal_refs.append(raw)
            legal_refs.append(raw.replace("/", " "))
            legal_refs.append(raw.replace("/", ""))
        entity_tokens = _dedupe(entity_tokens + [t for t in LEGAL_ENTITY_TOKENS if t in normalized])
        acronyms = [m.lower() for m in re.findall(r"\b[A-Z]{2,6}\b", question or "")]
        numeric_tokens_raw = [m.group(0) for m in NUMERIC_TOKEN_PATTERN.finditer(normalized)]
        numeric_tokens: list[str] = []
        for token in numeric_tokens_raw:
            digits = re.sub(r"\D", "", token)
            if len(digits) <= 1 and "%" not in token:
                continue
            numeric_tokens.append(token)
        numeric_tokens = _dedupe(numeric_tokens)
        generic_legal_terms = {
            "articulo",
            "articulos",
            "ley",
            "reglamento",
            "real",
            "decreto",
            "norma",
            "inversion",
        }
        generic_legal_terms |= GENERIC_QUERY_TERMS
        strict_terms: list[str] = []
        primary_terms: list[str] = []
        for token in keywords:
            has_digit = any(ch.isdigit() for ch in token)
            if has_digit or (len(token) >= 7 and token not in generic_legal_terms) or token in PROTECTED_CONTENT_TERMS:
                strict_terms.append(token)
                primary_terms.append(token)
        content_terms = [
            t
            for t in keywords
            if len(t) >= 4 and (t not in generic_legal_terms or t in PROTECTED_CONTENT_TERMS or t in LEGAL_ENTITY_TOKENS)
        ]
        content_roots: list[str] = []
        for term in content_terms:
            root = term[:5] if len(term) >= 5 else term
            if len(root) >= 4:
                content_roots.append(root)
        bigram_tokens = [t for t in all_tokens if t not in LEGAL_STOPWORDS and t not in GENERIC_QUERY_TERMS and len(t) >= 3]
        bigrams = _dedupe([f"{bigram_tokens[i]} {bigram_tokens[i + 1]}" for i in range(len(bigram_tokens) - 1)])
        strict_terms.extend(acronyms)
        strict_terms.extend(numeric_tokens)
        asks_article_numbers = intent == "article_list"
        asks_single_article = intent == "article_lookup"
        asks_definition = intent == "definition"
        asks_effective_date = intent == "effective_date"
        asks_yes_no = intent == "yes_no"
        asks_requirements = intent == "requirements"
        asks_comparison = intent == "comparison"
        asks_exclusion = intent == "exclusion"
        enumerative_need = bool(
            asks_article_numbers
            or asks_requirements
            or asks_exclusion
            or re.search(
                r"\b(que\s+entidades|que\s+articulos|que\s+requisitos|que\s+obligaciones|que\s+causas|que\s+supuestos)\b",
                normalized,
            )
        )
        asks_extreme = bool(
            re.search(
                r"\b(maximo|maxima|minimo|minima|mayor|menor|tope|limite|cuantia|importe|plazo\s+maximo)\b",
                normalized,
            )
        ) or any(
            re.search(r"(?:^|.*)(?:max|xim[oa]|min|nim[oa]|tope|limite|cuantia|importe)(?:.*|$)", token)
            for token in all_tokens
        )
        asks_modal = bool(
            asks_yes_no
            or re.search(r"\b(puede|permite|prohibe|prohibe|debe|obliga|obligatorio)\b", normalized)
        )

        return {
            "normalized_question": normalized,
            "token_count": len(all_tokens),
            "keywords": _dedupe(keywords),
            "keyword_roots": keyword_roots,
            "article_numbers": _dedupe(article_numbers),
            "legal_refs": _dedupe(legal_refs),
            "entity_tokens": _dedupe(entity_tokens),
            "acronyms": _dedupe(acronyms),
            "strict_terms": _dedupe(strict_terms),
            "primary_terms": _dedupe(primary_terms),
            "content_terms": _dedupe(content_terms),
            "content_roots": _dedupe(content_roots),
            "bigrams": bigrams[:20],
            "numeric_tokens": numeric_tokens,
            "asks_article_numbers": asks_article_numbers,
            "asks_single_article": asks_single_article,
            "asks_definition": asks_definition,
            "asks_effective_date": asks_effective_date,
            "asks_yes_no": asks_yes_no,
            "asks_requirements": asks_requirements,
            "asks_comparison": asks_comparison,
            "asks_exclusion": asks_exclusion,
            "enumerative_need": enumerative_need,
            "asks_extreme": asks_extreme,
            "asks_modal": asks_modal,
            "intent": intent,
            "intent_topics": topic_tokens,
            "intent_confidence": float(intent_data.get("confidence") or 0.0),
            "intent_source": str(intent_data.get("source") or "llm"),
        }

    def _candidate_rows(self, question: str, overfetch: int) -> list[dict[str, Any]]:
        signals = self._query_signals(question)
        keywords = signals["keywords"]
        keyword_roots = signals.get("keyword_roots", [])
        phrase = signals["normalized_question"]
        article_numbers = signals["article_numbers"]
        legal_refs = signals["legal_refs"]
        acronyms = signals["acronyms"]

        with self.driver.session(database=NEO4J_DATABASE) as session:
            rows = session.run(
                """
                MATCH (d:DocumentoNormativo)-[:CONTIENE|HAS_UNIT]->(u:UnidadNormativa)
                WITH d, u,
                     coalesce(u.text_norm, u.texto_norm, '') AS text_norm,
                     coalesce(u.text, u.texto, '') AS text,
                     substring(coalesce(u.text_norm, u.texto_norm, ''), 0, 320) AS head_norm,
                     coalesce(u.article, u.numero, '') AS article,
                     coalesce(u.position, u.orden_secuencial, 0) AS position,
                     coalesce(d.filename, d.titulo, d.title, d.id, '') AS doc_title,
                     coalesce(d.titulo_norm, '') AS doc_title_norm
                WHERE text <> ''
                RETURN coalesce(u.id, elementId(u)) AS unit_id,
                       text AS text,
                       text_norm AS text_norm,
                       article AS article,
                       position AS position,
                       coalesce(d.id, elementId(d)) AS documento_id,
                       doc_title AS documento_titulo,
                       doc_title_norm AS doc_title_norm,
                       reduce(h = 0, t IN $keywords | h + CASE WHEN text_norm CONTAINS t THEN 1 ELSE 0 END) AS kw_hits,
                       reduce(h = 0, t IN $keyword_roots | h + CASE WHEN text_norm CONTAINS t THEN 1 ELSE 0 END) AS root_hits,
                       reduce(h = 0, t IN $keywords | h + CASE WHEN head_norm CONTAINS t THEN 1 ELSE 0 END) AS head_kw_hits,
                       reduce(h = 0, t IN $keyword_roots | h + CASE WHEN head_norm CONTAINS t THEN 1 ELSE 0 END) AS head_root_hits,
                       reduce(h = 0, r IN $legal_refs | h + CASE WHEN (text_norm CONTAINS r OR doc_title_norm CONTAINS r) THEN 1 ELSE 0 END) AS ref_hits
                ORDER BY kw_hits DESC, ref_hits DESC, position ASC, article ASC, u.id ASC
                LIMIT $limit
                """,
                {
                    "keywords": keywords,
                    "keyword_roots": keyword_roots,
                    "article_numbers": article_numbers,
                    "legal_refs": legal_refs,
                    "acronyms": acronyms,
                    "phrase": phrase,
                    "limit": int(max(overfetch, 1200)),
                },
            ).data()
        return rows

    def _fetch_rows_by_articles(self, article_numbers: list[str], limit: int = 240) -> list[dict[str, Any]]:
        normalized = [_normalize_article_number(a) for a in (article_numbers or []) if a]
        normalized = _dedupe([a for a in normalized if a])
        if not normalized:
            return []
        article_regexes = [f"(?s).*\\barticulo\\s+{re.escape(a)}\\b.*" for a in normalized]
        with self.driver.session(database=NEO4J_DATABASE) as session:
            rows = session.run(
                """
                MATCH (d:DocumentoNormativo)-[:CONTIENE|HAS_UNIT]->(u:UnidadNormativa)
                WITH d, u,
                     toLower(replace(coalesce(u.article, u.numero, ''), ' ', '')) AS article_norm,
                     coalesce(u.text_norm, u.texto_norm, '') AS text_norm,
                     coalesce(u.text, u.texto, '') AS text,
                     coalesce(u.position, u.orden_secuencial, 0) AS position,
                     coalesce(d.filename, d.titulo, d.title, d.id, '') AS doc_title,
                     coalesce(d.titulo_norm, '') AS doc_title_norm
                WHERE text <> '' AND (
                    article_norm IN $articles OR
                    any(rx IN $article_regexes WHERE text_norm =~ rx)
                )
                RETURN coalesce(u.id, elementId(u)) AS unit_id,
                       text AS text,
                       text_norm AS text_norm,
                       article_norm AS article,
                       position AS position,
                       coalesce(d.id, elementId(d)) AS documento_id,
                       doc_title AS documento_titulo,
                       doc_title_norm AS doc_title_norm,
                       10 AS kw_hits,
                       0 AS root_hits,
                       0 AS ref_hits
                ORDER BY position ASC, article_norm ASC, u.id ASC
                LIMIT $limit
                """,
                {"articles": normalized, "article_regexes": article_regexes, "limit": int(max(limit, 40))},
            ).data()
        return rows

    def _fetch_rows_by_heading_terms(self, terms: list[str], limit: int = 240) -> list[dict[str, Any]]:
        normalized_terms = _dedupe([_normalize_for_search(t) for t in (terms or []) if t])
        normalized_terms = [t for t in normalized_terms if len(t) >= 4]
        if not normalized_terms:
            return []
        with self.driver.session(database=NEO4J_DATABASE) as session:
            rows = session.run(
                """
                MATCH (d:DocumentoNormativo)-[:CONTIENE|HAS_UNIT]->(u:UnidadNormativa)
                WITH d, u,
                     coalesce(u.text_norm, u.texto_norm, '') AS text_norm,
                     coalesce(u.text, u.texto, '') AS text,
                     substring(coalesce(u.text_norm, u.texto_norm, ''), 0, 320) AS head_norm,
                     toLower(replace(coalesce(u.article, u.numero, ''), ' ', '')) AS article_norm,
                     coalesce(u.position, u.orden_secuencial, 0) AS position,
                     coalesce(d.filename, d.titulo, d.title, d.id, '') AS doc_title,
                     coalesce(d.titulo_norm, '') AS doc_title_norm
                WHERE text <> ''
                WITH d, u, text_norm, text, head_norm, article_norm, position, doc_title, doc_title_norm,
                     reduce(h = 0, t IN $terms | h + CASE WHEN head_norm CONTAINS t THEN 1 ELSE 0 END) AS heading_hits
                WHERE heading_hits > 0
                RETURN coalesce(u.id, elementId(u)) AS unit_id,
                       text AS text,
                       text_norm AS text_norm,
                       article_norm AS article,
                       position AS position,
                       coalesce(d.id, elementId(d)) AS documento_id,
                       doc_title AS documento_titulo,
                       doc_title_norm AS doc_title_norm,
                       heading_hits AS kw_hits,
                       heading_hits AS root_hits,
                       heading_hits AS head_kw_hits,
                       heading_hits AS head_root_hits,
                       0 AS ref_hits
                ORDER BY heading_hits DESC, position ASC, article_norm ASC, u.id ASC
                LIMIT $limit
                """,
                {"terms": normalized_terms, "limit": int(max(limit, 40))},
            ).data()
        return rows

    def _score_row(
        self,
        row: dict[str, Any],
        signals: dict[str, Any],
        keyword_df: dict[str, int] | None = None,
    ) -> float:
        text_norm = str(row.get("text_norm") or "")
        doc_title_norm = str(row.get("doc_title_norm") or "")
        keywords = [str(t) for t in (signals.get("keywords") or []) if t]
        content_terms = [str(t) for t in (signals.get("content_terms") or []) if t]
        content_roots = [str(t) for t in (signals.get("content_roots") or []) if t]
        article_numbers = [str(a) for a in (signals.get("article_numbers") or []) if a]
        legal_refs = [str(r) for r in (signals.get("legal_refs") or []) if r]
        entity_tokens = [str(e) for e in (signals.get("entity_tokens") or []) if e]
        numeric_tokens = [str(n) for n in (signals.get("numeric_tokens") or []) if n]
        article = _normalize_article_number(row.get("article"))

        def coverage(tokens: list[str], haystack_a: str, haystack_b: str = "") -> tuple[int, float]:
            if not tokens:
                return 0, 0.0
            hits = sum(1 for token in tokens if token in haystack_a or (haystack_b and token in haystack_b))
            return hits, hits / max(len(tokens), 1)

        _, keyword_cov = coverage(keywords[:16], text_norm)
        _, content_cov = coverage(content_terms[:12], text_norm, doc_title_norm)
        _, root_cov = coverage(content_roots[:12], text_norm, doc_title_norm)
        long_terms = [t for t in content_terms if len(t) >= 9 or " " in t]
        long_hits, long_cov = coverage(long_terms[:8], text_norm, doc_title_norm)
        ref_hits, ref_cov = coverage(legal_refs[:8], text_norm, doc_title_norm)
        _, entity_cov = coverage(entity_tokens[:10], text_norm, doc_title_norm)

        score = (0.30 * keyword_cov) + (0.24 * content_cov) + (0.20 * root_cov) + (0.14 * long_cov) + (0.06 * ref_cov) + (0.06 * entity_cov)

        if article_numbers:
            if article and article in article_numbers:
                score += 1.20
            else:
                score *= 0.15

        if legal_refs and ref_hits == 0:
            score *= 0.30

        if numeric_tokens and LEGAL_REQUIRE_NUMERIC_OVERLAP_FOR_NUMERIC_QUERY:
            numeric_in_text = {m.group(0) for m in NUMERIC_TOKEN_PATTERN.finditer(text_norm)}
            overlap = sum(1 for token in numeric_tokens if token in numeric_in_text)
            if overlap == 0:
                score *= 0.40

        if signals.get("asks_modal") and long_terms and long_hits == 0:
            score *= 0.72

        if signals.get("asks_article_numbers"):
            row_text = str(row.get("text") or "")
            heading_line = next((ln.strip() for ln in row_text.splitlines() if ln and ln.strip()), row_text[:140])
            heading_norm = _normalize_for_search(heading_line)
            focus_window = text_norm[:1800]
            has_article_heading = bool(re.search(r"\barticulo\s+\d", heading_norm))
            if has_article_heading:
                score += 0.24
            else:
                score *= 0.72

            if len(text_norm) > 5000:
                score *= 0.45
            elif len(text_norm) > 3500:
                score *= 0.62

            if entity_tokens:
                entity_hits = sum(1 for token in entity_tokens if token in focus_window or token in doc_title_norm)
                if entity_hits == 0:
                    score *= 0.65
                else:
                    score += min(0.18, 0.06 * float(entity_hits))

        if signals.get("asks_comparison") and entity_tokens:
            entity_hits = sum(1 for token in entity_tokens if token in text_norm)
            if entity_hits >= 2:
                score += 0.38
            elif entity_hits == 1:
                score += 0.14
            else:
                score *= 0.78

        score += min(0.12, float(row.get("kw_hits") or 0.0) * 0.018)
        score += min(0.10, float(row.get("root_hits") or 0.0) * 0.014)
        score += min(
            0.28,
            (float(row.get("head_kw_hits") or 0.0) * 0.05) + (float(row.get("head_root_hits") or 0.0) * 0.035),
        )

        if signals.get("enumerative_need"):
            heading_hit_strength = float(row.get("head_kw_hits") or 0.0) + float(row.get("head_root_hits") or 0.0)
            score += min(0.20, heading_hit_strength * 0.03)
            if re.search(r"(^|\n)\s*[-*]\s+|(^|\n)\s*[a-z]\)\s+", str(row.get("text") or ""), flags=re.IGNORECASE):
                score += 0.14

        if signals.get("asks_exclusion"):
            heading_norm = _normalize_for_search(str(row.get("text") or "")[:280])
            has_exclusion = bool(
                re.search(r"\bexclu(?:ye|yen|id[oa]s?|ir|ira|iran|sion(?:es)?)\b", text_norm)
                or re.search(r"\bno\s+sera\s+de\s+aplicacion\b", text_norm)
            )
            heading_exclusion = bool(
                re.search(r"\barticulo\s+\d", heading_norm)
                and (
                    re.search(r"\bexclu(?:ye|yen|id[oa]s?|ir|ira|iran|sion(?:es)?)\b", heading_norm)
                    or re.search(r"\bno\s+sera\s+de\s+aplicacion\b", heading_norm)
                )
            )
            if heading_exclusion:
                score += 0.42
            elif has_exclusion:
                score += 0.26
            else:
                score *= 0.35

        if signals.get("asks_extreme"):
            if NUMERIC_TOKEN_PATTERN.search(text_norm):
                score += 0.25
            if re.search(r"\b(maxim|minim|hasta|superior|inferior|por\s+ciento|euros?)\b", text_norm):
                score += 0.18

        if text_norm.count("|") >= 2 or re.search(r"\.{8,}\s*\d+", text_norm):
            score *= 0.60

        return float(max(score, 0.0))

    def _rank_select_generic(
        self,
        rows: list[dict[str, Any]],
        signals: dict[str, Any],
        top_k: int,
        overfetch: int,
    ) -> list[GraphChunk]:
        for row in rows:
            if row.get("article"):
                continue
            text = str(row.get("text") or "")
            detected_article = self._article_from_text(text)
            if detected_article:
                row["article"] = detected_article

        keyword_df: dict[str, int] = {}
        if rows and signals["keywords"]:
            for k in signals["keywords"]:
                keyword_df[k] = sum(1 for row in rows if k in (row.get("text_norm") or ""))

        bm25_scores: dict[str, float] = {}
        bm25_terms = [t for t in (signals.get("content_terms") or signals.get("keywords") or []) if len(t) >= 3][:12]
        if rows and bm25_terms:
            row_tokens: list[list[str]] = []
            row_ids: list[str] = []
            for row in rows:
                text_norm = str(row.get("text_norm") or "")
                toks = re.findall(r"[a-z0-9]+", text_norm)
                row_tokens.append(toks)
                row_ids.append(str(row.get("unit_id") or ""))
            avg_dl = (sum(len(toks) for toks in row_tokens) / len(row_tokens)) if row_tokens else 1.0
            df: dict[str, int] = {}
            for term in bm25_terms:
                df[term] = sum(1 for toks in row_tokens if term in toks)
            k1 = 1.2
            b = 0.75
            for unit_id, toks in zip(row_ids, row_tokens):
                dl = max(len(toks), 1)
                score = 0.0
                for term in bm25_terms:
                    tf = toks.count(term)
                    if tf <= 0:
                        continue
                    n = max(df.get(term, 0), 0)
                    idf = max(0.0, (len(row_tokens) - n + 0.5) / (n + 0.5))
                    idf = float(math.log(1.0 + idf))
                    denom = tf + k1 * (1 - b + b * dl / max(avg_dl, 1.0))
                    score += idf * ((tf * (k1 + 1)) / max(denom, 1e-9))
                bm25_scores[unit_id] = score

        max_bm25 = max(bm25_scores.values()) if bm25_scores else 0.0

        ranked: list[tuple[float, dict[str, Any]]] = []
        for row in rows:
            score = self._score_row(row, signals, keyword_df=keyword_df)
            if bm25_scores:
                unit_id = str(row.get("unit_id") or "")
                bm25 = bm25_scores.get(unit_id, 0.0)
                bm25_norm = (bm25 / max_bm25) if max_bm25 > 0 else 0.0
                score += 0.45 * bm25_norm
            if score <= 0:
                continue
            ranked.append((score, row))
        ranked.sort(key=lambda x: x[0], reverse=True)

        selected: list[GraphChunk] = []
        seen_fingerprints: set[str] = set()
        article_counts: dict[str, int] = {}
        per_article_limit = 2 if (signals.get("asks_article_numbers") or signals.get("asks_comparison")) else 3
        for score, row in ranked:
            text = (row.get("text") or "").strip()
            if not text:
                continue
            fp = _normalize_for_search(text[:260])
            if fp in seen_fingerprints:
                continue
            seen_fingerprints.add(fp)
            article_value = row.get("article")
            if not article_value:
                article_value = self._article_from_text(text)
            article_key = _normalize_article_number(article_value)
            if article_key:
                current = int(article_counts.get(article_key, 0))
                if current >= per_article_limit:
                    continue
                article_counts[article_key] = current + 1
            metadata = {
                "unit_id": row.get("unit_id"),
                "tipo_unidad": "articulo" if article_value else "unidad",
                "numero": article_value,
                "documento_id": row.get("documento_id"),
                "documento_titulo": row.get("documento_titulo"),
                "position": row.get("position"),
            }
            selected.append(
                GraphChunk(
                    id=str(row.get("unit_id")),
                    text=text,
                    source=str(row.get("documento_titulo") or row.get("documento_id") or "documento_desconocido"),
                    score=score,
                    metadata=metadata,
                )
            )
            if len(selected) >= top_k:
                break

        self._last_search_debug = {
            "retrieval_strategies": {
                "keyword_ranking": [str(row.get("unit_id")) for _, row in ranked[:20]],
                "doc_ref_ranking": [],
                "exact_article_ranking": [str(row.get("unit_id")) for _, row in ranked[:20] if row.get("article")],
                "legal_ref_ranking": [],
                "constraint_ref_ranking": [],
                "state_graph_ranking": [],
                "fundtype_graph_ranking": [],
                "section_graph_ranking": [],
                "avcos_graph_ranking": [],
            },
            "retrieval_metrics": {
                "candidate_count": len(rows),
                "selected_count": len(selected),
                "overfetch_limit": overfetch,
                "rerank_top_n": top_k,
            },
            "rrf_scores": {c.id: round(float(c.score or 0.0), 6) for c in selected},
            "query_signals": signals,
            "reranker_trace": {
                "applied": bool(bm25_scores),
                "mode": "hybrid_lexical_bm25" if bm25_scores else "local_lexical",
                "model": "local_lexical_bm25" if bm25_scores else "local_lexical",
                "top_n": top_k,
                "fallback_reason": "not_configured",
            },
        }
        return selected

    def search_units(self, question: str, top_k: int = 8) -> list[GraphChunk]:
        top_k = max(1, int(top_k))
        signals = self._query_signals(question)

        if signals.get("asks_article_numbers") or signals.get("article_numbers") or signals.get("asks_comparison"):
            top_k = max(top_k, 12)
        elif signals.get("asks_yes_no"):
            top_k = max(top_k, 12)
        elif (
            signals.get("asks_definition")
            or signals.get("asks_requirements")
            or signals.get("asks_exclusion")
        ):
            top_k = max(top_k, 10)
        else:
            top_k = max(top_k, 8)

        overfetch = min(max(top_k * 24, 120), 1200)
        rows = self._candidate_rows(question=question, overfetch=overfetch)
        requested_articles = {str(a) for a in (signals.get("article_numbers") or []) if a}

        if not requested_articles:
            heading_terms = [str(t) for t in (signals.get("content_roots") or signals.get("content_terms") or []) if t]
            if signals.get("asks_article_numbers"):
                heading_terms = _dedupe(
                    heading_terms
                    + [str(t) for t in (signals.get("entity_tokens") or []) if t]
                    + [str(t) for t in (signals.get("bigrams") or []) if t]
                    + [str(t) for t in (signals.get("keywords") or []) if t]
                )[:24]
            heading_rows = self._fetch_rows_by_heading_terms(
                terms=heading_terms,
                limit=max(120, top_k * 20),
            )
            seen_ids = {str(r.get("unit_id") or "") for r in rows}
            for row in heading_rows:
                row_id = str(row.get("unit_id") or "")
                if row_id and row_id in seen_ids:
                    continue
                rows.append(row)
                if row_id:
                    seen_ids.add(row_id)

            if signals.get("asks_exclusion"):
                exclusion_heading_terms = _dedupe(heading_terms + ["exclu", "no sera de aplicacion", "entidades excluidas"])
                exclusion_rows = self._fetch_rows_by_heading_terms(
                    terms=exclusion_heading_terms,
                    limit=max(120, top_k * 24),
                )
                for row in exclusion_rows:
                    row_id = str(row.get("unit_id") or "")
                    if row_id and row_id in seen_ids:
                        continue
                    rows.append(row)
                    if row_id:
                        seen_ids.add(row_id)

        if requested_articles:
            article_rows = self._fetch_rows_by_articles(
                article_numbers=list(requested_articles),
                limit=max(80, top_k * 20),
            )
            seen_ids = {str(r.get("unit_id") or "") for r in rows}
            for row in article_rows:
                row_id = str(row.get("unit_id") or "")
                if row_id and row_id in seen_ids:
                    continue
                rows.append(row)
                if row_id:
                    seen_ids.add(row_id)
        selected = self._rank_select_generic(rows=rows, signals=signals, top_k=top_k, overfetch=overfetch)

        if signals.get("asks_exclusion"):
            exclusion_pattern = r"\bexclu(?:ye|yen|id[oa]s?|ir|ira|iran|sion(?:es)?)\b|\bno\s+sera\s+de\s+aplicacion\b"
            has_anchor = any(
                re.search(exclusion_pattern, _normalize_for_search((chunk.text or "")[:280]))
                and re.search(r"\barticulo\s+\d", _normalize_for_search((chunk.text or "")[:280]))
                for chunk in selected[: max(6, top_k)]
            )
            if not has_anchor:
                anchor_row: dict[str, Any] | None = None
                anchor_score = -1.0
                for row in rows:
                    text = str(row.get("text") or "")
                    if not text:
                        continue
                    heading_norm = _normalize_for_search(text[:280])
                    if not re.search(r"\barticulo\s+\d", heading_norm):
                        continue
                    text_norm = str(row.get("text_norm") or "")
                    if not re.search(exclusion_pattern, text_norm):
                        continue
                    row_score = self._score_row(row, signals)
                    if row_score > anchor_score:
                        anchor_score = row_score
                        anchor_row = row
                if anchor_row is not None:
                    anchor_text = str(anchor_row.get("text") or "").strip()
                    anchor_id = str(anchor_row.get("unit_id") or "")
                    if anchor_text and all(str(chunk.id or "") != anchor_id for chunk in selected):
                        article_value = _normalize_article_number(anchor_row.get("article")) or self._article_from_text(anchor_text) or ""
                        anchor_chunk = GraphChunk(
                            id=anchor_id or str(anchor_row.get("unit_id") or ""),
                            text=anchor_text,
                            source=str(anchor_row.get("documento_titulo") or anchor_row.get("documento_id") or "documento_desconocido"),
                            score=float(anchor_row.get("kw_hits") or 1.0) + 1.1,
                            metadata={
                                "unit_id": anchor_row.get("unit_id"),
                                "tipo_unidad": "articulo" if article_value else "unidad",
                                "numero": article_value or None,
                                "documento_id": anchor_row.get("documento_id"),
                                "documento_titulo": anchor_row.get("documento_titulo"),
                                "position": anchor_row.get("position"),
                            },
                        )
                        selected = [anchor_chunk] + selected
                        selected = selected[: max(top_k, 10)]

        if requested_articles:
            covered_articles = {
                _normalize_article_number((chunk.metadata or {}).get("numero")) or self._article_from_text(chunk.text) or ""
                for chunk in selected
            }
            missing_articles = {a for a in requested_articles if a and a not in covered_articles}
            if missing_articles:
                supplemental_rows = self._fetch_rows_by_articles(
                    article_numbers=list(missing_articles),
                    limit=max(120, top_k * 24),
                )
                seen_chunk_ids = {str(chunk.id or "") for chunk in selected}
                for row in supplemental_rows:
                    row_id = str(row.get("unit_id") or "")
                    if row_id and row_id in seen_chunk_ids:
                        continue
                    text = str(row.get("text") or "").strip()
                    if not text:
                        continue
                    article_value = _normalize_article_number(row.get("article")) or self._article_from_text(text) or ""
                    if article_value not in missing_articles:
                        continue
                    selected.append(
                        GraphChunk(
                            id=row_id or str(row.get("unit_id") or ""),
                            text=text,
                            source=str(row.get("documento_titulo") or row.get("documento_id") or "documento_desconocido"),
                            score=float(row.get("kw_hits") or 1.0) + 1.5,
                            metadata={
                                "unit_id": row.get("unit_id"),
                                "tipo_unidad": "articulo",
                                "numero": article_value,
                                "documento_id": row.get("documento_id"),
                                "documento_titulo": row.get("documento_titulo"),
                                "position": row.get("position"),
                            },
                        )
                    )
                    if row_id:
                        seen_chunk_ids.add(row_id)
                    missing_articles.discard(article_value)
                    if not missing_articles:
                        break
            selected.sort(
                key=lambda chunk: (
                    0 if _normalize_article_number((chunk.metadata or {}).get("numero")) in requested_articles else 1,
                    -float(chunk.score or 0.0),
                )
            )
        return selected

    def light_probe(self, question: str, top_k: int = 4) -> list[GraphChunk]:
        return self.search_units(question=question, top_k=max(1, min(top_k, 6)))

    def _evidence_block(self, chunks: list[GraphChunk], max_items: int | None = None, indices: list[int] | None = None) -> str:
        if not chunks:
            return ""
        lines = ["**Evidencia**"]
        if indices:
            seen: set[int] = set()
            normalized = [i for i in indices if 1 <= i <= len(chunks) and not (i in seen or seen.add(i))]
            for idx in normalized[:LEGAL_EVIDENCE_MAX_ITEMS]:
                chunk = chunks[idx - 1]
                snippet = _repair_visible_text(chunk.text[:LEGAL_SNIPPET_MAX_CHARS])
                lines.append(f"[{idx}] {chunk.source} (fragmento {idx}): {snippet}")
            return "\n".join(lines) if len(lines) > 1 else ""

        n = max(1, min(max_items or LEGAL_EVIDENCE_MAX_ITEMS, len(chunks)))
        lines = ["**Evidencia**"]
        for idx, chunk in enumerate(chunks[:n], start=1):
            snippet = _repair_visible_text(chunk.text[:LEGAL_SNIPPET_MAX_CHARS])
            lines.append(f"[{idx}] {chunk.source} (fragmento {idx}): {snippet}")
        return "\n".join(lines)

    def _normalize_citation_tags(self, answer: str) -> str:
        if not answer:
            return answer
        out = answer
        out = re.sub(r"\[(?:n|N)(\d+)\]", r"[\1]", out)
        out = re.sub(r"\[(?:n|N)\]", "[1]", out)
        return out

    def _article_from_text(self, text: str) -> str | None:
        lines = [ln.strip() for ln in (text or "").splitlines() if ln and ln.strip()]
        for line in lines[:8]:
            norm = _normalize_for_search(line)
            m = ARTICLE_HEADING_MD_PATTERN.match(norm) or ARTICLE_HEADING_INLINE_PATTERN.match(norm)
            if m:
                return _normalize_article_number(m.group(1))
            m2 = re.search(r"\barticulo\s+(\d+[a-z]?)\b", norm)
            if m2:
                return _normalize_article_number(m2.group(1))
        return None

    def _compose_answer_deterministic(self, question: str, chunks: list[GraphChunk]) -> str:
        raise RuntimeError("Deterministic synthesis is disabled: LLM-only mode is enforced")

    def _compose_answer_llm(self, question: str, chunks: list[GraphChunk], chat_history: list[dict] | None) -> str:
        llm = self._get_llm()
        if llm is None:
            raise RuntimeError("LLM answer generator unavailable: OPENAI_API_KEY is required")
        signals = self._query_signals(question)
        intent = str(signals.get("intent") or "generic")

        context_parts: list[str] = []
        for idx, chunk in enumerate(chunks[:8], start=1):
            snippet = _repair_visible_text((chunk.text or "")[:1800])
            if not snippet:
                continue
            context_parts.append(f"<fragmento id=\"{idx}\" fuente=\"{chunk.source}\">\n{snippet}\n</fragmento>")

        if not context_parts:
            return "NO ENCONTRADO EN EL DOCUMENTO"

        messages = [
            {
                "role": "system",
                "content": (
                    "Eres un asistente juridico de QA sobre normativa. "
                    "Responde usando SOLO informacion explicita de los fragmentos. "
                    "No inventes ni extrapoles. "
                    "Si hay evidencia parcial, responde de forma parcial y explicita el alcance. "
                    "Usa NO ENCONTRADO EN EL DOCUMENTO solo cuando no exista evidencia relevante en los fragmentos. "
                    "Prioriza contestar exactamente lo preguntado y evita texto lateral. "
                    "Incluye citas inline [n] solo de fragmentos usados. "
                    "No cites articulos o datos que no esten en esos fragmentos."
                ),
            }
        ]
        if chat_history:
            for msg in chat_history[-6:]:
                role = str(msg.get("role") or "").strip().lower()
                content = str(msg.get("content") or "")
                if role in {"user", "assistant"} and content:
                    messages.append({"role": role, "content": content})

        messages.append(
            {
                "role": "user",
                "content": (
                    "<contexto>\n"
                    + "\n\n".join(context_parts)
                    + "\n</contexto>\n\n"
                    + f"Pregunta: {question}\n\n"
                    + f"Intencion estimada: {intent}\n\n"
                    + "Formato:\n"
                    + "- Respuesta directa en 1-4 frases.\n"
                    + "- Si piden articulos, lista solo los articulos relevantes del contexto.\n"
                    + "- Si la cobertura es parcial, indicalo explicitamente.\n"
                    + "- No repitas el contexto completo.\n"
                    + "- Cada frase juridica debe incluir al menos una cita [n].\n"
                ),
            }
        )
        try:
            response = llm.chat.completions.create(
                model=LLM_MODEL,
                temperature=0.0,
                messages=messages,
            )
            text = self._normalize_citation_tags(_repair_visible_text(response.choices[0].message.content or ""))
            if not text:
                return "NO ENCONTRADO EN EL DOCUMENTO"
            return text
        except Exception as exc:
            raise RuntimeError(f"LLM answer generator failed: {exc}") from exc

    def _extractive_answer_by_intent(self, question: str, chunks: list[GraphChunk]) -> str | None:
        if not chunks:
            return None
        signals = self._query_signals(question)
        intent = str(signals.get("intent") or "").strip().lower()
        requested_articles = [str(a) for a in (signals.get("article_numbers") or []) if a]
        content_terms = [str(t) for t in (signals.get("content_terms") or []) if t]
        normalized_question = str(signals.get("normalized_question") or "")
        question_terms = _dedupe(
            [t for t in _tokens(normalized_question) if t and t not in LEGAL_STOPWORDS and t not in GENERIC_QUERY_TERMS]
        )
        question_term_set = set(question_terms)
        broad_question_terms = _dedupe([t for t in _tokens(normalized_question) if t and t not in LEGAL_STOPWORDS])
        broad_question_roots = _dedupe([t[:5] for t in broad_question_terms if len(t) >= 5])
        comparison_terms = _dedupe(
            [
                str(t)
                for t in (
                    (signals.get("entity_tokens") or [])
                    + (signals.get("acronyms") or [])
                    + (signals.get("content_terms") or [])
                )
                if str(t)
            ]
        )

        def first_sentence(text: str) -> str:
            cleaned = _repair_visible_text(str(text or "")).replace("\n", " ").strip()
            parts = re.split(r"(?<=[\.!?])\s+", cleaned)
            for part in parts:
                if len(_normalize_for_search(part)) >= 24:
                    return part.strip()[:420]
            return cleaned[:420]

        def extract_list_items(text: str, max_items: int = 5) -> list[str]:
            items: list[str] = []
            for raw_line in (text or "").splitlines():
                line = _repair_visible_text(raw_line).strip()
                if not line:
                    continue
                norm = _normalize_for_search(line)
                if re.match(r"^[-*]\s+", line) or re.match(r"^[a-z]\)\s+", norm) or re.match(r"^\d+[.)]\s+", norm):
                    clean = re.sub(r"^[-*]\s+|^[a-z]\)\s+|^\d+[.)]\s+", "", line, flags=re.IGNORECASE).strip(" .;")
                    if len(_normalize_for_search(clean)) >= 12:
                        items.append(clean)
                if len(items) >= max_items:
                    break
            return _dedupe(items)

        if intent == "article_lookup" and requested_articles:
            for requested in requested_articles:
                for idx, chunk in enumerate(chunks, start=1):
                    number = _normalize_article_number((chunk.metadata or {}).get("numero")) or self._article_from_text(chunk.text) or ""
                    if number != requested:
                        continue
                    sentence = first_sentence(chunk.text)
                    if sentence:
                        return f"articulo {requested}: {sentence} [{idx}]"

        if intent == "article_list":
            generic_article_terms = {
                "articulo",
                "articulos",
                "regimen",
                "juridico",
                "juridica",
                "juridicas",
                "ley",
                "norma",
                "regula",
                "regulan",
            }
            content_roots = _dedupe([t[:5] for t in content_terms if len(t) >= 5])
            question_focus_terms = [
                t
                for t in question_terms
                if len(t) >= 5 and t not in generic_article_terms and t not in LEGAL_STOPWORDS
            ]
            distinctive_terms = _dedupe([t for t in (content_terms + question_focus_terms) if t not in generic_article_terms])
            if not distinctive_terms:
                distinctive_terms = list(content_terms)
            distinctive_roots = _dedupe([t[:5] for t in distinctive_terms if len(t) >= 5])
            entity_terms = _dedupe([str(t) for t in (signals.get("entity_tokens") or []) if t])
            article_candidates: dict[str, dict[str, Any]] = {}

            for idx, chunk in enumerate(chunks, start=1):
                number = _normalize_article_number((chunk.metadata or {}).get("numero")) or self._article_from_text(chunk.text) or ""
                if not number:
                    continue
                text_norm = _normalize_for_search(chunk.text or "")
                if not text_norm:
                    continue
                focus_window = text_norm[:1600]
                raw_text = chunk.text or ""
                heading_line = next((ln.strip() for ln in raw_text.splitlines() if ln and ln.strip()), raw_text[:140])
                heading_norm = _normalize_for_search(heading_line)
                has_heading = bool(re.search(r"\barticulo\s+\d", heading_norm))
                term_hits = sum(1 for term in content_terms if term in focus_window)
                root_hits = sum(1 for root in content_roots if root in focus_window)
                heading_term_hits = sum(1 for term in content_terms if term in heading_norm)
                heading_root_hits = sum(1 for root in content_roots if root in heading_norm)
                distinct_hits = sum(1 for term in distinctive_terms if term in focus_window)
                distinct_root_hits = sum(1 for root in distinctive_roots if root in focus_window)
                heading_distinct_hits = sum(1 for term in distinctive_terms if term in heading_norm)
                heading_distinct_root_hits = sum(1 for root in distinctive_roots if root in heading_norm)
                heading_entity_hits = sum(1 for ent in entity_terms if ent in heading_norm)
                entity_hits = sum(1 for ent in entity_terms if ent in focus_window)
                if not has_heading:
                    continue
                if entity_terms and entity_hits == 0 and heading_entity_hits == 0:
                    continue
                if (heading_distinct_hits + heading_distinct_root_hits + distinct_hits + distinct_root_hits + heading_entity_hits) == 0:
                    continue
                if has_heading and (heading_term_hits + heading_root_hits) == 0 and term_hits < 2:
                    continue
                score = (2.2 if has_heading else 0.0) + (1.2 * float(heading_term_hits)) + (0.8 * float(heading_root_hits))
                score += float(term_hits) + (0.6 * float(root_hits)) + (0.8 * float(entity_hits))
                score += (1.6 * float(heading_distinct_hits)) + (1.0 * float(heading_distinct_root_hits))
                score += (0.9 * float(distinct_hits)) + (0.5 * float(distinct_root_hits))
                score += 0.22 * float(chunk.score or 0.0)
                if len(text_norm) > 4500:
                    score *= 0.52
                elif len(text_norm) > 3200:
                    score *= 0.68
                source = str(chunk.source or "")
                current = article_candidates.get(number)
                if current is None or score > float(current.get("score", -1.0)):
                    article_candidates[number] = {
                        "score": score,
                        "idx": idx,
                        "source": source,
                        "has_heading": has_heading,
                    }

            if article_candidates:
                source_scores: dict[str, float] = {}
                for data in article_candidates.values():
                    src = str(data.get("source") or "")
                    if not src:
                        continue
                    source_scores[src] = source_scores.get(src, 0.0) + float(data.get("score") or 0.0)
                preferred_source = max(source_scores.items(), key=lambda x: x[1])[0] if source_scores else ""

                ranked_articles = sorted(
                    article_candidates.items(),
                    key=lambda kv: (
                        1 if preferred_source and kv[1].get("source") == preferred_source else 0,
                        1 if kv[1].get("has_heading") else 0,
                        float(kv[1].get("score") or 0.0),
                    ),
                    reverse=True,
                )

                top_score = float(ranked_articles[0][1].get("score") or 0.0)
                items: list[str] = []
                for number, data in ranked_articles:
                    score = float(data.get("score") or 0.0)
                    if top_score > 0 and score < (top_score * 0.62):
                        continue
                    items.append(f"articulo {number} [{int(data.get('idx') or 1)}]")
                    if len(items) >= 5:
                        break
                if len(items) < 3:
                    for number, data in ranked_articles:
                        item = f"articulo {number} [{int(data.get('idx') or 1)}]"
                        if item in items:
                            continue
                        items.append(item)
                        if len(items) >= 3:
                            break
                if items:
                    return "Los articulos relevantes son: " + ", ".join(items) + "."

        if signals.get("asks_exclusion") or intent == "exclusion":
            exclusion_pattern = r"\bexclu(?:ye|yen|id[oa]s?|ir|ira|iran|sion(?:es)?)\b|\bno\s+sera\s+de\s+aplicacion\b"
            candidates: list[tuple[float, int, int, GraphChunk, str]] = []
            for idx, chunk in enumerate(chunks, start=1):
                text_norm = _normalize_for_search(chunk.text or "")
                if not text_norm:
                    continue
                if not re.search(exclusion_pattern, text_norm):
                    continue
                heading_norm = _normalize_for_search((chunk.text or "")[:280])
                heading_boost = 1.0 if re.search(r"\barticulo\s+\d", heading_norm) else 0.0
                if re.search(r"\bentidad(?:es)?\b", heading_norm) and re.search(exclusion_pattern, heading_norm):
                    heading_boost += 1.2
                list_items = extract_list_items(chunk.text)
                semantic_hits = sum(1 for root in broad_question_roots if root in heading_norm)
                semantic_hits += sum(1 for root in broad_question_roots if root in text_norm) * 0.25
                article = _normalize_article_number((chunk.metadata or {}).get("numero")) or self._article_from_text(chunk.text) or ""
                candidate_score = heading_boost + float(semantic_hits) + (0.20 * float(len(list_items)))
                candidates.append((candidate_score, semantic_hits, idx, chunk, article))
            if candidates:
                candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
                _, _, best_idx, best_chunk, article = candidates[0]
                items = extract_list_items(best_chunk.text)
                if items:
                    joined = "; ".join(items[:4])
                    if article:
                        return f"Respuesta parcial (exclusion): articulo {article}: {joined}. [{best_idx}]"
                    return f"Respuesta parcial (exclusion): {joined}. [{best_idx}]"
                sentence = first_sentence(best_chunk.text)
                if sentence:
                    if article:
                        return f"Respuesta parcial (exclusion): articulo {article}: {sentence} [{best_idx}]"
                    return f"Respuesta parcial (exclusion): {sentence} [{best_idx}]"

        if intent == "comparison" and comparison_terms:
            candidates: list[tuple[int, int, GraphChunk]] = []
            for idx, chunk in enumerate(chunks, start=1):
                text_norm = _normalize_for_search(chunk.text or "")
                if not text_norm:
                    continue
                hits = {term for term in comparison_terms if term in text_norm}
                if len(hits) < 2:
                    continue
                candidates.append((len(hits), idx, chunk))
            if candidates:
                candidates.sort(key=lambda item: item[0], reverse=True)
                _, best_idx, best_chunk = candidates[0]
                sentence = first_sentence(best_chunk.text)
                if sentence:
                    article = _normalize_article_number((best_chunk.metadata or {}).get("numero")) or self._article_from_text(best_chunk.text) or ""
                    if article:
                        return f"Comparacion parcial (contexto recuperado): articulo {article}: {sentence} [{best_idx}]"
                    return f"Comparacion parcial (contexto recuperado): {sentence} [{best_idx}]"

        if signals.get("asks_modal"):
            asks_negative_modal = bool(re.search(r"\b(no\s+puede|no\s+permite|prohibe|prohibido)\b", normalized_question))
            modal_terms = _dedupe(
                [
                    t
                    for t in content_terms
                    if len(t) >= 5 and " " not in t and t in question_term_set and t not in GENERIC_QUERY_TERMS
                ]
            )
            if len(modal_terms) >= 2:
                long_modal_terms = [t for t in modal_terms if len(t) >= 9]
                distinctive_modal_terms = [t for t in modal_terms if (" " in t) or len(t) >= 10]
                if not distinctive_modal_terms:
                    distinctive_modal_terms = long_modal_terms
                if len(distinctive_modal_terms) > 2:
                    distinctive_modal_terms = sorted(distinctive_modal_terms, key=len, reverse=True)[:2]
                modal_roots = _dedupe([t[:5] for t in modal_terms if len(t) >= 5])
                long_modal_roots = _dedupe([t[:6] for t in long_modal_terms if len(t) >= 6])
                distinctive_modal_roots = _dedupe([t[:6] for t in distinctive_modal_terms if len(t) >= 6 and " " not in t])
                candidates: list[tuple[float, float, float, int, GraphChunk, str]] = []
                for idx, chunk in enumerate(chunks, start=1):
                    text_norm = _normalize_for_search(chunk.text or "")
                    if not text_norm:
                        continue
                    hits = sum(1 for term in modal_terms if term in text_norm)
                    root_hits = sum(1 for root in modal_roots if root in text_norm)
                    effective_hits = hits + (0.6 * root_hits)
                    coverage = effective_hits / max(len(modal_terms), 1)
                    threshold = 0.40 if long_modal_terms else 0.67
                    if coverage < threshold:
                        continue
                    long_hits = sum(1 for term in long_modal_terms if term in text_norm)
                    long_root_hits = sum(1 for root in long_modal_roots if root in text_norm)
                    if long_modal_terms and (long_hits + long_root_hits) <= 0:
                        continue
                    distinct_hits = sum(1 for term in distinctive_modal_terms if term in text_norm)
                    distinct_root_hits = sum(1 for root in distinctive_modal_roots if root in text_norm)
                    if distinctive_modal_terms and (distinct_hits + distinct_root_hits) <= 0:
                        continue
                    pos_cues = float(
                        len(re.findall(r"\b(podra|podran|puede|pueden|se\s+permite)\b", text_norm))
                    )
                    neg_cues = float(
                        len(re.findall(r"\b(no\s+podra|no\s+podran|prohibe|prohibido|no\s+se\s+permite)\b", text_norm))
                    )
                    cue_score = (neg_cues - pos_cues) if asks_negative_modal else (pos_cues - neg_cues)
                    if cue_score <= 0:
                        continue
                    article = _normalize_article_number((chunk.metadata or {}).get("numero")) or self._article_from_text(chunk.text) or ""
                    candidates.append((coverage, cue_score, float(chunk.score or 0.0), idx, chunk, article))
                if candidates:
                    candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
                    _, _, _, best_idx, best_chunk, article = candidates[0]
                    sentence = first_sentence(best_chunk.text)
                    if sentence:
                        if article:
                            return f"Respuesta parcial (modal): articulo {article}: {sentence} [{best_idx}]"
                        return f"Respuesta parcial (modal): {sentence} [{best_idx}]"

        if signals.get("enumerative_need"):
            focus_terms = _dedupe([t for t in content_terms if len(t) >= 5 and t in question_term_set])
            if len(focus_terms) < 2:
                return None
            enum_candidates: list[tuple[int, int, GraphChunk, str]] = []
            for idx, chunk in enumerate(chunks, start=1):
                text_norm = _normalize_for_search(chunk.text or "")
                if not text_norm:
                    continue
                heading_norm = _normalize_for_search((chunk.text or "")[:220])
                if not re.search(r"\barticulo\s+\d", heading_norm):
                    continue
                article = _normalize_article_number((chunk.metadata or {}).get("numero")) or self._article_from_text(chunk.text) or ""
                if not article:
                    continue
                hits = sum(1 for term in focus_terms if term in text_norm)
                coverage = hits / max(len(focus_terms), 1)
                if coverage < 0.60:
                    continue
                enum_candidates.append((hits, idx, chunk, article))
            if enum_candidates:
                enum_candidates.sort(key=lambda item: item[0], reverse=True)
                _, best_idx, best_chunk, article = enum_candidates[0]
                sentence = first_sentence(best_chunk.text)
                if sentence:
                    return f"Respuesta parcial (enumeracion): articulo {article}: {sentence} [{best_idx}]"

        return None

    def _partial_answer_for_extreme(self, question: str, chunks: list[GraphChunk]) -> str | None:
        if not chunks:
            return None
        signals = self._query_signals(question)
        if not signals.get("asks_extreme"):
            return None
        if signals.get("asks_article_numbers"):
            return None
        if signals.get("asks_single_article") and (signals.get("article_numbers") or []):
            return None
        if signals.get("asks_single_article") and not (signals.get("article_numbers") or []):
            return None
        content_terms = [str(t) for t in (signals.get("content_terms") or []) if t]
        normalized_question = str(signals.get("normalized_question") or "")
        question_terms = _dedupe(
            [t for t in _tokens(normalized_question) if t and t not in LEGAL_STOPWORDS and t not in GENERIC_QUERY_TERMS]
        )
        question_term_set = set(question_terms)
        entity_terms = {str(t) for t in (signals.get("entity_tokens") or []) if t}
        focus_terms = _dedupe(
            [
                t
                for t in content_terms
                if len(t) >= 5 and t in question_term_set and t not in entity_terms and t not in GENERIC_QUERY_TERMS
            ]
        )
        if len(focus_terms) < 2:
            return None
        extreme_tokens = {"maximo", "maxima", "minimo", "minima", "limite", "tope", "plazo"}
        theme_terms = [t for t in focus_terms if t not in extreme_tokens]
        if not theme_terms:
            theme_terms = [t for t in focus_terms if t not in {"maximo", "maxima", "minimo", "minima"}]
        if not theme_terms:
            return None

        def first_sentence(text: str) -> str:
            cleaned = _repair_visible_text(str(text or "")).replace("\n", " ").strip()
            parts = re.split(r"(?<=[\.!?])\s+", cleaned)
            for part in parts:
                if len(_normalize_for_search(part)) >= 24:
                    return part.strip()[:420]
            return cleaned[:420]

        def extract_list_items(text: str, max_items: int = 5) -> list[str]:
            items: list[str] = []
            for raw_line in (text or "").splitlines():
                line = _repair_visible_text(raw_line).strip()
                if not line:
                    continue
                norm = _normalize_for_search(line)
                if re.match(r"^[-*]\s+", line) or re.match(r"^[a-z]\)\s+", norm) or re.match(r"^\d+[.)]\s+", norm):
                    clean = re.sub(r"^[-*]\s+|^[a-z]\)\s+|^\d+[.)]\s+", "", line, flags=re.IGNORECASE).strip(" .;")
                    if len(_normalize_for_search(clean)) >= 12:
                        items.append(clean)
                if len(items) >= max_items:
                    break
            return _dedupe(items)

        candidates: list[tuple[float, int, GraphChunk, str]] = []
        extreme_hint_pattern = (
            r"\b(maxim|minim|tope|limite|cuantia|importe|sancion|sanciones|multa|suspension|revocacion|"
            r"grave|muy\s+grave|hasta|por\s+ciento|euros?|plazo)\b"
        )
        for idx, chunk in enumerate(chunks, start=1):
            text_norm = _normalize_for_search(chunk.text or "")
            if not text_norm:
                continue
            theme_hits = sum(1 for term in theme_terms if term in text_norm)
            theme_coverage = theme_hits / max(len(theme_terms), 1)
            if theme_coverage < 0.60:
                continue
            if len(theme_terms) >= 2 and theme_hits < 2:
                continue
            has_numeric = bool(NUMERIC_TOKEN_PATTERN.search(text_norm))
            has_extreme_hint = bool(re.search(extreme_hint_pattern, text_norm))
            if not (has_numeric or has_extreme_hint):
                continue
            score = float(theme_hits) + (0.6 if has_extreme_hint else 0.0) + (0.8 if has_numeric else 0.0)
            article = _normalize_article_number((chunk.metadata or {}).get("numero")) or self._article_from_text(chunk.text) or ""
            candidates.append((score, idx, chunk, article))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        _, best_idx, best_chunk, article = candidates[0]
        sentence = first_sentence(best_chunk.text)
        if not sentence:
            return None
        if article:
            return (
                "Respuesta parcial (extremo): "
                f"se recupera el articulo {article} con contenido relacionado, "
                "pero no se identifica en estos fragmentos un maximo/minimo unico explicito. "
                f"[{best_idx}]"
            )
        return (
            "Respuesta parcial (extremo): se recupera contenido relacionado, "
            "pero no se identifica en estos fragmentos un maximo/minimo unico explicito. "
            f"[{best_idx}]"
        )

    def _apply_modal_guardrails(self, question: str, answer: str, chunks: list[GraphChunk]) -> str:
        if not answer or not chunks:
            return answer
        signals = self._query_signals(question)
        if not signals.get("asks_modal"):
            return answer

        answer_norm = _normalize_for_search(answer)
        is_negative = bool(re.search(r"\b(no\s+permite|no\s+puede|prohib|no\s+se\s+permite)\b", answer_norm))
        is_positive = bool(re.search(r"\b(se\s+permite|puede|permitid|si\b)\b", answer_norm)) and not is_negative
        if not (is_negative or is_positive):
            return answer

        text_norm = " ".join(_normalize_for_search(chunk.text or "") for chunk in chunks[:6] if chunk.text)
        has_habilitante = bool(re.search(r"\b(se\s+podra|podra|podran|se\s+permite|pueden|autoriza)\b", text_norm))
        has_prohibitiva = bool(re.search(r"\b(no\s+podra|no\s+podran|queda\s+prohibid|prohibe|prohibido|no\s+se\s+permite)\b", text_norm))
        has_procedural = bool(re.search(r"\b(debera|deberan|notificar|comunicar|registro|procedimiento|servicios|medidas)\b", text_norm))

        cite = "".join(f"[{i}]" for i in range(1, min(3, len(chunks)) + 1))
        if is_negative and not has_prohibitiva and (has_habilitante or has_procedural):
            return (
                "Respuesta parcial (modal): los fragmentos recuperados regulan condiciones/procedimiento, "
                "pero no acreditan una prohibicion general explicita. "
                f"{cite}"
            ).strip()
        if is_positive and has_prohibitiva and not has_habilitante:
            return (
                "Respuesta parcial (modal): hay indicios de limites/prohibiciones en los fragmentos, "
                "sin base habilitante suficiente para una afirmacion general. "
                f"{cite}"
            ).strip()
        return answer

    def _apply_operation_guardrails(self, question: str, answer: str, chunks: list[GraphChunk]) -> str:
        if not answer or not chunks:
            return answer
        signals = self._query_signals(question)
        evidence_text = " ".join(_normalize_for_search(chunk.text or "") for chunk in chunks[:6] if chunk.text)
        answer_norm = _normalize_for_search(answer)
        normalized_question = str(signals.get("normalized_question") or _normalize_for_search(question))
        exclusion_pattern = r"\bexclu(?:ye|yen|id[oa]s?|ir|ira|iran|sion(?:es)?)\b"
        is_exclusion_query = bool(signals.get("asks_exclusion")) or bool(re.search(exclusion_pattern, normalized_question))

        def first_sentence(text: str) -> str:
            cleaned = _repair_visible_text(str(text or "")).replace("\n", " ").strip()
            parts = re.split(r"(?<=[\.!?])\s+", cleaned)
            for part in parts:
                if len(_normalize_for_search(part)) >= 24:
                    return part.strip()[:420]
            return cleaned[:420]

        # Exclusion answers require explicit exclusion evidence in the retrieved context.
        if is_exclusion_query:
            anchors: list[tuple[int, GraphChunk]] = []
            for idx, chunk in enumerate(chunks[:6], start=1):
                heading_norm = _normalize_for_search((chunk.text or "")[:260])
                if re.search(r"\barticulo\s+\d", heading_norm) and re.search(exclusion_pattern, heading_norm):
                    anchors.append((idx, chunk))
            if not anchors and not re.search(exclusion_pattern, evidence_text):
                return "NO ENCONTRADO EN EL DOCUMENTO"
            if anchors and not re.search(exclusion_pattern, answer_norm):
                idx, chunk = anchors[0]
                sentence = first_sentence(chunk.text)
                if sentence:
                    return f"Respuesta parcial (exclusion): {sentence} [{idx}]"

        if signals.get("asks_comparison"):
            comparison_entities = _dedupe(
                [
                    str(t)
                    for t in (
                        (signals.get("entity_tokens") or [])
                        + (signals.get("acronyms") or [])
                        + (signals.get("content_terms") or [])
                    )
                    if len(str(t)) >= 3 and str(t) not in GENERIC_QUERY_TERMS
                ]
            )[:6]
            if len(comparison_entities) >= 2:
                covered_entities = [ent for ent in comparison_entities if ent in evidence_text]
                if len(set(covered_entities)) < 2:
                    fallback = self._extractive_answer_by_intent(question=question, chunks=chunks)
                    if fallback and "comparacion parcial" in _normalize_for_search(fallback):
                        return fallback
                    cite = "".join(f"[{i}]" for i in range(1, min(3, len(chunks)) + 1))
                    return (
                        "Comparacion parcial (cobertura insuficiente): no hay anclaje dual suficiente "
                        "para ambas entidades en los fragmentos recuperados. "
                        f"{cite}"
                    ).strip()

        return answer

    def generate_from_chunks(self, question: str, chunks: list[GraphChunk], chat_history: list[dict] | None = None) -> str:
        self._last_generation_debug = {
            "not_found_reason": None,
            "grounding_trace": {},
            "grounding_summary": {},
            "claim_candidates_total": 0,
            "claim_candidates_filtered": 0,
            "claim_filter_reasons": {},
            "question_relevance_stats": {},
            "context_dump_blocked": False,
            "best_effort_reason": None,
            "response_mode": "affirmative",
        }

        if not chunks:
            self._last_generation_debug.update(
                {
                    "not_found_reason": {"code": "no_chunks"},
                    "response_mode": "not_found",
                }
            )
            return "NO ENCONTRADO EN EL DOCUMENTO"

        answer = self._compose_answer_llm(question=question, chunks=chunks, chat_history=chat_history).strip()
        answer = self._normalize_citation_tags(answer)
        if not answer:
            answer = "NO ENCONTRADO EN EL DOCUMENTO"

        signals = self._query_signals(question)
        if signals.get("asks_article_numbers"):
            deterministic = self._extractive_answer_by_intent(question=question, chunks=chunks)
            if deterministic and deterministic.lower().startswith("los articulos relevantes son:"):
                answer = deterministic
                self._last_generation_debug.update(
                    {
                        "response_mode": "answered",
                        "best_effort_reason": "article_centric_list",
                    }
                )

        if _normalize_for_search(answer).startswith("no encontrado en el documento"):
            extracted = self._extractive_answer_by_intent(question=question, chunks=chunks)
            if extracted:
                answer = extracted
                self._last_generation_debug.update(
                    {
                        "response_mode": "partial",
                        "best_effort_reason": "extractive_partial",
                    }
                )
            else:
                extreme_partial = self._partial_answer_for_extreme(question=question, chunks=chunks)
                if extreme_partial:
                    answer = extreme_partial
                    self._last_generation_debug.update(
                        {
                            "response_mode": "partial",
                            "best_effort_reason": "extreme_partial_without_explicit_bound",
                        }
                    )
                else:
                    self._last_generation_debug.update(
                        {
                            "not_found_reason": {"code": "llm_or_deterministic_not_found"},
                            "response_mode": "not_found",
                        }
                    )
                    return "NO ENCONTRADO EN EL DOCUMENTO"

        if _normalize_for_search(answer).startswith("no encontrado en el documento"):
            self._last_generation_debug.update(
                {
                    "not_found_reason": {"code": "llm_or_deterministic_not_found"},
                    "response_mode": "not_found",
                }
            )
            return "NO ENCONTRADO EN EL DOCUMENTO"

        guarded_answer = self._apply_modal_guardrails(question=question, answer=answer, chunks=chunks)
        if guarded_answer != answer:
            answer = guarded_answer
            self._last_generation_debug.update(
                {
                    "response_mode": "partial",
                    "best_effort_reason": "modal_guardrail_partial",
                }
            )

        op_guarded_answer = self._apply_operation_guardrails(question=question, answer=answer, chunks=chunks)
        if op_guarded_answer != answer:
            answer = op_guarded_answer

        if _normalize_for_search(answer).startswith("no encontrado en el documento"):
            self._last_generation_debug.update(
                {
                    "not_found_reason": {"code": "operation_guardrail_not_satisfied"},
                    "response_mode": "not_found",
                }
            )
            return "NO ENCONTRADO EN EL DOCUMENTO"

        if self._last_generation_debug.get("response_mode") == "affirmative":
            answer_norm = _normalize_for_search(answer)
            if "respuesta parcial" in answer_norm or "comparacion parcial" in answer_norm:
                self._last_generation_debug["response_mode"] = "partial"
            else:
                self._last_generation_debug["response_mode"] = "answered"

        cited_indices = [int(m) for m in re.findall(r"\[(\d+)\]", answer)]
        max_items = max(cited_indices) if cited_indices else None
        evidence = self._evidence_block(chunks, max_items=max_items, indices=cited_indices or None)
        return f"{answer}\n\n{evidence}".strip() if evidence else answer

    def _select_answer_chunks(self, question: str, chunks: list[GraphChunk]) -> list[GraphChunk]:
        if not chunks:
            return []

        signals = self._query_signals(question)
        requested_articles = [str(a) for a in (signals.get("article_numbers") or []) if a]
        limit = 6 if (signals.get("asks_article_numbers") or signals.get("asks_comparison")) else 5
        if signals.get("asks_exclusion") or signals.get("asks_modal") or signals.get("asks_extreme"):
            limit = max(limit, 6)
        limit = max(1, min(limit, len(chunks)))

        def article_key(chunk: GraphChunk) -> str:
            return _normalize_article_number((chunk.metadata or {}).get("numero")) or self._article_from_text(chunk.text) or ""

        if not requested_articles:
            if signals.get("asks_article_numbers"):
                generic_article_terms = {
                    "articulo",
                    "articulos",
                    "regimen",
                    "juridico",
                    "juridica",
                    "juridicas",
                    "ley",
                    "norma",
                    "regula",
                    "regulan",
                }
                focus_terms = _dedupe([str(t) for t in (signals.get("content_terms") or []) if len(str(t)) >= 4])
                question_focus_terms = [
                    t
                    for t in _tokens(str(signals.get("normalized_question") or ""))
                    if len(t) >= 5 and t not in generic_article_terms and t not in LEGAL_STOPWORDS
                ]
                focus_roots = _dedupe([term[:5] for term in focus_terms if len(term) >= 5])
                distinctive_terms = _dedupe([term for term in (focus_terms + question_focus_terms) if term not in generic_article_terms])
                if not distinctive_terms:
                    distinctive_terms = list(focus_terms)
                distinctive_roots = _dedupe([term[:5] for term in distinctive_terms if len(term) >= 5])
                entity_terms = _dedupe([str(t) for t in (signals.get("entity_tokens") or []) if len(str(t)) >= 3])

                article_best: dict[str, tuple[float, GraphChunk, str, bool]] = {}
                for chunk in chunks:
                    number = article_key(chunk)
                    if not number:
                        continue
                    text_norm = _normalize_for_search(chunk.text or "")
                    if not text_norm:
                        continue
                    focus_window = text_norm[:1600]
                    raw_text = chunk.text or ""
                    heading_line = next((ln.strip() for ln in raw_text.splitlines() if ln and ln.strip()), raw_text[:140])
                    heading_norm = _normalize_for_search(heading_line)
                    has_heading = bool(re.search(r"\barticulo\s+\d", heading_norm))
                    term_hits = sum(1 for term in focus_terms if term in focus_window)
                    root_hits = sum(1 for root in focus_roots if root in focus_window)
                    heading_term_hits = sum(1 for term in focus_terms if term in heading_norm)
                    heading_root_hits = sum(1 for root in focus_roots if root in heading_norm)
                    distinct_hits = sum(1 for term in distinctive_terms if term in focus_window)
                    distinct_root_hits = sum(1 for root in distinctive_roots if root in focus_window)
                    heading_distinct_hits = sum(1 for term in distinctive_terms if term in heading_norm)
                    heading_distinct_root_hits = sum(1 for root in distinctive_roots if root in heading_norm)
                    heading_entity_hits = sum(1 for ent in entity_terms if ent in heading_norm)
                    entity_hits = sum(1 for ent in entity_terms if ent in focus_window)
                    if not has_heading:
                        continue
                    if entity_terms and entity_hits == 0 and heading_entity_hits == 0:
                        continue
                    if (heading_distinct_hits + heading_distinct_root_hits + distinct_hits + distinct_root_hits + heading_entity_hits) == 0:
                        continue
                    if has_heading and (heading_term_hits + heading_root_hits) == 0 and term_hits < 2:
                        continue
                    score = (2.0 if has_heading else 0.0) + (1.1 * float(heading_term_hits)) + (0.7 * float(heading_root_hits))
                    score += float(term_hits) + (0.55 * float(root_hits)) + (0.75 * float(entity_hits))
                    score += (1.5 * float(heading_distinct_hits)) + (0.9 * float(heading_distinct_root_hits))
                    score += (0.8 * float(distinct_hits)) + (0.45 * float(distinct_root_hits))
                    score += 0.18 * float(chunk.score or 0.0)
                    if len(text_norm) > 4500:
                        score *= 0.52
                    elif len(text_norm) > 3200:
                        score *= 0.68
                    source = str(chunk.source or "")
                    current = article_best.get(number)
                    if current is None or score > current[0]:
                        article_best[number] = (score, chunk, source, has_heading)

                if article_best:
                    source_weights: dict[str, float] = {}
                    for score, _, source, _ in article_best.values():
                        if source:
                            source_weights[source] = source_weights.get(source, 0.0) + score
                    preferred_source = max(source_weights.items(), key=lambda x: x[1])[0] if source_weights else ""

                    ranked = sorted(
                        article_best.items(),
                        key=lambda kv: (
                            1 if preferred_source and kv[1][2] == preferred_source else 0,
                            1 if kv[1][3] else 0,
                            kv[1][0],
                        ),
                        reverse=True,
                    )
                    top_score = ranked[0][1][0]
                    selected = [entry[1][1] for entry in ranked if entry[1][0] >= max(top_score * 0.62, 1.6)][:limit]
                    if len(selected) < limit:
                        for chunk in chunks:
                            if len(selected) >= limit:
                                break
                            if chunk in selected:
                                continue
                            selected.append(chunk)
                    return selected[:limit]

            if signals.get("asks_comparison"):
                comparison_entities = _dedupe(
                    [
                        str(t)
                        for t in (
                            (signals.get("entity_tokens") or [])
                            + (signals.get("acronyms") or [])
                            + (signals.get("content_terms") or [])
                        )
                        if len(str(t)) >= 3 and str(t) not in GENERIC_QUERY_TERMS
                    ]
                )[:6]
                if len(comparison_entities) >= 2:
                    selected: list[GraphChunk] = []
                    per_entity: dict[str, tuple[float, GraphChunk]] = {}
                    bridge: tuple[float, GraphChunk] | None = None

                    for chunk in chunks:
                        text_norm = _normalize_for_search(chunk.text or "")
                        if not text_norm:
                            continue
                        matched = [ent for ent in comparison_entities if ent in text_norm]
                        if not matched:
                            continue
                        score = float(len(matched)) + (0.25 * float(chunk.score or 0.0))
                        if len(matched) >= 2 and (bridge is None or score > bridge[0]):
                            bridge = (score, chunk)
                        for ent in matched:
                            current = per_entity.get(ent)
                            if current is None or score > current[0]:
                                per_entity[ent] = (score, chunk)

                    if bridge is not None:
                        selected.append(bridge[1])
                    for ent in comparison_entities:
                        if len(selected) >= limit:
                            break
                        best = per_entity.get(ent)
                        if best is None:
                            continue
                        if best[1] in selected:
                            continue
                        selected.append(best[1])
                    if selected:
                        for chunk in chunks:
                            if len(selected) >= limit:
                                break
                            if chunk in selected:
                                continue
                            selected.append(chunk)
                        return selected[:limit]

            if (
                signals.get("asks_modal")
                or signals.get("asks_extreme")
                or signals.get("enumerative_need")
                or signals.get("asks_exclusion")
            ):
                focus_terms = _dedupe(
                    [str(t) for t in (signals.get("content_terms") or []) if len(str(t)) >= 5 and str(t) not in GENERIC_QUERY_TERMS]
                )
                focus_roots = _dedupe([term[:5] for term in focus_terms if len(term) >= 5])
                long_terms = [term for term in focus_terms if len(term) >= 9]
                long_roots = _dedupe([term[:6] for term in long_terms if len(term) >= 6])
                distinctive_terms = [term for term in focus_terms if (" " in term) or len(term) >= 10]
                if not distinctive_terms:
                    distinctive_terms = long_terms
                if len(distinctive_terms) > 2:
                    distinctive_terms = sorted(distinctive_terms, key=len, reverse=True)[:2]
                distinctive_roots = _dedupe([term[:6] for term in distinctive_terms if len(term) >= 6 and " " not in term])
                exclusion_pattern = r"\bexclu(?:ye|yen|id[oa]s?|ir|ira|iran|sion(?:es)?)\b|\bno\s+sera\s+de\s+aplicacion\b"
                extreme_pattern = (
                    r"\b(maxim|minim|tope|limite|cuantia|importe|sancion|sanciones|multa|"
                    r"suspension|revocacion|grave|muy\s+grave|hasta|por\s+ciento|euros?|plazo)\b"
                )
                scored: list[tuple[float, float, GraphChunk]] = []
                for chunk in chunks:
                    text_norm = _normalize_for_search(chunk.text or "")
                    if not text_norm:
                        continue
                    heading_norm = _normalize_for_search((chunk.text or "")[:280])
                    base = float(chunk.score or 0.0)
                    exact_hits = sum(1 for term in focus_terms if term in text_norm)
                    root_hits = sum(1 for root in focus_roots if root in text_norm)
                    long_hits = sum(1 for term in long_terms if term in text_norm)
                    long_root_hits = sum(1 for root in long_roots if root in text_norm)
                    distinct_hits = sum(1 for term in distinctive_terms if term in text_norm)
                    distinct_root_hits = sum(1 for root in distinctive_roots if root in text_norm)

                    rank = (0.45 * base) + float(exact_hits) + (0.55 * float(root_hits))
                    rank += (0.65 * float(long_hits)) + (0.35 * float(long_root_hits))
                    rank += (0.85 * float(distinct_hits)) + (0.40 * float(distinct_root_hits))

                    if signals.get("enumerative_need"):
                        if re.search(r"\barticulo\s+\d", heading_norm):
                            rank += 0.75
                        if re.search(r"(^|\n)\s*[-*]\s+|(^|\n)\s*[a-z]\)\s+|(^|\n)\s*\d+[.)]\s+", chunk.text or "", flags=re.IGNORECASE):
                            rank += 0.55

                    if signals.get("asks_exclusion"):
                        has_exclusion = bool(re.search(exclusion_pattern, text_norm))
                        has_heading_exclusion = bool(re.search(r"\barticulo\s+\d", heading_norm) and re.search(exclusion_pattern, heading_norm))
                        if has_heading_exclusion:
                            rank += 1.50
                        elif has_exclusion:
                            rank += 0.90
                        else:
                            rank -= 1.20

                    if signals.get("asks_modal") and long_terms and (long_hits + long_root_hits) <= 0:
                        rank -= 0.90
                    if signals.get("asks_modal") and distinctive_terms and (distinct_hits + distinct_root_hits) <= 0:
                        rank -= 1.10

                    if signals.get("asks_extreme"):
                        if NUMERIC_TOKEN_PATTERN.search(text_norm):
                            rank += 0.90
                        if re.search(extreme_pattern, text_norm):
                            rank += 0.70

                    scored.append((rank, base, chunk))
                if scored:
                    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
                    diverse: list[GraphChunk] = []
                    seen_articles: set[str] = set()
                    for rank, _, chunk in scored:
                        if len(diverse) >= limit:
                            break
                        if rank <= -0.60:
                            continue
                        number = article_key(chunk)
                        if number and number in seen_articles:
                            continue
                        diverse.append(chunk)
                        if number:
                            seen_articles.add(number)
                    if signals.get("asks_modal") and distinctive_terms:
                        question_norm = str(signals.get("normalized_question") or "")
                        asks_negative_modal = bool(re.search(r"\b(no\s+puede|no\s+permite|prohibe|prohibido)\b", question_norm))

                        def has_distinctive(chunk: GraphChunk) -> bool:
                            text_norm = _normalize_for_search(chunk.text or "")
                            if not text_norm:
                                return False
                            has_distinctive_term = any(term in text_norm for term in distinctive_terms) or any(
                                root in text_norm for root in distinctive_roots
                            )
                            if not has_distinctive_term:
                                return False
                            if asks_negative_modal:
                                return bool(re.search(r"\b(no\s+podra|no\s+podran|prohibe|prohibido|no\s+se\s+permite)\b", text_norm))
                            return bool(re.search(r"\b(podra|podran|puede|pueden|se\s+permite|no\s+podra|no\s+podran)\b", text_norm))

                        if not any(has_distinctive(chunk) for chunk in diverse):
                            for _, _, chunk in scored:
                                if not has_distinctive(chunk):
                                    continue
                                if chunk in diverse:
                                    break
                                diverse = [chunk] + diverse
                                break
                    if signals.get("asks_exclusion"):
                        has_anchor = any(
                            re.search(exclusion_pattern, _normalize_for_search((chunk.text or "")[:280]))
                            and re.search(r"\barticulo\s+\d", _normalize_for_search((chunk.text or "")[:280]))
                            for chunk in diverse
                        )
                        if not has_anchor:
                            for _, _, chunk in scored:
                                heading_norm = _normalize_for_search((chunk.text or "")[:280])
                                if not re.search(r"\barticulo\s+\d", heading_norm):
                                    continue
                                if not re.search(exclusion_pattern, heading_norm):
                                    continue
                                if chunk in diverse:
                                    has_anchor = True
                                    break
                                diverse = [chunk] + diverse
                                has_anchor = True
                                break
                    if len(diverse) < limit:
                        for chunk in chunks:
                            if len(diverse) >= limit:
                                break
                            if chunk in diverse:
                                continue
                            diverse.append(chunk)
                    if diverse:
                        return diverse[:limit]

            if signals.get("asks_article_numbers") or signals.get("asks_comparison"):
                diverse: list[GraphChunk] = []
                seen_articles: set[str] = set()
                for chunk in chunks:
                    if len(diverse) >= limit:
                        break
                    number = article_key(chunk)
                    if number and number in seen_articles:
                        continue
                    diverse.append(chunk)
                    if number:
                        seen_articles.add(number)
                if len(diverse) < limit:
                    for chunk in chunks:
                        if len(diverse) >= limit:
                            break
                        if chunk in diverse:
                            continue
                        diverse.append(chunk)
                return diverse[:limit]

            return chunks[:limit]

        prioritized: list[GraphChunk] = []
        for article in requested_articles:
            for chunk in chunks:
                number = article_key(chunk)
                if number == article and chunk not in prioritized:
                    prioritized.append(chunk)

        for chunk in chunks:
            if len(prioritized) >= limit:
                break
            if chunk in prioritized:
                continue
            prioritized.append(chunk)

        return prioritized[:limit]

    def query(self, question: str, top_k: int = 8, chat_history: list[dict] | None = None) -> GraphRAGResult:
        signals = self._query_signals(question)
        adjusted_top_k = max(1, int(top_k))
        if signals.get("asks_article_numbers") or signals.get("article_numbers") or signals.get("asks_comparison"):
            adjusted_top_k = min(max(adjusted_top_k, 12), 20)
        else:
            adjusted_top_k = min(max(adjusted_top_k, 8), 16)
        chunks = self.search_units(question=question, top_k=adjusted_top_k)
        if not chunks:
            return GraphRAGResult(answer="NO ENCONTRADO EN EL DOCUMENTO", chunks=[], sources=[])
        answer_chunks = self._select_answer_chunks(question=question, chunks=chunks)
        answer = self.generate_from_chunks(question=question, chunks=answer_chunks, chat_history=chat_history)
        return GraphRAGResult(answer=answer, chunks=answer_chunks, sources=list({c.source for c in answer_chunks}))

    def find_bridge_states(self, unit_ids: list[str]) -> list[dict[str, Any]]:
        if not unit_ids:
            return []

        results: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()

        with self.driver.session(database=NEO4J_DATABASE) as session:
            rows = session.run(
                """
                MATCH (u:UnidadNormativa)-[:MENTIONS_STATE|IMPACTA_ESTADO]->(s:EstadoFinanciero)
                WHERE u.id IN $unit_ids
                RETURN u.id AS unit_id, s.id AS estado_id, coalesce(s.code, s.codigo, s.id) AS codigo_estado
                LIMIT 300
                """,
                {"unit_ids": unit_ids},
            ).data()

            for row in rows:
                uid = str(row.get("unit_id") or "")
                code = _normalize_for_search(str(row.get("codigo_estado") or ""))
                sid = str(row.get("estado_id") or f"estado_{code}")
                if not uid or not code:
                    continue
                key = (uid, code)
                if key in seen:
                    continue
                seen.add(key)
                results.append(
                    {
                        "unit_id": uid,
                        "estado_id": sid,
                        "codigo_estado": code,
                        "detalle": f"Mencion explicita a Estado {code.upper()}",
                    }
                )

            if results:
                return results

            # Fallback: derive from text if relation was not created.
            text_rows = session.run(
                """
                MATCH (u:UnidadNormativa)
                WHERE u.id IN $unit_ids
                RETURN u.id AS unit_id, coalesce(u.text_norm, u.texto_norm, '') AS text_norm
                """,
                {"unit_ids": unit_ids},
            ).data()
            for row in text_rows:
                uid = str(row.get("unit_id") or "")
                text_norm = str(row.get("text_norm") or "")
                for code in _extract_state_codes(text_norm):
                    key = (uid, code)
                    if key in seen:
                        continue
                    seen.add(key)
                    results.append(
                        {
                            "unit_id": uid,
                            "estado_id": f"estado_{code}",
                            "codigo_estado": code,
                            "detalle": f"Mencion explicita a Estado {code.upper()}",
                        }
                    )
        return results

    def get_last_debug_trace(self) -> dict[str, Any]:
        return {
            "retrieval_strategies": self._last_search_debug.get("retrieval_strategies", {}),
            "retrieval_metrics": self._last_search_debug.get("retrieval_metrics", {}),
            "rrf_scores": self._last_search_debug.get("rrf_scores", {}),
            "reranker_trace": self._last_search_debug.get("reranker_trace", {}),
            "grounding_trace": self._last_generation_debug.get("grounding_trace", {}),
            "grounding_summary": self._last_generation_debug.get("grounding_summary", {}),
            "claim_candidates_total": self._last_generation_debug.get("claim_candidates_total", 0),
            "claim_candidates_filtered": self._last_generation_debug.get("claim_candidates_filtered", 0),
            "claim_filter_reasons": self._last_generation_debug.get("claim_filter_reasons", {}),
            "question_relevance_stats": self._last_generation_debug.get("question_relevance_stats", {}),
            "context_dump_blocked": self._last_generation_debug.get("context_dump_blocked", False),
            "best_effort_reason": self._last_generation_debug.get("best_effort_reason"),
            "response_mode": self._last_generation_debug.get("response_mode", "affirmative"),
            "not_found_reason": self._last_generation_debug.get("not_found_reason"),
            "query_signals": self._last_search_debug.get("query_signals", {}),
        }

    def get_stats(self) -> dict[str, Any]:
        with self.driver.session(database=NEO4J_DATABASE) as session:
            docs = session.run("MATCH (d:DocumentoNormativo) RETURN count(d) AS c").single()["c"]
            units = session.run("MATCH (u:UnidadNormativa) RETURN count(u) AS c").single()["c"]
            states = session.run("MATCH (s:EstadoFinanciero) RETURN count(s) AS c").single()["c"]
        return {
            "documents": int(docs),
            "units": int(units),
            "states": int(states),
            "neo4j_uri": NEO4J_URI,
        }
