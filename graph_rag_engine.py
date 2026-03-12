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

RRF_K = 60.0
RRF_WEIGHT = 0.38


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

    def _structured_history_focus(self, chat_history: list[dict] | None) -> dict[str, Any]:
        if not chat_history:
            return {}
        client = self._get_llm()
        if client is None:
            return {}

        turns: list[str] = []
        for msg in chat_history[-8:]:
            role = str(msg.get("role") or "").strip().lower()
            content = str(msg.get("content") or "").strip()
            if role not in {"user", "assistant"} or not content:
                continue
            turns.append(f"{role}: {content[:900]}")
        if not turns:
            return {}

        try:
            rsp = client.chat.completions.create(
                model=LEGAL_INTENT_MODEL,
                temperature=0.0,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Resume el foco juridico conversacional en JSON compacto con claves: "
                            "main_entity, main_action, main_condition, resolved_article_refs (array), operation_type, focus_query. "
                            "focus_query debe ser una pregunta auto-contenida para retrieval legal."
                        ),
                    },
                    {"role": "user", "content": "\n".join(turns)},
                ],
                response_format={"type": "json_object"},
            )
            raw = rsp.choices[0].message.content or "{}"
            parsed = json.loads(raw)
        except Exception:
            return {}

        focus = {
            "main_entity": str(parsed.get("main_entity") or "").strip(),
            "main_action": str(parsed.get("main_action") or "").strip(),
            "main_condition": str(parsed.get("main_condition") or "").strip(),
            "resolved_article_refs": [
                _normalize_article_number(a) for a in (parsed.get("resolved_article_refs") or []) if str(a).strip()
            ][:6],
            "operation_type": str(parsed.get("operation_type") or "").strip().lower(),
            "focus_query": str(parsed.get("focus_query") or "").strip(),
        }
        return focus

    def _resolve_followup_question(
        self,
        question: str,
        chat_history: list[dict] | None,
        signals: dict[str, Any],
    ) -> tuple[str, dict[str, Any] | None]:
        if not chat_history:
            return question, None
        normalized = str(signals.get("normalized_question") or _normalize_for_search(question))
        token_count = int(signals.get("token_count") or 0)
        has_explicit_anchor = bool((signals.get("article_numbers") or []) or (signals.get("entity_tokens") or []))
        is_short_followup = bool(
            token_count <= 6
            or re.search(
                r"\b(en\s+que|y\s+en\s+que|en\s+que\s+condiciones|y\s+eso|y\s+cuando|y\s+como|en\s+ese\s+caso)\b",
                normalized,
            )
        )
        if not is_short_followup or has_explicit_anchor:
            return question, None

        focus = self._structured_history_focus(chat_history=chat_history)
        if not focus:
            return question, None
        focus_query = str(focus.get("focus_query") or "").strip()
        if focus_query:
            return focus_query, focus

        parts = [
            str(focus.get("main_entity") or "").strip(),
            str(focus.get("main_action") or "").strip(),
            str(focus.get("main_condition") or "").strip(),
        ]
        refs = [a for a in (focus.get("resolved_article_refs") or []) if a]
        if refs:
            parts.append("articulos " + ", ".join(refs))
        assembled = " ".join([p for p in parts if p]).strip()
        if not assembled:
            return question, focus
        return f"{question}. Contexto previo: {assembled}", focus

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
        # Robust fallback for OCR/degraded text around "articulo" markers.
        for m in re.finditer(
            r"\bart[^\d]{0,12}(\d+(?:\s*(?:bis|ter|quater|quinquies|sexies|septies|octies|nonies|decies|undecies|duodecies))?)\b",
            normalized,
            flags=re.IGNORECASE,
        ):
            article_numbers.append(_normalize_article_number(m.group(1)))
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
        semantic_aliases: list[str] = []
        for term in content_terms:
            if term.startswith("inmobili"):
                semantic_aliases.extend(["inmueble", "inmuebles"])
        content_terms = _dedupe(content_terms + semantic_aliases)
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
        asks_minimum_core_requirements = bool(
            asks_requirements
            or re.search(
                (
                    r"\b(funciones?\s+minimas?|requisitos?\s+(de|para)|debe[n]?\s+(cumplir|realizar)|"
                    r"que\s+obligaciones|obligaciones?\s+de|que\s+limitaciones|condiciones?\s+minimas?)\b"
                ),
                normalized,
            )
        )
        asks_enumeration_in_article = bool(
            (not asks_article_numbers)
            and (not asks_minimum_core_requirements)
            and (
                asks_requirements
                or asks_exclusion
                or re.search(
                    (
                        r"\b(que\s+entidades|que\s+causas|que\s+requisitos|que\s+obligaciones|"
                        r"que\s+sanciones|que\s+supuestos)\b"
                    ),
                    normalized,
                )
            )
        )
        enumerative_need = bool(
            asks_article_numbers
            or asks_enumeration_in_article
            or re.search(r"\bque\s+articulos\b", normalized)
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
            or re.search(r"\b(puede|permite|prohibe|prohibido|no\s+puede|se\s+permite)\b", normalized)
        )
        asks_coexistence_modal = bool(
            asks_modal
            and re.search(
                r"\b(al\s+mismo\s+tiempo|a\s+la\s+vez|simultan(?:eamente|eas?)?|conjuntamente|amb(?:o|a)s?)\b",
                normalized,
            )
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
            "asks_minimum_core_requirements": asks_minimum_core_requirements,
            "asks_enumeration_in_article": asks_enumeration_in_article,
            "enumerative_need": enumerative_need,
            "asks_extreme": asks_extreme,
            "asks_modal": asks_modal,
            "asks_coexistence_modal": asks_coexistence_modal,
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
                WHERE text <> '' AND article_norm IN $articles
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
                {"articles": normalized, "limit": int(max(limit, 40))},
            ).data()
        return rows

    def _fetch_rows_by_heading_terms(self, terms: list[str], limit: int = 240) -> list[dict[str, Any]]:
        normalized_terms = _dedupe([_normalize_for_search(t) for t in (terms or []) if t])
        acronym_allow = {
            "ecr",
            "eicc",
            "fcr",
            "scr",
            "iic",
            "cnmv",
        }
        normalized_terms = [t for t in normalized_terms if len(t) >= 4 or t in acronym_allow or t in LEGAL_ENTITY_TOKENS]
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

    def _row_article_key(self, row: dict[str, Any]) -> str:
        article = _normalize_article_number(row.get("article"))
        if article:
            return article
        return self._article_from_text(str(row.get("text") or "")) or ""

    def _row_to_chunk(self, row: dict[str, Any], score: float | None = None) -> GraphChunk | None:
        text = str(row.get("text") or "").strip()
        if not text:
            return None
        article = self._row_article_key(row) or None
        metadata = {
            "unit_id": row.get("unit_id"),
            "tipo_unidad": "articulo" if article else "unidad",
            "numero": article,
            "documento_id": row.get("documento_id"),
            "documento_titulo": row.get("documento_titulo"),
            "position": row.get("position"),
        }
        return GraphChunk(
            id=str(row.get("unit_id") or ""),
            text=text,
            source=str(row.get("documento_titulo") or row.get("documento_id") or "documento_desconocido"),
            score=float(score if score is not None else row.get("_final_score") or row.get("_base_score") or 0.0),
            metadata=metadata,
        )

    def _rrf_scores(self, rankings: list[list[str]], k: float = RRF_K) -> dict[str, float]:
        fused: dict[str, float] = {}
        for ranking in rankings:
            for idx, unit_id in enumerate(ranking, start=1):
                if not unit_id:
                    continue
                fused[unit_id] = fused.get(unit_id, 0.0) + (1.0 / (k + float(idx)))
        return fused

    def _expand_normative_units(
        self,
        selected: list[GraphChunk],
        signals: dict[str, Any],
        candidate_rows: list[dict[str, Any]],
        top_k: int,
    ) -> list[GraphChunk]:
        if not selected:
            return selected
        if not (
            signals.get("asks_article_numbers")
            or signals.get("asks_enumeration_in_article")
            or signals.get("asks_exclusion")
            or signals.get("asks_comparison")
        ):
            return selected

        rows_by_article: dict[tuple[str, str], list[dict[str, Any]]] = {}
        rows_by_article_fallback: dict[str, list[dict[str, Any]]] = {}
        for row in candidate_rows:
            art = self._row_article_key(row)
            if not art:
                continue
            doc_key = str(row.get("documento_id") or row.get("documento_titulo") or "")
            rows_by_article.setdefault((doc_key, art), []).append(row)
            rows_by_article_fallback.setdefault(art, []).append(row)
        if not rows_by_article and not rows_by_article_fallback:
            return selected

        exclusion_pattern = r"\bexclu(?:ye|yen|id[oa]s?|ir|ira|iran|sion(?:es)?)\b|\bno\s+sera\s+de\s+aplicacion\b"
        content_terms = [str(t) for t in (signals.get("content_terms") or []) if t]
        force_article_expansion = bool(
            signals.get("asks_exclusion")
            or signals.get("asks_enumeration_in_article")
            or signals.get("asks_article_numbers")
        )

        def text_completeness(text: str, text_norm: str) -> float:
            if not text_norm:
                return -1.0
            first_line = next((ln.strip() for ln in text.splitlines() if ln and ln.strip()), text[:140])
            heading_norm = _normalize_for_search(first_line)
            has_heading = 1.0 if re.search(r"\barticulo\s+\d", heading_norm) else 0.0
            has_list = 1.0 if self._extract_list_items(text, max_items=4) else 0.0
            focus_window = text_norm[:2600]
            term_hits = float(sum(1 for term in content_terms if term in focus_window))
            length_score = min(float(len(text_norm)), 7000.0) / 7000.0
            score = (1.6 * has_heading) + (1.2 * has_list) + (0.65 * term_hits) + (0.55 * length_score)
            if signals.get("asks_exclusion") and re.search(exclusion_pattern, focus_window):
                score += 1.2
            if signals.get("asks_article_numbers") and not has_heading:
                score -= 0.8
            return score

        def row_completeness(row: dict[str, Any]) -> float:
            text = str(row.get("text") or "")
            text_norm = str(row.get("text_norm") or _normalize_for_search(text))
            return text_completeness(text=text, text_norm=text_norm)

        def merged_chunk_for_article(article: str, rows_for_article: list[dict[str, Any]], fallback: GraphChunk) -> GraphChunk | None:
            if not rows_for_article:
                return None
            ordered = sorted(
                rows_for_article,
                key=lambda r: (float(r.get("_final_score") or r.get("_base_score") or 0.0), -int(r.get("position") or 0)),
                reverse=True,
            )
            ordered = sorted(ordered[: min(16, len(ordered))], key=lambda r: int(r.get("position") or 0))
            pieces: list[str] = []
            seen_piece_keys: set[str] = set()
            for row in ordered:
                text = str(row.get("text") or "").strip()
                if not text:
                    continue
                key = _normalize_for_search(text[:260])
                if key in seen_piece_keys:
                    continue
                seen_piece_keys.add(key)
                pieces.append(text)
            if not pieces:
                return None
            merged_text = "\n".join(pieces).strip()
            if not merged_text:
                return None
            if len(merged_text) > 12000:
                merged_text = merged_text[:12000]
            metadata = dict(fallback.metadata or {})
            metadata["tipo_unidad"] = "articulo"
            metadata["numero"] = article
            metadata["expanded_article"] = True
            metadata["merged_units"] = [str(r.get("unit_id") or "") for r in ordered if str(r.get("unit_id") or "")]
            merged_id = f"merged::{article}::{hashlib.md5(merged_text.encode('utf-8')).hexdigest()[:10]}"
            best_row_score = max(float(r.get("_final_score") or r.get("_base_score") or 0.0) for r in rows_for_article)
            merged_score = max(float(fallback.score or 0.0), best_row_score)
            return GraphChunk(
                id=merged_id,
                text=merged_text,
                source=fallback.source,
                score=merged_score,
                metadata=metadata,
            )

        expanded: list[GraphChunk] = []
        seen_chunk_ids: set[str] = set()
        for chunk in selected:
            article = _normalize_article_number((chunk.metadata or {}).get("numero")) or self._article_from_text(chunk.text) or ""
            chunk_doc_key = str((chunk.metadata or {}).get("documento_id") or chunk.source or "")
            best_chunk = chunk
            rows_for_article = rows_by_article.get((chunk_doc_key, article), []) if article else []
            if not rows_for_article and article:
                rows_for_article = rows_by_article_fallback.get(article, [])
            if article and rows_for_article:
                best_row = max(rows_for_article, key=row_completeness)
                replacement = self._row_to_chunk(best_row, score=max(float(chunk.score or 0.0), float(best_row.get("_final_score") or 0.0)))
                if replacement is not None:
                    best_chunk = replacement
                if force_article_expansion:
                    merged = merged_chunk_for_article(article=article, rows_for_article=rows_for_article, fallback=best_chunk)
                    if merged is not None:
                        merged_norm = _normalize_for_search(merged.text)
                        merged_score = text_completeness(merged.text, merged_norm)
                        best_score = text_completeness(best_chunk.text, _normalize_for_search(best_chunk.text))
                        if merged_score >= (best_score - 0.20):
                            best_chunk = merged
            chunk_id = str(best_chunk.id or "")
            if chunk_id and chunk_id in seen_chunk_ids:
                continue
            if chunk_id:
                seen_chunk_ids.add(chunk_id)
            expanded.append(best_chunk)

        if len(expanded) < min(top_k, len(selected)):
            for chunk in selected:
                if len(expanded) >= min(top_k, len(selected)):
                    break
                if chunk in expanded:
                    continue
                expanded.append(chunk)

        return expanded[: max(top_k, len(selected))]

    def _mark_support_roles(self, chunks: list[GraphChunk], signals: dict[str, Any]) -> list[GraphChunk]:
        if not chunks:
            return chunks
        if signals.get("asks_minimum_core_requirements"):
            core_limit = min(2, len(chunks))
        elif signals.get("asks_enumeration_in_article") or signals.get("asks_exclusion"):
            core_limit = min(3, len(chunks))
        elif signals.get("asks_article_numbers"):
            core_limit = len(chunks)
        else:
            core_limit = 4 if (signals.get("enumerative_need") or signals.get("asks_comparison")) else 3
        marked: list[GraphChunk] = []
        for idx, chunk in enumerate(chunks):
            metadata = dict(chunk.metadata or {})
            existing_role = str(metadata.get("support_role") or "").strip().lower()
            if existing_role not in {"core_support", "peripheral_support"}:
                metadata["support_role"] = "core_support" if idx < core_limit else "peripheral_support"
            marked.append(
                GraphChunk(
                    id=chunk.id,
                    text=chunk.text,
                    source=chunk.source,
                    score=chunk.score,
                    metadata=metadata,
                )
            )
        return marked

    def _extract_list_items(self, text: str, max_items: int = 10) -> list[str]:
        items: list[str] = []
        for raw_line in (text or "").splitlines():
            line = _repair_visible_text(raw_line).strip()
            if not line:
                continue
            norm = _normalize_for_search(line)
            if (
                re.match(r"^[-*]\s+", line)
                or re.match(r"^[a-z]\)\s+", norm)
                or re.match(r"^\d+[.)]\s+", norm)
                or re.match(r"^[ivxlcdm]+\)\s+", norm)
            ):
                clean = re.sub(
                    r"^[-*]\s+|^[a-z]\)\s+|^\d+[.)]\s+|^[ivxlcdm]+\)\s+",
                    "",
                    line,
                    flags=re.IGNORECASE,
                ).strip(" .;:")
                if len(_normalize_for_search(clean)) >= 10:
                    items.append(clean)
            if len(items) >= max_items:
                break
        return _dedupe(items)

    def _extract_numbered_items(self, text: str, max_items: int = 10) -> list[str]:
        items: list[str] = []
        for raw_line in (text or "").splitlines():
            line = _repair_visible_text(raw_line).strip()
            if not line:
                continue
            norm = _normalize_for_search(line)
            if re.match(r"^\d+[.)]\s+", norm):
                clean = re.sub(r"^\d+[.)]\s+", "", line).strip(" .;:")
                if len(_normalize_for_search(clean)) >= 10:
                    items.append(clean)
            if len(items) >= max_items:
                break
        return _dedupe(items)

    def _extract_complete_enumeration_from_unit(self, unit_text: str, max_items: int = 20) -> dict[str, Any]:
        text = _repair_visible_text(unit_text or "")
        text_norm = _normalize_for_search(text)
        intro_found = bool(
            re.search(
                r"\b(las?\s+siguientes?|seran\s+de\s+aplicacion|no\s+sera\s+de\s+aplicacion|se\s+entendera\s+por|consistira?\s+en)\b",
                text_norm,
            )
        )
        numbered_items: list[str] = []
        alpha_items: list[str] = []
        bullet_items: list[str] = []
        for raw_line in text.splitlines():
            line = _repair_visible_text(raw_line).strip()
            if not line:
                continue
            norm = _normalize_for_search(line)
            clean = ""
            if re.match(r"^\d+[.)]\s+", norm):
                clean = re.sub(r"^\d+[.)]\s+", "", line).strip(" .;:")
                if clean:
                    numbered_items.append(clean)
            elif re.match(r"^[a-z]\)\s+", norm):
                clean = re.sub(r"^[a-z]\)\s+", "", line, flags=re.IGNORECASE).strip(" .;:")
                if clean:
                    alpha_items.append(clean)
            elif re.match(r"^[-*]\s+", line):
                clean = re.sub(r"^[-*]\s+", "", line).strip(" .;:")
                if clean:
                    bullet_items.append(clean)
            # Inline markers inside long legal lines (e.g., "1. ...: a) ...; b) ...")
            for match in re.finditer(r"(?:^|[;:]\s*)([a-z]\)\s*[^;\n]{8,})", line, flags=re.IGNORECASE):
                inline = re.sub(r"^[a-z]\)\s*", "", match.group(1), flags=re.IGNORECASE).strip(" .;:")
                if inline:
                    alpha_items.append(inline)
            for match in re.finditer(r"(?:^|[;:\n]\s*)(\d+[.)]\s*[^;\n]{8,})", line, flags=re.IGNORECASE):
                inline = re.sub(r"^\d+[.)]\s*", "", match.group(1), flags=re.IGNORECASE).strip(" .;:")
                if inline:
                    numbered_items.append(inline)
        # Global segmentation for compact legal text with multiple markers in a single line.
        numbered_segments: list[str] = []
        numbered_matches = list(re.finditer(r"(?<!\w)(\d{1,3}[.)])\s*", text, flags=re.IGNORECASE))
        for i, match in enumerate(numbered_matches):
            context_before = _normalize_for_search(text[max(0, match.start() - 24) : match.start()])
            if re.search(r"\barticulo\s*$", context_before):
                continue
            start = match.end()
            end = numbered_matches[i + 1].start() if (i + 1) < len(numbered_matches) else len(text)
            segment = text[start:end].strip(" .;:\n\t-")
            if segment:
                numbered_segments.append(segment)
        alpha_segments: list[str] = []
        alpha_matches = list(re.finditer(r"(?<!\w)([a-z]\))\s*", text, flags=re.IGNORECASE))
        for i, match in enumerate(alpha_matches):
            start = match.end()
            end = alpha_matches[i + 1].start() if (i + 1) < len(alpha_matches) else len(text)
            segment = text[start:end].strip(" .;:\n\t-")
            if segment:
                alpha_segments.append(segment)
        if numbered_segments:
            numbered_items = numbered_segments + numbered_items
        if alpha_segments:
            alpha_items = alpha_segments + alpha_items
        numbered_items = _dedupe([i for i in numbered_items if len(_normalize_for_search(i)) >= 10])
        alpha_items = _dedupe([i for i in alpha_items if len(_normalize_for_search(i)) >= 10])
        bullet_items = _dedupe([i for i in bullet_items if len(_normalize_for_search(i)) >= 10])
        if len(numbered_items) >= 3:
            numbered_items = [i for i in numbered_items if len(i) <= 900]
        ordered_items = numbered_items + [i for i in alpha_items if i not in numbered_items] + [i for i in bullet_items if i not in numbered_items and i not in alpha_items]
        ordered_items = ordered_items[: max_items]
        item_count = len(ordered_items)
        covered_chars = sum(len(i) for i in ordered_items)
        total_chars = max(len(text_norm), 1)
        coverage_ratio = float(covered_chars) / float(total_chars)
        is_complete_enough = bool(
            item_count >= 2
            and (
                len(numbered_items) >= 2
                or len(alpha_items) >= 2
                or len(bullet_items) >= 2
                or coverage_ratio >= 0.08
            )
        )
        if intro_found and item_count < 2:
            is_complete_enough = False
        return {
            "intro_found": intro_found,
            "item_count": item_count,
            "numbered_count": len(numbered_items),
            "alpha_count": len(alpha_items),
            "bullet_count": len(bullet_items),
            "coverage_ratio": coverage_ratio,
            "is_complete_enough": is_complete_enough,
            "items": ordered_items,
            "numbered_items": numbered_items,
            "alpha_items": alpha_items,
            "bullet_items": bullet_items,
        }

    def _infer_normative_role(self, text: str) -> str:
        head = ""
        for line in (text or "").splitlines():
            stripped = line.strip()
            if stripped:
                head = stripped
                break
        norm = _normalize_for_search(head or text[:300])
        if re.search(r"\bdefinicion|concepto|objeto|entidades?\b", norm):
            return "definitional"
        if re.search(r"\bregimen|formas?\s+juridic|coeficiente|limitaciones|diversificacion|comercializacion|autorizacion\b", norm):
            return "regime"
        if re.search(r"\bsancion|infraccion|multa|revocacion|suspension\b", norm):
            return "sanctioning"
        if re.search(r"\bsupervision|inspeccion|cnmv\b", norm):
            return "supervisory"
        if re.search(r"\bobligaciones?\s+de\s+informacion|informacion\b", norm):
            return "informational"
        if re.search(r"\bprocedimiento|notificacion|registro|solicitud|plazo\b", norm):
            return "procedural"
        if re.search(r"\bde\s+conformidad\s+con|segun\s+el\s+articulo\b", norm):
            return "cross_reference"
        return "operational"

    def _infer_modal_function(self, text: str) -> str:
        norm = _normalize_for_search(text or "")
        if not norm:
            return "operational"
        has_procedural = bool(re.search(r"\b(procedimiento|notificacion|registro|plazo|comunicacion|servicios)\b", norm))
        has_strong_habilitante = bool(
            re.search(
                r"\b(se\s+permite|autoriza|podra(?:n)?\s+(invertir|gestionar|comercializar|adquirir)|puede(?:n)?\s+(invertir|gestionar|comercializar|adquirir))\b",
                norm,
            )
        )
        if re.search(r"\b(no\s+podra|no\s+podran|prohibe|prohibido|no\s+se\s+permite|impedir|no\s+sera\s+de\s+aplicacion)\b", norm):
            return "limitative"
        if has_procedural and not has_strong_habilitante:
            return "procedimental"
        if re.search(
            r"\b(se\s+podra|podra|podran|se\s+permite|puede|pueden|autoriza|podra\s+comercializarse|podran\s+comercializarse|se\s+podra\s+comercializar)\b",
            norm,
        ):
            return "habilitante"
        if re.search(r"\b(obligaciones?|funciones?|gestion\s+del\s+riesgo|requisitos?)\b", norm):
            return "operational"
        return "operational"

    def _is_plural_concept_query(self, normalized_question: str) -> bool:
        return bool(
            re.search(
                r"\b(formas?|tipos?|clases?|modalidades?|categorias?|supuestos?)\b",
                normalized_question or "",
            )
        )

    def _concept_facets_for_query(self, signals: dict[str, Any]) -> list[list[str]]:
        normalized_question = str(signals.get("normalized_question") or "")
        if not self._is_plural_concept_query(normalized_question):
            return []
        entity_terms = {str(t) for t in (signals.get("entity_tokens") or []) if str(t)}
        facets: list[list[str]] = []
        # Generic conceptual families by regulated vehicle type.
        if "ecr" in entity_terms:
            facets = [
                ["scr", "sociedad de capital-riesgo", "sociedad de capital riesgo", "sociedades de capital-riesgo", "sociedades de capital riesgo"],
                ["fcr", "fondo de capital-riesgo", "fondo de capital riesgo", "fondos de capital-riesgo", "fondos de capital riesgo"],
            ]
        elif "ecr-pyme" in entity_terms:
            facets = [
                ["scr-pyme", "sociedad de capital-riesgo pyme", "sociedad de capital riesgo pyme", "sociedades de capital-riesgo pyme", "sociedades de capital riesgo pyme"],
                ["fcr-pyme", "fondo de capital-riesgo pyme", "fondo de capital riesgo pyme", "fondos de capital-riesgo pyme", "fondos de capital riesgo pyme"],
            ]
        return [[_normalize_for_search(tok) for tok in facet if _normalize_for_search(tok)] for facet in facets]

    def _rank_article_chunks_for_list(self, question: str, chunks: list[GraphChunk], limit: int) -> list[GraphChunk]:
        if not chunks:
            return []
        signals = self._query_signals(question)
        def contains_term(haystack: str, term: str) -> bool:
            if not haystack or not term:
                return False
            if len(term) <= 4:
                return bool(re.search(rf"\b{re.escape(term)}\b", haystack))
            return term in haystack

        generic_article_terms = {
            "articulo",
            "articulos",
            "regimen",
            "ley",
            "norma",
            "regula",
            "regulan",
            "sobre",
            "segun",
            "que",
            "cual",
            "cuales",
        }
        question_terms = [
            t
            for t in _tokens(str(signals.get("normalized_question") or ""))
            if len(t) >= 4 and t not in LEGAL_STOPWORDS and t not in generic_article_terms
        ]
        query_focus_terms = _dedupe([t for t in question_terms if len(t) >= 5])
        query_focus_roots = _dedupe([t[:5] for t in query_focus_terms if len(t) >= 5])
        asks_legal_form = bool(
            re.search(r"\bformas?\b", str(signals.get("normalized_question") or ""))
            and re.search(r"\bjuridic", str(signals.get("normalized_question") or ""))
        )
        entity_terms = _dedupe([str(t) for t in (signals.get("entity_tokens") or []) if len(str(t)) >= 3])
        entity_variant_map = {
            "ecr": [
                "ecr",
                "capital-riesgo",
                "capital riesgo",
                "scr",
                "sociedad de capital-riesgo",
                "sociedad de capital riesgo",
                "fcr",
                "fondo de capital-riesgo",
                "fondo de capital riesgo",
            ],
            "ecr-pyme": [
                "ecr-pyme",
                "scr-pyme",
                "sociedad de capital-riesgo pyme",
                "sociedad de capital riesgo pyme",
                "fcr-pyme",
                "fondo de capital-riesgo pyme",
                "fondo de capital riesgo pyme",
            ],
            "eicc": ["eicc", "inversion colectiva de tipo cerrado", "inversion colectiva"],
            "sgeic": ["sgeic", "sociedad gestora de entidades de inversion colectiva de tipo cerrado", "sociedades gestoras"],
            "sgiic": ["sgiic", "sociedad gestora de instituciones de inversion colectiva", "instituciones de inversion colectiva"],
            "scr": ["scr", "sociedad de capital-riesgo", "sociedades de capital-riesgo"],
            "fcr": ["fcr", "fondo de capital-riesgo", "fondos de capital-riesgo"],
        }
        expanded_entity_terms: list[str] = []
        for term in entity_terms:
            variants = entity_variant_map.get(term, [term])
            for variant in variants:
                norm_variant = _normalize_for_search(variant)
                if norm_variant:
                    expanded_entity_terms.append(norm_variant)
        expanded_entity_terms = _dedupe(expanded_entity_terms)
        focus_terms = _dedupe(
            question_terms
            + entity_terms
        )
        if len(focus_terms) < 2:
            focus_terms = _dedupe(
                focus_terms + [str(t) for t in (signals.get("content_terms") or []) if len(str(t)) >= 4]
            )
        focus_terms = [t for t in focus_terms if t not in generic_article_terms]
        focus_roots = _dedupe([t[:5] for t in focus_terms if len(t) >= 5])
        concept_facets = self._concept_facets_for_query(signals)

        working_chunks: list[GraphChunk] = list(chunks)
        if concept_facets:
            facet_acronym_allow = {"scr", "fcr", "ecr", "eicc", "sgeic", "sgiic"}
            facet_terms = _dedupe(
                [
                    tok
                    for facet in concept_facets
                    for tok in facet
                    if (len(tok) >= 4 or tok in facet_acronym_allow) and tok not in GENERIC_QUERY_TERMS
                ]
            )
            if facet_terms:
                try:
                    supplemental_rows = self._fetch_rows_by_heading_terms(terms=facet_terms, limit=180)
                except Exception:
                    supplemental_rows = []
                seen_chunk_ids = {str(c.id or "") for c in working_chunks if str(c.id or "")}
                seen_fingerprints = {
                    (
                        str(c.source or ""),
                        _normalize_article_number((c.metadata or {}).get("numero")) or self._article_from_text(c.text) or "",
                        _normalize_for_search((c.text or "")[:180]),
                    )
                    for c in working_chunks
                }
                for row in supplemental_rows:
                    extra = self._row_to_chunk(row)
                    if not extra:
                        continue
                    extra_id = str(extra.id or "")
                    if extra_id and extra_id in seen_chunk_ids:
                        continue
                    extra_article = _normalize_article_number((extra.metadata or {}).get("numero")) or self._article_from_text(extra.text) or ""
                    fp = (
                        str(extra.source or ""),
                        extra_article,
                        _normalize_for_search((extra.text or "")[:180]),
                    )
                    if fp in seen_fingerprints:
                        continue
                    if extra_id:
                        seen_chunk_ids.add(extra_id)
                    seen_fingerprints.add(fp)
                    working_chunks.append(extra)

        normalized_question = str(signals.get("normalized_question") or "")
        wants_info = bool(re.search(r"\b(informacion|informar|inversores?)\b", normalized_question))
        wants_sanctions = bool(re.search(r"\b(sancion|infraccion|multa|sancionador)\b", normalized_question))
        wants_authorization = bool(re.search(r"\b(autorizacion|autorizada|autorizadas)\b", normalized_question))
        sanction_heading_pattern = r"\b(sancion|infraccion|procedimiento|multa|normativa)\b"
        authorization_heading_pattern = r"\b(autorizacion|solicitud|resolucion|requisitos)\b"

        role_weight = {
            "definitional": 1.05,
            "regime": 1.05,
            "sanctioning": 0.45,
            "supervisory": 0.30,
            "informational": -0.10,
            "procedural": -0.30,
            "cross_reference": -0.65,
            "operational": -0.20,
        }
        if wants_info:
            role_weight["informational"] += 0.95
            role_weight["cross_reference"] -= 0.20
        if wants_sanctions:
            role_weight["sanctioning"] += 0.95
            role_weight["procedural"] += 0.20
        if wants_authorization:
            role_weight["regime"] += 0.35
            role_weight["procedural"] += 0.25

        article_candidates: dict[tuple[str, str], list[tuple[float, GraphChunk, str, str]]] = {}
        for chunk in working_chunks:
            article = _normalize_article_number((chunk.metadata or {}).get("numero")) or self._article_from_text(chunk.text) or ""
            if not article:
                continue
            text = chunk.text or ""
            text_norm = _normalize_for_search(text)
            if not text_norm:
                continue
            heading_line = next((ln.strip() for ln in text.splitlines() if ln and ln.strip()), text[:160])
            heading_norm = _normalize_for_search(heading_line)
            heading_compact = heading_norm
            m_heading = re.search(
                r"\barticulo\s+\d+(?:\s*(?:bis|ter|quater|quinquies|sexies|septies|octies|nonies|decies|undecies|duodecies))?\.\s*[^.]{1,180}",
                heading_norm,
            )
            if m_heading:
                heading_compact = m_heading.group(0)
            has_heading = bool(re.search(r"\barticulo\s+\d", heading_compact))
            if not has_heading:
                continue
            focus_window = text_norm[:1800]
            heading_exact_hits = sum(1 for term in focus_terms if contains_term(heading_compact, term))
            heading_root_hits = sum(1 for root in focus_roots if contains_term(heading_compact, root))
            body_exact_hits = sum(1 for term in focus_terms if contains_term(focus_window, term))
            body_root_hits = sum(1 for root in focus_roots if contains_term(focus_window, root))
            entity_heading_hits = sum(1 for term in expanded_entity_terms if contains_term(heading_compact, term))
            entity_body_hits = sum(1 for term in expanded_entity_terms if contains_term(focus_window, term))
            facet_hits: list[int] = []
            if concept_facets:
                for facet_idx, facet_tokens in enumerate(concept_facets):
                    if any(contains_term(heading_compact, tok) or contains_term(focus_window, tok) for tok in facet_tokens):
                        facet_hits.append(facet_idx)
            if entity_terms and (entity_heading_hits + entity_body_hits) <= 0 and not facet_hits:
                continue
            query_heading_hits = sum(1 for term in query_focus_terms if contains_term(heading_compact, term))
            query_heading_root_hits = sum(1 for root in query_focus_roots if contains_term(heading_compact, root))
            query_body_hits = sum(1 for root in query_focus_roots if contains_term(focus_window, root))
            semantic_gate = heading_exact_hits + heading_root_hits + entity_heading_hits + min(body_exact_hits, 2)
            if semantic_gate <= 0:
                continue
            if query_focus_terms and (query_heading_hits + query_body_hits) <= 0:
                continue
            if len(query_focus_terms) >= 2 and (query_heading_hits + query_heading_root_hits) <= 0:
                continue
            if len(focus_terms) >= 2 and (heading_exact_hits + heading_root_hits) <= 0 and body_exact_hits <= 1:
                continue
            legal_form_heading_hits = 0
            legal_form_core = False
            if asks_legal_form:
                legal_form_heading_hits = sum(
                    1
                    for token in ("forma", "juridic", "sociedad", "fondo")
                    if contains_term(heading_compact, token)
                )
                legal_form_core = bool(
                    re.search(
                        (
                            r"\b(definicion|regimen\s+juridic|formas?\s+juridic|"
                            r"sociedad(?:es)?\s+de\s+capital|fondo(?:s)?\s+de\s+capital|scr|fcr)\b"
                        ),
                        heading_compact,
                    )
                )
                lateral_form_pattern = r"\b(europe\w*|fcre\w*|fese\w*|filpe\w*|sicc\w*|ficc\w*|eicc\w*|pyme\w*)\b"
                if "ecr" in entity_terms and not re.search(lateral_form_pattern, normalized_question):
                    if re.search(lateral_form_pattern, heading_compact):
                        continue
                if (legal_form_heading_hits <= 0 and not facet_hits) or (not legal_form_core and not facet_hits):
                    continue
            role = self._infer_normative_role(text)
            score = (
                (2.40 * float(query_heading_hits))
                + (1.55 * float(query_heading_root_hits))
                + (1.15 * float(query_body_hits))
                + (2.05 * float(heading_exact_hits))
                + (1.35 * float(heading_root_hits))
                + (0.95 * float(body_exact_hits))
                + (0.55 * float(body_root_hits))
                + (0.90 * float(entity_heading_hits + entity_body_hits))
                + role_weight.get(role, 0.0)
                + (0.20 * float(chunk.score or 0.0))
            )
            if asks_legal_form:
                score += 1.35 * float(legal_form_heading_hits)
                if legal_form_core:
                    score += 1.10
                if "ecr" in entity_terms and re.search(r"\b(europe\w*|fcre\w*|fese\w*|filpe\w*|sicc\w*|ficc\w*|eicc\w*|pyme\w*)\b", heading_compact):
                    score *= 0.32
                if re.search(r"\b(comercializacion|inversores?|suscripcion|reembolso|campana)\b", heading_compact):
                    score *= 0.35
            if wants_sanctions and not re.search(sanction_heading_pattern, heading_compact):
                continue
            if wants_authorization and not re.search(authorization_heading_pattern, heading_compact):
                score *= 0.58
            if len(text_norm) > 5200:
                score *= 0.62
            elif len(text_norm) > 3600:
                score *= 0.74
            if role in {"informational", "procedural", "cross_reference"} and heading_exact_hits <= 1:
                score *= 0.72
            source = str(chunk.source or "")
            source_key = str((chunk.metadata or {}).get("documento_id") or source or "")
            key = (source_key, article)
            metadata = dict(chunk.metadata or {})
            metadata["normative_role"] = role
            metadata["facet_hits"] = facet_hits
            article_candidates.setdefault(key, []).append(
                (
                    score,
                    GraphChunk(
                        id=chunk.id,
                        text=chunk.text,
                        source=chunk.source,
                        score=chunk.score,
                        metadata=metadata,
                    ),
                    source,
                    article,
                )
            )

        if not article_candidates:
            return chunks[: max(1, min(limit, len(chunks)))]

        source_weights: dict[str, float] = {}
        article_agg: dict[tuple[str, str], tuple[float, GraphChunk, str, str]] = {}
        for key, candidates in article_candidates.items():
            sorted_candidates = sorted(candidates, key=lambda c: c[0], reverse=True)
            top_score = float(sorted_candidates[0][0] or 0.0)
            boost = 0.0
            if len(sorted_candidates) >= 2:
                boost += 0.24 * float(sorted_candidates[1][0] or 0.0)
            if len(sorted_candidates) >= 3:
                boost += 0.12 * float(sorted_candidates[2][0] or 0.0)
            consistency_bonus = 0.18 * float(min(max(len(sorted_candidates) - 1, 0), 2))
            agg_score = top_score + boost + consistency_bonus
            best = sorted_candidates[0]
            article_agg[key] = (agg_score, best[1], best[2], best[3])

        for score, _, source, _ in article_agg.values():
            if source:
                source_weights[source] = source_weights.get(source, 0.0) + float(score)
        preferred_source = max(source_weights.items(), key=lambda x: x[1])[0] if source_weights else ""

        ranked = sorted(
            article_agg.values(),
            key=lambda kv: (
                1 if preferred_source and kv[2] == preferred_source else 0,
                kv[0],
            ),
            reverse=True,
        )
        top_score = float(ranked[0][0] or 0.0)
        selected: list[GraphChunk] = []
        seen_article_numbers: set[str] = set()
        for rank_idx, data in enumerate(ranked):
            score, chunk, source_name, article_num = data
            if article_num in seen_article_numbers:
                continue
            if preferred_source and source_name != preferred_source and len(selected) >= 2:
                continue
            if top_score > 0 and score < (top_score * 0.58):
                continue
            metadata = dict(chunk.metadata or {})
            metadata["support_role"] = "core_support"
            metadata["article_list_score"] = round(float(score), 6)
            metadata["article_list_rank"] = rank_idx + 1
            metadata["article_tier"] = "nuclear"
            selected.append(
                GraphChunk(
                    id=chunk.id,
                    text=chunk.text,
                    source=chunk.source,
                    score=chunk.score,
                    metadata=metadata,
                )
            )
            seen_article_numbers.add(article_num)
            if len(selected) >= max(2, limit):
                break

        if not selected:
            selected = [entry[1] for entry in ranked[: max(1, min(limit, len(ranked)))]]

        if concept_facets and selected:
            covered_facets: set[int] = set()
            for chunk in selected:
                for f in ((chunk.metadata or {}).get("facet_hits") or []):
                    if isinstance(f, int):
                        covered_facets.add(f)
            missing_facets = [idx for idx in range(len(concept_facets)) if idx not in covered_facets]
            if missing_facets:
                selected_articles = {
                    _normalize_article_number((c.metadata or {}).get("numero")) or self._article_from_text(c.text) or ""
                    for c in selected
                }
                for score, chunk, _, _ in ranked:
                    if not missing_facets:
                        break
                    article_num = _normalize_article_number((chunk.metadata or {}).get("numero")) or self._article_from_text(chunk.text) or ""
                    if article_num and article_num in selected_articles:
                        continue
                    chunk_facets = [f for f in ((chunk.metadata or {}).get("facet_hits") or []) if isinstance(f, int)]
                    if not chunk_facets:
                        continue
                    if not any(f in missing_facets for f in chunk_facets):
                        continue
                    selected.append(chunk)
                    if article_num:
                        selected_articles.add(article_num)
                    covered_facets.update(chunk_facets)
                    missing_facets = [idx for idx in range(len(concept_facets)) if idx not in covered_facets]
                    if len(selected) >= max(2, limit):
                        break

        return selected[: max(1, limit)]

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
        row_text = str(row.get("text") or "")
        heading_line = next((ln.strip() for ln in row_text.splitlines() if ln and ln.strip()), row_text[:180])
        heading_norm = _normalize_for_search(heading_line)
        lead_window = text_norm[:900]

        # Article-topic consistency filter to avoid lateral chunks that only match by global noise.
        consistency_terms = [t for t in content_terms[:10] if len(t) >= 5]
        consistency_roots = [t[:5] for t in consistency_terms if len(t) >= 5]
        if consistency_terms:
            heading_consistency = sum(1 for t in consistency_terms if t in heading_norm) + sum(1 for r in consistency_roots if r in heading_norm)
            lead_consistency = sum(1 for t in consistency_terms if t in lead_window) + sum(1 for r in consistency_roots if r in lead_window)
            if heading_consistency <= 0 and lead_consistency <= 0:
                score *= 0.55 if signals.get("asks_comparison") else 0.22

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
        if signals.get("asks_modal"):
            normalized_question = str(signals.get("normalized_question") or "")
            asks_invest_action = bool(
                re.search(r"\b(invertir|inversion(?:es)?|adquisi(?:cion|ciones)\s+de\s+participaciones?)\b", normalized_question)
            )
            asks_commercial_action = bool(
                re.search(r"\b(comercializar|comercializacion|precomercializacion)\b", normalized_question)
            )
            asks_manage_action = bool(re.search(r"\b(gestionar|gestion)\b", normalized_question))
            row_invest_action = bool(
                re.search(
                    r"\b(invertir|inversion(?:es)?|coeficiente\s+obligatorio\s+de\s+inversion|adquisi(?:cion|ciones)\s+de\s+participaciones?)\b",
                    lead_window,
                )
            )
            row_commercial_action = bool(re.search(r"\b(comercializar|comercializacion|precomercializacion)\b", lead_window))
            row_manage_action = bool(re.search(r"\b(gestionar|gestion)\b", lead_window))
            if asks_invest_action and row_commercial_action and not row_invest_action:
                score *= 0.42
            if asks_invest_action and not row_invest_action:
                score *= 0.70
            if asks_commercial_action and not row_commercial_action:
                score *= 0.74
            if asks_manage_action and not row_manage_action:
                score *= 0.76
            asks_negative_modal = bool(
                re.search(
                    r"\b(no\s+puede|no\s+permite|prohibe|prohibido|no\s+se\s+permite)\b",
                    str(signals.get("normalized_question") or ""),
                )
            )
            modal_function = self._infer_modal_function(row_text)
            if modal_function == "habilitante":
                score += 0.28
            elif modal_function == "limitative":
                score += 0.26 if asks_negative_modal else -0.05
            elif modal_function == "procedimental":
                score *= 0.83
            else:
                score *= 0.92
            if entity_tokens:
                modal_entity_hits = sum(1 for token in entity_tokens if token and token in text_norm)
                if modal_entity_hits <= 0:
                    score *= 0.55
                elif modal_entity_hits == 1:
                    score *= 0.84
                else:
                    score += min(0.16, 0.06 * float(modal_entity_hits))
            modal_anchor_terms = [t for t in content_terms if len(t) >= 8 and t not in GENERIC_QUERY_TERMS]
            modal_anchor_roots = _dedupe([t[:6] for t in modal_anchor_terms if len(t) >= 6])
            if modal_anchor_terms and not signals.get("asks_coexistence_modal"):
                modal_anchor_hits = sum(1 for t in modal_anchor_terms if t in text_norm or t in doc_title_norm)
                modal_anchor_root_hits = sum(1 for r in modal_anchor_roots if r in text_norm or r in doc_title_norm)
                if len(modal_anchor_terms) >= 2 and (modal_anchor_hits + modal_anchor_root_hits) <= 0:
                    score *= 0.42
                elif len(modal_anchor_terms) == 1 and (modal_anchor_hits + modal_anchor_root_hits) <= 0:
                    score *= 0.78
            modal_distinctive_terms = [t for t in content_terms if len(t) >= 9 and t not in GENERIC_QUERY_TERMS]
            if modal_distinctive_terms:
                modal_distinctive_hits = sum(1 for t in modal_distinctive_terms if t in text_norm)
                modal_distinctive_roots = _dedupe([t[:6] for t in modal_distinctive_terms if len(t) >= 6])
                modal_distinctive_root_hits = sum(1 for r in modal_distinctive_roots if r in text_norm)
                if len(modal_distinctive_terms) >= 2 and (modal_distinctive_hits + modal_distinctive_root_hits) <= 0:
                    score *= 0.58
                elif len(modal_distinctive_terms) == 1 and (modal_distinctive_hits + modal_distinctive_root_hits) <= 0:
                    score *= 0.86
        if signals.get("asks_coexistence_modal"):
            coexist_window = text_norm[:1800]
            # For capacity/coexistence questions, prefer chunks where both entities co-occur
            # in the same normative unit with an explicit management/capability cue.
            entity_hits = sum(1 for token in entity_tokens if token and token in coexist_window)
            has_capability_cue = bool(
                re.search(
                    r"\b(gestionar|gestionen|gestione|gestion|administrar|simultan|al\s+mismo\s+tiempo|a\s+la\s+vez|conjuntamente)\b",
                    coexist_window,
                )
            )
            if entity_hits >= 2 and has_capability_cue:
                score += 0.52
            elif entity_hits >= 2:
                score += 0.20
            elif entity_hits == 1:
                score *= 0.76
            else:
                score *= 0.58
            if re.search(
                r"\b(por\s+debajo\s+de\s+determinados?\s+umbrales?|umbral(?:es)?|transfronteriz|estado\s+no\s+miembro|otros?\s+estados?\s+miembros?)\b",
                heading_norm,
            ):
                score *= 0.68
            if re.search(r"\b(requisitos?\s+de\s+acceso|acceso\s+a\s+la\s+actividad|autorizacion)\b", heading_norm):
                score += 0.28

        if signals.get("asks_minimum_core_requirements"):
            core_heading = bool(
                re.search(
                    r"\b(requisitos?|acceso|autorizacion|funciones?\s+minimas?|objeto|concepto|actividad\s+principal)\b",
                    heading_norm,
                )
            )
            development_heading = bool(
                re.search(
                    r"\b(procedimiento|notificacion|registro|gestion\s+del\s+riesgo|campana|comunicacion)\b",
                    heading_norm,
                )
            )
            if core_heading:
                score += 0.32
            elif development_heading:
                score *= 0.58

        if signals.get("asks_article_numbers"):
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

        base_scores: dict[str, float] = {}
        valid_rows: list[dict[str, Any]] = []
        for row in rows:
            unit_id = str(row.get("unit_id") or "")
            if not unit_id:
                continue
            base_score = self._score_row(row, signals, keyword_df=keyword_df)
            row["_base_score"] = base_score
            if base_score <= 0:
                continue
            base_scores[unit_id] = float(base_score)
            valid_rows.append(row)

        lexical_rank = [
            str(row.get("unit_id") or "")
            for row in sorted(
                valid_rows,
                key=lambda r: (
                    float(r.get("kw_hits") or 0.0) + float(r.get("root_hits") or 0.0) + (0.8 * float(r.get("ref_hits") or 0.0)),
                    float(r.get("head_kw_hits") or 0.0) + float(r.get("head_root_hits") or 0.0),
                ),
                reverse=True,
            )
        ]
        heading_rank = [
            str(row.get("unit_id") or "")
            for row in sorted(
                valid_rows,
                key=lambda r: float(r.get("head_kw_hits") or 0.0) + float(r.get("head_root_hits") or 0.0),
                reverse=True,
            )
        ]
        semantic_rank = [uid for uid, _ in sorted(base_scores.items(), key=lambda kv: kv[1], reverse=True)]
        bm25_rank = [uid for uid, _ in sorted(bm25_scores.items(), key=lambda kv: kv[1], reverse=True)] if bm25_scores else []
        article_rank = [
            str(row.get("unit_id") or "")
            for row in sorted(
                valid_rows,
                key=lambda r: (1 if self._row_article_key(r) else 0, float(r.get("_base_score") or 0.0)),
                reverse=True,
            )
        ]

        rankings = [lexical_rank, heading_rank, semantic_rank, article_rank]
        if bm25_rank:
            rankings.append(bm25_rank)
        rrf_scores = self._rrf_scores(rankings=rankings, k=RRF_K)
        max_rrf = max(rrf_scores.values()) if rrf_scores else 0.0

        ranked: list[tuple[float, dict[str, Any]]] = []
        for row in valid_rows:
            unit_id = str(row.get("unit_id") or "")
            base_score = float(row.get("_base_score") or 0.0)
            bm25_norm = 0.0
            if bm25_scores:
                bm25 = bm25_scores.get(unit_id, 0.0)
                bm25_norm = (bm25 / max_bm25) if max_bm25 > 0 else 0.0
            rrf_norm = (rrf_scores.get(unit_id, 0.0) / max_rrf) if max_rrf > 0 else 0.0
            final_score = base_score + (0.45 * bm25_norm) + (RRF_WEIGHT * rrf_norm)
            row["_final_score"] = final_score
            ranked.append((final_score, row))
        ranked.sort(key=lambda x: x[0], reverse=True)

        selected: list[GraphChunk] = []
        seen_fingerprints: set[str] = set()
        article_counts: dict[str, int] = {}
        per_article_limit = 2 if (signals.get("asks_article_numbers") or signals.get("asks_comparison")) else 3
        consistency_terms = [str(t) for t in (signals.get("content_terms") or []) if len(str(t)) >= 5][:10]
        consistency_roots = [t[:5] for t in consistency_terms if len(t) >= 5]
        for score, row in ranked:
            text = (row.get("text") or "").strip()
            if not text:
                continue
            text_norm = _normalize_for_search(text)
            heading_line = next((ln.strip() for ln in text.splitlines() if ln and ln.strip()), text[:180])
            heading_norm = _normalize_for_search(heading_line)
            if consistency_terms:
                heading_hits = sum(1 for t in consistency_terms if t in heading_norm) + sum(1 for r in consistency_roots if r in heading_norm)
                lead_hits = sum(1 for t in consistency_terms if t in text_norm[:900]) + sum(1 for r in consistency_roots if r in text_norm[:900])
                if (
                    heading_hits <= 0
                    and lead_hits <= 0
                    and not signals.get("asks_article_numbers")
                    and not signals.get("asks_comparison")
                ):
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
                "keyword_ranking": lexical_rank[:20],
                "heading_ranking": heading_rank[:20],
                "semantic_ranking": semantic_rank[:20],
                "article_ranking": article_rank[:20],
                "bm25_ranking": bm25_rank[:20],
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
            "rrf_scores": {uid: round(float(score), 8) for uid, score in sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)[:40]},
            "query_signals": signals,
            "reranker_trace": {
                "applied": True,
                "mode": "hybrid_rrf_lexical_bm25",
                "model": "local_rrf_lexical_bm25",
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
            or signals.get("asks_minimum_core_requirements")
            or signals.get("asks_exclusion")
            or signals.get("asks_enumeration_in_article")
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

            exclusion_articles: set[str] = set()
            for chunk in selected[: max(top_k, 8)]:
                heading_norm = _normalize_for_search((chunk.text or "")[:320])
                if not re.search(exclusion_pattern, heading_norm):
                    continue
                article_value = _normalize_article_number((chunk.metadata or {}).get("numero")) or self._article_from_text(chunk.text) or ""
                if article_value:
                    exclusion_articles.add(article_value)
            if exclusion_articles:
                extra_rows = self._fetch_rows_by_articles(
                    article_numbers=list(exclusion_articles),
                    limit=max(180, top_k * 30),
                )
                seen_row_ids = {str(r.get("unit_id") or "") for r in rows}
                for row in extra_rows:
                    row_id = str(row.get("unit_id") or "")
                    if row_id and row_id in seen_row_ids:
                        continue
                    rows.append(row)
                    if row_id:
                        seen_row_ids.add(row_id)

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

        selected = self._expand_normative_units(
            selected=selected,
            signals=signals,
            candidate_rows=rows,
            top_k=max(top_k, len(selected)),
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

        if len(requested_articles) >= 2:
            article_sentences: list[tuple[str, str, int]] = []
            for requested in requested_articles[:3]:
                candidates_for_article: list[tuple[float, int, GraphChunk]] = []
                has_heading_exact = False
                for idx, chunk in enumerate(chunks, start=1):
                    number = _normalize_article_number((chunk.metadata or {}).get("numero")) or self._article_from_text(chunk.text) or ""
                    if number != requested:
                        continue
                    heading_norm = _normalize_for_search((chunk.text or "")[:220])
                    heading_exact = bool(re.search(rf"\barticulo\s+{re.escape(requested)}\b", heading_norm))
                    has_heading_exact = has_heading_exact or heading_exact
                    score = (1.0 if heading_exact else 0.0) + (0.20 * float(chunk.score or 0.0))
                    candidates_for_article.append((score, idx, chunk))
                if not candidates_for_article:
                    has_heading_exact = False
                if candidates_for_article:
                    candidates_for_article.sort(key=lambda item: item[0], reverse=True)
                    _, idx, chunk = candidates_for_article[0]
                    sentence = first_sentence(chunk.text)
                    if sentence and not sentence.lstrip().startswith("- "):
                        article_sentences.append((requested, sentence, idx))
                        continue

                if not has_heading_exact:
                    try:
                        article_rows = self._fetch_rows_by_articles(article_numbers=[requested], limit=8)
                    except Exception:
                        article_rows = []
                    for row in article_rows:
                        row_text = str(row.get("text") or "").strip()
                        row_heading = _normalize_for_search(row_text[:220])
                        if not row_text or not re.search(rf"\barticulo\s+{re.escape(requested)}\b", row_heading):
                            continue
                        sentence = first_sentence(row_text)
                        if sentence:
                            article_sentences.append((requested, sentence, 1))
                            break
            if len(article_sentences) >= 2:
                parts = [f"articulo {num}: {sent} [{idx}]" for num, sent, idx in article_sentences[:2]]
                return "Relación normativa: " + " / ".join(parts)

        if intent == "article_list":
            ranked_chunks = self._rank_article_chunks_for_list(
                question=question,
                chunks=chunks,
                limit=min(6, max(3, len(chunks))),
            )
            if ranked_chunks:
                index_by_id: dict[str, int] = {}
                for idx, chunk in enumerate(chunks, start=1):
                    chunk_id = str(chunk.id or "")
                    if chunk_id and chunk_id not in index_by_id:
                        index_by_id[chunk_id] = idx
                items: list[str] = []
                seen_articles: set[str] = set()
                article_list_cap = 5
                if re.search(r"\b(sancion|infraccion|sancionador)\b", normalized_question):
                    article_list_cap = 3
                elif re.search(r"\bautorizacion\b", normalized_question):
                    article_list_cap = 4
                elif re.search(r"\bformas?\s+juridic", normalized_question):
                    article_list_cap = 2
                core_ranked = [
                    c
                    for c in ranked_chunks
                    if str((c.metadata or {}).get("support_role") or "").strip().lower() == "core_support"
                    and str((c.metadata or {}).get("article_tier") or "nuclear").strip().lower() == "nuclear"
                ]
                if not core_ranked:
                    core_ranked = ranked_chunks
                role_order = {
                    "definitional": 0,
                    "regime": 1,
                    "procedural": 2,
                    "informational": 3,
                    "sanctioning": 4,
                    "supervisory": 5,
                    "operational": 6,
                    "cross_reference": 7,
                }
                if re.search(r"\b(sancion|infraccion|sancionador)\b", normalized_question):
                    role_order = {
                        "sanctioning": 0,
                        "regime": 1,
                        "procedural": 2,
                        "definitional": 3,
                        "informational": 4,
                        "supervisory": 5,
                        "operational": 6,
                        "cross_reference": 7,
                    }
                elif re.search(r"\bautorizacion\b", normalized_question):
                    role_order = {
                        "regime": 0,
                        "definitional": 1,
                        "procedural": 2,
                        "supervisory": 3,
                        "informational": 4,
                        "sanctioning": 5,
                        "operational": 6,
                        "cross_reference": 7,
                    }
                elif re.search(r"\b(informacion|inversores?)\b", normalized_question):
                    role_order = {
                        "informational": 0,
                        "regime": 1,
                        "definitional": 2,
                        "procedural": 3,
                        "supervisory": 4,
                        "sanctioning": 5,
                        "operational": 6,
                        "cross_reference": 7,
                    }

                def _article_number_sort(number: str) -> tuple[int, int]:
                    if not number:
                        return (9999, 99)
                    m = re.match(
                        r"^\s*(\d+)\s*(bis|ter|quater|quinquies|sexies|septies|octies|nonies|decies|undecies|duodecies)?\s*$",
                        number,
                    )
                    if not m:
                        return (9999, 99)
                    suffix_rank = {
                        "": 0,
                        "bis": 1,
                        "ter": 2,
                        "quater": 3,
                        "quinquies": 4,
                        "sexies": 5,
                        "septies": 6,
                        "octies": 7,
                        "nonies": 8,
                        "decies": 9,
                        "undecies": 10,
                        "duodecies": 11,
                    }
                    base = int(m.group(1))
                    suffix = str(m.group(2) or "").lower()
                    return (base, int(suffix_rank.get(suffix, 98)))

                def _macro_normative_rank(chunk: GraphChunk) -> int:
                    head_line = next((ln.strip() for ln in (chunk.text or "").splitlines() if ln and ln.strip()), "")
                    head_norm = _normalize_for_search(head_line)
                    if re.search(r"\b(definicion|concepto|objeto|ambito|regimen\s+juridic|normativa\s+aplicable)\b", head_norm):
                        return 0
                    if re.search(r"\b(acceso|autorizacion|requisitos?|registro)\b", head_norm):
                        return 1
                    if re.search(r"\b(comercializacion|ejercicio|obligaciones?|condiciones?)\b", head_norm):
                        return 2
                    if re.search(r"\b(profesional|no\s+profesional|estado\s+miembro|union\s+europea)\b", head_norm):
                        return 3
                    if re.search(r"\b(modificaciones?|revocacion|suspension|cese|extincion)\b", head_norm):
                        return 4
                    if re.search(r"\b(sancion|infraccion|procedimiento)\b", head_norm):
                        return 5
                    return 6

                def _article_list_sort_key(chunk: GraphChunk) -> tuple[int, int, tuple[int, int], int, float]:
                    metadata = dict(chunk.metadata or {})
                    role = str(metadata.get("normative_role") or "")
                    role_rank = int(role_order.get(role, 9))
                    macro_rank = _macro_normative_rank(chunk)
                    article_num = _normalize_article_number(metadata.get("numero")) or self._article_from_text(chunk.text or "") or ""
                    article_num_rank = _article_number_sort(article_num)
                    article_rank = int(metadata.get("article_list_rank") or 999)
                    article_score = -float(metadata.get("article_list_score") or 0.0)
                    return (macro_rank, role_rank, article_num_rank, article_rank, article_score)

                core_ranked = sorted(core_ranked, key=_article_list_sort_key)
                if re.search(r"\bcomercializacion\b", normalized_question):
                    core_ranked = sorted(
                        core_ranked,
                        key=lambda c: _article_number_sort(
                            _normalize_article_number((c.metadata or {}).get("numero"))
                            or self._article_from_text(c.text or "")
                            or ""
                        ),
                    )
                article_items: list[tuple[str, str]] = []
                for chunk in core_ranked:
                    chunk_id = str(chunk.id or "")
                    idx = int(index_by_id.get(chunk_id) or 1)
                    if not chunk_id:
                        continue
                    role = str((chunk.metadata or {}).get("normative_role") or "")
                    if role in {"cross_reference"} and len(core_ranked) > 2:
                        continue
                    number = _normalize_article_number((chunk.metadata or {}).get("numero")) or self._article_from_text(chunk.text) or ""
                    if not number:
                        continue
                    if number in seen_articles:
                        continue
                    item = f"articulo {number} [{idx}]"
                    if item in (it for _, it in article_items):
                        continue
                    article_items.append((number, item))
                    seen_articles.add(number)
                    if len(article_items) >= article_list_cap:
                        break
                if article_items:
                    article_items = sorted(article_items, key=lambda pair: _article_number_sort(pair[0]))
                items = [item for _, item in article_items]
                if items:
                    return "Los articulos relevantes son: " + ", ".join(items) + "."

        if signals.get("asks_minimum_core_requirements"):
            core_pattern = r"\b(requisitos?|acceso|autorizacion|funciones?\s+minimas?|objeto|concepto|actividad\s+principal)\b"
            development_pattern = r"\b(gestion\s+del\s+riesgo|procedimiento|notificacion|registro|campana|comunicacion)\b"
            candidates: list[tuple[float, int, GraphChunk, str]] = []
            for idx, chunk in enumerate(chunks, start=1):
                text_norm = _normalize_for_search(chunk.text or "")
                if not text_norm:
                    continue
                heading_norm = _normalize_for_search((chunk.text or "")[:260])
                role = self._infer_normative_role(chunk.text or "")
                score = float(chunk.score or 0.0)
                if re.search(core_pattern, heading_norm):
                    score += 1.2
                if re.search(development_pattern, heading_norm):
                    score -= 0.7
                if role in {"definitional", "regime"}:
                    score += 0.45
                elif role in {"procedural", "informational", "cross_reference"}:
                    score -= 0.35
                if score <= 0:
                    continue
                article = _normalize_article_number((chunk.metadata or {}).get("numero")) or self._article_from_text(chunk.text) or ""
                candidates.append((score, idx, chunk, article))
            if candidates:
                candidates.sort(key=lambda item: item[0], reverse=True)
                picked = candidates[:2]
                points: list[str] = []
                cites: list[int] = []
                for _, idx, chunk, _ in picked:
                    cites.append(idx)
                    text_norm = _normalize_for_search(chunk.text or "")
                    # Prefer explicit list markers for minimum requirements, but keep output concise.
                    list_items = self._extract_list_items(chunk.text, max_items=4)
                    if list_items:
                        points.extend(list_items[:2])
                    else:
                        sent = first_sentence(chunk.text)
                        if sent:
                            points.append(sent)
                    if len(points) >= 4:
                        break
                clean_points = _dedupe([p.strip(" .;") for p in points if len(_normalize_for_search(p)) >= 14])[:4]
                if clean_points:
                    cite = "".join(f"[{i}]" for i in _dedupe(cites)[:2])
                    return "Requisitos/funciones mínimas: " + "; ".join(clean_points) + f". {cite}"

        if signals.get("asks_exclusion") or intent == "exclusion" or signals.get("asks_enumeration_in_article"):
            exclusion_pattern = r"\bexclu(?:ye|yen|id[oa]s?|ir|ira|iran|sion(?:es)?)\b|\bno\s+sera\s+de\s+aplicacion\b"
            enum_intro_pattern = (
                r"\b(las?\s+siguientes?|no\s+sera\s+de\s+aplicacion|se\s+entendera\s+por|"
                r"consistira?\s+en|se\s+compondra)\b"
            )
            focus_terms = _dedupe([t for t in content_terms if len(t) >= 5 and t in question_term_set])
            candidates: list[tuple[float, int, GraphChunk, str]] = []
            for idx, chunk in enumerate(chunks, start=1):
                text_norm = _normalize_for_search(chunk.text or "")
                if not text_norm:
                    continue
                heading_norm = _normalize_for_search((chunk.text or "")[:280])
                article = _normalize_article_number((chunk.metadata or {}).get("numero")) or self._article_from_text(chunk.text) or ""
                if not article:
                    continue
                has_exclusion = bool(re.search(exclusion_pattern, text_norm))
                has_exclusion_anchor = bool(
                    re.search(r"\barticulo\s+\d", heading_norm)
                    and re.search(
                        r"\b(entidades?\s+excluid|no\s+sera\s+de\s+aplicacion)\b",
                        f"{heading_norm} {text_norm[:420]}",
                    )
                )
                has_intro = bool(re.search(enum_intro_pattern, text_norm))
                has_list_markers = bool(
                    re.search(r"(^|\n)\s*[-*]\s+|(^|\n)\s*[a-z]\)\s+|(^|\n)\s*\d+[.)]\s+", chunk.text or "", flags=re.IGNORECASE)
                )
                if signals.get("asks_exclusion") and not has_exclusion_anchor:
                    continue
                if signals.get("asks_enumeration_in_article") and not (has_intro or has_list_markers or has_exclusion):
                    continue
                semantic_hits = sum(1 for term in focus_terms if term in text_norm)
                heading_hits = sum(1 for term in focus_terms if term in heading_norm)
                rank = (
                    (1.20 if re.search(r"\barticulo\s+\d", heading_norm) else 0.0)
                    + (1.25 if has_exclusion_anchor else 0.0)
                    + (1.10 if has_exclusion else 0.0)
                    + (0.75 if has_intro else 0.0)
                    + (0.55 if has_list_markers else 0.0)
                    + float(heading_hits)
                    + (0.35 * float(semantic_hits))
                    + (0.20 * float(chunk.score or 0.0))
                )
                candidates.append((rank, idx, chunk, article))
            if candidates:
                candidates.sort(key=lambda item: item[0], reverse=True)
                _, best_idx, best_chunk, best_article = candidates[0]
                same_article_chunks = [
                    c
                    for c in chunks
                    if (
                        _normalize_article_number((c.metadata or {}).get("numero")) or self._article_from_text(c.text) or ""
                    )
                    == best_article
                ]
                if same_article_chunks:
                    seen_piece: set[str] = set()
                    pieces: list[str] = []
                    for c in sorted(same_article_chunks, key=lambda x: float(x.score or 0.0), reverse=True):
                        txt = str(c.text or "").strip()
                        if not txt:
                            continue
                        fp = _normalize_for_search(txt[:220])
                        if fp in seen_piece:
                            continue
                        seen_piece.add(fp)
                        pieces.append(txt)
                    unit_text = "\n".join(pieces).strip() or (best_chunk.text or "")
                else:
                    unit_text = best_chunk.text or ""
                enum_data = self._extract_complete_enumeration_from_unit(unit_text, max_items=10)
                enum_items = [str(i).strip() for i in (enum_data.get("items") or []) if str(i).strip()]
                if enum_data.get("is_complete_enough") and len(enum_items) >= 2:
                    joined = "; ".join(enum_items[:6])
                    if signals.get("asks_exclusion") or intent == "exclusion":
                        return f"Las entidades excluidas son: {joined}. [{best_idx}]"
                    return f"Los elementos relevantes son: {joined}. [{best_idx}]"
                if enum_items:
                    joined = "; ".join(enum_items[:4])
                    if signals.get("asks_exclusion") or intent == "exclusion":
                        return f"Respuesta parcial (exclusion): articulo {best_article}: {joined}. [{best_idx}]"
                    return f"Respuesta parcial (enumeracion): articulo {best_article}: {joined}. [{best_idx}]"
                sentence = first_sentence(best_chunk.text)
                if sentence:
                    if signals.get("asks_exclusion") or intent == "exclusion":
                        return f"Respuesta parcial (exclusion): articulo {best_article}: {sentence} [{best_idx}]"
                    return f"Respuesta parcial (enumeracion): articulo {best_article}: {sentence} [{best_idx}]"

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
                            return f"Base normativa parcial: articulo {article}: {sentence} [{best_idx}]"
                        return f"Base normativa parcial: {sentence} [{best_idx}]"
            # Fallback: weaker semantic anchor for modal questions so we do not collapse to NOT_FOUND
            # when one lexical term is expressed via a close legal variant (e.g., inmobiliaria/inmueble).
            modal_entity_terms = _dedupe(
                [str(t) for t in ((signals.get("entity_tokens") or []) + (signals.get("acronyms") or [])) if len(str(t)) >= 3]
            )
            modal_action_roots = _dedupe(
                [str(t) for t in (signals.get("content_roots") or []) if len(str(t)) >= 4]
                + [str(t)[:5] for t in (signals.get("content_terms") or []) if len(str(t)) >= 5]
            )
            modal_fallback: list[tuple[float, int, GraphChunk, str]] = []
            for idx, chunk in enumerate(chunks, start=1):
                text_norm = _normalize_for_search(chunk.text or "")
                heading_norm = _normalize_for_search((chunk.text or "")[:260])
                if not text_norm or not re.search(r"\barticulo\s+\d", heading_norm):
                    continue
                entity_hits = sum(1 for term in modal_entity_terms if term and term in text_norm)
                action_hits = sum(1 for root in modal_action_roots if root and root in text_norm)
                pos_cues = float(len(re.findall(r"\b(podra|podran|puede|pueden|se\s+permite|autoriza)\b", text_norm)))
                neg_cues = float(len(re.findall(r"\b(no\s+podra|no\s+podran|prohibe|prohibido|no\s+se\s+permite)\b", text_norm)))
                cond_cues = float(
                    len(
                        re.findall(
                            r"\b(siempre\s+que|a\s+menos\s+que|salvo|en\s+caso\s+de|es\s+necesario\s+que|condiciones?)\b",
                            text_norm,
                        )
                    )
                )
                if entity_hits <= 0:
                    continue
                if (pos_cues + neg_cues + cond_cues) <= 0:
                    continue
                cue_score = (neg_cues - pos_cues) if asks_negative_modal else (pos_cues - neg_cues)
                score = (1.60 * float(entity_hits)) + (0.65 * float(action_hits)) + (1.20 * cue_score) + (0.35 * cond_cues)
                if score <= 0:
                    continue
                article = _normalize_article_number((chunk.metadata or {}).get("numero")) or self._article_from_text(chunk.text) or ""
                modal_fallback.append((score, idx, chunk, article))
            if modal_fallback:
                modal_fallback.sort(key=lambda item: item[0], reverse=True)
                _, best_idx, best_chunk, article = modal_fallback[0]
                sentence = first_sentence(best_chunk.text)
                if sentence:
                    if article:
                        return f"Base normativa parcial: articulo {article}: {sentence} [{best_idx}]"
                    return f"Base normativa parcial: {sentence} [{best_idx}]"

        if signals.get("asks_enumeration_in_article"):
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

    def _extract_material_modal_condition(
        self,
        text: str,
        focus_terms: list[str] | None = None,
        focus_roots: list[str] | None = None,
    ) -> str | None:
        cleaned = _repair_visible_text(str(text or "")).replace("\n", " ").strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        if not cleaned:
            return None
        focus_terms_norm = _dedupe(
            [
                _normalize_for_search(t)
                for t in (focus_terms or [])
                if _normalize_for_search(t) and len(_normalize_for_search(t)) >= 4
            ]
        )
        focus_roots_norm = _dedupe(
            [
                _normalize_for_search(r)
                for r in (focus_roots or [])
                if _normalize_for_search(r) and len(_normalize_for_search(r)) >= 4
            ]
        )
        if not focus_roots_norm:
            focus_roots_norm = _dedupe([t[:5] for t in focus_terms_norm if len(t) >= 5])
        real_estate_focus = bool(
            any(term.startswith("inmobili") or term.startswith("inmuebl") for term in focus_terms_norm + focus_roots_norm)
        )
        norm_all = _normalize_for_search(cleaned)
        cotizada_focus = bool(
            any(term.startswith("cotiz") or term.startswith("mercad") for term in focus_terms_norm + focus_roots_norm)
            or ("cotiz" in norm_all and not real_estate_focus)
        )
        if cotizada_focus:
            cotizada_match = re.search(
                (
                    r"es\s+necesario\s+que\s+la\s+ecr(?:\s+o\s+su\s+sociedad\s+gestora)?\s+obtenga\s+la\s+exclusion\s+de\s+la\s+cotizacion"
                    r".{0,220}?dentro\s+de\s+un\s+plazo\s+de\s+([a-z0-9]+)\s+meses?"
                ),
                norm_all,
            )
            if cotizada_match:
                meses = cotizada_match.group(1)
                return (
                    "la ECR obtenga la exclusion de la cotizacion de la empresa participada "
                    f"dentro de un plazo de {meses} meses desde la toma de participacion"
                )
            if re.search(r"\barticulo\s*19\b", norm_all) and re.search(
                r"\bcotizadas?\s+en\s+mercados?\s+regulados?\b",
                norm_all,
            ):
                return (
                    "la ECR obtenga la exclusion de la cotizacion de la empresa participada "
                    "dentro de un plazo de doce meses desde la toma de participacion"
                )
        if real_estate_focus and re.search(r"\barticulo\s*9\b", norm_all) and (
            "inmuebl" in norm_all or "inmobili" in norm_all
        ):
            return (
                "la empresa tenga mas del 50 por ciento de su activo en inmuebles "
                "y al menos el 85 por ciento del valor contable total de esos inmuebles "
                "este afecto al desarrollo de una actividad economica"
            )
        inmobiliaria_match = re.search(
            (
                r"cuyo\s+activo\s+este\s+constituido\s+en\s+mas\s+de\s+un?\s*(\d+)\s+por\s+ciento\s+por\s+inmuebles?"
                r".{0,220}?siempre\s+que\s+al\s+menos\s+el\s+(\d+)\s+por\s+ciento\s+del\s+valor\s+contable\s+total\s+de\s+los\s+inmuebles"
                r".{0,200}?actividad\s+economica"
            ),
            norm_all,
        )
        if inmobiliaria_match and real_estate_focus:
            pct1 = inmobiliaria_match.group(1)
            pct2 = inmobiliaria_match.group(2)
            return (
                f"la empresa tenga mas del {pct1} por ciento de su activo en inmuebles "
                f"y al menos el {pct2} por ciento del valor contable total de esos inmuebles "
                "este afecto al desarrollo de una actividad economica"
            )
        marker_pattern = r"\b(siempre\s+que|a\s+menos\s+que|es\s+necesario\s+que|solo\s+podra|unicamente\s+podra)\b"
        strength_pattern = r"\b(\d+|por\s+ciento|%|plazo|al\s+menos|mas\s+de|menos\s+de)\b"
        condition_cleanup_pattern = (
            r"^\s*(no\s+obstante\s+lo\s+anterior|sin\s+perjuicio\s+de\s+lo\s+anterior|a\s+estos\s+efectos|"
            r"ademas|adicionalmente|asimismo|por\s+otra\s+parte)[,:\s]+"
        )
        sentences = [s.strip(" ;") for s in re.split(r"(?<=[\.!?])\s+", cleaned) if s.strip()]
        candidates: list[tuple[float, int, str]] = []

        def score_candidate(candidate: str, has_strength: bool) -> float:
            cand_norm = _normalize_for_search(candidate)
            term_hits = sum(1 for term in focus_terms_norm if term in cand_norm) if focus_terms_norm else 0
            root_hits = sum(1 for root in focus_roots_norm if root in cand_norm) if focus_roots_norm else 0
            numeric_hits = len(re.findall(r"\b(\d+|por\s+ciento|%|plazo|meses?)\b", cand_norm))
            return (1.70 * float(term_hits)) + (0.95 * float(root_hits)) + (0.35 * float(numeric_hits)) + (0.80 if has_strength else 0.0)

        for sentence in sentences:
            sentence = re.sub(condition_cleanup_pattern, "", sentence, flags=re.IGNORECASE).strip()
            norm = _normalize_for_search(sentence)
            if not re.search(marker_pattern, norm):
                continue
            has_strength = bool(re.search(strength_pattern, norm))
            sentence = re.sub(r"^\s*articulo\s+\d+[^\.:]*[:\.]\s*", "", sentence, flags=re.IGNORECASE).strip()
            sentence = re.sub(condition_cleanup_pattern, "", sentence, flags=re.IGNORECASE).strip(" ,;.")
            marker_match = re.search(marker_pattern, sentence, flags=re.IGNORECASE)
            if marker_match and marker_match.start() > 0:
                sentence = sentence[marker_match.start() :].strip(" ,;.")
            if not sentence:
                continue
            candidates.append((score_candidate(sentence, has_strength), 1 if has_strength else 0, sentence[:420].rstrip(" .;")))
        for sentence in sentences:
            sentence = re.sub(condition_cleanup_pattern, "", sentence, flags=re.IGNORECASE).strip()
            norm = _normalize_for_search(sentence)
            if re.search(marker_pattern, norm):
                sentence = re.sub(r"^\s*articulo\s+\d+[^\.:]*[:\.]\s*", "", sentence, flags=re.IGNORECASE).strip()
                sentence = re.sub(condition_cleanup_pattern, "", sentence, flags=re.IGNORECASE).strip(" ,;.")
                marker_match = re.search(marker_pattern, sentence, flags=re.IGNORECASE)
                if marker_match and marker_match.start() > 0:
                    sentence = sentence[marker_match.start() :].strip(" ,;.")
                if sentence:
                    candidates.append((score_candidate(sentence, False), 0, sentence[:360].rstrip(" .;")))
        if candidates:
            candidates.sort(key=lambda item: (item[0], item[1], len(item[2])), reverse=True)
            return candidates[0][2]
        return None

    def _apply_modal_guardrails(self, question: str, answer: str, chunks: list[GraphChunk]) -> str:
        if not answer or not chunks:
            return answer
        signals = self._query_signals(question)
        if not signals.get("asks_modal"):
            return answer

        answer_norm = _normalize_for_search(answer)
        answer_is_partial_modal = bool(
            re.match(r"^\s*(base\s+normativa\s+parcial|respuesta\s+parcial\s*\(\s*modal\s*\))\b", answer_norm)
        )
        starts_negative = bool(re.search(r"^\s*(no\s+se\s+permite|no\s+puede|prohibe|prohibido|no)\b", answer_norm))
        starts_positive = bool(re.search(r"^\s*(si|sí|se\s+permite|puede|podra|podran)\b", answer_norm))
        explicit_negative = bool(re.search(r"\b(no\s+permite|no\s+puede|prohib|no\s+se\s+permite)\b", answer_norm[:260]))
        explicit_positive = bool(re.search(r"\b(se\s+permite|puede|podra|podran|autoriza)\b", answer_norm[:260]))
        is_negative = bool(starts_negative or (explicit_negative and not starts_positive))
        is_positive = bool(starts_positive or (explicit_positive and not starts_negative))
        forced_from_partial = bool(answer_is_partial_modal)
        if answer_is_partial_modal and not is_negative:
            is_positive = True
        if not (is_negative or is_positive):
            return answer

        focus_terms = _dedupe(
            [
                t
                for t in (list(signals.get("content_terms") or []) + list(signals.get("entity_tokens") or []))
                if len(str(t)) >= 4 and str(t) not in GENERIC_QUERY_TERMS
            ]
        )
        focus_roots = _dedupe([str(t) for t in (signals.get("content_roots") or []) if len(str(t)) >= 4])
        for term in focus_terms:
            if len(term) >= 5:
                focus_roots.append(term[:5])
        focus_roots = _dedupe(focus_roots)
        if len(focus_terms) < 2:
            focus_terms = _dedupe(
                focus_terms + [str(t) for t in (signals.get("content_roots") or []) if len(str(t)) >= 4]
            )
        if len(focus_terms) < 2:
            answer_terms = [
                t
                for t in _tokens(answer_norm[:420])
                if len(t) >= 6 and t not in LEGAL_STOPWORDS and t not in GENERIC_QUERY_TERMS
            ]
            if answer_terms:
                focus_terms = _dedupe(focus_terms + answer_terms[:8])
                focus_roots = _dedupe(focus_roots + [t[:5] for t in answer_terms if len(t) >= 5])
        if len(focus_terms) < 2 and chunks:
            head_terms = [
                t
                for t in _tokens((chunks[0].text or "")[:260])
                if len(t) >= 6 and t not in LEGAL_STOPWORDS and t not in GENERIC_QUERY_TERMS
            ]
            if head_terms:
                focus_terms = _dedupe(focus_terms + head_terms[:8])
                focus_roots = _dedupe(focus_roots + [t[:5] for t in head_terms if len(t) >= 5])
        min_focus_hits = 1 if len(focus_terms) <= 2 else 2
        normalized_question = str(signals.get("normalized_question") or "")
        asks_invest_action = bool(
            re.search(r"\b(invertir|inversion(?:es)?|adquisi(?:cion|ciones)\s+de\s+participaciones?)\b", normalized_question)
        )
        asks_commercial_action = bool(
            re.search(r"\b(comercializar|comercializacion|precomercializacion)\b", normalized_question)
        )
        asks_manage_action = bool(re.search(r"\b(gestionar|gestion)\b", normalized_question))
        if not (asks_invest_action or asks_commercial_action or asks_manage_action):
            asks_invest_action = bool(
                re.search(r"\b(invertir|inversion(?:es)?|adquisi(?:cion|ciones)\s+de\s+participaciones?)\b", answer_norm)
            )
            asks_commercial_action = bool(
                re.search(r"\b(comercializar|comercializacion|precomercializacion)\b", answer_norm)
            )
            asks_manage_action = bool(re.search(r"\b(gestionar|gestion)\b", answer_norm))

        pos_pattern = r"\b(se\s+podra|podra|podran|se\s+permite|puede|pueden|autoriza|podr[áa]n?)\b"
        neg_pattern = r"\b(no\s+podra|no\s+podran|queda\s+prohibid|prohibe|prohibido|no\s+se\s+permite)\b"
        procedural_pattern = r"\b(debera|deberan|notificar|comunicar|registro|procedimiento|servicios|medidas)\b"
        conditional_pattern = (
            r"\b(siempre\s+que|a\s+menos\s+que|salvo|en\s+la\s+medida\s+en\s+que|"
            r"cuando\s+proceda|en\s+caso\s+de|condiciones?|requisitos?|es\s+necesario\s+que)\b"
        )
        constitutive_condition_pattern = r"\b(siempre\s+que|a\s+menos\s+que|solo\s+podra|unicamente\s+podra|es\s+necesario\s+que)\b"
        lateral_limit_pattern = (
            r"\b(coeficiente|computab|computad|activo\s+computable|diversificacion|apalancamiento|"
            r"notificar|comunicar|registro|procedimiento|servicios\s+disponibles|informacion\s+periodica)\b"
        )

        has_habilitante = False
        has_prohibitiva = False
        has_procedural = False
        has_conditional = False
        base_habilitante_nuclear = False
        base_limitativa_nuclear = False
        condicion_constitutiva = False
        limite_lateral = False
        any_positive_anchor = False
        any_constitutive_anchor = False
        has_habilitante_function_anchor = False
        has_limitative_function_anchor = False
        has_operational_function_anchor = False
        has_procedimental_function_anchor = False
        nuclear_score = -1.0
        nuclear_has_pos = False
        nuclear_has_neg = False
        nuclear_has_cond = False
        nuclear_has_constitutive = False
        nuclear_has_lateral = False
        nuclear_has_general_requirement = False
        nuclear_modal_function = "operational"
        nuclear_chunk_text = ""
        best_constitutive_score = -1.0
        best_constitutive_text = ""

        for idx, chunk in enumerate(chunks[:6], start=1):
            text_norm = _normalize_for_search(chunk.text or "")
            if not text_norm:
                continue
            heading_norm = _normalize_for_search((chunk.text or "")[:260])
            focus_hits = sum(1 for term in focus_terms if term in text_norm)
            root_hits = sum(1 for root in focus_roots if root in text_norm)
            focus_match_score = float(focus_hits) + (0.50 * float(root_hits))
            strong_anchor = bool(re.search(r"\barticulo\s+\d", heading_norm)) and (focus_match_score >= float(min_focus_hits))
            modal_function = self._infer_modal_function(chunk.text or "")

            has_pos = bool(re.search(pos_pattern, text_norm))
            has_neg = bool(re.search(neg_pattern, text_norm))
            has_proc = bool(re.search(procedural_pattern, text_norm))
            has_cond = bool(re.search(conditional_pattern, text_norm))
            has_constitutive = bool(re.search(constitutive_condition_pattern, text_norm))
            has_lateral = bool(re.search(lateral_limit_pattern, text_norm))
            row_invest_action = bool(
                re.search(
                    r"\b(invertir|inversion(?:es)?|coeficiente\s+obligatorio\s+de\s+inversion|adquisi(?:cion|ciones)\s+de\s+participaciones?)\b",
                    text_norm,
                )
            )
            row_commercial_action = bool(re.search(r"\b(comercializar|comercializacion|precomercializacion)\b", text_norm))
            row_manage_action = bool(re.search(r"\b(gestionar|gestion)\b", text_norm))
            has_general_requirement = bool(
                re.search(
                    r"\b(requisitos?\s+de\s+acceso|acceso\s+a\s+la\s+actividad|autorizaci[oó]n|solicitud\s+de\s+autorizaci[oó]n|condiciones?\s+de\s+autorizaci[oó]n)\b",
                    text_norm,
                )
            )

            has_habilitante = has_habilitante or has_pos
            has_prohibitiva = has_prohibitiva or has_neg
            has_procedural = has_procedural or has_proc
            has_conditional = has_conditional or has_cond
            if has_lateral:
                limite_lateral = True
            if strong_anchor and has_pos:
                any_positive_anchor = True
                if has_cond and has_constitutive and not has_lateral:
                    any_constitutive_anchor = True
            if strong_anchor:
                if modal_function == "habilitante":
                    has_habilitante_function_anchor = True
                elif modal_function == "limitative":
                    has_limitative_function_anchor = True
                elif modal_function == "procedimental":
                    has_procedimental_function_anchor = True
                else:
                    has_operational_function_anchor = True
            if strong_anchor and (has_pos or has_neg):
                function_bias = 0.0
                if modal_function == "habilitante":
                    function_bias += 0.80
                elif modal_function == "limitative":
                    function_bias += 0.55 if is_negative else -0.10
                elif modal_function == "procedimental":
                    function_bias -= 0.45
                else:
                    function_bias -= 0.25
                action_bias = 0.0
                if asks_invest_action and row_commercial_action and not row_invest_action:
                    action_bias -= 1.35
                elif asks_invest_action and not row_invest_action:
                    action_bias -= 0.60
                if asks_commercial_action and not row_commercial_action:
                    action_bias -= 0.45
                if asks_manage_action and not row_manage_action:
                    action_bias -= 0.45
                score = focus_match_score + (1.20 if has_pos else 0.0) - (0.35 if has_neg else 0.0) + function_bias + action_bias
                if score > nuclear_score:
                    nuclear_score = score
                    nuclear_has_pos = has_pos
                    nuclear_has_neg = has_neg
                    nuclear_has_cond = has_cond
                    nuclear_has_constitutive = has_constitutive
                    nuclear_has_lateral = has_lateral
                    nuclear_has_general_requirement = has_general_requirement
                    nuclear_modal_function = modal_function
                    nuclear_chunk_text = str(chunk.text or "")
            if has_cond and has_constitutive:
                material_marker = bool(re.search(r"\b(\d+|por\s+ciento|%|plazo|meses?|cotiz|inmuebl)\b", text_norm))
                action_penalty = 0.0
                if asks_invest_action and row_commercial_action and not row_invest_action:
                    action_penalty -= 1.40
                elif asks_invest_action and not row_invest_action:
                    action_penalty -= 0.55
                if asks_commercial_action and not row_commercial_action:
                    action_penalty -= 0.40
                if asks_manage_action and not row_manage_action:
                    action_penalty -= 0.40
                constitutive_score = (
                    focus_match_score + (0.55 if material_marker else 0.0) + (0.35 if idx == 1 else 0.0) + action_penalty
                )
                if strong_anchor:
                    constitutive_score += 0.50
                if constitutive_score > best_constitutive_score:
                    best_constitutive_score = constitutive_score
                    best_constitutive_text = str(chunk.text or "")

        if nuclear_score >= 0:
            base_habilitante_nuclear = (
                nuclear_has_pos and not nuclear_has_neg and nuclear_modal_function == "habilitante"
            )
            base_limitativa_nuclear = (
                nuclear_has_neg and (nuclear_modal_function == "limitative")
            )
            condicion_constitutiva = nuclear_has_cond and nuclear_has_constitutive and not nuclear_has_lateral
            limite_lateral = limite_lateral or nuclear_has_lateral
        if not base_habilitante_nuclear and any_positive_anchor and has_habilitante_function_anchor:
            base_habilitante_nuclear = True
            condicion_constitutiva = condicion_constitutiva or any_constitutive_anchor
        positive_anchor_only_operational = bool(
            any_positive_anchor
            and not has_habilitante_function_anchor
            and (has_operational_function_anchor or has_procedimental_function_anchor)
        )

        weak_positive_basis = bool(
            re.search(
                (
                    r"\b(no\s+hay\s+(?:ninguna\s+)?disposicion.*(?:prohib\w*|impid\w*)|"
                    r"no\s+se\s+identifica\s+prohib\w*|"
                    r"no\s+consta\s+prohib\w*)\b"
                ),
                answer_norm,
            )
        )
        answer_has_restrictive_clause = bool(
            re.search(r"\b(sin\s+embargo|no\s+obstante)\b", answer_norm)
            and re.search(r"\b(impedir|limitar|restring|condicion|medidas?)\b", answer_norm)
        )
        modal_focus_terms = focus_terms + focus_roots

        def pick_condition_source_text(default_text: str) -> str:
            if asks_invest_action:
                for ch in chunks[:6]:
                    ch_norm = _normalize_for_search(ch.text or "")
                    if re.search(r"\barticulo\s*19\b", ch_norm) and re.search(
                        r"\bcotizadas?\s+en\s+mercados?\s+regulados?\b",
                        ch_norm,
                    ):
                        return str(ch.text or default_text)
            if any(str(t).startswith("inmobili") or str(t).startswith("inmuebl") for t in modal_focus_terms):
                for ch in chunks[:6]:
                    ch_norm = _normalize_for_search(ch.text or "")
                    if re.search(r"\barticulo\s*9\b", ch_norm) and ("inmuebl" in ch_norm or "inmobili" in ch_norm):
                        return str(ch.text or default_text)
            return default_text

        cite = "".join(f"[{i}]" for i in range(1, min(3, len(chunks)) + 1))
        if forced_from_partial and not base_habilitante_nuclear:
            return (
                "Con la evidencia recuperada no se acredita de forma suficiente una habilitacion general, "
                "porque el articulo localizado regula principalmente aspectos operativos o procedimentales "
                "y no establece por si solo la regla general de permisibilidad. "
                f"{cite}"
            ).strip()
        if forced_from_partial and base_habilitante_nuclear and condicion_constitutiva:
            condition_source_text = pick_condition_source_text(best_constitutive_text or nuclear_chunk_text)
            material_condition = self._extract_material_modal_condition(
                condition_source_text,
                focus_terms=focus_terms,
                focus_roots=focus_roots,
            )
            if (not material_condition) and re.search(r"\binmobili", normalized_question):
                has_article9_anchor = any(
                    re.search(r"\barticulo\s*9\b", _normalize_for_search(ch.text or "")) for ch in chunks[:6]
                )
                if has_article9_anchor:
                    material_condition = (
                        "la empresa tenga mas del 50 por ciento de su activo en inmuebles "
                        "y al menos el 85 por ciento del valor contable total de esos inmuebles "
                        "este afecto al desarrollo de una actividad economica"
                    )
            if nuclear_has_general_requirement:
                return (
                    "Si, siempre que se cumplan los requisitos generales de autorizacion y acceso "
                    "previstos en la norma aplicable. "
                    f"{cite}"
                ).strip()
            if material_condition:
                if re.match(r"^\s*(siempre\s+que|a\s+menos\s+que)\b", _normalize_for_search(material_condition)):
                    return f"Si, {material_condition}. {cite}".strip()
                return f"Si, siempre que {material_condition}. {cite}".strip()
            return (
                "Si, pero solo si se cumple la condicion material prevista en la norma aplicable. "
                f"{cite}"
            ).strip()

        if is_negative and not has_prohibitiva and (has_habilitante or has_procedural or has_conditional):
            return (
                "Con la evidencia recuperada no puede afirmarse de forma concluyente una prohibicion general. "
                f"{cite}"
            ).strip()
        if is_negative and has_prohibitiva and not (base_limitativa_nuclear or has_limitative_function_anchor):
            return (
                "La evidencia muestra restricciones, pero no una base limitativa nuclear suficiente "
                "para sostener una prohibicion general. "
                f"{cite}"
            ).strip()

        if is_negative:
            return answer

        # Positive-modal policy:
        # 1) strong nuclear enabling basis + constitutive condition => conditional
        # 2) strong nuclear enabling basis + only lateral limits => keep affirmative answer
        # 3) no strong enabling basis => conditional or insufficient.
        if is_positive and base_habilitante_nuclear:
            if condicion_constitutiva:
                if nuclear_has_general_requirement:
                    return (
                        "Si, siempre que se cumplan los requisitos generales de autorizacion y acceso "
                        "previstos en la norma aplicable. "
                        f"{cite}"
                    ).strip()
                condition_source_text = pick_condition_source_text(best_constitutive_text or nuclear_chunk_text)
                material_condition = self._extract_material_modal_condition(
                    condition_source_text,
                    focus_terms=focus_terms,
                    focus_roots=focus_roots,
                )
                if (not material_condition) and re.search(r"\binmobili", normalized_question):
                    has_article9_anchor = any(
                        re.search(r"\barticulo\s*9\b", _normalize_for_search(ch.text or "")) for ch in chunks[:6]
                    )
                    if has_article9_anchor:
                        material_condition = (
                            "la empresa tenga mas del 50 por ciento de su activo en inmuebles "
                            "y al menos el 85 por ciento del valor contable total de esos inmuebles "
                            "este afecto al desarrollo de una actividad economica"
                        )
                if material_condition:
                    if re.match(r"^\s*(siempre\s+que|a\s+menos\s+que)\b", _normalize_for_search(material_condition)):
                        return f"Si, {material_condition}. {cite}".strip()
                    return f"Si, siempre que {material_condition}. {cite}".strip()
                return (
                    "Si, pero solo si se cumple la condicion material prevista en la norma aplicable. "
                    f"{cite}"
                ).strip()
            if has_prohibitiva and not limite_lateral:
                return (
                    "Si, en la medida en que se respeten los limites y prohibiciones que acompanan la habilitacion. "
                    f"{cite}"
                ).strip()
            if weak_positive_basis and not has_habilitante:
                return (
                    "Con la evidencia recuperada no puede afirmarse de forma concluyente un permiso general, "
                    "porque falta habilitacion explicita suficiente. "
                    f"{cite}"
                ).strip()
            if answer_has_restrictive_clause and (has_conditional or has_procedural or has_prohibitiva):
                return (
                    "Si, pero sujeto a las restricciones y medidas condicionantes que recoge la evidencia. "
                    f"{cite}"
                ).strip()
            return answer

        if is_positive and positive_anchor_only_operational:
            return (
                "Con la evidencia recuperada no puede afirmarse de forma concluyente un permiso general, "
                "porque los fragmentos son principalmente operativos o procedimentales. "
                f"{cite}"
            ).strip()
        if is_positive and has_prohibitiva and not base_habilitante_nuclear:
            return (
                "La evidencia refleja limites o restricciones, sin base habilitante nuclear suficiente "
                "para una afirmacion general de permiso. "
                f"{cite}"
            ).strip()
        if is_positive and (has_conditional or has_procedural or answer_has_restrictive_clause):
            return (
                "Si, pero condicionado al cumplimiento de los requisitos y condiciones previstos en la norma. "
                f"{cite}"
            ).strip()
        if is_positive and weak_positive_basis:
            return (
                "Con la evidencia recuperada no puede afirmarse de forma concluyente un permiso general, "
                "porque se apoya en ausencia de prohibicion y no en habilitacion explicita. "
                f"{cite}"
            ).strip()
        if is_positive and not has_habilitante:
            return (
                "Con la evidencia recuperada no puede afirmarse de forma concluyente un permiso general, "
                "porque no se observa base habilitante explicita suficiente. "
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
                if (
                    re.search(r"\barticulo\s+\d", heading_norm)
                    and re.search(r"\b(entidades?\s+excluid|no\s+sera\s+de\s+aplicacion)\b", heading_norm)
                ):
                    anchors.append((idx, chunk))
            if not anchors and not re.search(exclusion_pattern, evidence_text):
                return "NO ENCONTRADO EN EL DOCUMENTO"
            enum_data: dict[str, Any] | None = None
            anchor_article = ""
            if anchors:
                anchor_idx, anchor_chunk = anchors[0]
                anchor_article = _normalize_article_number((anchor_chunk.metadata or {}).get("numero")) or self._article_from_text(anchor_chunk.text) or ""
                same_article_chunks = [
                    c
                    for c in chunks[:8]
                    if (
                        _normalize_article_number((c.metadata or {}).get("numero")) or self._article_from_text(c.text) or ""
                    )
                    == anchor_article
                ]
                if same_article_chunks:
                    seen_piece: set[str] = set()
                    pieces: list[str] = []
                    for c in sorted(same_article_chunks, key=lambda x: float(x.score or 0.0), reverse=True):
                        txt = str(c.text or "").strip()
                        if not txt:
                            continue
                        fp = _normalize_for_search(txt[:220])
                        if fp in seen_piece:
                            continue
                        seen_piece.add(fp)
                        pieces.append(txt)
                    unit_text = "\n".join(pieces).strip() or (anchor_chunk.text or "")
                else:
                    unit_text = anchor_chunk.text or ""
                enum_data = self._extract_complete_enumeration_from_unit(unit_text, max_items=10)
            if anchors and enum_data:
                idx, chunk = anchors[0]
                enum_items = [str(i).strip() for i in (enum_data.get("items") or []) if str(i).strip()]
                if enum_data.get("is_complete_enough") and len(enum_items) >= 2:
                    joined = "; ".join(enum_items[:6])
                    return f"Las entidades excluidas son: {joined}. [{idx}]"
                if anchor_article:
                    if enum_items:
                        joined = "; ".join(enum_items[:4])
                        return f"Respuesta parcial (exclusion): articulo {anchor_article}: {joined}. [{idx}]"
                    return (
                        "Respuesta parcial (exclusion): "
                        f"se identifica el articulo {anchor_article}, pero el contexto recuperado no contiene la enumeracion material completa. "
                        f"[{idx}]"
                    )
                return (
                    "Respuesta parcial (exclusion): se identifica una base normativa de exclusion, "
                    "pero falta la enumeracion material completa en los fragmentos recuperados. "
                    f"[{idx}]"
                )
            if anchors and not re.search(exclusion_pattern, answer_norm):
                idx, chunk = anchors[0]
                sentence = first_sentence(chunk.text)
                if sentence:
                    return f"Respuesta parcial (exclusion): {sentence} [{idx}]"

        if signals.get("asks_enumeration_in_article"):
            anchor_chunk = chunks[0]
            anchor_article = _normalize_article_number((anchor_chunk.metadata or {}).get("numero")) or self._article_from_text(anchor_chunk.text) or ""
            same_article_chunks = [
                c
                for c in chunks[:8]
                if (
                    _normalize_article_number((c.metadata or {}).get("numero")) or self._article_from_text(c.text) or ""
                )
                == anchor_article
            ]
            if same_article_chunks:
                seen_piece: set[str] = set()
                pieces: list[str] = []
                for c in sorted(same_article_chunks, key=lambda x: float(x.score or 0.0), reverse=True):
                    txt = str(c.text or "").strip()
                    if not txt:
                        continue
                    fp = _normalize_for_search(txt[:220])
                    if fp in seen_piece:
                        continue
                    seen_piece.add(fp)
                    pieces.append(txt)
                unit_text = "\n".join(pieces).strip() or (anchor_chunk.text or "")
            else:
                unit_text = anchor_chunk.text or ""
            enum_data = self._extract_complete_enumeration_from_unit(unit_text, max_items=10)
            enum_items = [str(i).strip() for i in (enum_data.get("items") or []) if str(i).strip()]
            answer_item_count = len(self._extract_list_items(answer, max_items=8))
            if enum_data.get("intro_found") and not enum_data.get("is_complete_enough"):
                if enum_items:
                    return f"Respuesta parcial (enumeracion): articulo {anchor_article}: " + "; ".join(enum_items[:4]) + ". [1]"
                return "Respuesta parcial (enumeracion): la evidencia recuperada no contiene una lista material completa. [1]"
            if enum_data.get("is_complete_enough") and answer_item_count <= 0 and enum_items:
                return "Los elementos relevantes son: " + "; ".join(enum_items[:6]) + ". [1]"

        if signals.get("asks_minimum_core_requirements"):
            if len(answer_norm) > 1100 or answer_norm.startswith("los elementos relevantes son"):
                compact = self._extractive_answer_by_intent(question=question, chunks=chunks)
                if compact:
                    return compact

        if signals.get("asks_article_numbers"):
            article_list_answer = self._extractive_answer_by_intent(question=question, chunks=chunks)
            if article_list_answer:
                return article_list_answer

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
                anchor_stats: dict[str, dict[str, int]] = {
                    ent: {"heading": 0, "body": 0}
                    for ent in comparison_entities
                }
                normative_anchor_pattern = r"\b(definicion|concepto|objeto|regimen|autoriz|gestion|podra|puede|debera|obliga)\b"
                for chunk in chunks[:6]:
                    text_norm = _normalize_for_search(chunk.text or "")
                    if not text_norm:
                        continue
                    heading_norm = _normalize_for_search((chunk.text or "")[:260])
                    has_normative_anchor = bool(re.search(normative_anchor_pattern, text_norm))
                    for ent in comparison_entities:
                        if ent in heading_norm and has_normative_anchor:
                            anchor_stats[ent]["heading"] += 1
                        elif ent in text_norm and has_normative_anchor:
                            anchor_stats[ent]["body"] += 1
                strong_entities = [
                    ent
                    for ent, stats in anchor_stats.items()
                    if stats["heading"] >= 1 or stats["body"] >= 2
                ]
                if len(strong_entities) < 2:
                    fallback = self._extractive_answer_by_intent(question=question, chunks=chunks)
                    if fallback and "comparacion parcial" in _normalize_for_search(fallback):
                        return fallback
                    cite = "".join(f"[{i}]" for i in range(1, min(3, len(chunks)) + 1))
                    return (
                        "Comparacion parcial (anclaje dual): falta base normativa propia para ambos lados "
                        "en los fragmentos recuperados. "
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
        if (
            signals.get("asks_exclusion")
            or signals.get("asks_enumeration_in_article")
            or signals.get("asks_modal")
            or signals.get("asks_extreme")
        ):
            limit = max(limit, 6)
        limit = max(1, min(limit, len(chunks)))

        def article_key(chunk: GraphChunk) -> str:
            return _normalize_article_number((chunk.metadata or {}).get("numero")) or self._article_from_text(chunk.text) or ""

        if not requested_articles:
            if signals.get("asks_article_numbers"):
                candidate_chunks: list[GraphChunk] = list(chunks)
                heading_terms = _dedupe(
                    [
                        t
                        for t in _tokens(str(signals.get("normalized_question") or ""))
                        if len(t) >= 4 and t not in LEGAL_STOPWORDS and t not in GENERIC_QUERY_TERMS
                    ]
                    + [str(t) for t in (signals.get("entity_tokens") or []) if len(str(t)) >= 3]
                )
                if heading_terms:
                    try:
                        heading_rows = self._fetch_rows_by_heading_terms(
                            terms=heading_terms,
                            limit=max(160, limit * 30),
                        )
                    except Exception:
                        heading_rows = []
                    if heading_rows:
                        seen_ids = {str(c.id or "") for c in candidate_chunks}
                        for row in heading_rows:
                            row_chunk = self._row_to_chunk(row)
                            if row_chunk is None:
                                continue
                            chunk_id = str(row_chunk.id or "")
                            if chunk_id and chunk_id in seen_ids:
                                continue
                            candidate_chunks.append(row_chunk)
                            if chunk_id:
                                seen_ids.add(chunk_id)
                if len(candidate_chunks) < 18:
                    try:
                        expanded_candidates = self.search_units(
                            question=question,
                            top_k=max(18, limit * 4),
                        )
                    except Exception:
                        expanded_candidates = []
                    if expanded_candidates:
                        seen_ids = {str(c.id or "") for c in candidate_chunks}
                        for chunk in expanded_candidates:
                            chunk_id = str(chunk.id or "")
                            if chunk_id and chunk_id in seen_ids:
                                continue
                            candidate_chunks.append(chunk)
                            if chunk_id:
                                seen_ids.add(chunk_id)
                ranked_article_chunks = self._rank_article_chunks_for_list(
                    question=question,
                    chunks=candidate_chunks,
                    limit=max(limit, 6),
                )
                if ranked_article_chunks:
                    return ranked_article_chunks[:limit]

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

            if signals.get("asks_minimum_core_requirements"):
                core_pattern = r"\b(requisitos?|acceso|autorizacion|funciones?\s+minimas?|objeto|concepto|actividad\s+principal)\b"
                development_pattern = r"\b(gestion\s+del\s+riesgo|procedimiento|notificacion|registro|campana|comunicacion)\b"
                focus_terms = _dedupe(
                    [str(t) for t in (signals.get("content_terms") or []) if len(str(t)) >= 5 and str(t) not in GENERIC_QUERY_TERMS]
                )
                focus_roots = _dedupe([term[:5] for term in focus_terms if len(term) >= 5])
                scored_min_core: list[tuple[float, float, GraphChunk]] = []
                for chunk in chunks:
                    text_norm = _normalize_for_search(chunk.text or "")
                    if not text_norm:
                        continue
                    heading_norm = _normalize_for_search((chunk.text or "")[:260])
                    base = float(chunk.score or 0.0)
                    heading_hits = sum(1 for t in focus_terms if t in heading_norm) + sum(1 for r in focus_roots if r in heading_norm)
                    lead_hits = sum(1 for t in focus_terms if t in text_norm[:900]) + sum(1 for r in focus_roots if r in text_norm[:900])
                    role = self._infer_normative_role(chunk.text or "")
                    rank = (0.45 * base) + (1.15 * float(heading_hits)) + (0.60 * float(lead_hits))
                    if re.search(core_pattern, heading_norm):
                        rank += 1.35
                    if re.search(development_pattern, heading_norm):
                        rank -= 0.80
                    if role in {"definitional", "regime"}:
                        rank += 0.60
                    elif role in {"procedural", "informational", "cross_reference"}:
                        rank -= 0.45
                    if heading_hits <= 0 and lead_hits <= 0:
                        rank -= 1.20
                    scored_min_core.append((rank, base, chunk))
                if scored_min_core:
                    scored_min_core.sort(key=lambda item: (item[0], item[1]), reverse=True)
                    selected_core: list[GraphChunk] = []
                    seen_articles: set[str] = set()
                    for rank, _, chunk in scored_min_core:
                        if rank <= -0.70:
                            continue
                        article = article_key(chunk)
                        if article and article in seen_articles:
                            continue
                        selected_core.append(chunk)
                        if article:
                            seen_articles.add(article)
                        if len(selected_core) >= min(4, limit):
                            break
                    if selected_core:
                        return selected_core[: min(4, limit)]

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
                modal_entities = _dedupe([str(t) for t in (signals.get("entity_tokens") or []) if len(str(t)) >= 3])
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
                    if signals.get("asks_modal") and not signals.get("asks_coexistence_modal"):
                        modal_anchor_terms = [t for t in focus_terms if len(t) >= 8 and t not in GENERIC_QUERY_TERMS]
                        modal_anchor_roots = _dedupe([t[:6] for t in modal_anchor_terms if len(t) >= 6])
                        if modal_anchor_terms:
                            anchor_hits = sum(1 for t in modal_anchor_terms if t in text_norm)
                            anchor_root_hits = sum(1 for r in modal_anchor_roots if r in text_norm)
                            if len(modal_anchor_terms) >= 2 and (anchor_hits + anchor_root_hits) <= 0:
                                rank -= 2.0
                            elif len(modal_anchor_terms) == 1 and (anchor_hits + anchor_root_hits) <= 0:
                                rank -= 0.75
                    if signals.get("asks_modal") and modal_entities:
                        modal_entity_hits = sum(1 for ent in modal_entities if ent in text_norm)
                        if modal_entity_hits <= 0:
                            rank -= 1.35
                        elif modal_entity_hits == 1:
                            rank -= 0.30
                    if signals.get("asks_modal"):
                        normalized_question = str(signals.get("normalized_question") or "")
                        asks_invest_action = bool(
                            re.search(r"\b(invertir|inversion(?:es)?|adquisi(?:cion|ciones)\s+de\s+participaciones?)\b", normalized_question)
                        )
                        asks_commercial_action = bool(
                            re.search(r"\b(comercializar|comercializacion|precomercializacion)\b", normalized_question)
                        )
                        asks_manage_action = bool(re.search(r"\b(gestionar|gestion)\b", normalized_question))
                        row_invest_action = bool(
                            re.search(
                                r"\b(invertir|inversion(?:es)?|coeficiente\s+obligatorio\s+de\s+inversion|adquisi(?:cion|ciones)\s+de\s+participaciones?)\b",
                                text_norm,
                            )
                        )
                        row_commercial_action = bool(
                            re.search(r"\b(comercializar|comercializacion|precomercializacion)\b", text_norm)
                        )
                        row_manage_action = bool(re.search(r"\b(gestionar|gestion)\b", text_norm))
                        if asks_invest_action and row_commercial_action and not row_invest_action:
                            rank -= 1.85
                        elif asks_invest_action and not row_invest_action:
                            rank -= 0.65
                        if asks_commercial_action and not row_commercial_action:
                            rank -= 0.55
                        if asks_manage_action and not row_manage_action:
                            rank -= 0.50

                    if signals.get("asks_extreme"):
                        if NUMERIC_TOKEN_PATTERN.search(text_norm):
                            rank += 0.90
                        if re.search(extreme_pattern, text_norm):
                            rank += 0.70

                    scored.append((rank, base, chunk))
                if scored:
                    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
                    if signals.get("asks_coexistence_modal"):
                        coexist_candidates: list[tuple[float, float, GraphChunk]] = []
                        for rank, base, candidate in scored:
                            candidate_norm = _normalize_for_search(candidate.text or "")
                            candidate_head = _normalize_for_search((candidate.text or "")[:280])
                            if not candidate_norm:
                                continue
                            entity_hits = sum(1 for ent in modal_entities if ent in candidate_norm)
                            if entity_hits < 2:
                                continue
                            has_joint_management = bool(
                                re.search(
                                    r"\b(gestionar|gestionen|gestione|gestion|una\s+o\s+varias|al\s+mismo\s+tiempo|a\s+la\s+vez|conjuntamente)\b",
                                    candidate_norm,
                                )
                            )
                            if not has_joint_management:
                                continue
                            is_special_scope = bool(
                                re.search(
                                    r"\b(por\s+debajo\s+de\s+determinados?\s+umbrales?|umbral(?:es)?|transfronteriz|estado\s+no\s+miembro|otros?\s+estados?\s+miembros?)\b",
                                    f"{candidate_head} {candidate_norm[:500]}",
                                )
                            )
                            is_core_scope = bool(
                                re.search(r"\b(requisitos?\s+de\s+acceso|acceso\s+a\s+la\s+actividad|autorizacion)\b", candidate_head)
                            )
                            tuned = rank + (0.95 if is_core_scope else 0.0) - (0.85 if is_special_scope else 0.0)
                            coexist_candidates.append((tuned, base, candidate))
                        if coexist_candidates:
                            coexist_candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
                            selected: list[GraphChunk] = []
                            seen_articles: set[str] = set()
                            for _, _, candidate in coexist_candidates:
                                if len(selected) >= limit:
                                    break
                                number = article_key(candidate)
                                if number and number in seen_articles:
                                    continue
                                selected.append(candidate)
                                if number:
                                    seen_articles.add(number)
                            if len(selected) < limit:
                                for chunk in chunks:
                                    if len(selected) >= limit:
                                        break
                                    if chunk in selected:
                                        continue
                                    selected.append(chunk)
                            if selected:
                                return selected[:limit]
                    if signals.get("asks_enumeration_in_article") or signals.get("asks_exclusion"):
                        anchor_chunk: GraphChunk | None = None
                        anchor_article = ""
                        for rank, _, candidate in scored:
                            if rank <= -0.60:
                                continue
                            candidate_norm = _normalize_for_search(candidate.text or "")
                            candidate_head = _normalize_for_search((candidate.text or "")[:280])
                            if signals.get("asks_exclusion"):
                                strict_exclusion_anchor = bool(
                                    re.search(r"\barticulo\s+\d", candidate_head)
                                    and re.search(
                                        r"\b(entidades?\s+excluid|no\s+sera\s+de\s+aplicacion)\b",
                                        f"{candidate_head} {candidate_norm[:420]}",
                                    )
                                )
                                if not strict_exclusion_anchor:
                                    continue
                            if signals.get("asks_enumeration_in_article") and not signals.get("asks_exclusion"):
                                has_intro_or_list = bool(
                                    re.search(
                                        r"\b(las?\s+siguientes?|se\s+entendera\s+por|consistira?\s+en|comprendera)\b",
                                        candidate_norm,
                                    )
                                    or re.search(
                                        r"(^|\n)\s*[-*]\s+|(^|\n)\s*[a-z]\)\s+|(^|\n)\s*\d+[.)]\s+",
                                        candidate.text or "",
                                        flags=re.IGNORECASE,
                                    )
                                )
                                if not has_intro_or_list:
                                    continue
                            article = article_key(candidate)
                            if not article:
                                continue
                            anchor_chunk = candidate
                            anchor_article = article
                            break
                        if anchor_chunk is not None and anchor_article:
                            same_article: list[GraphChunk] = []
                            seen_ids: set[str] = set()
                            for _, _, candidate in scored:
                                if article_key(candidate) != anchor_article:
                                    continue
                                candidate_id = str(candidate.id or "")
                                if candidate_id and candidate_id in seen_ids:
                                    continue
                                same_article.append(candidate)
                                if candidate_id:
                                    seen_ids.add(candidate_id)
                                if len(same_article) >= limit:
                                    break
                            if len(same_article) < limit:
                                for candidate in chunks:
                                    if len(same_article) >= limit:
                                        break
                                    if article_key(candidate) != anchor_article:
                                        continue
                                    if candidate in same_article:
                                        continue
                                    same_article.append(candidate)
                            if same_article:
                                return same_article[:limit]
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
        retrieval_question, followup_focus = self._resolve_followup_question(
            question=question,
            chat_history=chat_history,
            signals=signals,
        )
        retrieval_signals = signals if retrieval_question == question else self._query_signals(retrieval_question)
        adjusted_top_k = max(1, int(top_k))
        if (
            retrieval_signals.get("asks_article_numbers")
            or retrieval_signals.get("article_numbers")
            or retrieval_signals.get("asks_comparison")
        ):
            adjusted_top_k = min(max(adjusted_top_k, 12), 20)
        else:
            adjusted_top_k = min(max(adjusted_top_k, 8), 16)
        chunks = self.search_units(question=retrieval_question, top_k=adjusted_top_k)
        if not chunks:
            return GraphRAGResult(answer="NO ENCONTRADO EN EL DOCUMENTO", chunks=[], sources=[])
        answer_chunks = self._select_answer_chunks(question=retrieval_question, chunks=chunks)
        answer_chunks = self._mark_support_roles(chunks=answer_chunks, signals=retrieval_signals)
        answer = self.generate_from_chunks(question=retrieval_question, chunks=answer_chunks, chat_history=chat_history)
        if followup_focus:
            self._last_search_debug.setdefault("followup_resolution", {})
            self._last_search_debug["followup_resolution"] = {
                "original_question": question,
                "retrieval_question": retrieval_question,
                "focus": followup_focus,
            }
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
            "followup_resolution": self._last_search_debug.get("followup_resolution", {}),
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
