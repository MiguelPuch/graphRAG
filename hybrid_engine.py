"""Hybrid orchestrator: technical Milvus RAG + legal Neo4j GraphRAG."""

from __future__ import annotations

import logging
import re
import threading
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import (
    NORMATIVA_SOURCE_DIR,
    ORCH_LEGAL_KEYWORD_COVERAGE,
    ORCH_LEGAL_MIN_CHUNKS,
    ORCH_LEGAL_MIN_SCORE,
    ORCH_PROBE_TOP_K,
    ORCH_TECH_MIN_CHUNKS,
    ORCH_TECH_MIN_SCORE,
    SUPPORTED_INGEST_EXTENSIONS,
)
from graph_rag_engine import GraphChunk, GraphRAGEngine
from rag_engine import Chunk, RAGEngine
from routing import (
    QueryRouter,
    classify_document_name,
    is_bridge_legal_document,
    normalize_for_matching,
)

logger = logging.getLogger(__name__)


@dataclass
class HybridQueryResult:
    answer: str
    sources: list[str]
    chunks: list[Chunk]
    route: str
    route_reason: str
    route_confidence: float


class HybridMetrics:
    def __init__(self):
        self._lock = threading.Lock()
        self.query_total = 0
        self.query_by_route = {"legal": 0, "technical": 0, "both": 0}
        self.query_failures = 0
        self.ingest_total = 0
        self.ingest_failures = 0
        self.last_query_ts = None

    def mark_query(self, route: str, success: bool) -> None:
        with self._lock:
            self.query_total += 1
            if route in self.query_by_route:
                self.query_by_route[route] += 1
            if not success:
                self.query_failures += 1
            self.last_query_ts = int(time.time())

    def mark_ingest(self, success: bool) -> None:
        with self._lock:
            self.ingest_total += 1
            if not success:
                self.ingest_failures += 1

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "query_total": self.query_total,
                "query_by_route": dict(self.query_by_route),
                "query_failures": self.query_failures,
                "ingest_total": self.ingest_total,
                "ingest_failures": self.ingest_failures,
                "last_query_ts": self.last_query_ts,
            }


class HybridRAGEngine:
    def __init__(self):
        self.technical = RAGEngine()
        self.legal = GraphRAGEngine()
        self.router = QueryRouter()
        self.metrics = HybridMetrics()

    def close(self) -> None:
        self.legal.close()

    @staticmethod
    def _is_markdown_name(filename: str) -> bool:
        return Path(filename).suffix.lower() in {".md", ".markdown"}

    @staticmethod
    def _decode_markdown_bytes(content: bytes) -> str:
        return content.decode("utf-8", errors="ignore")

    def _should_bridge_legal_to_technical(self, filename: str, markdown: str) -> bool:
        if is_bridge_legal_document(filename):
            return True
        text = normalize_for_matching((markdown or "")[:250000])
        if not text:
            return False
        markers = 0
        patterns = [
            r"\b(lqb|bg)\s*[-_]?\s*\d",
            r"\bestado\s+[a-z]{1,4}\d{1,4}(?:-[a-z]+)?\b",
            r"\ba0\d\b",
            r"\b(celda|columna|codigo|cumplimentacion|tramo|ratio)\b",
        ]
        for pattern in patterns:
            hits = re.findall(pattern, text)
            markers += len(hits)
        return markers >= 6

    def ingest_technical_bytes(self, content: bytes, filename: str) -> dict[str, Any]:
        if self._is_markdown_name(filename):
            markdown = self._decode_markdown_bytes(content)
            result = self.technical.ingest_markdown_content(markdown=markdown, filename=filename)
        else:
            result = self.technical.ingest_bytes(content, filename)
        self.metrics.mark_ingest(success=True)
        return {"target": "technical", **result}

    def ingest_legal_bytes(self, content: bytes, filename: str) -> dict[str, Any]:
        if self._is_markdown_name(filename):
            markdown = self._decode_markdown_bytes(content)
            doc_slug = self.technical.converter._create_slug(filename)
        else:
            markdown, doc_slug = self.technical.converter.convert_bytes_to_markdown(content, filename)
        result = self.legal.ingest_markdown(markdown, filename)
        bridge_to_technical = self._should_bridge_legal_to_technical(filename=filename, markdown=markdown)
        technical_result = None
        if bridge_to_technical:
            technical_result = self.technical.ingest_markdown_content(markdown=markdown, filename=filename, doc_slug=doc_slug)
        self.metrics.mark_ingest(success=True)
        payload = {
            "target": "both" if bridge_to_technical else "legal",
            "doc_slug": doc_slug,
            "filename": filename,
            "legal": result,
        }
        if technical_result is not None:
            payload["technical"] = technical_result
        return payload

    def ingest_auto_bytes(self, content: bytes, filename: str) -> dict[str, Any]:
        classification = classify_document_name(filename)
        kind = classification.domain
        if kind == "legal":
            return self.ingest_legal_bytes(content, filename)
        if kind == "technical":
            return self.ingest_technical_bytes(content, filename)

        # If ambiguous, index in both engines.
        legal = self.ingest_legal_bytes(content, filename)
        technical = self.ingest_technical_bytes(content, filename)
        return {
            "target": "both",
            "filename": filename,
            "legal": legal,
            "technical": technical,
        }

    def ingest_corpus_folder(self, root: str | Path | None = None) -> dict[str, Any]:
        folder = Path(root or NORMATIVA_SOURCE_DIR)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")

        results: list[dict[str, Any]] = []
        failed: list[dict[str, Any]] = []

        for path in sorted(folder.rglob("*")):
            if (not path.is_file()) or (path.suffix.lower() not in SUPPORTED_INGEST_EXTENSIONS):
                continue
            try:
                content = path.read_bytes()
                if path.suffix.lower() == ".md":
                    kind = classify_document_name(path.name).domain
                    if kind == "legal":
                        res = self.ingest_legal_bytes(content=content, filename=path.name)
                        results.append({"file": str(path), **res})
                    else:
                        res = self.technical.ingest_markdown_file(path, filename=path.name)
                        self.metrics.mark_ingest(success=True)
                        results.append({"file": str(path), "target": "technical", "result": res})
                    continue

                res = self.ingest_auto_bytes(content=content, filename=path.name)
                results.append({"file": str(path), **res})
            except Exception as exc:
                self.metrics.mark_ingest(success=False)
                failed.append({"file": str(path), "error": str(exc)})

        return {
            "root": str(folder),
            "processed": len(results),
            "failed": len(failed),
            "results": results,
            "errors": failed,
        }

    def _query_keywords(self, question: str) -> list[str]:
        norm = normalize_for_matching(question or "")
        tokens = re.findall(r"[a-z0-9]{4,}", norm)
        stop = {
            "como",
            "donde",
            "cual",
            "cuales",
            "sobre",
            "para",
            "esta",
            "este",
            "que",
            "con",
            "sin",
            "pero",
            "cuando",
            "legalmente",
            "tecnicamente",
            "operativamente",
        }
        out: list[str] = []
        for token in tokens:
            if token in stop:
                continue
            if token in out:
                continue
            out.append(token)
        return out[:20]

    @staticmethod
    def _is_anaphoric_question(question: str) -> bool:
        q = normalize_for_matching(question or "")
        q = "".join(ch for ch in unicodedata.normalize("NFKD", q) if not unicodedata.combining(ch))
        if not q:
            return False
        if len(q) <= 40 and re.search(r"^(y|entonces|en que condiciones|que condiciones|como|y eso|cual de)", q):
            return True
        return bool(re.search(r"\b(eso|ello|en ese caso|en este caso|en que condiciones)\b", q))

    def _expand_followup_question(self, question: str, chat_history: list[dict] | None) -> str:
        if not chat_history or not self._is_anaphoric_question(question):
            return question
        prev_user = ""
        for msg in reversed(chat_history):
            if str(msg.get("role") or "").strip().lower() != "user":
                continue
            content = str(msg.get("content") or "").strip()
            if content:
                prev_user = content
                break
        if not prev_user:
            return question
        return f"{prev_user} {question}".strip()

    def _keyword_coverage(self, question: str, texts: list[str]) -> float:
        kws = self._query_keywords(question)
        if not kws:
            return 0.0
        text = normalize_for_matching(" ".join(texts)[:20000])
        hits = sum(1 for k in kws if k in text)
        return hits / max(len(kws), 1)

    def _evaluate_technical_evidence(self, question: str, chunks: list[Chunk]) -> dict[str, Any]:
        chunk_count = len(chunks)
        top_score = float(chunks[0].score or 0.0) if chunks else 0.0
        keyword_coverage = self._keyword_coverage(question, [c.text for c in chunks[:8]])
        score_pass = top_score >= ORCH_TECH_MIN_SCORE
        coverage_pass = keyword_coverage >= 0.45
        evidence_ok = chunk_count >= ORCH_TECH_MIN_CHUNKS and (score_pass or coverage_pass)
        reason = (
            f"Evidencia tecnica {'suficiente' if evidence_ok else 'insuficiente'} "
            f"(top_score={top_score:.3f}, chunks={chunk_count}, coverage={keyword_coverage:.2f})"
        )
        return {
            "evidence_ok": evidence_ok,
            "chunk_count": chunk_count,
            "top_score": top_score,
            "keyword_coverage": keyword_coverage,
            "required_min_score": ORCH_TECH_MIN_SCORE,
            "reason": reason,
        }

    def _evaluate_legal_evidence(self, question: str, chunks: list[GraphChunk]) -> dict[str, Any]:
        chunk_count = len(chunks)
        top_score = float(chunks[0].score or 0.0) if chunks else 0.0
        keyword_coverage = self._keyword_coverage(question, [c.text for c in chunks[:8]])
        score_pass = top_score >= ORCH_LEGAL_MIN_SCORE
        coverage_pass = keyword_coverage >= ORCH_LEGAL_KEYWORD_COVERAGE
        evidence_ok = (chunk_count >= ORCH_LEGAL_MIN_CHUNKS and (score_pass or coverage_pass)) or (
            chunk_count >= 1 and top_score >= 0.45
        )
        reason = (
            f"Evidencia legal {'suficiente' if evidence_ok else 'insuficiente'} "
            f"(top_score={top_score:.3f}, chunks={chunk_count}, coverage={keyword_coverage:.2f})"
        )
        return {
            "evidence_ok": evidence_ok,
            "chunk_count": chunk_count,
            "top_score": top_score,
            "keyword_coverage": keyword_coverage,
            "required_min_score": ORCH_LEGAL_MIN_SCORE,
            "reason": reason,
        }

    def _enrich_question_with_states(self, question: str, bridge_states: list[dict[str, Any]]) -> str:
        codes = sorted({str(item.get("codigo_estado") or "").strip() for item in bridge_states if item.get("codigo_estado")})
        if not codes:
            return question
        states_hint = " ".join(f"estado {code}" for code in codes)
        return f"{question} {states_hint}".strip()

    def _build_answer(
        self,
        route_final: str,
        question: str,
        legal_chunks: list[GraphChunk],
        technical_chunks: list[Chunk],
        chat_history: list[dict] | None,
    ) -> tuple[str, dict[str, Any], list[GraphChunk], list[Chunk]]:
        legal_answer = ""
        technical_answer = ""
        legal_answer_chunks = legal_chunks
        technical_answer_chunks = technical_chunks

        if route_final in {"legal", "both"}:
            try:
                legal_answer_chunks = self.legal._select_answer_chunks(question=question, chunks=legal_chunks)  # noqa: SLF001
            except Exception:
                legal_answer_chunks = legal_chunks
            legal_answer = self.legal.generate_from_chunks(
                question=question,
                chunks=legal_answer_chunks,
                chat_history=chat_history,
            )
        if route_final in {"technical", "both"}:
            technical_answer = self.technical.generate_from_chunks(
                question=question,
                chunks=technical_answer_chunks,
                chat_history=chat_history,
            )

        if route_final == "legal":
            return legal_answer, {
                "mode": "legal",
                "legal_answer_used": True,
                "technical_answer_used": False,
                "llm_synthesis_used": False,
            }, legal_answer_chunks, technical_answer_chunks
        if route_final == "technical":
            return technical_answer, {
                "mode": "technical",
                "legal_answer_used": False,
                "technical_answer_used": True,
                "llm_synthesis_used": False,
            }, legal_answer_chunks, technical_answer_chunks

        answer = (
            "**Bloque legal:**\n"
            f"{legal_answer}\n\n"
            "**Bloque tecnico:**\n"
            f"{technical_answer}"
        )
        return answer, {
            "mode": "both",
            "legal_answer_used": True,
            "technical_answer_used": True,
            "llm_synthesis_used": False,
        }, legal_answer_chunks, technical_answer_chunks

    def _build_output_chunks(
        self,
        route_final: str,
        legal_chunks: list[GraphChunk],
        technical_chunks: list[Chunk],
    ) -> tuple[list[Chunk], list[str]]:
        merged_chunks: list[Chunk] = []
        merged_sources: list[str] = []

        if route_final in {"technical", "both"}:
            merged_chunks.extend(technical_chunks)
            merged_sources.extend([c.filename for c in technical_chunks])

        if route_final in {"legal", "both"}:
            for c in legal_chunks:
                merged_chunks.append(
                    Chunk(
                        id=c.id,
                        doc_slug="legal_graph",
                        filename=c.source,
                        text=c.text,
                        score=c.score,
                        metadata=c.metadata or {},
                    )
                )
            merged_sources.extend([c.source for c in legal_chunks])

        return merged_chunks, merged_sources

    def _build_technical_trace(
        self,
        chunks: list[Chunk],
        evaluation: dict[str, Any],
        include_raw_text: bool,
        probe_used: bool,
    ) -> dict[str, Any]:
        trace_chunks = []
        for c in chunks:
            trace_chunks.append(
                {
                    "chunk_id": c.id,
                    "doc_slug": c.doc_slug,
                    "filename": c.filename,
                    "score": c.score,
                    "snippet": c.text[:280],
                    "text": c.text if include_raw_text else None,
                }
            )
        return {
            "queried": bool(chunks),
            "probe_used": probe_used,
            "evaluation": evaluation,
            "chunks": trace_chunks,
        }

    def _build_legal_trace(
        self,
        chunks: list[GraphChunk],
        evaluation: dict[str, Any],
        include_raw_text: bool,
        probe_used: bool,
        bridge_states: list[dict[str, Any]],
    ) -> dict[str, Any]:
        legal_debug = {}
        try:
            legal_debug = self.legal.get_last_debug_trace() or {}
        except Exception:
            legal_debug = {}

        nodes = []
        for c in chunks:
            md = c.metadata or {}
            nodes.append(
                {
                    "unit_id": md.get("unit_id") or c.id,
                    "tipo_unidad": md.get("tipo_unidad"),
                    "numero": md.get("numero"),
                    "documento_id": md.get("documento_id"),
                    "documento_titulo": md.get("documento_titulo") or c.source,
                    "score": c.score,
                    "snippet": c.text[:280],
                    "text": c.text if include_raw_text else None,
                }
            )

        return {
            "queried": bool(chunks),
            "probe_used": probe_used,
            "evaluation": evaluation,
            "nodes": nodes,
            "bridge_states": bridge_states,
            "retrieval_strategies": legal_debug.get("retrieval_strategies", {}),
            "retrieval_metrics": legal_debug.get("retrieval_metrics", {}),
            "rrf_scores": legal_debug.get("rrf_scores", {}),
            "reranker_trace": legal_debug.get("reranker_trace", {}),
            "grounding_trace": legal_debug.get("grounding_trace", {}),
            "grounding_summary": legal_debug.get("grounding_summary", {}),
            "claim_candidates_total": legal_debug.get("claim_candidates_total", 0),
            "claim_candidates_filtered": legal_debug.get("claim_candidates_filtered", 0),
            "claim_filter_reasons": legal_debug.get("claim_filter_reasons", {}),
            "question_relevance_stats": legal_debug.get("question_relevance_stats", {}),
            "context_dump_blocked": legal_debug.get("context_dump_blocked", False),
            "best_effort_reason": legal_debug.get("best_effort_reason"),
            "response_mode": legal_debug.get("response_mode"),
            "not_found_reason": legal_debug.get("not_found_reason"),
            "query_signals": legal_debug.get("query_signals", {}),
        }

    def _resolve_route(
        self,
        route_initial: str,
        legal_eval: dict[str, Any],
        technical_eval: dict[str, Any],
    ) -> str:
        if route_initial == "legal":
            if legal_eval["evidence_ok"] and technical_eval["evidence_ok"]:
                return "both"
            if legal_eval["evidence_ok"]:
                return "legal"
            if technical_eval["evidence_ok"]:
                return "technical"
            return "legal"

        if route_initial == "technical":
            if technical_eval["evidence_ok"] and legal_eval["evidence_ok"]:
                return "both"
            if technical_eval["evidence_ok"]:
                return "technical"
            if legal_eval["evidence_ok"]:
                return "legal"
            return "technical"

        # both
        if legal_eval["evidence_ok"] and technical_eval["evidence_ok"]:
            return "both"
        if legal_eval["evidence_ok"]:
            return "legal"
        if technical_eval["evidence_ok"]:
            return "technical"
        return "both"

    def query(
        self,
        question: str,
        top_k: int = 8,
        doc_slugs: list[str] | None = None,
        chat_history: list[dict] | None = None,
        request_id: str | None = None,
    ) -> HybridQueryResult:
        debug = self.query_debug(
            question=question,
            top_k=top_k,
            doc_slugs=doc_slugs,
            chat_history=chat_history,
            debug_probe_mode="auto",
            include_raw_text=False,
            request_id=request_id,
        )

        output_chunks = []
        for item in debug.get("_merged_chunks", []):
            output_chunks.append(item)

        result = HybridQueryResult(
            answer=debug["answer"],
            sources=debug.get("sources", []),
            chunks=output_chunks,
            route=debug["route_final"],
            route_reason=debug["route_reason"],
            route_confidence=debug["route_confidence"],
        )
        return result

    def query_debug(
        self,
        question: str,
        top_k: int = 8,
        doc_slugs: list[str] | None = None,
        chat_history: list[dict] | None = None,
        debug_probe_mode: str = "auto",
        include_raw_text: bool = False,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        effective_question = self._expand_followup_question(question=question, chat_history=chat_history)
        route_debug = self.router.route_with_debug(question=effective_question, chat_history=chat_history)
        final_decision = route_debug["final_decision"]
        route_initial = final_decision.route

        force_legal = debug_probe_mode == "force_legal"
        force_technical = debug_probe_mode == "force_technical"
        force_both = debug_probe_mode == "force_both"

        legal_chunks: list[GraphChunk] = []
        technical_chunks: list[Chunk] = []
        bridge_states: list[dict[str, Any]] = []
        legal_probe_used = False
        technical_probe_used = False

        need_legal = force_legal or force_both or route_initial in {"legal", "both"}
        need_technical = force_technical or force_both or route_initial in {"technical", "both"}

        if need_legal:
            legal_chunks = self.legal.search_units(question=effective_question, top_k=top_k)
        if need_technical:
            technical_chunks = self.technical.retrieve(question=effective_question, top_k=top_k, doc_slugs=doc_slugs)

        legal_eval = self._evaluate_legal_evidence(effective_question, legal_chunks)
        technical_eval = self._evaluate_technical_evidence(effective_question, technical_chunks)

        # Probe fallback only when requested route has weak evidence.
        if route_initial == "legal" and (not legal_eval["evidence_ok"]) and not force_legal:
            technical_probe_used = True
            technical_chunks = self.technical.light_probe(question=effective_question, top_k=min(top_k, ORCH_PROBE_TOP_K), doc_slugs=doc_slugs)
            technical_eval = self._evaluate_technical_evidence(effective_question, technical_chunks)
        elif route_initial == "technical" and (not technical_eval["evidence_ok"]) and not force_technical:
            legal_probe_used = True
            legal_chunks = self.legal.light_probe(question=effective_question, top_k=min(top_k, ORCH_PROBE_TOP_K))
            legal_eval = self._evaluate_legal_evidence(effective_question, legal_chunks)

        if legal_chunks:
            unit_ids = [c.id for c in legal_chunks if c.id]
            bridge_states = self.legal.find_bridge_states(unit_ids)
            if bridge_states and technical_chunks:
                enriched_question = self._enrich_question_with_states(question=question, bridge_states=bridge_states)
                technical_chunks = self.technical.retrieve(
                    question=enriched_question,
                    top_k=top_k,
                    doc_slugs=doc_slugs,
                )
                technical_eval = self._evaluate_technical_evidence(effective_question, technical_chunks)

        route_final = self._resolve_route(
            route_initial,
            legal_eval=legal_eval,
            technical_eval=technical_eval,
        )
        route_reason = final_decision.reason
        route_confidence = float(final_decision.confidence)

        answer, synthesis_trace, legal_chunks_used, technical_chunks_used = self._build_answer(
            route_final=route_final,
            question=effective_question,
            legal_chunks=legal_chunks,
            technical_chunks=technical_chunks,
            chat_history=chat_history,
        )

        merged_chunks, merged_sources = self._build_output_chunks(
            route_final=route_final,
            legal_chunks=legal_chunks_used,
            technical_chunks=technical_chunks_used,
        )
        sources = sorted(set(merged_sources))

        success = not normalize_for_matching(answer).startswith("no encontrado en el documento")
        self.metrics.mark_query(route=route_final, success=success)

        debug_payload = {
            "answer": answer,
            "route_initial": route_initial,
            "route_final": route_final,
            "route_reason": route_reason,
            "route_confidence": route_confidence,
            "engines_used": [
                engine
                for engine, enabled in {
                    "legal": bool(legal_chunks),
                    "technical": bool(technical_chunks),
                }.items()
                if enabled
            ],
            "bridge_fallback_used": bool(bridge_states),
            "router_trace": route_debug,
            "legal_trace": self._build_legal_trace(
                chunks=legal_chunks,
                evaluation=legal_eval,
                include_raw_text=include_raw_text,
                probe_used=legal_probe_used,
                bridge_states=bridge_states,
            ),
            "technical_trace": self._build_technical_trace(
                chunks=technical_chunks,
                evaluation=technical_eval,
                include_raw_text=include_raw_text,
                probe_used=technical_probe_used,
            ),
            "synthesis_trace": synthesis_trace,
            "request_id": request_id,
            "sources": sources,
            "_merged_chunks": merged_chunks,
        }
        return debug_payload

    def get_stats(self) -> dict[str, Any]:
        return {
            "technical": self.technical.get_stats(),
            "legal_graph": self.legal.get_stats(),
            "routing": {
                "enabled": True,
                "strategy": "router + evidence arbitration",
            },
            "observability": self.metrics.to_dict(),
            "bridge_docs_hint": is_bridge_legal_document("dummy.pdf"),
        }
