"""Hybrid RAG API: technical Milvus RAG + legal Neo4j GraphRAG with routing."""

from __future__ import annotations

import logging
import os
import time
import uuid
from pathlib import Path
from typing import Optional, Any, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

from config import API_HOST, API_PORT, NORMATIVA_SOURCE_DIR, validate_runtime_config
from hybrid_engine import HybridRAGEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ChatMessage(BaseModel):
    role: str = Field(..., description="user o assistant")
    content: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=10, ge=1, le=50)
    doc_slugs: Optional[list[str]] = None
    chat_history: Optional[list[ChatMessage]] = None


class ChunkResponse(BaseModel):
    id: str
    doc_slug: str
    filename: str
    text: str
    score: Optional[float] = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    route: str
    route_reason: str
    route_confidence: float
    chunks: list[ChunkResponse]


class HealthResponse(BaseModel):
    status: str
    details: dict


class RouteDebugRequest(BaseModel):
    question: str = Field(..., min_length=1)
    chat_history: Optional[list[ChatMessage]] = None


class RouteDebugResponse(BaseModel):
    original_question: Optional[str] = None
    normalized_question: str
    hard_rule_applied: Optional[bool] = None
    hard_rule_name: Optional[str] = None
    heuristic_scores: dict[str, Any]
    heuristic_decision: dict[str, Any]
    llm_decision: Optional[dict[str, Any]] = None
    final_decision: dict[str, Any]


class QueryDebugRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=10, ge=1, le=50)
    doc_slugs: Optional[list[str]] = None
    chat_history: Optional[list[ChatMessage]] = None
    debug_probe_mode: Literal["auto", "force_legal", "force_technical", "force_both"] = "auto"
    include_raw_text: bool = False


class QueryDebugResponse(BaseModel):
    answer: str
    route_initial: str
    route_final: str
    route_reason: str
    route_confidence: float
    engines_used: list[str]
    bridge_fallback_used: bool
    router_trace: dict[str, Any]
    legal_trace: dict[str, Any]
    technical_trace: dict[str, Any]
    synthesis_trace: dict[str, Any]
    request_id: Optional[str] = None
    sources: list[str]


def create_app(engine: Optional[HybridRAGEngine] = None) -> FastAPI:
    app = FastAPI(
        title="Hybrid RAG API - Normativa",
        description="Router LLM + Milvus RAG tecnico + Neo4j GraphRAG legal",
        version="2.0.0",
        root_path=os.getenv("FASTAPI_ROOT_PATH", ""),
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    errors = validate_runtime_config()
    for err in errors:
        logger.error(err)
    if errors:
        raise RuntimeError(f"Invalid runtime configuration: {errors}")

    app.state.engine = engine

    def get_engine() -> HybridRAGEngine:
        if app.state.engine is None:
            app.state.engine = HybridRAGEngine()
        return app.state.engine

    @app.middleware("http")
    async def request_context_middleware(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        start = time.time()
        try:
            response: Response = await call_next(request)
        except Exception:
            elapsed_ms = int((time.time() - start) * 1000)
            logger.exception(
                "request_error request_id=%s method=%s path=%s elapsed_ms=%s",
                request_id,
                request.method,
                request.url.path,
                elapsed_ms,
            )
            raise
        elapsed_ms = int((time.time() - start) * 1000)
        content_type = response.headers.get("content-type", "")
        if content_type.startswith("application/json") and "charset=" not in content_type.lower():
            response.headers["content-type"] = f"{content_type}; charset=utf-8"
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "request_done request_id=%s method=%s path=%s status=%s elapsed_ms=%s",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response

    @app.on_event("shutdown")
    def _shutdown() -> None:
        current_engine = app.state.engine
        if current_engine is not None:
            current_engine.close()

    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    def health_check():
        try:
            return HealthResponse(status="healthy", details=get_engine().get_stats())
        except Exception as exc:
            logger.exception("Health check failed")
            raise HTTPException(status_code=503, detail=str(exc))

    @app.post("/query", response_model=QueryResponse, tags=["Query"])
    def query_rag(request: QueryRequest, http_request: Request):
        try:
            history = None
            if request.chat_history:
                history = [{"role": m.role, "content": m.content} for m in request.chat_history]

            result = get_engine().query(
                question=request.question,
                top_k=request.top_k,
                doc_slugs=request.doc_slugs,
                chat_history=history,
                request_id=getattr(http_request.state, "request_id", None),
            )

            chunks = [
                ChunkResponse(
                    id=c.id,
                    doc_slug=c.doc_slug,
                    filename=c.filename,
                    text=c.text,
                    score=c.score,
                )
                for c in result.chunks
            ]

            return QueryResponse(
                answer=result.answer,
                sources=result.sources,
                route=result.route,
                route_reason=result.route_reason,
                route_confidence=result.route_confidence,
                chunks=chunks,
            )
        except Exception as exc:
            logger.exception("Query failed")
            raise HTTPException(status_code=500, detail=f"Error procesando consulta: {exc}")

    @app.post("/route-debug", response_model=RouteDebugResponse, tags=["Query"])
    def route_debug(request: RouteDebugRequest):
        try:
            history = None
            if request.chat_history:
                history = [{"role": m.role, "content": m.content} for m in request.chat_history]

            debug = get_engine().router.route_with_debug(question=request.question, chat_history=history)

            def _serialize(decision):
                if decision is None:
                    return None
                return {
                    "route": decision.route,
                    "confidence": decision.confidence,
                    "reason": decision.reason,
                    "used_llm": decision.used_llm,
                    "fell_back_to_both": decision.fell_back_to_both,
                }

            return RouteDebugResponse(
                original_question=debug.get("original_question"),
                normalized_question=debug["normalized_question"],
                hard_rule_applied=debug.get("hard_rule_applied"),
                hard_rule_name=debug.get("hard_rule_name"),
                heuristic_scores=debug["heuristic_scores"],
                heuristic_decision=_serialize(debug["heuristic_decision"]) or {},
                llm_decision=_serialize(debug["llm_decision"]),
                final_decision=_serialize(debug["final_decision"]) or {},
            )
        except Exception as exc:
            logger.exception("Route debug failed")
            raise HTTPException(status_code=500, detail=f"Error en route-debug: {exc}")

    @app.post("/query-debug", response_model=QueryDebugResponse, tags=["Query"])
    def query_debug(request: QueryDebugRequest, http_request: Request):
        try:
            history = None
            if request.chat_history:
                history = [{"role": m.role, "content": m.content} for m in request.chat_history]

            debug = get_engine().query_debug(
                question=request.question,
                top_k=request.top_k,
                doc_slugs=request.doc_slugs,
                chat_history=history,
                debug_probe_mode=request.debug_probe_mode,
                include_raw_text=request.include_raw_text,
                request_id=getattr(http_request.state, "request_id", None),
            )

            return QueryDebugResponse(
                answer=debug["answer"],
                route_initial=debug["route_initial"],
                route_final=debug["route_final"],
                route_reason=debug["route_reason"],
                route_confidence=debug["route_confidence"],
                engines_used=debug["engines_used"],
                bridge_fallback_used=debug["bridge_fallback_used"],
                router_trace=debug["router_trace"],
                legal_trace=debug["legal_trace"],
                technical_trace=debug["technical_trace"],
                synthesis_trace=debug["synthesis_trace"],
                request_id=debug.get("request_id"),
                sources=debug.get("sources", []),
            )
        except Exception as exc:
            logger.exception("Query debug failed")
            raise HTTPException(status_code=500, detail=f"Error en query-debug: {exc}")

    @app.post("/ingest", tags=["Ingest"])
    async def ingest_document(
        file: UploadFile = File(...),
        target: str = Form(default="auto", description="auto | legal | technical"),
    ):
        if not file.filename:
            raise HTTPException(status_code=400, detail="Se requiere nombre de fichero")

        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Fichero vacio")

        try:
            if target == "legal":
                result = get_engine().ingest_legal_bytes(content, file.filename)
            elif target == "technical":
                result = get_engine().ingest_technical_bytes(content, file.filename)
            else:
                result = get_engine().ingest_auto_bytes(content, file.filename)
            return result
        except Exception as exc:
            logger.exception("Error ingestando %s", file.filename)
            raise HTTPException(status_code=500, detail=f"Error en ingesta: {exc}")

    @app.post("/ingest-corpus", tags=["Ingest"])
    def ingest_corpus(folder: Optional[str] = Form(default=None)):
        try:
            root = folder or NORMATIVA_SOURCE_DIR
            return get_engine().ingest_corpus_folder(root)
        except Exception as exc:
            logger.exception("Error ingestando corpus")
            raise HTTPException(status_code=500, detail=f"Error ingestando corpus: {exc}")

    @app.get("/stats", tags=["Health"])
    def get_stats():
        return get_engine().get_stats()

    @app.get("/list-docs", tags=["Management"])
    def list_docs():
        docs_folder = Path("/app/docs")
        if not docs_folder.exists():
            return {"files": []}

        files = []
        for f in docs_folder.iterdir():
            if f.is_file():
                files.append(
                    {
                        "name": f.name,
                        "size_kb": round(f.stat().st_size / 1024, 2),
                        "extension": f.suffix,
                    }
                )
        return {"files": files}

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=API_HOST, port=API_PORT)
