"""Centralized settings for hybrid RAG (technical + legal graph)."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def _resolve_env_name() -> str:
    env_hint = (os.getenv("APP_ENV") or os.getenv("ENV") or os.getenv("ENVIRONMENT") or "").strip().lower()
    if env_hint in {"prod", "production"}:
        return "production"
    return "development"


def _load_env_files() -> None:
    project_root = Path(__file__).resolve().parents[1]
    rag_dir = Path(__file__).resolve().parent
    env_name = _resolve_env_name()

    candidates = [
        project_root / f".env.{env_name}",
        rag_dir / ".env",
    ]

    for env_path in candidates:
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=False)


_load_env_files()


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


# LLM
LLM_PROVIDER = os.getenv("RAG_LLM_PROVIDER", "openai").lower()
LLM_MODEL = os.getenv("RAG_LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = float(os.getenv("RAG_LLM_TEMPERATURE", "0.1"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Router
ROUTER_MODEL = os.getenv("RAG_ROUTER_MODEL", LLM_MODEL)
ROUTER_TEMPERATURE = float(os.getenv("RAG_ROUTER_TEMPERATURE", "0.0"))
ROUTER_USE_LLM = _env_bool("RAG_ROUTER_USE_LLM", True)
ROUTER_LOW_CONFIDENCE_THRESHOLD = _env_float("RAG_ROUTER_LOW_CONFIDENCE_THRESHOLD", 0.65)
ROUTER_SCORE_MARGIN = _env_float("RAG_ROUTER_SCORE_MARGIN", 2.0)

# Embeddings
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "intfloat/multilingual-e5-base")
EMBED_DEVICE = os.getenv("RAG_EMBED_DEVICE", "auto")

# Chunking
CHUNK_MAX_TOKENS = int(os.getenv("RAG_CHUNK_MAX_TOKENS", "530"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "50"))

# Milvus (technical docs)
MILVUS_URI = os.getenv("MILVUS_URI", "http://milvus:19530")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "rag_documents")

# Neo4j (legal graph)
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4jpassword")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# Storage
DATA_DIR = os.getenv("RAG_DATA_DIR", "./data")
DOCS_DIR = os.getenv("RAG_DOCS_DIR", f"{DATA_DIR}/documents")
NORMATIVA_SOURCE_DIR = os.getenv("RAG_NORMATIVA_SOURCE_DIR", "/app/normativa")

# API
API_HOST = os.getenv("RAG_API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("RAG_API_PORT", "8001"))

# Connectivity / startup hardening
STARTUP_MAX_RETRIES = _env_int("RAG_STARTUP_MAX_RETRIES", 20)
STARTUP_RETRY_INTERVAL_SECONDS = _env_float("RAG_STARTUP_RETRY_INTERVAL_SECONDS", 2.0)
MILVUS_CONNECT_TIMEOUT_SECONDS = _env_float("RAG_MILVUS_CONNECT_TIMEOUT_SECONDS", 5.0)
NEO4J_CONNECT_TIMEOUT_SECONDS = _env_float("RAG_NEO4J_CONNECT_TIMEOUT_SECONDS", 5.0)

# Hybrid orchestration evidence thresholds
ORCH_PROBE_TOP_K = _env_int("RAG_ORCH_PROBE_TOP_K", 4)
ORCH_TECH_MIN_SCORE = _env_float("RAG_ORCH_TECH_MIN_SCORE", 0.80)
ORCH_TECH_MIN_CHUNKS = _env_int("RAG_ORCH_TECH_MIN_CHUNKS", 2)
ORCH_LEGAL_MIN_SCORE = _env_float("RAG_ORCH_LEGAL_MIN_SCORE", 0.20)
ORCH_LEGAL_MIN_CHUNKS = _env_int("RAG_ORCH_LEGAL_MIN_CHUNKS", 2)
ORCH_LEGAL_KEYWORD_COVERAGE = _env_float("RAG_ORCH_LEGAL_KEYWORD_COVERAGE", 0.30)

# Legal retrieval / reranker / grounding
LEGAL_OVERFETCH_K = _env_int("RAG_LEGAL_OVERFETCH_K", 32)
LEGAL_RERANKER_ENABLED = _env_bool("RAG_LEGAL_RERANKER_ENABLED", True)
LEGAL_RERANKER_ENDPOINT = os.getenv("RAG_LEGAL_RERANKER_ENDPOINT", "")
LEGAL_RERANKER_MODEL = os.getenv("RAG_LEGAL_RERANKER_MODEL", "bge-reranker-v2-m3")
LEGAL_RERANKER_TIMEOUT_MS = _env_int("RAG_LEGAL_RERANKER_TIMEOUT_MS", 2500)
LEGAL_RERANKER_TOP_N = _env_int("RAG_LEGAL_RERANKER_TOP_N", 10)
LEGAL_GROUNDING_ENABLED = _env_bool("RAG_LEGAL_GROUNDING_ENABLED", True)
LEGAL_GROUNDING_SIM_THRESHOLD = _env_float("RAG_LEGAL_GROUNDING_SIM_THRESHOLD", 0.35)
LEGAL_GROUNDING_LLM_FALLBACK = _env_bool("RAG_LEGAL_GROUNDING_LLM_FALLBACK", True)
LEGAL_QREL_THRESHOLD = _env_float("RAG_LEGAL_QREL_THRESHOLD", 0.26)
LEGAL_COMBINED_THRESHOLD = _env_float("RAG_LEGAL_COMBINED_THRESHOLD", 0.46)
LEGAL_EVIDENCE_MAX_ITEMS = _env_int("RAG_LEGAL_EVIDENCE_MAX_ITEMS", 2)
LEGAL_SNIPPET_MAX_CHARS = _env_int("RAG_LEGAL_SNIPPET_MAX_CHARS", 220)
LEGAL_REQUIRE_NUMERIC_OVERLAP_FOR_NUMERIC_QUERY = _env_bool(
    "RAG_LEGAL_REQUIRE_NUMERIC_OVERLAP_FOR_NUMERIC_QUERY",
    True,
)
LEGAL_INTENT_MODEL = os.getenv("RAG_LEGAL_INTENT_MODEL", LLM_MODEL)

SUPPORTED_INGEST_EXTENSIONS = {".pdf", ".docx", ".pptx", ".xlsx", ".md", ".markdown"}


def get_device() -> str:
    """Resolve model device (cuda/mps/cpu)."""
    if EMBED_DEVICE != "auto":
        return EMBED_DEVICE
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def validate_runtime_config() -> list[str]:
    """Return validation errors for critical runtime settings."""
    errors: list[str] = []
    if not MILVUS_URI:
        errors.append("MILVUS_URI is required")
    if not NEO4J_URI:
        errors.append("NEO4J_URI is required")
    if not NEO4J_USER:
        errors.append("NEO4J_USER is required")
    if not NEO4J_PASSWORD:
        errors.append("NEO4J_PASSWORD is required")
    if STARTUP_MAX_RETRIES < 1:
        errors.append("RAG_STARTUP_MAX_RETRIES must be >= 1")
    if STARTUP_RETRY_INTERVAL_SECONDS <= 0:
        errors.append("RAG_STARTUP_RETRY_INTERVAL_SECONDS must be > 0")
    if not OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY is required for LLM-only routing/intent/generation")
    return errors
