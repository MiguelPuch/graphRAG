"""Sincroniza PDFs de normativa con Markdown e ingesta en RAG."""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

from rag_engine import RAGEngine

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        logger.warning("Valor invalido para %s=%r. Usando %d", name, value, default)
        return default

    if parsed < 1:
        logger.warning("Valor invalido para %s=%r. Usando %d", name, value, default)
        return default
    return parsed


MD_ROOT = Path(os.getenv("RAG_NORMATIVA_MD_ROOT", "/app/data/normativa"))
PDF_DIR = Path(os.getenv("RAG_NORMATIVA_PDF_DIR", "/app/data/documents"))
POR_PROCESAR = Path(os.getenv("RAG_NORMATIVA_POR_PROCESAR", str(MD_ROOT / "por_procesar")))
PROCESADOS = Path(os.getenv("RAG_NORMATIVA_PROCESADOS", str(MD_ROOT / "procesados")))
POLL_SECONDS = _env_int("RAG_NORMATIVA_POLL_SECONDS", 30)
SYNC_ENABLED = _env_bool("RAG_NORMATIVA_SYNC", True)


def _ensure_dirs() -> None:
    POR_PROCESAR.mkdir(parents=True, exist_ok=True)
    PROCESADOS.mkdir(parents=True, exist_ok=True)


def sync_once(engine: RAGEngine) -> int:
    """Procesa PDFs nuevos de normativa y devuelve cuántos se ingirieron."""
    _ensure_dirs()
    if not PDF_DIR.exists():
        logger.warning("Carpeta de PDFs no existe: %s", PDF_DIR)
        return 0

    processed = 0
    pdfs = sorted(p for p in PDF_DIR.rglob("*.pdf") if p.is_file())
    for pdf in pdfs:
        slug = engine.converter._create_slug(pdf.name)
        processed_md = PROCESADOS / f"{slug}.md"
        pending_md = POR_PROCESAR / f"{slug}.md"

        if processed_md.exists():
            continue

        if not pending_md.exists():
            try:
                engine.converter.convert_to_markdown(pdf, pdf.name, output_dir=POR_PROCESAR)
            except Exception as e:
                logger.exception("Error convirtiendo %s: %s", pdf.name, e)
                continue

        # Ingestar desde Markdown pendiente
        try:
            result = engine.ingest_markdown_file(pending_md, pdf.name, doc_slug=slug)
            if result.get("status") == "success":
                target = PROCESADOS / pending_md.name
                target.parent.mkdir(parents=True, exist_ok=True)
                pending_md.replace(target)
                processed += 1
            else:
                logger.warning("Ingesta vacía para %s (%s)", pdf.name, result.get("status"))
        except Exception as e:
            logger.exception("Error ingestando %s: %s", pdf.name, e)

    return processed


def _loop(engine: RAGEngine) -> None:
    while True:
        try:
            count = sync_once(engine)
            if count:
                logger.info("Ingestados %d PDFs de normativa", count)
        except Exception as e:
            logger.exception("Error en sincronización de normativa: %s", e)
        time.sleep(POLL_SECONDS)


def start_normativa_sync(engine: RAGEngine) -> None:
    if not SYNC_ENABLED:
        logger.info("Sincronización de normativa deshabilitada")
        return

    thread = threading.Thread(target=_loop, args=(engine,), daemon=True)
    thread.start()
    logger.info("Sincronización de normativa iniciada (intervalo=%ss)", POLL_SECONDS)
