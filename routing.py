"""Routing and document classification for hybrid RAG."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import OPENAI_API_KEY, ROUTER_MODEL, ROUTER_TEMPERATURE, ROUTER_USE_LLM

LEGAL_DOC_TOKENS = {
    "ley",
    "reglamento",
    "real",
    "decreto",
    "directiva",
    "circular",
    "resolucion",
    "acuerdo",
    "boe",
    "cnmv",
    "doue",
}

TECH_DOC_TOKENS = {
    "manual",
    "cumplimentacion",
    "reporte",
    "reporting",
    "tecnico",
    "tecnica",
    "estado",
    "xbrl",
    "xml",
    "lqb",
    "bg",
}

@dataclass
class DocumentClassification:
    domain: str
    confidence: float
    reason: str


@dataclass
class RouteDecision:
    route: str
    confidence: float
    reason: str
    used_llm: bool
    fell_back_to_both: bool = False


def normalize_for_matching(text: str) -> str:
    value = str(text or "").lower()
    value = value.replace("¿", " ").replace("¡", " ")
    value = "".join(ch for ch in unicodedata.normalize("NFKD", value) if not unicodedata.combining(ch))
    value = re.sub(r"[^a-z0-9/%\s_\-]+", " ", value)
    value = re.sub(r"[_\-]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]{2,}", normalize_for_matching(text))


def _score_tokens(tokens: list[str], vocabulary: set[str]) -> int:
    if not tokens:
        return 0
    return sum(1 for t in tokens if t in vocabulary)


def classify_document_name(filename: str) -> DocumentClassification:
    name_norm = normalize_for_matching(Path(filename).stem)
    tokens = _tokenize(name_norm)
    legal_score = _score_tokens(tokens, LEGAL_DOC_TOKENS)
    tech_score = _score_tokens(tokens, TECH_DOC_TOKENS)

    if re.search(r"\b(ley|circular|acuerdo|resolucion|reglamento|real decreto|boe)\b", name_norm):
        legal_score += 2
    if re.search(r"\b(manual|cumplimentacion|estado|xbrl|xml|plantilla|formulario)\b", name_norm):
        tech_score += 2

    if legal_score == 0 and tech_score == 0:
        return DocumentClassification(domain="unknown", confidence=0.2, reason="Nombre sin senales claras")
    if legal_score > tech_score:
        return DocumentClassification(domain="legal", confidence=0.9, reason="Nombre con senales legales")
    if tech_score > legal_score:
        return DocumentClassification(domain="technical", confidence=0.9, reason="Nombre con senales tecnicas")
    return DocumentClassification(domain="mixed", confidence=0.7, reason="Nombre mixto legal/tecnico")


def is_bridge_legal_document(filename: str) -> bool:
    name_norm = normalize_for_matching(Path(filename).stem)
    tokens = set(_tokenize(name_norm))
    has_legal = bool(tokens & LEGAL_DOC_TOKENS)
    has_tech = bool(tokens & TECH_DOC_TOKENS)
    if re.search(r"\bmanual\b", name_norm) and re.search(r"\b(circular|ley|boe|cnmv)\b", name_norm):
        return True
    return has_legal and has_tech


class QueryRouter:
    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        if not (ROUTER_USE_LLM and OPENAI_API_KEY):
            return None
        try:
            from openai import OpenAI

            self._client = OpenAI(api_key=OPENAI_API_KEY)
            return self._client
        except Exception:
            return None

    def _llm_decision(self, question: str) -> RouteDecision | None:
        client = self._get_client()
        if client is None:
            return None

        system = (
            "Clasifica preguntas para enrutado RAG. Responde SOLO JSON valido con claves: "
            "route (legal|technical|both), confidence (0..1), reason."
        )
        user = (
            "Pregunta: " + question + "\n"
            "Reglas estrictas: "
            "elige legal por defecto para preguntas sobre ley/articulos/CNMV/ECR/EICC/SGEIC/SGIIC/SCR/FCR, "
            "definiciones, autorizacion, comercializacion, inversiones, requisitos, sanciones o regimen juridico. "
            "Elige technical SOLO si la pregunta trata de cumplimentacion operativa de estados/celdas/campos/codigos/XBRL/XML/manual. "
            "Elige both solo si pide expresamente mezcla legal y tecnica. "
            "Si hay duda entre legal y technical, elige legal."
        )

        try:
            rsp = client.chat.completions.create(
                model=ROUTER_MODEL,
                temperature=float(ROUTER_TEMPERATURE),
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
            )
            raw = rsp.choices[0].message.content or "{}"
            data = json.loads(raw)
            route = str(data.get("route", "both")).strip().lower()
            if route not in {"legal", "technical", "both"}:
                route = "both"
            confidence = float(data.get("confidence", 0.6))
            reason = str(data.get("reason", "LLM arbitration"))[:220]
            return RouteDecision(
                route=route,
                confidence=max(0.0, min(confidence, 1.0)),
                reason=reason,
                used_llm=True,
            )
        except Exception:
            return None

    def route(self, question: str, chat_history: list[dict] | None = None) -> RouteDecision:
        debug = self.route_with_debug(question=question, chat_history=chat_history)
        return debug["final_decision"]

    def route_with_debug(self, question: str, chat_history: list[dict] | None = None) -> dict[str, Any]:
        llm = self._llm_decision(question)
        if llm is None:
            raise RuntimeError("LLM router unavailable: enable OPENAI_API_KEY and RAG_ROUTER_USE_LLM=true")
        final = llm

        return {
            "original_question": question,
            "normalized_question": normalize_for_matching(question),
            "heuristic_scores": {},
            "heuristic_decision": None,
            "llm_decision": llm,
            "final_decision": final,
        }
