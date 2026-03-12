"""Routing-only helpers for deciding legal vs technical questions."""

from __future__ import annotations

from dataclasses import dataclass

from routing import QueryRouter


@dataclass
class RouteResult:
    route: str
    confidence: float
    reason: str
    used_llm: bool
    fell_back_to_both: bool


def classify_question(question: str, chat_history: list[dict] | None = None) -> RouteResult:
    """Classify a question into legal/technical/both using shared router."""
    decision = QueryRouter().route(question=question, chat_history=chat_history)
    return RouteResult(
        route=decision.route,
        confidence=decision.confidence,
        reason=decision.reason,
        used_llm=decision.used_llm,
        fell_back_to_both=decision.fell_back_to_both,
    )


def is_legal_route(question: str, chat_history: list[dict] | None = None) -> bool:
    """Return True if route is legal or both."""
    route = classify_question(question=question, chat_history=chat_history).route
    return route in {"legal", "both"}

