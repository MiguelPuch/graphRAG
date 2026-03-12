"""Shared lexical utilities for legal GraphRAG (generic, low-bias)."""

from __future__ import annotations

import re
import unicodedata

LEGAL_STOPWORDS = {
    "como",
    "donde",
    "cual",
    "cuales",
    "para",
    "sobre",
    "esta",
    "este",
    "estos",
    "estas",
    "que",
    "con",
    "sin",
    "por",
    "segun",
    "ley",
    "real",
    "decreto",
    "del",
    "los",
    "las",
    "una",
    "unos",
    "unas",
    "norma",
    "articulo",
    "entre",
    "hay",
    "significa",
    "quien",
    "cual",
}

ARTICLE_REF_PATTERN = re.compile(r"\barticulo\s+(\d+[a-z]?)\b", flags=re.IGNORECASE)
LEGAL_REF_PATTERN = re.compile(
    r"\b(ley|reglamento|directiva|circular|real decreto(?:\s+ley|-ley)?|orden|resolucion|acuerdo)\s+(\d{1,4}/\d{4})\b",
    flags=re.IGNORECASE,
)
STATE_REF_PATTERN = re.compile(r"\bestado\s+([a-z]{1,4}\d{1,4}(?:-[a-z]+)?)\b", flags=re.IGNORECASE)
STATE_CODE_INLINE_PATTERN = re.compile(r"\b([a-z]{2,6})\s*(\d{1,4})(?:-([a-z]+))?\b", flags=re.IGNORECASE)

LEGAL_ENTITY_TOKENS = {
    "cnmv",
    "banco de espana",
    "esi",
    "sgiic",
    "sgeic",
    "iic",
    "ecr",
    "eicc",
    "scr",
    "fcr",
    "ecr pyme",
}

_HEADING_DUMP_PATTERNS = [
    re.compile(r"^\s*##\s*", flags=re.IGNORECASE),
    re.compile(r"^\s*boletin oficial del estado", flags=re.IGNORECASE),
    re.compile(r"^\s*comision nacional del mercado de valores", flags=re.IGNORECASE),
    re.compile(r"^\s*\|", flags=re.IGNORECASE),
]

MOJIBAKE_MARKERS = ("Ã", "Â", "â", "ï¿½", "�")


def _repair_visible_text(text: str) -> str:
    """Normalize spacing and repair common mojibake artifacts lightly."""
    if not text:
        return ""
    value = str(text).replace("\ufeff", "").replace("\u200b", " ").replace("\xa0", " ")
    # Iterative utf-8/latin1 repair for common double-encoded text.
    for _ in range(2):
        if not any(marker in value for marker in MOJIBAKE_MARKERS):
            break
        try:
            candidate = value.encode("latin1", errors="ignore").decode("utf-8", errors="ignore")
        except Exception:
            break
        if not candidate or candidate == value:
            break
        value = candidate
    # Lightweight mojibake repairs (non-aggressive).
    value = value.replace("Ã¡", "a").replace("Ã©", "e").replace("Ã­", "i").replace("Ã³", "o").replace("Ãº", "u")
    value = value.replace("Ã±", "n").replace("Â", " ")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _normalize_for_search(text: str) -> str:
    """Accent-insensitive, punctuation-light normalizer for lexical matching."""
    value = _repair_visible_text(text or "").lower()
    value = value.replace("¿", " ").replace("¡", " ")
    value = "".join(ch for ch in unicodedata.normalize("NFKD", value) if not unicodedata.combining(ch))
    value = re.sub(r"[^a-z0-9/%\s_\-]+", " ", value)
    value = re.sub(r"[_\-]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    # Common mojibake side-effect in Spanish questions ("qué" -> "qu").
    value = re.sub(r"\bqu\b", "que", value)
    return value


def _normalize_article_number(value: str | None) -> str:
    return re.sub(r"\s+", "", _normalize_for_search(value or ""))


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]{3,}", _normalize_for_search(text))


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _canonical_state_code(value: str) -> str:
    return re.sub(r"\s+", "", _normalize_for_search(value or ""))


def _state_code_terms(code: str) -> list[str]:
    compact = _canonical_state_code(code)
    if not compact:
        return []
    terms = [compact, f"estado {compact}"]
    match = re.match(r"^([a-z]{2,6})(\d{1,4}(?:-[a-z]+)?)$", compact)
    if match:
        spaced = f"{match.group(1)} {match.group(2)}"
        terms.extend([spaced, f"estado {spaced}"])
    return _dedupe(terms)


def _extract_state_codes(text_norm: str) -> list[str]:
    codes: list[str] = []
    for match in STATE_REF_PATTERN.finditer(text_norm):
        code = _canonical_state_code(match.group(1))
        if code:
            codes.append(code)
    for match in STATE_CODE_INLINE_PATTERN.finditer(text_norm):
        prefix = _normalize_for_search(match.group(1))
        number = match.group(2)
        suffix = match.group(3) or ""
        code = f"{prefix}{number}{('-' + suffix) if suffix else ''}"
        c = _canonical_state_code(code)
        if c:
            codes.append(c)
    return _dedupe(codes)


def looks_like_heading_dump(text: str) -> bool:
    cleaned = _normalize_for_search(text)
    return any(p.search(cleaned) for p in _HEADING_DUMP_PATTERNS)
