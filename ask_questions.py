"""Run questions from a text file against the RAG API and export answers to JSON."""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
import unicodedata
from pathlib import Path
from typing import Any


MOJIBAKE_MARKERS = ("Ã", "Â", "â", "ï¿½", "�")


def _mojibake_score(text: str) -> int:
    if not text:
        return 0
    score = 0
    for marker in MOJIBAKE_MARKERS:
        score += text.count(marker)
    score += text.count("??")
    return score


def _repair_mojibake(text: str) -> str:
    if not text:
        return text
    repaired = text
    for _ in range(4):
        if _mojibake_score(repaired) == 0:
            break
        candidates = [repaired]
        for source_encoding in ("latin1", "cp1252"):
            try:
                candidates.append(repaired.encode(source_encoding, errors="ignore").decode("utf-8", errors="ignore"))
            except Exception:
                continue
        best = min((c for c in candidates if c), key=_mojibake_score, default=repaired)
        if (not best) or best == repaired:
            break
        repaired = best
    return repaired


def _normalize_question_line(line: str, repair_mojibake: bool = True) -> str:
    text = line.strip()
    if not text:
        return ""
    if repair_mojibake:
        text = _repair_mojibake(text)
    text = text.replace("Â¿", "¿").replace("Â¡", "¡").replace("Ã¿", "¿")
    if text.startswith("?") and not text.startswith("¿"):
        text = f"¿{text[1:].strip()}"
    text = re.sub(r"^\s*(?:[-*]\s+|\d+[).:-]\s+|pregunta\s+\d+\s*:\s*)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_questions(path: Path, repair_mojibake: bool = True) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Questions file not found: {path}")

    raw = path.read_bytes().decode("utf-8", errors="ignore")
    if repair_mojibake:
        raw = _repair_mojibake(raw)

    lines = [ln.strip() for ln in raw.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    if not lines:
        return []

    cleaned: list[str] = []
    for q in lines:
        qq = _normalize_question_line(q, repair_mojibake=repair_mojibake)
        qq = qq.replace("Â¿Â¿", "¿").strip()
        qq = re.sub(r"\s+", " ", qq).strip()
        if len(qq) < 6:
            continue
        cleaned.append(qq)
    return cleaned


def post_json(base_url: str, endpoint: str, payload: dict[str, Any], timeout: int = 180) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _is_anaphoric_question(question: str) -> bool:
    q = re.sub(r"\s+", " ", (question or "").strip().lower())
    q = "".join(ch for ch in unicodedata.normalize("NFKD", q) if not unicodedata.combining(ch))
    if len(q) <= 40:
        if re.search(r"^(y|entonces|en que condiciones|que condiciones|como|y eso|y cual|cual de)", q):
            return True
    return bool(re.search(r"\b(eso|ello|ellas|ellos|en ese caso|en este caso|en que condiciones)\b", q))


def run_questions(
    base_url: str,
    endpoint: str,
    questions: list[str],
    top_k: int,
    timeout: int,
    carry_history: bool = True,
    history_turns: int = 4,
    history_mode: str = "anaphoric",
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    started_at = int(time.time())
    history: list[dict[str, str]] = []

    for idx, question in enumerate(questions, start=1):
        payload = {"question": question, "top_k": top_k}
        use_history = carry_history and bool(history)
        if use_history and history_mode == "anaphoric":
            use_history = _is_anaphoric_question(question)
        if use_history:
            payload["chat_history"] = history[-max(1, history_turns * 2) :]
        t0 = time.time()
        try:
            response = post_json(base_url=base_url, endpoint=endpoint, payload=payload, timeout=timeout)
            elapsed_ms = int((time.time() - t0) * 1000)
            row: dict[str, Any] = {
                "num": idx,
                "question": question,
                "elapsed_ms": elapsed_ms,
                "status": "ok",
            }
            if endpoint.strip("/").lower() == "query-debug":
                row.update(
                    {
                        "answer": response.get("answer"),
                        "sources": response.get("sources", []),
                        "route_initial": response.get("route_initial"),
                        "route_final": response.get("route_final"),
                        "route_reason": response.get("route_reason"),
                        "route_confidence": response.get("route_confidence"),
                        "engines_used": response.get("engines_used", []),
                    }
                )
            else:
                row.update(
                    {
                        "answer": response.get("answer"),
                        "sources": response.get("sources", []),
                        "route": response.get("route"),
                        "route_reason": response.get("route_reason"),
                        "route_confidence": response.get("route_confidence"),
                    }
                )
            rows.append(row)
            if carry_history:
                answer_text = str(response.get("answer") or "")
                history.append({"role": "user", "content": question})
                if answer_text:
                    history.append({"role": "assistant", "content": answer_text[:2000]})
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="ignore")
            rows.append(
                {
                    "num": idx,
                    "question": question,
                    "status": "error",
                    "error_type": "http_error",
                    "http_status": exc.code,
                    "detail": details,
                }
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "num": idx,
                    "question": question,
                    "status": "error",
                    "error_type": "exception",
                    "detail": str(exc),
                }
            )

    finished_at = int(time.time())
    return {
        "base_url": base_url,
        "endpoint": endpoint,
        "top_k": top_k,
        "carry_history": bool(carry_history),
        "history_turns": int(history_turns),
        "history_mode": history_mode,
        "count": len(questions),
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_seconds": max(finished_at - started_at, 0),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="questions.txt", help="Text file with one question per line")
    parser.add_argument("--output", default="data/questions_answers.json", help="JSON output file")
    parser.add_argument("--base-url", default="http://localhost:8001", help="API base URL")
    parser.add_argument("--endpoint", default="query", choices=["query", "query-debug"], help="API endpoint")
    parser.add_argument("--top-k", type=int, default=8, help="top_k sent to API")
    parser.add_argument("--timeout", type=int, default=180, help="HTTP timeout per question (seconds)")
    parser.set_defaults(carry_history=True)
    parser.add_argument(
        "--carry-history",
        dest="carry_history",
        action="store_true",
        help="Send prior Q/A turns as chat_history to support follow-up questions",
    )
    parser.add_argument(
        "--no-carry-history",
        dest="carry_history",
        action="store_false",
        help="Disable chat_history carry between questions",
    )
    parser.add_argument("--history-turns", type=int, default=4, help="Number of previous turns to include when carrying history")
    parser.add_argument(
        "--history-mode",
        choices=["anaphoric", "all"],
        default="anaphoric",
        help="When carrying history, use it only for anaphoric questions or for all questions",
    )
    parser.add_argument(
        "--no-repair-mojibake",
        action="store_true",
        help="Disable lightweight mojibake repair for question lines",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    questions = load_questions(input_path, repair_mojibake=not args.no_repair_mojibake)
    report = run_questions(
        base_url=args.base_url,
        endpoint=args.endpoint,
        questions=questions,
        top_k=args.top_k,
        timeout=args.timeout,
        carry_history=bool(args.carry_history),
        history_turns=max(1, int(args.history_turns)),
        history_mode=str(args.history_mode),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Questions processed: {len(questions)}")
    print(f"Output written to: {output_path}")


if __name__ == "__main__":
    main()
