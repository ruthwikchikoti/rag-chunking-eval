"""LLM-as-judge metrics: chunk relevance (-> Precision@3), faithfulness, answer relevance.

All judges run at temperature 0 in JSON mode and degrade conservatively: a parse
failure yields the *pessimistic* score so the pipeline never silently inflates a metric.
"""
from __future__ import annotations

from .llm import OllamaClient

_JUDGE_SYSTEM = (
    "You are a strict, literal evaluation judge for a retrieval-augmented QA system. "
    "Base every decision ONLY on the text provided. Respond with JSON only."
)


def chunk_relevance(client: OllamaClient, question: str, chunk_text: str) -> dict:
    """Binary relevance of a retrieved chunk to the query (the basis for Precision@3)."""
    prompt = (
        f"Question:\n{question}\n\n"
        f"Retrieved passage:\n\"\"\"{chunk_text}\"\"\"\n\n"
        "Does this passage contain information that helps answer the question? "
        'Reply JSON: {"relevant": true|false, "reason": "<short>"}'
    )
    try:
        out = client.generate_json(prompt, system=_JUDGE_SYSTEM)
        return {"relevant": bool(out.get("relevant", False)), "reason": out.get("reason", "")}
    except ValueError:
        return {"relevant": False, "reason": "judge-parse-failure"}


def faithfulness(client: OllamaClient, context: str, answer: str) -> dict:
    """Fraction of the answer's claims that are supported by the retrieved context (0-1)."""
    if not answer.strip() or answer.strip().lower().startswith("i don't know"):
        # An honest abstention is trivially faithful (it makes no unsupported claims).
        return {"score": 1.0, "supported": 0, "total": 0, "reason": "abstained"}
    prompt = (
        f"Context (the only allowed source of truth):\n\"\"\"{context}\"\"\"\n\n"
        f"Answer to verify:\n\"\"\"{answer}\"\"\"\n\n"
        "Break the answer into its individual factual claims. Count how many are directly "
        "supported by the context. Reply JSON: "
        '{"total_claims": <int>, "supported_claims": <int>, "reason": "<short>"}'
    )
    try:
        out = client.generate_json(prompt, system=_JUDGE_SYSTEM)
        total = max(int(out.get("total_claims", 0)), 0)
        supported = max(min(int(out.get("supported_claims", 0)), total), 0)
        score = 1.0 if total == 0 else supported / total
        return {"score": score, "supported": supported, "total": total, "reason": out.get("reason", "")}
    except (ValueError, TypeError):
        return {"score": 0.0, "supported": 0, "total": 0, "reason": "judge-parse-failure"}


def answer_relevance(client: OllamaClient, question: str, answer: str) -> dict:
    """How well the answer addresses the question, independent of factual support (0-1)."""
    if not answer.strip():
        return {"score": 0.0, "reason": "empty"}
    prompt = (
        f"Question:\n{question}\n\n"
        f"Answer:\n\"\"\"{answer}\"\"\"\n\n"
        "Rate from 0 to 10 how directly and completely the answer addresses the question "
        "(ignore whether it is factually correct). An honest 'I don't know' when the answer "
        "is genuinely unavailable should score around 3. Reply JSON: "
        '{"rating": <0-10>, "reason": "<short>"}'
    )
    try:
        out = client.generate_json(prompt, system=_JUDGE_SYSTEM)
        rating = max(0.0, min(10.0, float(out.get("rating", 0))))
        return {"score": rating / 10.0, "reason": out.get("reason", "")}
    except (ValueError, TypeError):
        return {"score": 0.0, "reason": "judge-parse-failure"}
