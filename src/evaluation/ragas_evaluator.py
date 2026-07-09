"""Text-only RAGAS metrics backed by Groq and local FastEmbed embeddings."""

import json
from functools import lru_cache
from math import sqrt
from typing import Any

from openai import AsyncOpenAI
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    ContextUtilization,
    FactualCorrectness,
    Faithfulness,
    SemanticSimilarity,
)

from src.core.config import settings
from src.llms.provider import get_embeddings

REFERENCE_FREE_GROUNDED = ["answer_relevancy", "faithfulness", "context_utilization"]
REFERENCE_FREE_GENERAL = ["answer_relevancy"]
REFERENCE_ADDITIONS = ["factual_correctness", "semantic_similarity"]
REFERENCE_CONTEXT_ADDITIONS = ["context_precision", "context_recall"]


def metric_names_for(route: str, *, has_contexts: bool, has_reference: bool) -> list[str]:
    names = list(
        REFERENCE_FREE_GROUNDED
        if route in {"index", "search"} and has_contexts
        else REFERENCE_FREE_GENERAL
    )
    if has_reference:
        names.extend(REFERENCE_ADDITIONS)
        if has_contexts:
            names.extend(REFERENCE_CONTEXT_ADDITIONS)
    return names


class LocalFastEmbedRagasEmbedding(BaseRagasEmbedding):
    """Expose the application's ONNX embedding model through RAGAS's text interface."""

    def embed_text(self, text: str, **kwargs: Any) -> list[float]:
        return get_embeddings().embed_query(text)

    async def aembed_text(self, text: str, **kwargs: Any) -> list[float]:
        return await get_embeddings().aembed_query(text)

    def embed_texts(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        return get_embeddings().embed_documents(texts)

    async def aembed_texts(
        self, texts: list[str], **kwargs: Any
    ) -> list[list[float]]:
        return await get_embeddings().aembed_documents(texts)


@lru_cache
def get_ragas_llm():
    client = AsyncOpenAI(
        api_key=settings.groq_api_key,
        base_url=settings.ragas_judge_base_url,
        timeout=90.0,
        max_retries=0,
    )
    return llm_factory(
        settings.effective_ragas_judge_model,
        provider="openai",
        client=client,
        temperature=0.0,
        top_p=1.0,
        max_tokens=settings.groq_max_output_tokens,
        system_prompt=(
            "You are a strict JSON-only evaluator. Return exactly the JSON object "
            "requested by the schema. Do not include markdown, XML tags, prose, "
            "or explanations outside the JSON object."
        ),
    )


@lru_cache
def get_ragas_embeddings() -> LocalFastEmbedRagasEmbedding:
    return LocalFastEmbedRagasEmbedding()


@lru_cache
def get_metrics() -> dict[str, Any]:
    llm = get_ragas_llm()
    embeddings = get_ragas_embeddings()
    return {
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings, strictness=1),
        "faithfulness": Faithfulness(llm=llm),
        "context_utilization": ContextUtilization(llm=llm),
        "factual_correctness": FactualCorrectness(llm=llm, mode="f1"),
        "semantic_similarity": SemanticSimilarity(embeddings=embeddings),
        "context_precision": ContextPrecision(llm=llm),
        "context_recall": ContextRecall(llm=llm),
    }


def _plain_contexts(snapshot: dict) -> list[str]:
    """Return bounded strings only; RAGAS receives no path or URL loading instructions."""
    output = []
    used = 0
    for item in snapshot.get("contexts", [])[: settings.ragas_max_contexts]:
        remaining = settings.ragas_max_context_chars - used
        if remaining <= 0:
            break
        text = str(item.get("content", "")).strip()[:remaining]
        if text:
            output.append(text)
            used += len(text)
    return output


async def score_metric(
    metric_name: str, snapshot: dict, reference: str | None
) -> dict[str, Any]:
    metric = get_metrics()[metric_name]
    common = {"user_input": snapshot["question"], "response": snapshot["answer"]}
    contexts = _plain_contexts(snapshot)
    try:
        if metric_name == "answer_relevancy":
            result = await metric.ascore(**common)
        elif metric_name in {"faithfulness", "context_utilization"}:
            result = await metric.ascore(**common, retrieved_contexts=contexts)
        elif metric_name in {"factual_correctness", "semantic_similarity"}:
            result = await metric.ascore(response=snapshot["answer"], reference=reference)
        elif metric_name == "context_precision":
            result = await metric.ascore(
                user_input=snapshot["question"],
                reference=reference,
                retrieved_contexts=contexts,
            )
        elif metric_name == "context_recall":
            result = await metric.ascore(
                user_input=snapshot["question"],
                reference=reference,
                retrieved_contexts=contexts,
            )
        else:
            raise ValueError(f"Unsupported evaluation metric: {metric_name}")
    except Exception as exc:
        if _is_structured_json_failure(exc):
            recovered = await _try_direct_json_judge_recovery(
                metric_name, snapshot, reference, contexts, exc
            )
            if recovered is not None:
                return recovered
            return await _local_structured_output_fallback(metric_name, snapshot, reference, contexts)
        raise
    value = float(result.value)
    reason = getattr(result, "reason", None)
    return {"score": value, "reason": str(reason) if reason else None}


def _is_structured_json_failure(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "json_validate_failed" in message
        or "failed to validate json" in message
        or "instructorretryexception" in exc.__class__.__name__.lower()
    )


@lru_cache
def get_direct_json_judge_client() -> AsyncOpenAI:
    """Return a plain OpenAI-compatible Groq client for schema-recovery judge calls."""
    return AsyncOpenAI(
        api_key=settings.groq_api_key,
        base_url=settings.ragas_judge_base_url,
        timeout=45.0,
        max_retries=0,
    )


def _json_recovery_supported(metric_name: str) -> bool:
    return metric_name in {"faithfulness", "context_utilization"}


def _truncate_for_judge(text: str, limit: int = 8000) -> str:
    stripped = str(text or "").strip()
    if len(stripped) <= limit:
        return stripped
    return stripped[:limit] + "\n...[truncated]"


def _json_recovery_prompt(
    metric_name: str,
    snapshot: dict,
    contexts: list[str],
) -> str:
    joined_contexts = "\n\n--- CONTEXT BREAK ---\n\n".join(
        _truncate_for_judge(context, 3000) for context in contexts
    )
    question = _truncate_for_judge(snapshot["question"], 1200)
    answer = _truncate_for_judge(snapshot["answer"], 3500)
    if metric_name == "faithfulness":
        rubric = (
            "Evaluate faithfulness. Score 1.0 when every factual claim in the answer "
            "is directly supported by the retrieved contexts. Penalize unsupported, "
            "contradicted, or invented details. If the answer is mostly supported but "
            "has minor unsupported wording, use a score between 0.6 and 0.9."
        )
    else:
        rubric = (
            "Evaluate context utilization. Score 1.0 when the retrieved contexts are "
            "clearly useful and sufficient for producing the answer. Penalize contexts "
            "that are irrelevant, unused, or weakly connected to the answer."
        )
    return f"""
{rubric}

Return one valid JSON object only, with exactly these keys:
{{"score": <number from 0 to 1>, "reason": "<one concise sentence>"}}

Do not include markdown, XML tags, code fences, or extra keys.

Question:
{question}

Answer:
{answer}

Retrieved contexts:
{joined_contexts}
""".strip()


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Judge JSON response must be an object")
    return parsed


def _coerce_score(value: Any) -> float:
    score = float(value)
    if score != score:
        raise ValueError("Judge score cannot be NaN")
    return max(0.0, min(1.0, score))


async def _try_direct_json_judge_recovery(
    metric_name: str,
    snapshot: dict,
    reference: str | None,
    contexts: list[str],
    original_error: Exception,
) -> dict[str, Any] | None:
    """Recover RAGAS schema failures with a simpler judge-backed JSON contract.

    RAGAS 0.4.x uses Instructor/Pydantic schemas internally. Some Groq-hosted
    models occasionally miss those richer schemas for multi-step metrics even
    when the same model can return a simple JSON object. This keeps the metric
    judge-backed before falling all the way back to local embedding similarity.
    """
    del reference
    if not _json_recovery_supported(metric_name) or not contexts:
        return None
    try:
        response = await get_direct_json_judge_client().chat.completions.create(
            model=settings.effective_ragas_judge_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict evaluation judge. You only return valid JSON "
                        "that matches the requested shape."
                    ),
                },
                {
                    "role": "user",
                    "content": _json_recovery_prompt(metric_name, snapshot, contexts),
                },
            ],
            temperature=0,
            max_tokens=min(settings.groq_max_output_tokens, 700),
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or ""
        parsed = _extract_json_object(content)
        score = _coerce_score(parsed["score"])
        reason = str(parsed.get("reason") or "Recovered with a direct JSON judge call.")
        return {
            "score": score,
            "reason": (
                f"{reason} RAGAS structured-output validation failed for this metric, "
                "so the project recovered with a simpler Groq JSON judge prompt instead "
                "of using the local embedding fallback."
            ),
        }
    except Exception:
        return None


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=False))
    left_norm = sqrt(sum(a * a for a in left))
    right_norm = sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return max(0.0, min(1.0, dot / (left_norm * right_norm)))


async def _max_similarity(text: str, candidates: list[str]) -> float:
    if not text.strip() or not candidates:
        return 0.0
    embeddings = get_embeddings()
    query_vector = await embeddings.aembed_query(text)
    candidate_vectors = await embeddings.aembed_documents(candidates)
    return max((_cosine(query_vector, vector) for vector in candidate_vectors), default=0.0)


async def _local_structured_output_fallback(
    metric_name: str,
    snapshot: dict,
    reference: str | None,
    contexts: list[str],
) -> dict[str, Any]:
    """Fallback when a judge model fails RAGAS structured JSON validation."""
    if metric_name in {"faithfulness", "context_utilization", "context_precision", "context_recall"}:
        score = await _max_similarity(snapshot["answer"], contexts)
    elif metric_name == "factual_correctness" and reference:
        score = await _max_similarity(snapshot["answer"], [reference])
    else:
        score = await _max_similarity(snapshot["question"], [snapshot["answer"]])
    return {
        "score": score,
        "reason": (
            "Local FastEmbed semantic-similarity fallback was used because the selected "
            "judge model did not return the structured RAGAS schema for this metric. "
            "Treat this as a weaker diagnostic than a normal judge-backed RAGAS score."
        ),
    }
