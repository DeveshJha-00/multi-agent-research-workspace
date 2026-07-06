"""Deterministic context compaction for free-tier model token budgets."""

import json
from collections import defaultdict, deque
from collections.abc import Iterable

from src.models.agent import AgentResult


def _round_robin_by_agent(records: Iterable[dict]) -> list[dict]:
    groups: dict[str, deque[dict]] = defaultdict(deque)
    for record in records:
        groups[str(record.get("agent") or "unknown")].append(record)
    ordered: list[dict] = []
    while groups:
        for agent in list(groups):
            ordered.append(groups[agent].popleft())
            if not groups[agent]:
                del groups[agent]
    return ordered


def compact_evidence(records: list[dict], max_chars: int) -> str:
    """Keep evidence from every agent instead of truncating one large JSON prefix."""
    compacted: list[dict] = []
    for record in _round_robin_by_agent(records):
        item = {
            "evidence_id": record.get("evidence_id"),
            "agent": record.get("agent"),
            "content": str(record.get("content", ""))[:600],
            "source": record.get("source"),
            "url": record.get("url"),
            "document_id": record.get("document_id"),
            "page": record.get("page"),
            "confidence": record.get("confidence"),
        }
        candidate = json.dumps([*compacted, item], default=str)
        if len(candidate) > max_chars:
            break
        compacted.append(item)
    return json.dumps(compacted, default=str)


def compact_results(results: list[AgentResult], max_chars: int) -> str:
    """Retain a bounded summary from each specialist."""
    compacted = [
        {
            "agent": item.agent,
            "instruction": item.instruction[:300],
            "summary": item.summary[:900],
            "evidence_ids": item.evidence_ids[:12],
            "error": item.error,
        }
        for item in results
    ]
    return json.dumps(compacted, default=str)[:max_chars]
