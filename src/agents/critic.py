"""Evidence-auditing agent that can request targeted follow-up work."""

import logging

from langchain_core.prompts import ChatPromptTemplate

from src.core.config import settings
from src.core.prompt_budget import compact_evidence, compact_results
from src.db.evidence_store import get_evidence
from src.llms.provider import get_structured_llm
from src.models.agent import AgentResult, Critique

logger = logging.getLogger(__name__)


class EvidenceCriticAgent:
    """Audit bounded evidence in one structured call to conserve free-tier tokens."""

    name = "evidence_critic"

    async def review(
        self,
        *,
        task_id: str,
        session_id: str,
        objective: str,
        results: list[AgentResult],
        allow_follow_ups: bool,
    ) -> Critique:
        del session_id  # Evidence is already isolated by the task created for this request.
        evidence = await get_evidence(task_id, limit=30)
        deterministic_success = bool(results) and all(
            item.agent in {"data_analyst", "repository_analyst"}
            and not item.error
            and item.evidence_ids
            for item in results
        )
        if deterministic_success and evidence:
            return Critique(
                approved=True,
                coverage_score=0.9,
                problems=[],
                follow_up_tasks=[],
            )
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Return a strict structured audit. Approve only when the objective is substantially "
                    "answered by traceable evidence. Follow-up agents may only be "
                    "document_investigator, web_researcher, data_analyst, or repository_analyst. "
                    "Do not request follow-ups when disabled.",
                ),
                (
                    "human",
                    "Objective:\n{objective}\n\nSpecialist results:\n{results}\n\nEvidence:\n{evidence}\n\n"
                    "Follow-ups allowed: {allow_follow_ups}",
                ),
            ]
        )
        structured = get_structured_llm(Critique)
        try:
            critique = await (prompt | structured).ainvoke(
                {
                    "objective": objective,
                    "results": compact_results(results, settings.critic_results_chars),
                    "evidence": compact_evidence(evidence, settings.critic_evidence_chars),
                    "allow_follow_ups": allow_follow_ups,
                }
            )
        except Exception as exc:
            logger.warning(
                "critic_structured_output_failed error=%s; using deterministic audit",
                type(exc).__name__,
            )
            failed_agents = [item.agent for item in results if item.error]
            has_evidence = bool(evidence)
            approved = has_evidence and not failed_agents
            problems = []
            if not has_evidence:
                problems.append("No traceable evidence was collected.")
            if failed_agents:
                problems.append(f"Specialist failures: {', '.join(sorted(set(failed_agents)))}")
            critique = Critique(
                approved=approved,
                coverage_score=0.85 if approved else (0.4 if has_evidence else 0.0),
                problems=problems,
                follow_up_tasks=[],
            )
        if not allow_follow_ups:
            critique.follow_up_tasks = []
        unique_follow_ups = []
        assigned_agents = set()
        for item in critique.follow_up_tasks:
            if (
                item.agent
                not in {
                    "document_investigator",
                    "web_researcher",
                    "data_analyst",
                    "repository_analyst",
                }
                or item.agent in assigned_agents
            ):
                continue
            unique_follow_ups.append(item)
            assigned_agents.add(item.agent)
            if len(unique_follow_ups) >= settings.supervisor_max_workers:
                break
        critique.follow_up_tasks = unique_follow_ups
        return critique


evidence_critic = EvidenceCriticAgent()
