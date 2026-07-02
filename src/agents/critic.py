"""Evidence-auditing agent that can request targeted follow-up work."""

import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

from src.agents.base import AgentContext, ToolCallingAgent
from src.core.config import settings
from src.db.evidence_store import get_evidence
from src.llms.openai import get_llm
from src.models.agent import AgentResult, Critique


class EvidenceCriticAgent(ToolCallingAgent):
    name = "evidence_critic"
    system_prompt = (
        "Audit whether specialist findings answer the objective. Inspect the evidence ledger, identify "
        "unsupported claims, missing perspectives, contradictions, and weak sources."
    )

    def build_tools(self, context: AgentContext):
        @tool
        async def inspect_evidence(limit: int = 60) -> list[dict]:
            """Read evidence collected by all specialists for this task."""
            return await get_evidence(context.task_id, limit=max(1, min(limit, 100)))

        return [inspect_evidence]

    async def review(
        self,
        *,
        task_id: str,
        session_id: str,
        objective: str,
        results: list[AgentResult],
        allow_follow_ups: bool,
    ) -> Critique:
        context = AgentContext(
            task_id=task_id,
            session_id=session_id,
            objective=objective,
            instruction="Audit the collected work and evidence for completeness and reliability.",
            prior_results=results,
        )
        audit = await self.run(context)
        evidence = await get_evidence(task_id, limit=100)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Return a strict structured audit. Approve only when the objective is substantially "
                    "answered by traceable evidence. Follow-up agents may only be document_investigator "
                    "or web_researcher. Do not request follow-ups when they are disabled.",
                ),
                (
                    "human",
                    "Objective:\n{objective}\n\nSpecialist results:\n{results}\n\nEvidence:\n{evidence}\n\n"
                    "Audit notes:\n{audit}\n\nFollow-ups allowed: {allow_follow_ups}",
                ),
            ]
        )
        structured = get_llm().with_structured_output(Critique)
        critique = await (prompt | structured).ainvoke(
            {
                "objective": objective,
                "results": json.dumps([item.model_dump() for item in results], default=str)[:16000],
                "evidence": json.dumps(evidence, default=str)[:24000],
                "audit": audit.summary,
                "allow_follow_ups": allow_follow_ups,
            }
        )
        if not allow_follow_ups:
            critique.follow_up_tasks = []
        critique.follow_up_tasks = [
            item
            for item in critique.follow_up_tasks
            if item.agent in {"document_investigator", "web_researcher"}
        ][: settings.supervisor_max_workers]
        return critique


evidence_critic = EvidenceCriticAgent()
