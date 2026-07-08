"""Agent that synthesizes audited evidence into downloadable deliverables."""

from langchain_core.prompts import ChatPromptTemplate

from src.agents.base import AgentContext
from src.core.config import settings
from src.core.idempotency import operation_key
from src.core.prompt_budget import compact_evidence, compact_results
from src.db.artifact_store import list_artifacts, save_artifact
from src.db.evidence_store import get_evidence
from src.llms.provider import get_llm
from src.models.agent import AgentResult, ArtifactRecord


class DeliverableBuilderAgent:
    """Build and persist a report in one bounded model call."""

    name = "deliverable_builder"
    system_prompt = (
        "Build a polished Markdown research report from audited evidence. Include an executive summary, "
        "findings, limitations, and a source list. Cite evidence IDs inline."
    )

    @staticmethod
    def _fallback_report(context: AgentContext, evidence: list[dict]) -> str:
        """Build a non-empty report when the model spends its output budget on reasoning."""
        sections = [
            "# Research report",
            "",
            "## Objective",
            context.objective,
            "",
            "## Specialist findings",
        ]
        if context.prior_results:
            for result in context.prior_results:
                sections.extend(
                    [
                        "",
                        f"### {result.agent.replace('_', ' ').title()}",
                        result.summary or "No summary was returned.",
                    ]
                )
        else:
            sections.append("No specialist findings were returned.")
        sections.extend(["", "## Evidence sources"])
        if evidence:
            for item in evidence[:20]:
                source = str(item.get("source") or "Unknown source")
                evidence_id = str(item.get("evidence_id") or "unidentified")
                sections.append(f"- `{evidence_id}` — {source}")
        else:
            sections.append("- No evidence was collected.")
        sections.extend(
            [
                "",
                "## Limitations",
                "This deterministic report was produced because the language model returned no final text.",
            ]
        )
        return "\n".join(sections).strip()

    @staticmethod
    def _data_report(context: AgentContext, evidence: list[dict]) -> str:
        """Preserve computed data-agent output without introducing generative numeric claims."""
        sections = [
            "# Data analysis report",
            "",
            "## Objective",
            context.objective,
            "",
            "## Computed findings",
        ]
        for result in context.prior_results:
            sections.extend(["", result.summary or "No computed summary was returned."])
        sections.extend(["", "## Evidence sources"])
        for item in evidence[:20]:
            source = str(item.get("source") or "Unknown source")
            evidence_id = str(item.get("evidence_id") or "unidentified")
            sections.append(f"- `{evidence_id}` — {source}")
        return "\n".join(sections).strip()

    @staticmethod
    def _repository_report(context: AgentContext, evidence: list[dict]) -> str:
        """Preserve static code findings and their paths without a second model dependency."""
        sections = [
            "# Repository analysis report",
            "",
            "## Objective",
            context.objective,
            "",
            "## Explanation and findings",
        ]
        for result in context.prior_results:
            sections.extend(["", result.summary or "No repository summary was returned."])
        sections.extend(["", "## Evidence sources"])
        for item in evidence[:20]:
            source = str(item.get("source") or "Unknown source")
            evidence_id = str(item.get("evidence_id") or "unidentified")
            sections.append(f"- `{evidence_id}` - {source}")
        sections.extend(
            [
                "",
                "## Execution boundary",
                "Uploaded repository code was inspected as text and was not executed.",
            ]
        )
        return "\n".join(sections).strip()

    async def build(self, context: AgentContext) -> tuple[AgentResult, list[ArtifactRecord]]:
        evidence = await get_evidence(context.task_id, limit=30)
        prior = compact_results(context.prior_results, settings.critic_results_chars)
        evidence_text = compact_evidence(evidence, settings.critic_evidence_chars)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                (
                    "human",
                    "Objective:\n{objective}\n\nSpecialist findings:\n{results}\n\n"
                    "Evidence ledger:\n{evidence}\n\nReturn the report as Markdown.",
                ),
            ]
        )
        data_only = bool(context.prior_results) and all(
            result.agent == "data_analyst" for result in context.prior_results
        )
        repository_only = bool(context.prior_results) and all(
            result.agent == "repository_analyst" for result in context.prior_results
        )
        if data_only:
            markdown = self._data_report(context, evidence)
        elif repository_only:
            markdown = self._repository_report(context, evidence)
        else:
            response = await (prompt | get_llm()).ainvoke(
                {"objective": context.objective, "results": prior, "evidence": evidence_text}
            )
            markdown = str(response.content).strip()
            if not markdown:
                markdown = self._fallback_report(context, evidence)
        artifact_id = await save_artifact(
            task_id=context.task_id,
            session_id=context.session_id,
            name="research-report.md",
            media_type="text/markdown",
            content=markdown.encode("utf-8")[:1_000_000],
            operation_key=operation_key("final_report", context.objective),
        )
        context.artifact_ids.append(artifact_id)
        result = AgentResult(
            agent=self.name,
            instruction=context.instruction,
            summary=markdown,
            evidence_ids=[
                str(item.get("evidence_id")) for item in evidence if item.get("evidence_id")
            ],
            tool_calls=0,
        )
        records = await list_artifacts(context.task_id)
        return result, [
            ArtifactRecord(
                artifact_id=item["artifact_id"],
                name=item["name"],
                media_type=item["media_type"],
            )
            for item in records
        ]


deliverable_builder = DeliverableBuilderAgent()
