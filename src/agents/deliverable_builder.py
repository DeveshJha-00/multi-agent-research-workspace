"""Agent that synthesizes audited evidence into downloadable deliverables."""

from langchain_core.tools import tool

from src.agents.base import AgentContext, ToolCallingAgent
from src.db.artifact_store import get_artifact, list_artifacts, save_artifact
from src.db.evidence_store import get_evidence
from src.models.agent import AgentResult, ArtifactRecord


class DeliverableBuilderAgent(ToolCallingAgent):
    name = "deliverable_builder"
    system_prompt = (
        "Build a polished Markdown research report from audited evidence. Include an executive summary, "
        "findings, limitations, and a source list. Cite evidence IDs inline. Save the report with your tool."
    )

    def build_tools(self, context: AgentContext):
        @tool
        async def read_evidence(limit: int = 100) -> list[dict]:
            """Read the shared evidence ledger before writing the deliverable."""
            return await get_evidence(context.task_id, limit=max(1, min(limit, 100)))

        @tool
        async def save_markdown_report(title: str, markdown: str) -> dict:
            """Save the completed report as a downloadable Markdown artifact."""
            safe_name = "".join(char for char in title if char.isalnum() or char in " -_").strip()
            artifact_id = await save_artifact(
                task_id=context.task_id,
                session_id=context.session_id,
                name=f"{safe_name or 'research-report'}.md",
                media_type="text/markdown",
                content=markdown.encode("utf-8")[:1_000_000],
            )
            context.artifact_ids.append(artifact_id)
            return {"artifact_id": artifact_id, "name": f"{safe_name}.md"}

        return [read_evidence, save_markdown_report]

    async def build(self, context: AgentContext) -> tuple[AgentResult, list[ArtifactRecord]]:
        result = await self.run(context)
        if not context.artifact_ids:
            artifact_id = await save_artifact(
                task_id=context.task_id,
                session_id=context.session_id,
                name="research-report.md",
                media_type="text/markdown",
                content=result.summary.encode("utf-8"),
            )
            context.artifact_ids.append(artifact_id)
        primary = await get_artifact(context.artifact_ids[0], context.session_id)
        if primary is not None:
            result.summary = bytes(primary["content"]).decode("utf-8", errors="replace")
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
