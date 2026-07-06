"""Agent specialized in uploaded-document investigation."""

import json

from langchain_core.tools import tool

from src.agents.base import AgentContext, ToolCallingAgent
from src.db.evidence_store import add_evidence
from src.models.agent import AgentResult
from src.rag.retriever_setup import retrieve_documents


class DocumentInvestigatorAgent(ToolCallingAgent):
    name = "document_investigator"
    system_prompt = (
        "Investigate uploaded documents. Run multiple focused searches when needed, compare passages, "
        "and preserve page/source provenance. State clearly when the documents do not contain an answer."
    )

    def build_tools(self, context: AgentContext):
        @tool
        async def search_uploaded_documents(query: str, limit: int = 8) -> list[dict]:
            """Search the current workspace's uploaded documents for relevant passages."""
            documents = await retrieve_documents(
                query,
                session_id=context.session_id,
                limit=max(1, min(limit, 15)),
            )
            results = []
            for document in documents:
                metadata = document.metadata
                evidence_id = await add_evidence(
                    task_id=context.task_id,
                    session_id=context.session_id,
                    agent=self.name,
                    content=document.page_content,
                    source=str(metadata.get("source", "Uploaded document")),
                    document_id=metadata.get("document_id"),
                    page=metadata.get("page"),
                    confidence=float(metadata.get("vector_score", 0.7)),
                    metadata={"query": query},
                )
                context.evidence_ids.append(evidence_id)
                results.append(
                    {
                        "evidence_id": evidence_id,
                        "content": document.page_content,
                        "source": metadata.get("source"),
                        "page": metadata.get("page"),
                        "score": metadata.get("vector_score"),
                    }
                )
            return results

        return [search_uploaded_documents]

    async def run(self, context: AgentContext) -> AgentResult:
        """Collect document evidence directly to avoid unreliable free-model tool calls."""
        try:
            tool = self.build_tools(context)[0]
            findings = await tool.ainvoke({"query": context.instruction, "limit": 5})
            summary = json.dumps(findings, default=str)[:5000]
            return AgentResult(
                agent=self.name,
                instruction=context.instruction,
                summary=summary or "No relevant uploaded-document passages were found.",
                evidence_ids=context.evidence_ids,
                tool_calls=1,
            )
        except Exception as exc:
            return AgentResult(
                agent=self.name,
                instruction=context.instruction,
                summary="Document evidence collection failed.",
                evidence_ids=context.evidence_ids,
                tool_calls=1,
                error=str(exc),
            )


document_investigator = DocumentInvestigatorAgent()
