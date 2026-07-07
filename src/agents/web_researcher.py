"""Agent specialized in current web research and source provenance."""

import json

from langchain_core.tools import tool
from tavily import AsyncTavilyClient

from src.agents.base import AgentContext, ToolCallingAgent
from src.core.config import settings
from src.core.idempotency import operation_key
from src.db.evidence_store import add_evidence
from src.models.agent import AgentResult


class WebResearcherAgent(ToolCallingAgent):
    name = "web_researcher"
    system_prompt = (
        "Research current external information. Use focused searches, prefer primary and authoritative "
        "sources, compare dates, and preserve URLs. Do not treat a search snippet as stronger than it is."
    )

    def build_tools(self, context: AgentContext):
        @tool
        async def search_web(query: str, max_results: int = 5) -> list[dict]:
            """Search the web and add result snippets with URLs to the shared evidence ledger."""
            query = " ".join(query.split())[:400]
            client = AsyncTavilyClient(api_key=settings.tavily_api_key)
            response = await client.search(
                query,
                max_results=max(1, min(max_results, 10)),
                include_answer=False,
                search_depth="basic",
            )
            output = []
            for item in response.get("results", []):
                content = str(item.get("content", ""))
                if not content:
                    continue
                evidence_id = await add_evidence(
                    task_id=context.task_id,
                    session_id=context.session_id,
                    agent=self.name,
                    content=content,
                    source=str(item.get("title") or item.get("url") or "Web result"),
                    url=item.get("url"),
                    confidence=float(item.get("score", 0.7)),
                    metadata={"query": query},
                    operation_key=operation_key("web", query, item.get("url"), content),
                )
                context.evidence_ids.append(evidence_id)
                output.append(
                    {
                        "evidence_id": evidence_id,
                        "title": item.get("title"),
                        "url": item.get("url"),
                        "content": content,
                        "score": item.get("score"),
                    }
                )
            return output

        return [search_web]

    async def run(self, context: AgentContext) -> AgentResult:
        """Collect web evidence directly to conserve Groq calls and tokens."""
        try:
            tool = self.build_tools(context)[0]
            findings = await tool.ainvoke({"query": context.instruction, "max_results": 4})
            summary = json.dumps(findings, default=str)[:5000]
            return AgentResult(
                agent=self.name,
                instruction=context.instruction,
                summary=summary or "No usable web-search results were found.",
                evidence_ids=context.evidence_ids,
                tool_calls=1,
            )
        except Exception as exc:
            return AgentResult(
                agent=self.name,
                instruction=context.instruction,
                summary="Web evidence collection failed.",
                evidence_ids=context.evidence_ids,
                tool_calls=1,
                error=str(exc),
            )


web_researcher = WebResearcherAgent()
