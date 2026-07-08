"""Explicit specialist registry and dispatch."""

from src.agents.base import AgentContext
from src.agents.data_analyst import data_analyst
from src.agents.document_investigator import document_investigator
from src.agents.repository_analyst import repository_analyst
from src.agents.web_researcher import web_researcher
from src.models.agent import AgentResult, AgentTask

AGENTS = {
    "document_investigator": document_investigator,
    "web_researcher": web_researcher,
    "data_analyst": data_analyst,
    "repository_analyst": repository_analyst,
}


async def run_specialist(
    *,
    task_id: str,
    session_id: str,
    objective: str,
    task: AgentTask,
    prior_results: list[AgentResult],
) -> AgentResult:
    agent = AGENTS.get(task.agent)
    if agent is None:
        return AgentResult(
            agent=task.agent,
            instruction=task.instruction,
            summary="The requested specialist is not available.",
            error="agent_unavailable",
        )
    return await agent.run(
        AgentContext(
            task_id=task_id,
            session_id=session_id,
            objective=objective,
            instruction=task.instruction,
            prior_results=prior_results,
        )
    )
