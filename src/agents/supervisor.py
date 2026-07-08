"""Supervisor planning agent."""

import logging

from langchain_core.prompts import ChatPromptTemplate

from src.core.config import settings
from src.llms.provider import get_structured_llm
from src.models.agent import AgentTask, ResearchPlan

AVAILABLE_AGENTS = {
    "document_investigator",
    "web_researcher",
    "data_analyst",
    "repository_analyst",
}
logger = logging.getLogger(__name__)


def _fallback_plan(objective: str, available_data: list[str]) -> ResearchPlan:
    """Create a safe plan when the free model returns malformed structured output."""
    declared = " ".join(available_data).lower()
    objective_lower = objective.lower()
    tasks = []
    if "dataset" in declared or any(
        term in objective_lower for term in ("csv", "dataset", "xlsx", "spreadsheet", "sales")
    ):
        tasks.append(
            AgentTask(
                agent="data_analyst",
                instruction=f"Analyze the available workspace dataset for: {objective}",
                rationale="The objective requires calculations from tabular data.",
            )
        )
    if "repository id" in declared or any(
        term in objective_lower
        for term in ("codebase", "repository", "source code", "dependencies", "entry point")
    ):
        tasks.append(
            AgentTask(
                agent="repository_analyst",
                instruction=f"Inspect the available source repository for: {objective}",
                rationale="The objective requires static repository and source-code analysis.",
            )
        )
    if "uploaded document" in declared or "document" in objective_lower or "policy" in objective_lower:
        tasks.append(
            AgentTask(
                agent="document_investigator",
                instruction=f"Find relevant evidence in uploaded documents for: {objective}",
                rationale="The objective refers to workspace documents.",
            )
        )
    if any(
        term in objective_lower
        for term in ("current", "latest", "web", "external", "guidance", "compare")
    ):
        tasks.append(
            AgentTask(
                agent="web_researcher",
                instruction=f"Find authoritative current web evidence for: {objective}",
                rationale="The objective requires current or external evidence.",
            )
        )
    if not tasks:
        tasks.append(
            AgentTask(
                agent="web_researcher",
                instruction=f"Research reliable external evidence for: {objective}",
                rationale="Fallback research assignment.",
            )
        )
    return ResearchPlan(objective=objective, tasks=tasks[: settings.supervisor_max_workers])


async def create_research_plan(objective: str, available_data: list[str]) -> ResearchPlan:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You supervise specialist agents. Break complex objectives into independent, bounded "
                "assignments that can run in parallel. Available agents in this phase are "
                "document_investigator for uploaded documents and web_researcher for external/current "
                "information, data_analyst for uploaded CSV/JSON/Excel datasets, and "
                "repository_analyst for uploaded source repository ZIPs. Do not assign unavailable "
                "agents. Avoid duplicate work and assign specialists only when their data is relevant.",
            ),
            (
                "human",
                "Objective: {objective}\nAvailable workspace data: {available_data}\n"
                "Create at most {max_workers} initial tasks.",
            ),
        ]
    )
    planner = get_structured_llm(ResearchPlan)
    try:
        plan = await (prompt | planner).ainvoke(
            {
                "objective": objective,
                "available_data": ", ".join(available_data) or "No declared uploads",
                "max_workers": settings.supervisor_max_workers,
            }
        )
    except Exception:
        logger.exception("supervisor_structured_output_failed; using deterministic plan")
        plan = _fallback_plan(objective, available_data)
    unique_tasks = []
    assigned_agents = set()
    for task in plan.tasks:
        if task.agent not in AVAILABLE_AGENTS or task.agent in assigned_agents:
            continue
        unique_tasks.append(task)
        assigned_agents.add(task.agent)
        if len(unique_tasks) >= settings.supervisor_max_workers:
            break
    plan.tasks = unique_tasks
    if not plan.tasks:
        plan.tasks = [
            AgentTask(
                agent="web_researcher",
                instruction=f"Research reliable external evidence for: {objective}",
                rationale="Fallback assignment because no valid specialist task was planned.",
            )
        ]
    return plan
