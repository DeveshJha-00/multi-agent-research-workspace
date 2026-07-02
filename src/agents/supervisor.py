"""Supervisor planning agent."""

from langchain_core.prompts import ChatPromptTemplate

from src.core.config import settings
from src.llms.openai import get_llm
from src.models.agent import ResearchPlan

PHASE_ONE_AGENTS = {"document_investigator", "web_researcher"}


async def create_research_plan(objective: str, available_data: list[str]) -> ResearchPlan:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You supervise specialist agents. Break complex objectives into independent, bounded "
                "assignments that can run in parallel. Available agents in this phase are "
                "document_investigator for uploaded documents and web_researcher for external/current "
                "information. Do not assign unavailable agents. Avoid duplicate work.",
            ),
            (
                "human",
                "Objective: {objective}\nAvailable workspace data: {available_data}\n"
                "Create at most {max_workers} initial tasks.",
            ),
        ]
    )
    planner = get_llm().with_structured_output(ResearchPlan)
    plan = await (prompt | planner).ainvoke(
        {
            "objective": objective,
            "available_data": ", ".join(available_data) or "No declared uploads",
            "max_workers": settings.supervisor_max_workers,
        }
    )
    plan.tasks = [task for task in plan.tasks if task.agent in PHASE_ONE_AGENTS][
        : settings.supervisor_max_workers
    ]
    if not plan.tasks:
        from src.models.agent import AgentTask

        plan.tasks = [
            AgentTask(
                agent="web_researcher",
                instruction=f"Research reliable external evidence for: {objective}",
                rationale="Fallback assignment because no valid specialist task was planned.",
            )
        ]
    return plan
