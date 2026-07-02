"""Supervisor-worker research graph with bounded critique and revision."""

from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.types import Send

from src.agents.base import AgentContext
from src.agents.critic import evidence_critic
from src.agents.deliverable_builder import deliverable_builder
from src.agents.registry import run_specialist
from src.agents.supervisor import create_research_plan
from src.core.config import settings
from src.models.agent import AgentTask, OrchestrationState, WorkerState


async def plan_work(state: OrchestrationState) -> dict:
    plan = await create_research_plan(state["objective"], state.get("available_data", []))
    return {"plan": plan.tasks, "worker_results": [], "revision_count": 0}


def dispatch_workers(state: OrchestrationState):
    return [
        Send(
            "run_worker",
            {
                "task_id": state["task_id"],
                "session_id": state["session_id"],
                "objective": state["objective"],
                "task": task,
                "worker_results": state.get("worker_results", []),
            },
        )
        for task in state.get("plan", [])
    ]


async def run_worker(state: WorkerState) -> dict:
    result = await run_specialist(
        task_id=state["task_id"],
        session_id=state["session_id"],
        objective=state["objective"],
        task=state["task"],
        prior_results=state.get("worker_results", []),
    )
    return {"worker_results": [result]}


async def critique_work(state: OrchestrationState) -> dict:
    allow_follow_ups = state.get("revision_count", 0) < settings.agent_max_revisions
    critique = await evidence_critic.review(
        task_id=state["task_id"],
        session_id=state["session_id"],
        objective=state["objective"],
        results=state.get("worker_results", []),
        allow_follow_ups=allow_follow_ups,
    )
    return {"critique": critique}


def after_critique(state: OrchestrationState) -> str:
    if state["critique"].follow_up_tasks and not state["critique"].approved:
        return "prepare_revision"
    return "build_deliverable"


async def prepare_revision(state: OrchestrationState) -> dict:
    tasks = [
        AgentTask(agent=item.agent, instruction=item.instruction, rationale=item.rationale)
        for item in state["critique"].follow_up_tasks
    ]
    return {"plan": tasks, "revision_count": state.get("revision_count", 0) + 1}


async def build_deliverable(state: OrchestrationState) -> dict:
    result, artifacts = await deliverable_builder.build(
        AgentContext(
            task_id=state["task_id"],
            session_id=state["session_id"],
            objective=state["objective"],
            instruction=(
                "Create the final report. Incorporate the specialist findings and the critic's audit."
            ),
            prior_results=state.get("worker_results", []),
        )
    )
    return {"final_answer": result.summary, "artifacts": artifacts}


graph = StateGraph(OrchestrationState)
graph.add_node("plan", plan_work)
graph.add_node("run_worker", run_worker)
graph.add_node("critique", critique_work)
graph.add_node("prepare_revision", prepare_revision)
graph.add_node("build_deliverable", build_deliverable)
graph.add_edge(START, "plan")
graph.add_conditional_edges("plan", dispatch_workers, ["run_worker"])
graph.add_edge("run_worker", "critique")
graph.add_conditional_edges(
    "critique",
    after_critique,
    {"prepare_revision": "prepare_revision", "build_deliverable": "build_deliverable"},
)
graph.add_conditional_edges("prepare_revision", dispatch_workers, ["run_worker"])
graph.add_edge("build_deliverable", END)

research_orchestrator = graph.compile()
