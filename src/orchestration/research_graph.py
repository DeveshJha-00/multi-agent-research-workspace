"""Supervisor-worker research graph for durable multi-agent reports."""

from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.types import Send

from src.agents.base import AgentContext
from src.agents.deliverable_builder import deliverable_builder
from src.agents.registry import run_specialist
from src.agents.supervisor import create_research_plan
from src.db.checkpoint_store import checkpoint_saver
from src.db.research_job_store import append_event, ensure_job_not_cancelled
from src.models.agent import OrchestrationState, WorkerState


async def plan_work(state: OrchestrationState) -> dict:
    await ensure_job_not_cancelled(state["task_id"])
    await append_event(
        state["task_id"],
        event="planning_started",
        stage="planning",
        progress=5,
        message="Supervisor is planning the research",
    )
    plan = await create_research_plan(state["objective"], state.get("available_data", []))
    await append_event(
        state["task_id"],
        event="planning_completed",
        stage="delegating",
        progress=15,
        message=f"Supervisor delegated {len(plan.tasks)} specialist task(s)",
        details={"agents": [task.agent for task in plan.tasks]},
    )
    return {"plan": plan.tasks, "worker_results": []}


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
    await ensure_job_not_cancelled(state["task_id"])
    agent_name = state["task"].agent
    await append_event(
        state["task_id"],
        event="specialist_started",
        stage="specialists",
        progress=25,
        message=f"{agent_name.replace('_', ' ').title()} started",
        details={"agent": agent_name, "instruction": state["task"].instruction[:500]},
    )
    result = await run_specialist(
        task_id=state["task_id"],
        session_id=state["session_id"],
        objective=state["objective"],
        task=state["task"],
        prior_results=state.get("worker_results", []),
    )
    await append_event(
        state["task_id"],
        event="specialist_completed",
        stage="specialists",
        progress=60,
        message=f"{agent_name.replace('_', ' ').title()} completed",
        details={
            "agent": agent_name,
            "evidence_items": len(result.evidence_ids),
            "error": result.error,
        },
    )
    return {"worker_results": [result]}


async def build_deliverable(state: OrchestrationState) -> dict:
    await ensure_job_not_cancelled(state["task_id"])
    await append_event(
        state["task_id"],
        event="report_started",
        stage="reporting",
        progress=90,
        message="Deliverable builder is creating the final report",
    )
    result, artifacts = await deliverable_builder.build(
        AgentContext(
            task_id=state["task_id"],
            session_id=state["session_id"],
            objective=state["objective"],
            instruction="Create the final report from the specialist findings and evidence.",
            prior_results=state.get("worker_results", []),
        )
    )
    await append_event(
        state["task_id"],
        event="report_completed",
        stage="reporting",
        progress=98,
        message="Final report and artifacts are ready",
        details={"artifacts": len(artifacts)},
    )
    return {"final_answer": result.summary, "artifacts": artifacts}


graph = StateGraph(OrchestrationState)
graph.add_node("plan", plan_work)
graph.add_node("run_worker", run_worker)
graph.add_node("build_deliverable", build_deliverable)
graph.add_edge(START, "plan")
graph.add_conditional_edges("plan", dispatch_workers, ["run_worker"])
graph.add_edge("run_worker", "build_deliverable")
graph.add_edge("build_deliverable", END)

research_orchestrator = graph.compile(checkpointer=checkpoint_saver)
