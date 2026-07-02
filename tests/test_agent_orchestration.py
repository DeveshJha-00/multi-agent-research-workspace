import pytest
from pydantic import ValidationError

from src.agents.registry import AGENTS
from src.models.agent import AgentTask, Critique
from src.orchestration.research_graph import after_critique, dispatch_workers


def test_phase_one_registry_contains_distinct_specialists():
    assert set(AGENTS) == {"document_investigator", "web_researcher"}
    assert AGENTS["document_investigator"].build_tools is not AGENTS["web_researcher"].build_tools


def test_agent_task_rejects_unknown_specialist():
    with pytest.raises(ValidationError):
        AgentTask(agent="generic_agent", instruction="Do some generic work", rationale="test")


def test_supervisor_dispatches_one_worker_per_planned_task():
    state = {
        "task_id": "task-123",
        "session_id": "session-123",
        "objective": "Compare uploaded policy with current external guidance",
        "plan": [
            AgentTask(
                agent="document_investigator",
                instruction="Extract policy requirements",
                rationale="Internal evidence",
            ),
            AgentTask(
                agent="web_researcher",
                instruction="Find current external guidance",
                rationale="External evidence",
            ),
        ],
        "worker_results": [],
    }
    sends = dispatch_workers(state)
    assert len(sends) == 2
    assert {send.arg["task"].agent for send in sends} == {
        "document_investigator",
        "web_researcher",
    }


def test_critic_requests_revision_when_followups_exist():
    state = {
        "critique": Critique(
            approved=False,
            coverage_score=0.5,
            problems=["Missing primary source"],
            follow_up_tasks=[
                {
                    "agent": "web_researcher",
                    "instruction": "Find the missing primary source",
                    "rationale": "Close evidence gap",
                }
            ],
        )
    }
    assert after_critique(state) == "prepare_revision"


def test_critic_routes_approved_work_to_deliverable():
    state = {
        "critique": Critique(
            approved=True,
            coverage_score=0.95,
            problems=[],
            follow_up_tasks=[],
        )
    }
    assert after_critique(state) == "build_deliverable"
