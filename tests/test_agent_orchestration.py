import pytest
from pydantic import ValidationError

from src.agents import web_researcher as web_module
from src.agents.base import AgentContext
from src.agents.registry import AGENTS
from src.agents.supervisor import _fallback_plan
from src.models.agent import AgentTask
from src.orchestration.research_graph import dispatch_workers


def test_phase_one_registry_contains_distinct_specialists():
    assert set(AGENTS) == {
        "document_investigator",
        "web_researcher",
        "data_analyst",
        "repository_analyst",
    }
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


def test_supervisor_fallback_assigns_data_agent_for_dataset():
    plan = _fallback_plan(
        "Analyze sales.csv and create a chart",
        ["Dataset ID dataset-123: sales.csv"],
    )
    assert [task.agent for task in plan.tasks] == ["data_analyst"]


def test_supervisor_fallback_assigns_repository_agent_for_source_code():
    plan = _fallback_plan(
        "Analyze this codebase architecture and entry points",
        ["Repository ID repository-123: project.zip"],
    )
    assert [task.agent for task in plan.tasks] == ["repository_analyst"]


@pytest.mark.asyncio
async def test_web_agent_bounds_tavily_query(monkeypatch):
    captured = {}

    class FakeTavilyClient:
        def __init__(self, api_key):
            pass

        async def search(self, query, **kwargs):
            captured["query"] = query
            return {"results": []}

    monkeypatch.setattr(web_module, "AsyncTavilyClient", FakeTavilyClient)
    result = await web_module.web_researcher.run(
        AgentContext(
            task_id="task-123",
            session_id="session-123",
            objective="Research current guidance",
            instruction="guidance " * 100,
        )
    )
    assert len(captured["query"]) <= 400
    assert result.error is None
