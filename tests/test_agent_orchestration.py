import pytest
from pydantic import ValidationError

from src.agents import critic as critic_module
from src.agents import web_researcher as web_module
from src.agents.base import AgentContext
from src.agents.registry import AGENTS
from src.agents.supervisor import _fallback_plan
from src.models.agent import AgentResult, AgentTask, Critique
from src.orchestration.research_graph import after_critique, dispatch_workers


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
async def test_critic_falls_back_when_structured_output_is_invalid(monkeypatch):
    async def fake_evidence(task_id, limit):
        return [{"evidence_id": "ev-1", "agent": "data_analyst", "content": "totals"}]

    class FailingStructuredModel:
        async def ainvoke(self, value, config=None, **kwargs):
            raise ValueError("invalid structured output")

    monkeypatch.setattr(critic_module, "get_evidence", fake_evidence)
    monkeypatch.setattr(
        critic_module,
        "get_structured_llm",
        lambda schema: FailingStructuredModel(),
    )
    critique = await critic_module.evidence_critic.review(
        task_id="task-123",
        session_id="session-123",
        objective="Analyze sales",
        results=[
            AgentResult(
                agent="data_analyst",
                instruction="Analyze",
                summary="Totals calculated",
            )
        ],
        allow_follow_ups=True,
    )
    assert critique.approved is True
    assert critique.coverage_score == 0.85


@pytest.mark.asyncio
async def test_successful_data_analysis_does_not_request_duplicate_revision(monkeypatch):
    async def fake_evidence(task_id, limit):
        return [{"evidence_id": "ev-1", "agent": "data_analyst", "content": "totals"}]

    class FakeChain:
        async def ainvoke(self, value, config=None, **kwargs):
            return Critique(
                approved=False,
                coverage_score=0.6,
                problems=["Request more analysis"],
                follow_up_tasks=[
                    {
                        "agent": "data_analyst",
                        "instruction": "Analyze again",
                        "rationale": "More detail",
                    },
                    {
                        "agent": "data_analyst",
                        "instruction": "Create another chart",
                        "rationale": "More charts",
                    },
                ],
            )

    class FakePrompt:
        def __or__(self, other):
            return FakeChain()

    monkeypatch.setattr(critic_module, "get_evidence", fake_evidence)
    monkeypatch.setattr(critic_module, "get_structured_llm", lambda schema: object())
    monkeypatch.setattr(critic_module.ChatPromptTemplate, "from_messages", lambda messages: FakePrompt())
    critique = await critic_module.evidence_critic.review(
        task_id="task-123",
        session_id="session-123",
        objective="Analyze sales",
        results=[
            AgentResult(
                agent="data_analyst",
                instruction="Analyze",
                summary="Complete",
                evidence_ids=["ev-1"],
            )
        ],
        allow_follow_ups=True,
    )
    assert critique.approved is True
    assert critique.coverage_score == 0.9
    assert critique.follow_up_tasks == []


@pytest.mark.asyncio
async def test_successful_repository_analysis_uses_deterministic_audit(monkeypatch):
    async def fake_evidence(task_id, limit):
        return [{"evidence_id": "ev-1", "agent": "repository_analyst", "content": "paths"}]

    monkeypatch.setattr(critic_module, "get_evidence", fake_evidence)
    monkeypatch.setattr(
        critic_module,
        "get_structured_llm",
        lambda schema: (_ for _ in ()).throw(AssertionError("LLM must not be called")),
    )
    critique = await critic_module.evidence_critic.review(
        task_id="task-123",
        session_id="session-123",
        objective="Analyze repository architecture",
        results=[
            AgentResult(
                agent="repository_analyst",
                instruction="Inspect repository",
                summary="Static findings",
                evidence_ids=["ev-1"],
            )
        ],
        allow_follow_ups=True,
    )
    assert critique.approved is True
    assert critique.coverage_score == 0.9


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
