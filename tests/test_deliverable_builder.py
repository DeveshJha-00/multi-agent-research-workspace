from types import SimpleNamespace

import pytest
from langchain_core.runnables import RunnableLambda

from src.agents import deliverable_builder as module
from src.agents.base import AgentContext
from src.models.agent import AgentResult


@pytest.mark.asyncio
async def test_empty_model_output_uses_non_empty_fallback(monkeypatch):
    saved = {}
    evidence = [{"evidence_id": "ev-1", "source": "sales.csv", "content": "Revenue: 100"}]

    async def fake_evidence(task_id, limit):
        return evidence

    async def fake_save(**kwargs):
        saved.update(kwargs)
        return "artifact-123"

    async def fake_list(task_id):
        return [
            {
                "artifact_id": "artifact-123",
                "name": "research-report.md",
                "media_type": "text/markdown",
            }
        ]

    monkeypatch.setattr(module, "get_evidence", fake_evidence)
    monkeypatch.setattr(module, "save_artifact", fake_save)
    monkeypatch.setattr(module, "list_artifacts", fake_list)
    monkeypatch.setattr(
        module,
        "get_llm",
        lambda: RunnableLambda(lambda _: SimpleNamespace(content="")),
    )

    result, artifacts = await module.deliverable_builder.build(
        AgentContext(
            task_id="task-123",
            session_id="session-123",
            objective="Analyze sales performance",
            instruction="Build report",
            prior_results=[
                AgentResult(
                    agent="web_researcher",
                    instruction="Research sales",
                    summary="External benchmark revenue is 100.",
                )
            ],
        )
    )

    assert result.summary.startswith("# Research report")
    assert saved["content"]
    assert artifacts[0].artifact_id == "artifact-123"


@pytest.mark.asyncio
async def test_data_only_report_does_not_call_llm(monkeypatch):
    saved = {}

    async def fake_evidence(task_id, limit):
        return [{"evidence_id": "ev-1", "source": "sales.csv", "content": "profit: 40"}]

    async def fake_save(**kwargs):
        saved.update(kwargs)
        return "artifact-123"

    async def fake_list(task_id):
        return [
            {
                "artifact_id": "artifact-123",
                "name": "research-report.md",
                "media_type": "text/markdown",
            }
        ]

    monkeypatch.setattr(module, "get_evidence", fake_evidence)
    monkeypatch.setattr(module, "save_artifact", fake_save)
    monkeypatch.setattr(module, "list_artifacts", fake_list)
    monkeypatch.setattr(
        module,
        "get_llm",
        lambda: (_ for _ in ()).throw(AssertionError("LLM must not be called")),
    )

    result, _ = await module.deliverable_builder.build(
        AgentContext(
            task_id="task-123",
            session_id="session-123",
            objective="Analyze sales",
            instruction="Build report",
            prior_results=[
                AgentResult(
                    agent="data_analyst",
                    instruction="Analyze",
                    summary="Revenue total: 300. Profit total: 120.",
                )
            ],
        )
    )

    assert result.summary.startswith("# Data analysis report")
    assert "Profit total: 120" in result.summary
    assert saved["content"]


@pytest.mark.asyncio
async def test_repository_only_report_does_not_call_llm(monkeypatch):
    saved = {}

    async def fake_evidence(task_id, limit):
        return [{"evidence_id": "ev-1", "source": "project.zip", "content": "main.py"}]

    async def fake_save(**kwargs):
        saved.update(kwargs)
        return "artifact-123"

    async def fake_list(task_id):
        return [
            {
                "artifact_id": "artifact-123",
                "name": "research-report.md",
                "media_type": "text/markdown",
            }
        ]

    monkeypatch.setattr(module, "get_evidence", fake_evidence)
    monkeypatch.setattr(module, "save_artifact", fake_save)
    monkeypatch.setattr(module, "list_artifacts", fake_list)
    monkeypatch.setattr(
        module,
        "get_llm",
        lambda: (_ for _ in ()).throw(AssertionError("LLM must not be called")),
    )

    result, _ = await module.deliverable_builder.build(
        AgentContext(
            task_id="task-123",
            session_id="session-123",
            objective="Analyze repository architecture",
            instruction="Build report",
            prior_results=[
                AgentResult(
                    agent="repository_analyst",
                    instruction="Inspect repository",
                    summary="## Repository overview\n- Entry point: `main.py`",
                    evidence_ids=["ev-1"],
                )
            ],
        )
    )

    assert result.summary.startswith("# Repository analysis report")
    assert "main.py" in result.summary
    assert "was not executed" in result.summary
    assert saved["content"]
