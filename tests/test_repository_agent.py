from io import BytesIO
from types import SimpleNamespace
from zipfile import ZipFile

import pytest
from langchain_core.runnables import RunnableLambda

from src.agents import repository_analyst as module
from src.agents.base import AgentContext
from src.data.repository_ingestion import _parse_repository


def _repository_zip() -> bytes:
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr("demo/main.py", "from fastapi import FastAPI\napp = FastAPI()\n")
        archive.writestr("demo/tests/test_main.py", "def test_app():\n    assert app\n")
        archive.writestr("demo/requirements.txt", "fastapi==1.0\n")
        archive.writestr("demo/.env.example", "API_KEY=replace-me\n")
        archive.writestr("demo/node_modules/dependency.js", "ignored = true\n")
        archive.writestr("../escape.py", "escaped = True\n")
        archive.writestr("demo/binary.py", b"\x00\x01\x02")
    return buffer.getvalue()


def test_repository_zip_parser_keeps_safe_source_files_only():
    files, languages = _parse_repository(_repository_zip())
    paths = {item["path"] for item in files}
    assert paths == {
        "demo/main.py",
        "demo/tests/test_main.py",
        "demo/requirements.txt",
        "demo/.env.example",
    }
    assert languages == {"Python": 2}


def test_repository_zip_parser_rejects_non_zip_content():
    with pytest.raises(ValueError, match="valid ZIP"):
        _parse_repository(b"not a zip")


def test_repository_inspection_selects_manifests_and_runtime_files():
    files = [
        {"path": "project/README.md", "size": 100, "extension": ".md"},
        {"path": "project/package.json", "size": 100, "extension": ".json"},
        {"path": "project/src/app/page.tsx", "size": 100, "extension": ".tsx"},
        {"path": "project/src/app/api/speech/route.ts", "size": 100, "extension": ".ts"},
        {"path": "project/src/lib/audio.ts", "size": 100, "extension": ".ts"},
        {"path": "project/public/data.json", "size": 100, "extension": ".json"},
    ]
    selected = module._inspection_paths(files, limit=5)
    assert "project/README.md" in selected
    assert "project/package.json" in selected
    assert "project/src/app/page.tsx" in selected
    assert "project/src/app/api/speech/route.ts" in selected
    assert "project/public/data.json" not in selected


def test_repository_fallback_is_an_explanation_not_only_an_inventory():
    explanation = module._fallback_explanation(
        "Explain how the application runs",
        {
            "filename": "project.zip",
            "file_count": 3,
        },
        [
            {"path": "README.md"},
            {"path": "src/main.py"},
            {"path": "src/api/routes.py"},
        ],
        {"FastAPI": ["src/main.py:1"]},
        ["README.md", "src/main.py"],
        {"README.md": "1: # Demo API\n2: A service for language learning."},
    )
    assert "## What this repository is for" in explanation
    assert "## How the application starts and runs" in explanation
    assert "A service for language learning" in explanation
    assert "src/main.py:1" in explanation


def test_repository_synthesis_validator_only_rejects_extreme_failures():
    allowed = {"README.md", "src/main.py", "src/api/routes.py", "src/db/store.py"}
    grounded = "\n".join(
        [
            "The README describes the product as a research workspace [README.md:3-8].",
            "FastAPI creates the application during startup [src/main.py:20-28].",
            "The API module registers request handlers [src/api/routes.py:10-35].",
            "The store module persists application records [src/db/store.py:12-40].",
        ]
    )
    assert module._is_grounded_explanation(grounded, allowed) is True
    assert (
        module._is_grounded_explanation(
            grounded + "\nAn invented service exists [src/missing.py:1].", allowed
        )
        is True
    )
    overwhelmingly_unknown = (
        grounded
        + "\n"
        + "\n".join(f"Unsupported component [missing-{index}.py:1]." for index in range(20))
    )
    assert module._is_grounded_explanation(overwhelmingly_unknown, allowed) is False
    assert (
        module._is_grounded_explanation(
            "A fluent uncited architecture description that remains useful to the reader. " * 5,
            allowed,
        )
        is True
    )
    assert module._is_grounded_explanation("Too short to be useful.", allowed) is False


def test_repository_citation_normalization_resolves_unique_zip_root_paths():
    normalized = module._normalize_citation_paths(
        "Dependencies are declared in (`package.json:2-10`).",
        {"project-main/package.json", "project-main/src/app/page.tsx"},
    )
    assert "[project-main/package.json:2-10]" in normalized


@pytest.mark.asyncio
async def test_repository_synthesis_prefers_usable_model_explanation(monkeypatch):
    model_text = (
        "## Purpose\nThis service gives users a grounded research workflow [README.md:3-8].\n\n"
        "## Runtime flow\nThe API application initializes its dependencies during startup "
        "[src/main.py:27-45].\n\n"
        + "The explanation remains detailed and readable for a human audience. "
        * 8
    )

    class FakeModel:
        def bind(self, **kwargs):
            return RunnableLambda(lambda _: SimpleNamespace(content=model_text))

    monkeypatch.setattr(module, "get_llm", lambda: FakeModel())
    explanation = await module._synthesize_explanation(
        objective="Explain the repository",
        metadata={
            "filename": "project.zip",
            "description": "Demo",
            "file_count": 2,
            "languages": {"Python": 1},
        },
        files=[{"path": "README.md"}, {"path": "src/main.py"}],
        technology_sources={"FastAPI": ["src/main.py:7"]},
        inspected_paths=["README.md", "src/main.py"],
        excerpts={
            "README.md": "3: A grounded research workflow.",
            "src/main.py": "27: async def lifespan(app):",
        },
        matches=[],
    )
    assert explanation == model_text.strip()
    assert "This fallback" not in explanation


@pytest.mark.asyncio
async def test_repository_synthesis_preserves_uncited_model_output_and_adds_sources(monkeypatch):
    model_text = "## Explanation\n" + (
        "This repository has a coherent architecture explained in accessible prose. " * 8
    )

    class FakeModel:
        def bind(self, **kwargs):
            return RunnableLambda(lambda _: SimpleNamespace(content=model_text))

    monkeypatch.setattr(module, "get_llm", lambda: FakeModel())
    explanation = await module._synthesize_explanation(
        objective="Explain the repository",
        metadata={
            "filename": "project.zip",
            "description": "Demo",
            "file_count": 2,
            "languages": {"Python": 1},
        },
        files=[{"path": "README.md"}, {"path": "src/main.py"}],
        technology_sources={"FastAPI": ["src/main.py:7"]},
        inspected_paths=["README.md", "src/main.py"],
        excerpts={
            "README.md": "3: A grounded research workflow.",
            "src/main.py": "27: async def lifespan(app):",
        },
        matches=[],
    )
    assert model_text.strip() in explanation
    assert "## Inspected source references" in explanation
    assert "[README.md:3]" in explanation
    assert "This explanation is bounded by the selected excerpts" not in explanation


@pytest.mark.asyncio
async def test_repository_agent_produces_static_evidence_backed_summary(monkeypatch):
    metadata = {
        "repository_id": "repository-123",
        "filename": "demo.zip",
        "description": "Demo API",
        "file_count": 3,
        "total_bytes": 120,
        "languages": {"Python": 2},
    }
    files = [
        {"path": "demo/main.py", "size": 60, "extension": ".py"},
        {"path": "demo/tests/test_main.py", "size": 40, "extension": ".py"},
        {"path": "demo/requirements.txt", "size": 20, "extension": ".txt"},
    ]

    async def fake_list_repositories(session_id):
        return [metadata]

    async def fake_get_repository(repository_id, session_id):
        return metadata

    async def fake_list_files(repository_id, session_id, **kwargs):
        return files

    async def fake_search(repository_id, session_id, query, **kwargs):
        return [{"path": "demo/main.py", "line": 2, "excerpt": "app = FastAPI()"}]

    async def fake_read(repository_id, session_id, path, **kwargs):
        return {"content": "1: from fastapi import FastAPI", "path": path}

    async def fake_add_evidence(**kwargs):
        assert kwargs["metadata"]["repository_id"] == "repository-123"
        return "evidence-123"

    checkpoint = {}
    events = []

    async def fake_get_checkpoint(task_id, analysis_key):
        return None

    async def fake_save_stage(**kwargs):
        checkpoint.update(kwargs["values"])
        checkpoint["stage"] = kwargs["stage"]
        return checkpoint.copy()

    async def fake_event(task_id, **kwargs):
        events.append(kwargs["event"])

    async def fake_cancel_check(task_id):
        return None

    async def fake_synthesis(**kwargs):
        return (
            "## What the project does\nA FastAPI service exposes an application API "
            "[`demo/main.py:1-2`].\n\n## How it works\nThe app is created at startup."
        )

    monkeypatch.setattr(module, "list_repositories", fake_list_repositories)
    monkeypatch.setattr(module, "get_repository", fake_get_repository)
    monkeypatch.setattr(module, "list_repository_files", fake_list_files)
    monkeypatch.setattr(module, "search_repository_code", fake_search)
    monkeypatch.setattr(module, "read_repository_file", fake_read)
    monkeypatch.setattr(module, "add_evidence", fake_add_evidence)
    monkeypatch.setattr(module, "get_repository_analysis_checkpoint", fake_get_checkpoint)
    monkeypatch.setattr(module, "save_repository_analysis_stage", fake_save_stage)
    monkeypatch.setattr(module, "append_event", fake_event)
    monkeypatch.setattr(module, "ensure_job_not_cancelled", fake_cancel_check)
    monkeypatch.setattr(module, "_synthesize_explanation", fake_synthesis)

    result = await module.repository_analyst.run(
        AgentContext(
            task_id="task-123",
            session_id="session-123",
            objective="Analyze repository-123 FastAPI entry point",
            instruction="Inspect repository-123 and identify the FastAPI entry point",
        )
    )

    assert result.agent == "repository_analyst"
    assert result.error is None
    assert result.evidence_ids == ["evidence-123"]
    assert "demo/main.py" in result.summary
    assert "Test files: 1" in result.summary
    assert "FastAPI" in result.summary
    assert "How it works" in result.summary
    assert checkpoint["result"]["agent"] == "repository_analyst"
    assert events == [
        "repository_inventory_started",
        "repository_inventory_completed",
        "repository_inspection_started",
        "repository_inspection_completed",
        "repository_search_started",
        "repository_search_completed",
        "repository_synthesis_started",
        "repository_synthesis_completed",
        "repository_analysis_completed",
    ]


@pytest.mark.asyncio
async def test_repository_agent_reuses_completed_internal_checkpoint(monkeypatch):
    stored_result = {
        "agent": "repository_analyst",
        "instruction": "Inspect repository-123",
        "summary": "Previously completed analysis",
        "evidence_ids": ["evidence-123"],
        "tool_calls": 4,
        "error": None,
    }
    events = []

    async def fake_list(session_id):
        return [{"repository_id": "repository-123"}]

    async def fake_metadata(repository_id, session_id):
        return {"repository_id": repository_id, "filename": "demo.zip"}

    async def fake_checkpoint(task_id, analysis_key):
        return {"stage": "completed", "result": stored_result}

    async def fake_event(task_id, **kwargs):
        events.append(kwargs["event"])

    monkeypatch.setattr(module, "list_repositories", fake_list)
    monkeypatch.setattr(module, "get_repository", fake_metadata)
    monkeypatch.setattr(module, "get_repository_analysis_checkpoint", fake_checkpoint)
    monkeypatch.setattr(module, "append_event", fake_event)

    result = await module.repository_analyst.run(
        AgentContext(
            task_id="task-123",
            session_id="session-123",
            objective="Inspect repository-123",
            instruction="Inspect repository-123",
        )
    )

    assert result.summary == "Previously completed analysis"
    assert events == ["repository_analysis_reused"]


@pytest.mark.asyncio
async def test_repository_agent_resumes_after_inspection_checkpoint(monkeypatch):
    metadata = {
        "repository_id": "repository-123",
        "filename": "demo.zip",
        "file_count": 1,
        "total_bytes": 40,
        "languages": {"Python": 1},
    }
    state = {
        "stage": "inspection_completed",
        "inventory": {
            "metadata": metadata,
            "files": [{"path": "main.py", "size": 40, "extension": ".py"}],
            "inspection_paths": ["main.py"],
            "terms": ["fastapi"],
        },
        "technology_sources": {"FastAPI": ["main.py"]},
        "excerpts": {"main.py": "1: from fastapi import FastAPI"},
    }
    events = []

    async def fake_list(session_id):
        return [metadata]

    async def fake_metadata(repository_id, session_id):
        return metadata

    async def fake_checkpoint(task_id, analysis_key):
        return state.copy()

    async def fake_save(**kwargs):
        state.update(kwargs["values"])
        state["stage"] = kwargs["stage"]
        return state.copy()

    async def fake_search(repository_id, session_id, query, **kwargs):
        return [{"path": "main.py", "line": 1, "excerpt": "from fastapi import FastAPI"}]

    async def fake_evidence(**kwargs):
        return "evidence-123"

    async def fake_event(task_id, **kwargs):
        events.append(kwargs["event"])

    async def fake_cancel(task_id):
        return None

    async def fake_synthesis(**kwargs):
        return "## How it works\nFastAPI starts in [`main.py:1`]."

    async def should_not_run(*args, **kwargs):
        raise AssertionError("completed inventory/inspection stage ran again")

    monkeypatch.setattr(module, "list_repositories", fake_list)
    monkeypatch.setattr(module, "get_repository", fake_metadata)
    monkeypatch.setattr(module, "get_repository_analysis_checkpoint", fake_checkpoint)
    monkeypatch.setattr(module, "save_repository_analysis_stage", fake_save)
    monkeypatch.setattr(module, "search_repository_code", fake_search)
    monkeypatch.setattr(module, "add_evidence", fake_evidence)
    monkeypatch.setattr(module, "append_event", fake_event)
    monkeypatch.setattr(module, "ensure_job_not_cancelled", fake_cancel)
    monkeypatch.setattr(module, "_synthesize_explanation", fake_synthesis)
    monkeypatch.setattr(module, "list_repository_files", should_not_run)
    monkeypatch.setattr(module, "read_repository_file", should_not_run)

    result = await module.repository_analyst.run(
        AgentContext(
            task_id="task-123",
            session_id="session-123",
            objective="Analyze repository-123 FastAPI",
            instruction="Inspect repository-123 FastAPI",
        )
    )

    assert result.error is None
    assert state["stage"] == "completed"
    assert events == [
        "repository_analysis_resumed",
        "repository_search_started",
        "repository_search_completed",
        "repository_synthesis_started",
        "repository_synthesis_completed",
        "repository_analysis_completed",
    ]


def test_repository_agent_exposes_bounded_inspection_tools():
    tools = module.repository_analyst.build_tools(
        AgentContext(
            task_id="task-123",
            session_id="session-123",
            objective="Inspect code",
            instruction="Inspect code",
        )
    )
    assert {tool.name for tool in tools} == {"list_code_files", "read_code_file", "search_code"}
