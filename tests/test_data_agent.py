from io import BytesIO

import pandas as pd
import pytest
from fastapi import UploadFile

from src.agents import data_analyst as data_module
from src.agents.base import AgentContext
from src.data import ingestion


@pytest.mark.asyncio
async def test_csv_ingestion_validates_and_persists(monkeypatch):
    captured = {}

    async def fake_save(frame, **kwargs):
        captured["frame"] = frame
        captured.update(kwargs)
        return {
            "dataset_id": "dataset-123",
            "filename": kwargs["filename"],
            "description": kwargs["description"],
            "columns": list(frame.columns),
            "row_count": len(frame),
        }

    monkeypatch.setattr(ingestion, "save_dataframe", fake_save)
    upload = UploadFile(filename="sales.CSV", file=BytesIO(b"region,revenue\nNorth,10\nSouth,20"))
    result = await ingestion.ingest_dataset(
        file=upload, session_id="session-123", description="Sales data"
    )
    assert result["dataset_id"] == "dataset-123"
    assert result["row_count"] == 2
    assert list(captured["frame"].columns) == ["region", "revenue"]


@pytest.mark.asyncio
async def test_data_agent_tools_calculate_and_create_chart(monkeypatch):
    frame = pd.DataFrame({"region": ["North", "South", "North"], "revenue": [10, 20, 30]})
    metadata = {"filename": "sales.csv", "columns": list(frame.columns), "row_count": 3}
    evidence = []
    artifacts = []

    async def fake_load(dataset_id, session_id):
        assert dataset_id == "dataset-123"
        assert session_id == "session-123"
        return frame, metadata

    async def fake_list(session_id):
        return [{"dataset_id": "dataset-123", **metadata}]

    async def fake_evidence(**kwargs):
        evidence.append(kwargs)
        return f"evidence-{len(evidence)}"

    async def fake_artifact(**kwargs):
        artifacts.append(kwargs)
        return "artifact-123"

    monkeypatch.setattr(data_module, "load_dataframe", fake_load)
    monkeypatch.setattr(data_module, "list_datasets", fake_list)
    monkeypatch.setattr(data_module, "add_evidence", fake_evidence)
    monkeypatch.setattr(data_module, "save_artifact", fake_artifact)

    context = AgentContext(
        task_id="task-123",
        session_id="session-123",
        objective="Analyze revenue",
        instruction="Compare regions",
    )
    tools = {tool.name: tool for tool in data_module.data_analyst.build_tools(context)}
    aggregate = await tools["aggregate_dataset"].ainvoke(
        {
            "dataset_id": "dataset-123",
            "group_by": "region",
            "metric": "revenue",
            "aggregation": "sum",
        }
    )
    chart = await tools["create_chart"].ainvoke(
        {
            "dataset_id": "dataset-123",
            "chart_type": "bar",
            "x": "region",
            "y": "revenue",
            "title": "Revenue",
        }
    )

    assert aggregate["result"] == {"North": 40, "South": 20}
    assert chart["artifact_id"] == "artifact-123"
    assert artifacts[0]["media_type"] == "image/png"
    assert artifacts[0]["content"].startswith(b"\x89PNG")
    assert context.evidence_ids == ["evidence-1"]


@pytest.mark.asyncio
async def test_data_agent_run_produces_analysis_evidence_and_chart(monkeypatch):
    frame = pd.DataFrame(
        {
            "region": ["North", "South", "North"],
            "quarter": ["Q1", "Q1", "Q2"],
            "revenue": [100, 80, 120],
            "cost": [60, 50, 70],
        }
    )
    metadata = {"filename": "sales.csv", "columns": list(frame.columns), "row_count": 3}
    evidence = []
    artifacts = []

    async def fake_list(session_id):
        return [{"dataset_id": "dataset-123", **metadata}]

    async def fake_load(dataset_id, session_id):
        return frame, metadata

    async def fake_evidence(**kwargs):
        evidence.append(kwargs)
        return "evidence-123"

    async def fake_artifact(**kwargs):
        artifacts.append(kwargs)
        return "chart-123"

    monkeypatch.setattr(data_module, "list_datasets", fake_list)
    monkeypatch.setattr(data_module, "load_dataframe", fake_load)
    monkeypatch.setattr(data_module, "add_evidence", fake_evidence)
    monkeypatch.setattr(data_module, "save_artifact", fake_artifact)

    result = await data_module.data_analyst.run(
        AgentContext(
            task_id="task-123",
            session_id="session-123",
            objective="Analyze dataset dataset-123",
            instruction="Calculate revenue, profit, regional and quarterly performance, and chart it.",
        )
    )

    assert result.error is None
    assert "profit" in result.summary
    assert "North" in result.summary
    assert "Top by profit" in result.summary
    assert evidence and '"profit"' in evidence[0]["content"]
    assert artifacts[0]["media_type"] == "image/png"
    assert artifacts[0]["content"].startswith(b"\x89PNG")
