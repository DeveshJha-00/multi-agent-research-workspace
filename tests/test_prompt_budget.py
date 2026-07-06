import json

from src.core.prompt_budget import compact_evidence, compact_results
from src.models.agent import AgentResult


def test_evidence_compaction_keeps_multiple_agents():
    records = [
        {"agent": "document_investigator", "content": "d" * 5000, "evidence_id": "doc-1"},
        {"agent": "document_investigator", "content": "d" * 5000, "evidence_id": "doc-2"},
        {"agent": "web_researcher", "content": "w" * 5000, "evidence_id": "web-1"},
    ]
    compacted = json.loads(compact_evidence(records, max_chars=2000))
    assert {item["agent"] for item in compacted} == {
        "document_investigator",
        "web_researcher",
    }
    assert len(json.dumps(compacted)) <= 2000


def test_result_compaction_obeys_budget():
    results = [
        AgentResult(agent="a", instruction="work", summary="x" * 5000),
        AgentResult(agent="b", instruction="work", summary="y" * 5000),
    ]
    assert len(compact_results(results, max_chars=1200)) <= 1200
