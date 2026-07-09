from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from src.api.routes import _evaluation_response, router
from src.db import evaluation_job_store, rag_response_store
from src.evaluation import job_runner, ragas_evaluator
from src.models.api import EvaluationStatusResponse, QueryResponse
from src.rag.graph_builder import _evaluation_contexts


def test_metric_selection_depends_on_route_context_and_reference():
    assert ragas_evaluator.metric_names_for(
        "general", has_contexts=False, has_reference=False
    ) == ["answer_relevancy"]
    assert ragas_evaluator.metric_names_for(
        "index", has_contexts=True, has_reference=False
    ) == ["answer_relevancy", "faithfulness", "context_utilization"]
    assert ragas_evaluator.metric_names_for(
        "search", has_contexts=True, has_reference=True
    ) == [
        "answer_relevancy",
        "faithfulness",
        "context_utilization",
        "factual_correctness",
        "semantic_similarity",
        "context_precision",
        "context_recall",
    ]


def test_ragas_telemetry_is_disabled_and_multimodal_metrics_are_not_imported():
    import os
    from pathlib import Path

    assert os.environ["RAGAS_DO_NOT_TRACK"] == "true"
    source = Path("src/evaluation/ragas_evaluator.py").read_text(encoding="utf-8")
    assert "MultiModal" not in source


def test_query_response_exposes_stable_response_id():
    response = QueryResponse(
        response_id="response-123",
        content="Answer",
        route="general",
        sources=[],
    )
    assert response.response_id == "response-123"


def test_evaluation_routes_are_async_and_queryable():
    routes = {(route.path, frozenset(route.methods or [])): route for route in router.routes}
    submit = routes[("/rag/evaluations", frozenset({"POST"}))]
    assert submit.status_code == 202
    assert ("/rag/evaluations/{evaluation_id}", frozenset({"GET"})) in routes
    assert ("/rag/evaluations", frozenset({"GET"})) in routes
    assert ("/rag/documents", frozenset({"GET"})) in routes


def test_evaluation_status_serializes_incremental_metrics():
    now = datetime.now(timezone.utc)
    status = EvaluationStatusResponse(
        evaluation_id="evaluation-123",
        response_id="response-123",
        status="running",
        progress=50,
        attempts=1,
        metric_names=["answer_relevancy"],
        metrics={
            "answer_relevancy": {
                "name": "answer_relevancy",
                "status": "completed",
                "score": 0.9,
            }
        },
        created_at=now,
        updated_at=now,
    )
    assert status.metrics["answer_relevancy"].score == 0.9


def test_evaluation_response_does_not_duplicate_internal_fields():
    now = datetime.now(timezone.utc)
    response = _evaluation_response(
        {
            "evaluation_id": "evaluation-123",
            "session_id": "session-123",
            "response_id": "response-123",
            "status": "failed",
            "progress": 50,
            "attempts": 1,
            "metric_names": ["faithfulness"],
            "metrics": {
                "faithfulness": {
                    "name": "faithfulness",
                    "status": "failed",
                    "error": "judge unavailable",
                }
            },
            "reference": "expected answer",
            "reference_supplied": True,
            "idempotency_key": "hidden",
            "request_hash": "hidden",
            "created_at": now,
            "updated_at": now,
        },
        context_count=2,
    )
    assert response.reference_supplied is True
    assert response.context_count == 2
    assert response.metrics["faithfulness"].status == "failed"


def test_response_contexts_are_bounded(monkeypatch):
    monkeypatch.setattr(rag_response_store.settings, "ragas_max_contexts", 2)
    monkeypatch.setattr(rag_response_store.settings, "ragas_max_context_chars", 7)
    contexts = rag_response_store._bounded_contexts(
        [
            {"content": "abcd", "source": "one"},
            {"content": "efgh", "source": "two"},
            {"content": "ignored", "source": "three"},
        ]
    )
    assert [item["content"] for item in contexts] == ["abcd", "efg"]


@pytest.mark.asyncio
async def test_response_lookup_always_filters_by_session(monkeypatch):
    captured = {}

    class FakeCollection:
        async def find_one(self, query):
            captured.update(query)
            return None

    monkeypatch.setattr(rag_response_store, "collection", FakeCollection())
    await rag_response_store.get_rag_response("response-1", "session-123")
    assert captured == {"response_id": "response-1", "session_id": "session-123"}


@pytest.mark.asyncio
async def test_evaluation_idempotency_reuses_only_the_same_request(monkeypatch):
    class FakeJobs:
        def __init__(self):
            self.documents = []

        async def find_one(self, query):
            return next(
                (
                    item
                    for item in self.documents
                    if all(item.get(key) == value for key, value in query.items())
                ),
                None,
            )

        async def insert_one(self, document):
            self.documents.append(document.copy())

    fake = FakeJobs()
    monkeypatch.setattr(evaluation_job_store, "jobs", fake)
    arguments = {
        "session_id": "session-123",
        "response_id": "response-123",
        "reference": None,
        "metric_names": ["answer_relevancy"],
        "idempotency_key": "same-key",
    }
    first, first_reused = await evaluation_job_store.create_evaluation_job(**arguments)
    second, second_reused = await evaluation_job_store.create_evaluation_job(**arguments)
    assert first_reused is False
    assert second_reused is True
    assert second["evaluation_id"] == first["evaluation_id"]
    with pytest.raises(evaluation_job_store.EvaluationIdempotencyConflictError):
        await evaluation_job_store.create_evaluation_job(
            **{**arguments, "response_id": "different-response"}
        )


def test_graph_context_snapshot_contains_text_without_loading_sources(monkeypatch):
    monkeypatch.setattr(rag_response_store.settings, "ragas_max_contexts", 3)
    documents = [
        Document(
            page_content="policy evidence",
            metadata={"source": "policy.txt", "url": "https://example.test/policy"},
        )
    ]
    snapshot = _evaluation_contexts(documents)
    assert snapshot[0]["content"] == "policy evidence"
    assert snapshot[0]["url"] == "https://example.test/policy"


@pytest.mark.asyncio
async def test_metric_scoring_returns_value_and_reason(monkeypatch):
    class FakeMetric:
        async def ascore(self, **kwargs):
            assert kwargs == {"user_input": "Question", "response": "Answer"}
            return SimpleNamespace(value=0.75, reason="Directly addresses the question")

    monkeypatch.setattr(
        ragas_evaluator, "get_metrics", lambda: {"answer_relevancy": FakeMetric()}
    )
    result = await ragas_evaluator.score_metric(
        "answer_relevancy",
        {"question": "Question", "answer": "Answer", "contexts": []},
        None,
    )
    assert result == {"score": 0.75, "reason": "Directly addresses the question"}


@pytest.mark.asyncio
async def test_metric_scoring_recovers_json_validation_with_direct_judge(monkeypatch):
    class BrokenMetric:
        async def ascore(self, **kwargs):
            raise RuntimeError("Error code: 400 - json_validate_failed")

    async def fake_recovery(metric_name, snapshot, reference, contexts, original_error):
        assert metric_name == "faithfulness"
        assert contexts == ["A"]
        assert "json_validate_failed" in str(original_error)
        return {"score": 0.82, "reason": "Recovered with direct judge."}

    monkeypatch.setattr(
        ragas_evaluator,
        "get_metrics",
        lambda: {"faithfulness": BrokenMetric()},
    )
    monkeypatch.setattr(
        ragas_evaluator,
        "_try_direct_json_judge_recovery",
        fake_recovery,
    )
    result = await ragas_evaluator.score_metric(
        "faithfulness",
        {"question": "Q", "answer": "A", "contexts": [{"content": "A"}]},
        None,
    )
    assert result == {"score": 0.82, "reason": "Recovered with direct judge."}


@pytest.mark.asyncio
async def test_metric_scoring_uses_local_fallback_after_judge_recovery_fails(monkeypatch):
    class BrokenMetric:
        async def ascore(self, **kwargs):
            raise RuntimeError("Error code: 400 - json_validate_failed")

    async def fake_no_recovery(*args, **kwargs):
        return None

    class FakeEmbeddings:
        async def aembed_query(self, text):
            return [1.0, 0.0]

        async def aembed_documents(self, texts):
            return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(
        ragas_evaluator, "get_metrics", lambda: {"faithfulness": BrokenMetric()}
    )
    monkeypatch.setattr(
        ragas_evaluator,
        "_try_direct_json_judge_recovery",
        fake_no_recovery,
    )
    monkeypatch.setattr(ragas_evaluator, "get_embeddings", lambda: FakeEmbeddings())
    result = await ragas_evaluator.score_metric(
        "faithfulness",
        {"question": "Q", "answer": "A", "contexts": [{"content": "A"}]},
        None,
    )
    assert result["score"] == 1.0
    assert "Local FastEmbed semantic-similarity fallback" in result["reason"]


def test_metric_errors_are_user_friendly():
    assert (
        job_runner._friendly_metric_error(RuntimeError("429 rate_limit_exceeded"))
        == "RAGAS judge rate limit was reached. Retry when provider limits reset."
    )
    assert "structured JSON" in job_runner._friendly_metric_error(
        RuntimeError("<failed_attempts> json_validate_failed lots of noisy XML")
    )


@pytest.mark.asyncio
async def test_local_embedding_adapter_reuses_application_model(monkeypatch):
    class FakeEmbeddings:
        def embed_query(self, text):
            return [1.0]

        def embed_documents(self, texts):
            return [[1.0] for _ in texts]

        async def aembed_query(self, text):
            return [2.0]

        async def aembed_documents(self, texts):
            return [[2.0] for _ in texts]

    monkeypatch.setattr(ragas_evaluator, "get_embeddings", lambda: FakeEmbeddings())
    adapter = ragas_evaluator.LocalFastEmbedRagasEmbedding()
    assert adapter.embed_text("one") == [1.0]
    assert adapter.embed_texts(["one", "two"]) == [[1.0], [1.0]]
    assert await adapter.aembed_text("one") == [2.0]


@pytest.mark.asyncio
async def test_worker_resumes_without_repeating_completed_metrics(monkeypatch):
    scored = []
    recorded = []
    completed = []

    async def fake_snapshot(response_id, session_id):
        return {"question": "Q", "answer": "A", "contexts": []}

    async def fake_score(name, snapshot, reference):
        scored.append(name)
        return {"score": 0.8, "reason": None}

    async def fake_record(evaluation_id, worker_id, name, result):
        recorded.append((name, result["status"]))
        return True

    async def fake_complete(evaluation_id, worker_id, duration):
        completed.append(evaluation_id)
        return True

    monkeypatch.setattr(job_runner, "get_rag_response", fake_snapshot)
    monkeypatch.setattr(job_runner, "score_metric", fake_score)
    monkeypatch.setattr(job_runner, "record_metric_result", fake_record)
    monkeypatch.setattr(job_runner, "complete_evaluation_job", fake_complete)
    monkeypatch.setattr(job_runner.settings, "evaluation_metric_delay_seconds", 0)

    runner = job_runner.EvaluationJobRunner()
    await runner._execute(
        {
            "evaluation_id": "evaluation-1",
            "response_id": "response-1",
            "session_id": "session-123",
            "reference": None,
            "metric_names": ["answer_relevancy", "faithfulness"],
            "metrics": {"answer_relevancy": {"status": "completed", "score": 0.9}},
        }
    )
    assert scored == ["faithfulness"]
    assert recorded == [("faithfulness", "completed")]
    assert completed == ["evaluation-1"]
