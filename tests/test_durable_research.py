from datetime import datetime, timezone

from src.api.agent_routes import router
from src.core.idempotency import canonical_hash, operation_key
from src.db.checkpoint_store import checkpoint_saver
from src.models.api import ResearchJobStatusResponse
from src.orchestration.research_graph import research_orchestrator


def test_idempotency_hash_is_stable_across_mapping_order():
    assert canonical_hash({"a": 1, "b": [2, 3]}) == canonical_hash({"b": [2, 3], "a": 1})
    assert operation_key("report", "objective") != operation_key("chart", "objective")


def test_research_graph_uses_persistent_checkpoint_saver():
    assert research_orchestrator.checkpointer is checkpoint_saver


def test_research_submission_is_async_and_exposes_job_lifecycle_routes():
    routes = {(route.path, frozenset(route.methods or [])): route for route in router.routes}
    submit = routes[("/agents/research", frozenset({"POST"}))]
    assert submit.status_code == 202
    assert ("/agents/tasks/{task_id}/events/stream", frozenset({"GET"})) in routes
    assert ("/agents/tasks/{task_id}/retry", frozenset({"POST"})) in routes
    assert ("/agents/tasks/{task_id}", frozenset({"DELETE"})) in routes
    assert ("/agents/repositories/upload", frozenset({"POST"})) in routes


def test_job_status_model_accepts_durable_lifecycle_fields():
    now = datetime.now(timezone.utc)
    model = ResearchJobStatusResponse(
        task_id="task-1",
        objective="Research a sufficiently detailed topic",
        status="running",
        stage="specialists",
        progress=50,
        attempts=1,
        created_at=now,
        updated_at=now,
    )
    assert model.status == "running"
    assert model.progress == 50
