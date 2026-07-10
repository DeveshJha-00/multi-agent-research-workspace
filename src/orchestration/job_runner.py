"""Leased background runner for durable research orchestration."""

import asyncio
import logging
from contextlib import suppress
from uuid import uuid4

from src.core.config import settings
from src.db.research_job_store import (
    ResearchJobCancelled,
    append_event,
    cancel_job,
    claim_research_job,
    complete_job,
    fail_job,
    get_research_job,
    heartbeat_job,
    is_cancel_requested,
    release_job,
)
from src.models.api import ResearchResponse
from src.orchestration.research_graph import research_orchestrator

logger = logging.getLogger(__name__)


class ResearchJobRunner:
    """Poll, lease, execute, and safely resume persisted research jobs."""

    def __init__(self) -> None:
        self.worker_id = str(uuid4())
        self._stopping = asyncio.Event()
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        if self._task is None:
            if self._stopping.is_set():
                self._stopping = asyncio.Event()
            self._task = asyncio.create_task(self._run(), name="research-job-runner")

    async def stop(self) -> None:
        self._stopping.set()
        if self._task:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _run(self) -> None:
        while not self._stopping.is_set():
            try:
                job = await claim_research_job(self.worker_id)
                if job:
                    await self._execute(job)
                    continue
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("research_job_poll_failed worker_id=%s", self.worker_id)
            try:
                await asyncio.wait_for(
                    self._stopping.wait(), timeout=settings.research_worker_poll_seconds
                )
            except TimeoutError:
                pass

    async def _heartbeat(self, task_id: str) -> None:
        interval = max(5.0, settings.research_job_lease_seconds / 3)
        while True:
            await asyncio.sleep(interval)
            try:
                if not await heartbeat_job(task_id, self.worker_id):
                    return
            except Exception:
                logger.exception("research_job_heartbeat_failed task_id=%s", task_id)

    async def _execute(self, job: dict) -> None:
        task_id = job["task_id"]
        heartbeat = asyncio.create_task(self._heartbeat(task_id))
        await append_event(
            task_id,
            event="job_started",
            stage="starting",
            progress=max(job.get("progress", 0), 1),
            message=(
                "Research job resumed from its durable checkpoint"
                if job["attempts"] > 1
                else "Research worker started"
            ),
            details={"attempt": job["attempts"]},
        )
        config = {
            "configurable": {"thread_id": task_id, "checkpoint_ns": ""},
            "recursion_limit": settings.graph_recursion_limit,
        }
        try:
            snapshot = await research_orchestrator.aget_state(config)
            if snapshot.values and not snapshot.next and snapshot.values.get("final_answer"):
                result = snapshot.values
            elif snapshot.values:
                result = await research_orchestrator.ainvoke(None, config=config)
            else:
                result = await research_orchestrator.ainvoke(
                    {
                        "task_id": task_id,
                        "session_id": job["session_id"],
                        "objective": job["objective"],
                        "available_data": job.get("available_data", []),
                        "worker_results": [],
                    },
                    config=config,
                )
            if await is_cancel_requested(task_id):
                raise ResearchJobCancelled("Research job was cancelled")
            response = ResearchResponse(
                task_id=task_id,
                content=result["final_answer"],
                worker_results=result.get("worker_results", []),
                artifacts=result.get("artifacts", []),
            ).model_dump(mode="json")
            if await complete_job(task_id, self.worker_id, response):
                await append_event(
                    task_id,
                    event="job_completed",
                    stage="completed",
                    progress=100,
                    message="Research job completed",
                )
        except ResearchJobCancelled:
            current = await get_research_job(task_id)
            if await cancel_job(task_id, self.worker_id):
                await append_event(
                    task_id,
                    event="job_cancelled",
                    stage="cancelled",
                    progress=(current or job).get("progress", 0),
                    message="Research job cancelled",
                )
        except asyncio.CancelledError:
            logger.info("research_job_interrupted task_id=%s", task_id)
            await release_job(task_id, self.worker_id)
            raise
        except Exception as exc:
            if await is_cancel_requested(task_id):
                current = await get_research_job(task_id)
                cancelled = await cancel_job(task_id, self.worker_id)
                if cancelled:
                    await append_event(
                        task_id,
                        event="job_cancelled",
                        stage="cancelled",
                        progress=(current or job).get("progress", 0),
                        message="Research job cancelled",
                    )
            else:
                logger.exception("research_job_failed task_id=%s", task_id)
                current = await get_research_job(task_id)
                if await fail_job(task_id, self.worker_id, str(exc)):
                    await append_event(
                        task_id,
                        event="job_failed",
                        stage="failed",
                        progress=(current or job).get("progress", 0),
                        message="Research job failed and can be retried",
                        details={"error": str(exc)[:500]},
                    )
        finally:
            heartbeat.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await heartbeat


research_job_runner = ResearchJobRunner()
