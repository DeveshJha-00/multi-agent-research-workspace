"""Single-concurrency durable worker for RAGAS evaluation jobs."""

import asyncio
import logging
from contextlib import suppress
from time import monotonic
from uuid import uuid4

from src.core.config import settings
from src.db.evaluation_job_store import (
    claim_evaluation_job,
    complete_evaluation_job,
    fail_evaluation_job,
    heartbeat_evaluation_job,
    record_metric_result,
    release_evaluation_job,
)
from src.db.rag_response_store import get_rag_response
from src.evaluation.ragas_evaluator import score_metric

logger = logging.getLogger(__name__)


def _friendly_metric_error(exc: Exception) -> str:
    message = str(exc)
    lower = message.lower()
    if "rate_limit" in lower or "too many requests" in lower or "429" in lower:
        return "RAGAS judge rate limit was reached. Retry when provider limits reset."
    if "json_validate_failed" in lower or "failed to validate json" in lower:
        return (
            "RAGAS judge returned invalid structured JSON for this metric. "
            "Retry later or use a judge model with stronger structured-output support."
        )
    first_line = message.strip().splitlines()[0] if message.strip() else exc.__class__.__name__
    return first_line[:500]


class EvaluationJobRunner:
    def __init__(self) -> None:
        self.worker_id = str(uuid4())
        self._stopping = asyncio.Event()
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        if self._task is None and settings.ragas_enabled:
            if self._stopping.is_set():
                self._stopping = asyncio.Event()
            self._task = asyncio.create_task(self._run(), name="evaluation-job-runner")

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
                job = await claim_evaluation_job(self.worker_id)
                if job:
                    await self._execute(job)
                    continue
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("evaluation_job_poll_failed worker_id=%s", self.worker_id)
            try:
                await asyncio.wait_for(
                    self._stopping.wait(), timeout=settings.evaluation_worker_poll_seconds
                )
            except TimeoutError:
                pass

    async def _heartbeat(self, evaluation_id: str) -> None:
        interval = max(10.0, settings.evaluation_job_lease_seconds / 3)
        while True:
            await asyncio.sleep(interval)
            if not await heartbeat_evaluation_job(evaluation_id, self.worker_id):
                return

    async def _execute(self, job: dict) -> None:
        evaluation_id = job["evaluation_id"]
        started = monotonic()
        heartbeat = asyncio.create_task(self._heartbeat(evaluation_id))
        try:
            snapshot = await get_rag_response(job["response_id"], job["session_id"])
            if snapshot is None:
                raise RuntimeError("The Chat response snapshot no longer exists")
            completed = job.get("metrics", {})
            pending = [name for name in job["metric_names"] if name not in completed]
            for index, metric_name in enumerate(pending):
                metric_started = monotonic()
                try:
                    scored = await score_metric(metric_name, snapshot, job.get("reference"))
                    result = {
                        "name": metric_name,
                        "status": "completed",
                        **scored,
                        "duration_seconds": round(monotonic() - metric_started, 3),
                        "error": None,
                    }
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.exception(
                        "evaluation_metric_failed evaluation_id=%s metric=%s",
                        evaluation_id,
                        metric_name,
                    )
                    result = {
                        "name": metric_name,
                        "status": "failed",
                        "score": None,
                        "reason": None,
                        "duration_seconds": round(monotonic() - metric_started, 3),
                        "error": _friendly_metric_error(exc),
                    }
                if not await record_metric_result(
                    evaluation_id, self.worker_id, metric_name, result
                ):
                    raise RuntimeError("Evaluation lease was lost while recording a metric")
                if index < len(pending) - 1 and settings.evaluation_metric_delay_seconds:
                    await asyncio.sleep(settings.evaluation_metric_delay_seconds)
            await complete_evaluation_job(
                evaluation_id, self.worker_id, round(monotonic() - started, 3)
            )
        except asyncio.CancelledError:
            await release_evaluation_job(evaluation_id, self.worker_id)
            raise
        except Exception as exc:
            logger.exception("evaluation_job_failed evaluation_id=%s", evaluation_id)
            await fail_evaluation_job(
                evaluation_id,
                self.worker_id,
                str(exc),
                round(monotonic() - started, 3),
            )
        finally:
            heartbeat.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await heartbeat


evaluation_job_runner = EvaluationJobRunner()
