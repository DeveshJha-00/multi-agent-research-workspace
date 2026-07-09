"""Public RAG API routes. Authentication is intentionally not enabled."""

import logging
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, File, Header, HTTPException, UploadFile, status
from groq import APIError, AuthenticationError, RateLimitError
from langchain_core.messages import AIMessage, HumanMessage

from src.core.config import settings
from src.core.integration_errors import groq_error_detail, groq_rate_limit_detail
from src.db.evaluation_job_store import (
    EvaluationIdempotencyConflictError,
    clear_evaluation_jobs,
    create_evaluation_job,
    get_evaluation_job,
    list_evaluation_jobs,
)
from src.db.rag_response_store import (
    clear_rag_responses,
    get_rag_response,
    save_rag_response,
)
from src.evaluation.ragas_evaluator import metric_names_for
from src.memory.chat_history_mongo import ChatHistory
from src.models.api import (
    DeleteResponse,
    EvaluationCreated,
    EvaluationRequest,
    EvaluationStatusResponse,
    IndexedDocumentResponse,
    QueryResponse,
    UploadResponse,
)
from src.models.query_request import QueryRequest
from src.rag.document_upload import documents
from src.rag.graph_builder import builder
from src.rag.retriever_setup import delete_document, list_documents

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/query", response_model=QueryResponse)
async def rag_query(req: QueryRequest) -> QueryResponse:
    history = ChatHistory.get_session_history(req.session_id)
    messages = await history.get_messages()
    user_message = HumanMessage(content=req.query)
    messages.append(user_message)
    try:
        result = await builder.ainvoke(
            {"messages": messages, "session_id": req.session_id},
            config={"recursion_limit": settings.graph_recursion_limit},
        )
        answer = str(result["answer"])
        route = result["route"]
        sources = result.get("sources", [])
    except Exception as exc:
        logger.exception("query_failed session_id=%s", req.session_id)
        if isinstance(exc, AuthenticationError):
            detail = "Groq rejected GROQ_API_KEY. Update .env and restart the API."
        elif isinstance(exc, RateLimitError):
            detail = groq_rate_limit_detail(exc)
        elif isinstance(exc, APIError):
            detail = groq_error_detail(exc)
        else:
            detail = "The answer service is temporarily unavailable"
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail,
        ) from exc

    await history.add_messages([user_message, AIMessage(content=answer)])
    response_id = str(uuid4())
    try:
        await save_rag_response(
            response_id=response_id,
            session_id=req.session_id,
            question=req.query,
            answer=answer,
            route=route,
            sources=sources,
            contexts=result.get("evaluation_contexts", []),
        )
    except Exception:
        logger.exception("response_snapshot_failed response_id=%s", response_id)
    return QueryResponse(
        response_id=response_id, content=answer, route=route, sources=sources
    )


def _evaluation_response(job: dict, context_count: int = 0) -> EvaluationStatusResponse:
    public_job = {
        key: value
        for key, value in job.items()
        if key
        not in {
            "_id",
            "session_id",
            "reference",
            "reference_supplied",
            "idempotency_key",
            "request_hash",
            "lease_owner",
            "lease_expires_at",
        }
    }
    return EvaluationStatusResponse(
        **public_job,
        context_count=context_count,
        reference_supplied=bool(job.get("reference_supplied", job.get("reference"))),
    )


@router.post(
    "/evaluations", response_model=EvaluationCreated, status_code=status.HTTP_202_ACCEPTED
)
async def create_evaluation(
    request: EvaluationRequest,
    session_id: Annotated[str, Header(alias="X-Session-ID", min_length=8, max_length=200)],
    idempotency_key: Annotated[
        str | None, Header(alias="Idempotency-Key", min_length=1, max_length=200)
    ] = None,
) -> EvaluationCreated:
    if not settings.ragas_enabled:
        raise HTTPException(status_code=503, detail="RAGAS evaluation is disabled")
    snapshot = await get_rag_response(request.response_id, session_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="Chat response snapshot not found")
    metric_names = metric_names_for(
        snapshot["route"],
        has_contexts=bool(snapshot.get("contexts")),
        has_reference=bool(request.reference and request.reference.strip()),
    )
    try:
        job, reused = await create_evaluation_job(
            session_id=session_id,
            response_id=request.response_id,
            reference=request.reference,
            metric_names=metric_names,
            idempotency_key=idempotency_key,
        )
    except EvaluationIdempotencyConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return EvaluationCreated(
        evaluation_id=job["evaluation_id"],
        response_id=job["response_id"],
        status=job["status"],
        reused=reused,
    )


@router.get("/evaluations", response_model=list[EvaluationStatusResponse])
async def get_evaluations(
    session_id: Annotated[str, Header(alias="X-Session-ID", min_length=8, max_length=200)],
    response_id: str | None = None,
) -> list[EvaluationStatusResponse]:
    output = []
    for job in await list_evaluation_jobs(session_id, response_id=response_id):
        snapshot = await get_rag_response(job["response_id"], session_id)
        output.append(_evaluation_response(job, len((snapshot or {}).get("contexts", []))))
    return output


@router.get("/evaluations/{evaluation_id}", response_model=EvaluationStatusResponse)
async def get_evaluation(
    evaluation_id: str,
    session_id: Annotated[str, Header(alias="X-Session-ID", min_length=8, max_length=200)],
) -> EvaluationStatusResponse:
    job = await get_evaluation_job(evaluation_id, session_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Evaluation job not found")
    snapshot = await get_rag_response(job["response_id"], session_id)
    return _evaluation_response(job, len((snapshot or {}).get("contexts", [])))


@router.post("/documents/upload", response_model=UploadResponse)
async def upload_file(
    file: Annotated[UploadFile, File()],
    session_id: Annotated[str, Header(alias="X-Session-ID", min_length=8, max_length=200)],
    description: Annotated[str, Header(alias="X-Description", max_length=500)] = "",
) -> UploadResponse:
    result = await documents(description, file, session_id)
    return UploadResponse(**result)


@router.get("/documents", response_model=list[IndexedDocumentResponse])
async def indexed_documents(
    session_id: Annotated[str, Header(alias="X-Session-ID", min_length=8, max_length=200)],
) -> list[IndexedDocumentResponse]:
    return [IndexedDocumentResponse(**item) for item in await list_documents(session_id=session_id)]


@router.delete("/documents/{document_id}", response_model=DeleteResponse)
async def remove_document(
    document_id: str,
    session_id: Annotated[str, Header(alias="X-Session-ID", min_length=8, max_length=200)],
) -> DeleteResponse:
    await delete_document(session_id=session_id, document_id=document_id)
    return DeleteResponse(status=True, document_id=document_id)


@router.delete("/history", status_code=status.HTTP_204_NO_CONTENT)
async def clear_history(
    session_id: Annotated[str, Header(alias="X-Session-ID", min_length=8, max_length=200)],
) -> None:
    await ChatHistory.get_session_history(session_id).clear()
    await clear_evaluation_jobs(session_id)
    await clear_rag_responses(session_id)
