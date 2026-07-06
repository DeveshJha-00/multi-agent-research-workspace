"""Public RAG API routes. Authentication is intentionally not enabled."""

import logging
from typing import Annotated

from fastapi import APIRouter, File, Header, HTTPException, UploadFile, status
from groq import APIError, AuthenticationError, RateLimitError
from langchain_core.messages import AIMessage, HumanMessage

from src.core.config import settings
from src.core.integration_errors import groq_error_detail, groq_rate_limit_detail
from src.memory.chat_history_mongo import ChatHistory
from src.models.api import DeleteResponse, QueryResponse, UploadResponse
from src.models.query_request import QueryRequest
from src.rag.document_upload import documents
from src.rag.graph_builder import builder
from src.rag.retriever_setup import delete_document

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
    return QueryResponse(content=answer, route=route, sources=sources)


@router.post("/documents/upload", response_model=UploadResponse)
async def upload_file(
    file: Annotated[UploadFile, File()],
    description: Annotated[str, Header(alias="X-Description", min_length=1, max_length=500)],
    session_id: Annotated[str, Header(alias="X-Session-ID", min_length=8, max_length=200)],
) -> UploadResponse:
    result = await documents(description, file, session_id)
    return UploadResponse(**result)


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
