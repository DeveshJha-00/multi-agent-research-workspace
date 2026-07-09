"""LangGraph state schema."""

from typing import Annotated, Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages


class State(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    session_id: str
    route: Literal["index", "general", "search"]
    classifier_route: Literal["index", "general", "search"]
    latest_query: str
    documents: list[Document]
    reranked_documents: list[Document]
    retry_count: int
    answer: str
    sources: list[dict]
    verification_attempts: int
    verification_context: str
    evaluation_contexts: list[dict]
    faithful: bool
