"""Deterministic LangGraph routing functions."""

from typing import Literal

from src.core.config import settings
from src.models.state import State


def routing_tool(state: State) -> Literal["rerank", "general_llm", "web_search"]:
    return {
        "index": "rerank",
        "general": "general_llm",
        "search": "web_search",
    }[state["route"]]


def retrieval_decision(state: State) -> Literal["generate", "rewrite", "web_search"]:
    documents = state.get("reranked_documents", [])
    if documents and state.get("classifier_route") == "search":
        return "web_search"
    if documents:
        return "generate"
    if state.get("retry_count", 0) < settings.max_retrieval_retries:
        return "rewrite"
    return "web_search"


def generation_decision(state: State) -> Literal["__end__", "verify"]:
    if state.get("route") == "index":
        return "__end__"
    return "verify"


def verification_decision(state: State) -> Literal["__end__", "generate", "safe_fallback"]:
    if state.get("faithful", False):
        return "__end__"
    if state.get("verification_attempts", 0) <= 1:
        return "generate"
    return "safe_fallback"
