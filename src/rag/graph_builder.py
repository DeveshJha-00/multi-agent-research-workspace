"""Adaptive retrieval graph with explicit reranking and answer verification."""

import logging
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.prompts import PromptTemplate
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from tavily import AsyncTavilyClient

from src.config.settings import Config
from src.core.config import settings
from src.llms.provider import get_llm, get_structured_llm, rerank_passages
from src.models.route_identifier import RouteIdentifier
from src.models.state import State
from src.models.verification_result import VerificationResult
from src.rag.retriever_setup import retrieve_documents
from src.tools.graph_tools import retrieval_decision, routing_tool, verification_decision

logger = logging.getLogger(__name__)
config = Config()


def _history(messages: list[BaseMessage], limit: int = 10) -> str:
    selected = messages[-limit:]
    return "\n".join(f"{message.type}: {message.content}" for message in selected)


def _format_documents(documents: list[Document], max_chars: int = 4000) -> str:
    blocks = []
    used = 0
    for index, doc in enumerate(documents, 1):
        block = f"[{index}] {doc.page_content.strip()}"
        if used + len(block) > max_chars and blocks:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks) if blocks else "No relevant indexed excerpts found."


def _sources(documents: list[Document]) -> list[dict]:
    output: list[dict] = []
    seen: set[tuple[Any, ...]] = set()
    for doc in documents:
        metadata = doc.metadata
        item = {
            "source": str(metadata.get("source") or metadata.get("title") or "Unknown source"),
            "document_id": metadata.get("document_id"),
            "page": metadata.get("page"),
            "url": metadata.get("url"),
        }
        key = tuple(item.values())
        if key not in seen:
            seen.add(key)
            output.append(item)
    return output


async def query_classifier(state: State) -> dict:
    question = state["messages"][-1].content
    documents = await retrieve_documents(question, session_id=state["session_id"])
    prompt = PromptTemplate.from_template(config.prompt("classify_prompt"))
    classifier = get_structured_llm(RouteIdentifier)
    result = await (prompt | classifier).ainvoke(
        {
            "question": question,
            "history": _history(state["messages"][:-1]),
            "context": _format_documents(documents, max_chars=5000),
        }
    )
    logger.info("query_routed route=%s candidates=%d", result.route, len(documents))
    return {
        "route": result.route,
        "latest_query": question,
        "documents": documents,
        "retry_count": 0,
        "verification_attempts": 0,
    }


async def general_llm(state: State) -> dict:
    result = await get_llm().ainvoke(state["messages"])
    return {
        "messages": [result],
        "answer": result.content,
        "sources": [],
        "route": "general",
    }


async def retriever_node(state: State) -> dict:
    documents = await retrieve_documents(
        state["latest_query"],
        session_id=state["session_id"],
    )
    return {"documents": documents}


async def rerank(state: State) -> dict:
    candidates = state.get("documents", [])
    if not candidates:
        return {"reranked_documents": []}

    try:
        results = await rerank_passages(
            state["latest_query"], [doc.page_content for doc in candidates]
        )
        ranked = []
        for index, score in results:
            if index < 0 or index >= len(candidates):
                continue
            doc = candidates[index]
            doc.metadata["rerank_score"] = score
            ranked.append(doc)
            if len(ranked) >= settings.rerank_top_n:
                break
    except Exception:
        logger.exception("Local cross-encoder reranking failed; using vector-score order")
        ranked = sorted(
            candidates,
            key=lambda doc: float(doc.metadata.get("vector_score", 0.0)),
            reverse=True,
        )[: settings.rerank_top_n]
        for doc in ranked:
            doc.metadata["rerank_score"] = float(doc.metadata.get("vector_score", 0.0))
    return {"reranked_documents": ranked}


async def rewrite_query(state: State) -> dict:
    prompt = PromptTemplate.from_template(config.prompt("rewrite_prompt"))
    result = await (prompt | get_llm()).ainvoke(
        {
            "query": state["latest_query"],
            "history": _history(state["messages"]),
        }
    )
    return {
        "latest_query": result.content.strip(),
        "retry_count": state.get("retry_count", 0) + 1,
    }


async def web_search(state: State) -> dict:
    search_client = AsyncTavilyClient(api_key=settings.tavily_api_key)
    response = await search_client.search(
        state["latest_query"],
        max_results=5,
        include_answer=False,
        search_depth="basic",
    )
    raw_results = response.get("results", [])
    documents: list[Document] = []
    if isinstance(raw_results, list):
        for item in raw_results:
            if not isinstance(item, dict) or not item.get("content"):
                continue
            documents.append(
                Document(
                    page_content=item["content"],
                    metadata={
                        "source": item.get("title") or item.get("url") or "Web result",
                        "title": item.get("title"),
                        "url": item.get("url"),
                    },
                )
            )
    if not documents:
        raise RuntimeError("Web search returned no usable results")
    return {"route": "search", "reranked_documents": documents}


async def generate(state: State) -> dict:
    documents = state.get("reranked_documents", [])
    evidence = _format_documents(documents, max_chars=14000)
    prompt = PromptTemplate.from_template(config.prompt("generate_prompt"))
    retry_instruction = ""
    if state.get("verification_attempts", 0):
        retry_instruction = (
            "Previous verification failed. Remove every claim not explicitly supported."
        )
    result = await (prompt | get_llm()).ainvoke(
        {
            "question": state["messages"][-1].content,
            "history": _history(state["messages"][:-1]),
            "context": evidence,
            "retry_instruction": retry_instruction,
        }
    )
    return {
        "answer": result.content,
        "sources": _sources(documents),
        "verification_context": evidence,
    }


async def verify(state: State) -> dict:
    prompt = PromptTemplate.from_template(config.prompt("verify_prompt"))
    verifier = get_structured_llm(VerificationResult)
    result = await (prompt | verifier).ainvoke(
        {
            "question": state["messages"][-1].content,
            "context": state["verification_context"],
            "final_answer": state["answer"],
        }
    )
    attempts = state.get("verification_attempts", 0) + (0 if result.faithful else 1)
    updates: dict = {"faithful": result.faithful, "verification_attempts": attempts}
    if result.faithful:
        updates["messages"] = [AIMessage(content=state["answer"])]
    return updates


async def safe_fallback(state: State) -> dict:
    answer = "I could not produce an answer that was fully supported by the available sources."
    return {"answer": answer, "sources": [], "messages": [AIMessage(content=answer)]}


graph = StateGraph(State)
graph.add_node("query_analysis", query_classifier)
graph.add_node("general_llm", general_llm)
graph.add_node("retriever", retriever_node)
graph.add_node("rerank", rerank)
graph.add_node("rewrite", rewrite_query)
graph.add_node("web_search", web_search)
graph.add_node("generate", generate)
graph.add_node("verify", verify)
graph.add_node("safe_fallback", safe_fallback)

graph.add_edge(START, "query_analysis")
graph.add_conditional_edges("query_analysis", routing_tool)
graph.add_conditional_edges("rerank", retrieval_decision)
graph.add_edge("rewrite", "retriever")
graph.add_edge("retriever", "rerank")
graph.add_edge("web_search", "generate")
graph.add_edge("generate", "verify")
graph.add_conditional_edges("verify", verification_decision)
graph.add_edge("general_llm", END)
graph.add_edge("safe_fallback", END)

builder = graph.compile()
