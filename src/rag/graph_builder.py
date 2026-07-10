"""Adaptive retrieval graph with explicit reranking and answer verification."""

import logging
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
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
from src.tools.graph_tools import (
    generation_decision,
    retrieval_decision,
    routing_tool,
    verification_decision,
)

logger = logging.getLogger(__name__)
config = Config()


def _history(messages: list[BaseMessage], limit: int = 10) -> str:
    selected = messages[-limit:]
    return "\n".join(f"{message.type}: {message.content}" for message in selected)


def _session_instructions(messages: list[BaseMessage], limit: int | None = None) -> str:
    """Extract user-provided session preferences/instructions from recent turns."""
    selected = messages[-(limit or settings.max_history_messages) :]
    instruction_terms = (
        "address me",
        "call me",
        "my name is",
        "i am ",
        "i'm ",
        "remember",
        "from now",
        "for future",
        "in future",
        "throughout",
        "always",
        "every response",
        "all your responses",
        "all responses",
        "respond",
        "reply",
        "answer me",
        "format",
        "tone",
        "style",
        "instruction",
        "follow",
        "use my",
        "don't",
        "do not",
    )
    lines = []
    used = 0
    for message in selected:
        if message.type != "human":
            continue
        content = str(message.content).strip()
        if not content:
            continue
        lowered = content.casefold()
        if not any(term in lowered for term in instruction_terms):
            continue
        line = f"- {content}"
        if used + len(line) > 2500 and lines:
            break
        lines.append(line)
        used += len(line)
    if not lines:
        return "No explicit session-level user preferences or instructions found."
    return "\n".join(lines)


def _conversation_memory_documents(messages: list[BaseMessage]) -> list[Document]:
    """Expose recent session chat as evidence for user-provided facts."""
    if not messages:
        return []
    lines = []
    used = 0
    selected = messages[-settings.max_history_messages:]
    for index, message in enumerate(reversed(selected)):
        if message.type != "human":
            continue
        content = str(message.content).strip()
        if not content:
            continue
        is_current = index == 0 and message is messages[-1]
        prefix = "Current user message" if is_current else "User previously said"
        line = f"{prefix}: {content}"
        if used + len(line) > 4000 and lines:
            break
        lines.append(line)
        used += len(line)
    if not lines:
        return []
    lines.reverse()
    return [
        Document(
            page_content="\n".join(lines),
            metadata={
                "source": "Recent conversation memory",
                "source_kind": "conversation_memory",
                "document_id": "session-memory",
                "rerank_score": 0.75,
                "metadata_match_score": 0.75,
            },
        )
    ]


async def _retrieve_with_memory(
    question: str,
    *,
    session_id: str,
    messages: list[BaseMessage],
) -> list[Document]:
    documents = await retrieve_documents(question, session_id=session_id)
    return _conversation_memory_documents(messages) + documents


def _language_instruction(answer_language: str | None, query_language: str | None = None) -> str:
    if not answer_language or answer_language == "auto":
        return "Answer in the current user's question language, not older conversation turns."
    return (
        f"The current query language is {query_language or answer_language}. "
        f"Answer strictly in {answer_language}, even if earlier conversation turns used a "
        "different language. Keep quoted source text, citations, code, URLs, names, "
        "and exact document phrases unchanged."
    )


def _behavior_instruction(messages: list[BaseMessage], answer_language: str | None, query_language: str | None = None) -> str:
    return (
        f"{_language_instruction(answer_language, query_language)}\n"
        "Session-level user preferences and instructions:\n"
        f"{_session_instructions(messages)}\n"
        "Obey these user preferences in the current answer unless they conflict with safety, "
        "the current user message, or higher-priority system/developer instructions. "
        "For example, if the user asked to be addressed by name, start relevant future answers "
        "with that name. If the user points out you missed an instruction, acknowledge it and "
        "follow it going forward."
    )


def _is_conversation_management_query(question: str) -> bool:
    """Detect questions mainly about the current chat/session behavior."""
    lowered = question.casefold()
    markers = (
        "address me",
        "call me",
        "my name",
        "remember me",
        "why did you",
        "why didn't you",
        "why didnt you",
        "previous answer",
        "previous query",
        "earlier answer",
        "earlier query",
        "last response",
        "future response",
        "future messages",
        "all your responses",
        "throughout this session",
    )
    return any(marker in lowered for marker in markers)


def _format_documents(documents: list[Document], max_chars: int = 4000) -> str:
    blocks = []
    used = 0
    for index, doc in enumerate(documents, 1):
        metadata = doc.metadata
        source_kind = metadata.get("source_kind")
        if not source_kind:
            source_kind = "web_search" if metadata.get("url") else "uploaded_document"
        if source_kind == "web_search":
            source_label = "Web search result"
        elif source_kind == "conversation_memory":
            source_label = "Conversation memory"
        else:
            source_label = "Uploaded document"
        source_name = metadata.get("source") or metadata.get("title") or "Unknown source"
        block = f"[{index}] {source_label} — {source_name}\n{doc.page_content.strip()}"
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
            "source_type": metadata.get("source_kind")
            or ("web_search" if metadata.get("url") else "uploaded_document"),
            "document_id": metadata.get("document_id"),
            "page": metadata.get("page"),
            "url": metadata.get("url"),
        }
        key = tuple(item.values())
        if key not in seen:
            seen.add(key)
            output.append(item)
    return output


def _evaluation_contexts(documents: list[Document]) -> list[dict]:
    """Return plain-text evidence snapshots; evaluators never load paths or URLs."""
    output = []
    used = 0
    for doc in documents[: settings.ragas_max_contexts]:
        remaining = settings.ragas_max_context_chars - used
        if remaining <= 0:
            break
        content = doc.page_content.strip()[:remaining]
        if not content:
            continue
        metadata = doc.metadata
        output.append(
            {
                "content": content,
                "source": str(metadata.get("source") or metadata.get("title") or "Unknown source"),
                "document_id": metadata.get("document_id"),
                "page": metadata.get("page"),
                "url": metadata.get("url"),
            }
        )
        used += len(content)
    return output


def _document_key(document: Document) -> str:
    metadata = document.metadata
    return str(metadata.get("document_id") or metadata.get("source") or "")


def _candidate_score(document: Document) -> float:
    metadata = document.metadata
    return max(
        float(metadata.get("rerank_score", 0.0)),
        float(metadata.get("metadata_match_score", 0.0)),
        float(metadata.get("document_candidate_score", 0.0)),
        float(metadata.get("vector_score", 0.0)),
    )


def _log_documents(stage: str, documents: list[Document], *, limit: int = 8) -> None:
    for index, document in enumerate(documents[:limit]):
        metadata = document.metadata
        scores = {
            key: metadata.get(key)
            for key in (
                "vector_score",
                "metadata_match_score",
                "document_candidate_score",
                "rerank_score",
                "extraction_quality",
            )
            if metadata.get(key) is not None
        }
        logger.info(
            "%s[%d] source=%s document_id=%s chunk=%s scores=%s preview=%r",
            stage,
            index,
            metadata.get("source") or metadata.get("title"),
            metadata.get("document_id"),
            metadata.get("chunk_index"),
            scores,
            document.page_content[:180].replace("\n", " "),
        )


def _diversify_ranked_documents(
    ranked: list[Document],
    candidates: list[Document],
    limit: int,
) -> list[Document]:
    if not ranked or limit <= 1:
        return ranked[:limit]
    candidate_keys = {_document_key(document) for document in candidates if _document_key(document)}
    if len(candidate_keys) <= 1:
        return ranked[:limit]

    selected = ranked[:limit]
    selected_keys = {_document_key(document) for document in selected if _document_key(document)}
    if candidate_keys <= selected_keys:
        return selected

    selected_ids = {id(document) for document in selected}
    additions = sorted(
        [
            document
            for document in candidates
            if _document_key(document)
            and _document_key(document) not in selected_keys
            and id(document) not in selected_ids
        ],
        key=_candidate_score,
        reverse=True,
    )
    for document in additions:
        if len(selected) < limit:
            selected.append(document)
        else:
            selected[-1] = document
        selected_keys.add(_document_key(document))
        selected_ids.add(id(document))
        if candidate_keys <= selected_keys:
            break
    return selected[:limit]


async def query_classifier(state: State) -> dict:
    question = state["messages"][-1].content
    documents = await _retrieve_with_memory(
        question,
        session_id=state["session_id"],
        messages=state["messages"],
    )
    _log_documents("retrieved", documents)
    has_non_memory_documents = any(
        doc.metadata.get("source_kind") != "conversation_memory" for doc in documents
    )
    prompt = PromptTemplate.from_template(config.prompt("classify_prompt"))
    classifier = get_structured_llm(RouteIdentifier)
    try:
        result = await (prompt | classifier).ainvoke(
            {
                "question": question,
                "history": _history(state["messages"][:-1]),
                "context": _format_documents(documents, max_chars=5000),
            }
        )
    except Exception:
        logger.exception("query_classifier_failed; using deterministic fallback route")
        fallback_route = "index" if has_non_memory_documents else "general"
        result = RouteIdentifier(
            route=fallback_route,
            reason="Classifier structured output failed; selected a safe fallback route.",
        )
    should_force_index = has_non_memory_documents and not _is_conversation_management_query(
        str(question)
    )
    effective_route = "index" if should_force_index and result.route != "index" else result.route
    logger.info(
        "query_routed route=%s effective_route=%s candidates=%d",
        result.route,
        effective_route,
        len(documents),
    )
    return {
        "route": effective_route,
        "classifier_route": result.route,
        "latest_query": question,
        "documents": documents,
        "retry_count": 0,
        "verification_attempts": 0,
    }


async def general_llm(state: State) -> dict:
    messages = [
        SystemMessage(
            content=_behavior_instruction(
                state["messages"],
                state.get("answer_language"),
                state.get("query_language"),
            )
        ),
        *state["messages"],
    ]
    result = await get_llm().ainvoke(messages)
    return {
        "messages": [result],
        "answer": result.content,
        "sources": [],
        "route": "general",
        "evaluation_contexts": [],
    }


async def retriever_node(state: State) -> dict:
    documents = await _retrieve_with_memory(
        state["latest_query"],
        session_id=state["session_id"],
        messages=state["messages"],
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
            doc.metadata.setdefault("source_kind", "uploaded_document")
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
            doc.metadata.setdefault("source_kind", "uploaded_document")
            doc.metadata["rerank_score"] = float(doc.metadata.get("vector_score", 0.0))
    diversified = _diversify_ranked_documents(
        ranked,
        candidates,
        settings.rerank_top_n,
    )
    _log_documents("reranked", diversified)
    return {
        "reranked_documents": diversified
    }


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
    uploaded_documents = [
        doc
        for doc in state.get("reranked_documents", [])
        if doc.metadata.get("source_kind") != "web_search"
    ]
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
                        "source_kind": "web_search",
                        "source": item.get("title") or item.get("url") or "Web result",
                        "title": item.get("title"),
                        "url": item.get("url"),
                    },
                )
            )
    if not documents:
        raise RuntimeError("Web search returned no usable results")
    if uploaded_documents:
        return {"route": "search", "reranked_documents": uploaded_documents + documents}
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
            "session_instructions": _session_instructions(state["messages"]),
            "context": evidence,
            "retry_instruction": retry_instruction,
            "answer_language": state.get("answer_language", "auto"),
            "query_language": state.get("query_language", "auto"),
        }
    )
    return {
        "messages": [AIMessage(content=result.content)],
        "answer": result.content,
        "sources": _sources(documents),
        "verification_context": evidence,
        "evaluation_contexts": _evaluation_contexts(documents),
    }


async def verify(state: State) -> dict:
    prompt = PromptTemplate.from_template(config.prompt("verify_prompt"))
    verifier = get_structured_llm(VerificationResult)
    try:
        result = await (prompt | verifier).ainvoke(
            {
                "question": state["messages"][-1].content,
                "context": state["verification_context"][:6000],
                "final_answer": state["answer"],
            }
        )
    except Exception:
        logger.exception("Answer verification failed; returning generated answer")
        return {
            "faithful": True,
            "messages": [AIMessage(content=state["answer"])],
        }
    attempts = state.get("verification_attempts", 0) + (0 if result.faithful else 1)
    updates: dict = {"faithful": result.faithful, "verification_attempts": attempts}
    if result.faithful:
        updates["messages"] = [AIMessage(content=state["answer"])]
    return updates


async def safe_fallback(state: State) -> dict:
    documents = state.get("reranked_documents", [])
    answer = state.get("answer") or (
        "I could not produce an answer that was fully supported by the available sources."
    )
    return {
        "answer": answer,
        "sources": _sources(documents),
        "verification_context": state.get("verification_context", ""),
        "evaluation_contexts": state.get("evaluation_contexts", _evaluation_contexts(documents)),
        "messages": [AIMessage(content=answer)],
    }


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
graph.add_conditional_edges("generate", generation_decision)
graph.add_conditional_edges("verify", verification_decision)
graph.add_edge("general_llm", END)
graph.add_edge("safe_fallback", END)

builder = graph.compile()
