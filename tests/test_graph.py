import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from src.rag.graph_builder import (
    _conversation_memory_documents,
    _format_documents,
    _is_conversation_management_query,
    _session_instructions,
    builder,
    safe_fallback,
)


def test_graph_contains_reranking_and_verification():
    nodes = set(builder.get_graph().nodes)
    assert {"query_analysis", "rerank", "generate", "verify", "safe_fallback"} <= nodes


@pytest.mark.asyncio
async def test_safe_fallback_preserves_retrieved_evidence():
    document = Document(
        page_content="Languages: Java, Python, JavaScript",
        metadata={"source": "resume.pdf", "document_id": "doc-1", "page": 0},
    )

    result = await safe_fallback(
        {
            "answer": "Generated answer",
            "reranked_documents": [document],
        }
    )

    assert result["answer"] == "Generated answer"
    assert result["sources"][0]["source"] == "resume.pdf"
    assert result["evaluation_contexts"][0]["content"] == "Languages: Java, Python, JavaScript"


def test_recent_user_turns_become_conversation_memory_evidence():
    messages = [
        HumanMessage(content="I am in class 9. Can I take this test?"),
        AIMessage(content="The document says class 10."),
        HumanMessage(content="Which class am I in?"),
    ]

    documents = _conversation_memory_documents(messages)

    assert len(documents) == 1
    assert documents[0].metadata["source_kind"] == "conversation_memory"
    assert "I am in class 9" in documents[0].page_content
    assert "Current user message: Which class am I in?" in documents[0].page_content
    assert "The document says class 10" not in documents[0].page_content


def test_conversation_memory_is_formatted_as_evidence():
    document = Document(
        page_content="User previously said: I am in class 9.",
        metadata={"source": "Recent conversation memory", "source_kind": "conversation_memory"},
    )

    formatted = _format_documents([document])

    assert "Conversation memory" in formatted
    assert "I am in class 9" in formatted


def test_session_instructions_capture_persistent_user_preferences():
    messages = [
        HumanMessage(content="I am Aryan. Address me by my name in all future responses."),
        AIMessage(content="Sure, Aryan."),
        HumanMessage(content="List the projects."),
    ]

    instructions = _session_instructions(messages)

    assert "I am Aryan" in instructions
    assert "Address me by my name" in instructions


def test_conversation_management_queries_do_not_force_index_route():
    assert _is_conversation_management_query("Why didn't you address me by my name?")
    assert _is_conversation_management_query("What is my name?")
    assert not _is_conversation_management_query("What is the candidate's name?")
