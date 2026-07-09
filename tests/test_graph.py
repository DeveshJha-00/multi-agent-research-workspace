import pytest
from langchain_core.documents import Document

from src.rag.graph_builder import builder, safe_fallback


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
