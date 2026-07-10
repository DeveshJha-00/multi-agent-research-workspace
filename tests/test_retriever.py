from types import SimpleNamespace

import pytest
from langchain_core.documents import Document
from qdrant_client import AsyncQdrantClient

from src.rag import retriever_setup


class FakeEmbeddings:
    async def aembed_documents(self, values):
        return [[0.1] * 1536 for _ in values]

    async def aembed_query(self, value):
        return [0.2] * 1536


class SmallFakeEmbeddings:
    async def aembed_documents(self, values):
        return [[1.0, 0.0, 0.0] for _ in values]

    async def aembed_query(self, value):
        return [1.0, 0.0, 0.0]


@pytest.mark.asyncio
async def test_indexing_upserts_without_deleting(monkeypatch):
    calls = []

    async def fake_upsert(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(retriever_setup, "get_embeddings", lambda: FakeEmbeddings())
    monkeypatch.setattr(retriever_setup.client, "upsert", fake_upsert)

    count = await retriever_setup.index_documents(
        [Document(page_content="one", metadata={"source": "test.txt"})],
        session_id="session-123",
        document_id="document-123",
    )

    assert count == 1
    assert len(calls) == 1
    assert calls[0]["points"][0].payload["session_id"] == "session-123"


@pytest.mark.asyncio
async def test_retrieval_filters_by_session(monkeypatch):
    captured = {}

    async def fake_query_points(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            points=[
                SimpleNamespace(
                    payload={"content": "found", "source": "test.txt"},
                    score=0.9,
                )
            ]
        )

    async def fake_scroll(**kwargs):
        return [], None

    monkeypatch.setattr(retriever_setup, "get_embeddings", lambda: FakeEmbeddings())
    monkeypatch.setattr(retriever_setup.client, "query_points", fake_query_points)
    monkeypatch.setattr(retriever_setup.client, "scroll", fake_scroll)
    documents = await retriever_setup.retrieve_documents("question", session_id="session-123")

    condition = captured["query_filter"].must[0]
    assert condition.key == "session_id"
    assert condition.match.value == "session-123"
    assert documents[0].metadata["vector_score"] == 0.9


@pytest.mark.asyncio
async def test_retrieval_adds_metadata_matches_for_named_documents(monkeypatch):
    async def fake_query_points(**kwargs):
        return SimpleNamespace(points=[])

    async def fake_scroll(**kwargs):
        return [
            SimpleNamespace(
                payload={
                    "content": "This is a Hindi policy document.",
                    "source": "hindi_doc.pdf",
                    "description": "hindi_document",
                    "document_id": "document-hindi",
                    "chunk_index": 0,
                }
            )
        ], None

    monkeypatch.setattr(retriever_setup, "get_embeddings", lambda: FakeEmbeddings())
    monkeypatch.setattr(retriever_setup.client, "query_points", fake_query_points)
    monkeypatch.setattr(retriever_setup.client, "scroll", fake_scroll)

    documents = await retriever_setup.retrieve_documents(
        "what is the uploaded hindi document about?",
        session_id="session-123",
    )
    assert documents[0].metadata["source"] == "hindi_doc.pdf"
    assert documents[0].metadata["metadata_match_score"] > 0


@pytest.mark.asyncio
async def test_retrieval_keeps_relevant_candidates_from_multiple_documents(monkeypatch):
    async def fake_query_points(**kwargs):
        must = kwargs["query_filter"].must
        document_filter = next(
            (condition for condition in must if condition.key == "document_id"),
            None,
        )
        if document_filter and document_filter.match.value == "document-hindi":
            return SimpleNamespace(
                points=[
                    SimpleNamespace(
                        payload={
                            "content": "hindi document agriculture policy details",
                            "source": "hindi_doc.pdf",
                            "description": "hindi_document",
                            "document_id": "document-hindi",
                            "chunk_index": 0,
                        },
                        score=0.7,
                    )
                ]
            )
        return SimpleNamespace(
            points=[
                SimpleNamespace(
                    payload={
                        "content": "resume skills education projects",
                        "source": "resume.pdf",
                        "description": "candidate resume",
                        "document_id": "document-resume",
                        "chunk_index": index,
                    },
                    score=0.8 - index * 0.01,
                )
                for index in range(6)
            ]
        )

    async def fake_scroll(**kwargs):
        return [
            SimpleNamespace(payload={"document_id": "document-resume"}),
            SimpleNamespace(payload={"document_id": "document-hindi"}),
        ], None

    monkeypatch.setattr(retriever_setup, "get_embeddings", lambda: FakeEmbeddings())
    monkeypatch.setattr(retriever_setup.client, "query_points", fake_query_points)
    monkeypatch.setattr(retriever_setup.client, "scroll", fake_scroll)

    documents = await retriever_setup.retrieve_documents(
        "summarize the agriculture policy",
        session_id="session-123",
        limit=6,
    )
    assert {document.metadata["document_id"] for document in documents} == {
        "document-resume",
        "document-hindi",
    }


@pytest.mark.asyncio
async def test_retrieval_filters_binary_and_prioritizes_relevant_resume_chunks(monkeypatch):
    async def fake_query_points(**kwargs):
        must = kwargs["query_filter"].must
        document_filter = next(
            (condition for condition in must if condition.key == "document_id"),
            None,
        )
        if document_filter and document_filter.match.value == "document-hindi":
            return SimpleNamespace(
                points=[
                    SimpleNamespace(
                        payload={
                            "content": "PK\x03\x04\ufffd\ufffd\ufffd\x00\x01garbage",
                            "source": "hindi_doc.pdf",
                            "description": "hindi document",
                            "document_id": "document-hindi",
                            "chunk_index": 0,
                        },
                        score=0.7,
                    )
                ]
            )
        return SimpleNamespace(
            points=[
                SimpleNamespace(
                    payload={
                        "content": (
                            "TECHNICAL SKILLS AND COURSEWORK\n"
                            "Languages: Java, Python, JavaScript\n"
                            "Databases: MongoDB, Redis"
                        ),
                        "source": "resume.pdf",
                        "description": "candidate resume",
                        "document_id": "document-resume",
                        "chunk_index": 0,
                        "extraction_quality": 0.9,
                    },
                    score=0.8,
                )
            ]
        )

    async def fake_scroll(**kwargs):
        return [
            SimpleNamespace(
                payload={
                    "content": (
                        "TECHNICAL SKILLS AND COURSEWORK\n"
                        "Languages: Java, Python, JavaScript\n"
                        "Databases: MongoDB, Redis"
                    ),
                    "source": "resume.pdf",
                    "description": "candidate resume",
                    "document_id": "document-resume",
                    "chunk_index": 0,
                    "extraction_quality": 0.9,
                }
            ),
            SimpleNamespace(
                payload={
                    "content": "PK\x03\x04\ufffd\ufffd\ufffd\x00\x01garbage",
                    "source": "hindi_doc.pdf",
                    "description": "hindi document",
                    "document_id": "document-hindi",
                    "chunk_index": 0,
                }
            ),
        ], None

    monkeypatch.setattr(retriever_setup, "get_embeddings", lambda: FakeEmbeddings())
    monkeypatch.setattr(retriever_setup.client, "query_points", fake_query_points)
    monkeypatch.setattr(retriever_setup.client, "scroll", fake_scroll)

    documents = await retriever_setup.retrieve_documents(
        "List the technical skills the candidate has",
        session_id="session-123",
        limit=6,
    )

    assert documents
    assert documents[0].metadata["document_id"] == "document-resume"
    assert "TECHNICAL SKILLS" in documents[0].page_content
    assert all(document.metadata["document_id"] != "document-hindi" for document in documents)


@pytest.mark.asyncio
async def test_list_documents_groups_chunks_by_document(monkeypatch):
    async def fake_scroll(**kwargs):
        return [
            SimpleNamespace(
                payload={
                    "source": "hindi_doc.pdf",
                    "description": "hindi_document",
                    "document_id": "document-hindi",
                    "chunk_index": 0,
                    "parser_provider": "local",
                    "detected_language": "hi-IN",
                    "script": "Deva",
                }
            ),
            SimpleNamespace(
                payload={
                    "source": "hindi_doc.pdf",
                    "description": "hindi_document",
                    "document_id": "document-hindi",
                    "chunk_index": 1,
                    "parser_provider": "local",
                    "detected_language": "hi-IN",
                    "script": "Deva",
                }
            ),
        ], None

    monkeypatch.setattr(retriever_setup.client, "scroll", fake_scroll)
    documents = await retriever_setup.list_documents(session_id="session-123")
    assert documents == [
        {
            "document_id": "document-hindi",
            "filename": "hindi_doc.pdf",
            "description": "hindi_document",
            "chunks_indexed": 2,
            "parser_provider": "local",
            "detected_language": "hi-IN",
            "script": "Deva",
        }
    ]


@pytest.mark.asyncio
async def test_document_like_query_uses_representative_chunks_when_vector_search_misses(monkeypatch):
    async def fake_query_points(**kwargs):
        return SimpleNamespace(points=[])

    async def fake_scroll(**kwargs):
        return [
            SimpleNamespace(
                payload={
                    "content": "प्रश्न पत्र\nसमय: 90 मिनट\nखंड अ और खंड ब",
                    "source": "hindi_doc.pdf",
                    "document_id": "document-hindi",
                    "chunk_index": 0,
                    "session_id": "session-123",
                }
            )
        ], None

    monkeypatch.setattr(retriever_setup, "get_embeddings", lambda: FakeEmbeddings())
    monkeypatch.setattr(retriever_setup.client, "query_points", fake_query_points)
    monkeypatch.setattr(retriever_setup.client, "scroll", fake_scroll)

    documents = await retriever_setup.retrieve_documents(
        "How many sections does the exam have?",
        session_id="session-123",
    )

    assert documents
    assert documents[0].metadata["document_id"] == "document-hindi"
    assert documents[0].metadata["retrieval_fallback"] == "representative_uploaded_chunk"


@pytest.mark.asyncio
async def test_real_qdrant_local_collection_round_trip(monkeypatch):
    local_client = AsyncQdrantClient(":memory:")
    monkeypatch.setattr(retriever_setup, "client", local_client)
    monkeypatch.setattr(retriever_setup.settings, "qdrant_collection", "test_documents")
    monkeypatch.setattr(retriever_setup.settings, "embedding_dimensions", 3)
    monkeypatch.setattr(retriever_setup, "get_embeddings", lambda: SmallFakeEmbeddings())

    await retriever_setup.initialize_qdrant()
    await retriever_setup.index_documents(
        [Document(page_content="private session content", metadata={"source": "notes.txt"})],
        session_id="session-one",
        document_id="document-one",
    )

    matching = await retriever_setup.retrieve_documents("content", session_id="session-one")
    isolated = await retriever_setup.retrieve_documents("content", session_id="session-two")
    assert matching[0].page_content == "private session content"
    assert isolated == []
    await local_client.close()
