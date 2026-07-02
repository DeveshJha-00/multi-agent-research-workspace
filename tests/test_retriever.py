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

    monkeypatch.setattr(retriever_setup, "get_embeddings", lambda: FakeEmbeddings())
    monkeypatch.setattr(retriever_setup.client, "query_points", fake_query_points)
    documents = await retriever_setup.retrieve_documents("question", session_id="session-123")

    condition = captured["query_filter"].must[0]
    assert condition.key == "session_id"
    assert condition.match.value == "session-123"
    assert documents[0].metadata["vector_score"] == 0.9


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
