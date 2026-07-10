import pytest

from src.llms import provider


class FakeVector:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class FakeEmbeddingModel:
    def embed(self, texts, batch_size):
        return iter(FakeVector([float(index), 1.0]) for index, _ in enumerate(texts))

    def query_embed(self, text):
        return iter([FakeVector([0.5, 1.0])])


def test_structured_llm_uses_native_json_schema(monkeypatch):
    captured = {}

    class FakeChatModel:
        def with_structured_output(self, schema, method):
            captured.update(schema=schema, method=method)
            return "structured-model"

    monkeypatch.setattr(provider, "get_llm", lambda: FakeChatModel())
    assert provider.get_structured_llm(dict) == "structured-model"
    assert captured == {"schema": dict, "method": "json_schema"}


@pytest.mark.asyncio
async def test_fastembed_adapter_runs_sync_inference_asynchronously():
    embeddings = object.__new__(provider.FastEmbedEmbeddings)
    embeddings._model = FakeEmbeddingModel()

    assert await embeddings.aembed_documents(["one", "two"]) == [[0.0, 1.0], [1.0, 1.0]]
    assert await embeddings.aembed_query("question") == [0.5, 1.0]


@pytest.mark.asyncio
async def test_hash_embeddings_are_deterministic_and_normalized():
    embeddings = provider.HashEmbeddings(dimensions=16)

    first = await embeddings.aembed_query("Java Python MongoDB")
    second = await embeddings.aembed_query("Java Python MongoDB")
    different = await embeddings.aembed_query("Hindi exam eligibility")

    assert first == second
    assert first != different
    assert len(first) == 16
    assert sum(value * value for value in first) == pytest.approx(1.0)


def test_get_embeddings_can_use_low_memory_hash_provider(monkeypatch):
    provider.get_embeddings.cache_clear()
    monkeypatch.setattr(provider.settings, "embedding_provider", "hash")

    try:
        assert isinstance(provider.get_embeddings(), provider.HashEmbeddings)
    finally:
        provider.get_embeddings.cache_clear()


@pytest.mark.asyncio
async def test_reranker_preserves_original_passage_indexes(monkeypatch):
    class FakeRanker:
        def rerank(self, request):
            return [
                {"id": 1, "text": "relevant", "score": 0.91},
                {"id": 0, "text": "weak", "score": 0.12},
            ]

    monkeypatch.setattr(provider, "get_reranker", lambda: FakeRanker())
    assert await provider.rerank_passages("query", ["weak", "relevant"]) == [
        (1, 0.91),
        (0, 0.12),
    ]
