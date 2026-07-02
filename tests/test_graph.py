from src.rag.graph_builder import builder


def test_graph_contains_reranking_and_verification():
    nodes = set(builder.get_graph().nodes)
    assert {"query_analysis", "rerank", "generate", "verify", "safe_fallback"} <= nodes
