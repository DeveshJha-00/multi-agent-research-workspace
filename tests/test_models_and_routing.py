import pytest
from langchain_core.documents import Document
from pydantic import ValidationError

from src.models.query_request import QueryRequest
from src.models.route_identifier import RouteIdentifier
from src.tools.graph_tools import generation_decision, retrieval_decision, routing_tool


def test_query_request_strips_text_and_validates_session():
    request = QueryRequest(query="  hello  ", session_id="session-123")
    assert request.query == "hello"
    assert request.query_language == "auto"
    assert request.answer_language == "auto"
    with pytest.raises(ValidationError):
        QueryRequest(query="hello", session_id="bad id")


def test_route_is_strict_literal():
    with pytest.raises(ValidationError):
        RouteIdentifier(route="anything", reason="invalid")


def test_route_reason_allows_verbose_model_explanations():
    result = RouteIdentifier(route="general", reason="x" * 1000)
    assert result.reason == "x" * 1000


@pytest.mark.parametrize(
    ("route", "expected"),
    [("index", "rerank"), ("general", "general_llm"), ("search", "web_search")],
)
def test_primary_routing(route, expected):
    assert routing_tool({"route": route}) == expected


def test_retrieval_decision_generates_for_relevant_documents():
    document = Document(page_content="answer", metadata={"rerank_score": 0.95})
    assert retrieval_decision({"reranked_documents": [document], "retry_count": 0}) == "generate"


def test_retrieval_decision_uses_documents_before_rewrite_or_search():
    document = Document(page_content="answer", metadata={"rerank_score": 0.2})
    assert retrieval_decision({"reranked_documents": [document], "retry_count": 0}) == "generate"


def test_retrieval_decision_still_uses_extremely_weak_document_evidence():
    document = Document(page_content="answer", metadata={"rerank_score": 0.0})
    assert retrieval_decision({"reranked_documents": [document], "retry_count": 0}) == "generate"


def test_retrieval_decision_hybridizes_when_classifier_requested_search():
    document = Document(page_content="answer", metadata={"rerank_score": 0.95})
    assert (
        retrieval_decision(
            {
                "classifier_route": "search",
                "reranked_documents": [document],
                "retry_count": 0,
            }
        )
        == "web_search"
    )


def test_retrieval_decision_is_bounded():
    assert retrieval_decision({"reranked_documents": [], "retry_count": 0}) == "rewrite"
    assert retrieval_decision({"reranked_documents": [], "retry_count": 1}) == "web_search"


def test_index_generation_skips_extra_verification_call():
    assert generation_decision({"route": "index"}) == "__end__"
    assert generation_decision({"route": "search"}) == "verify"
