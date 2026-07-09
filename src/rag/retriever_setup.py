"""Persistent, session-partitioned Qdrant document storage and retrieval.

FAISS was previously used here. It is intentionally disabled because an in-process
index is not durable and diverges across API workers. Qdrant is the only active
vector backend.
"""

import logging
import re
from collections.abc import Sequence
from uuid import uuid4

from langchain_core.documents import Document
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from src.core.config import settings
from src.llms.provider import get_embeddings
from src.rag.document_parsers import is_binary_like_text, text_quality_score

logger = logging.getLogger(__name__)
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "about",
    "document",
    "documents",
    "file",
    "is",
    "it",
    "of",
    "the",
    "this",
    "to",
    "uploaded",
    "what",
}

client = AsyncQdrantClient(
    url=settings.qdrant_url,
    api_key=settings.qdrant_api_key,
    timeout=settings.qdrant_timeout_seconds,
)


async def initialize_qdrant() -> None:
    """Create the collection and filter indexes when absent."""
    exists = await client.collection_exists(settings.qdrant_collection)
    if not exists:
        try:
            await client.create_collection(
                collection_name=settings.qdrant_collection,
                vectors_config=models.VectorParams(
                    size=settings.embedding_dimensions,
                    distance=models.Distance.COSINE,
                ),
            )
            logger.info("Created Qdrant collection %s", settings.qdrant_collection)
        except UnexpectedResponse as exc:
            if exc.status_code != 409:
                raise

    info = await client.get_collection(settings.qdrant_collection)
    vectors = info.config.params.vectors
    actual_size = getattr(vectors, "size", None)
    if actual_size is not None and actual_size != settings.embedding_dimensions:
        raise RuntimeError(
            f"Qdrant collection vector size is {actual_size}, expected "
            f"{settings.embedding_dimensions}. Use a new QDRANT_COLLECTION after changing models."
        )

    for field_name in ("session_id", "document_id"):
        try:
            await client.create_payload_index(
                collection_name=settings.qdrant_collection,
                field_name=field_name,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except UnexpectedResponse as exc:
            if exc.status_code != 409:
                raise


async def qdrant_ready() -> bool:
    try:
        return await client.collection_exists(settings.qdrant_collection)
    except Exception:
        return False


async def close_qdrant() -> None:
    await client.close()


async def index_documents(
    chunks: Sequence[Document],
    *,
    session_id: str,
    document_id: str,
) -> int:
    """Embed and upsert chunks without replacing existing documents."""
    if not chunks:
        return 0

    embeddings = get_embeddings()
    indexed = 0
    for start in range(0, len(chunks), settings.embedding_batch_size):
        batch = list(chunks[start : start + settings.embedding_batch_size])
        vectors = await embeddings.aembed_documents([doc.page_content for doc in batch])
        points = []
        for offset, (doc, vector) in enumerate(zip(batch, vectors, strict=True)):
            metadata = dict(doc.metadata)
            payload = {
                "content": doc.page_content,
                "session_id": session_id,
                "document_id": document_id,
                "chunk_index": start + offset,
                **metadata,
            }
            points.append(models.PointStruct(id=str(uuid4()), vector=vector, payload=payload))

        await client.upsert(
            collection_name=settings.qdrant_collection,
            points=points,
            wait=True,
        )
        indexed += len(points)
    return indexed


def _document_from_payload(payload: dict, *, score: float, score_key: str) -> Document | None:
    payload = dict(payload)
    content = str(payload.pop("content", ""))
    if not content:
        return None
    if bool(payload.get("binary_like")) or is_binary_like_text(content):
        return None
    if float(payload.get("extraction_quality", text_quality_score(content))) < 0.05:
        return None
    payload[score_key] = score
    payload.setdefault("vector_score", score if score_key == "vector_score" else 0.0)
    return Document(page_content=content, metadata=payload)


def _tokenize(value: str) -> set[str]:
    tokens = {
        token
        for token in re.split(r"[^a-z0-9]+", value.lower().replace("_", " "))
        if len(token) > 1 and token not in STOPWORDS
    }
    if "hindi" in tokens:
        tokens.add("hi")
    return tokens


def _lexical_score(query_tokens: set[str], payload: dict) -> float:
    haystack = " ".join(
        str(payload.get(key, ""))
        for key in (
            "source",
            "description",
            "detected_language",
            "script",
            "parser_provider",
            "content",
        )
    )
    tokens = _tokenize(haystack)
    if not query_tokens or not tokens:
        return 0.0
    overlap = query_tokens & tokens
    score = len(overlap) / len(query_tokens)
    source_text = f"{payload.get('source', '')} {payload.get('description', '')}".lower()
    if overlap and any(token in source_text for token in overlap):
        score += 0.25
    return min(1.0, score)


def _document_score(document: Document) -> float:
    metadata = document.metadata
    return max(
        float(metadata.get("metadata_match_score", 0.0)),
        float(metadata.get("vector_score", 0.0)),
        float(metadata.get("document_candidate_score", 0.0)),
    )


async def _metadata_matches(query: str, *, session_id: str, limit: int) -> list[Document]:
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []
    records, _ = await client.scroll(
        collection_name=settings.qdrant_collection,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="session_id",
                    match=models.MatchValue(value=session_id),
                )
            ]
        ),
        limit=max(limit * 4, settings.retrieval_top_k),
        with_payload=True,
        with_vectors=False,
    )
    scored: list[tuple[float, Document]] = []
    for record in records:
        payload = dict(record.payload or {})
        score = _lexical_score(query_tokens, payload)
        if score <= 0:
            continue
        document = _document_from_payload(
            payload,
            score=score,
            score_key="metadata_match_score",
        )
        if document:
            scored.append((score, document))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [document for _, document in scored[:limit]]


async def _session_document_ids(*, session_id: str, limit: int = 200) -> list[str]:
    records, _ = await client.scroll(
        collection_name=settings.qdrant_collection,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="session_id",
                    match=models.MatchValue(value=session_id),
                )
            ]
        ),
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    document_ids: list[str] = []
    seen: set[str] = set()
    for record in records:
        payload = dict(record.payload or {})
        document_id = str(payload.get("document_id") or "")
        if document_id and document_id not in seen:
            seen.add(document_id)
            document_ids.append(document_id)
    return document_ids


async def _per_document_candidates(
    query_vector: list[float],
    *,
    query: str,
    session_id: str,
    limit: int,
    minimum_score: float,
) -> list[Document]:
    document_ids = await _session_document_ids(session_id=session_id)
    if len(document_ids) <= 1:
        return []

    candidates: list[Document] = []
    query_tokens = _tokenize(query)
    per_document_limit = 1 if len(document_ids) >= limit else 2
    for document_id in document_ids[:limit]:
        result = await client.query_points(
            collection_name=settings.qdrant_collection,
            query=query_vector,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="session_id",
                        match=models.MatchValue(value=session_id),
                    ),
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id),
                    ),
                ]
            ),
            limit=per_document_limit,
            with_payload=True,
            with_vectors=False,
        )
        for point in result.points[:per_document_limit]:
            document = _document_from_payload(
                dict(point.payload or {}),
                score=float(point.score),
                score_key="document_candidate_score",
            )
            if document:
                lexical_score = _lexical_score(
                    query_tokens,
                    {
                        **document.metadata,
                        "content": document.page_content,
                    },
                )
                candidate_score = float(document.metadata.get("document_candidate_score", 0.0))
                if lexical_score <= 0 and candidate_score < max(minimum_score, 0.75):
                    continue
                if lexical_score > 0:
                    document.metadata["metadata_match_score"] = max(
                        float(document.metadata.get("metadata_match_score", 0.0)),
                        lexical_score,
                    )
                candidates.append(document)
    return candidates[:limit]


def _merge_documents(primary: list[Document], supplemental: list[Document], limit: int) -> list[Document]:
    merged: list[Document] = []
    seen: set[tuple[str, int | str]] = set()
    ordered = sorted([*primary, *supplemental], key=_document_score, reverse=True)
    for document in ordered:
        key = (
            str(document.metadata.get("document_id") or document.metadata.get("source") or ""),
            document.metadata.get("chunk_index", document.page_content[:80]),
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(document)
        if len(merged) >= limit:
            break
    selected_documents = {
        str(document.metadata.get("document_id") or document.metadata.get("source") or "")
        for document in merged
    }
    for document in sorted(supplemental, key=_document_score, reverse=True):
        document_key = str(document.metadata.get("document_id") or document.metadata.get("source") or "")
        if not document_key or document_key in selected_documents:
            continue
        if _document_score(document) < 0.5:
            continue
        if len(merged) < limit:
            merged.append(document)
        elif merged:
            merged[-1] = document
        selected_documents.add(document_key)
    return merged


async def retrieve_documents(
    query: str,
    *,
    session_id: str,
    limit: int | None = None,
) -> list[Document]:
    """Retrieve semantically similar chunks restricted to one session."""
    query_vector = await get_embeddings().aembed_query(query)
    result = await client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="session_id",
                    match=models.MatchValue(value=session_id),
                )
            ]
        ),
        limit=limit or settings.retrieval_top_k,
        score_threshold=settings.retrieval_score_threshold,
        with_payload=True,
        with_vectors=False,
    )

    documents = []
    for point in result.points:
        document = _document_from_payload(
            dict(point.payload or {}),
            score=float(point.score),
            score_key="vector_score",
        )
        if document:
            documents.append(document)
    try:
        metadata_matches = await _metadata_matches(
            query,
            session_id=session_id,
            limit=limit or settings.retrieval_top_k,
        )
    except Exception:
        logger.exception("metadata_match_retrieval_failed session_id=%s", session_id)
        metadata_matches = []
    if documents or metadata_matches:
        try:
            best_existing_score = max(
                [_document_score(document) for document in [*documents, *metadata_matches]],
                default=0.0,
            )
            document_candidates = await _per_document_candidates(
                query_vector,
                query=query,
                session_id=session_id,
                limit=limit or settings.retrieval_top_k,
                minimum_score=max(0.5, best_existing_score * 0.85),
            )
        except Exception:
            logger.exception("per_document_candidate_retrieval_failed session_id=%s", session_id)
            document_candidates = []
    else:
        document_candidates = []
    return _merge_documents(
        documents,
        [*metadata_matches, *document_candidates],
        limit or settings.retrieval_top_k,
    )


async def delete_document(*, session_id: str, document_id: str) -> None:
    """Delete one document while preventing cross-session deletion."""
    await client.delete(
        collection_name=settings.qdrant_collection,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="session_id", match=models.MatchValue(value=session_id)
                    ),
                    models.FieldCondition(
                        key="document_id", match=models.MatchValue(value=document_id)
                    ),
                ]
            )
        ),
        wait=True,
    )


async def list_documents(*, session_id: str, limit: int = 200) -> list[dict]:
    """List indexed documents for one session from Qdrant payloads."""
    records, _ = await client.scroll(
        collection_name=settings.qdrant_collection,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="session_id",
                    match=models.MatchValue(value=session_id),
                )
            ]
        ),
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    documents: dict[str, dict] = {}
    for record in records:
        payload = dict(record.payload or {})
        document_id = str(payload.get("document_id") or payload.get("source") or "unknown")
        item = documents.setdefault(
            document_id,
            {
                "document_id": document_id,
                "filename": str(payload.get("source") or "Unknown source"),
                "description": str(payload.get("description") or ""),
                "chunks_indexed": 0,
                "parser_provider": str(payload.get("parser_provider") or "unknown"),
                "detected_language": str(payload.get("detected_language") or "unknown"),
                "script": str(payload.get("script") or "unknown"),
            },
        )
        item["chunks_indexed"] += 1
    return sorted(documents.values(), key=lambda item: item["filename"])


# Compatibility shim for older imports. New code must use the async functions above.
def get_retriever():
    raise RuntimeError("The FAISS/LangChain retriever was removed; use retrieve_documents().")
