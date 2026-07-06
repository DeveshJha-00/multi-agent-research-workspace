"""Persistent, session-partitioned Qdrant document storage and retrieval.

FAISS was previously used here. It is intentionally disabled because an in-process
index is not durable and diverges across API workers. Qdrant is the only active
vector backend.
"""

import logging
from collections.abc import Sequence
from uuid import uuid4

from langchain_core.documents import Document
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from src.core.config import settings
from src.llms.provider import get_embeddings

logger = logging.getLogger(__name__)

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
        payload = dict(point.payload or {})
        content = str(payload.pop("content", ""))
        if not content:
            continue
        payload["vector_score"] = float(point.score)
        documents.append(Document(page_content=content, metadata=payload))
    return documents


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


# Compatibility shim for older imports. New code must use the async functions above.
def get_retriever():
    raise RuntimeError("The FAISS/LangChain retriever was removed; use retrieve_documents().")
