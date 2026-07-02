"""Validated PDF/TXT ingestion into Qdrant."""

import asyncio
import os
import tempfile
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException, UploadFile
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.config import settings
from src.rag.retriever_setup import index_documents

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


def _load_and_split(path: str, extension: str):
    loader = PyPDFLoader(path) if extension == ".pdf" else TextLoader(path, encoding="utf-8")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True,
    )
    return splitter.split_documents(documents)


async def documents(description: str, file: UploadFile, session_id: str) -> dict:
    """Validate, chunk, enrich with metadata, and persist an uploaded document."""
    filename = Path(file.filename or "").name
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported")
    if not description.strip():
        raise HTTPException(status_code=400, detail="Document description is required")

    content = await file.read(settings.max_upload_bytes + 1)
    await file.close()
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty")
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {settings.max_upload_bytes // (1024 * 1024)} MB limit",
        )

    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name
        chunks = await asyncio.to_thread(_load_and_split, tmp_path, extension)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Unable to parse document: {exc}") from exc
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    if not chunks:
        raise HTTPException(status_code=422, detail="No readable text was found in the document")

    document_id = str(uuid4())
    for chunk in chunks:
        chunk.metadata.update(
            {
                "source": filename,
                "description": description.strip(),
                "document_id": document_id,
            }
        )
    count = await index_documents(
        chunks,
        session_id=session_id,
        document_id=document_id,
    )
    return {
        "status": True,
        "document_id": document_id,
        "filename": filename,
        "chunks_indexed": count,
    }
