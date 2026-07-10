"""Validated PDF/TXT ingestion into Qdrant."""

import logging
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException, UploadFile

from src.core.config import settings
from src.rag.document_parsers import SUPPORTED_EXTENSIONS, choose_document_parser
from src.rag.retriever_setup import index_documents

logger = logging.getLogger(__name__)


async def documents(description: str, file: UploadFile, session_id: str) -> dict:
    """Validate, chunk, enrich with metadata, and persist an uploaded document."""
    filename = Path(file.filename or "").name
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported")
    normalized_description = description.strip() or Path(filename).stem.replace("_", " ").replace("-", " ")

    content = await file.read(settings.max_upload_bytes + 1)
    await file.close()
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty")
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {settings.max_upload_bytes // (1024 * 1024)} MB limit",
        )

    try:
        parser, parsed_preview, warnings = await choose_document_parser(
            content=content,
            filename=filename,
            extension=extension,
            description=normalized_description,
        )
        if parsed_preview and parser.provider == "sarvam":
            try:
                parsed = await parser.parse(
                    content=content,
                    filename=filename,
                    extension=extension,
                )
            except Exception as exc:
                logger.exception(
                    "sarvam_parse_failed_falling_back session_id=%s filename=%s",
                    session_id,
                    filename,
                )
                warnings.append(
                    "Sarvam document digitization failed or timed out, so this file was "
                    "indexed with best-effort local extraction. Results may be incomplete."
                )
                warnings.append(f"Sarvam parser error: {exc}")
                parsed = parsed_preview
        else:
            parsed = parsed_preview or await parser.parse(
                content=content,
                filename=filename,
                extension=extension,
            )
        chunks = parsed.chunks
        warnings.extend(parsed.warnings)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("document_parse_failed session_id=%s filename=%s", session_id, filename)
        raise HTTPException(status_code=422, detail=f"Unable to parse document: {exc}") from exc

    if not chunks:
        raise HTTPException(status_code=422, detail="No readable text was found in the document")

    document_id = str(uuid4())
    for chunk in chunks:
        chunk.metadata.update(
            {
                "source": filename,
                "description": normalized_description,
                "document_id": document_id,
            }
        )
        chunk.page_content = (
            f"Document filename: {filename}\n"
            f"Document description: {normalized_description}\n"
            f"Detected language: {parsed.detected_language}\n\n"
            f"{chunk.page_content}"
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
        "parser_provider": parsed.parser_provider,
        "detected_language": parsed.detected_language,
        "script": parsed.script,
        "warnings": warnings,
    }
