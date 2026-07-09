from io import BytesIO

import pytest
from fastapi import HTTPException, UploadFile

from src.rag import document_upload


@pytest.mark.asyncio
async def test_txt_upload_adds_metadata_and_indexes(monkeypatch):
    captured = {}

    async def fake_index(chunks, *, session_id, document_id):
        captured["chunks"] = chunks
        captured["session_id"] = session_id
        captured["document_id"] = document_id
        return len(chunks)

    monkeypatch.setattr(document_upload, "index_documents", fake_index)
    upload = UploadFile(filename="notes.TXT", file=BytesIO(b"Adaptive RAG test content."))
    result = await document_upload.documents("Test notes", upload, "session-123")

    assert result["status"] is True
    assert result["chunks_indexed"] == 1
    assert result["parser_provider"] == "local"
    assert result["detected_language"] == "en-IN"
    assert captured["session_id"] == "session-123"
    assert captured["chunks"][0].metadata["source"] == "notes.TXT"
    assert captured["chunks"][0].metadata["parser_provider"] == "local"


@pytest.mark.asyncio
async def test_document_description_is_optional_and_defaults_from_filename(monkeypatch):
    captured = {}

    async def fake_index(chunks, *, session_id, document_id):
        captured["chunks"] = chunks
        return len(chunks)

    monkeypatch.setattr(document_upload, "index_documents", fake_index)
    upload = UploadFile(filename="Resume_Devesh_Jha.txt", file=BytesIO(b"Candidate resume text."))
    result = await document_upload.documents("", upload, "session-123")

    assert result["status"] is True
    assert captured["chunks"][0].metadata["description"] == "Resume Devesh Jha"
    assert "Document description: Resume Devesh Jha" in captured["chunks"][0].page_content


@pytest.mark.asyncio
async def test_unsupported_upload_is_rejected():
    upload = UploadFile(filename="payload.exe", file=BytesIO(b"bad"))
    with pytest.raises(HTTPException) as error:
        await document_upload.documents("Executable", upload, "session-123")
    assert error.value.status_code == 400
