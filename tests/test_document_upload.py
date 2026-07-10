from io import BytesIO

import pytest
from fastapi import HTTPException, UploadFile
from langchain_core.documents import Document

from src.rag import document_upload
from src.rag.document_parsers import ParsedDocument


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
async def test_sarvam_failure_falls_back_to_local_preview(monkeypatch):
    captured = {}
    preview = ParsedDocument(
        chunks=[Document(page_content="स्थानीय निकाला गया पाठ")],
        parser_provider="local",
        detected_language="hi-IN",
        script="Deva",
    )

    class FailingSarvamParser:
        provider = "sarvam"

        async def parse(self, *, content, filename, extension):
            raise TimeoutError("Sarvam document parsing timed out")

    async def fake_choose(**kwargs):
        return FailingSarvamParser(), preview, ["Trying Sarvam first."]

    async def fake_index(chunks, *, session_id, document_id):
        captured["chunks"] = chunks
        return len(chunks)

    monkeypatch.setattr(document_upload, "choose_document_parser", fake_choose)
    monkeypatch.setattr(document_upload, "index_documents", fake_index)

    upload = UploadFile(filename="hindi_doc.pdf", file=BytesIO(b"%PDF"))
    result = await document_upload.documents("", upload, "session-123")

    assert result["status"] is True
    assert result["parser_provider"] == "local"
    assert result["chunks_indexed"] == 1
    assert "Sarvam document digitization failed" in result["warnings"][1]
    assert "स्थानीय" in captured["chunks"][0].page_content


@pytest.mark.asyncio
async def test_unsupported_upload_is_rejected():
    upload = UploadFile(filename="payload.exe", file=BytesIO(b"bad"))
    with pytest.raises(HTTPException) as error:
        await document_upload.documents("Executable", upload, "session-123")
    assert error.value.status_code == 400
