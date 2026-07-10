import time
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
from langchain_core.documents import Document

from src.rag import document_parsers
from src.rag.document_parsers import (
    ParsedDocument,
    _best_pdf_extraction,
    _download_markdown,
    _repair_pdf_text,
    choose_document_parser,
    is_binary_like_text,
    text_quality_score,
)
from src.services.language import LanguagePreferences, detect_text_language


def test_language_preferences_model_prepares_multilingual_sessions():
    preferences = LanguagePreferences(
        ui_language="hi-IN",
        query_language="auto",
        answer_language="hi-IN",
    )
    assert preferences.ui_language == "hi-IN"
    assert preferences.answer_language == "hi-IN"


def test_heuristic_language_detection_distinguishes_latin_and_indic():
    assert detect_text_language("Candidate resume and education").is_english_latin
    detection = detect_text_language("यह उम्मीदवार का रिज्यूमे है")
    assert detection.language_code == "hi-IN"
    assert detection.script_code == "Deva"


def test_pdf_text_repair_collapses_character_spaced_resume_text():
    repaired, changed = _repair_pdf_text(
        "D e v e s h  J h a\n"
        "M S  R a m a i a h  I n s t i t u t e  O f  T e c h n o l o g y\n"
        "T E C H N I C A L  S K I L L S  A N D  C O U R S E W O R K\n"
        "L a n g u a g e s  :  J a v a ,  P y t h o n ,  J a v a S c r i p t"
    )

    assert changed is True
    assert "Devesh Jha" in repaired
    assert "MS Ramaiah Institute Of Technology" in repaired
    assert "TECHNICAL SKILLS AND COURSEWORK" in repaired
    assert "Languages: Java, Python, JavaScript" in repaired


def test_text_quality_detects_binary_and_clean_resume_text():
    clean = "TECHNICAL SKILLS\nLanguages: Java, Python, JavaScript\nDatabases: MongoDB"
    binary = "PK\x03\x04\x00\x00\ufffd\ufffd\ufffd\x00\x01\x02"

    assert text_quality_score(clean) > 0.4
    assert is_binary_like_text(binary)


def test_best_pdf_extraction_selects_higher_quality_engine(monkeypatch):
    def fake_pymupdf(path, filename):
        return [
            Document(
                page_content=(
                    "TECHNICAL SKILLS\n"
                    "Languages: Java, Python, JavaScript\n"
                    "Databases: MongoDB, Redis"
                )
            )
        ]

    def fake_pypdf(path):
        return [Document(page_content="T E C H N I C A L  S K I L L S")]

    monkeypatch.setattr(document_parsers, "_load_pymupdf", fake_pymupdf)
    monkeypatch.setattr(document_parsers, "_load_pypdf", fake_pypdf)

    selected = _best_pdf_extraction("resume.pdf", "resume.pdf")

    assert selected.engine == "pymupdf"
    assert "JavaScript" in selected.documents[0].page_content


@pytest.mark.asyncio
async def test_auto_parser_uses_local_for_english_documents(monkeypatch):
    parsed = ParsedDocument(
        chunks=[Document(page_content="English resume text")],
        parser_provider="local",
        detected_language="en-IN",
        script="Latn",
    )

    async def fake_parse(self, *, content, filename, extension):
        return parsed

    monkeypatch.setattr(document_parsers.LocalDocumentParser, "parse", fake_parse)
    monkeypatch.setattr(document_parsers.settings, "document_parser_provider", "auto")
    parser, preview, warnings = await choose_document_parser(
        content=b"pdf",
        filename="resume.pdf",
        extension=".pdf",
    )
    assert parser.provider == "local"
    assert preview is parsed
    assert warnings == []


@pytest.mark.asyncio
async def test_auto_parser_routes_indic_documents_to_sarvam_when_configured(monkeypatch):
    parsed = ParsedDocument(
        chunks=[Document(page_content="हिंदी दस्तावेज")],
        parser_provider="local",
        detected_language="hi-IN",
        script="Deva",
    )

    async def fake_parse(self, *, content, filename, extension):
        return parsed

    monkeypatch.setattr(document_parsers.LocalDocumentParser, "parse", fake_parse)
    monkeypatch.setattr(document_parsers.settings, "document_parser_provider", "auto")
    monkeypatch.setattr(document_parsers.settings, "enable_multilingual_docs", True)
    monkeypatch.setattr(document_parsers.settings, "sarvam_api_key", "key")
    parser, preview, warnings = await choose_document_parser(
        content=b"pdf",
        filename="policy.pdf",
        extension=".pdf",
    )
    assert parser.provider == "sarvam"
    assert preview is parsed
    assert any("Trying Sarvam" in warning for warning in warnings)


@pytest.mark.asyncio
async def test_auto_parser_warns_and_falls_back_without_sarvam_key(monkeypatch):
    parsed = ParsedDocument(
        chunks=[Document(page_content="हिंदी दस्तावेज")],
        parser_provider="local",
        detected_language="hi-IN",
        script="Deva",
    )

    async def fake_parse(self, *, content, filename, extension):
        return parsed

    monkeypatch.setattr(document_parsers.LocalDocumentParser, "parse", fake_parse)
    monkeypatch.setattr(document_parsers.settings, "document_parser_provider", "auto")
    monkeypatch.setattr(document_parsers.settings, "enable_multilingual_docs", True)
    monkeypatch.setattr(document_parsers.settings, "sarvam_api_key", "")
    parser, preview, warnings = await choose_document_parser(
        content=b"pdf",
        filename="policy.pdf",
        extension=".pdf",
    )
    assert parser.provider == "local"
    assert preview is parsed
    assert any("Sarvam is not configured" in warning for warning in warnings)


@pytest.mark.asyncio
async def test_auto_parser_routes_garbled_non_english_pdf_to_sarvam(monkeypatch):
    parsed = ParsedDocument(
        chunks=[
            Document(
                page_content=(
                    "rtr\n"
                    "sfaaet vTYYT (2023-24)\n"
                    "f ( s sT 085\n"
                    "qutr 80\n"
                    "ztat ust 18 veaiat st gt 36 t fara Bl"
                )
            )
        ],
        parser_provider="local",
        detected_language="en-IN",
        script="Latn",
    )

    async def fake_parse(self, *, content, filename, extension):
        return parsed

    monkeypatch.setattr(document_parsers.LocalDocumentParser, "parse", fake_parse)
    monkeypatch.setattr(document_parsers.settings, "document_parser_provider", "auto")
    monkeypatch.setattr(document_parsers.settings, "enable_multilingual_docs", True)
    monkeypatch.setattr(document_parsers.settings, "sarvam_api_key", "key")
    parser, preview, warnings = await choose_document_parser(
        content=b"pdf",
        filename="hindi_doc.pdf",
        extension=".pdf",
        description="hindi document",
    )
    assert parser.provider == "sarvam"
    assert preview is parsed
    assert any("Trying Sarvam" in warning for warning in warnings)


@pytest.mark.asyncio
async def test_auto_parser_warns_for_garbled_pdf_without_sarvam(monkeypatch):
    parsed = ParsedDocument(
        chunks=[Document(page_content="rtr\nf ( s sT 085\nqutr 80\n(1*5=5 )")],
        parser_provider="local",
        detected_language="en-IN",
        script="Latn",
    )

    async def fake_parse(self, *, content, filename, extension):
        return parsed

    monkeypatch.setattr(document_parsers.LocalDocumentParser, "parse", fake_parse)
    monkeypatch.setattr(document_parsers.settings, "document_parser_provider", "auto")
    monkeypatch.setattr(document_parsers.settings, "enable_multilingual_docs", True)
    monkeypatch.setattr(document_parsers.settings, "sarvam_api_key", "")
    parser, preview, warnings = await choose_document_parser(
        content=b"pdf",
        filename="hindi_doc.pdf",
        extension=".pdf",
        description="hindi document",
    )
    assert parser.provider == "local"
    assert preview is parsed
    assert any("Sarvam is not configured" in warning for warning in warnings)


@pytest.mark.asyncio
async def test_sarvam_parser_rejects_documents_over_page_limit(monkeypatch):
    monkeypatch.setattr(document_parsers.settings, "sarvam_api_key", "key")
    monkeypatch.setattr(document_parsers.settings, "sarvam_max_pages_per_job", 10)
    monkeypatch.setattr(document_parsers, "_pdf_page_count", lambda content: 11)
    parser = document_parsers.SarvamDocumentParser()
    with pytest.raises(RuntimeError, match="up to 10 pages"):
        await parser.parse(content=b"pdf", filename="large.pdf", extension=".pdf")


def test_sarvam_blob_upload_uses_required_azure_headers(monkeypatch):
    captured_put = {}

    class FakeResponse:
        def __init__(self, payload=None, text=""):
            self._payload = payload or {}
            self.text = text
            self.headers = {"content-type": "text/markdown"}
            self.content = text.encode()

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    post_payloads = [
        {"job_id": "job-123"},
        {"upload_urls": {"policy.pdf": "https://blob.test/upload?sig=abc"}},
        {},
        {"download_urls": {"policy.md": "https://blob.test/output.md"}},
    ]

    def fake_post(*args, **kwargs):
        return FakeResponse(post_payloads.pop(0))

    def fake_put(*args, **kwargs):
        captured_put.update(kwargs)
        return FakeResponse()

    def fake_get(url, **kwargs):
        if url.endswith("/status"):
            return FakeResponse({"job_state": "Completed"})
        return FakeResponse(text="यह नीति दस्तावेज है")

    monkeypatch.setattr(document_parsers.requests, "post", fake_post)
    monkeypatch.setattr(document_parsers.requests, "put", fake_put)
    monkeypatch.setattr(document_parsers.requests, "get", fake_get)
    monkeypatch.setattr(document_parsers.settings, "sarvam_api_key", "key")
    parser = document_parsers.SarvamDocumentParser()

    parsed = parser._parse_sync(b"%PDF", "policy.pdf")

    assert parsed.parser_provider == "sarvam"
    assert captured_put["headers"]["x-ms-blob-type"] == "BlockBlob"
    assert captured_put["headers"]["Content-Type"] == "application/pdf"


def test_sarvam_download_detects_zip_by_magic_bytes(monkeypatch, tmp_path):
    archive_path = tmp_path / "sarvam-output.zip"
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("document.md", "# Policy\n\nयह नीति दस्तावेज है")

    class FakeResponse:
        headers = {"content-type": "application/octet-stream"}
        content = archive_path.read_bytes()
        text = "this should not be used"

        def raise_for_status(self):
            return None

    monkeypatch.setattr(document_parsers.requests, "get", lambda *args, **kwargs: FakeResponse())

    markdown = _download_markdown({"document.bin": "https://blob.test/output"})

    assert "नीति दस्तावेज" in markdown
    assert not markdown.startswith("PK")


def test_prompt_preserves_resume_institution_names_and_abbreviations():
    prompt = Path("src/config/prompts.yaml").read_text(encoding="utf-8")
    assert "M.S. Ramaiah Institute of Technology" in prompt
    assert "do not interpret" in prompt


def test_sarvam_timeout_reports_last_status(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "job_state": "Running",
                "page_metrics": {
                    "total_pages": 3,
                    "pages_processed": 1,
                    "pages_succeeded": 1,
                    "pages_failed": 0,
                },
            }

    monkeypatch.setattr(document_parsers.requests, "get", lambda *args, **kwargs: FakeResponse())
    monkeypatch.setattr(document_parsers.settings, "sarvam_job_timeout_seconds", 30)
    monkeypatch.setattr(document_parsers.settings, "sarvam_job_poll_seconds", 10)
    times = iter([0, 1, 31])
    monkeypatch.setattr(document_parsers.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(time, "sleep", lambda seconds: None)

    parser = document_parsers.SarvamDocumentParser()
    with pytest.raises(TimeoutError) as error:
        parser._wait_for_completion("job-123")

    assert "state=Running" in str(error.value)
    assert "processed=1/3" in str(error.value)
