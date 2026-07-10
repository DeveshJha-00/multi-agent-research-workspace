"""Language-aware document parsing with optional Sarvam digitization."""

from __future__ import annotations

import asyncio
import json
import os
import re
import tempfile
import time
import zipfile
from dataclasses import dataclass, field
from typing import Protocol

import fitz
import requests
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from src.core.config import settings
from src.services.language import detect_text_language

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}
NON_ENGLISH_HINTS = {
    "bangla",
    "bengali",
    "gujarati",
    "hindi",
    "indic",
    "kannada",
    "malayalam",
    "marathi",
    "odia",
    "punjabi",
    "sanskrit",
    "tamil",
    "telugu",
    "urdu",
}


@dataclass(frozen=True)
class ParsedDocument:
    chunks: list[Document]
    parser_provider: str
    detected_language: str
    script: str
    extraction_confidence: float | None = None
    warnings: list[str] = field(default_factory=list)


class DocumentParser(Protocol):
    provider: str

    async def parse(self, *, content: bytes, filename: str, extension: str) -> ParsedDocument:
        """Parse file bytes into chunked LangChain documents."""


@dataclass(frozen=True)
class ExtractionCandidate:
    documents: list[Document]
    engine: str
    quality_score: float
    text_repaired: bool = False


def _split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True,
    )
    return splitter.split_documents(documents)


def _load_text(path: str) -> list[Document]:
    return TextLoader(path, encoding="utf-8").load()


def _load_pypdf(path: str) -> list[Document]:
    return PyPDFLoader(path).load()


def _load_pymupdf(path: str, filename: str) -> list[Document]:
    pdf = fitz.open(path)
    try:
        documents: list[Document] = []
        total_pages = pdf.page_count
        metadata = dict(pdf.metadata or {})
        for index, page in enumerate(pdf):
            text = page.get_text("text", sort=True)
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        **metadata,
                        "source": filename,
                        "total_pages": total_pages,
                        "page": index,
                        "page_label": str(index + 1),
                    },
                )
            )
        return documents
    finally:
        pdf.close()


def _segment_is_character_spaced(segment: str) -> bool:
    tokens = re.findall(r"\S+", segment)
    if len(tokens) < 3:
        lengths = [len(re.sub(r"[^A-Za-z0-9]", "", token)) for token in tokens]
        return len(tokens) == 2 and all(length <= 2 for length in lengths) and any(
            length == 1 for length in lengths
        )
    compact_tokens = [
        token
        for token in tokens
        if len(re.sub(r"[^A-Za-z0-9]", "", token)) <= 2
    ]
    single_char_tokens = [
        token
        for token in tokens
        if len(re.sub(r"[^A-Za-z0-9]", "", token)) == 1
    ]
    return len(single_char_tokens) / len(tokens) >= 0.55 or len(compact_tokens) / len(tokens) >= 0.8


def _repair_character_spaced_line(line: str) -> str:
    parts = re.split(r"(\s{2,})", line.rstrip())
    repaired: list[str] = []
    for part in parts:
        if not part:
            continue
        if part.isspace():
            repaired.append(" ")
            continue
        if _segment_is_character_spaced(part):
            repaired.append("".join(re.findall(r"\S+", part)))
        else:
            repaired.append(part)
    line = "".join(repaired)
    line = re.sub(r"\s+([,:;.!?)%])", r"\1", line)
    line = re.sub(r"([(])\s+", r"\1", line)
    line = re.sub(r"\s{2,}", " ", line)
    return line.strip()


def _repair_pdf_text(text: str) -> tuple[str, bool]:
    repaired_lines = [_repair_character_spaced_line(line) for line in text.splitlines()]
    repaired = "\n".join(line for line in repaired_lines if line)
    return repaired, repaired != text.strip()


def _binary_character_ratio(text: str) -> float:
    if not text:
        return 1.0
    bad = sum(
        1
        for char in text
        if char == "\ufffd" or (ord(char) < 32 and char not in "\n\r\t")
    )
    return bad / len(text)


def _real_word_ratio(text: str) -> float:
    tokens = re.findall(r"\S+", text)
    if not tokens:
        return 0.0
    real_words = re.findall(r"[A-Za-z]{3,}|[\u0900-\u097F]{2,}", text)
    return min(1.0, len(real_words) / len(tokens))


def _single_character_token_ratio(text: str) -> float:
    tokens = re.findall(r"\S+", text)
    if not tokens:
        return 1.0
    compact_tokens = [
        token
        for token in tokens
        if len(re.sub(r"[^A-Za-z0-9\u0900-\u097F]", "", token)) <= 1
    ]
    return len(compact_tokens) / len(tokens)


def _section_signal_score(text: str) -> float:
    lowered = text.lower()
    signals = (
        "education",
        "experience",
        "project",
        "skills",
        "coursework",
        "certification",
        "achievement",
        "languages",
        "frameworks",
        "databases",
    )
    return min(1.0, sum(1 for signal in signals if signal in lowered) / 4)


def text_quality_score(text: str) -> float:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return 0.0
    length_score = min(1.0, len(normalized) / 800)
    word_score = _real_word_ratio(normalized)
    single_penalty = _single_character_token_ratio(normalized)
    binary_penalty = _binary_character_ratio(normalized)
    line_score = min(1.0, len([line for line in text.splitlines() if line.strip()]) / 8)
    section_score = _section_signal_score(normalized)
    score = (
        0.25 * length_score
        + 0.30 * word_score
        + 0.15 * line_score
        + 0.15 * section_score
        + 0.15 * (1 - single_penalty)
        - 0.75 * binary_penalty
    )
    return max(0.0, min(1.0, score))


def is_binary_like_text(text: str) -> bool:
    if not text.strip():
        return True
    return _binary_character_ratio(text) > 0.02 or (
        len(text) > 200 and _real_word_ratio(text) < 0.08
    )


def _combined_text(parsed: ParsedDocument) -> str:
    return "\n".join(chunk.page_content for chunk in parsed.chunks)


def _has_non_english_hint(*values: str) -> bool:
    text = " ".join(values).lower().replace("_", " ")
    tokens = set(re.split(r"[^a-z]+", text))
    return bool(tokens & NON_ENGLISH_HINTS)


def _looks_like_bad_pdf_extraction(text: str) -> bool:
    """Detect mojibake/gibberish that often means PDF text extraction failed.

    This intentionally does not reject normal short English docs. It looks for PDFs that
    are mostly low-information fragments: few real words, many isolated tokens, and no
    Indic script despite non-English hints or auto multilingual parsing.
    """
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return True

    words = re.findall(r"[A-Za-z]{3,}", normalized)
    tokens = re.findall(r"\S+", normalized)
    if not tokens:
        return True
    if len(normalized) < 40:
        return len(words) < 2
    real_word_ratio = len(words) / len(tokens)
    single_or_symbol_tokens = sum(
        1 for token in tokens if len(re.sub(r"[^A-Za-z0-9]", "", token)) <= 2
    )
    low_information_ratio = single_or_symbol_tokens / len(tokens)
    has_sentence_shape = bool(re.search(r"[A-Za-z]{3,}\s+[A-Za-z]{3,}\s+[A-Za-z]{3,}", normalized))
    return (
        real_word_ratio < 0.45
        or low_information_ratio > 0.45
        or (len(tokens) >= 8 and not has_sentence_shape)
    )


def _repair_documents(documents: list[Document]) -> tuple[list[Document], bool]:
    repaired_documents: list[Document] = []
    any_repaired = False
    for document in documents:
        repaired_text, repaired = _repair_pdf_text(document.page_content)
        any_repaired = any_repaired or repaired
        repaired_documents.append(
            Document(page_content=repaired_text, metadata=dict(document.metadata))
        )
    return repaired_documents, any_repaired


def _candidate(documents: list[Document], *, engine: str) -> ExtractionCandidate:
    repaired_documents, text_repaired = _repair_documents(documents)
    text = "\n".join(document.page_content for document in repaired_documents)
    return ExtractionCandidate(
        documents=repaired_documents,
        engine=engine,
        quality_score=text_quality_score(text),
        text_repaired=text_repaired,
    )


def _best_pdf_extraction(path: str, filename: str) -> ExtractionCandidate:
    candidates: list[ExtractionCandidate] = []
    try:
        candidates.append(_candidate(_load_pymupdf(path, filename), engine="pymupdf"))
    except Exception:
        candidates.append(ExtractionCandidate([], engine="pymupdf", quality_score=0.0))
    try:
        candidates.append(_candidate(_load_pypdf(path), engine="pypdf"))
    except Exception:
        candidates.append(ExtractionCandidate([], engine="pypdf", quality_score=0.0))
    return max(candidates, key=lambda item: item.quality_score)


def _parse_local_sync(content: bytes, filename: str, extension: str) -> ParsedDocument:
    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name
        extraction_engine = "text"
        extraction_quality = 1.0
        text_repaired = False
        if extension == ".pdf":
            selected = _best_pdf_extraction(tmp_path, filename)
            documents = selected.documents
            extraction_engine = selected.engine
            extraction_quality = selected.quality_score
            text_repaired = selected.text_repaired
        else:
            documents = _load_text(tmp_path)
            extraction_quality = text_quality_score(
                "\n".join(document.page_content for document in documents)
            )
        text = "\n".join(document.page_content for document in documents)
        detection = detect_text_language(text)
        chunks = _split_documents(documents)
        for chunk in chunks:
            page = chunk.metadata.get("page")
            chunk.metadata.update(
                {
                    "parser_provider": "local",
                    "detected_language": detection.language_code,
                    "script": detection.script_code,
                    "source_page": page,
                    "extraction_confidence": detection.confidence,
                    "text_repaired": text_repaired,
                    "extraction_engine": extraction_engine,
                    "extraction_quality": extraction_quality,
                    "binary_like": is_binary_like_text(chunk.page_content),
                }
            )
        return ParsedDocument(
            chunks=chunks,
            parser_provider="local",
            detected_language=detection.language_code,
            script=detection.script_code,
            extraction_confidence=detection.confidence,
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


class LocalDocumentParser:
    provider = "local"

    async def parse(self, *, content: bytes, filename: str, extension: str) -> ParsedDocument:
        return await asyncio.to_thread(_parse_local_sync, content, filename, extension)


class SarvamDocumentParser:
    provider = "sarvam"

    def __init__(self) -> None:
        self.base_url = settings.sarvam_base_url.rstrip("/")

    async def parse(self, *, content: bytes, filename: str, extension: str) -> ParsedDocument:
        if not settings.sarvam_api_key:
            raise RuntimeError("SARVAM_API_KEY is required for Sarvam document parsing")
        if extension != ".pdf":
            raise RuntimeError("Sarvam document parsing currently supports PDF uploads")
        page_count = await asyncio.to_thread(_pdf_page_count, content)
        if page_count > settings.sarvam_max_pages_per_job:
            raise RuntimeError(
                f"Sarvam document parsing supports up to {settings.sarvam_max_pages_per_job} "
                "pages per job. Split this PDF or use local parsing."
            )
        return await asyncio.to_thread(self._parse_sync, content, filename)

    def _headers(self) -> dict[str, str]:
        return {
            "api-subscription-key": settings.sarvam_api_key,
            "Content-Type": "application/json",
        }

    def _parse_sync(self, content: bytes, filename: str) -> ParsedDocument:
        language = settings.sarvam_document_language
        if language == "auto":
            language = "hi-IN"
        job_response = requests.post(
            f"{self.base_url}/doc-digitization/job/v1",
            headers=self._headers(),
            json={
                "job_parameters": {
                    "language": language,
                    "output_format": settings.sarvam_document_output_format,
                }
            },
            timeout=30,
        )
        job_response.raise_for_status()
        job_id = job_response.json()["job_id"]

        upload_response = requests.post(
            f"{self.base_url}/doc-digitization/job/v1/upload-files",
            headers=self._headers(),
            json={"job_id": job_id, "files": [filename]},
            timeout=30,
        )
        upload_response.raise_for_status()
        upload_url = _extract_file_url(upload_response.json()["upload_urls"], filename)
        put_response = requests.put(
            upload_url,
            data=content,
            headers={
                "x-ms-blob-type": "BlockBlob",
                "Content-Type": "application/pdf",
            },
            timeout=60,
        )
        put_response.raise_for_status()

        start_response = requests.post(
            f"{self.base_url}/doc-digitization/job/v1/{job_id}/start",
            headers=self._headers(),
            json={},
            timeout=30,
        )
        start_response.raise_for_status()
        final_status = self._wait_for_completion(job_id)
        download_response = requests.post(
            f"{self.base_url}/doc-digitization/job/v1/{job_id}/download-files",
            headers=self._headers(),
            json={},
            timeout=30,
        )
        download_response.raise_for_status()
        markdown = _download_markdown(download_response.json()["download_urls"])
        detection = detect_text_language(markdown)
        warnings: list[str] = []
        if is_binary_like_text(markdown):
            warnings.append(
                "Sarvam returned unreadable or binary-looking text. The document was not indexed."
            )
            return ParsedDocument(
                chunks=[],
                parser_provider="sarvam",
                detected_language=detection.language_code,
                script=detection.script_code,
                extraction_confidence=detection.confidence,
                warnings=warnings,
            )
        documents = [
            Document(
                page_content=markdown,
                metadata={
                    "source": filename,
                    "page": None,
                    "source_page": None,
                    "sarvam_job_id": job_id,
                    "sarvam_job_state": final_status.get("job_state"),
                },
            )
        ]
        chunks = _split_documents(documents)
        for chunk in chunks:
            chunk.metadata.update(
                {
                    "parser_provider": "sarvam",
                    "detected_language": detection.language_code,
                    "script": detection.script_code,
                    "extraction_confidence": detection.confidence,
                    "extraction_engine": "sarvam",
                    "extraction_quality": text_quality_score(markdown),
                    "binary_like": is_binary_like_text(chunk.page_content),
                }
            )
        return ParsedDocument(
            chunks=chunks,
            parser_provider="sarvam",
            detected_language=detection.language_code,
            script=detection.script_code,
            extraction_confidence=detection.confidence,
            warnings=warnings,
        )

    def _wait_for_completion(self, job_id: str) -> dict:
        deadline = time.monotonic() + settings.sarvam_job_timeout_seconds
        while time.monotonic() < deadline:
            response = requests.get(
                f"{self.base_url}/doc-digitization/job/v1/{job_id}/status",
                headers=self._headers(),
                timeout=30,
            )
            response.raise_for_status()
            status = response.json()
            state = status.get("job_state")
            if state in {"Completed", "PartiallyCompleted"}:
                return status
            if state == "Failed":
                raise RuntimeError(status.get("error_message") or "Sarvam parsing failed")
            time.sleep(settings.sarvam_job_poll_seconds)
        raise TimeoutError("Sarvam document parsing timed out")


def _pdf_page_count(content: bytes) -> int:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(content)
        tmp_path = tmp_file.name
    try:
        return len(PdfReader(tmp_path).pages)
    finally:
        os.unlink(tmp_path)


def _extract_file_url(upload_urls: dict, filename: str) -> str:
    item = upload_urls.get(filename) or next(iter(upload_urls.values()))
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return str(item.get("file_url") or item.get("url") or item.get("upload_url"))
    raise RuntimeError("Sarvam did not return a usable upload URL")


def _download_markdown(download_urls: dict) -> str:
    preferred = None
    fallback = None
    for name, item in download_urls.items():
        url = item if isinstance(item, str) else item.get("file_url")
        if not url:
            continue
        fallback = fallback or url
        if str(name).lower().endswith((".md", ".markdown", ".zip", ".json")):
            preferred = url
            break
    url = preferred or fallback
    if not url:
        raise RuntimeError("Sarvam did not return a usable download URL")

    tmp_path, first_bytes, content_type = _download_to_temp_file(url)
    try:
        if first_bytes.startswith(b"PK") or "zip" in content_type or url.lower().endswith(".zip"):
            return _read_text_from_zip_path(tmp_path)
        if url.lower().endswith(".json"):
            text = _read_text_file_capped(tmp_path)
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                return text
            return json.dumps(payload, ensure_ascii=False, indent=2)[
                : settings.sarvam_document_max_output_chars
            ]
        return _read_text_file_capped(tmp_path)
    finally:
        os.unlink(tmp_path)


def _download_to_temp_file(url: str) -> tuple[str, bytes, str]:
    response = requests.get(url, timeout=60, stream=True)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "")
    limit = settings.sarvam_document_download_max_bytes
    total = 0
    first_bytes = b""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".sarvam") as tmp_file:
        tmp_path = tmp_file.name
        try:
            chunks = (
                response.iter_content(chunk_size=128 * 1024)
                if hasattr(response, "iter_content")
                else [response.content]
            )
            for chunk in chunks:
                if not chunk:
                    continue
                total += len(chunk)
                if total > limit:
                    raise RuntimeError(
                        "Sarvam document output is too large for this demo deployment. "
                        "Use a smaller document or increase SARVAM_DOCUMENT_DOWNLOAD_MAX_BYTES."
                    )
                if len(first_bytes) < 8:
                    first_bytes += chunk[: 8 - len(first_bytes)]
                tmp_file.write(chunk)
        except Exception:
            tmp_file.close()
            os.unlink(tmp_path)
            raise
    return tmp_path, first_bytes, content_type


def _read_text_file_capped(path: str) -> str:
    with open(path, "rb") as file:
        content = file.read(settings.sarvam_document_max_output_chars + 1)
    return content.decode("utf-8", errors="replace")[: settings.sarvam_document_max_output_chars]


def _read_text_from_zip_path(path: str) -> str:
    max_chars = settings.sarvam_document_max_output_chars
    used = 0
    parts: list[str] = []
    with zipfile.ZipFile(path) as archive:
        names = sorted(
            name
            for name in archive.namelist()
            if name.lower().endswith((".md", ".txt", ".json", ".html"))
        )
        for name in names:
            if used >= max_chars:
                break
            remaining = max_chars - used
            with archive.open(name) as file:
                raw = file.read(remaining + 1)
            text = raw.decode("utf-8", errors="replace")[:remaining]
            if text:
                parts.append(text)
                used += len(text)
    return "\n\n".join(parts)


async def choose_document_parser(
    *, content: bytes, filename: str, extension: str, description: str = ""
) -> tuple[DocumentParser, ParsedDocument | None, list[str]]:
    local_parser = LocalDocumentParser()
    provider = settings.document_parser_provider
    warnings: list[str] = []
    if provider == "local" or extension == ".txt":
        return local_parser, None, warnings
    if provider == "sarvam":
        return SarvamDocumentParser(), None, warnings

    preview = await local_parser.parse(content=content, filename=filename, extension=extension)
    preview_text = _combined_text(preview)
    has_language_hint = _has_non_english_hint(filename, description)
    bad_pdf_extraction = extension == ".pdf" and _looks_like_bad_pdf_extraction(preview_text)
    non_english_detection = preview.detected_language != "en-IN" or preview.script != "Latn"
    should_try_sarvam = (
        settings.enable_multilingual_docs
        and bool(settings.sarvam_api_key)
        and (non_english_detection or has_language_hint)
    )
    if should_try_sarvam:
        warnings.append(
            "This document appears non-English/Indic or OCR-like. Trying Sarvam document "
            "digitization first; local extraction is retained as a fallback."
        )
        return SarvamDocumentParser(), preview, warnings
    if preview.detected_language == "en-IN" and preview.script == "Latn" and not has_language_hint:
        if bad_pdf_extraction:
            warnings.append(
                "Local PDF text extraction may still be incomplete after cleanup. "
                "Using repaired local text because this appears to be an English document."
            )
        return local_parser, preview, warnings
    if bad_pdf_extraction:
        warnings.append(
            "Local PDF text extraction looked incomplete or garbled. Configure "
            "SARVAM_API_KEY to use Sarvam document digitization for this file."
        )
    warnings.append(
        "This document may need multilingual/OCR-grade parsing, but Sarvam is not "
        "configured. Using best-effort local parsing."
    )
    return local_parser, preview, warnings
