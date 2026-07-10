"""Language detection and preference primitives.

The default detector is intentionally local and lightweight. Sarvam Language
Identification can be plugged in later behind the same interface without
changing RAG or upload flows.
"""

from dataclasses import dataclass
from typing import Protocol

import httpx
from pydantic import BaseModel, Field

from src.core.config import settings

INDIC_SCRIPT_RANGES = (
    ("Deva", "\u0900", "\u097F", "hi-IN"),
    ("Beng", "\u0980", "\u09FF", "bn-IN"),
    ("Guru", "\u0A00", "\u0A7F", "pa-IN"),
    ("Gujr", "\u0A80", "\u0AFF", "gu-IN"),
    ("Orya", "\u0B00", "\u0B7F", "od-IN"),
    ("Taml", "\u0B80", "\u0BFF", "ta-IN"),
    ("Telu", "\u0C00", "\u0C7F", "te-IN"),
    ("Knda", "\u0C80", "\u0CFF", "kn-IN"),
    ("Mlym", "\u0D00", "\u0D7F", "ml-IN"),
)


class LanguagePreferences(BaseModel):
    """Session/workspace language preferences for future multilingual UI."""

    ui_language: str = Field(default="en-IN", min_length=2, max_length=20)
    query_language: str = Field(default="auto", min_length=2, max_length=20)
    answer_language: str = Field(default="auto", min_length=2, max_length=20)


@dataclass(frozen=True)
class LanguageDetection:
    language_code: str
    script_code: str
    confidence: float | None = None

    @property
    def is_english_latin(self) -> bool:
        return self.language_code == "en-IN" and self.script_code == "Latn"


class LanguageDetectionService(Protocol):
    async def detect_text(self, text: str) -> LanguageDetection:
        """Detect the dominant language/script of text."""


class HeuristicLanguageDetectionService:
    """Small local detector for routing English vs Indic document parsing."""

    async def detect_text(self, text: str) -> LanguageDetection:
        return detect_text_language(text)


class SarvamLanguageDetectionService:
    """Sarvam-backed text language identification with heuristic fallback."""

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._client = client

    async def detect_text(self, text: str) -> LanguageDetection:
        if not settings.sarvam_configured:
            return detect_text_language(text)
        snippet = text.strip()[:1000]
        if not snippet:
            return detect_text_language(text)
        close_client = self._client is None
        client = self._client or httpx.AsyncClient(timeout=20.0)
        try:
            response = await client.post(
                f"{settings.sarvam_base_url.rstrip('/')}/text-lid",
                headers={"api-subscription-key": settings.sarvam_api_key},
                json={"input": snippet},
            )
            response.raise_for_status()
            payload = response.json()
            language = payload.get("language_code") or "en-IN"
            script = payload.get("script_code") or "Latn"
            return LanguageDetection(language_code=language, script_code=script)
        except Exception:
            return detect_text_language(text)
        finally:
            if close_client:
                await client.aclose()


async def detect_query_language(text: str) -> LanguageDetection:
    """Detect query language using Sarvam when available, otherwise locally."""
    if is_obvious_latin_english_query(text):
        return LanguageDetection(language_code="en-IN", script_code="Latn", confidence=1.0)
    return await SarvamLanguageDetectionService().detect_text(text)


def detect_text_language(text: str) -> LanguageDetection:
    sample = text[:20_000]
    counts: dict[tuple[str, str], int] = {}
    latin = 0
    for char in sample:
        codepoint = ord(char)
        if "A" <= char <= "Z" or "a" <= char <= "z":
            latin += 1
            continue
        for script, start, end, language in INDIC_SCRIPT_RANGES:
            if ord(start) <= codepoint <= ord(end):
                key = (script, language)
                counts[key] = counts.get(key, 0) + 1
                break
    if counts:
        (script, language), count = max(counts.items(), key=lambda item: item[1])
        total = max(1, sum(counts.values()) + latin)
        return LanguageDetection(language_code=language, script_code=script, confidence=count / total)
    return LanguageDetection(language_code="en-IN", script_code="Latn", confidence=None)


def _latin_letter_ratio(text: str) -> float:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return 0.0
    latin = sum(1 for char in letters if "A" <= char <= "Z" or "a" <= char <= "z")
    return latin / len(letters)


def is_obvious_latin_english_query(text: str) -> bool:
    """Fast-path typed English so prior multilingual turns cannot bias language."""
    sample = text.strip()
    if not sample:
        return False
    heuristic = detect_text_language(sample)
    return (
        heuristic.language_code == "en-IN"
        and heuristic.script_code == "Latn"
        and _latin_letter_ratio(sample) >= 0.85
    )
