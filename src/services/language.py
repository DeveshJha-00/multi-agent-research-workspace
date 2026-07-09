"""Language detection and preference primitives.

The default detector is intentionally local and lightweight. Sarvam Language
Identification can be plugged in later behind the same interface without
changing RAG or upload flows.
"""

from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, Field

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
