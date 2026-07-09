"""Provider-neutral speech service boundaries for future Sarvam STT/TTS work."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SpeechTranscript:
    text: str
    language_code: str | None = None
    duration_seconds: float | None = None


@dataclass(frozen=True)
class SpeechSynthesis:
    audio: bytes
    mime_type: str
    language_code: str


class SpeechToTextService(Protocol):
    async def transcribe(self, audio: bytes, *, filename: str, language_code: str = "auto") -> SpeechTranscript:
        """Convert user speech into text for the existing chat/RAG flow."""


class TextToSpeechService(Protocol):
    async def synthesize(self, text: str, *, language_code: str, voice: str | None = None) -> SpeechSynthesis:
        """Convert assistant text into audio for a future answer playback UI."""
