"""Sarvam-backed speech-to-text and text-to-speech services."""

import base64
import re
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from src.core.config import settings

TTS_SUPPORTED_LANGUAGES = {
    "en-IN": "English",
    "hi-IN": "Hindi",
    "bn-IN": "Bengali",
    "gu-IN": "Gujarati",
    "kn-IN": "Kannada",
    "ml-IN": "Malayalam",
    "mr-IN": "Marathi",
    "od-IN": "Odia",
    "pa-IN": "Punjabi",
    "ta-IN": "Tamil",
    "te-IN": "Telugu",
}

TTS_SPEAKERS = [
    "auto",
    "shubh",
    "aditya",
    "rahul",
    "rohan",
    "amit",
    "dev",
    "ratan",
    "varun",
    "manan",
    "sumit",
    "kabir",
    "aayan",
    "ashutosh",
    "advait",
    "anand",
    "tarun",
    "sunny",
    "mani",
    "gokul",
    "vijay",
    "mohit",
    "rehan",
    "soham",
    "ritu",
    "priya",
    "neha",
    "pooja",
    "simran",
    "kavya",
    "ishita",
    "shreya",
    "roopa",
    "tanya",
    "shruti",
    "suhani",
    "kavitha",
    "rupali",
]

TTS_RECOMMENDED_SPEAKERS = {
    "en-IN": "ishita",
    "hi-IN": "priya",
    "bn-IN": "roopa",
    "gu-IN": "priya",
    "kn-IN": "ishita",
    "ml-IN": "pooja",
    "mr-IN": "priya",
    "od-IN": "pooja",
    "pa-IN": "mani",
    "ta-IN": "ishita",
    "te-IN": "priya",
}

AUDIO_MIME_TYPES = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "linear16": "audio/l16",
    "mulaw": "audio/basic",
    "alaw": "audio/basic",
    "opus": "audio/ogg",
    "flac": "audio/flac",
    "aac": "audio/aac",
}


class SarvamSpeechError(RuntimeError):
    """Friendly Sarvam integration error."""


@dataclass(frozen=True)
class SpeechTranscript:
    text: str
    language_code: str | None = None
    language_probability: float | None = None
    request_id: str | None = None
    duration_seconds: float | None = None


@dataclass(frozen=True)
class SpeechSynthesis:
    audio: bytes
    audio_base64: str
    mime_type: str
    language_code: str
    speaker: str
    spoken_text: str
    shortened: bool
    request_id: str | None = None


class SpeechToTextService(Protocol):
    async def transcribe(
        self,
        audio: bytes,
        *,
        filename: str,
        content_type: str | None = None,
        language_code: str = "unknown",
    ) -> SpeechTranscript:
        """Convert user speech into text for the existing chat/RAG flow."""


class TextToSpeechService(Protocol):
    async def synthesize(
        self,
        text: str,
        *,
        language_code: str,
        voice: str | None = None,
        pace: float | None = None,
    ) -> SpeechSynthesis:
        """Convert assistant text into audio for playback UI."""


def sarvam_headers() -> dict[str, str]:
    return {"api-subscription-key": settings.sarvam_api_key}


def require_voice_enabled() -> None:
    if not settings.enable_voice_features:
        raise SarvamSpeechError("Voice features are disabled")
    if not settings.sarvam_configured:
        raise SarvamSpeechError("SARVAM_API_KEY is required for voice features")


def friendly_sarvam_error(exc: Exception) -> str:
    if isinstance(exc, SarvamSpeechError):
        return str(exc)
    if isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code
        try:
            payload = exc.response.json()
        except ValueError:
            payload = exc.response.text
        if status_code == 403:
            return "Sarvam rejected SARVAM_API_KEY. Update .env and restart the API."
        if status_code == 429:
            return "Sarvam rate limit or credits were exhausted. Retry later."
        if status_code in {400, 422}:
            return f"Sarvam could not process this audio/text: {payload}"
        return f"Sarvam service returned HTTP {status_code}: {payload}"
    return "Sarvam speech service is temporarily unavailable"


def normalize_tts_language(language_code: str | None) -> str:
    language = (language_code or "en-IN").strip()
    if language == "auto" or language == "unknown":
        return "en-IN"
    return language


def tts_language_supported(language_code: str | None) -> bool:
    return normalize_tts_language(language_code) in TTS_SUPPORTED_LANGUAGES


def resolve_tts_speaker(language_code: str, speaker: str | None = None) -> str:
    selected = (speaker or settings.sarvam_tts_default_speaker or "auto").strip().lower()
    if selected and selected != "auto" and selected in TTS_SPEAKERS:
        return selected
    return TTS_RECOMMENDED_SPEAKERS.get(normalize_tts_language(language_code), "shubh")


def _strip_markdown(text: str) -> str:
    cleaned = re.sub(r"`([^`]*)`", r"\1", text)
    cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)
    cleaned = re.sub(r"[*_>#|]", "", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _sentences_or_lines(text: str) -> list[str]:
    lines = [line.strip(" -\t") for line in text.splitlines() if line.strip()]
    if len(lines) >= 2:
        return lines
    return [item.strip() for item in re.split(r"(?<=[.!?।])\s+", text) if item.strip()]


def spoken_preview_text(text: str) -> tuple[str, bool]:
    """Return full short answers, or a concise preview for long answers."""
    cleaned = _strip_markdown(text)
    if len(cleaned) <= settings.sarvam_tts_long_answer_char_limit:
        return cleaned[: settings.sarvam_tts_max_chars], False

    selected: list[str] = []
    used = 0
    for line in _sentences_or_lines(cleaned):
        candidate = line.strip()
        if not candidate:
            continue
        if used + len(candidate) + 1 > settings.sarvam_tts_long_answer_char_limit:
            break
        selected.append(candidate)
        used += len(candidate) + 1
        if len(selected) >= 4:
            break
    preview = " ".join(selected).strip() or cleaned[: settings.sarvam_tts_long_answer_char_limit]
    return preview[: settings.sarvam_tts_max_chars], True


class SarvamSpeechToTextService:
    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._client = client

    async def transcribe(
        self,
        audio: bytes,
        *,
        filename: str,
        content_type: str | None = None,
        language_code: str = "unknown",
    ) -> SpeechTranscript:
        require_voice_enabled()
        close_client = self._client is None
        client = self._client or httpx.AsyncClient(timeout=90.0)
        try:
            response = await client.post(
                f"{settings.sarvam_base_url.rstrip('/')}/speech-to-text",
                headers=sarvam_headers(),
                data={
                    "model": settings.sarvam_stt_model,
                    "mode": settings.sarvam_stt_mode,
                    "language_code": language_code or settings.sarvam_stt_language,
                },
                files={
                    "file": (
                        filename,
                        audio,
                        content_type or "application/octet-stream",
                    )
                },
            )
            response.raise_for_status()
            payload = response.json()
            return SpeechTranscript(
                text=str(payload.get("transcript") or "").strip(),
                language_code=payload.get("language_code"),
                language_probability=payload.get("language_probability"),
                request_id=payload.get("request_id"),
            )
        finally:
            if close_client:
                await client.aclose()


class SarvamTextToSpeechService:
    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._client = client

    async def synthesize(
        self,
        text: str,
        *,
        language_code: str,
        voice: str | None = None,
        pace: float | None = None,
    ) -> SpeechSynthesis:
        require_voice_enabled()
        language = normalize_tts_language(language_code)
        if language not in TTS_SUPPORTED_LANGUAGES:
            raise SarvamSpeechError(
                f"Bulbul TTS does not support {language}. Showing text only."
            )
        spoken_text, shortened = spoken_preview_text(text)
        if not spoken_text:
            raise SarvamSpeechError("No text was available for voice synthesis")
        speaker = resolve_tts_speaker(language, voice)
        close_client = self._client is None
        client = self._client or httpx.AsyncClient(timeout=90.0)
        try:
            response = await client.post(
                f"{settings.sarvam_base_url.rstrip('/')}/text-to-speech",
                headers={**sarvam_headers(), "Content-Type": "application/json"},
                json={
                    "text": spoken_text,
                    "target_language_code": language,
                    "model": settings.sarvam_tts_model,
                    "speaker": speaker,
                    "pace": pace if pace is not None else settings.sarvam_tts_default_pace,
                    "speech_sample_rate": settings.sarvam_tts_sample_rate,
                    "output_audio_codec": settings.sarvam_tts_audio_format,
                },
            )
            response.raise_for_status()
            payload = response.json()
            audios = payload.get("audios") or []
            if not audios:
                raise SarvamSpeechError("Sarvam did not return audio")
            audio_base64 = "".join(str(item) for item in audios)
            return SpeechSynthesis(
                audio=base64.b64decode(audio_base64),
                audio_base64=audio_base64,
                mime_type=AUDIO_MIME_TYPES.get(
                    settings.sarvam_tts_audio_format,
                    "audio/wav",
                ),
                language_code=language,
                speaker=speaker,
                spoken_text=spoken_text,
                shortened=shortened,
                request_id=payload.get("request_id"),
            )
        finally:
            if close_client:
                await client.aclose()


def voice_capabilities() -> dict[str, Any]:
    return {
        "enabled": settings.enable_voice_features and settings.sarvam_configured,
        "supported_languages": TTS_SUPPORTED_LANGUAGES,
        "speakers": TTS_SPEAKERS,
        "recommended_speakers": TTS_RECOMMENDED_SPEAKERS,
        "default_speaker": settings.sarvam_tts_default_speaker,
        "default_pace": settings.sarvam_tts_default_pace,
        "audio_format": settings.sarvam_tts_audio_format,
        "max_chars": settings.sarvam_tts_max_chars,
    }
