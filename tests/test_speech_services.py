import base64
from types import SimpleNamespace

import pytest

from src.core.config import settings
from src.services import language, speech


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class FakeAsyncClient:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    async def post(self, url, **kwargs):
        self.calls.append(SimpleNamespace(url=url, kwargs=kwargs))
        return FakeResponse(self.payload)


@pytest.mark.asyncio
async def test_sarvam_stt_builds_expected_request(monkeypatch):
    monkeypatch.setattr(settings, "sarvam_api_key", "sarvam-key")
    client = FakeAsyncClient(
        {
            "transcript": "नमस्ते",
            "language_code": "hi-IN",
            "language_probability": 0.94,
            "request_id": "request-1",
        }
    )

    result = await speech.SarvamSpeechToTextService(client).transcribe(
        b"audio",
        filename="question.wav",
        content_type="audio/wav",
        language_code="unknown",
    )

    assert result.text == "नमस्ते"
    call = client.calls[0]
    assert call.url.endswith("/speech-to-text")
    assert call.kwargs["headers"]["api-subscription-key"] == "sarvam-key"
    assert call.kwargs["data"]["model"] == settings.sarvam_stt_model
    assert call.kwargs["data"]["mode"] == settings.sarvam_stt_mode
    assert call.kwargs["data"]["language_code"] == "unknown"
    assert call.kwargs["files"]["file"][0] == "question.wav"


@pytest.mark.asyncio
async def test_sarvam_tts_decodes_audio_and_uses_recommended_speaker(monkeypatch):
    monkeypatch.setattr(settings, "sarvam_api_key", "sarvam-key")
    audio = base64.b64encode(b"wav-bytes").decode()
    client = FakeAsyncClient({"audios": [audio], "request_id": "tts-1"})

    result = await speech.SarvamTextToSpeechService(client).synthesize(
        "Short answer.",
        language_code="hi-IN",
        voice="auto",
        pace=1.1,
    )

    assert result.audio == b"wav-bytes"
    assert result.audio_base64 == audio
    assert result.speaker == speech.TTS_RECOMMENDED_SPEAKERS["hi-IN"]
    call = client.calls[0]
    assert call.url.endswith("/text-to-speech")
    assert call.kwargs["json"]["target_language_code"] == "hi-IN"
    assert call.kwargs["json"]["output_audio_codec"] == settings.sarvam_tts_audio_format
    assert call.kwargs["json"]["pace"] == 1.1


def test_unsupported_tts_language_is_rejected():
    assert speech.tts_language_supported("as-IN") is False
    with pytest.raises(speech.SarvamSpeechError):
        if not speech.tts_language_supported("as-IN"):
            raise speech.SarvamSpeechError("Bulbul TTS does not support as-IN")


def test_spoken_preview_shortens_long_answers(monkeypatch):
    monkeypatch.setattr(settings, "sarvam_tts_long_answer_char_limit", 120)
    short, shortened = speech.spoken_preview_text("One line.\nTwo lines.")
    assert shortened is False
    assert short == "One line.\nTwo lines."

    long_text = "\n".join(
        [
            "Summary: this answer has a concise opening.",
            "First important detail.",
            "Second important detail.",
            "Third important detail.",
            "Fourth important detail that should be skipped.",
        ]
    )
    preview, shortened = speech.spoken_preview_text(long_text)
    assert shortened is True
    assert len(preview) <= 120
    assert "Summary" in preview
    assert "Fourth important" not in preview


@pytest.mark.asyncio
async def test_sarvam_language_detection_uses_text_lid(monkeypatch):
    monkeypatch.setattr(settings, "sarvam_api_key", "sarvam-key")
    client = FakeAsyncClient({"language_code": "ta-IN", "script_code": "Taml"})

    result = await language.SarvamLanguageDetectionService(client).detect_text(
        "தமிழ் கேள்வி"
    )

    assert result.language_code == "ta-IN"
    assert result.script_code == "Taml"
    call = client.calls[0]
    assert call.url.endswith("/text-lid")
    assert call.kwargs["json"]["input"] == "தமிழ் கேள்வி"


@pytest.mark.asyncio
async def test_language_detection_falls_back_without_sarvam(monkeypatch):
    monkeypatch.setattr(settings, "sarvam_api_key", "")
    result = await language.SarvamLanguageDetectionService().detect_text("हिंदी सवाल")
    assert result.language_code == "hi-IN"
