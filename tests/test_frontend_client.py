from pathlib import Path

from streamlit_app.utils import api_client


class FakeResponse:
    def __init__(self, payload=None, content=b"", headers=None):
        self._payload = payload
        self.content = content
        self.headers = headers or {}
        self.status_code = 200
        self.text = ""

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_research_client_sends_workspace_and_available_data(monkeypatch):
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return FakeResponse({"task_id": "task-1", "content": "done"})

    monkeypatch.setattr(api_client.requests, "post", fake_post)
    result = api_client.run_research(
        "Analyze the uploaded dataset",
        "session-123",
        ["Dataset ID dataset-1"],
    )
    assert result["task_id"] == "task-1"
    assert captured["json"]["session_id"] == "session-123"
    assert captured["json"]["available_data"] == ["Dataset ID dataset-1"]
    assert captured["headers"]["Idempotency-Key"]


def test_artifact_download_returns_content_and_media_type(monkeypatch):
    monkeypatch.setattr(
        api_client.requests,
        "get",
        lambda *args, **kwargs: FakeResponse(
            content=b"report", headers={"Content-Type": "text/markdown"}
        ),
    )
    assert api_client.download_artifact("artifact-1", "session-123") == (
        b"report",
        "text/markdown",
    )


def test_repository_client_uploads_zip_and_lists_workspace_repositories(monkeypatch):
    captured = {}

    class UploadedFile:
        name = "project.zip"
        type = "application/zip"

        @staticmethod
        def getvalue():
            return b"zip-content"

    def fake_post(url, **kwargs):
        captured["post_url"] = url
        captured["post"] = kwargs
        return FakeResponse({"repository_id": "repository-1", "reused": False})

    def fake_get(url, **kwargs):
        captured["get_url"] = url
        captured["get"] = kwargs
        return FakeResponse([{"repository_id": "repository-1", "filename": "project.zip"}])

    monkeypatch.setattr(api_client.requests, "post", fake_post)
    monkeypatch.setattr(api_client.requests, "get", fake_get)

    uploaded = api_client.repository_upload(UploadedFile(), "Demo source", "session-123")
    repositories = api_client.get_repositories("session-123")

    assert uploaded["repository_id"] == "repository-1"
    assert captured["post"]["files"]["file"][0] == "project.zip"
    assert captured["post"]["headers"]["X-Session-ID"] == "session-123"
    assert repositories[0]["repository_id"] == "repository-1"
    assert captured["get"]["headers"]["X-Session-ID"] == "session-123"


def test_research_ui_exposes_repository_upload_and_durable_analysis():
    content = Path("streamlit_app/pages/research.py").read_text(encoding="utf-8")
    assert '"Document", "Dataset", "Repository"' in content
    assert "repository_inventory_completed" in content
    assert "Analyze repository" in content


def test_frontend_image_exposes_project_package_path():
    content = Path("Dockerfile.streamlit").read_text(encoding="utf-8")
    assert "PYTHONPATH=/app" in content
    assert "HOME=/home/app" in content


def test_evaluation_client_submits_reference_and_workspace(monkeypatch):
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return FakeResponse({"evaluation_id": "evaluation-1", "status": "queued"})

    monkeypatch.setattr(api_client.requests, "post", fake_post)
    result = api_client.create_evaluation(
        "response-1", "session-123", "Reference answer", "idempotency-1"
    )
    assert result["evaluation_id"] == "evaluation-1"
    assert captured["json"]["reference"] == "Reference answer"
    assert captured["headers"]["X-Session-ID"] == "session-123"
    assert captured["headers"]["Idempotency-Key"] == "idempotency-1"


def test_chat_client_sends_language_metadata(monkeypatch):
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return FakeResponse({"response_id": "response-1", "content": "उत्तर"})

    monkeypatch.setattr(api_client.requests, "post", fake_post)
    result = api_client.query_backend(
        "प्रश्न",
        "session-123",
        query_language="hi-IN",
        answer_language="hi-IN",
    )
    assert result["content"] == "उत्तर"
    assert captured["json"]["query_language"] == "hi-IN"
    assert captured["json"]["answer_language"] == "hi-IN"


def test_chat_history_client_loads_workspace_messages(monkeypatch):
    captured = {}

    def fake_get(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return FakeResponse(
            {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there"},
                ]
            }
        )

    monkeypatch.setattr(api_client.requests, "get", fake_get)
    messages = api_client.get_chat_history("session-123")
    assert messages[0]["role"] == "user"
    assert messages[1]["content"] == "Hi there"
    assert captured["headers"]["X-Session-ID"] == "session-123"


def test_voice_clients_call_speech_endpoints(monkeypatch):
    captured = {}

    class AudioFile:
        name = "question.wav"
        type = "audio/wav"

        @staticmethod
        def getvalue():
            return b"audio"

    def fake_post(url, **kwargs):
        captured.setdefault("calls", []).append((url, kwargs))
        if url.endswith("/transcribe"):
            return FakeResponse({"transcript": "hello", "language_code": "en-IN"})
        return FakeResponse({"audio_base64": "d2F2", "mime_type": "audio/wav"})

    monkeypatch.setattr(api_client.requests, "post", fake_post)

    transcript = api_client.transcribe_speech(AudioFile(), "session-123")
    audio = api_client.synthesize_speech(
        "hello",
        "session-123",
        "en-IN",
        "auto",
        1.0,
    )

    assert transcript["transcript"] == "hello"
    assert audio["audio_base64"] == "d2F2"
    assert captured["calls"][0][1]["headers"]["X-Session-ID"] == "session-123"
    assert captured["calls"][1][1]["json"]["speaker"] is None


def test_indexed_documents_client_loads_workspace_docs(monkeypatch):
    captured = {}

    def fake_get(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return FakeResponse([{"document_id": "doc-1", "filename": "hindi_doc.pdf"}])

    monkeypatch.setattr(api_client.requests, "get", fake_get)
    result = api_client.get_indexed_documents("session-123")
    assert result[0]["filename"] == "hindi_doc.pdf"
    assert captured["headers"]["X-Session-ID"] == "session-123"


def test_chat_ui_exposes_ragas_controls():
    content = Path("streamlit_app/pages/chat.py").read_text(encoding="utf-8")
    assert "Evaluate response" in content
    assert "Optional reference answer" in content
    assert "Scores are model-based diagnostics" in content
    assert "_render_chat_message(assistant_message)" in content
    assert "get_indexed_documents" in content
    assert "Enable voice answers" in content
    assert "st.audio_input" in content
    assert "Upload audio file" in content
    assert "Browser recorder (experimental)" in content
    assert "Transcribe into message box" in content
    assert "Send voice transcript" in content
    assert "st.chat_input" in content
    assert "components.html" in content
    assert "autoplay" in content
