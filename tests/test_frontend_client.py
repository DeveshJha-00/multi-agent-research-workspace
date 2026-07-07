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


def test_frontend_image_exposes_project_package_path():
    content = Path("Dockerfile.streamlit").read_text(encoding="utf-8")
    assert "PYTHONPATH=/app" in content
    assert "HOME=/home/app" in content
