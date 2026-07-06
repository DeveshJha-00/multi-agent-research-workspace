"""Resilient HTTP client for the Python backend."""

import logging
import os

import requests

logger = logging.getLogger(__name__)
PYTHON_BASE_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT = int(os.getenv("RAG_REQUEST_TIMEOUT_SECONDS", "300"))


def _error(exc: requests.RequestException, fallback: str) -> str:
    logger.exception(fallback)
    if exc.response is not None:
        try:
            detail = exc.response.json().get("detail", "")
        except ValueError:
            detail = exc.response.text
        if detail:
            return str(detail)
    return fallback


def query_backend(query: str, session_id: str) -> dict:
    try:
        response = requests.post(
            f"{PYTHON_BASE_URL}/rag/query",
            json={"query": query, "session_id": session_id},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        return {
            "content": _error(exc, "The backend request failed."),
            "route": "error",
            "sources": [],
        }


def document_upload_rag(file, description: str, session_id: str) -> dict:
    try:
        response = requests.post(
            f"{PYTHON_BASE_URL}/rag/documents/upload",
            files={"file": (file.name, file.getvalue(), file.type)},
            headers={"X-Description": description, "X-Session-ID": session_id},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        return {"error": _error(exc, "Document upload failed")}


def dataset_upload(file, description: str, session_id: str) -> dict:
    try:
        response = requests.post(
            f"{PYTHON_BASE_URL}/agents/datasets/upload",
            files={"file": (file.name, file.getvalue(), file.type)},
            headers={"X-Description": description, "X-Session-ID": session_id},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        return {"error": _error(exc, "Dataset upload failed")}


def get_datasets(session_id: str) -> list[dict]:
    try:
        response = requests.get(
            f"{PYTHON_BASE_URL}/agents/datasets",
            headers={"X-Session-ID": session_id},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        _error(exc, "Unable to load datasets")
        return []


def run_research(objective: str, session_id: str, available_data: list[str]) -> dict:
    try:
        response = requests.post(
            f"{PYTHON_BASE_URL}/agents/research",
            json={
                "objective": objective,
                "session_id": session_id,
                "available_data": available_data,
            },
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        return {"error": _error(exc, "Multi-agent research failed")}


def download_artifact(artifact_id: str, session_id: str) -> tuple[bytes, str] | None:
    try:
        response = requests.get(
            f"{PYTHON_BASE_URL}/agents/artifacts/{artifact_id}",
            headers={"X-Session-ID": session_id},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return response.content, response.headers.get("Content-Type", "application/octet-stream")
    except requests.RequestException as exc:
        _error(exc, "Artifact download failed")
        return None
