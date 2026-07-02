"""Resilient HTTP client for the Python RAG backend."""

import logging
import os

import requests

logger = logging.getLogger(__name__)
PYTHON_BASE_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT = int(os.getenv("RAG_REQUEST_TIMEOUT_SECONDS", "120"))


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
        logger.exception("RAG query failed")
        detail = ""
        if exc.response is not None:
            try:
                detail = exc.response.json().get("detail", "")
            except ValueError:
                detail = exc.response.text
        return {
            "content": f"The backend request failed. {detail}".strip(),
            "route": "general",
            "sources": [],
        }


def document_upload_rag(file, description: str, session_id: str) -> dict | None:
    try:
        response = requests.post(
            f"{PYTHON_BASE_URL}/rag/documents/upload",
            files={"file": (file.name, file.getvalue(), file.type)},
            headers={"X-Description": description, "X-Session-ID": session_id},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        logger.exception("Document upload failed")
        return None
