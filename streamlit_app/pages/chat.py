"""Chat and document-management page."""

import time
from uuid import uuid4

import streamlit as st

from streamlit_app.utils.api_client import (
    create_evaluation,
    document_upload_rag,
    get_evaluation,
    get_evaluations,
    get_indexed_documents,
    query_backend,
)

st.set_page_config(page_title="Adaptive RAG Chat", page_icon="💬", layout="wide")

if "session_id" not in st.session_state:
    st.warning("Open a workspace first.")
    st.page_link("home.py", label="Go to home")
    st.stop()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = {}
st.session_state.setdefault("response_evaluations", {})

indexed_documents = get_indexed_documents(st.session_state.session_id)
st.session_state.uploaded_files = {
    item["document_id"]: item for item in indexed_documents
}

METRIC_HELP = {
    "answer_relevancy": "How directly the answer addresses the question.",
    "faithfulness": "How well answer claims are supported by retrieved evidence.",
    "context_utilization": "Whether the retrieved passages were useful for the answer.",
    "factual_correctness": "Factual overlap between the answer and your reference answer.",
    "semantic_similarity": "Meaning-level similarity to your reference answer.",
    "context_precision": "Whether relevant passages were ranked ahead of irrelevant ones.",
    "context_recall": "How much of the reference answer is supported by retrieved passages.",
}


def _render_evaluation(job: dict) -> None:
    status = job.get("status", "unknown")
    st.caption(
        f"Status: {status} · {job.get('progress', 0)}% · "
        f"Contexts: {job.get('context_count', 0)}"
    )
    metrics = job.get("metrics", {})
    completed = [item for item in metrics.values() if item.get("status") == "completed"]
    if completed:
        columns = st.columns(min(3, len(completed)))
        for index, item in enumerate(completed):
            with columns[index % len(columns)]:
                score = item.get("score")
                st.metric(item["name"].replace("_", " ").title(), f"{score:.3f}")
                st.caption(METRIC_HELP.get(item["name"], "RAGAS evaluation metric."))
                if item.get("reason"):
                    st.caption(item["reason"])
    failures = [item for item in metrics.values() if item.get("status") == "failed"]
    for item in failures:
        st.warning(f"{item['name'].replace('_', ' ').title()} unavailable: {item.get('error')}")
    if job.get("duration_seconds") is not None:
        st.caption(f"Evaluation duration: {job['duration_seconds']:.1f} seconds")
    if job.get("error"):
        st.error(job["error"])


def _watch_evaluation(evaluation_id: str) -> dict:
    placeholder = st.empty()
    latest = {}
    for _ in range(300):
        latest = get_evaluation(evaluation_id, st.session_state.session_id)
        if latest.get("error") or latest.get("status") in {"completed", "failed"}:
            break
        placeholder.caption(
            f"Evaluating in the background: {latest.get('progress', 0)}% complete"
        )
        time.sleep(1)
    placeholder.empty()
    return latest


def _evaluation_controls(message: dict) -> None:
    response_id = message.get("response_id")
    if not response_id:
        return
    with st.expander("RAGAS evaluation", expanded=False):
        st.caption(
            "Scores are model-based diagnostics, not proof that an answer is correct. "
            "Leave the reference blank for reference-free evaluation."
        )
        reference = st.text_area(
            "Optional reference answer",
            key=f"evaluation-reference-{response_id}",
            max_chars=12_000,
        )
        if st.button("Evaluate response", key=f"evaluate-{response_id}"):
            created = create_evaluation(
                response_id,
                st.session_state.session_id,
                reference.strip() or None,
                str(uuid4()),
            )
            if created.get("error"):
                st.error(created["error"])
            else:
                with st.spinner("RAGAS is evaluating this response..."):
                    job = _watch_evaluation(created["evaluation_id"])
                st.session_state.response_evaluations[response_id] = job
        job = st.session_state.response_evaluations.get(response_id)
        if not job:
            history = get_evaluations(response_id, st.session_state.session_id)
            if history:
                job = history[0]
                st.session_state.response_evaluations[response_id] = job
        if job:
            _render_evaluation(job)


def _render_chat_message(message: dict) -> None:
    st.markdown(message["content"])
    if message.get("route"):
        st.caption(f"Route: {message['route']}")
    for source in message.get("sources", []):
        label = source.get("source", "Source")
        if source.get("url"):
            st.markdown(f"- [{label}]({source['url']})")
        else:
            page = source.get("page")
            st.markdown(f"- {label}" + (f" (page {page + 1})" if page is not None else ""))
    _evaluation_controls(message)


st.title("Adaptive RAG Chat")
st.caption(f"Workspace: {st.session_state.session_id}")

navigation_left, navigation_right = st.columns([1, 4])
with navigation_left:
    if st.button("Research workspace", use_container_width=True):
        st.switch_page("pages/research.py")
st.info(
    "Chat selects one route per question. Use Research workspace when a task must combine "
    "uploaded documents, web sources, or datasets."
)

with st.sidebar:
    st.header("Documents")
    uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])
    description = st.text_input(
        "Document description (optional)",
        max_chars=500,
        placeholder="Example: Product handbook for the Acme API",
    )
    if st.button("Index document", disabled=not uploaded_file):
        with st.spinner("Parsing, embedding, and indexing..."):
            result = document_upload_rag(
                uploaded_file,
                description,
                st.session_state.session_id,
            )
        if not result.get("error"):
            st.session_state.uploaded_files[result["document_id"]] = result
            refreshed = get_indexed_documents(st.session_state.session_id)
            if refreshed:
                st.session_state.uploaded_files = {
                    item["document_id"]: item for item in refreshed
                }
            st.success(f"Indexed {result['filename']} ({result['chunks_indexed']} chunks)")
            st.caption(
                f"Parser: {result.get('parser_provider', 'local')} · "
                f"Language: {result.get('detected_language', 'unknown')} · "
                f"Script: {result.get('script', 'unknown')}"
            )
            for warning in result.get("warnings", []):
                st.warning(warning)
        else:
            st.error(result.get("error", "Document upload failed."))

    if st.session_state.uploaded_files:
        st.subheader("Indexed this session")
        for item in st.session_state.uploaded_files.values():
            st.caption(f"{item['filename']} · {item['chunks_indexed']} chunks")

    if st.button("New workspace"):
        for key in list(st.session_state):
            del st.session_state[key]
        st.switch_page("home.py")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        _render_chat_message(message)

if user_input := st.chat_input("Ask a question..."):
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = query_backend(user_input, st.session_state.session_id)
        assistant_message = {"role": "assistant", **response}
        _render_chat_message(assistant_message)
    st.session_state.chat_history.append(assistant_message)
