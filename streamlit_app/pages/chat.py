"""Chat and document-management page."""

import base64
import time
from uuid import uuid4

import streamlit as st
import streamlit.components.v1 as components

from streamlit_app.utils.api_client import (
    create_evaluation,
    document_upload_rag,
    get_evaluation,
    get_evaluations,
    get_indexed_documents,
    get_voice_capabilities,
    query_backend,
    synthesize_speech,
    transcribe_speech,
)
from streamlit_app.utils.ui import apply_custom_css

st.set_page_config(page_title="Adaptive RAG Chat", page_icon="💬", layout="wide")
apply_custom_css()

if "session_id" not in st.session_state:
    st.warning("Open a workspace first.")
    st.page_link("home.py", label="Go to home")
    st.stop()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = {}
st.session_state.setdefault("response_evaluations", {})
st.session_state.setdefault("voice_transcript", "")
st.session_state.setdefault("voice_language", "auto")
st.session_state.setdefault("voice_warnings", [])

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


def _keep_evaluation_expanded(response_id: str) -> None:
    st.session_state[f"evaluation-expanded-{response_id}"] = True


def _evaluation_controls(message: dict) -> None:
    response_id = message.get("response_id")
    if not response_id:
        return
    expanded_key = f"evaluation-expanded-{response_id}"
    expanded = bool(
        st.session_state.get(expanded_key)
        or st.session_state.response_evaluations.get(response_id)
    )
    with st.expander("RAGAS evaluation", expanded=expanded):
        st.caption(
            "Scores are model-based diagnostics, not proof that an answer is correct. "
            "Leave the reference blank for reference-free evaluation."
        )
        reference = st.text_area(
            "Optional reference answer",
            key=f"evaluation-reference-{response_id}",
            max_chars=12_000,
        )
        if st.button(
            "Evaluate response",
            key=f"evaluate-{response_id}",
            on_click=_keep_evaluation_expanded,
            args=(response_id,),
        ):
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
                st.session_state[expanded_key] = True
        job = st.session_state.response_evaluations.get(response_id)
        if not job:
            history = get_evaluations(response_id, st.session_state.session_id)
            if history:
                job = history[0]
                st.session_state.response_evaluations[response_id] = job
                st.session_state[expanded_key] = True
        if job:
            _render_evaluation(job)


def _render_chat_message(message: dict) -> None:
    st.markdown(message["content"])
    if message.get("tts_audio_base64"):
        mime_type = message.get("tts_mime_type", "audio/wav")
        audio_base64 = message["tts_audio_base64"]
        if message.get("tts_autoplay"):
            components.html(
                (
                    '<audio controls autoplay style="width: 100%;">'
                    f'<source src="data:{mime_type};base64,{audio_base64}" type="{mime_type}">'
                    "Your browser does not support audio playback."
                    "</audio>"
                ),
                height=54,
            )
            message["tts_autoplay"] = False
        else:
            st.audio(
                base64.b64decode(audio_base64),
                format=mime_type,
            )
        if message.get("tts_shortened"):
            st.caption("Voice playback is a shortened preview; full answer is above.")
    if message.get("tts_error"):
        st.warning(message["tts_error"])
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


def _voice_enabled() -> bool:
    return bool(st.session_state.get("voice_answers_enabled"))


def _attach_voice_response(message: dict) -> dict:
    if not _voice_enabled() or message.get("route") == "error":
        return message
    language = message.get("answer_language") or st.session_state.get("voice_language") or "en-IN"
    with st.spinner("Generating voice response..."):
        audio = synthesize_speech(
            message["content"],
            st.session_state.session_id,
            language,
            st.session_state.get("tts_speaker"),
            st.session_state.get("tts_pace"),
        )
    if audio.get("error"):
        message["tts_error"] = audio["error"]
        return message
    message.update(
        {
            "tts_audio_base64": audio.get("audio_base64"),
            "tts_mime_type": audio.get("mime_type"),
            "tts_spoken_text": audio.get("spoken_text"),
            "tts_shortened": audio.get("shortened", False),
            "tts_speaker": audio.get("speaker"),
            "tts_autoplay": True,
        }
    )
    if audio.get("warning"):
        message["tts_error"] = audio["warning"]
    return message


def _submit_user_query(
    query: str,
    *,
    query_language: str = "auto",
    answer_language: str = "auto",
) -> dict:
    st.session_state.chat_history.append({"role": "user", "content": query})
    response = query_backend(
        query,
        st.session_state.session_id,
        query_language=query_language,
        answer_language=answer_language,
    )
    assistant_message = {"role": "assistant", **response}
    assistant_message = _attach_voice_response(assistant_message)
    st.session_state.chat_history.append(assistant_message)
    return assistant_message


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
    voice_capabilities = get_voice_capabilities()
    st.header("Voice")
    st.session_state.voice_answers_enabled = st.toggle(
        "Enable voice answers",
        value=st.session_state.get("voice_answers_enabled", False),
        disabled=not voice_capabilities.get("enabled", False),
    )
    speakers = voice_capabilities.get("speakers") or ["auto"]
    current_speaker = st.session_state.get("tts_speaker", "auto")
    speaker_index = speakers.index(current_speaker) if current_speaker in speakers else 0
    st.session_state.tts_speaker = st.selectbox(
        "TTS speaker",
        speakers,
        index=speaker_index,
        disabled=not voice_capabilities.get("enabled", False),
    )
    st.session_state.tts_pace = st.slider(
        "TTS pace",
        min_value=0.5,
        max_value=2.0,
        value=float(st.session_state.get("tts_pace", 1.0)),
        step=0.1,
        disabled=not voice_capabilities.get("enabled", False),
    )
    if not voice_capabilities.get("enabled", False):
        st.caption(voice_capabilities.get("error") or "Configure SARVAM_API_KEY to enable voice.")
    st.divider()

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

st.subheader("Voice input")
audio_input = st.audio_input("Record a question")
if audio_input is not None:
    if st.button("Transcribe voice"):
        with st.spinner("Transcribing with Sarvam..."):
            transcript = transcribe_speech(audio_input, st.session_state.session_id)
        if transcript.get("error"):
            st.error(transcript["error"])
        else:
            st.session_state.voice_transcript = transcript.get("transcript", "")
            st.session_state.voice_language = transcript.get("language_code") or "auto"
            st.session_state.voice_warnings = transcript.get("warnings", [])
for warning in st.session_state.voice_warnings:
    st.warning(warning)
if st.session_state.voice_transcript:
    st.caption(f"Detected language: {st.session_state.voice_language}")
    edited_transcript = st.text_area(
        "Review/edit transcript before sending",
        value=st.session_state.voice_transcript,
        key="voice_transcript_editor",
        max_chars=8000,
    )
    if st.button("Send voice transcript"):
        cleaned = edited_transcript.strip()
        if cleaned:
            with st.spinner("Thinking..."):
                _submit_user_query(
                    cleaned,
                    query_language=st.session_state.voice_language,
                    answer_language=st.session_state.voice_language,
                )
            st.session_state.voice_transcript = ""
            st.session_state.voice_warnings = []
            st.rerun()

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        _render_chat_message(message)

if user_input := st.chat_input("Ask a question..."):
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            assistant_message = _submit_user_query(user_input)
        _render_chat_message(assistant_message)
