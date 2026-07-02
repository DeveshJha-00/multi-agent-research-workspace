"""Chat and document-management page."""

import streamlit as st

from streamlit_app.utils.api_client import document_upload_rag, query_backend

st.set_page_config(page_title="Adaptive RAG Chat", page_icon="💬", layout="wide")

if "session_id" not in st.session_state:
    st.warning("Open a workspace first.")
    st.page_link("home.py", label="Go to home")
    st.stop()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = {}

st.title("Adaptive RAG Chat")
st.caption(f"Workspace: {st.session_state.session_id}")

with st.sidebar:
    st.header("Documents")
    uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])
    description = st.text_input(
        "Document description",
        max_chars=500,
        placeholder="Example: Product handbook for the Acme API",
    )
    if st.button("Index document", disabled=not (uploaded_file and description.strip())):
        with st.spinner("Parsing, embedding, and indexing..."):
            result = document_upload_rag(
                uploaded_file,
                description,
                st.session_state.session_id,
            )
        if result:
            st.session_state.uploaded_files[result["document_id"]] = result
            st.success(f"Indexed {result['filename']} ({result['chunks_indexed']} chunks)")
        else:
            st.error("Document upload failed. Check the backend logs.")

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

if user_input := st.chat_input("Ask a question..."):
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = query_backend(user_input, st.session_state.session_id)
        st.markdown(response["content"])
        st.caption(f"Route: {response.get('route', 'unknown')}")
        for source in response.get("sources", []):
            label = source.get("source", "Source")
            if source.get("url"):
                st.markdown(f"- [{label}]({source['url']})")
            else:
                page = source.get("page")
                st.markdown(f"- {label}" + (f" (page {page + 1})" if page is not None else ""))
    st.session_state.chat_history.append({"role": "assistant", **response})
