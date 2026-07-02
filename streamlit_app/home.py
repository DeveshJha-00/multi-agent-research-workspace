"""Public landing page for the Streamlit application."""

from uuid import uuid4

import streamlit as st

st.set_page_config(page_title="Adaptive RAG", page_icon="💬")
st.title("Adaptive RAG")
st.write("Upload documents, ask grounded questions, or use general and web-search answers.")
st.caption("Authentication is not enabled. Workspace IDs organize data but are not access control.")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid4())

workspace = st.text_input(
    "Workspace ID",
    value=st.session_state.session_id,
    help="Keep this value if you want to return to the same documents and conversation.",
)

if st.button("Open chat", type="primary"):
    workspace = workspace.strip()
    if len(workspace) < 8:
        st.error("Workspace ID must contain at least 8 characters.")
    else:
        st.session_state.session_id = workspace
        st.switch_page("pages/chat.py")
