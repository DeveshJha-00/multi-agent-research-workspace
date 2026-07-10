"""Public landing page for the Streamlit application."""

import streamlit as st

from streamlit_app.utils.ui import (
    activate_workspace,
    apply_custom_css,
    ensure_workspace_state,
    remember_workspace,
    render_workspace_switcher,
)

st.set_page_config(page_title="Agentic Research Workspace", page_icon="🧭")
apply_custom_css()
ensure_workspace_state()
st.title("Agentic Research Workspace")
st.write("Chat with documents or delegate deeper work to specialized research and data agents.")
st.caption("Authentication is not enabled. Workspace IDs organize data but are not access control.")

with st.sidebar:
    render_workspace_switcher(compact=True)

workspace = st.text_input(
    "Workspace ID",
    value=st.session_state.session_id,
    help="Keep this value if you want to return to the same documents and conversation.",
)

chat_column, research_column = st.columns(2)


def open_page(page: str, selected_workspace: str) -> None:
    selected_workspace = selected_workspace.strip()
    if len(selected_workspace) < 8:
        st.error("Workspace ID must contain at least 8 characters.")
    else:
        remember_workspace(selected_workspace)
        activate_workspace(selected_workspace)
        st.switch_page(page)


with chat_column:
    if st.button("Open chat", type="primary", use_container_width=True):
        open_page("pages/chat.py", workspace)

with research_column:
    if st.button("Open research workspace", use_container_width=True):
        open_page("pages/research.py", workspace)
