"""Public landing page for the Streamlit application."""

import streamlit as st

from streamlit_app.utils.ui import (
    activate_workspace,
    apply_custom_css,
    ensure_workspace_state,
    remember_workspace,
    render_workspace_switcher,
)

st.set_page_config(page_title="AgentForge", page_icon="🧭", layout="wide")
apply_custom_css()
ensure_workspace_state()

with st.sidebar:
    render_workspace_switcher(compact=True)


def open_page(page: str, selected_workspace: str) -> None:
    selected_workspace = selected_workspace.strip()
    if len(selected_workspace) < 8:
        st.error("Workspace ID must contain at least 8 characters.")
    else:
        remember_workspace(selected_workspace)
        activate_workspace(selected_workspace)
        st.switch_page(page)


st.title("AgentForge - Agentic Research Workspace")
st.subheader("A demo workspace for document chat, web-augmented answers, and agentic reports.")
st.write(
    "Upload documents, ask grounded questions, compare information with the web, analyze datasets, "
    "inspect repositories, and try multilingual voice interactions. Everything is grouped by the "
    "active workspace in the sidebar."
)
st.caption(
    "Authentication is intentionally not enabled. Workspace IDs organize demo data but are not access "
    "control."
)

st.divider()

chat_column, research_column = st.columns(2)

with chat_column:
    st.markdown("### Chat workspace")
    st.write(
        "Use Chat when you want a fast answer. It chooses the best route for each question: uploaded "
        "documents, general model knowledge, web search, or a hybrid document-plus-web answer."
    )
    st.markdown(
        """
        Best for:

        - asking questions about uploaded PDFs/TXT files
        - quick resume, policy, or handbook extraction
        - multilingual text/voice Q&A
        - optional RAGAS diagnostics on individual answers
        """
    )
    if st.button("Open Chat", type="primary", use_container_width=True):
        open_page("pages/chat.py", st.session_state.session_id)

with research_column:
    st.markdown("### Research workspace")
    st.write(
        "Use Research when the task needs a longer report, multiple evidence sources, datasets, or "
        "repository/codebase analysis. It runs a durable multi-agent workflow and stores downloadable "
        "artifacts."
    )
    st.markdown(
        """
        Best for:

        - document + web comparison reports
        - CSV/Excel analysis with charts
        - repository ZIP architecture explanations
        - longer objectives that should survive refreshes
        """
    )
    if st.button("Open Research", type="primary", use_container_width=True):
        open_page("pages/research.py", st.session_state.session_id)

st.divider()

st.markdown("### Suggested demo flow")
step_one, step_two, step_three = st.columns(3)
with step_one:
    st.markdown("#### 1. Pick a workspace")
    st.write("Use the sidebar to create, rename, or switch demo workspaces. Chat and Research share data.")
with step_two:
    st.markdown("#### 2. Add evidence")
    st.write("Upload documents in Chat, or documents/datasets/repository ZIPs in Research.")
with step_three:
    st.markdown("#### 3. Ask or delegate")
    st.write("Use Chat for direct Q&A; use Research for report-style tasks and generated artifacts.")

st.info(
    "Tip: documents uploaded in Chat are available to Research when the same workspace is active."
)
