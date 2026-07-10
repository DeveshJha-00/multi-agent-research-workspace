from uuid import uuid4

import streamlit as st

WORKSPACE_SCOPED_KEYS = {
    "chat_history",
    "loaded_chat_session_id",
    "uploaded_files",
    "uploaded_datasets",
    "uploaded_repositories",
    "research_runs",
    "response_evaluations",
    "voice_transcript",
    "voice_language",
    "voice_warnings",
}


def apply_custom_css():
    """Applies custom CSS for redesigning the Streamlit frontend with premium UI touches."""
    st.markdown("""
    <style>
    /* Base dark mode enforcement and typography */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Smooth transitions globally */
    * {
        transition: background-color 0.2s ease, border-color 0.2s ease, transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    /* Buttons styling */
    .stButton > button {
        border-radius: 8px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        background-color: rgba(255, 255, 255, 0.05) !important;
        cursor: pointer !important;
        font-weight: 500 !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.3) !important;
        background-color: rgba(255, 255, 255, 0.1) !important;
        border-color: rgba(255, 255, 255, 0.2) !important;
        color: #ffffff !important;
    }
    
    .stButton > button:active {
        transform: translateY(1px);
        box-shadow: 0 0 0 rgba(0,0,0,0) !important;
    }

    /* Primary button override */
    .stButton > button[kind="primary"] {
        background-color: #2e66ff !important;
        border: 1px solid #477aff !important;
        color: white !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #477aff !important;
        box-shadow: 0 6px 16px rgba(46, 102, 255, 0.3) !important;
    }

    /* Chat messages */
    .stChatMessage {
        border-radius: 12px;
        padding: 1.25rem !important;
        margin-bottom: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
        background: rgba(25, 25, 25, 0.4);
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .stChatMessage:hover {
        border: 1px solid rgba(255, 255, 255, 0.1);
        background: rgba(25, 25, 25, 0.6);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        font-weight: 500;
        border-radius: 8px;
    }
    .streamlit-expanderHeader:hover {
        color: #fff;
        background-color: rgba(255,255,255,0.05);
    }
    
    /* Metrics (RAGAS) */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 600 !important;
    }
    [data-testid="stMetricLabel"] {
        font-weight: 500 !important;
        opacity: 0.8 !important;
    }
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.03);
        padding: 1.25rem !important;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        margin-bottom: 0.5rem;
    }
    [data-testid="stMetric"]:hover {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.15);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    
    /* Inputs */
    .stTextInput input, .stTextArea textarea, .stSelectbox select {
        border-radius: 8px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        background-color: rgba(0, 0, 0, 0.2) !important;
    }
    
    .stTextInput input:focus, .stTextArea textarea:focus, .stSelectbox select:focus {
        border-color: #2e66ff !important;
        box-shadow: 0 0 0 1px #2e66ff !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #121212;
        border-right: 1px solid rgba(255,255,255,0.05);
    }

    /* Native multipage navigation labels */
    [data-testid="stSidebarNav"] a span {
        text-transform: capitalize;
    }
    
    /* Audio input / players */
    audio {
        border-radius: 8px;
        outline: none;
    }
    
    /* Alerts & Warnings */
    .stAlert {
        border-radius: 8px !important;
    }
    
    </style>
    """, unsafe_allow_html=True)


def ensure_workspace_state() -> None:
    """Initialize a lightweight, demo-friendly workspace registry."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid4())
    if "workspaces" not in st.session_state:
        st.session_state.workspaces = {}
    workspace_id = st.session_state.session_id
    if workspace_id not in st.session_state.workspaces:
        st.session_state.workspaces[workspace_id] = {
            "id": workspace_id,
            "name": f"Workspace {len(st.session_state.workspaces) + 1}",
        }


def clear_workspace_caches() -> None:
    """Clear page-level cached state when switching workspaces."""
    for key in WORKSPACE_SCOPED_KEYS:
        st.session_state.pop(key, None)


def remember_workspace(workspace_id: str, name: str | None = None) -> None:
    ensure_workspace_state()
    workspace_id = workspace_id.strip()
    if not workspace_id:
        return
    st.session_state.workspaces.setdefault(
        workspace_id,
        {
            "id": workspace_id,
            "name": name or f"Workspace {len(st.session_state.workspaces) + 1}",
        },
    )


def activate_workspace(workspace_id: str) -> bool:
    """Activate a workspace ID and return True when the active workspace changed."""
    ensure_workspace_state()
    workspace_id = workspace_id.strip()
    if not workspace_id or workspace_id == st.session_state.session_id:
        return False
    remember_workspace(workspace_id)
    st.session_state.session_id = workspace_id
    clear_workspace_caches()
    return True


def create_workspace(name: str | None = None) -> str:
    ensure_workspace_state()
    workspace_id = str(uuid4())
    st.session_state.workspaces[workspace_id] = {
        "id": workspace_id,
        "name": name or f"Workspace {len(st.session_state.workspaces) + 1}",
    }
    activate_workspace(workspace_id)
    return workspace_id


def workspace_label(workspace_id: str) -> str:
    workspace = st.session_state.workspaces.get(workspace_id, {})
    name = workspace.get("name") or workspace_id[:8]
    return f"{name} · {workspace_id[:8]}"


def render_workspace_switcher(*, compact: bool = False) -> None:
    """Render controls for demo-only multi-workspace navigation."""
    ensure_workspace_state()
    if compact:
        st.subheader("Workspace")
    else:
        st.header("Workspaces")

    workspace_ids = list(st.session_state.workspaces)
    active_id = st.session_state.session_id
    selected_id = st.selectbox(
        "Active workspace",
        workspace_ids,
        index=workspace_ids.index(active_id),
        format_func=workspace_label,
        key="workspace-selector",
    )
    if selected_id != active_id and activate_workspace(selected_id):
        st.rerun()

    rename_key = f"workspace-rename-{st.session_state.session_id}"
    current_name = st.session_state.workspaces[st.session_state.session_id]["name"]
    new_name = st.text_input("Workspace name", value=current_name, key=rename_key)
    if st.button("Save name", key="save-workspace-name"):
        st.session_state.workspaces[st.session_state.session_id]["name"] = (
            new_name.strip() or current_name
        )
        st.rerun()

    if st.button("New workspace", key="new-workspace"):
        create_workspace()
        st.rerun()

    st.caption(f"ID: `{st.session_state.session_id}`")
