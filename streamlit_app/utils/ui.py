import streamlit as st


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
