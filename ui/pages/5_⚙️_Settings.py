"""Settings Page - Configuration options."""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ui.components.sidebar import render_sidebar
from ui.utils.session import init_session_state


st.set_page_config(
    page_title="Settings - DataVault",
    page_icon="⚙️",
    layout="wide",
)

init_session_state()
render_sidebar()


st.markdown("## ⚙️ Settings")
st.caption("Configure DataVault options")

# LLM Settings
st.markdown("### 🤖 LLM Configuration")

col1, col2 = st.columns(2)

with col1:
    backend = st.selectbox(
        "LLM Backend",
        ["auto", "mlx", "ollama"],
        index=["auto", "mlx", "ollama"].index(st.session_state.get("llm_backend", "auto")),
        help="auto: MLX on Apple Silicon, Ollama elsewhere",
    )
    st.session_state.llm_backend = backend

with col2:
    if backend == "mlx":
        model = st.text_input(
            "MLX Model",
            value="mlx-community/Llama-3.2-3B-Instruct-4bit",
        )
    elif backend == "ollama":
        model = st.text_input(
            "Ollama Model",
            value="llama3.2",
        )
    else:
        model = st.text_input(
            "Model",
            value="auto",
            disabled=True,
        )
    st.session_state.model_name = model

st.divider()

# Langfuse Settings
st.markdown("### 📊 Observability (Langfuse)")

langfuse_enabled = st.toggle(
    "Enable Langfuse Tracing",
    value=st.session_state.get("langfuse_enabled", False),
)
st.session_state.langfuse_enabled = langfuse_enabled

if langfuse_enabled:
    col1, col2 = st.columns(2)
    
    with col1:
        public_key = st.text_input(
            "Public Key",
            type="password",
            placeholder="pk-lf-...",
        )
    
    with col2:
        secret_key = st.text_input(
            "Secret Key",
            type="password",
            placeholder="sk-lf-...",
        )
    
    host = st.text_input(
        "Langfuse Host",
        value="http://localhost:3000",
    )
    
    if st.button("🔗 Test Connection"):
        try:
            import requests
            response = requests.get(f"{host}/api/public/health", timeout=5)
            if response.status_code == 200:
                st.success(f"✅ Connected to Langfuse {response.json().get('version', '')}")
            else:
                st.error(f"Connection failed: {response.status_code}")
        except Exception as e:
            st.error(f"Connection failed: {e}")

st.divider()

# Database Settings
st.markdown("### 🗄️ Database")

col1, col2 = st.columns(2)

with col1:
    if st.session_state.get("mcp_client"):
        st.markdown("🟢 **DuckDB Connected**")
        st.caption(":memory: (in-memory database)")
    else:
        st.markdown("🔴 **Not Connected**")
        st.caption("Load data to connect")

with col2:
    table_count = len(st.session_state.get("tables", []))
    st.metric("Loaded Tables", table_count)

st.divider()

# About
st.markdown("### ℹ️ About")

st.markdown("""
**DataVault** - Privacy-Preserving Data Analytics Agent

- 🏛️ All data stays local in DuckDB
- 🤖 Multi-agent analysis with LangGraph
- 🧠 Semantic memory for context
- 📊 Langfuse observability

Built with Streamlit, DuckDB, and MLX/Ollama.
""")

# Version info
st.caption("Version 1.0.0")
